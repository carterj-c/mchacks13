import pandas as pd
import numpy as np
import joblib  # Import joblib for saving the model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier

# 1. Load Data
files = [
    "received_trades_team_alpha_flash_crash.csv",
    "received_trades_team_alpha_hft_dominated.csv",
    "received_trades_team_alpha_mini_flash_crash.csv",
    "received_trades_team_alpha_normal_market.csv",
    "received_trades_team_alpha_stressed_market.csv",
]

dfs = []
for f in files:
    try:
        temp_df = pd.read_csv(f)
        dfs.append(temp_df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

df = pd.concat(dfs, ignore_index=True)

# 2. Pre-processing & Feature Engineering
df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["mid"] > 0)].copy()

# Group by scenario to ensure rolling windows stay within their own simulations
grouped = df.groupby("scenario")

# Calculate rolling features
df["vol_20"] = grouped["mid"].transform(lambda x: x.rolling(window=20).std())
df["vol_50"] = grouped["mid"].transform(lambda x: x.rolling(window=50).std())
df["spread_rel"] = (df["ask"] - df["bid"]) / df["mid"]
df["mid_change_abs"] = grouped["mid"].transform(lambda x: x.diff().abs())
df["velocity_ema"] = grouped["mid_change_abs"].transform(
    lambda x: x.ewm(span=20, adjust=False).mean()
)
df["spread_ma_50"] = grouped["spread_rel"].transform(
    lambda x: x.rolling(window=50).mean()
)
df["spread_ratio"] = df["spread_rel"] / df["spread_ma_50"]

# MERGE CATEGORIES: Merge 'mini_flash_crash' into 'flash_crash'
df["scenario"] = df["scenario"].replace({"mini_flash_crash": "flash_crash"})

# 3. THE "GROUPING" STEP
WINDOW_SIZE = 50
df_windowed = df.iloc[::WINDOW_SIZE].copy()
df_windowed = df_windowed.dropna()

# 4. Build X and y
le = LabelEncoder()
df_windowed["y"] = le.fit_transform(df_windowed["scenario"])

# Drop columns specific to a single point in time or identifiers
X = df_windowed.drop(
    columns=["scenario", "y", "timestamp", "step", "bid", "ask", "mid", "spread"],
    errors="ignore",
)

y = df_windowed["y"]

print(f"Analyzing {len(X)} distinct market windows...")
print("Class Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 6. Model Training
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    max_features=0.8,
    random_state=42,
)

model.fit(X_train, y_train)

# 7. Results
y_pred = model.predict(X_test)
print("\n" + "=" * 40)
print("WINDOWED CLASSIFICATION REPORT")
print("=" * 40)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 8. Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop Predictors of Market Scenario:")
print(importances.sort_values(ascending=False))

# 9. SAVE THE MODEL
output_filename = "market_scenario_model.pkl"
joblib.dump(model, output_filename)
print(f"\nModel successfully saved to {output_filename}")
