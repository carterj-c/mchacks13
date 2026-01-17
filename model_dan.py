import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load Data
df = pd.read_parquet("master_training_data.parquet")

# 1.2. Filter out crashes
df = df[~df['scenario'].str.contains("crash", case=False)].copy()

# 2. Pre-processing & Feature Engineering
df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["mid"] > 0)].copy()

# Group by scenario to ensure rolling windows stay within their own simulations
grouped = df.groupby('scenario')

# These features essentially "summarize" the group of rows leading up to the current tick
df['vol_20'] = grouped['mid'].transform(lambda x: x.rolling(window=20).std())
df['vol_50'] = grouped['mid'].transform(lambda x: x.rolling(window=50).std())
df['spread_rel'] = (df['ask'] - df['bid']) / df['mid']
df['mid_change_abs'] = grouped['mid'].transform(lambda x: x.diff().abs())
df['velocity_ema'] = grouped['mid_change_abs'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
df['spread_ma_50'] = grouped['spread_rel'].transform(lambda x: x.rolling(window=50).mean())
df['spread_ratio'] = df['spread_rel'] / df['spread_ma_50']

# 3. THE "GROUPING" STEP
# Instead of keeping every row, we take every 50th row.
# Each of these rows now carries the 'summary' (volatility, velocity) of the 50 rows before it.
WINDOW_SIZE = 50
df_windowed = df.iloc[::WINDOW_SIZE].copy()

# Clean up NaNs from the start of the windows
df_windowed = df_windowed.dropna()

# 4. Build X and y
le = LabelEncoder()
df_windowed["y"] = le.fit_transform(df_windowed["scenario"])

# Drop columns that are specific to a single point in time or identifiers
X = df_windowed.drop(columns=[
    "scenario", "y", "timestamp", "step", 
    "bid", "ask", "mid", "spread" # Drop raw prices to force behavior learning
], errors='ignore')

y = df_windowed["y"]

print(f"Analyzing {len(X)} distinct market windows...")
print("Class Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# 5. Train/Test Split
# Because we windowed the data, the 'leakage' between test and train is significantly reduced.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 6. XGBoost Model
model = XGBClassifier(
    n_estimators=200,
    max_depth=4, # Slightly shallower to prevent overfitting on specific windows
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(le.classes_)
)

model.fit(X_train, y_train)

# 7. Results
y_pred = model.predict(X_test)
print("\n" + "="*40)
print("WINDOWED CLASSIFICATION REPORT")
print("="*40)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 8. Feature Importance (See what actually defines a "Normal" vs "HFT" market)
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop Predictors of Market Scenario:")
print(importances.sort_values(ascending=False))