import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_parquet("master_training_data.parquet")

df = df[~df['scenario'].str.contains("crash", case=False)].copy()

# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
print("Generating features...")

grouped = df.groupby('scenario')

df['spread_rel'] = (df['ask'] - df['bid']) / df['mid']

df['vol_20'] = grouped['mid'].transform(lambda x: x.rolling(window=20).std())
df['vol_50'] = grouped['mid'].transform(lambda x: x.rolling(window=50).std())

df['mid_change_abs'] = grouped['mid'].transform(lambda x: x.diff().abs())
df['velocity_ema'] = grouped['mid_change_abs'].transform(lambda x: x.ewm(span=20, adjust=False).mean())

df['spread_ma_50'] = grouped['spread_rel'].transform(lambda x: x.rolling(window=50).mean())
df['spread_ratio'] = df['spread_rel'] / df['spread_ma_50']

df = df.dropna().copy()

# ==============================================================================
# MODEL TRAINING
# ==============================================================================

# 4. Prepare X and y
# CRITICAL: Drop absolute identifiers and raw prices
features_to_drop = [
    "scenario", "label", "timestamp", "step", "run_id", 
    "bid", "ask", "mid", "spread" # Drop raw prices, keep derived features
]

# (Handle columns that might not exist if data_collector changed)
existing_cols_to_drop = [c for c in features_to_drop if c in df.columns]

le = LabelEncoder()
df["label"] = le.fit_transform(df["scenario"])

# ------------------------------------------------------------------------------
# SPLITTING STRATEGY: CHRONOLOGICAL (Prevents Leakage)
# ------------------------------------------------------------------------------
# We cannot use random shuffle (train_test_split) because rolling window features 
# leak info from t-1 into t. If we shuffle, the model memorizes the neighbors.
# We must split strictly by time: Train on first 75%, Test on last 25%.
print("Splitting data chronologically (preventing time-series leakage)...")

X_train_list = []
y_train_list = []
X_test_list = []
y_test_list = []

# We iterate through each scenario to ensure we get a 75/25 split for EACH type
for scenario_name in df['scenario'].unique():
    # Get all rows for this scenario, preserving time order
    scenario_data = df[df['scenario'] == scenario_name]
    
    # Calculate split index
    split_idx = int(len(scenario_data) * 0.75)
    
    # Split chronologically
    train_subset = scenario_data.iloc[:split_idx]
    test_subset = scenario_data.iloc[split_idx:]
    
    # Append to lists (dropping non-feature columns)
    X_train_list.append(train_subset.drop(columns=existing_cols_to_drop + ["label"]))
    y_train_list.append(train_subset["label"])
    
    X_test_list.append(test_subset.drop(columns=existing_cols_to_drop + ["label"]))
    y_test_list.append(test_subset["label"])

# Recombine into final sets
X_train = pd.concat(X_train_list)
y_train = pd.concat(y_train_list)
X_test = pd.concat(X_test_list)
y_test = pd.concat(y_test_list)

print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
print(f"Training with features: {list(X_train.columns)}")

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(le.classes_)
)

model.fit(X_train, y_train)

# ==============================================================================
# RESULTS
# ==============================================================================
y_pred = model.predict(X_test)

print("\n" + "="*40)
print("Classification Report")
print("="*40)
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nFeature Importances:")
importances = pd.Series(model.feature_importances_, index=X_train.columns)
print(importances.sort_values(ascending=False))

# Optional: Save the model so your bot can load it
model.save_model("market_classifier.json")
print("\nModel saved to market_classifier.json")
