import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier

os.makedirs("output", exist_ok=True)

# ════════════════════════════════════════
# 1. Merged CSV load 
# ════════════════════════════════════════
print("[1/5] Loading merged data...")
df = pd.read_csv("output/ornl_merged.csv")
print(f"      Shape: {df.shape}")

# ════════════════════════════════════════
# 2. Clean
# ════════════════════════════════════════
print("[2/5] Cleaning...")

# Infinity → 0
df.replace([np.inf, -np.inf], 0, inplace=True)

# Missing → median
df.fillna(df.median(numeric_only=True), inplace=True)

# Label encode: NoEvents=0, Attack=1, Natural=2 (alphabetical)
le = LabelEncoder()
y = le.fit_transform(df["marker"])
print(f"      Labels: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Features separate
X = df.drop(columns=["marker"])
print(f"      X: {X.shape}")

# Normalize 0-1
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ════════════════════════════════════════
# 3. Low-variance features skip
# ════════════════════════════════════════
print("[3/5] Removing low-variance features...")
vt = VarianceThreshold(threshold=0.01)
X_var = pd.DataFrame(
    vt.fit_transform(X_scaled),
    columns=X_scaled.columns[vt.get_support()]
)
print(f"      {X_scaled.shape[1]} → {X_var.shape[1]} features")

# ════════════════════════════════════════
# 4. Correlated features skip
# ════════════════════════════════════════
print("[4/5] Removing correlated features...")
corr = X_var.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_cols = [c for c in upper.columns if any(upper[c] > 0.95)]
X_clean = X_var.drop(columns=drop_cols)
print(f"      {X_var.shape[1]} → {X_clean.shape[1]} features")

# ════════════════════════════════════════
# 5. Random Forest importance  ranking
# ════════════════════════════════════════
print("[5/5] Ranking features with Random Forest...")
print("      (take times 2/3 minutes.)")

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_clean, y)

importance = pd.Series(
    rf.feature_importances_, index=X_clean.columns
).sort_values(ascending=False)

# Top features 
print("\n      Top 10 most important features:")
print(importance.head(10).to_string())

# Chart save 
plt.figure(figsize=(10, 8))
importance.head(30).sort_values().plot(kind='barh', color='steelblue')
plt.xlabel("Importance Score")
plt.title("Top 30 Features — Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("output/feature_importance.png", dpi=150)
plt.close()
print("      ✅ Saved: output/feature_importance.png")

# Feature sets save 
print("\n      Saving feature sets...")
for n in [20, 30, 40, 51]:
    top_names = importance.head(n).index.tolist()
    out_df = X_clean[top_names].copy()
    out_df["label"] = y
    path = f"output/features_top_{n}.csv"
    out_df.to_csv(path, index=False)
    print(f"      ✅ Saved: {path}")

