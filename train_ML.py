import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# ===========================
# Load dataset
# ===========================
df = pd.read_csv("output/features_top_40.csv")

X = df.drop(columns=["marker"])
y = df["marker"]

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# ===========================
# Train-test split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   
)

# ===========================
# Random Forest (balanced)
# ===========================
print("\n--- Random Forest ---")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ===========================
# Decision Tree (balanced)
# ===========================
print("\n--- Decision Tree ---")

dt = DecisionTreeClassifier(
    random_state=42,
    class_weight='balanced'
)

dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))