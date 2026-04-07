import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -------------------------------------------------
# 1. Load dataset
# -------------------------------------------------
df = pd.read_csv("../data/health_data.csv")

print("Dataset Shape:", df.shape)
print("\nClass Distribution:\n", df["condition"].value_counts())

# -------------------------------------------------
# 2. Features and Target (NO BALANCING)
# -------------------------------------------------
X = df.drop("condition", axis=1)
y = df["condition"]

# -------------------------------------------------
# 3. Train-Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=600,     
    max_depth=18,            
    min_samples_split=3,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# -------------------------------------------------
# 5. Test Evaluation
# -------------------------------------------------
y_pred = model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------------------------
# 6. Cross Validation
# -------------------------------------------------
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross-validation scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# -------------------------------------------------
# 7. Feature Importance (for analysis)
# -------------------------------------------------
print("\nFeature Importance:\n")

importances = model.feature_importances_

for col, imp in zip(X.columns, importances):
    print(f"{col}: {imp:.4f}")

# -------------------------------------------------
# 8. Save Model
# -------------------------------------------------
os.makedirs("../model", exist_ok=True)

joblib.dump(model, "../model/health_risk_model.pkl")

print("\nModel trained and saved successfully (No Balancing, Tuned RF).")





