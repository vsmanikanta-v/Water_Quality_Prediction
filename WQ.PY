import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
import joblib

# ---- Load data safely ----
data = pd.read_csv("water_quality.csv")  # Update path if needed

# Define features and target (ensure these columns exist)
features = [
    "ph",
    "hardness",
    "solids",
    "chloramines",
    "sulfate",
    "conductivity",
    "organic_carbon",
    "trihalomethanes",
    "turbidity",
]
target = "quality"

# Drop rows with missing target
data = data.dropna(subset=[target])

X = data[features]
y = data[target]

# Quick class balance check (helps decide if imbalance handling needed)
print("Target distribution:\n", y.value_counts(normalize=True))

# Stratified split to preserve class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Build pipeline: imputing missing values (median is robust) + classifier
pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),  # handles missing feature values
        (
            "clf",
            RandomForestClassifier(
                random_state=42,
                class_weight="balanced"  # helpful if classes are imbalanced
            ),
        ),
    ]
)

# Hyperparameter search space (can be expanded)
param_dist = {
    "clf__n_estimators": [100, 300, 500],
    "clf__max_depth": [None, 10, 20, 30],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2", 0.5],
}

# Use stratified CV for robust estimation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=50,
    cv=cv,
    scoring="f1_weighted",  # more reliable than raw accuracy if imbalance exists
    n_jobs=-1,
    verbose=1,
    random_state=42,
)

# Fit the tuning/search
search.fit(X_train, y_train)
best_model = search.best_estimator_

# Evaluation on test set
y_pred = best_model.predict(X_test)

# If binary classification, get predicted probabilities for ROC-AUC
is_binary = len(np.unique(y)) == 2
y_proba = None
if is_binary:
    try:
        y_proba = best_model.predict_proba(X_test)[:, 1]
    except Exception:
        pass  # some classifiers might not support predict_proba

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"Weighted F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall (weighted): {recall_score(y_test, y_pred, average='weighted'):.4f}")
if is_binary and y_proba is not None:
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Feature importance via permutation (less biased than raw impurity)
perm_result = permutation_importance(
    best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
importance_df = pd.DataFrame(
    {
        "feature": features,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }
).sort_values(by="importance_mean", ascending=False)
print("\nPermutation feature importances:")
print(importance_df)

# Save the entire pipeline (with imputer + tuned classifier)
joblib.dump(best_model, "water_quality_rf_pipeline.joblib")
print("\nSaved best model to 'water_quality_rf_pipeline.joblib'.")

# Predict on a new sample (will go through same preprocessing)
sample = pd.DataFrame(
    [
        {
            "ph": 7.0,
            "hardness": 150,
            "solids": 5000,
            "chloramines": 7.5,
            "sulfate": 300,
            "conductivity": 400,
            "organic_carbon": 10,
            "trihalomethanes": 80,
            "turbidity": 3,
        }
    ]
)
sample_pred = best_model.predict(sample)
print("Predicted water quality for sample:", sample_pred[0])
if is_binary:
    try:
        sample_proba = best_model.predict_proba(sample)[0]
        print("Prediction probabilities:", sample_proba)
    except Exception:
        pass
