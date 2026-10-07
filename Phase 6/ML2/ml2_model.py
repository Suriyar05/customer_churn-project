import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ML1_FILE = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "ML1",
        "ml1_model_input.csv"
    )
)

TARGET_COLUMN = "churn"

RANDOM_STATE = 42


# =========================================================
# 1. LOAD DATA
# =========================================================

print("\n======================================")
print("ML2 — LOAD ML1 DATA")
print("======================================")

print(
    "Reading:",
    ML1_FILE
)

if not os.path.exists(ML1_FILE):

    raise FileNotFoundError(
        f"ML1 output not found:\n{ML1_FILE}\n"
        "Run ML1 first."
    )


df = pd.read_csv(
    ML1_FILE
)


print(
    "\nRows loaded:",
    len(df)
)

print(
    "Columns:",
    df.columns.tolist()
)


# =========================================================
# 2. CREATE X AND y
# =========================================================

print("\n======================================")
print("CREATE X AND y")
print("======================================")


if TARGET_COLUMN not in df.columns:

    raise ValueError(
        "churn column not found"
    )


y = df[
    TARGET_COLUMN
].astype(int)


X = df.drop(
    columns=[
        TARGET_COLUMN
    ]
)


# =========================================================
# 3. CHECK ENCODING
# =========================================================

print("\n======================================")
print("FEATURE VALIDATION")
print("======================================")


object_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


if object_columns:

    raise ValueError(
        "Unencoded categorical columns found: "
        f"{object_columns}"
    )


print(
    "PASS — all features are numeric"
)


# Convert boolean columns if any remain

X = X.astype(int)


print(
    "\nX shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)


# =========================================================
# 4. CLASS DISTRIBUTION
# =========================================================

print("\n======================================")
print("CLASS DISTRIBUTION")
print("======================================")


print(
    y.value_counts()
)


print(
    "\nClass proportions:"
)

print(
    y.value_counts(
        normalize=True
    ).round(4)
)


# =========================================================
# 5. TRAIN / TEST SPLIT
# =========================================================

print("\n======================================")
print("TRAIN / TEST SPLIT")
print("======================================")


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    stratify=y,

    random_state=RANDOM_STATE
)


print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)


# =========================================================
# 6. CREATE MODELS
# =========================================================

print("\n======================================")
print("CREATE MODELS")
print("======================================")


logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE
)


tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=RANDOM_STATE
)


# =========================================================
# 7. TRAIN MODELS
# =========================================================

print("\n======================================")
print("TRAIN MODELS")
print("======================================")


print(
    "\nTraining Logistic Regression..."
)

logistic_model.fit(
    X_train,
    y_train
)

print(
    "PASS — Logistic Regression trained"
)


print(
    "\nTraining Decision Tree..."
)

tree_model.fit(
    X_train,
    y_train
)

print(
    "PASS — Decision Tree trained"
)


# =========================================================
# 8. PREDICTIONS
# =========================================================

logistic_pred = logistic_model.predict(
    X_test
)

tree_pred = tree_model.predict(
    X_test
)


# =========================================================
# 9. CLASSIFICATION REPORTS
# =========================================================

print("\n======================================")
print("LOGISTIC REGRESSION")
print("======================================")


print(
    classification_report(
        y_test,
        logistic_pred,
        target_names=[
            "Active",
            "Churned"
        ],
        zero_division=0
    )
)


print("\n======================================")
print("DECISION TREE")
print("======================================")


print(
    classification_report(
        y_test,
        tree_pred,
        target_names=[
            "Active",
            "Churned"
        ],
        zero_division=0
    )
)


# =========================================================
# 10. CONFUSION MATRICES
# =========================================================

print("\n======================================")
print("CONFUSION MATRICES")
print("======================================")


print(
    "\nLogistic Regression:"
)

print(
    confusion_matrix(
        y_test,
        logistic_pred
    )
)


print(
    "\nDecision Tree:"
)

print(
    confusion_matrix(
        y_test,
        tree_pred
    )
)


# =========================================================
# 11. CROSS VALIDATION
# =========================================================

print("\n======================================")
print("5-FOLD CROSS VALIDATION")
print("======================================")


logistic_cv = cross_val_score(
    logistic_model,
    X,
    y,
    cv=5,
    scoring="f1"
)


tree_cv = cross_val_score(
    tree_model,
    X,
    y,
    cv=5,
    scoring="f1"
)


print(
    "\nLogistic Regression CV F1:"
)

print(
    logistic_cv.round(4)
)

print(
    "Mean:",
    round(
        logistic_cv.mean(),
        4
    )
)


print(
    "\nDecision Tree CV F1:"
)

print(
    tree_cv.round(4)
)

print(
    "Mean:",
    round(
        tree_cv.mean(),
        4
    )
)


# =========================================================
# 12. CALCULATE TEST METRICS
# =========================================================

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

tree_accuracy = accuracy_score(
    y_test,
    tree_pred
)


logistic_precision = precision_score(
    y_test,
    logistic_pred,
    pos_label=1,
    zero_division=0
)

tree_precision = precision_score(
    y_test,
    tree_pred,
    pos_label=1,
    zero_division=0
)


logistic_recall = recall_score(
    y_test,
    logistic_pred,
    pos_label=1,
    zero_division=0
)

tree_recall = recall_score(
    y_test,
    tree_pred,
    pos_label=1,
    zero_division=0
)


logistic_f1 = f1_score(
    y_test,
    logistic_pred,
    pos_label=1,
    zero_division=0
)

tree_f1 = f1_score(
    y_test,
    tree_pred,
    pos_label=1,
    zero_division=0
)


# =========================================================
# 13. SINGLE MODEL COMPARISON TABLE
# =========================================================

comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree"
    ],

    "Accuracy": [
        logistic_accuracy,
        tree_accuracy
    ],

    "Precision": [
        logistic_precision,
        tree_precision
    ],

    "Recall": [
        logistic_recall,
        tree_recall
    ],

    "F1 Score": [
        logistic_f1,
        tree_f1
    ],

    "CV F1": [
        logistic_cv.mean(),
        tree_cv.mean()
    ]

})


# =========================================================
# 14. DISPLAY COMPARISON
# =========================================================

print("\n======================================")
print("MODEL COMPARISON")
print("======================================")

print(
    comparison.round(4).to_string(
        index=False
    )
)


# =========================================================
# 15. SELECT BEST MODEL
# =========================================================

print("\n======================================")
print("MODEL SELECTION")
print("======================================")


if logistic_f1 >= tree_f1:

    selected_model = (
        "Logistic Regression"
    )

    selected_model_object = (
        logistic_model
    )

else:

    selected_model = (
        "Decision Tree"
    )

    selected_model_object = (
        tree_model
    )


print(
    "Primary metric: F1 Score"
)

print(
    f"Selected Model: {selected_model}"
)


# =========================================================
# 16. SAVE COMPARISON TABLE
# =========================================================

comparison_file = os.path.join(
    BASE_DIR,
    "model_comparison.csv"
)


comparison.round(4).to_csv(
    comparison_file,
    index=False
)


print(
    "\nComparison table saved:"
)

print(
    comparison_file
)


# =========================================================
# 17. SAVE MODELS
# =========================================================

models_dir = os.path.join(
    BASE_DIR,
    "models"
)


os.makedirs(
    models_dir,
    exist_ok=True
)


joblib.dump(
    logistic_model,
    os.path.join(
        models_dir,
        "logistic_churn.pkl"
    )
)


joblib.dump(
    tree_model,
    os.path.join(
        models_dir,
        "tree_churn.pkl"
    )
)


# =========================================================
# 18. SAVE SELECTED MODEL
# =========================================================

joblib.dump(
    selected_model_object,
    os.path.join(
        models_dir,
        "selected_churn_model.pkl"
    )
)


print("\n======================================")
print("MODELS SAVED")
print("======================================")


print(
    "models/logistic_churn.pkl"
)

print(
    "models/tree_churn.pkl"
)

print(
    "models/selected_churn_model.pkl"
)


# =========================================================
# COMPLETE
# =========================================================

print("\n======================================")
print("ML2 COMPLETE")
print("======================================")