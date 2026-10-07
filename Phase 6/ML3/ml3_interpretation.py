import pandas as pd
import joblib
import matplotlib.pyplot as plt
 
from pathlib import Path
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
 
 
# ============================================================
# DATABASE CONNECTION
# ============================================================
 
DATABASE_URL = (
    "mysql+pymysql://root:root@localhost/"
    "telecom_intelligence"
)
 
engine = create_engine(
    DATABASE_URL,
    echo=False
)
 
 
# ============================================================
# PATHS
# ============================================================
 
BASE_DIR = Path(__file__).resolve().parents[2]
 
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "tree_churn.pkl"
)
 
 
# ============================================================
# ML3 — MODEL INTERPRETATION
# ============================================================
 
print("\n========== ML3 MODEL INTERPRETATION ==========\n")
 
 
# ============================================================
# 1. LOAD CUSTOMER ML FEATURES
# ============================================================
 
print("Loading customer_ml_features...")
 
df = pd.read_sql(
    "SELECT * FROM customer_ml_features",
    engine
)
 
print(
    f"Rows loaded: {len(df)}"
)
 
print("\nAvailable columns:")
 
print(
    list(df.columns)
)
 
 
# ============================================================
# 2. DEFINE FEATURES
# ============================================================
 
numeric_features = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "service_count",
    "high_charge_flag",
    "is_long_term_customer",
    "auto_pay_flag",
    "has_streaming_bundle"
]
 
 
categorical_features = [
    "contract_type",
    "internet_service"
]
 
 
target_column = "churn"
 
 
# ============================================================
# 3. CREATE X AND y
# ============================================================
 
X = df[
    numeric_features + categorical_features
].copy()
 
y = df[
    target_column
].astype(int)
 
 
# ============================================================
# 4. ONE-HOT ENCODE
# ============================================================
 
print("\nApplying one-hot encoding...")
 
X = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=False,
    dtype=int
)
 
 
print(
    f"Final feature count: {X.shape[1]}"
)
 
 
# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================
 
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X,
    y,
    df["customer_id"],
    test_size=0.2,
    stratify=y,
    random_state=42
)
 
 
print(
    f"Test customers: {len(X_test)}"
)
 
 
# ============================================================
# 6. LOAD BEST MODEL
# ============================================================
 
print("\nLoading Decision Tree model...")
 
model = joblib.load(
    
    r"C:\Users\suriya.r\Documents\customer intelligence labs\Phase 6\ML2\models\tree_churn.pkl"
)
 
print(
    "Decision Tree loaded successfully."
)
 
 
# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
 
print("\n========== FEATURE IMPORTANCE ==========\n")
 
 
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)
 
 
feature_importance = (
    feature_importance
    .sort_values(
        ascending=False
    )
)
 
 
print(
    feature_importance
)
 
 
# ============================================================
# 8. TOP 10 FEATURES
# ============================================================
 
top_10 = (
    feature_importance
    .head(10)
)
 
 
print("\n========== TOP 10 FEATURES ==========\n")
 
print(
    top_10
)
 
 
# ============================================================
# 9. PLOT TOP 10
# ============================================================
 
plt.figure(
    figsize=(10, 6)
)
 
top_10.sort_values().plot(
    kind="barh"
)
 
plt.title(
    "Top 10 Decision Tree Feature Importances"
)
 
plt.xlabel(
    "Importance"
)
 
plt.ylabel(
    "Feature"
)
 
plt.tight_layout()
 
plt.show()
 
 
# ============================================================
# 10. TOP 3 FEATURES
# ============================================================
 
print("\n========== TOP 3 CHURN DRIVERS ==========\n")
 
top_3 = (
    feature_importance
    .head(3)
)
 
 
for feature, importance in top_3.items():
 
    print(
        f"{feature}: "
        f"{importance:.6f}"
    )
 
 
# ============================================================
# 11. PREDICT CHURN PROBABILITIES
# ============================================================
 
print(
    "\n========== CUSTOMER RISK SCORING ==========\n"
)
 
risk_probabilities = model.predict_proba(
    X_test
)[:, 1]
 
 
# ============================================================
# 12. CREATE RISK DATAFRAME
# ============================================================
 
risk_df = pd.DataFrame({
 
    "customer_id": ids_test.values,
 
    "churn_probability":
        risk_probabilities,
 
    "actual_churn":
        y_test.values
 
})
 
 
# ============================================================
# 13. SORT HIGHEST RISK
# ============================================================
 
top_risk_customers = (
    risk_df
    .sort_values(
        "churn_probability",
        ascending=False
    )
    .head(20)
)
 
 
print(
    "\n========== TOP 20 HIGHEST-RISK CUSTOMERS ==========\n"
)
 
print(
    top_risk_customers.to_string(
        index=False
    )
)
 
 
# ============================================================
# 14. SQL6 COMPARISON
# ============================================================
 
print(
    "\n========== SQL6 OVERLAP CHECK ==========\n"
)
 
print(
    "This section compares ML high-risk customers "
    "with the SQL6 rule-based high-risk list."
)
 
print(
    "If your SQL6 high-risk customer IDs are available, "
    "place them in the sql6_high_risk_ids set below."
)
 
 
# ------------------------------------------------------------
# Replace these IDs with the customer IDs produced by SQL6
# ------------------------------------------------------------
 
sql6_high_risk_ids = set()
 
 
ml_high_risk_ids = set(
    top_risk_customers[
        "customer_id"
    ]
)
 
 
if sql6_high_risk_ids:
 
    overlap = (
        ml_high_risk_ids
        .intersection(
            sql6_high_risk_ids
        )
    )
 
    print(
        f"ML top-20 customers: "
        f"{len(ml_high_risk_ids)}"
    )
 
    print(
        f"SQL6 high-risk customers: "
        f"{len(sql6_high_risk_ids)}"
    )
 
    print(
        f"Overlap: "
        f"{len(overlap)}"
    )
 
    print(
        "\nOverlapping customer IDs:"
    )
 
    print(
        sorted(overlap)
    )
 
else:
 
    print(
        "SQL6 IDs have not been entered yet."
    )
 
 
# ============================================================
# 15. BUSINESS INTERPRETATION
# ============================================================
 
print(
    "\n========== BUSINESS INTERPRETATION ==========\n"
)
 
 
top_feature_names = list(
    top_3.index
)
 
 
print(
    "Customers most likely to churn are those "
    f"whose behaviour is strongly associated with "
    f"{top_feature_names[0]}, "
    f"{top_feature_names[1]}, and "
    f"{top_feature_names[2]}."
)
 
print(
    "These features provide the strongest signals "
    "used by the Decision Tree when separating "
    "churned and non-churned customers."
)
 
print(
    "Customers receiving high churn probabilities "
    "should be prioritised by the retention team "
    "for proactive intervention."
)
 
print(
    "The model risk scores can help the business "
    "focus retention efforts on a smaller group of "
    "customers instead of contacting every customer."
)
 
print(
    "The ML-based high-risk list can also be compared "
    "with the SQL6 rule-based list to identify customers "
    "that both approaches consider risky."
)
 
 
# ============================================================
# FINAL
# ============================================================
 
print(
    "\n========== ML3 COMPLETE ==========\n"
)
 
print(
    "PASS - Decision Tree loaded"
)
 
print(
    "PASS - Feature importances calculated"
)
 
print(
    "PASS - Top 10 features identified"
)
 
print(
    "PASS - Top 20 customers scored"
)
 
print(
    "PASS - Business interpretation generated"
)