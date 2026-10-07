import pandas as pd
 
# ============================================================
# ML1 — CUSTOMER CHURN PROBLEM DEFINITION
#  ============================================================
 
print("\n========== ML1 CUSTOMER CHURN ML SETUP ==========\n")
 
# ============================================================
# 1. LOAD CUSTOMER ML FEATURES
# ============================================================
 
import pandas as pd
from sqlalchemy import create_engine
 
DATABASE_URL = (
    "mysql+pymysql://root:root@localhost/"
    "telecom_intelligence"
)
 
engine = create_engine(DATABASE_URL)
 
df = pd.read_sql(
    "SELECT * FROM customer_ml_features",
    engine
)
 
print("Rows loaded:", len(df))
print(df.columns.tolist())
 
# ============================================================
# 2. IDENTIFY TARGET
# ============================================================
 
target_column = "churn"
 
print(
    f"\nTarget column: {target_column}"
)
 
 
# ============================================================
# 3. IDENTIFY DROP COLUMN
# ============================================================
 
drop_columns = [
    "customer_id"
]
 
print(
    f"Drop columns: {drop_columns}"
)
 
 
# ============================================================
# 4. NUMERIC FEATURES
# ============================================================
 
numeric_columns = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "service_count",
    "high_charge_flag",
    "is_long_term_customer",
    "auto_pay_flag",
    "has_streaming_bundle"
]
 
print("\nNumeric features:")
 
for column in numeric_columns:
 
    print(
        f" - {column}"
    )
 
 
# ============================================================
# 5. CATEGORICAL FEATURES
# ============================================================
 
categorical_columns = [
    "contract_type",
    "internet_service"
]
 
print("\nCategorical features:")
 
for column in categorical_columns:
 
    print(
        f" - {column}"
    )
 
 
# ============================================================
# 6. CREATE TARGET y
# ============================================================
 
y = df[target_column].astype(int)
 
 
# ============================================================
# 7. CREATE FEATURE MATRIX
# ============================================================
 
X = df[
    numeric_columns + categorical_columns
].copy()
 
 
# ============================================================
# 8. ONE-HOT ENCODE CATEGORICAL FEATURES
# ============================================================
 
print(
    "\nApplying one-hot encoding..."
)
 
X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=False,
    dtype=int
)
 
 
# ============================================================
# 9. PRINT FINAL FEATURE SET
# ============================================================
 
print(
    "\n========== FINAL FEATURE SET ==========\n"
)
 
print(
    X.columns.tolist()
)
 
 
# ============================================================
# 10. PRINT SHAPES
# ============================================================
 
print(
    "\n========== SHAPES ==========\n"
)
 
print(
    f"X shape: {X.shape}"
)
 
print(
    f"y shape: {y.shape}"
)
 
 
# ============================================================
# 11. PRINT SAMPLE FEATURES
# ============================================================
 
print(
    "\n========== X SAMPLE ==========\n"
)
 
print(
    X.head()
)
 
 
print(
    "\n========== y SAMPLE ==========\n"
)
 
print(
    y.head()
)
 
 
# ============================================================
# 12. CLASS BALANCE
# ============================================================
 
print(
    "\n========== CLASS BALANCE ==========\n"
)
 
class_balance = (
    y.value_counts(
        normalize=True
    )
    .sort_index()
)
 
print(class_balance)
 
 
print(
    "\nClass counts:"
)
 
print(
    y.value_counts()
    .sort_index()
)
 
 
# ============================================================
# 13. ML1 SUMMARY
# ============================================================
 
print(
    "\n========== ML1 SUMMARY ==========\n"
)
 
print(
    "Target     : churn"
)
 
print(
    "Dropped    : customer_id"
)
 
print(
    f"Numeric features     : {len(numeric_columns)}"
)
 
print(
    f"Categorical features : {len(categorical_columns)}"
)
 
print(
    f"Final X columns      : {X.shape[1]}"
)
 
print(
    f"Samples               : {X.shape[0]}"
)
 
print(
    "\nPrimary evaluation metric should NOT be accuracy."
)
 
print(
    "Use churn-focused metrics such as "
    "precision, recall, and F1-score."
)
 
print(
    "\nML1 problem definition completed successfully."
)