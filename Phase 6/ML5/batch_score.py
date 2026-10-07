import os
import joblib
import pandas as pd

from datetime import datetime
from sqlalchemy import create_engine, text


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_URL = (
    "mysql+pymysql://root:root@localhost:3306/telecom_intelligence"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ML2",
    "models",
    "tree_churn.pkl"
)


# ============================================================
# REQUIRED DATABASE COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "customer_id",
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract",
    "internet_service",
    "service_count",
    "high_charge_flag",
    "is_long_term_customer",
    "has_streaming_bundle",
    "auto_pay_flag"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("\n========================================")
print("LOADING CHURN MODEL")
print("========================================")

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print(
    f"Model loaded successfully: {MODEL_PATH}"
)


# ============================================================
# CHECK DATABASE SCHEMA
# ============================================================

def check_database_schema():

    print("\n========================================")
    print("CHECK DATABASE SCHEMA")
    print("========================================")

    with engine.connect() as connection:

        result = connection.execute(
            text("""
                DESCRIBE customer_ml_features
            """)
        )

        actual_columns = [
            row[0]
            for row in result
        ]

    print("Available columns:")

    print(actual_columns)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in actual_columns
    ]

    if missing:

        raise ValueError(
            "\nMissing columns in customer_ml_features:\n"
            f"{missing}\n\n"
            "Fix SQL5 before running ML5."
        )

    print(
        "PASS — all required ML5 columns exist"
    )


# ============================================================
# LOAD CUSTOMERS
# ============================================================

def load_customers():

    print("\n========================================")
    print("ML5 — LOAD CUSTOMER DATA")
    print("========================================")

    check_database_schema()

    query = text("""
        SELECT
            customer_id,
            tenure,
            monthly_charges,
            total_charges,
            contract,
            internet_service,
            service_count,
            high_charge_flag,
            is_long_term_customer,
            has_streaming_bundle,
            auto_pay_flag

        FROM customer_ml_features
    """)

    df = pd.read_sql(
        query,
        engine
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    if len(df) == 0:

        raise ValueError(
            "customer_ml_features is empty"
        )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    print("\n========================================")
    print("PREPARE FEATURES")
    print("========================================")

    features = df[
        [
            "tenure",
            "monthly_charges",
            "total_charges",
            "service_count",
            "high_charge_flag",
            "is_long_term_customer",
            "auto_pay_flag",
            "has_streaming_bundle",
            "contract",
            "internet_service"
        ]
    ].copy()

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

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

    for column in numeric_columns:

        features[column] = pd.to_numeric(
            features[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Check numeric NULLs
    # --------------------------------------------------------

    if features[numeric_columns].isnull().any().any():

        raise ValueError(
            "Numeric feature columns contain NULL/invalid values"
        )

    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    features = pd.get_dummies(
        features,
        columns=[
            "contract",
            "internet_service"
        ],
        drop_first=False
    )

    # --------------------------------------------------------
    # Convert boolean columns to integers
    # --------------------------------------------------------

    features = features.astype(int)

    print(
        "Features prepared successfully."
    )

    print(
        "Feature columns:"
    )

    for column in features.columns:

        print(
            f" - {column}"
        )

    return features


# ============================================================
# ALIGN MODEL FEATURES
# ============================================================

def align_model_features(features):

    print("\n========================================")
    print("ALIGN MODEL FEATURES")
    print("========================================")

    # --------------------------------------------------------
    # sklearn tree model stores training feature names
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_names_in_"
    ):

        expected_columns = list(
            model.feature_names_in_
        )

        print(
            f"Model expects {len(expected_columns)} features."
        )

        # ----------------------------------------------------
        # Add missing training columns as zero
        # ----------------------------------------------------

        for column in expected_columns:

            if column not in features.columns:

                features[column] = 0

        # ----------------------------------------------------
        # Remove extra columns
        # ----------------------------------------------------

        features = features[
            expected_columns
        ]

    else:

        print(
            "WARNING — model does not contain "
            "feature_names_in_."
        )

    return features


# ============================================================
# SCORE ONE CUSTOMER
# ============================================================

def score_customer(
    row,
    feature_row
):

    prediction_value = model.predict(
        feature_row
    )[0]

    probabilities = model.predict_proba(
        feature_row
    )[0]

    risk_score = float(
        probabilities[1]
    )

    confidence = float(
        max(probabilities)
    )

    if prediction_value == 1:

        prediction = "Likely to churn"

    else:

        prediction = "Unlikely to churn"

    return {
        "customer_id": row["customer_id"],
        "risk_score": risk_score,
        "prediction": prediction,
        "confidence": confidence
    }


# ============================================================
# BATCH SCORE
# ============================================================

def batch_score(df):

    print("\n========================================")
    print("BATCH SCORING")
    print("========================================")

    features = prepare_features(
        df
    )

    features = align_model_features(
        features
    )

    results = []

    for index in range(
        len(df)
    ):

        original_row = df.iloc[index]

        feature_row = features.iloc[
            index:index + 1
        ]

        result = score_customer(
            original_row,
            feature_row
        )

        results.append(
            result
        )

    result_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Add scoring date
    # --------------------------------------------------------

    result_df["scoring_date"] = (
        datetime.today()
        .strftime("%Y-%m-%d")
    )

    print(
        f"Customers scored: {len(result_df)}"
    )

    return result_df


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(result_df):

    print("\n========================================")
    print("SAVE CUSTOMER RISK TABLE")
    print("========================================")

    output_columns = [
        "customer_id",
        "risk_score",
        "prediction",
        "confidence",
        "scoring_date"
    ]

    result_df[
        output_columns
    ].to_sql(
        "customer_risk_table",
        engine,
        if_exists="replace",
        index=False
    )

    print(
        "PASS — customer_risk_table populated"
    )

    print(
        f"Rows written: {len(result_df)}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(result_df):

    print("\n========================================")
    print("PREDICTION DISTRIBUTION")
    print("========================================")

    distribution = (
        result_df["prediction"]
        .value_counts()
    )

    print(
        distribution
    )

    total = len(
        result_df
    )

    likely = int(
        (
            result_df["prediction"]
            == "Likely to churn"
        ).sum()
    )

    percentage = (
        likely / total * 100
    )

    print(
        f"\nLikely to churn: "
        f"{likely}"
    )

    print(
        f"Unlikely to churn: "
        f"{total - likely}"
    )

    print(
        f"Predicted churn percentage: "
        f"{percentage:.2f}%"
    )


# ============================================================
# TOP 10 HIGH-RISK CUSTOMERS
# ============================================================

def print_top_risk(result_df):

    print("\n========================================")
    print("TOP 10 HIGH-RISK CUSTOMERS")
    print("========================================")

    top10 = (
        result_df
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(10)
    )

    print(
        top10[
            [
                "customer_id",
                "risk_score",
                "prediction",
                "confidence"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("ML5 — CUSTOMER BATCH SCORING")
    print("========================================")

    try:

        # ----------------------------------------------------
        # 1. Load
        # ----------------------------------------------------

        df = load_customers()

        # ----------------------------------------------------
        # 2. Score
        # ----------------------------------------------------

        result_df = batch_score(
            df
        )

        # ----------------------------------------------------
        # 3. Save
        # ----------------------------------------------------

        save_results(
            result_df
        )

        # ----------------------------------------------------
        # 4. Summary
        # ----------------------------------------------------

        print_summary(
            result_df
        )

        # ----------------------------------------------------
        # 5. Top customers
        # ----------------------------------------------------

        print_top_risk(
            result_df
        )

        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        print("\n========================================")
        print("ML5 COMPLETE")
        print("========================================")

    except Exception as error:

        print("\n========================================")
        print("ML5 FAILED")
        print("========================================")

        print(
            f"Reason: {error}"
        )

        raise


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()