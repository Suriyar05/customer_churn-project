import pandas as pd

from sqlalchemy import create_engine, text


# ============================================================
# DATABASE CONNECTION
# ============================================================

DATABASE_URL = (
    "mysql+pymysql://root:root@localhost/telecom_intelligence"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


# ============================================================
# DE4 — BUILD FEATURES
# ============================================================

def build_features(engine):

    print("\n========================================")
    print("DE4 — BUILD CUSTOMER FEATURES")
    print("========================================")

    # --------------------------------------------------------
    # 1. READ CLEANED DATA
    # --------------------------------------------------------

    df = pd.read_sql(
        "SELECT * FROM cleaned_customers",
        engine
    )

    print(
        f"Cleaned customers: {len(df)}"
    )

    if df.empty:
        raise ValueError(
            "cleaned_customers table is empty"
        )


    # --------------------------------------------------------
    # 2. CALCULATE MEDIAN MONTHLY CHARGES
    #
    # CP4 definition:
    # high_charge_flag = 1 when monthly_charges
    # is greater than the current batch median
    # --------------------------------------------------------

    median_monthly_charges = df[
        "monthly_charges"
    ].median()

    print(
        f"Monthly charges median: "
        f"{median_monthly_charges:.2f}"
    )


    # --------------------------------------------------------
    # 3. SERVICE COUNT
    #
    # Count subscribed services.
    #
    # Yes/no service columns:
    # phone_service
    # multiple_lines
    # online_security
    # online_backup
    # device_protection
    # tech_support
    # streaming_tv
    # streaming_movies
    # --------------------------------------------------------

    service_columns = [
        "phone_service",
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies"
    ]

    df["service_count"] = 0

    for column in service_columns:

        if column in df.columns:

            df["service_count"] += (
                df[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["yes", "1", "true"])
                .astype(int)
            )


    # --------------------------------------------------------
    # 4. TENURE BUCKET
    #
    # 0–12     = 0
    # 13–24    = 1
    # 25–48    = 2
    # 49+      = 3
    # --------------------------------------------------------

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=[0, 1, 2, 3]
    ).astype(int)


    # --------------------------------------------------------
    # 5. HIGH CHARGE FLAG
    #
    # 1 = above current batch median
    # 0 = median or below
    # --------------------------------------------------------

    df["high_charge_flag"] = (
        df["monthly_charges"] >
        median_monthly_charges
    ).astype(int)


    # --------------------------------------------------------
    # 6. LONG TERM CUSTOMER
    #
    # 1 = tenure >= 24 months
    # 0 = otherwise
    # --------------------------------------------------------

    df["is_long_term_customer"] = (
        df["tenure"] >= 24
    ).astype(int)


    # --------------------------------------------------------
    # 7. STREAMING BUNDLE
    #
    # Customer has both StreamingTV and StreamingMovies
    # --------------------------------------------------------

    streaming_tv = (
        df["streaming_tv"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    streaming_movies = (
        df["streaming_movies"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["has_streaming_bundle"] = (
        (streaming_tv == "yes") &
        (streaming_movies == "yes")
    ).astype(int)


    # --------------------------------------------------------
    # 8. AUTO PAY FLAG
    #
    # Payment methods:
    # Bank transfer (automatic)
    # Credit card (automatic)
    # --------------------------------------------------------

    payment = (
        df["payment_method"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["auto_pay_flag"] = (
        payment.isin([
            "bank transfer (automatic)",
            "credit card (automatic)"
        ])
    ).astype(int)


    # ========================================================
    # 9. CREATE FEATURE TABLE
    # ========================================================

    feature_columns = [
        "customer_id",
        "service_count",
        "tenure_bucket",
        "high_charge_flag",
        "is_long_term_customer",
        "has_streaming_bundle",
        "auto_pay_flag",
        "monthly_charges",
        "total_charges",
        "churn"
    ]

    features = df[feature_columns].copy()


    # ========================================================
    # 10. FORCE CORRECT DATA TYPES
    # ========================================================

    integer_columns = [
        "service_count",
        "tenure_bucket",
        "high_charge_flag",
        "is_long_term_customer",
        "has_streaming_bundle",
        "auto_pay_flag",
        "churn"
    ]

    for column in integer_columns:

        features[column] = (
            pd.to_numeric(
                features[column],
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
        )


    features["monthly_charges"] = pd.to_numeric(
        features["monthly_charges"],
        errors="coerce"
    )

    features["total_charges"] = pd.to_numeric(
        features["total_charges"],
        errors="coerce"
    )


    # ========================================================
    # 11. NULL CHECK
    # ========================================================

    print("\nFeature NULL check:")

    feature_check_columns = [
        "service_count",
        "tenure_bucket",
        "high_charge_flag",
        "is_long_term_customer",
        "has_streaming_bundle",
        "auto_pay_flag"
    ]

    for column in feature_check_columns:

        null_count = features[column].isna().sum()

        if null_count == 0:

            print(
                f"PASS — {column}: "
                f"0 NULLs"
            )

        else:

            print(
                f"FAIL — {column}: "
                f"{null_count} NULLs"
            )

            raise AssertionError(
                f"{column} contains NULL values"
            )


    # ========================================================
    # 12. WRITE FEATURE TABLE
    # ========================================================

    features.to_sql(
        "customer_ml_features",
        engine,
        if_exists="replace",
        index=False
    )

    print(
        "\nPASS — customer_ml_features created"
    )


    # ========================================================
    # 13. ROW COUNT VALIDATION
    # ========================================================

    cleaned_count = len(df)
    feature_count = len(features)

    print("\nRow Count Validation")

    print(
        f"cleaned_customers : {cleaned_count}"
    )

    print(
        f"customer_ml_features : {feature_count}"
    )

    if cleaned_count != feature_count:

        raise AssertionError(
            "Feature table row count does not "
            "match cleaned_customers"
        )

    print(
        "PASS — row counts match"
    )


    # ========================================================
    # 14. SCHEMA VALIDATION
    # ========================================================

    required_features = [
        "service_count",
        "tenure_bucket",
        "high_charge_flag",
        "is_long_term_customer",
        "has_streaming_bundle",
        "auto_pay_flag"
    ]

    print("\nSchema Validation")

    missing_features = [
        column
        for column in required_features
        if column not in features.columns
    ]

    if missing_features:

        raise AssertionError(
            f"Missing features: {missing_features}"
        )

    print(
        "PASS — all required features present"
    )


    # ========================================================
    # 15. DISPLAY SAMPLE
    # ========================================================

    print("\nFeature Sample")

    print(
        features[
            [
                "customer_id",
                "service_count",
                "tenure_bucket",
                "high_charge_flag",
                "is_long_term_customer",
                "has_streaming_bundle",
                "auto_pay_flag"
            ]
        ].head(3).to_string(index=False)
    )


    return features


# ============================================================
# VERIFY DATABASE TABLE
# ============================================================

def verify_feature_table(engine):

    print("\n========================================")
    print("FEATURE TABLE VERIFICATION")
    print("========================================")

    with engine.connect() as connection:

        count = connection.execute(
            text(
                "SELECT COUNT(*) "
                "FROM customer_ml_features"
            )
        ).scalar()

        print(
            f"customer_ml_features rows: {count}"
        )

        result = connection.execute(
            text("""
                SELECT
                    customer_id,
                    service_count,
                    tenure_bucket,
                    high_charge_flag,
                    is_long_term_customer,
                    has_streaming_bundle,
                    auto_pay_flag,
                    monthly_charges,
                    total_charges,
                    churn
                FROM customer_ml_features
                LIMIT 3
            """)
        )

        rows = result.mappings().all()

        print("\nSample rows:")

        for row in rows:
            print(dict(row))


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        build_features(engine)

        verify_feature_table(engine)

        print("\n========================================")
        print("DE4 COMPLETE")
        print("========================================")

    except Exception as e:

        print("\n========================================")
        print("DE4 FAILED")
        print("========================================")

        print(
            f"Reason: {e}"
        )

        raise


if __name__ == "__main__":
    main()