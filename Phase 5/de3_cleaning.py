import pandas as pd

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from customer_cleaner import CustomerCleaner


# ============================================================
# DATABASE
# ============================================================

DATABASE_URL = (
    "mysql+pymysql://root:root@localhost/telecom_intelligence"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


# ============================================================
# REQUIRED CLEANED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "customer_id",
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    "contract",
    "payment_method",
    "monthly_charges",
    "total_charges",
    "churn"
]


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_columns(df):

    """
    Convert possible raw / CustomerCleaner column names
    into the standard names expected by DE3.
    """

    mapping = {

        # Customer ID
        "customerID": "customer_id",
        "CustomerID": "customer_id",
        "customer_id": "customer_id",
        "customer_i_d": "customer_id",

        # Basic customer attributes
        "gender": "gender",
        "Gender": "gender",

        "SeniorCitizen": "senior_citizen",
        "senior_citizen": "senior_citizen",
        "senior_citizen_flag": "senior_citizen",

        "Partner": "partner",
        "partner": "partner",

        "Dependents": "dependents",
        "dependents": "dependents",

        # Tenure
        "tenure": "tenure",
        "Tenure": "tenure",

        # Services
        "PhoneService": "phone_service",
        "phone_service": "phone_service",

        "MultipleLines": "multiple_lines",
        "multiple_lines": "multiple_lines",

        "InternetService": "internet_service",
        "internet_service": "internet_service",

        "OnlineSecurity": "online_security",
        "online_security": "online_security",

        "OnlineBackup": "online_backup",
        "online_backup": "online_backup",

        "DeviceProtection": "device_protection",
        "device_protection": "device_protection",

        "TechSupport": "tech_support",
        "tech_support": "tech_support",

        "StreamingTV": "streaming_tv",
        "streaming_tv": "streaming_tv",

        "StreamingMovies": "streaming_movies",
        "streaming_movies": "streaming_movies",

        # Contract
        "Contract": "contract",
        "contract": "contract",
        "contract_type": "contract",

        # Billing
        "PaperlessBilling": "paperless_billing",
        "paperless_billing": "paperless_billing",

        "PaymentMethod": "payment_method",
        "payment_method": "payment_method",

        # Charges
        "MonthlyCharges": "monthly_charges",
        "monthly_charges": "monthly_charges",

        "TotalCharges": "total_charges",
        "total_charges": "total_charges",

        # Target
        "Churn": "churn",
        "churn": "churn"
    }

    rename_dict = {}

    for column in df.columns:

        if column in mapping:

            rename_dict[column] = mapping[column]

    df = df.rename( columns=rename_dict )

    return df


# ============================================================
# CLEAN STAGING
# ============================================================

def clean_staging(engine):

    print("\n========================================")
    print("DE3 — CLEAN STAGING")
    print("========================================")

    # --------------------------------------------------------
    # READ STAGING
    # --------------------------------------------------------

    df = pd.read_sql(
        "SELECT * FROM stg_customer_raw",
        engine
    )

    staging_count = len(df)

    print(
        f"Staging rows: {staging_count}"
    )

    # --------------------------------------------------------
    # RUN CUSTOMER CLEANER
    # --------------------------------------------------------

    print("\nRunning CustomerCleaner...")

    cleaner = CustomerCleaner(df)

    cleaned_df = cleaner.clean()

    print("\nCLEANED CUSTOMER COLUMNS:")
    print(cleaned_df.columns.tolist())

    if cleaned_df is None:

        raise ValueError(
            "CustomerCleaner.clean() returned None"
        )

    print(
        f"Rows after CustomerCleaner: "
        f"{len(cleaned_df)}"
    )

    # --------------------------------------------------------
    # SHOW ORIGINAL CLEANED COLUMNS
    # --------------------------------------------------------

    print("\nColumns returned by CustomerCleaner:")

    print(
        cleaned_df.columns.tolist()
    )

    # --------------------------------------------------------
    # NORMALIZE COLUMN NAMES
    # --------------------------------------------------------

    cleaned_df = normalize_columns(
        cleaned_df
    )

    print("\nNormalized columns:")

    print(
        cleaned_df.columns.tolist()
    )

    # --------------------------------------------------------
    # REQUIRED COLUMN CHECK
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in cleaned_df.columns
    ]

    if missing_columns:

        print("\nERROR — Missing columns:")

        print(
            missing_columns
        )

        raise ValueError(
            "CustomerCleaner output is missing "
            f"required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # CONVERT NUMERIC COLUMNS
    # --------------------------------------------------------

    cleaned_df["tenure"] = pd.to_numeric(
        cleaned_df["tenure"],
        errors="coerce"
    )

    cleaned_df["monthly_charges"] = pd.to_numeric(
        cleaned_df["monthly_charges"],
        errors="coerce"
    )

    cleaned_df["total_charges"] = pd.to_numeric(
        cleaned_df["total_charges"],
        errors="coerce"
    )

    cleaned_df["senior_citizen"] = pd.to_numeric(
        cleaned_df["senior_citizen"],
        errors="coerce"
    )

    cleaned_df["churn"] = pd.to_numeric(
        cleaned_df["churn"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # ROW COUNT CHECK
    # --------------------------------------------------------

    cleaned_count = len(cleaned_df)

    print("\nRow Count Check")

    print(
        f"Staging : {staging_count}"
    )

    print(
        f"Cleaned : {cleaned_count}"
    )

    if cleaned_count == staging_count:

        print(
            "PASS — row counts match"
        )

    else:

        print(
            "FAIL — row counts do not match"
        )

        raise AssertionError(
            "Cleaning stage dropped rows"
        )

    # --------------------------------------------------------
    # WRITE CLEANED TABLE
    # --------------------------------------------------------

    cleaned_df.to_sql(
        "cleaned_customers",
        engine,
        if_exists="replace",
        index=False
    )

    print(
        "\nPASS — cleaned_customers "
        "table created"
    )

    return cleaned_df


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

def run_quality_checks(cleaned_df):

    print("\n========================================")
    print("DATA QUALITY REPORT")
    print("========================================")

    # --------------------------------------------------------
    # CUSTOMER ID NULL CHECK
    # --------------------------------------------------------

    null_customer_id = (
        cleaned_df["customer_id"]
        .isna()
        .sum()
    )

    if null_customer_id == 0:

        print(
            f"PASS — customer_id NULL count = "
            f"{null_customer_id}"
        )

    else:

        print(
            f"FAIL — customer_id NULL count = "
            f"{null_customer_id}"
        )

        raise AssertionError(
            "customer_id contains NULL values"
        )

    # --------------------------------------------------------
    # MONTHLY CHARGES NULL CHECK
    # --------------------------------------------------------

    null_monthly = (
        cleaned_df["monthly_charges"]
        .isna()
        .sum()
    )

    if null_monthly == 0:

        print(
            f"PASS — monthly_charges NULL count = "
            f"{null_monthly}"
        )

    else:

        print(
            f"FAIL — monthly_charges NULL count = "
            f"{null_monthly}"
        )

        raise AssertionError(
            "monthly_charges contains NULL values"
        )

    # --------------------------------------------------------
    # CHURN CHECK
    # --------------------------------------------------------

    invalid_churn = (
        ~cleaned_df["churn"]
        .isin([0, 1])
    ).sum()

    if invalid_churn == 0:

        print(
            "PASS — churn contains only 0 and 1"
        )

    else:

        print(
            f"FAIL — invalid churn count = "
            f"{invalid_churn}"
        )

        print(
            "Invalid values:",
            cleaned_df.loc[
                ~cleaned_df["churn"].isin([0, 1]),
                "churn"
            ].unique()
        )

        raise AssertionError(
            "churn contains values other than 0 or 1"
        )

    # --------------------------------------------------------
    # DUPLICATE CUSTOMER ID CHECK
    # --------------------------------------------------------

    duplicate_ids = (
        cleaned_df["customer_id"]
        .duplicated()
        .sum()
    )

    if duplicate_ids == 0:

        print(
            "PASS — duplicate customer_id count = 0"
        )

    else:

        print(
            f"WARNING — duplicate customer_id count = "
            f"{duplicate_ids}"
        )

    print(
        "\nAll required data-quality checks passed."
    )


# ============================================================
# BUILD CURATED TABLES
# ============================================================

def build_curated_tables(engine):

    print("\n========================================")
    print("BUILDING CURATED TABLES")
    print("========================================")

    with engine.begin() as connection:

        # ====================================================
        # 1. DIM CONTRACT
        # ====================================================

        print("\n1. Building dim_contract...")

        connection.execute(
            text("""
                INSERT INTO dim_contract
                (
                    contract_name
                )

                SELECT DISTINCT
                    contract

                FROM cleaned_customers

                WHERE contract IS NOT NULL
                  AND TRIM(contract) <> ''

                ON DUPLICATE KEY UPDATE
                    contract_name = VALUES(contract_name)
            """)
        )

        print("PASS — dim_contract populated")


        # ====================================================
        # 2. DIM PAYMENT
        # ====================================================

        print("\n2. Building dim_payment...")

        connection.execute(
            text("""
                INSERT INTO dim_payment
                (
                    payment_method
                )

                SELECT DISTINCT
                    payment_method

                FROM cleaned_customers

                WHERE payment_method IS NOT NULL
                  AND TRIM(payment_method) <> ''

                ON DUPLICATE KEY UPDATE
                    payment_method = VALUES(payment_method)
            """)
        )

        print("PASS — dim_payment populated")


        # ====================================================
        # 3. CUSTOMERS
        # ====================================================

        print("\n3. Building customers...")

        connection.execute(
            text("""
                INSERT INTO customers
                (
                    customer_id,
                    gender,
                    senior_citizen,
                    partner,
                    dependents
                )

                SELECT
                    customer_id,
                    gender,
                    senior_citizen,
                    partner,
                    dependents

                FROM cleaned_customers

                ON DUPLICATE KEY UPDATE
                    gender = VALUES(gender),
                    senior_citizen = VALUES(senior_citizen),
                    partner = VALUES(partner),
                    dependents = VALUES(dependents)
            """)
        )

        print("PASS — customers populated")


        # ====================================================
        # 4. FACT CUSTOMER ACCOUNT
        # ====================================================

        print("\n4. Building fact_customer_account...")

        connection.execute(
            text("""
                INSERT INTO fact_customer_account
                (
                    customer_id,
                    contract_id,
                    payment_id,
                    tenure,
                    monthly_charges,
                    total_charges,
                    churn
                )

                SELECT
                    cc.customer_id,
                    dc.contract_id,
                    dp.payment_id,
                    cc.tenure,
                    cc.monthly_charges,
                    cc.total_charges,
                    cc.churn

                FROM cleaned_customers cc

                LEFT JOIN dim_contract dc
                    ON cc.contract = dc.contract_name

                LEFT JOIN dim_payment dp
                    ON cc.payment_method = dp.payment_method

                ON DUPLICATE KEY UPDATE
                    contract_id = VALUES(contract_id),
                    payment_id = VALUES(payment_id),
                    tenure = VALUES(tenure),
                    monthly_charges = VALUES(monthly_charges),
                    total_charges = VALUES(total_charges),
                    churn = VALUES(churn)
            """)
        )

        print("PASS — fact_customer_account populated")

    print("\n========================================")
    print("CURATED TABLES COMPLETE")
    print("========================================")



# ============================================================
# VERIFY CURATED TABLES
# ============================================================

def verify_curated_tables(engine):

    print("\n========================================")
    print("CURATED TABLE COUNTS")
    print("========================================")

    tables = [
        "dim_contract",
        "dim_payment",
        "customers",
        "fact_customer_account"
    ]

    with engine.connect() as connection:

        for table in tables:

            result = connection.execute(
                text(
                    f"""
                    SELECT COUNT(*)
                    FROM {table}
                    """
                )
            )

            count = result.scalar()

            print(
                f"{table}: {count}"
            )

    # --------------------------------------------------------
    # Expected checks
    # --------------------------------------------------------

    customers_count = pd.read_sql(
        "SELECT COUNT(*) AS count FROM customers",
        engine
    )["count"].iloc[0]

    fact_count = pd.read_sql(
        """
        SELECT COUNT(*) AS count
        FROM fact_customer_account
        """,
        engine
    )["count"].iloc[0]

    if customers_count == 7043:

        print(
            "PASS — customers contains 7043 rows"
        )

    else:

        print(
            f"FAIL — customers contains "
            f"{customers_count} rows"
        )

    if fact_count == 7043:

        print(
            "PASS — fact_customer_account "
            "contains 7043 rows"
        )

    else:

        print(
            f"FAIL — fact_customer_account "
            f"contains {fact_count} rows"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # STEP 1
        # staging → cleaned
        # ----------------------------------------------------

        cleaned_df = clean_staging(
            engine
        )

        # ----------------------------------------------------
        # STEP 2
        # quality checks
        # ----------------------------------------------------

        run_quality_checks(
            cleaned_df
        )

        # ----------------------------------------------------
        # STEP 3
        # cleaned → curated
        # ----------------------------------------------------

        build_curated_tables(
            engine
        )

        # ----------------------------------------------------
        # STEP 4
        # verification
        # ----------------------------------------------------

        verify_curated_tables(
            engine
        )

        print("\n========================================")
        print("DE3 COMPLETE")
        print("========================================")

    except Exception as e:

        print("\n========================================")
        print("DE3 FAILED")
        print("========================================")

        print(
            f"Reason: {e}"
        )

        raise


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()