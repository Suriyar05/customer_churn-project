import json
from pathlib import Path
from datetime import datetime

import pandas as pd

from sqlalchemy import create_engine, text


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

QUALITY_REPORT_PATH = (
    BASE_DIR / "data" / "quality_report.json"
)


# ------------------------------------------------------------
# DATABASE CONFIGURATION
# ------------------------------------------------------------

DB_USER = "root"
DB_PASSWORD = "root"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "telecom_intelligence"


DATABASE_URL = (
    f"mysql+pymysql://"
    f"{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}"
    f"/{DB_NAME}"
)


# ============================================================
# CREATE DATABASE ENGINE
# ============================================================

def create_db_engine():

    print("\n========================================")
    print("DATABASE CONNECTION")
    print("========================================")

    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True
    )

    with engine.connect() as connection:

        connection.execute(
            text("SELECT 1")
        )

    print(
        "PASS — Database connection successful."
    )

    return engine


# ============================================================
# LOAD DATA
# ============================================================

def load_cleaned_data(engine):

    print("\n========================================")
    print("LOAD CLEANED DATA")
    print("========================================")

    query = text("""
        SELECT *
        FROM cleaned_customers
    """)

    df = pd.read_sql(
        query,
        engine
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    return df


# ============================================================
# QUALITY CHECK 1
# NULL RATE
# ============================================================

def check_null_rate(
    df,
    col,
    threshold
):

    null_count = (
        df[col]
        .isna()
        .sum()
    )

    total_rows = len(df)

    if total_rows == 0:

        null_rate = 100.0

    else:

        null_rate = (
            null_count
            / total_rows
            * 100
        )


    passed = (
        null_rate <= threshold
    )


    return {
        "check_name":
            f"Null rate — {col}",

        "status":
            "PASS" if passed else "FAIL",

        "value_found":
            round(null_rate, 4),

        "threshold":
            f"<= {threshold}%",

        "severity":
            "CRITICAL"
    }


# ============================================================
# QUALITY CHECK 2
# VALUE RANGE
# ============================================================

def check_value_range(
    df,
    col,
    minimum,
    maximum=None
):

    # Convert values to numeric where possible

    values = pd.to_numeric(
        df[col],
        errors="coerce"
    )


    invalid_mask = (
        values < minimum
    )


    if maximum is not None:

        invalid_mask = (
            invalid_mask
            | (values > maximum)
        )


    invalid_count = (
        invalid_mask
        .sum()
    )


    passed = (
        invalid_count == 0
    )


    if maximum is None:

        threshold_text = (
            f">= {minimum}"
        )

    else:

        threshold_text = (
            f"{minimum}–{maximum}"
        )


    return {
        "check_name":
            f"Value range — {col}",

        "status":
            "PASS" if passed else "FAIL",

        "value_found":
            int(invalid_count),

        "threshold":
            threshold_text,

        "severity":
            "CRITICAL"
    }


# ============================================================
# QUALITY CHECK 3
# ALLOWED VALUES
# ============================================================

def check_allowed_values(
    df,
    col,
    allowed_set
):

    invalid_mask = (
        ~df[col].isin(
            allowed_set
        )
    )


    invalid_count = (
        invalid_mask
        .sum()
    )


    passed = (
        invalid_count == 0
    )


    return {
        "check_name":
            f"Allowed values — {col}",

        "status":
            "PASS" if passed else "FAIL",

        "value_found":
            int(invalid_count),

        "threshold":
            sorted(
                list(allowed_set)
            ),

        "severity":
            "CRITICAL"
    }


# ============================================================
# QUALITY CHECK 4
# ROW COUNT
# ============================================================

def check_row_count(
    df,
    expected_min
):

    actual_count = len(df)

    passed = (
        actual_count >= expected_min
    )


    return {
        "check_name":
            "Row count",

        "status":
            "PASS" if passed else "FAIL",

        "value_found":
            actual_count,

        "threshold":
            f">= {expected_min}",

        "severity":
            "CRITICAL"
    }


# ============================================================
# QUALITY CHECK 5
# DUPLICATES
# ============================================================

def check_no_duplicates(
    df,
    key_col
):

    duplicate_count = (
        df[key_col]
        .duplicated()
        .sum()
    )


    passed = (
        duplicate_count == 0
    )


    return {
        "check_name":
            f"Duplicate check — {key_col}",

        "status":
            "PASS" if passed else "FAIL",

        "value_found":
            int(duplicate_count),

        "threshold":
            0,

        "severity":
            "CRITICAL"
    }


# ============================================================
# QUALITY CHECK 6
# CHURN DISTRIBUTION
# ============================================================

def check_churn_distribution(
    df
):

    # Make sure churn is numeric

    churn = pd.to_numeric(
        df["churn"],
        errors="coerce"
    )


    invalid_count = (
        (~churn.isin([0, 1]))
        .sum()
    )


    churn_1_count = (
        (churn == 1)
        .sum()
    )


    churn_0_count = (
        (churn == 0)
        .sum()
    )


    passed = (
        invalid_count == 0
        and churn_1_count > 0
        and churn_0_count > 0
    )


    return {
        "check_name":
            "Churn distribution",

        "status":
            "PASS" if passed else "FAIL",

        "value_found": {
            "churn_0":
                int(churn_0_count),

            "churn_1":
                int(churn_1_count),

            "invalid":
                int(invalid_count)
        },

        "threshold":
            "Values must be 0/1 and both classes must exist",

        "severity":
            "CRITICAL"
    }


# ============================================================
# RUN ALL QUALITY CHECKS
# ============================================================

def run_quality_checks(df):

    print("\n========================================")
    print("RUN DATA QUALITY CHECKS")
    print("========================================")


    results = []


    # --------------------------------------------------------
    # 1. Monthly charges NULL
    # --------------------------------------------------------

    result = check_null_rate(
        df,
        "monthly_charges",
        0
    )

    results.append(result)


    # --------------------------------------------------------
    # 2. Tenure NULL
    # --------------------------------------------------------

    result = check_null_rate(
        df,
        "tenure",
        0
    )

    results.append(result)


    # --------------------------------------------------------
    # 3. Tenure range
    # --------------------------------------------------------

    result = check_value_range(
        df,
        "tenure",
        0,
        100
    )

    results.append(result)


    # --------------------------------------------------------
    # 4. Monthly charges > 0
    # --------------------------------------------------------

    result = check_value_range(
        df,
        "monthly_charges",
        0
    )

    results.append(result)


    # --------------------------------------------------------
    # 5. Contract allowed values
    # --------------------------------------------------------

    allowed_contracts = {
        "Month-to-month",
        "One year",
        "Two year"
    }


    result = check_allowed_values(
        df,
        "contract",
        allowed_contracts
    )

    results.append(result)


    # --------------------------------------------------------
    # 6. Row count
    # --------------------------------------------------------

    result = check_row_count(
        df,
        7000
    )

    results.append(result)


    # --------------------------------------------------------
    # 7. Duplicate customer IDs
    # --------------------------------------------------------

    result = check_no_duplicates(
        df,
        "customer_id"
    )

    results.append(result)


    # --------------------------------------------------------
    # 8. Churn distribution
    # --------------------------------------------------------

    result = check_churn_distribution(
        df
    )

    results.append(result)


    return results


# ============================================================
# PRINT QUALITY REPORT
# ============================================================

def print_quality_report(
    results
):

    print("\n========================================")
    print("DATA QUALITY REPORT")
    print("========================================")


    for result in results:

        status = result["status"]

        severity = result["severity"]

        check_name = result["check_name"]

        value_found = result["value_found"]

        threshold = result["threshold"]


        print(
            f"{status} "
            f"[{severity}] "
            f"{check_name}"
        )

        print(
            f"    Value: {value_found}"
        )

        print(
            f"    Threshold: {threshold}"
        )


# ============================================================
# SAVE JSON REPORT
# ============================================================

def save_quality_report(
    results
):

    print("\n========================================")
    print("SAVE QUALITY REPORT")
    print("========================================")


    report = {

        "generated_at":
            datetime.now()
            .isoformat(),

        "total_checks":
            len(results),

        "passed":
            sum(
                1
                for result in results
                if result["status"] == "PASS"
            ),

        "failed":
            sum(
                1
                for result in results
                if result["status"] == "FAIL"
            ),

        "checks":
            results
    }


    QUALITY_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        QUALITY_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            default=str
        )


    print(
        f"PASS — Report written to:"
    )

    print(
        QUALITY_REPORT_PATH
    )


# ============================================================
# QUALITY GATE
# ============================================================

def enforce_quality_gate(
    results
):

    print("\n========================================")
    print("QUALITY GATE")
    print("========================================")


    critical_failures = [

        result
        for result in results

        if (
            result["status"] == "FAIL"
            and result["severity"] == "CRITICAL"
        )
    ]


    if critical_failures:

        print(
            "QUALITY GATE FAILED."
        )

        print(
            f"Critical failures: "
            f"{len(critical_failures)}"
        )


        for failure in critical_failures:

            print(
                f"  - "
                f"{failure['check_name']}"
            )


        raise RuntimeError(
            "Pipeline stopped because "
            "one or more CRITICAL "
            "data-quality checks failed."
        )


    print(
        "QUALITY GATE PASSED."
    )

    print(
        "All critical checks passed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("DE7 — DATA QUALITY PIPELINE")
    print("========================================")


    engine = (
        create_db_engine()
    )


    # --------------------------------------------------------
    # Load cleaned data
    # --------------------------------------------------------

    df = (
        load_cleaned_data(
            engine
        )
    )


    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [

        "customer_id",
        "tenure",
        "monthly_charges",
        "contract",
        "churn"
    ]


    missing_columns = [

        col
        for col in required_columns
        if col not in df.columns
    ]


    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )


    # --------------------------------------------------------
    # Run checks
    # --------------------------------------------------------

    results = (
        run_quality_checks(
            df
        )
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_quality_report(
        results
    )


    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    save_quality_report(
        results
    )


    # --------------------------------------------------------
    # Stop pipeline if critical check failed
    # --------------------------------------------------------

    enforce_quality_gate(
        results
    )


    print("\n========================================")
    print("DE7 COMPLETE")
    print("========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()