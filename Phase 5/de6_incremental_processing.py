import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

from sqlalchemy import (
    create_engine,
    text
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "customer_churn.csv"
)


# ------------------------------------------------------------
# CHANGE THESE IF YOUR DATABASE DETAILS ARE DIFFERENT
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
# DATABASE ENGINE
# ============================================================

def create_db_engine():

    print("\n========================================")
    print("DATABASE CONNECTION")
    print("========================================")

    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True
    )

    # Test connection

    with engine.connect() as connection:

        connection.execute(
            text("SELECT 1")
        )

    print(
        "PASS — Database connection successful."
    )

    return engine


# ============================================================
# CREATE PIPELINE LOG TABLE
# ============================================================

def create_pipeline_log_table(engine):

    print("\n========================================")
    print("CREATE PIPELINE LOG TABLE")
    print("========================================")

    sql = text("""
        CREATE TABLE IF NOT EXISTS pipeline_run_log (

            run_id INT AUTO_INCREMENT PRIMARY KEY,

            file_name VARCHAR(255) NOT NULL,

            rows_inserted INT NOT NULL,

            rows_updated INT NOT NULL,

            rows_unchanged INT NOT NULL,

            run_timestamp DATETIME NOT NULL

        )
    """)

    with engine.begin() as connection:

        connection.execute(sql)

    print(
        "PASS — pipeline_run_log is ready."
    )


# ============================================================
# READ CSV
# ============================================================

def read_csv():

    print("\n========================================")
    print("READ DAILY EXTRACT")
    print("========================================")

    print(
        f"File: {CSV_PATH}"
    )

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH}"
        )

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"Rows read: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    return df


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

def standardize_columns(df):

    df = df.copy()

    # Match the cleaned naming convention

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    rename_map = {

        "customerid":
            "customer_id",

        "seniorcitizen":
            "senior_citizen",

        "monthlycharges":
            "monthly_charges",

        "totalcharges":
            "total_charges"
    }

    df = df.rename(
        columns=rename_map
    )

    return df


# ============================================================
# NORMALIZE CUSTOMER DATA
# ============================================================

def prepare_customer_data(df):

    print("\n========================================")
    print("PREPARE CUSTOMER DATA")
    print("========================================")

    df = standardize_columns(
        df
    )

    required_columns = [

        "customer_id",
        "gender",
        "senior_citizen",
        "partner",
        "dependents"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )


    # --------------------------------------------------------
    # Normalize gender
    # --------------------------------------------------------

    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.strip()
    )


    # --------------------------------------------------------
    # Normalize SeniorCitizen
    # --------------------------------------------------------

    df["senior_citizen"] = (
        pd.to_numeric(
            df["senior_citizen"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )


    # --------------------------------------------------------
    # Normalize Partner
    # --------------------------------------------------------

    df["partner"] = (
        df["partner"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0
        })
        .fillna(0)
        .astype(int)
    )


    # --------------------------------------------------------
    # Normalize Dependents
    # --------------------------------------------------------

    df["dependents"] = (
        df["dependents"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0
        })
        .fillna(0)
        .astype(int)
    )


    # --------------------------------------------------------
    # Remove invalid customer IDs
    # --------------------------------------------------------

    df = df[
        df["customer_id"].notna()
    ]

    df["customer_id"] = (
        df["customer_id"]
        .astype(str)
        .str.strip()
    )


    # --------------------------------------------------------
    # Keep only columns matching customers table
    # --------------------------------------------------------

    df = df[
        [
            "customer_id",
            "gender",
            "senior_citizen",
            "partner",
            "dependents"
        ]
    ]


    # --------------------------------------------------------
    # Remove duplicate customer IDs inside the incoming file
    # --------------------------------------------------------

    duplicate_count = (
        df["customer_id"]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        print(
            f"WARNING — {duplicate_count} "
            f"duplicate customer IDs found."
        )

        df = (
            df
            .drop_duplicates(
                subset=["customer_id"],
                keep="last"
            )
        )


    print(
        f"Prepared rows: {len(df)}"
    )

    return df


# ============================================================
# GET EXISTING CUSTOMERS
# ============================================================

def get_existing_customers(
    engine,
    customer_ids
):

    if not customer_ids:

        return pd.DataFrame(
            columns=[
                "customer_id",
                "gender",
                "senior_citizen",
                "partner",
                "dependents"
            ]
        )


    # --------------------------------------------------------
    # Read existing customers.
    #
    # For 7,043 rows this is completely fine.
    # In a very large production system, this would be
    # optimized/batched differently.
    # --------------------------------------------------------

    existing = pd.read_sql(
        text("""
            SELECT
                customer_id,
                gender,
                senior_citizen,
                partner,
                dependents
            FROM customers
        """),
        engine
    )

    return existing


# ============================================================
# CLASSIFY ROWS
# ============================================================

def classify_changes(
    incoming_df,
    existing_df
):

    print("\n========================================")
    print("CLASSIFY CHANGES")
    print("========================================")


    # --------------------------------------------------------
    # No existing customers
    # --------------------------------------------------------

    if existing_df.empty:

        incoming_df = incoming_df.copy()

        incoming_df["_change_type"] = "INSERT"

        print(
            f"INSERT: {len(incoming_df)}"
        )

        print(
            "UPDATE: 0"
        )

        print(
            "UNCHANGED: 0"
        )

        return incoming_df


    # --------------------------------------------------------
    # Set customer_id as index
    # --------------------------------------------------------

    incoming = (
        incoming_df
        .set_index("customer_id")
    )

    existing = (
        existing_df
        .set_index("customer_id")
    )


    # IDs that are completely new

    new_ids = (
        incoming.index
        .difference(
            existing.index
        )
    )


    # IDs that already exist

    existing_ids = (
        incoming.index
        .intersection(
            existing.index
        )
    )


    # --------------------------------------------------------
    # Detect changed records
    # --------------------------------------------------------

    changed_ids = []

    unchanged_ids = []


    compare_columns = [

        "gender",
        "senior_citizen",
        "partner",
        "dependents"
    ]


    for customer_id in existing_ids:

        incoming_row = (
            incoming.loc[
                customer_id
            ]
        )

        existing_row = (
            existing.loc[
                customer_id
            ]
        )


        changed = False


        for column in compare_columns:

            incoming_value = (
                incoming_row[column]
            )

            existing_value = (
                existing_row[column]
            )


            # Normalize string comparison

            if isinstance(
                incoming_value,
                str
            ):

                incoming_value = (
                    incoming_value.strip()
                )


            if isinstance(
                existing_value,
                str
            ):

                existing_value = (
                    existing_value.strip()
                )


            if incoming_value != existing_value:

                changed = True

                break


        if changed:

            changed_ids.append(
                customer_id
            )

        else:

            unchanged_ids.append(
                customer_id
            )


    # --------------------------------------------------------
    # Create classification dataframe
    # --------------------------------------------------------

    result = incoming_df.copy()

    result["_change_type"] = "UNCHANGED"


    result.loc[
        result["customer_id"].isin(
            new_ids
        ),
        "_change_type"
    ] = "INSERT"


    result.loc[
        result["customer_id"].isin(
            changed_ids
        ),
        "_change_type"
    ] = "UPDATE"


    print(
        f"INSERT: "
        f"{len(new_ids)}"
    )

    print(
        f"UPDATE: "
        f"{len(changed_ids)}"
    )

    print(
        f"UNCHANGED: "
        f"{len(unchanged_ids)}"
    )


    return result


# ============================================================
# UPSERT CUSTOMERS
# ============================================================

def upsert_customers(
    engine,
    classified_df
):

    print("\n========================================")
    print("UPSERT CUSTOMERS")
    print("========================================")


    # --------------------------------------------------------
    # MySQL syntax:
    #
    # INSERT ... ON DUPLICATE KEY UPDATE
    #
    # customer_id is PRIMARY KEY.
    # --------------------------------------------------------

    upsert_sql = text("""
        INSERT INTO customers
        (
            customer_id,
            gender,
            senior_citizen,
            partner,
            dependents
        )
        VALUES
        (
            :customer_id,
            :gender,
            :senior_citizen,
            :partner,
            :dependents
        )

        ON DUPLICATE KEY UPDATE

            gender =
                VALUES(gender),

            senior_citizen =
                VALUES(senior_citizen),

            partner =
                VALUES(partner),

            dependents =
                VALUES(dependents)
    """)


    rows_to_upsert = (
        classified_df[
            classified_df["_change_type"]
            .isin([
                "INSERT",
                "UPDATE"
            ])
        ]
        .copy()
    )


    if rows_to_upsert.empty:

        print(
            "No INSERT or UPDATE required."
        )

        return


    records = (
        rows_to_upsert[
            [
                "customer_id",
                "gender",
                "senior_citizen",
                "partner",
                "dependents"
            ]
        ]
        .to_dict(
            orient="records"
        )
    )


    with engine.begin() as connection:

        connection.execute(
            upsert_sql,
            records
        )


    print(
        f"Upserted rows: "
        f"{len(records)}"
    )


# ============================================================
# LOG PIPELINE RUN
# ============================================================

def log_pipeline_run(
    engine,
    file_name,
    rows_inserted,
    rows_updated,
    rows_unchanged
):

    sql = text("""
        INSERT INTO pipeline_run_log
        (
            file_name,
            rows_inserted,
            rows_updated,
            rows_unchanged,
            run_timestamp
        )
        VALUES
        (
            :file_name,
            :rows_inserted,
            :rows_updated,
            :rows_unchanged,
            :run_timestamp
        )
    """)


    with engine.begin() as connection:

        connection.execute(
            sql,
            {
                "file_name":
                    file_name,

                "rows_inserted":
                    rows_inserted,

                "rows_updated":
                    rows_updated,

                "rows_unchanged":
                    rows_unchanged,

                "run_timestamp":
                    datetime.now()
            }
        )


    print(
        "PASS — Pipeline run logged."
    )


# ============================================================
# VERIFY CUSTOMER COUNT
# ============================================================

def get_customer_count(engine):

    with engine.connect() as connection:

        result = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM customers
            """)
        )

        return result.scalar()


# ============================================================
# MAIN INCREMENTAL PIPELINE
# ============================================================

def process_incremental():

    print("\n========================================")
    print("DE6 — INCREMENTAL PROCESSING")
    print("========================================")


    # --------------------------------------------------------
    # 1. Connect to database
    # --------------------------------------------------------

    engine = (
        create_db_engine()
    )


    # --------------------------------------------------------
    # 2. Create logging table
    # --------------------------------------------------------

    create_pipeline_log_table(
        engine
    )


    # --------------------------------------------------------
    # 3. Read daily extract
    # --------------------------------------------------------

    raw_df = read_csv()


    # --------------------------------------------------------
    # 4. Prepare customer records
    # --------------------------------------------------------

    incoming_df = (
        prepare_customer_data(
            raw_df
        )
    )


    # --------------------------------------------------------
    # 5. Get current database state
    # --------------------------------------------------------

    existing_df = (
        get_existing_customers(
            engine,
            incoming_df[
                "customer_id"
            ]
            .tolist()
        )
    )


    print(
        f"\nExisting customers in DB: "
        f"{len(existing_df)}"
    )


    # --------------------------------------------------------
    # 6. Classify
    # --------------------------------------------------------

    classified_df = (
        classify_changes(
            incoming_df,
            existing_df
        )
    )


    # --------------------------------------------------------
    # 7. Counts
    # --------------------------------------------------------

    rows_inserted = (
        classified_df[
            classified_df["_change_type"]
            == "INSERT"
        ]
        .shape[0]
    )


    rows_updated = (
        classified_df[
            classified_df["_change_type"]
            == "UPDATE"
        ]
        .shape[0]
    )


    rows_unchanged = (
        classified_df[
            classified_df["_change_type"]
            == "UNCHANGED"
        ]
        .shape[0]
    )


    # --------------------------------------------------------
    # 8. Upsert
    # --------------------------------------------------------

    upsert_customers(
        engine,
        classified_df
    )


    # --------------------------------------------------------
    # 9. Log run
    # --------------------------------------------------------

    log_pipeline_run(
        engine,
        CSV_PATH.name,
        rows_inserted,
        rows_updated,
        rows_unchanged
    )


    # --------------------------------------------------------
    # 10. Final count
    # --------------------------------------------------------

    final_count = (
        get_customer_count(
            engine
        )
    )


    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("\n========================================")
    print("DE6 RUN SUMMARY")
    print("========================================")

    print(
        f"File: {CSV_PATH.name}"
    )

    print(
        f"Rows inserted: "
        f"{rows_inserted}"
    )

    print(
        f"Rows updated: "
        f"{rows_updated}"
    )

    print(
        f"Rows unchanged: "
        f"{rows_unchanged}"
    )

    print(
        f"Customers in database: "
        f"{final_count}"
    )


    print("\nDE6 completed successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    process_incremental()