import os
import csv
import shutil
import pandas as pd

from sqlalchemy import create_engine, text


# =========================================================
# CONFIGURATION
# =========================================================

LANDING_DIR = "data/landing"
RAW_DIR = "data/raw"
REJECTED_DIR = "data/rejected"


DATABASE_URL = (
    "mysql+pymysql://root:root@localhost:3306/telecom_intelligence"
)

engine = create_engine(DATABASE_URL)


# =========================================================
# EXPECTED CSV COLUMNS
# =========================================================

EXPECTED_COLS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]


# =========================================================
# 1. CREATE REQUIRED DIRECTORIES
# =========================================================

def create_directories():

    os.makedirs(LANDING_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)

    print("Directories ready.")


# =========================================================
# 2. DETECT CSV FILES
# =========================================================

def detect_files():

    files = []

    for filename in os.listdir(LANDING_DIR):

        if filename.lower().endswith(".csv"):

            filepath = os.path.join(
                LANDING_DIR,
                filename
            )

            files.append(filepath)

    return files


# =========================================================
# 3. VALIDATE CSV SCHEMA
# =========================================================

def validate_schema(filepath):

    with open(
        filepath,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        header = next(
            csv.reader(file)
        )

    is_valid = (
        header == EXPECTED_COLS
    )

    return is_valid, header


# =========================================================
# 4. LOAD DATA INTO STAGING TABLE
# =========================================================

def load_to_staging(filepath):

    df = pd.read_csv(filepath)

    df.to_sql(
        "stg_customer_raw",
        engine,
        if_exists="replace",
        index=False
    )

    return len(df)


# =========================================================
# 5. CREATE INGESTION LOG TABLE
# =========================================================

def create_ingestion_log_table():

    query = text("""
        CREATE TABLE IF NOT EXISTS ingestion_log (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255),
            status VARCHAR(20),
            row_count INT,
            reason VARCHAR(500),
            loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    with engine.begin() as connection:

        connection.execute(query)


# =========================================================
# 6. LOG INGESTION RESULT
# =========================================================

def log_ingestion(
    filename,
    status,
    rows,
    reason
):

    query = text("""
        INSERT INTO ingestion_log
        (
            filename,
            status,
            row_count,
            reason
        )
        VALUES
        (
            :filename,
            :status,
            :rows,
            :reason
        )
    """)

    with engine.begin() as connection:

        connection.execute(
            query,
            {
                "filename": filename,
                "status": status,
                "rows": rows,
                "reason": reason
            }
        )


# =========================================================
# 7. PROCESS LANDING FILES
# =========================================================

def process_landing():

    files = detect_files()

    if not files:

        print("No CSV files found in landing folder.")
        return

    print(
        f"\nFound {len(files)} CSV file(s).\n"
    )

    for filepath in files:

        filename = os.path.basename(filepath)

        print(
            f"Processing: {filename}"
        )

        # -------------------------------------------------
        # Schema Validation
        # -------------------------------------------------

        try:

            valid, header = validate_schema(
                filepath
            )

        except Exception as e:

            print(
                f"ERROR reading {filename}: {e}"
            )

            log_ingestion(
                filename,
                "REJECTED",
                0,
                str(e)
            )

            continue


        # -------------------------------------------------
        # Invalid Schema
        # -------------------------------------------------

        if not valid:

            reason = (
                "Column mismatch. "
                f"Expected {len(EXPECTED_COLS)} columns, "
                f"found {len(header)}."
            )

            print(
                f"REJECTED: {filename}"
            )

            print(
                f"Reason: {reason}"
            )

            log_ingestion(
                filename,
                "REJECTED",
                0,
                reason
            )

            # Move invalid file
            rejected_path = os.path.join(
                REJECTED_DIR,
                filename
            )

            shutil.move(
                filepath,
                rejected_path
            )

            continue


        # -------------------------------------------------
        # Valid File → Load to Staging
        # -------------------------------------------------

        try:

            rows = load_to_staging(
                filepath
            )

            print(
                f"LOADED: {filename}"
            )

            print(
                f"Rows loaded: {rows}"
            )

            log_ingestion(
                filename,
                "LOADED",
                rows,
                "Schema validation successful"
            )

            # Move successfully processed file
            raw_path = os.path.join(
                RAW_DIR,
                filename
            )

            shutil.move(
                filepath,
                raw_path
            )

        except Exception as e:

            print(
                f"FAILED: {filename}"
            )

            print(
                f"Reason: {e}"
            )

            log_ingestion(
                filename,
                "REJECTED",
                0,
                str(e)
            )

            rejected_path = os.path.join(
                REJECTED_DIR,
                filename
            )

            shutil.move(
                filepath,
                rejected_path
            )


# =========================================================
# 8. VERIFY STAGING ROW COUNT
# =========================================================

def verify_staging():

    query = text("""
        SELECT COUNT(*) AS total_rows
        FROM stg_customer_raw
    """)

    with engine.connect() as connection:

        result = connection.execute(
            query
        ).scalar()

    print(
        f"\nStaging table row count: {result}"
    )


# =========================================================
# 9. DISPLAY INGESTION LOG
# =========================================================

def show_ingestion_log():

    query = text("""
        SELECT
            id,
            filename,
            status,
            row_count,
            reason,
            loaded_at
        FROM ingestion_log
        ORDER BY id DESC
    """)

    with engine.connect() as connection:

        results = connection.execute(
            query
        )

        print("\n========== INGESTION LOG ==========")

        for row in results:

            print(
                f"ID: {row.id}"
            )

            print(
                f"File: {row.filename}"
            )

            print(
                f"Status: {row.status}"
            )

            print(
                f"Rows: {row.row_count}"
            )

            print(
                f"Reason: {row.reason}"
            )

            print(
                f"Loaded At: {row.loaded_at}"
            )

            print(
                "----------------------------------"
            )


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    print(
        "======================================"
    )

    print(
        "   CUSTOMER DATA INGESTION PIPELINE"
    )

    print(
        "======================================"
    )


    # Create folders
    create_directories()


    # Create ingestion log table
    create_ingestion_log_table()


    # Process landing files
    process_landing()


    # Verify staging table
    try:

        verify_staging()

    except Exception as e:

        print(
            f"\nStaging verification failed: {e}"
        )


    # Display ingestion log
    try:

        show_ingestion_log()

    except Exception as e:

        print(
            f"\nCould not display ingestion log: {e}"
        )


    print(
        "\n======================================"
    )

    print(
        "       INGESTION COMPLETED"
    )

    print(
        "======================================"
    )