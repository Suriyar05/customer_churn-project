from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "spark_output"
    / "contract_summary"
)


# ============================================================
# FIND CUSTOMER CSV AUTOMATICALLY
# ============================================================

def find_customer_csv():

    print("\n========================================")
    print("LOCATING CUSTOMER CSV")
    print("========================================")

    matches = list(
        PROJECT_ROOT.rglob("customer_churn.csv")
    )

    if not matches:

        raise FileNotFoundError(
            "customer_churn.csv was not found anywhere "
            f"inside:\n{PROJECT_ROOT}"
        )

    csv_path = matches[0]

    print(
        f"PASS — customer_churn.csv found:\n"
        f"{csv_path}"
    )

    return csv_path


# ============================================================
# CREATE SPARK SESSION
# ============================================================

def create_spark():

    print("\n========================================")
    print("CREATE SPARK SESSION")
    print("========================================")

    spark = (
        SparkSession.builder
        .appName("DE5 Customer Processing")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("PASS — SparkSession created")

    return spark


# ============================================================
# MANUAL CSV SCHEMA
# ============================================================

def get_customer_schema():

    return StructType([

        StructField(
            "customerID",
            StringType(),
            True
        ),

        StructField(
            "gender",
            StringType(),
            True
        ),

        StructField(
            "SeniorCitizen",
            IntegerType(),
            True
        ),

        StructField(
            "Partner",
            StringType(),
            True
        ),

        StructField(
            "Dependents",
            StringType(),
            True
        ),

        StructField(
            "tenure",
            IntegerType(),
            True
        ),

        StructField(
            "PhoneService",
            StringType(),
            True
        ),

        StructField(
            "MultipleLines",
            StringType(),
            True
        ),

        StructField(
            "InternetService",
            StringType(),
            True
        ),

        StructField(
            "OnlineSecurity",
            StringType(),
            True
        ),

        StructField(
            "OnlineBackup",
            StringType(),
            True
        ),

        StructField(
            "DeviceProtection",
            StringType(),
            True
        ),

        StructField(
            "TechSupport",
            StringType(),
            True
        ),

        StructField(
            "StreamingTV",
            StringType(),
            True
        ),

        StructField(
            "StreamingMovies",
            StringType(),
            True
        ),

        StructField(
            "Contract",
            StringType(),
            True
        ),

        StructField(
            "PaperlessBilling",
            StringType(),
            True
        ),

        StructField(
            "PaymentMethod",
            StringType(),
            True
        ),

        StructField(
            "MonthlyCharges",
            DoubleType(),
            True
        ),

        # IMPORTANT:
        # Keep TotalCharges as String initially.
        # Some rows contain blanks.
        StructField(
            "TotalCharges",
            StringType(),
            True
        ),

        StructField(
            "Churn",
            StringType(),
            True
        )
    ])


# ============================================================
# READ CUSTOMER CSV
# ============================================================

def read_customer_csv(spark, csv_path):

    print("\n========================================")
    print("READ CUSTOMER CSV")
    print("========================================")

    schema = get_customer_schema()

    df = (
        spark.read
        .option("header", True)
        .option("mode", "PERMISSIVE")
        .schema(schema)
        .csv(str(csv_path))
    )

    print("\nSpark DataFrame schema:")
    df.printSchema()

    row_count = df.count()

    print(
        f"\nRow count: {row_count}"
    )

    if row_count != 7043:

        raise AssertionError(
            f"Expected 7043 rows, got {row_count}"
        )

    print(
        "PASS — 7043 customer rows loaded"
    )

    return df


# ============================================================
# CLEAN TOTAL CHARGES
# ============================================================

def clean_total_charges(df):

    print("\n========================================")
    print("CLEAN TOTAL CHARGES")
    print("========================================")

    df = df.withColumn(
        "TotalCharges",
        F.regexp_replace(
            F.trim(F.col("TotalCharges")),
            r"^\s*$",
            None
        ).cast(DoubleType())
    )

    print(
        "PASS — TotalCharges converted "
        "to DoubleType"
    )

    return df


# ============================================================
# ENCODE CHURN
# ============================================================

def encode_churn(df):

    print("\n========================================")
    print("ENCODE CHURN")
    print("========================================")

    df = df.withColumn(
        "churn_encoded",
        F.when(
            F.col("Churn") == "Yes",
            1
        ).otherwise(0)
    )

    print(
        "PASS — churn_encoded created"
    )

    return df


# ============================================================
# CONTRACT CHURN SUMMARY
# ============================================================

def build_contract_summary(df):

    print("\n========================================")
    print("CONTRACT CHURN SUMMARY")
    print("========================================")

    summary = (
        df.groupBy("Contract")
        .agg(
            F.count("*").alias(
                "customer_count"
            ),

            F.avg(
                "churn_encoded"
            ).alias(
                "churn_rate"
            )
        )
        .orderBy("Contract")
    )

    summary.show(
        truncate=False
    )

    return summary


# ============================================================
# INTERNET SERVICE SUMMARY
# ============================================================

def build_internet_summary(df):

    print("\n========================================")
    print("INTERNET SERVICE SUMMARY")
    print("========================================")

    summary = (
        df.groupBy("InternetService")
        .agg(

            F.avg(
                "MonthlyCharges"
            ).alias(
                "avg_monthly_charges"
            ),

            F.avg(
                "churn_encoded"
            ).alias(
                "churn_rate"
            )
        )
        .orderBy("InternetService")
    )

    summary.show(
        truncate=False
    )

    return summary


# ============================================================
# CROSS VALIDATION
# ============================================================

def cross_validate_contract(summary):

    print("\n========================================")
    print("CROSS VALIDATION")
    print("========================================")

    print(
        """
Spark contract churn rates:

Month-to-month ≈ 0.427097
One year       ≈ 0.112695
Two year       ≈ 0.028319

Compare these values with your CP3
pandas groupby results.
"""
    )

    rows = (
        summary
        .select(
            "Contract",
            "customer_count",
            "churn_rate"
        )
        .collect()
    )

    for row in rows:

        print(
            f"{row['Contract']}: "
            f"customers={row['customer_count']}, "
            f"churn_rate={row['churn_rate']:.6f}"
        )


# ============================================================
# WRITE PARQUET USING PANDAS + PYARROW
# ============================================================

def write_contract_parquet_pandas(
    contract_summary
):

    print("\n========================================")
    print("WRITE PARQUET")
    print("========================================")

    print(
        "Using pandas + PyArrow for Parquet output."
    )

    print(
        "Hadoop is NOT required for this step."
    )

    # --------------------------------------------------------
    # Convert Spark DataFrame -> pandas
    # --------------------------------------------------------

    pandas_df = contract_summary.toPandas()

    print(
        f"Converted Spark summary to pandas: "
        f"{len(pandas_df)} rows"
    )

    # --------------------------------------------------------
    # Remove old output directory
    # --------------------------------------------------------

    if OUTPUT_PATH.exists():

        import shutil

        shutil.rmtree(
            OUTPUT_PATH
        )

    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Write partitioned Parquet
    #
    # Same requirement as:
    #
    # partitionBy("Contract")
    # --------------------------------------------------------

    table = pa.Table.from_pandas(
        pandas_df,
        preserve_index=False
    )

    pq.write_to_dataset(
        table,
        root_path=str(OUTPUT_PATH),
        partition_cols=["Contract"]
    )

    print(
        f"PASS — Parquet written to:\n"
        f"{OUTPUT_PATH}"
    )

    # --------------------------------------------------------
    # Show generated partition folders
    # --------------------------------------------------------

    print("\nGenerated Parquet partitions:")

    for path in OUTPUT_PATH.rglob("*"):

        if path.is_file():

            print(
                f"  {path.relative_to(OUTPUT_PATH)}"
            )


# ============================================================
# READ PARQUET BACK USING PYARROW
# ============================================================

def read_parquet_back():

    print("\n========================================")
    print("PARQUET ROUND-TRIP VALIDATION")
    print("========================================")

    if not OUTPUT_PATH.exists():

        raise FileNotFoundError(
            "Parquet output directory does not exist"
        )

    # --------------------------------------------------------
    # Read dataset using PyArrow
    # --------------------------------------------------------

    dataset = pq.ParquetDataset(
        str(OUTPUT_PATH)
    )

    table = dataset.read()

    pandas_df = table.to_pandas()

    # --------------------------------------------------------
    # Print schema
    # --------------------------------------------------------

    print("\nParquet schema:")

    print(
        table.schema
    )

    # --------------------------------------------------------
    # Count
    # --------------------------------------------------------

    count = len(
        pandas_df
    )

    print(
        f"\nRound-trip row count: {count}"
    )

    # Contract summary should contain
    # 3 contract rows.

    if count == 3:

        print(
            "PASS — Parquet round-trip "
            "row count = 3"
        )

    else:

        raise AssertionError(
            f"Expected 3 rows after round-trip, "
            f"got {count}"
        )

    print("\nRound-trip data:")

    print(
        pandas_df.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    spark = None

    try:

        print("\n========================================")
        print("DE5 — PYSPARK CUSTOMER PIPELINE")
        print("========================================")

        # ----------------------------------------------------
        # 1. Find CSV
        # ----------------------------------------------------

        csv_path = find_customer_csv()

        # ----------------------------------------------------
        # 2. Create Spark
        # ----------------------------------------------------

        spark = create_spark()

        # ----------------------------------------------------
        # 3. Read CSV with manual schema
        # ----------------------------------------------------

        df = read_customer_csv(
            spark,
            csv_path
        )

        # ----------------------------------------------------
        # 4. Clean TotalCharges
        # ----------------------------------------------------

        df = clean_total_charges(
            df
        )

        # ----------------------------------------------------
        # 5. Encode Churn
        # ----------------------------------------------------

        df = encode_churn(
            df
        )

        # ----------------------------------------------------
        # 6. Contract summary
        # ----------------------------------------------------

        contract_summary = (
            build_contract_summary(
                df
            )
        )

        # ----------------------------------------------------
        # 7. Internet summary
        # ----------------------------------------------------

        build_internet_summary(
            df
        )

        # ----------------------------------------------------
        # 8. Cross validation
        # ----------------------------------------------------

        cross_validate_contract(
            contract_summary
        )

        # ----------------------------------------------------
        # 9. Write Parquet
        #
        # PyArrow instead of Spark/Hadoop writer.
        # ----------------------------------------------------

        write_contract_parquet_pandas(
            contract_summary
        )

        # ----------------------------------------------------
        # 10. Read Parquet back
        # ----------------------------------------------------

        read_parquet_back()

        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        print("\n========================================")
        print("DE5 COMPLETE")
        print("========================================")

        print(
            "\nPySpark processing completed successfully."
        )

        print(
            "Parquet was written using "
            "pandas + PyArrow."
        )

       
    except Exception as e:

        print("\n========================================")
        print("DE5 FAILED")
        print("========================================")

        print(
            f"Reason: {e}"
        )

        raise

    finally:

        if spark is not None:

            spark.stop()

            print(
                "\nSparkSession stopped."
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()