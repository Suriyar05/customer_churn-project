import pandas as pd
import logging
 
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="customer_cleaner.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
 
class CustomerCleaner:
    def __init__(self, data_frame):
        self.DataFrame = data_frame.copy()
 
    def standardize_column_names(self):
        def clean_column_name(column_name):
            if column_name == "customerID":
                return "customer_id"
            if column_name == "StreamingTV":
                return "streaming_tv"
            new_column_name = column_name[0].lower()
 
            for i in range(1, len(column_name)):
                if column_name[i].isupper():
                    new_column_name += "_"
                new_column_name += column_name[i].lower()
 
            return new_column_name
 
        new_columns = []
 
        for cols in self.DataFrame.columns:
            new_columns.append(clean_column_name(cols))
 
        self.DataFrame.columns = new_columns
 
        logger.info("Column names standardized to snake_case")
 
        return self
 
    def fix_total_charges(self):
        try:
            before_nulls = self.DataFrame["total_charges"].isna().sum()
 
            self.DataFrame["total_charges"] = (
                self.DataFrame["total_charges"]
                .replace("", pd.NA)
            )
 
            self.DataFrame["total_charges"] = pd.to_numeric(
                self.DataFrame["total_charges"],
                errors="coerce"
            )
 
            after_nulls = self.DataFrame["total_charges"].isna().sum()
            affected_rows = after_nulls - before_nulls
 
            logger.info(
                "Total charges converted to float. Rows affected: %s",
                affected_rows
            )
 
        except Exception as error:
            logger.error(
                "Unexpected error while fixing total charges: %s",
                error
            )
 
        return self
 
    def normalize_binary_columns(self):
        def make_binary(value):
            return 1 if value == "Yes" else 0
 
        columns = [
            "partner",
            "dependents",
            "phone_service",
            "paperless_billing",
            "churn"
        ]
 
        for col in columns:
            if col in self.DataFrame.columns:
                self.DataFrame[col] = self.DataFrame[col].apply(make_binary)
 
        logger.info(
            "Binary columns normalized: %s",
            columns
        )
 
        return self
 
    def handle_nulls(self):
        null_count = self.DataFrame["total_charges"].isna().sum()
 
        self.DataFrame["total_charges"] = self.DataFrame[
            "total_charges"
        ].fillna(
            self.DataFrame["monthly_charges"]
        )
 
        logger.info(
            "Handled total charges nulls. Rows filled: %s",
            null_count
        )
 
        return self
 
    def clean(self):
        self.standardize_column_names()
        self.fix_total_charges()
        self.normalize_binary_columns()
        self.handle_nulls()
 
        logger.info(
            "Cleaning completed. Final shape: %s",
            self.DataFrame.shape
        )
 
        return self.DataFrame
 