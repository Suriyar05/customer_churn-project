from CP2.customer_cleaner import CustomerCleaner
import pandas as pd
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="customer_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

data_frame_path = "../../dataset/telco_customer.csv"

def load_data():
    try:
        df = pd.read_csv(data_frame_path)
        logger.info("DataFrame Loaded")
        logger.info(f"DataFrame Rows {df.shape[0]}")
        return df
    except Exception as e:
        logger.error(f"Pipeline failed : {e}")
        raise

def build_features(original_df):
    try:
        df= original_df.copy()

        # Feature 1 - tenure_bucket

        bins = [0, 12, 24, 48, 72]
        labels = ["0-12", "13-24", "25-48", "49-72"]
        df["tenure_bucket"] = pd.cut(
            df["tenure"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        # Feature 2 - high_change_flag

        median = df["monthly_charges"].mean()
        def fill_high_charge(charge):
            return 1 if charge > median else 0

        df["high_charge_flag"] = df["monthly_charges"].apply(fill_high_charge)

        # Feature 3 - serivce_count

        def fill_service_count(row):
            service_columns = [
                "online_security",
                "online_backup",
                "device_protection",
                "tech_support",
                "streaming_tv",
                "streaming_movies"
            ]
            count = 0
            for cols in service_columns:
                count += 1 if row[cols] == "Yes" else 0
            return count

        df["service_count"] = df.apply(fill_service_count, axis=1)

        # Feature 4 - is_long_term_customer

        def fill_is_long_term_customer(tenure):
            return 1 if tenure >= 24 else 0

        df["is_long_term_customer"] = df["tenure"].apply(fill_is_long_term_customer)

        # Feature 5 - has_streaming_bundle

        def fill_has_streaming_bundle(row):
            return 1 if row["streaming_tv"] == "Yes" and row["streaming_movies"] == "Yes" else 0

        df["has_streaming_bundle"] = df[["streaming_tv", "streaming_movies"]].apply(fill_has_streaming_bundle, axis = 1)

        # Feature 6 - auto_pay_flag

        def fill_auto_pay_flag(value):
            return 1 if "automatic" in value else 0

        df["auto_pay_flag"] = df["payment_method"].apply(fill_auto_pay_flag)

        return df
    except Exception as e:
        logger.error(f"Pipeline failed : {e}")
        raise

def validate_data(cleaned_df):
    try:
        assert cleaned_df["monthly_charges"].notna().all(), \
            "monthly_charges contains null values"

        assert cleaned_df["churn"].isin([0, 1]).all(), \
            "churn contains values other than 0 or 1"

        logger.info("Data quality checks passed")

    except Exception as e:
        logger.error(f"Data quality check failed: {e}")
        raise

def save_outputs(clean_df, feature_df, output_dir):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Clean dataframe
        clean_df_filename = f"clean_df_{timestamp}.csv"
        clean_df_path = os.path.join(output_dir, clean_df_filename)
        clean_df.to_csv(clean_df_path, index=False)

        # Feature dataframe
        feature_df_filename = f"feature_df_{timestamp}.csv"
        feature_df_path = os.path.join(output_dir, feature_df_filename)
        feature_df.to_csv(feature_df_path, index=False)

        logger.info(f"Clean dataframe saved to: {clean_df_path}")
        logger.info(f"Feature dataframe saved to: {feature_df_path}")
    except Exception as e:
        logger.error(f"Pipeline failed : {e}")
        raise

def main():
    try:
        logger.info("Pipeline started")

        main_df = load_data()
        logger.info("Data loading completed")

        cleaner = CustomerCleaner(data_frame=main_df)
        cleaned_df = cleaner.clean()
        logger.info(f"Data cleaning completed. Rows: {cleaned_df.shape[0]}")

        feature_df = build_features(cleaned_df)
        logger.info(
            f"Feature engineering completed. Rows: {feature_df.shape[0]}"
        )

        validate_data(cleaned_df)

        output_dir = "../../dataset"
        save_outputs(cleaned_df, feature_df, output_dir)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()