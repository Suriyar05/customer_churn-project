# ============================================================
# ML4 — REAL CHURN PREDICTION
# ============================================================

import os
import joblib
import pandas as pd


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
# LOAD MODEL ONCE
# ============================================================

print("Loading churn model...")

model = joblib.load(
    MODEL_PATH
)

print(
    f"Model loaded successfully: {MODEL_PATH}"
)


# ============================================================
# TRAINING FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [

    "tenure",

    "monthly_charges",

    "total_charges",

    "service_count",

    "high_charge_flag",

    "is_long_term_customer",

    "auto_pay_flag",

    "has_streaming_bundle",

    "contract_type_Month-to-month",

    "contract_type_One year",

    "contract_type_Two year",

    "internet_service_DSL",

    "internet_service_Fiber optic",

    "internet_service_No"
]


# ============================================================
# PREPROCESS INPUT
# ============================================================

def preprocess_input(
    tenure,
    monthly_charges,
    contract_type,
    service_count
):

    # --------------------------------------------------------
    # Calculate engineered features
    # --------------------------------------------------------

    # NOTE:
    # For production inference, these thresholds should come
    # from the same feature pipeline used during training.
    #
    # These values are based on the current Telco dataset.

    high_charge_threshold = 70.35

    high_charge_flag = int(
        monthly_charges >
        high_charge_threshold
    )

    is_long_term_customer = int(
        tenure >= 24
    )

    # The API does not currently receive payment method,
    # streaming information, or internet service.
    #
    # Use the neutral/default values required by the model.

    auto_pay_flag = 0

    has_streaming_bundle = 0

    internet_service = "No"


    # --------------------------------------------------------
    # Total charges
    # --------------------------------------------------------

    # API does not receive total_charges.
    #
    # Estimate it using tenure and monthly charges.
    #
    # IMPORTANT:
    # This is an approximation. A production API should
    # ideally receive total_charges or retrieve it by customer_id.

    total_charges = (
        tenure *
        monthly_charges
    )


    # --------------------------------------------------------
    # Create one-row DataFrame
    # --------------------------------------------------------

    data = {

        "tenure": [
            tenure
        ],

        "monthly_charges": [
            monthly_charges
        ],

        "total_charges": [
            total_charges
        ],

        "service_count": [
            service_count
        ],

        "high_charge_flag": [
            high_charge_flag
        ],

        "is_long_term_customer": [
            is_long_term_customer
        ],

        "auto_pay_flag": [
            auto_pay_flag
        ],

        "has_streaming_bundle": [
            has_streaming_bundle
        ],

        "contract_type_Month-to-month": [
            0
        ],

        "contract_type_One year": [
            0
        ],

        "contract_type_Two year": [
            0
        ],

        "internet_service_DSL": [
            0
        ],

        "internet_service_Fiber optic": [
            0
        ],

        "internet_service_No": [
            0
        ]
    }


    # --------------------------------------------------------
    # Set contract one-hot value
    # --------------------------------------------------------

    contract_column = (
        f"contract_type_{contract_type}"
    )


    if contract_column not in data:

        raise ValueError(
            f"Invalid contract_type: "
            f"{contract_type}"
        )


    data[contract_column] = [1]


    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    X = pd.DataFrame(
        data
    )


    # --------------------------------------------------------
    # Ensure exact training column order
    # --------------------------------------------------------

    X = X[
        FEATURE_COLUMNS
    ]


    return X


# ============================================================
# PREDICT CHURN
# ============================================================

def predict_churn(
    tenure,
    monthly_charges,
    contract_type,
    service_count
):

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    X = preprocess_input(
        tenure=tenure,
        monthly_charges=monthly_charges,
        contract_type=contract_type,
        service_count=service_count
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        X
    )[0]


    # --------------------------------------------------------
    # Probability of churn
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X
    )[0]


    # Class 1 = churn

    risk_score = float(
        probabilities[1]
    )


    # --------------------------------------------------------
    # Prediction label
    # --------------------------------------------------------

    if prediction == 1:

        prediction_label = (
            "Likely to churn"
        )

    else:

        prediction_label = (
            "Unlikely to churn"
        )


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = float(
        max(probabilities)
    )


    return {

        "risk_score": risk_score,

        "prediction": prediction_label,

        "confidence": confidence

    }