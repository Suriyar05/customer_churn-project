from fastapi import FastAPI
from pydantic import BaseModel

from predict import predict_churn


app = FastAPI(
    title="Customer Churn API",
    description="Customer churn prediction API",
    version="1.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChurnRequest(BaseModel):

    tenure: int

    monthly_charges: float

    contract_type: str

    service_count: int


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Customer Churn API is running"
    }


# ============================================================
# PREDICT CHURN
# ============================================================

@app.post("/predict-churn")
def predict(request: ChurnRequest):

    result = predict_churn(

        tenure=request.tenure,

        monthly_charges=request.monthly_charges,

        contract_type=request.contract_type,

        service_count=request.service_count

    )


    return result