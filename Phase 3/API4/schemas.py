from pydantic import BaseModel


# =====================================================
# API1
# =====================================================

class CustomerResponse(BaseModel):
    customer_id: str
    tenure: int
    contract_type: str
    internet_service: str
    monthly_charges: float
    churn: int


# =====================================================
# API2
# =====================================================

class ContractChurn(BaseModel):
    contract_type: str
    churn_rate: float


class InternetServiceChurn(BaseModel):
    internet_service: str
    churn_rate: float


class ChurnSummaryResponse(BaseModel):
    total_customers: int
    churned: int
    churn_rate: float
    by_contract: list[ContractChurn]
    by_internet_service: list[InternetServiceChurn]


# =====================================================
# API3 — High-Risk Customer
# =====================================================

class HighRiskCustomerResponse(BaseModel):
    customer_id: str
    tenure: int
    monthly_charges: float
    contract_type: str
    risk_reason: str

# =====================================================
# API4 — Customer ML Features  117 
# =====================================================

class CustomerFeaturesResponse(BaseModel):
    service_count: int
    tenure_bucket: str
    high_charge_flag: int
    is_long_term_customer: int
    has_streaming_bundle: int
    auto_pay_flag: int
    monthly_charges: float
    total_charges: float