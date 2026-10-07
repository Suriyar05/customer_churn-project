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