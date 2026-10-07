from pydantic import BaseModel


class CustomerResponse(BaseModel):
    customer_id: str
    tenure: int
    contract_type: str
    internet_service: str
    monthly_charges: float
    churn: int

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