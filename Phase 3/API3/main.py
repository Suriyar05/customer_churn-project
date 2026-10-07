from fastapi import FastAPI, HTTPException,Query
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from schemas import (CustomerResponse,ChurnSummaryResponse,HighRiskCustomerResponse)
from services import get_churn_summary



# =========================================================
# 90. Create FastAPI Application
# =========================================================

app = FastAPI(
    title="Telecom Customer Intelligence API",
    description="API for querying telecom customer intelligence",
    version="1.0.0"
)


# =========================================================
# 91. SQLAlchemy Database Connection
# =========================================================

DATABASE_URL = (
    "mysql+pymysql://root:root@localhost:3306/telecom_intelligence"
)

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)



# =========================================================
# 93-94. Customer Profile Endpoint
# =========================================================

@app.get(
    "/customer/{customer_id}",
    response_model=CustomerResponse
)
def get_customer(customer_id: str):

    session = SessionLocal()

    try:

        query = text("""
            SELECT
                c.customer_id,
                f.tenure,
                dc.contract_name AS contract_type,
                f.internet_service,
                f.monthly_charges,
                f.churn

            FROM standardising_customers c

            JOIN fact_customer_account f
                ON c.customer_id = f.customer_id

            JOIN dim_contract dc
                ON f.contract_id = dc.contract_id

            WHERE c.customer_id = :customer_id
        """)

        result = session.execute(
            query,
            {
                "customer_id": customer_id
            }
        ).mappings().first()

        # Customer not found
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Customer '{customer_id}' not found"
            )

        # Return Pydantic response
        return CustomerResponse(
            customer_id=result["customer_id"],
            tenure=result["tenure"],
            contract_type=result["contract_type"],
            internet_service=result["internet_service"],
            monthly_charges=float(
                result["monthly_charges"]
            ),
            churn=result["churn"]
        )

    finally:
        session.close()

# =========================================================
# API2 — Churn Summary
# =========================================================

@app.get(
    "/churn/summary",
    response_model=ChurnSummaryResponse
)
def churn_summary():

    session = SessionLocal()

    try:
        return get_churn_summary(session)

    finally:
        session.close()

# =========================================================
# API3 — High-Risk Customers
# =========================================================

@app.get(
    "/customers/high-risk",
    response_model=list[HighRiskCustomerResponse]
)
def get_high_risk_customers(
    limit: int = Query(50, ge=1, le=500),
    min_tenure: int | None = Query(None, ge=0),
    max_tenure: int | None = Query(None, ge=0)
):

    session = SessionLocal()

    try:

        query = """
            SELECT
                customer_id,
                tenure,
                monthly_charges,
                contract_name AS contract_type,
                risk_reason
            FROM v_high_risk_customers
            WHERE 1 = 1
        """

        params = {
            "limit": limit
        }

        # Minimum tenure filter
        if min_tenure is not None:
            query += """
                AND tenure >= :min_tenure
            """
            params["min_tenure"] = min_tenure

        # Maximum tenure filter
        if max_tenure is not None:
            query += """
                AND tenure <= :max_tenure
            """
            params["max_tenure"] = max_tenure

        # Ranking
        query += """
            ORDER BY monthly_charges DESC
            LIMIT :limit
        """

        result = session.execute(
            text(query),
            params
        ).mappings().all()

        return [
            HighRiskCustomerResponse(
                customer_id=row["customer_id"],
                tenure=row["tenure"],
                monthly_charges=float(row["monthly_charges"]),
                contract_type=row["contract_type"],
                risk_reason=row["risk_reason"]
            )
            for row in result
        ]

    finally:
        session.close()