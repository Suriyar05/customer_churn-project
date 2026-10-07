import os
import logging

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Header,
    Depends,
    Request
)

from fastapi.responses import JSONResponse

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from schemas import (
    CustomerResponse,
    ChurnSummaryResponse,
    HighRiskCustomerResponse,
    CustomerFeaturesResponse,
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    ErrorResponse
)

from services import get_churn_summary

from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Telecom Customer Intelligence API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)

# =========================================================
# Environment Variables
# =========================================================

load_dotenv()

API_KEY = os.getenv("API_KEY")


# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(
    title="Telecom Customer Intelligence API",
    description="Production-ready telecom customer intelligence API",
    version="1.0.0"
)


# =========================================================
# Database Connection
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
# 131. API KEY AUTHENTICATION
# =========================================================

def verify_api_key(
    x_api_key: str | None = Header(default=None)
):

    if not API_KEY:
        logger.error("API_KEY environment variable is not configured")

        raise HTTPException(
            status_code=500,
            detail="API authentication is not configured"
        )

    if x_api_key != API_KEY:

        logger.warning("Unauthorized request")

        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

    return True


# =========================================================
# 129. Global 404 Handler
# =========================================================

@app.exception_handler(404)
async def not_found_handler(
    request: Request,
    exc: HTTPException
):

    logger.warning(
        f"404 | {request.method} {request.url.path}"
    )

    return JSONResponse(
        status_code=404,
        content={
            "detail": "Resource not found",
            "status_code": 404
        }
    )


# =========================================================
# Global HTTPException Handler
# =========================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException
):

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": str(exc.detail),
            "status_code": exc.status_code
        }
    )


# =========================================================
# Home
# =========================================================

@app.get("/")
def home(
    _: bool = Depends(verify_api_key)
):

    logger.info("GET /")

    return {
        "message": "Telecom Customer Intelligence API is running"
    }


# =========================================================
# API1 — Customer Profile
# =========================================================

@app.get(
    "/customer/{customer_id}",
    response_model=CustomerResponse,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
def get_customer(
    customer_id: str,
    _: bool = Depends(verify_api_key)
):

    logger.info(
        f"GET /customer/{customer_id}"
    )

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
            {"customer_id": customer_id}
        ).mappings().first()

        if result is None:

            logger.warning(
                f"Customer not found: {customer_id}"
            )

            raise HTTPException(
                status_code=404,
                detail=f"Customer '{customer_id}' not found"
            )

        logger.info(
            f"Response status: 200 | customer={customer_id}"
        )

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

    except HTTPException:
        raise

    except SQLAlchemyError as e:

        logger.error(
            f"Database error: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Database unavailable"
        )

    finally:
        session.close()


# =========================================================
# API2 — Churn Summary
# =========================================================

@app.get(
    "/churn/summary",
    response_model=ChurnSummaryResponse,
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
def churn_summary(
    _: bool = Depends(verify_api_key)
):

    logger.info("GET /churn/summary")

    session = SessionLocal()

    try:

        result = get_churn_summary(session)

        logger.info(
            "Response status: 200 | /churn/summary"
        )

        return result

    except SQLAlchemyError as e:

        logger.error(
            f"Database error: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Database unavailable"
        )

    finally:
        session.close()


# =========================================================
# API3 — High-Risk Customers
# =========================================================

@app.get(
    "/customers/high-risk",
    response_model=list[HighRiskCustomerResponse],
    responses={
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
def get_high_risk_customers(
    limit: int = Query(
        50,
        ge=1,
        le=500
    ),

    min_tenure: int | None = Query(
        None,
        ge=0
    ),

    max_tenure: int | None = Query(
        None,
        ge=0
    ),

    _: bool = Depends(verify_api_key)
):

    logger.info(
        f"GET /customers/high-risk "
        f"limit={limit}, "
        f"min_tenure={min_tenure}, "
        f"max_tenure={max_tenure}"
    )

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

        if min_tenure is not None:

            query += """
                AND tenure >= :min_tenure
            """

            params["min_tenure"] = min_tenure

        if max_tenure is not None:

            query += """
                AND tenure <= :max_tenure
            """

            params["max_tenure"] = max_tenure

        query += """
            ORDER BY monthly_charges DESC
            LIMIT :limit
        """

        result = session.execute(
            text(query),
            params
        ).mappings().all()

        logger.info(
            f"Response status: 200 | "
            f"high-risk count={len(result)}"
        )

        return [
            HighRiskCustomerResponse(
                customer_id=row["customer_id"],
                tenure=row["tenure"],
                monthly_charges=float(
                    row["monthly_charges"]
                ),
                contract_type=row["contract_type"],
                risk_reason=row["risk_reason"]
            )

            for row in result
        ]

    except SQLAlchemyError as e:

        logger.error(
            f"Database error: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Database unavailable"
        )

    finally:
        session.close()


# =========================================================
# API4 — Customer Features
# =========================================================

@app.get(
    "/customer/{customer_id}/features",
    response_model=CustomerFeaturesResponse,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
def get_customer_features(
    customer_id: str,
    _: bool = Depends(verify_api_key)
):

    logger.info(
        f"GET /customer/{customer_id}/features"
    )

    session = SessionLocal()

    try:

        query = text("""
            SELECT
                service_count,
                tenure_bucket,
                high_charge_flag,
                is_long_term_customer,
                has_streaming_bundle,
                auto_pay_flag,
                monthly_charges,
                total_charges

            FROM customer_ml_features

            WHERE customer_id = :customer_id
        """)

        result = session.execute(
            query,
            {"customer_id": customer_id}
        ).mappings().first()

        if result is None:

            raise HTTPException(
                status_code=404,
                detail=f"Features for customer '{customer_id}' not found"
            )

        logger.info(
            f"Response status: 200 | "
            f"features={customer_id}"
        )

        return CustomerFeaturesResponse(
            service_count=result["service_count"],
            tenure_bucket=result["tenure_bucket"],
            high_charge_flag=result["high_charge_flag"],
            is_long_term_customer=result[
                "is_long_term_customer"
            ],
            has_streaming_bundle=result[
                "has_streaming_bundle"
            ],
            auto_pay_flag=result["auto_pay_flag"],
            monthly_charges=float(
                result["monthly_charges"]
            ),
            total_charges=float(
                result["total_charges"]
            )
        )

    except HTTPException:
        raise

    except SQLAlchemyError as e:

        logger.error(
            f"Database error: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Database unavailable"
        )

    finally:
        session.close()


# =========================================================
# API5 — Prediction Stub
# =========================================================

@app.post(
    "/predict-churn",
    response_model=ChurnPredictionResponse,
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
def predict_churn(
    request: ChurnPredictionRequest,
    _: bool = Depends(verify_api_key)
):

    logger.info(
        f"POST /predict-churn | "
        f"tenure={request.tenure}, "
        f"monthly_charges={request.monthly_charges}, "
        f"contract={request.contract_type}, "
        f"service_count={request.service_count}"
    )

    logger.info(
        "Response status: 200 | /predict-churn"
    )

    return ChurnPredictionResponse(
        customer_id="N/A",
        risk_score=0.78,
        prediction="Likely to churn",
        note="stub — real model in ML4"
    )