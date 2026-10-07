from sqlalchemy import text


def get_churn_summary(db):

    # =====================================================
    # 100. Total customers, churned customers, churn rate
    # =====================================================

    summary_query = text("""
        SELECT
            COUNT(*) AS total_customers,
            SUM(churn) AS total_churned,
            AVG(churn) AS churn_rate
        FROM fact_customer_account
    """)

    summary = db.execute(summary_query).mappings().first()


    # =====================================================
    # 101. Churn rate by contract type
    # =====================================================

    contract_query = text("""
        SELECT
            dc.contract_name AS contract_type,
            AVG(f.churn) AS churn_rate
        FROM fact_customer_account f
        JOIN dim_contract dc
            ON f.contract_id = dc.contract_id
        GROUP BY dc.contract_name
        ORDER BY churn_rate DESC
    """)

    contract_results = (
        db.execute(contract_query)
        .mappings()
        .all()
    )


    # =====================================================
    # 102. Churn rate by internet service
    # =====================================================

    internet_query = text("""
        SELECT
            internet_service,
            AVG(churn) AS churn_rate
        FROM fact_customer_account
        GROUP BY internet_service
        ORDER BY churn_rate DESC
    """)

    internet_results = (
        db.execute(internet_query)
        .mappings()
        .all()
    )


    # =====================================================
    # 103. Return structured response
    # =====================================================

    return {
        "total_customers": summary["total_customers"],
        "churned": summary["total_churned"],
        "churn_rate": round(
            float(summary["churn_rate"]), 4
        ),

        "by_contract": [
            {
                "contract_type": row["contract_type"],
                "churn_rate": round(
                    float(row["churn_rate"]), 4
                )
            }
            for row in contract_results
        ],

        "by_internet_service": [
            {
                "internet_service": row["internet_service"],
                "churn_rate": round(
                    float(row["churn_rate"]), 4
                )
            }
            for row in internet_results
        ]
    }