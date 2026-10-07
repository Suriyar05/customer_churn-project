from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# =========================================================
# Customer Model
# =========================================================

class Customer(Base):
    __tablename__ = "standardising_customers"

    customer_id = Column(String(20), primary_key=True)
    gender = Column(String(10))
    senior_citizen = Column(Integer)
    partner = Column(Integer)
    dependents = Column(Integer)

    account = relationship(
        "CustomerAccount",
        back_populates="customer",
        uselist=False
    )


# =========================================================
# Contract Model
# =========================================================

class Contract(Base):
    __tablename__ = "dim_contract"

    contract_id = Column(Integer, primary_key=True)
    contract_name = Column(String(30), nullable=False)

    accounts = relationship(
        "CustomerAccount",
        back_populates="contract"
    )


# =========================================================
# Customer Account Model
# =========================================================

class CustomerAccount(Base):
    __tablename__ = "fact_customer_account"

    customer_id = Column(
        String(20),
        primary_key=True
    )

    tenure = Column(Integer)

    internet_service = Column(String(30))
    monthly_charges = Column(Integer)

    contract_id = Column(
        Integer,
        nullable=False
    )

    churn = Column(Integer)

    customer = relationship(
        "Customer",
        back_populates="account"
    )

    contract = relationship(
        "Contract",
        back_populates="accounts"
    )