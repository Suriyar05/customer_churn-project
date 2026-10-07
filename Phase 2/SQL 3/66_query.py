from database import get_db
from schema import Customer
 
 
db = get_db()
 
try:
    customer = db.get(Customer, "7590-VHVEG")
 
    if customer:
        print("Customer Details")
        print("-" * 30)
        print(f"Customer ID     : {customer.customer_id}")
        print(f"Gender          : {customer.gender}")
        print(f"Senior Citizen  : {customer.senior_citizen}")
    else:
        print("Customer not found.")
 
finally:
    db.close()
 
 
 
class Contract(Base):
    __tablename__ = "dim_contract"
 
    contract_key = Column(Integer, primary_key=True)
    contract_type = Column(String(30))
 