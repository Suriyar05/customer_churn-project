
from database import Base
from sqlalchemy import Column
from sqlalchemy.types import String, Integer
 
class Customer(Base):
    __tablename__= "customers"
 
    customer_id = Column(String(20), primary_key=True)
    gender = Column(String(20))
    senior_citizen = Column(Integer)
    partner = Column(Integer)
    dependents = Column(Integer)
 
