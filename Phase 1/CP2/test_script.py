import pandas as pd
from customer_cleaner import CustomerCleaner
 
df = pd.read_csv("../../Dataset/customer_churn.csv")
 
print("Before cleaning:")
print(df.shape)
print(df.dtypes)
 
cleaned_df = CustomerCleaner(df).clean()
 
print("\nAfter cleaning:")
print(cleaned_df.shape)
print(cleaned_df.dtypes)