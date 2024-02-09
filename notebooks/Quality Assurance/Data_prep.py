import pandas as pd 
df = pd.read_csv("C:\\Users\\kiran\\OneDrive\\Desktop\\Card Fraud Detection\\application_data.csv")
# null_values = df.isnull().sum()
null_columns = df.columns[df.isnull().any()]
null_dtypes = df[null_columns].dtypes
null_count = df[null_columns].isnull().sum()
# null_values.to_csv("C:\\Users\\kiran\\OneDrive\\Desktop\\Card Fraud Detection\\nulls.csv")
null_info = pd.DataFrame({
    'Datatype':null_dtypes,
    'nUllcount':null_count
})
# print(null_values)
# print(null_dtypes)
null_info.to_csv("C:\\Users\\kiran\\OneDrive\\Desktop\\Card Fraud Detection\\nulls.csv")
print(null_info)