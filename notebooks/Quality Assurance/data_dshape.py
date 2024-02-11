import pandas as pd

df = pd.read_csv("C:\\Users\\kiran\\OneDrive\\Desktop\\Card Fraud Detection\\application_data.csv")

shape_info = df.shape


column_details = df.info()

with open("C:\\Users\\kiran\\OneDrive\\Desktop\\Card Fraud Detection\\data_dshape_info.txt", "w") as f:
    f.write(f"Shape of DataFrame: {shape_info[0]} rows, {shape_info[1]} columns\n\n")
    df.info(buf=f)
