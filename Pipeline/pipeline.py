import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier


def uppercase_strings(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.upper()
    return df
def one_hot_encode(df):
    object_cols = df.select_dtypes(include=['object']).columns
    encoded_cols = []
    for col in object_cols:
        encoded_col = pd.get_dummies(df[col], prefix=col)
        encoded_cols.append(encoded_col)
        encoded_df = pd.concat([df] + encoded_cols, axis=1)
        encoded_df.drop(columns=object_cols, inplace=True)

    return encoded_df

def one_hot_decode(encoded_df):
    
    decoded_df = encoded_df
    for column in encoded_df.columns:
        
        prefix, _, category = column.partition('_')

       
        if prefix not in decoded_df.columns:
            
            decoded_df[prefix] = encoded_df[column].apply(lambda x: category if x == 1 else None)

    return decoded_df

def pipeline(filename):
    df=pd.read_csv('fraud_detection.csv')
    print(df)
    
    # Step 1: to uppercase
    print("Step 2:convert to uppercase...")
    upper_data = uppercase_strings(df)
    print(upper_data)
    
    # Step 2:one-hot encoding
    onehot_data=one_hot_encode(upper_data)
    print(onehot_data)
    #step 3:one-hot decoding
    onehotdecode_data=one_hot_decode(onehot_data)
    print(onehot_data)
    
def main():
    df=pd.read_csv('fraud_detection.csv')
    pipeline(df)
    
# Execute the pipeline
if __name__ == "__main__":
    main()
    


