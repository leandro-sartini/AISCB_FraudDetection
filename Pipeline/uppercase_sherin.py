import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier



df=pd.read_csv('fraud_detection.csv')

def uppercase_strings(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.upper()
    return df
new_df=uppercase_strings(df)
new_df







