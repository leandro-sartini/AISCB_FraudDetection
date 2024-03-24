import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

df=pd.read_csv('fraud_detection.csv')

def one_hot_encode(df):
    object_cols = df.select_dtypes(include=['object']).columns
    encoded_cols = []
    for col in object_cols:
        encoded_col = pd.get_dummies(df[col], prefix=col)
        encoded_cols.append(encoded_col)
        encoded_df = pd.concat([df] + encoded_cols, axis=1)
        encoded_df.drop(columns=object_cols, inplace=True)

    return encoded_df


encoded_df=one_hot_encode(df)

encoded_df





