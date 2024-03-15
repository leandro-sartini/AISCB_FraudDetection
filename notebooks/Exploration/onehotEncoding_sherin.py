#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier


# In[2]:


df=pd.read_csv('fraud_detection.csv')


# In[3]:


df


# In[4]:


def one_hot_encode(df):
    object_cols = df.select_dtypes(include=['object']).columns
    #print(object_cols)

    # Perform one-hot encoding for each selected column
    encoded_cols = []
    for col in object_cols:
        encoded_col = pd.get_dummies(df[col], prefix=col)
        encoded_cols.append(encoded_col)
        #print("ENCODED COLUMNS")
        #print(encoded_cols)

    # Concatenate the encoded columns with the original DataFrame
    encoded_df = pd.concat([df] + encoded_cols, axis=1)

    # Drop the original columns after encoding
    encoded_df.drop(columns=object_cols, inplace=True)

    return encoded_df


# In[5]:


encoded_df=one_hot_encode(df)


# In[6]:


encoded_df


# In[ ]:




