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


#function to convert strings to uppercase


# In[5]:


def uppercase_strings(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.upper()
    return df


# In[6]:


new_df=uppercase_strings(df)


# In[7]:


new_df


# In[ ]:




