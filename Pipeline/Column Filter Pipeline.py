#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import os

def column_filter(df):
    
    parquet_path = '../../data/interim/fraud_detection.parquet'
    csv_path = 'fraud_detection.csv'

    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

    df.to_parquet(parquet_path)

    df.to_csv(csv_path, index=False)

def main():
    column_filter(df)

if __name__ == "__main__":
    main()


# In[ ]:




