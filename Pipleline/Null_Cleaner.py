#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings 
warnings.filterwarnings("ignore")


# In[3]:


cd


# In[11]:


import pandas as pd
df = pd.read_csv('fraud_detection.csv')
def null():
    
    
    df = pd.read_csv('fraud_detection.csv')
    transformed_column_AMT_ANNUITY = df.groupby(['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN'])['AMT_ANNUITY'].transform('median')
    filled_column = transformed_column_AMT_ANNUITY.fillna(transformed_column_AMT_ANNUITY.median())
    df['AMT_ANNUITY'] = filled_column

    transformed_column_AMT_GOODS_PRICE = df.groupby(['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN'])['AMT_GOODS_PRICE'].transform('median')
    filled_column_AMT_GOODS_PRICE = transformed_column_AMT_GOODS_PRICE.fillna(transformed_column_AMT_GOODS_PRICE.median())
    df['AMT_GOODS_PRICE'] = filled_column_AMT_GOODS_PRICE

    transformed_column_CNT_FAM_MEMBERS = df.groupby(['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN'])['CNT_FAM_MEMBERS'].transform('median')
    filled_column_CNT_FAM_MEMBERS = transformed_column_CNT_FAM_MEMBERS.fillna(transformed_column_CNT_FAM_MEMBERS.median())
    df['CNT_FAM_MEMBERS'] = filled_column_CNT_FAM_MEMBERS

    df.drop(columns=['OWN_CAR_AGE'], inplace=True)

    df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].fillna('No_Ref')

    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('No_Ref')

    
if __name__ == "__main__":
    null()



# In[ ]:




