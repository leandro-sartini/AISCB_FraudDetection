#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier


# In[12]:


import nbimporter
import onehotEncoding_sherin


# In[13]:


df=pd.read_csv('fraud_detection.csv')


# In[14]:


df


# In[15]:


encoded_df=onehotEncoding_sherin.one_hot_encode(df)


# In[16]:


encoded_df


# In[17]:


import pandas as pd

def one_hot_decode(encoded_df):
    
    decoded_df = pd.DataFrame()

    # Iterate over each column in the encoded DataFrame
    for column in encoded_df.columns:
        # Split the column name to get prefix and category
        prefix, _, category = column.partition('_')

        # Check if the prefix already exists in decoded_df
        if prefix not in decoded_df.columns:
            # Create a new column in decoded_df for the prefix
            decoded_df[prefix] = encoded_df[column].apply(lambda x: category if x == 1 else None)

    return decoded_df



# In[18]:


encoded_df1 = pd.DataFrame(encoded_df)


# In[19]:


# Reverse one-hot encoding
decoded_df = one_hot_decode(encoded_df1)


# In[20]:


# Display the decoded DataFrame
print("Decoded DataFrame:")
print(decoded_df)


# In[ ]:




