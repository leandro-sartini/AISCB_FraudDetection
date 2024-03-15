#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier


# In[63]:


df=pd.read_csv('fraud_detection.csv')


# In[64]:


df


# In[65]:


#function to convert strings to uppercase


# In[66]:


def uppercase_strings(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.upper()
    return df


# In[67]:


new_df=uppercase_strings(df)


# In[68]:


new_df


# In[69]:


#function to do one_hot_encoding and reverse


# In[70]:


df


# In[71]:


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



# In[72]:


encoded_df=one_hot_encode(df)


# In[73]:


encoded_df


# In[ ]:


#CODE TO REVERSE ONE HOT ENCODING


# In[74]:


import pandas as pd

def one_hot_decode(encoded_df):
    """
    Reverse one-hot encoding for columns with object datatype in a DataFrame.

    Args:
    - encoded_df (pandas.DataFrame): Input DataFrame with one-hot encoded columns.

    Returns:
    - decoded_df (pandas.DataFrame): DataFrame with original categorical columns.
    """
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



# In[78]:


encoded_df1 = pd.DataFrame(encoded_df)




# In[79]:


# Reverse one-hot encoding
decoded_df = one_hot_decode(encoded_df1)


# In[76]:


# Display the decoded DataFrame
print("Decoded DataFrame:")
print(decoded_df)


# In[ ]:




