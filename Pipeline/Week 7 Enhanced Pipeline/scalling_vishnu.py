#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import numpy as np
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split,cross_val_score,KFold,RandomizedSearchCV


# In[2]:


def scaling(X_train,X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    return X_train,X_test


# In[ ]:




