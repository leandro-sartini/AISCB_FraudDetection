#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import numpy as np
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split,cross_val_score,KFold,RandomizedSearchCV


# In[4]:


def oneh(df,X_train,X_test):
    object_cols = df.select_dtypes(include=['object']).columns
    oneh = OneHotEncoder(sparse_output=False)
    oneh.fit(X_train[object_cols])
    col_names = oneh.get_feature_names_out()
    X_train_dummies = pd.DataFrame(oneh.transform(X_train[object_cols]),columns = col_names)
    X_train_dummies.index = X_train.index
    X_train = pd.merge(X_train,X_train_dummies,left_index=True,right_index=True)

    X_test_dummies = pd.DataFrame(oneh.transform(X_test[object_cols]),columns = col_names)
    X_test_dummies.index = X_test.index
    X_test = pd.merge(X_test,X_test_dummies,left_index=True,right_index=True)
    return X_train,X_test


# In[ ]:




