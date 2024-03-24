#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_auc_score,f1_score
#from sklearn.model_selection import train_test_split,cross_val_score,KFold,RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NearMiss
from sklearn.datasets import make_classification
import numpy as np
from collections import Counter
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


# In[3]:


def mid_sampling(X_train,Y_train,target_minority_samples=180000):
    
    def oversample_with_adasyn(X, y, target_minority_samples=180000):
        # Identify minority class
        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        current_minority_samples = counts[np.argmin(counts)]

    # Calculate how many new samples we need to generate for the minority class
        if current_minority_samples >= target_minority_samples:
            raise ValueError("The minority class already has more or equal samples than the target_minority_samples.")
    
        n_samples_to_generate = target_minority_samples - current_minority_samples

    # Set the sampling strategy for ADASYN
    # This will tell ADASYN to generate the specified number of samples for the minority class
        sampling_strategy = {minority_class: n_samples_to_generate}

    # Create the ADASYN instance and fit it to the data
        adasyn = ADASYN(sampling_strategy=sampling_strategy, n_neighbors=5, random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)

        return X_resampled, y_resampled
    def undersample_with_nearmiss(X, y):
    
    # Identify majority and minority classes
        classes, counts = np.unique(y, return_counts=True)
        minority_class_count = min(counts)
    
    # Set the sampling strategy for NearMiss
    # This will balance the majority class with the minority class
        nm = NearMiss()
    
    # Perform the undersampling
        X_resampled, y_resampled = nm.fit_resample(X, y)
    
    # Check if undersampling was successful
        if len(np.unique(y_resampled, return_counts=True)[1]) != 2 or np.unique(y_resampled, return_counts=True)[1][0] != minority_class_count:
            raise ValueError("NearMiss did not undersample correctly.")
    
        return X_resampled, y_resampled

    X_resampled,y_resampled = oversample_with_adasyn(X_train,Y_train,target_minority_samples)
    X_train,Y_train = undersample_with_nearmiss(X_resampled,y_resampled)
    
    return X_train,Y_train


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


# In[5]:


def fillnulls(data,grouped_cols,imput_dict):
    imputation_dict = imput_dict
    
    col_x,col_y,col_z =grouped_cols
    
    for col in imputation_dict:
        for val in imputation_dict[col]:
            val_x,val_y,val_z = val
            data.loc[(data[col_x] == val_x) & (data[col_z] == val_z) & (data[col_z] == val_z),col] = imputation_dict[col][val]
    return data


# In[6]:


def null_dict(df,grouped_cols):
    imputation_dict = {}
    # Calculate the sum of null values for each column
    null_sum = df.isnull().sum()

    # Filter columns where the sum of null values is greater than 0
    columns_with_nulls = null_sum[null_sum > 0].index

    # Select only the columns from the original DataFrame that have more than 0 null values
    null_cols = df.loc[:, columns_with_nulls].columns
    for col in null_cols:
        if (df[col].dtypes == 'int64') | (df[col].dtypes == 'float64'):
            imputation_dict.update({col : df.groupby(grouped_cols)[col].median().to_dict()})
        elif (df[col].dtypes == 'O'):
            imputation_dict.update({col : df.groupby(grouped_cols)[col].agg(pd.Series.mode).to_dict()})
                
    return imputation_dict


# In[7]:


def drop_high_nulls_col(df):
    # Grouped median calculation for filling missing values
    
    for cols in df.columns:
        if (df[cols].notnull().sum() / df.shape[0]) < 0.4:
            df.drop(columns=[cols], inplace=True)
            
    return df


# In[8]:


def uppercase_str(df):
    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].apply(lambda x: x.str.upper())
    return df


# In[9]:


def pipeline(df):
    # read the file name
    df = pd.read_csv(r"C:\Users\vishnu\fraud_detection.csv")
    df.drop(columns = ['SK_ID_CURR','AMT_GOODS_PRICE','CODE_GENDER'],inplace = True)
    df = uppercase_str(df)
    if (df['AMT_ANNUITY'].dtypes == 'int64') | (df['AMT_ANNUITY'].dtypes == 'float64'):
        print('x')
    df = drop_high_nulls_col(df)
    X_col = df.drop(columns = 'TARGET').columns
    y_col = ['TARGET']
    group_cols = ['NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    X_train,X_test,y_train,y_test = train_test_split(df[X_col],df[y_col],stratify=df[group_cols],train_size=0.8,test_size = 0.2,random_state=37)
    imput_dict = null_dict(X_train,group_cols)
    X_train = fillnulls(X_train,group_cols,imput_dict)
    X_test = fillnulls(X_test,group_cols,imput_dict)
    x,y = oneh(df,X_train,X_test)
    X_train = x
    x_test = y
    x, y = mid_sampling(X_train,y_train)
    X_train = x
    Y_train = y
    x, y =scaling(X_train,X_test)
    X_train = x
    X_test   = y
    return X_train,X_test,y_train,y_test
    


# In[ ]:


def main():
    df = pd.read_csv(r"C:\Users\vishnu\fraud_detection.csv")
    pipeline(df)
if __name__ == "__main__":
    main()

