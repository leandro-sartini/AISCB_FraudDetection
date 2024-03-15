#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import numpy as np
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split,cross_val_score,KFold,RandomizedSearchCV


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

    X_resampled,y_resampled = oversample_with_adasyn(X_train,y_train,target_minority_samples)
    X_train,Y_train = undersample_with_nearmiss(X_resampled,y_resampled)
    
    return X_train,Y_train


# In[ ]:




