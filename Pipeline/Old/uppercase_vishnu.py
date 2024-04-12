


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import numpy as np
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NearMiss





def uppercase_str(df):
    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].apply(lambda x: x.str.upper())
    return df







