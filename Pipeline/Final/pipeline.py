import pandas as pd
import numpy as np
from joblib import load
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pickle


def one_hot_encode(data):
    oneh = load('../../models/onehot_model.joblib')
    col_names = oneh.get_feature_names_out()
    object_cols = data.select_dtypes(include=['object']).columns
    data_dummies = pd.DataFrame(oneh.transform(data[object_cols]), columns=col_names)
    data_dummies.index = data.index
    data = pd.merge(data, data_dummies, left_index=True, right_index=True)
    data.drop(columns=object_cols, inplace=True)
    return data


def scaling(data):
    scaler = load('../../models/scaling_model.joblib')
    data_scaled = scaler.transform(data)
    data = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
    return data


def load_columns():
    with open('../../models/columns.txt', 'r') as f:
        cols = [line.strip() for line in f]
    return list(cols)


def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


def predict_data(data, cols, loaded_model):
    data['Prediction'] = loaded_model.predict(data)
    return data


def pipeline(data, filename):
    print("Step 2:One Hot\n\n")
    data = one_hot_encode(data)
    print("Step 3:Scale\n\n")
    data = scaling(data)
    print("Step 4:Load Columns\n\n")
    cols = load_columns()
    print("Step 5:Load Model\n\n")
    loaded_model = load_model(filename)
    print("Step 6:Predict\n\n")
    data = predict_data(data, cols, loaded_model)

    data['ID'] = data.index
    prediction_cols = data.drop(columns=['Prediction', 'ID']).columns
    data['Probabilities'] = loaded_model.predict_proba(data[prediction_cols])[:, 1]
    data = data[['ID', 'Prediction', 'Probabilities']]
    data.to_csv('../../data/predicted/Predicted_file.csv', index=False)
    return data

def main():
    df = pd.read_csv('../../data/processed/Sample_Data.csv')

    # Continue with your pipeline...
    pipeline(df, '../../models/best_logistic.sav')


if __name__ == "__main__":
    main()
