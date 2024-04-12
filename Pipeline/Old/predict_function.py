import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def predict_data(data, path_to_model):
    with open(path_to_model, 'rb') as file:
        fraud_predict = pickle.load(file)

    data['PredictedData'] = fraud_predict.predict(data).round(2)

    return data
