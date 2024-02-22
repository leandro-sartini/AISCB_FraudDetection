import pandas as pd

def clean_data(df):
    # Grouped median calculation for filling missing values
    grouped_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN']
    for col in ['AMT_ANNUITY', 'AMT_GOODS_PRICE', 'CNT_FAM_MEMBERS']:
        df[col].fillna(df.groupby(grouped_cols)[col].transform('median'), inplace=True)

    # Drop column 'OWN_CAR_AGE' if it exists and all values are missing
    if 'OWN_CAR_AGE' in df.columns and df['OWN_CAR_AGE'].isnull().all():
        df.drop(columns=['OWN_CAR_AGE'], inplace=True)

    # Replace missing values in 'NAME_TYPE_SUITE' and 'OCCUPATION_TYPE'
    df['NAME_TYPE_SUITE'].fillna('No_Ref', inplace=True)
    df['OCCUPATION_TYPE'].fillna('No_Ref', inplace=True)

    return df

if __name__ == "__main__":
    df = pd.read_csv('fraud_detection.csv')
    cleaned_df = clean_data(df)
    cleaned_df.to_csv("fraud_detection_null_cleaner.csv", index=False)




