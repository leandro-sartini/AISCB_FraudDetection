def column_filter(df, parquet_path, csv_path):
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

    # Save DataFrame as Parquet
    df.to_parquet(parquet_path)

    # Save DataFrame as CSV
    df.to_csv(csv_path, index=False)

def main():
    df = pd.read_csv('fraud_detection.csv')  # Assuming 'fraud_detection.csv' exists
    parquet_path = '../../data/interim/fraud_detection.parquet'
    csv_path = 'fraud_detection.csv'
    column_filter(df, parquet_path, csv_path)

if __name__ == "__main__":
    main()



