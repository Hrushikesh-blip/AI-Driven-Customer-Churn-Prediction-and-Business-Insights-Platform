import pandas as pd

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Clean and preprocess the dataset."""
    df = df.dropna()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = pd.get_dummies(df, drop_first=True)
    return df

if __name__ == "__main__":
    data = load_data("data/raw/telco_customer_churn.csv")
    clean_data = preprocess_data(data)
    clean_data.to_csv("data/processed/cleaned_data.csv", index=False)
