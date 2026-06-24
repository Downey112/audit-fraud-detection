import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split

def load_and_split_data():
    print("Connecting to database...")
    conn = sqlite3.connect('fraud_analytics.db')
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()
    
    print(f"Total records fetched: {df.shape[0]}")
    
    target_col = 'is_fraud'
    
    if target_col not in df.columns:
        print(f"Error: Target column not found.")
        return None, None, None, None

    # Separate features (X) and target (y)
    # We also drop 'unnamed: 0' because it is just a useless duplicate ID column
    cols_to_drop = [target_col]
    if 'unnamed: 0' in df.columns:
        cols_to_drop.append('unnamed: 0')
        
    X = df.drop(columns=cols_to_drop)
    
    # CRITICAL: Keep only numeric columns for our Pearson correlation and ML models
    X = X.select_dtypes(include=['int64', 'float64'])
    
    y = df[target_col]
    
    # Stratified Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"\n--- Stratified Split Completed ---")
    print(f"Training set size: {X_train.shape[0]} rows")
    print(f"Testing set size:  {X_test.shape[0]} rows")
    print(f"Fraud ratio in Training Set: {y_train.mean():.4%}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()