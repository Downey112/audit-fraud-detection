import pandas as pd
import sqlite3
import os

# Point to the train.csv file you have in the data folder
csv_path = os.path.join('data', 'train.csv')
db_path = 'fraud_analytics.db'

if not os.path.exists(csv_path):
    print(f"Error: I cannot find 'train.csv'. Make sure it is inside the 'data' folder.")
else:
    print("Loading CSV into memory... (This might take a few seconds)")
    df = pd.read_csv(csv_path)
    # Lowercase all column names for easier SQL querying later
    df.columns = [c.lower() for c in df.columns]

    print("Connecting to SQLite and writing table...")
    conn = sqlite3.connect(db_path)
    df.to_sql('transactions', conn, if_exists='replace', index=False)
    conn.close()
    print("Success! Database created as 'fraud_analytics.db'.")