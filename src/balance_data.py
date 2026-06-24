import pandas as pd
from imblearn.over_sampling import SMOTE
import os

# Import the split function again
from eda_modeling import load_and_split_data

print("--- Starting Phase 3: SMOTE Data Balancing ---")
X_train, X_test, y_train, y_test = load_and_split_data()

if X_train is not None:
    print(f"\nOriginal Training Fraud Count: {sum(y_train == 1)}")
    print(f"Original Training Normal Count: {sum(y_train == 0)}")
    
    print("\nApplying SMOTE... (This synthesizes data and takes a few seconds)")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print("\n--- SMOTE Complete ---")
    print(f"Balanced Training Fraud Count: {sum(y_train_balanced == 1)}")
    print(f"Balanced Training Normal Count: {sum(y_train_balanced == 0)}")
    print(f"New Fraud Ratio: {y_train_balanced.mean():.2%}")