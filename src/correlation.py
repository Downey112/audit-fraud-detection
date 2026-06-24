import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Import the function you wrote in the other file
from eda_modeling import load_and_split_data

print("--- Starting Phase 2: Correlation Analysis ---")
X_train, X_test, y_train, y_test = load_and_split_data()

if X_train is not None:
    print("\nCalculating Pearson correlations...")
    
    # 1. Temporarily recombine features and target just for the math
    train_data = X_train.copy()
    train_data['is_fraud'] = y_train

    # 2. Compute the Pearson correlation matrix
    corr_matrix = train_data.corr(method='pearson')

    # 3. Isolate how everything correlates specifically to 'is_fraud'
    fraud_corr = corr_matrix[['is_fraud']].drop('is_fraud').sort_values(by='is_fraud', ascending=False)
    
    print("\nTop Feature Correlations with Fraud:")
    print(fraud_corr)

    # 4. Generate the professional heatmap for your GitHub README
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Pearson Correlation Matrix - Fraud Detection Features", fontsize=14, pad=20)
    plt.tight_layout()
    
    # 5. Save the image to your root folder
    image_path = "correlation_heatmap.png"
    plt.savefig(image_path, dpi=300)
    print(f"\nSuccess! Visual saved as '{image_path}'. You can use this in your GitHub README.")