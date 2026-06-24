import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import warnings

# Ignore minor convergence warnings to keep the terminal clean
warnings.filterwarnings("ignore")

# Import our data pipeline function
from eda_modeling import load_and_split_data

print("--- Phase 4: Model Training & Evaluation ---")
X_train, X_test, y_train, y_test = load_and_split_data()

if X_train is not None:
    print("\n1. Applying SMOTE to training data (Balancing to 50/50)...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    print("2. Scaling features (Crucial for Neural Networks)...")
    scaler = StandardScaler()
    # FIT on the training data, then TRANSFORM both train and test
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    print("3. Training Artificial Neural Network...")
    print("   -> Architecture: 8 Hidden Units")
    # Setting hidden_layer_sizes to (8,) for an efficient network
    ann_model = MLPClassifier(hidden_layer_sizes=(8,), max_iter=100, random_state=42)
    ann_model.fit(X_train_scaled, y_train_bal)
    
    print("4. Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train_bal)

    print("\n======================================================")
    print("--- Model Evaluation (On UNTOUCHED Test Data) ---")
    print("======================================================")
    
    # ANN Evaluation
    ann_preds = ann_model.predict(X_test_scaled)
    print("\n[ Artificial Neural Network Results ]")
    print(classification_report(y_test, ann_preds, target_names=['Normal', 'Fraud']))
    
    # LR Evaluation
    lr_preds = lr_model.predict(X_test_scaled)
    print("\n[ Logistic Regression Results ]")
    print(classification_report(y_test, lr_preds, target_names=['Normal', 'Fraud']))