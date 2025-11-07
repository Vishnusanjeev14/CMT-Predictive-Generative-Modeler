"""
create_models.py

Loads the synthetic dataset, trains RandomForestClassifier, 
and saves the model, encoder, and scaler to disk.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_and_preprocess(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Define categorical and numerical features
    cat_cols = ["gene_variant_id", "mutation_type"]
    num_cols = ["age_of_onset", "motor_score", "sensory_score"]
    
    # 2. Fit OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(df[cat_cols])
    
    # 3. Fit StandardScaler
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols].values.astype(float))
    
    # 4. Combine features
    X = np.hstack([X_num, X_cat])
    
    # 5. Encode target variable
    label_map = {"mild":0, "moderate":1, "severe":2}
    y = df["disease_severity"].map(label_map).values
    
    print("Preprocessing complete.")
    return X, y, encoder, scaler, label_map

def train_and_save(X, y, encoder, scaler, out_dir="."):
    print("Splitting data and training RandomForest model...")
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Train the RandomForest model
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"\nModel training complete. Accuracy: {acc_rf:.4f}")
    print(classification_report(y_test, y_pred_rf, target_names=['mild','moderate','severe']))
    
    # --- Save the files ---
    out_path = Path(out_dir)
    
    model_path = out_path / "rf_model.pkl"
    encoder_path = out_path / "encoder.pkl"
    scaler_path = out_path / "scaler.pkl"
    
    joblib.dump(rf, model_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nSuccessfully saved:")
    print(f"  Model -> {model_path}")
    print(f"  Encoder -> {encoder_path}")
    print(f"  Scaler -> {scaler_path}")

if __name__ == "__main__":
    # Ensure the dataset is in the same folder as this script
    dataset_file = "cmt_synthetic_dataset.csv"
    
    X, y, encoder, scaler, label_map = load_and_preprocess(dataset_file)
    train_and_save(X, y, encoder, scaler)