# src/train_model.py
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_data(path):
    return pd.read_csv(path)

def train(data_path='data/synthetic_creditcard.csv', model_out='models/model.joblib'):
    df = load_data(data_path)
    features = [c for c in df.columns if c != 'Class']
    X = df[features].values
    y = df['Class'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE on training data only
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
    print("After SMOTE:", np.bincount(y_res.astype(int)))

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42,
                                 class_weight='balanced_subsample')
    clf.fit(X_res, y_res)

    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    print(classification_report(y_test, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("PR AUC (avg precision):", average_precision_score(y_test, y_proba))

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump({'model': clf, 'scaler': scaler, 'features': features}, model_out)
    print("Saved model to", model_out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic_creditcard.csv")
    p.add_argument("--out", default="models/model.joblib")
    args = p.parse_args()
    train(args.data, args.out)
