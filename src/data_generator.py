# src/data_generator.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import os
import argparse

def generate_creditcard_like_csv(output_path='data/synthetic_creditcard.csv',
                                 n_samples=100000,
                                 imbalance_ratio=0.002,
                                 random_state=42):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_features = 28  # like V1..V28
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=10,
                               n_redundant=8,
                               n_clusters_per_class=2,
                               weights=[1 - imbalance_ratio, imbalance_ratio],
                               flip_y=0.001,
                               class_sep=1.0,
                               random_state=random_state)
    df = pd.DataFrame(X, columns=[f'V{i}' for i in range(1, n_features + 1)])
    rng = np.random.RandomState(random_state)
    df['Time'] = rng.randint(0, 172800, size=n_samples)              # seconds over 2 days
    df['Amount'] = np.round(np.abs(rng.normal(50, 200, size=n_samples)), 2)
    df['Class'] = y
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} ({n_samples} rows). Fraud count: {df['Class'].sum()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/synthetic_creditcard.csv")
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--imbalance", type=float, default=0.002)
    args = p.parse_args()
    generate_creditcard_like_csv(args.output, args.n, args.imbalance)
