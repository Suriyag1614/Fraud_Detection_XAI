# src/explainers.py
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

def load_artifact(path='models/model.joblib'):
    obj = joblib.load(path)
    return obj['model'], obj['scaler'], obj['features']

def shap_summary_for_test(model, X_test, features, out='shap_summary.png'):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)  # for binary: shap_vals[1] is class-1
    plt.figure(figsize=(8,6))
    # summary plot (class=1)
    shap.summary_plot(shap_vals[1], X_test, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig(out)
    print("Saved", out)

if __name__ == "__main__":
    model, scaler, features = load_artifact()
    # example: load a sample test X (you should pass your X_test)
    # shap_summary_for_test(model, X_test_scaled, features)
