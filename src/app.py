# src/app.py
import streamlit as st
import pandas as pd
import joblib
import sqlite3
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title="Fraud XAI Dashboard", layout="wide")
st.title("Fraud Detection with Explainable AI")
# ---------------------
# Load model artifacts
@st.cache_resource
def load_artifacts(path='models/model.joblib'):
    obj = joblib.load(path)
    return obj['model'], obj['scaler'], obj['features']

model, scaler, features = load_artifacts()

# ---------------------
# Data loading
uploaded = st.file_uploader(
    "Upload transactions CSV (leave empty for demo)", type=["csv"]
)
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("No upload — using demo synthetic data")
    df = pd.read_csv("data/synthetic_creditcard.csv").sample(200, random_state=1).reset_index(drop=True)

df = df.fillna(0)

# ---------------------
# Prediction function
def predict_df(model, scaler, features, df):
    X = df[features].values
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[:,1]
    pred = (proba >= 0.5).astype(int)
    df2 = df.copy()
    df2['fraud_proba'] = proba
    df2['predicted_class'] = pred
    return df2

df_preds = predict_df(model, scaler, features, df)

# ---------------------
# Sidebar settings
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Flag threshold probability", 0.01, 0.99, 0.5)
row_idx = st.sidebar.number_input(
    "Select row index for explanations", 0, len(df_preds)-1, 0
)

# ---------------------
# Tabs layout
tabs = st.tabs(["Dashboard", "Transaction Table", "Explanations"])

# ---------------------
# Tab 1: Dashboard plots
with tabs[0]:
    st.header("Fraud Analytics Overview")
    # Fraud probability histogram
    fig = px.histogram(df_preds, x="fraud_proba", nbins=50, title="Fraud Probability Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # Top flagged transactions
    flags = df_preds[df_preds['fraud_proba'] >= threshold].sort_values('fraud_proba', ascending=False)
    fig2 = px.bar(flags.head(20), x='fraud_proba', y='Class', orientation='h', title="Top 20 Flagged Transactions")
    st.plotly_chart(fig2, use_container_width=True)
    st.write(f"Total flagged transactions (threshold {threshold:.2f}): {len(flags)}")

    # Download flagged CSV
    csv = flags.to_csv(index=False).encode()
    st.download_button(
        label="Download flagged transactions CSV",
        data=csv,
        file_name='flagged_transactions.csv',
        mime='text/csv'
    )

# ---------------------
# Tab 2: Transaction Table
with tabs[1]:
    st.header("All Transactions Table")
    st.dataframe(df_preds)

    # Optional: save flagged to SQLite
    if st.button("Save flagged transactions to SQLite"):
        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect("data/predictions.db")
        flags.to_sql("flags", conn, if_exists='append', index=False)
        conn.close()
        st.success("Saved flagged transactions to data/predictions.db")

# ---------------------
# Tab 3: Explanations
with tabs[2]:
    st.header("Explainability")

    # SHAP Explainer
    if st.button("Show Local SHAP (waterfall)"):
        Xs = scaler.transform(df_preds[features].values)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(Xs)

        st.write(f"Explaining row {row_idx} (prob={df_preds.iloc[row_idx]['fraud_proba']:.3f})")
        fig, ax = plt.subplots(figsize=(8,4))
        shap.plots._waterfall.waterfall_legacy(shap_vals[1][row_idx], feature_names=features, max_display=10, show=False)
        st.pyplot(fig)

        # Top contributions
        contribs = sorted(zip(features, shap_vals[1][row_idx]), key=lambda x: abs(x[1]), reverse=True)
        st.write("Top feature contributions:", contribs[:10])

    if st.button("Show Global SHAP Summary"):
        Xs = scaler.transform(df_preds[features].values)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(Xs)
        fig = plt.figure(figsize=(8,6))
        shap.summary_plot(shap_vals[1], Xs, feature_names=features, show=False)
        st.pyplot(fig)

    # LIME explanation
    if st.button("Show LIME Explanation"):
        Xs = scaler.transform(df_preds[features].values)
        explainer = LimeTabularExplainer(
            training_data=Xs,
            feature_names=features,
            class_names=['legit','fraud'],
            discretize_continuous=True
        )
        exp = explainer.explain_instance(Xs[row_idx], model.predict_proba, num_features=10)
        st.write("LIME Explanation (list):", exp.as_list())
        st.components.v1.html(exp.as_html(), height=400, scrolling=True)

st.sidebar.markdown("**Notes:** SHAP and LIME require the 'shap' and 'lime' packages installed.")
