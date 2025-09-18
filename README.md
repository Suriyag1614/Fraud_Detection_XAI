# Fraud Detection with Explainable AI (XAI)

Demo project: trains a fraud classifier on a credit-card-like dataset, uses SMOTE for imbalance, and explains predictions with SHAP and LIME. Includes an interactive Streamlit dashboard.

## Quickstart
1. python -m venv .venv && source .venv/bin/activate
2. pip install -r requirements.txt
3. python src/data_generator.py
4. python src/train_model.py --data data/synthetic_creditcard.csv
5. streamlit run src/app.py

## Notes
- Kaggle original dataset (benchmark): 492 frauds out of 284,807 transactions. :contentReference[oaicite:11]{index=11}
- SHAP docs: `pip install shap`. :contentReference[oaicite:12]{index=12}
- LIME docs: `pip install lime`. :contentReference[oaicite:13]{index=13}
- SMOTE via imbalanced-learn. :contentReference[oaicite:14]{index=14}

## Dashboard
[Click Here](https://fraud-detection-xai.streamlit.app/) to view the streamlit deployment.
