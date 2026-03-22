import streamlit as st
import pandas as pd
import os

st.title("Model Performance & System Impact")

st.header("Problem Overview")

st.write("""
The current blood supply system suffers from:

- Inaccurate demand forecasting → leading to shortages and expired blood waste  
- Inefficient donor management → difficulty identifying reliable and rare donors  

This system addresses both challenges using AI-based forecasting and donor intelligence.
""")

st.header("Demand Forecasting Performance")

# You manually input your results here (from your experiments)
lstm_rmse = 1.98
arima_rmse = 6.73

st.metric("LSTM RMSE (Better Model)", lstm_rmse)
st.metric("ARIMA RMSE", arima_rmse)

st.success("LSTM significantly improves forecasting accuracy compared to ARIMA.")

# --- Donor Model Evaluation Metrics (clean column layout) ---
st.header("Donor Model Evaluation Metrics")

if os.path.exists("outputs/donor_metrics.csv"):
    metrics_df = pd.read_csv("outputs/donor_metrics.csv")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy",  round(metrics_df.loc[0, "Value"], 3))
    col2.metric("Precision", round(metrics_df.loc[1, "Value"], 3))
    col3.metric("Recall",    round(metrics_df.loc[2, "Value"], 3))

    col1.metric("F1 Score",  round(metrics_df.loc[3, "Value"], 3))
    col2.metric("AUC",       round(metrics_df.loc[4, "Value"], 3))

else:
    st.warning("Run donor_model.py to generate metrics.")

# --- Confusion Matrix ---
st.header("Confusion Matrix")

if os.path.exists("outputs/confusion_matrix.png"):
    st.image("outputs/confusion_matrix.png", caption="Donor Model Confusion Matrix")
else:
    st.warning("Confusion matrix not found. Generate it from donor model.")

# --- ROC Curve ---
st.header("ROC Curve")

if os.path.exists("outputs/roc_curve.png"):
    st.image("outputs/roc_curve.png", caption="ROC Curve - Donor Prediction")
else:
    st.warning("ROC curve not found. Run donor_model.py")

# --- Feature Importance ---
st.header("Feature Importance Ranking")

if os.path.exists("outputs/feature_importance.csv"):
    importance_df = pd.read_csv("outputs/feature_importance.csv")

    st.dataframe(importance_df)

    st.bar_chart(importance_df.set_index("Feature"))

else:
    st.warning("Feature importance not found.")

# --- Explainable AI Insights ---
st.header("Explainable AI Insights")

st.write("""
SHAP explanations provide transparency in predictions:
- Identify key factors affecting donor return
- Explain demand drivers such as holidays and seasonal effects
""")

if os.path.exists("outputs/xai/donor_feature_importance.png"):
    st.image("outputs/xai/donor_feature_importance.png", caption="Donor Feature Importance")

if os.path.exists("outputs/xai/demand_feature_importance.png"):
    st.image("outputs/xai/demand_feature_importance.png", caption="Demand Feature Importance")

# --- System Improvement ---
st.header("System Improvement (Before vs After)")

if os.path.exists("outputs/system_comparison.csv"):
    df = pd.read_csv("outputs/system_comparison.csv")

    st.dataframe(df)

    st.bar_chart(df.set_index("Scenario"))

    st.success("The AI system significantly reduces forecasting error and improves decision-making.")

else:
    st.warning("Run system_comparison.py first.")

# --- System Impact ---
st.header("System Impact")

st.success("""
This system transforms a reactive process into a proactive decision support system:

- Reduces blood shortages through accurate forecasting  
- Minimizes wastage of expired blood units  
- Identifies high-priority and rare donors  
- Improves donor retention strategies  
""")

# --- Real-Time Model Re-Evaluation ---
st.header("Run Model Evaluation")

if st.button("Recalculate Model Metrics"):
    os.system("python src/donor_management/donor_model.py")
    st.success("Model re-evaluated successfully.")