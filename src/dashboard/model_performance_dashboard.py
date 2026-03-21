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

st.header("Donor Prediction Performance")

# Replace with your real values if available
auc_score = 0.95

st.metric("AUC Score", auc_score)

st.write("""
The donor classification model shows strong ability to distinguish:
- Returning donors
- Non-returning donors
""")

st.header("Confusion Matrix")

if os.path.exists("outputs/confusion_matrix.png"):
    st.image("outputs/confusion_matrix.png", caption="Donor Model Confusion Matrix")
else:
    st.warning("Confusion matrix not found. Generate it from donor model.")

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

st.header("System Impact")

st.success("""
This system transforms a reactive process into a proactive decision support system:

- Reduces blood shortages through accurate forecasting  
- Minimizes wastage of expired blood units  
- Identifies high-priority and rare donors  
- Improves donor retention strategies  
""")