import streamlit as st
import subprocess
import sys
import os

st.set_page_config(page_title="AI Blood Supply Management System", layout="wide")

st.title("AI-Driven Blood Supply Management System")
st.write("Final integrated dashboard for demand forecasting, donor management, reminders, and explainable AI.")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Add Donor",
        "Demand Forecast",
        "Reminder Monitoring",
        "Explainable AI",
        "Reminder Settings",
        "Model Performance" 
    ]
)

if page == "Home":
    st.header("System Overview")
    st.write("""
    This integrated system supports:
    - Blood demand forecasting using ARIMA and LSTM
    - Donor return prediction and ranking
    - 4-month donor reminder generation
    - Explainable AI using SHAP
    """)

elif page == "Add Donor":
    exec(open("src/dashboard/donor_input_dashboard.py").read())

elif page == "Demand Forecast":
    exec(open("src/dashboard/demand_forecast_dashboard.py").read())

elif page == "Reminder Monitoring":
    exec(open("src/dashboard/reminder_dashboard.py").read())

elif page == "Explainable AI":
    exec(open("src/dashboard/xai_dashboard.py").read())
    
elif page == "Model Performance":
    exec(open("src/dashboard/model_performance_dashboard.py").read())
    
elif page == "Reminder Settings":
    exec(open("src/dashboard/reminder_settings_dashboard.py").read())