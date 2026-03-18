import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from demand_forecasting.forecast_engine import generate_forecast

st.title("Demand Forecast Dashboard")
st.write("Generate blood demand forecasts by blood bank and blood type.")

blood_bank = st.text_input("Enter Blood Bank", "Accident Service")
blood_type = st.selectbox("Select Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

if st.button("Generate Forecast"):
    try:
        generate_forecast(blood_bank, blood_type)
        safe_bank = blood_bank.replace("/", "-").replace(" ", "_")
        safe_type = blood_type.replace("+", "pos").replace("-", "neg")
        file_path = f"outputs/{safe_bank}_{safe_type}_forecast.csv"

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            st.success("Forecast generated successfully.")
            st.dataframe(df)
        else:
            st.warning("Forecast file was not found after generation.")
    except Exception as e:
        st.error(f"Error generating forecast: {e}")