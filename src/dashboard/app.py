import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from auth.auth import init_auth_state, logout_user, get_allowed_pages

st.set_page_config(page_title="RedPredict", page_icon="🩸", layout="wide")

init_auth_state()

if not st.session_state["logged_in"]:
    exec(open("src/dashboard/login_page.py").read())
    st.stop()

st.title("RedPredict | AI-Driven Blood Supply Management System")
st.write("Integrated decision support system for demand forecasting, donor management, reminders, explainable AI, and performance monitoring.")

st.sidebar.success(f"Logged in as: {st.session_state['username']}")
st.sidebar.info(f"Role: {st.session_state['role']}")

if st.sidebar.button("Logout"):
    logout_user()
    st.rerun()

allowed_pages = get_allowed_pages(st.session_state["role"])

page = st.sidebar.radio("Go to", allowed_pages)

if page == "Home":
    st.header("System Overview")
    st.markdown("""
### Core Problems Addressed
- Inaccurate blood demand forecasting, causing shortages and expired blood waste
- Inefficient donor management, making it difficult to identify reliable and rare blood donors

### Main Modules
- **Demand Forecasting** using ARIMA and LSTM
- **Donor Management** using classification and prioritization
- **Reminder System** based on configurable donation intervals
- **Explainable AI** using SHAP
- **Performance & Evaluation** using RMSE, MAE, MAPE, Accuracy, Precision, Recall, F1, and AUC

### Operational Features
- Donor registration
- Reminder generation
- Communication logging
- Shortage alerts
- Dashboard-based monitoring
""")

elif page == "Add Donor":
    exec(open("src/dashboard/donor_input_dashboard.py").read())

elif page == "Demand Forecast":
    exec(open("src/dashboard/demand_forecast_dashboard.py").read())

elif page == "Reminder Monitoring":
    exec(open("src/dashboard/reminder_dashboard.py").read())

elif page == "Reminder Settings":
    exec(open("src/dashboard/reminder_settings_dashboard.py").read())

elif page == "Communication Log":
    exec(open("src/dashboard/communication_log_dashboard.py").read())

elif page == "Model Performance":
    exec(open("src/dashboard/model_performance_dashboard.py").read())

elif page == "Explainable AI":
    exec(open("src/dashboard/xai_dashboard.py").read())
    
elif page == "User Management":
    exec(open("src/dashboard/user_management_dashboard.py").read())

elif page == "Performance":
    exec(open("src/dashboard/performance_dashboard.py").read())