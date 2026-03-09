import streamlit as st
import os

st.title("Explainable AI Dashboard")

st.write("This dashboard shows SHAP-based explanations for donor prediction and demand forecasting.")

st.subheader("Donor Prediction Explanations")

if os.path.exists("outputs/xai/donor_feature_importance.png"):
    st.image("outputs/xai/donor_feature_importance.png", caption="Donor Feature Importance")
else:
    st.warning("donor_feature_importance.png not found")

if os.path.exists("outputs/xai/donor_feature_summary.png"):
    st.image("outputs/xai/donor_feature_summary.png", caption="Donor SHAP Summary")
else:
    st.warning("donor_feature_summary.png not found")

if os.path.exists("outputs/xai/donor_local_explanation.png"):
    st.image("outputs/xai/donor_local_explanation.png", caption="Donor Local Explanation")
else:
    st.warning("donor_local_explanation.png not found")

st.subheader("Demand Forecast Explanations")

if os.path.exists("outputs/xai/demand_feature_importance.png"):
    st.image("outputs/xai/demand_feature_importance.png", caption="Demand SHAP Summary")
else:
    st.warning("demand_feature_importance.png not found")

if os.path.exists("outputs/xai/demand_feature_importance_bar.png"):
    st.image("outputs/xai/demand_feature_importance_bar.png", caption="Demand SHAP Bar Importance")
else:
    st.warning("demand_feature_importance_bar.png not found")

if os.path.exists("outputs/xai/demand_local_explanation.png"):
    st.image("outputs/xai/demand_local_explanation.png", caption="Demand Local Explanation")
else:
    st.warning("demand_local_explanation.png not found")