import streamlit as st
from PIL import Image
import os

st.title("Explainable AI Dashboard")

st.subheader("Donor Prediction Explanation")
if os.path.exists("outputs/xai/donor_feature_importance.png"):
    st.image("outputs/xai/donor_feature_importance.png", caption="Donor Global Feature Importance")
if os.path.exists("outputs/xai/donor_local_explanation.png"):
    st.image("outputs/xai/donor_local_explanation.png", caption="Donor Local Explanation")

st.subheader("Demand Forecast Explanation")
if os.path.exists("outputs/xai/demand_feature_importance.png"):
    st.image("outputs/xai/demand_feature_importance.png", caption="Demand Global Feature Importance")
if os.path.exists("outputs/xai/demand_local_explanation.png"):
    st.image("outputs/xai/demand_local_explanation.png", caption="Demand Local Explanation")