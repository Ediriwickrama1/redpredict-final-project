import streamlit as st
import pandas as pd

st.title("Blood Donor Reminder Dashboard")

st.write("This dashboard displays donors who are due for reminders based on the 4-month donation rule.")

# Load reminder list
data_path = "outputs/reminder_list.csv"

try:
    df = pd.read_csv(data_path)
except:
    st.error("Reminder list not found. Please run reminder_engine.py first.")
    st.stop()

st.subheader("Total Donors Due for Reminder")

st.metric("Reminder Count", len(df))

# Filters
st.sidebar.header("Filters")

blood_type_filter = st.sidebar.selectbox(
    "Filter by Blood Type",
    ["All"] + sorted(df["blood_type"].unique().tolist())
)

hospital_filter = st.sidebar.selectbox(
    "Filter by Hospital",
    ["All"] + sorted(df["hospital"].unique().tolist())
)

filtered_df = df.copy()

if blood_type_filter != "All":
    filtered_df = filtered_df[filtered_df["blood_type"] == blood_type_filter]

if hospital_filter != "All":
    filtered_df = filtered_df[filtered_df["hospital"] == hospital_filter]

st.subheader("Reminder List")

st.dataframe(filtered_df)

# Show top priority donors
st.subheader("Top Priority Donors")

top_donors = filtered_df.sort_values(
    "reminder_priority_score",
    ascending=False
).head(10)

st.dataframe(top_donors)

# Download button
st.download_button(
    label="Download Reminder List",
    data=filtered_df.to_csv(index=False),
    file_name="reminder_list.csv",
    mime="text/csv"
)