import streamlit as st
import pandas as pd

st.title("Blood Donor Reminder Dashboard")
st.write("This dashboard shows donors due for reminders based on the 4-month donation rule.")

data_path = "outputs/reminder_list.csv"

try:
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()
except Exception as e:
    st.error(f"Reminder list not found or could not be loaded: {e}")
    st.stop()

st.subheader("Total Donors Due for Reminder")
st.metric("Reminder Count", len(df))

st.sidebar.header("Filters")

blood_type_options = ["All"] + sorted(df["blood_type"].dropna().unique().tolist()) if "blood_type" in df.columns else ["All"]
hospital_options = ["All"] + sorted(df["hospital"].dropna().unique().tolist()) if "hospital" in df.columns else ["All"]

blood_type_filter = st.sidebar.selectbox("Filter by Blood Type", blood_type_options)
hospital_filter = st.sidebar.selectbox("Filter by Hospital", hospital_options)

filtered_df = df.copy()

if "blood_type" in filtered_df.columns and blood_type_filter != "All":
    filtered_df = filtered_df[filtered_df["blood_type"] == blood_type_filter]

if "hospital" in filtered_df.columns and hospital_filter != "All":
    filtered_df = filtered_df[filtered_df["hospital"] == hospital_filter]

st.subheader("Reminder List")
st.dataframe(filtered_df)

st.subheader("Top Priority Donors")

if "reminder_priority_score" in filtered_df.columns:
    top_donors = filtered_df.sort_values("reminder_priority_score", ascending=False).head(10)
    st.dataframe(top_donors)
else:
    st.warning("Column 'reminder_priority_score' not found. Run reminder_engine.py again to regenerate the reminder list.")

st.download_button(
    label="Download Reminder List",
    data=filtered_df.to_csv(index=False),
    file_name="reminder_list.csv",
    mime="text/csv"
)