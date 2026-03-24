import streamlit as st
import pandas as pd
import os

st.title("System Performance Dashboard")
st.write("This page shows measured execution times for key system tasks.")

path = "outputs/performance_log.csv"

if os.path.exists(path):
    df = pd.read_csv(path)

    st.metric("Logged Performance Events", len(df))
    st.dataframe(df)

    if "Task" in df.columns and "Runtime_Seconds" in df.columns:
        avg_df = df.groupby("Task", as_index=False)["Runtime_Seconds"].mean()
        st.subheader("Average Runtime by Task")
        st.dataframe(avg_df)
        st.bar_chart(avg_df.set_index("Task"))
else:
    st.warning("No performance log found. Run system modules first.")