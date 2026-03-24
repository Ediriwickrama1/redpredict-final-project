import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from database.mysql_connection import get_connection

st.title("Donor Communication Log")

st.write("This page shows recorded reminder and communication history for donors.")

try:
    conn = get_connection()
    query = "SELECT * FROM donor_communications ORDER BY communication_date DESC"
    df = pd.read_sql(query, conn)
    conn.close()

    st.metric("Total Communication Records", len(df))
    st.dataframe(df)

except Exception as e:
    st.error(f"Could not load communication log: {e}")