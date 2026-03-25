import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from database.mysql_connection import get_connection

st.title("User Management")

st.write("Add new system users for dashboard access.")

with st.form("user_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    role = st.selectbox("Role", ["Blood Bank Manager", "Donor Coordinator", "Hospital Staff"])
    add_btn = st.form_submit_button("Add User")

if add_btn:
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO system_users (username, password, role)
            VALUES (%s, %s, %s)
        """, (username, password, role))

        conn.commit()
        cursor.close()
        conn.close()

        st.success("User added successfully.")

    except Exception as e:
        st.error(f"Could not add user: {e}")