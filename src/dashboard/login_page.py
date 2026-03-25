import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from auth.auth import login_user, init_auth_state

init_auth_state()

st.title("RedPredict Login")
st.write("Please log in to access the AI-Driven Blood Supply Management System.")

with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.form_submit_button("Login")

if login_btn:
    success, role = login_user(username, password)

    if success:
        st.session_state["logged_in"] = True
        st.session_state["username"] = username
        st.session_state["role"] = role
        st.success(f"Login successful. Welcome, {role}.")
        st.rerun()
    else:
        st.error("Invalid username or password.")