import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from database.mysql_connection import get_connection


ROLE_PAGES = {
    "Blood Bank Manager": [
        "Home",
        "Add Donor",
        "Demand Forecast",
        "Reminder Monitoring",
        "Reminder Settings",
        "Communication Log",
        "Model Performance",
        "Explainable AI",
        "User Management",
        "Performance"
    ],
    "Donor Coordinator": [
        "Home",
        "Add Donor",
        "Reminder Monitoring",
        "Reminder Settings",
        "Communication Log"
    ],
    "Hospital Staff": [
        "Home",
        "Demand Forecast",
        "Model Performance",
        "Explainable AI"
    ]
}


def login_user(username, password):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT username, password, role
        FROM system_users
        WHERE username = %s
    """, (username,))

    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if user and user["password"] == password:
        return True, user["role"]

    return False, None


def logout_user():
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.session_state["role"] = None


def init_auth_state():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "role" not in st.session_state:
        st.session_state["role"] = None


def get_allowed_pages(role):
    return ROLE_PAGES.get(role, ["Home"])