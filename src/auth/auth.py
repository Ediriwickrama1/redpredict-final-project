import streamlit as st

USERS = {
    "manager1": {
        "password": "manager123",
        "role": "Blood Bank Manager"
    },
    "coordinator1": {
        "password": "coord123",
        "role": "Donor Coordinator"
    },
    "hospital1": {
        "password": "hospital123",
        "role": "Hospital Staff"
    }
}

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
    if username in USERS and USERS[username]["password"] == password:
        return True, USERS[username]["role"]
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