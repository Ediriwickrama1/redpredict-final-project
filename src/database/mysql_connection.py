import os
import mysql.connector
import streamlit as st


def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


def get_connection():
    host = get_secret("DB_HOST", "localhost")
    port = int(get_secret("DB_PORT", "3306"))
    user = get_secret("DB_USER", "root")
    password = get_secret("DB_PASSWORD", "")
    database = get_secret("DB_NAME", "blood_donor_system")

    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )