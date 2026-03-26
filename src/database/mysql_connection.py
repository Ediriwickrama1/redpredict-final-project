import os
import mysql.connector
import streamlit as st


def _get_secret(name: str, default: str | None = None):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


def get_connection():
    return mysql.connector.connect(
        host=_get_secret("DB_HOST"),
        port=int(_get_secret("DB_PORT", "3306")),
        user=_get_secret("DB_USER"),
        password=_get_secret("DB_PASSWORD"),
        database=_get_secret("DB_NAME"),
    )