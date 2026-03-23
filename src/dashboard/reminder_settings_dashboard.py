import streamlit as st
import json
import os

SETTINGS_PATH = "outputs/reminder_settings.json"

st.title("Reminder Settings")

default_settings = {
    "reminder_interval_months": 4,
    "minimum_days_since_last": 120
}

if os.path.exists(SETTINGS_PATH):
    with open(SETTINGS_PATH, "r") as f:
        settings = json.load(f)
else:
    settings = default_settings

st.write("Customize reminder behavior for donor notifications.")

interval = st.slider(
    "Reminder Interval (months)",
    3, 6,
    settings.get("reminder_interval_months", 4)
)

min_days = st.slider(
    "Minimum Days Since Last Donation",
    90, 180,
    settings.get("minimum_days_since_last", 120)
)

if st.button("Save Settings"):
    new_settings = {
        "reminder_interval_months": interval,
        "minimum_days_since_last": min_days
    }

    with open(SETTINGS_PATH, "w") as f:
        json.dump(new_settings, f, indent=2)

    st.success("Reminder settings saved successfully.")