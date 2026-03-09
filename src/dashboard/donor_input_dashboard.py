import streamlit as st
import mysql.connector
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from database.mysql_connection import get_connection

st.title("Add New Donor")

st.write("Use this form to add a new donor to the donor management database.")

with st.form("donor_form"):
    donor_id = st.text_input("Donor ID")
    name = st.text_input("Name")
    hospital = st.text_input("Hospital")
    blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    last_donation_date = st.date_input("Last Donation Date")
    total_donations = st.number_input("Total Donations", min_value=0, step=1)
    contact = st.text_input("Contact")
    days_since_last = st.number_input("Days Since Last Donation", min_value=0, step=1)
    status = st.selectbox("Status", ["active", "eligible", "available", "inactive"])
    age_group = st.selectbox("Age Group", ["18-25", "26-35", "36-45", "46-55", "56+"])
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    will_return = st.selectbox("Will Return", [0, 1])

    submitted = st.form_submit_button("Add Donor")

if submitted:
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO donors (
                donor_id, name, hospital, blood_type, last_donation_date,
                total_donations, contact, days_since_last, status,
                age_group, gender, will_return
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            donor_id,
            name,
            hospital,
            blood_type,
            str(last_donation_date),
            int(total_donations),
            contact,
            int(days_since_last),
            status,
            age_group,
            gender,
            int(will_return)
        ))

        conn.commit()
        cursor.close()
        conn.close()

        st.success("Donor added successfully to MySQL database.")

    except mysql.connector.Error as e:
        st.error(f"Database error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")