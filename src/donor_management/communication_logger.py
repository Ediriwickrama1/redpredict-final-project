from datetime import datetime
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from database.mysql_connection import get_connection


def log_communication(donor_id, communication_type="Reminder", communication_status="Pending", notes=""):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO donor_communications
        (donor_id, communication_date, communication_type, communication_status, notes)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        donor_id,
        datetime.now(),
        communication_type,
        communication_status,
        notes
    ))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"Communication logged for donor {donor_id}")


if __name__ == "__main__":
    log_communication("D0001")