import pandas as pd
from database.mysql_connection import get_connection


def load_donors_from_mysql():
    conn = get_connection()
    query = "SELECT * FROM donors"
    df = pd.read_sql(query, conn)
    conn.close()
    return df