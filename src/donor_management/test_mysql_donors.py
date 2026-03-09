import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from database.mysql_connection import get_connection


def main():
    conn = get_connection()

    query = "SELECT * FROM donors"

    df = pd.read_sql(query, conn)

    print("Connected successfully to XAMPP MySQL")
    print("Number of donors:", df.shape[0])
    print(df.head())


if __name__ == "__main__":
    main()