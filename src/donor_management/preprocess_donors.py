import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from database.mysql_connection import get_connection

OUTPUT_PATH = "data/processed_donors.csv"


def load_data():
    print("Loading donor data from MySQL...")

    conn = get_connection()
    query = "SELECT * FROM donors"
    df = pd.read_sql(query, conn)
    conn.close()

    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))

    return df


def clean_data(df):
    print("Cleaning donor data...")

    df = df.drop_duplicates()
    df.columns = [col.strip() for col in df.columns]

    print("Columns after cleaning names:", list(df.columns))

    # Convert columns carefully
    df["last_donation_date"] = pd.to_datetime(df["last_donation_date"], errors="coerce")
    df["total_donations"] = pd.to_numeric(df["total_donations"], errors="coerce")
    df["days_since_last"] = pd.to_numeric(df["days_since_last"], errors="coerce")
    df["will_return"] = pd.to_numeric(df["will_return"], errors="coerce")

    string_cols = [
        "donor_id",
        "name",
        "hospital",
        "blood_type",
        "contact",
        "status",
        "age_group",
        "gender"
    ]

    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()

    print("\nMissing values before dropping:")
    print(df.isna().sum())

    # Only drop rows missing truly critical model fields
    df = df.dropna(subset=[
        "donor_id",
        "blood_type",
        "last_donation_date",
        "total_donations",
        "days_since_last",
        "will_return"
    ])

    print("Remaining rows after dropna:", len(df))

    return df


def create_features(df):
    print("Creating donor system features...")

    df["reminder_due_date"] = df["last_donation_date"] + pd.DateOffset(months=4)
    df["eligible_by_time"] = df["days_since_last"] >= 120
    df["frequent_donor"] = df["total_donations"] >= 5

    rare_types = ["O-", "A-", "B-", "AB-"]
    df["rare_blood_type"] = df["blood_type"].isin(rare_types)

    df["has_valid_contact"] = df["contact"].notna() & (df["contact"].astype(str).str.strip() != "")
    df["active_status"] = df["status"].str.lower().isin(["active", "eligible", "available", "yes"])

    return df


def save_processed_data(df):
    print("Saving processed donor dataset...")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Saved to:", OUTPUT_PATH)


def main():
    df = load_data()
    print("After loading:", df.shape)

    df = clean_data(df)
    print("After cleaning:", df.shape)

    df = create_features(df)
    print("After feature creation:", df.shape)

    save_processed_data(df)

    print("\nProcessed donor data preview:")
    print(df.head())


if __name__ == "__main__":
    main()