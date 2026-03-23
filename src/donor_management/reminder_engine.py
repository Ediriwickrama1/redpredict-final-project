import pandas as pd
from datetime import datetime
import json
import os

DATA_PATH = "data/processed_donors.csv"
OUTPUT_PATH = "outputs/reminder_list.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower()

    df["last_donation_date"] = pd.to_datetime(df["last_donation_date"])
    df["reminder_due_date"] = pd.to_datetime(df["reminder_due_date"])

    print("Loaded columns:", list(df.columns))
    return df


def load_settings():
    path = "outputs/reminder_settings.json"

    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    return {
        "reminder_interval_months": 4,
        "minimum_days_since_last": 120
    }


def generate_reminder_list(df):
    settings = load_settings()
    interval_months = settings["reminder_interval_months"]
    min_days = settings["minimum_days_since_last"]

    today = pd.Timestamp(datetime.today().date())

    df["reminder_due_date"] = df["last_donation_date"] + pd.DateOffset(months=interval_months)

    reminder_df = df[
        (df["reminder_due_date"] <= today) &
        (df["eligible_by_time"] == True) &
        (df["active_status"] == True) &
        (df["has_valid_contact"] == True) &
        (df["days_since_last"] >= min_days)
    ].copy()

    reminder_df["days_overdue"] = (today - reminder_df["reminder_due_date"]).dt.days

    max_days_overdue = reminder_df["days_overdue"].max() if len(reminder_df) > 0 else 1
    if max_days_overdue == 0:
        max_days_overdue = 1

    reminder_df["reminder_priority_score"] = (
        0.40 * reminder_df["rare_blood_type"].astype(int) +
        0.30 * reminder_df["frequent_donor"].astype(int) +
        0.30 * (reminder_df["days_overdue"] / max_days_overdue)
    )

    reminder_df = reminder_df.sort_values(
        by=["reminder_priority_score", "days_overdue"],
        ascending=[False, False]
    )

    return reminder_df


def save_output(df):
    df.to_csv(OUTPUT_PATH, index=False)
    print("Reminder list saved to:", OUTPUT_PATH)
    print("Reminder donor count:", len(df))


def main():
    df = load_data()
    reminder_df = generate_reminder_list(df)
    save_output(reminder_df)

    print("\nReminder preview:")
    print(
        reminder_df[
            [
                "donor_id",
                "name",
                "blood_type",
                "hospital",
                "last_donation_date",
                "reminder_due_date",
                "days_since_last",
                "days_overdue",
                "rare_blood_type",
                "frequent_donor",
                "reminder_priority_score"
            ]
        ].head()
    )


if __name__ == "__main__":
    main()