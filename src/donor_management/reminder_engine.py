import pandas as pd
from datetime import datetime

DATA_PATH = "data/processed_donors.csv"
OUTPUT_PATH = "outputs/reminder_list.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    # make all column names lowercase
    df.columns = df.columns.str.lower()

    df["last_donation_date"] = pd.to_datetime(df["last_donation_date"])
    df["reminder_due_date"] = pd.to_datetime(df["reminder_due_date"])

    print("Loaded columns:", list(df.columns))

    return df


def generate_reminder_list(df):
    today = pd.Timestamp(datetime.today().date())

    reminder_df = df[
        (df["reminder_due_date"] <= today) &
        (df["eligible_by_time"] == True) &
        (df["active_status"] == True) &
        (df["has_valid_contact"] == True)
    ].copy()

    reminder_df["days_overdue"] = (today - reminder_df["reminder_due_date"]).dt.days

    reminder_df = reminder_df.sort_values(
        by=["rare_blood_type", "frequent_donor", "days_overdue"],
        ascending=[False, False, False]
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
                "blood_type",
                "last_donation_date",
                "reminder_due_date",
                "days_overdue",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()