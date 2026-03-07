import pandas as pd
from datetime import datetime

DATA_PATH = "data/processed_donors.csv"
OUTPUT_PATH = "outputs/reminder_list.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Last_Donation_Date"] = pd.to_datetime(df["Last_Donation_Date"])
    df["Reminder_Due_Date"] = pd.to_datetime(df["Reminder_Due_Date"])
    return df


def generate_reminder_list(df):
    today = pd.Timestamp(datetime.today().date())

    reminder_df = df[
        (df["Reminder_Due_Date"] <= today) &
        (df["Eligible_By_Time"] == True) &
        (df["Active_Status"] == True) &
        (df["Has_Valid_Contact"] == True)
    ].copy()

    reminder_df["Days_Overdue"] = (today - reminder_df["Reminder_Due_Date"]).dt.days

    reminder_df = reminder_df.sort_values(
        by=["Rare_Blood_Type", "Frequent_Donor", "Days_Overdue"],
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
            ["Donor_ID", "Blood_Type", "Last_Donation_Date", "Reminder_Due_Date", "Days_Overdue"]
        ].head()
    )


if __name__ == "__main__":
    main()