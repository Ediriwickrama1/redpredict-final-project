import pandas as pd

DATA_PATH = "data/donor_management.csv"
OUTPUT_PATH = "data/processed_donors.csv"


def load_data():
    print("Loading donor dataset...")
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    return df


def clean_data(df):
    print("Cleaning donor data...")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    # Convert date column
    df["Last_Donation_Date"] = pd.to_datetime(df["Last_Donation_Date"], errors="coerce")

    # Convert numeric columns
    df["Total_Donations"] = pd.to_numeric(df["Total_Donations"], errors="coerce")
    df["Days_Since_Last"] = pd.to_numeric(df["Days_Since_Last"], errors="coerce")
    df["Will_Return"] = pd.to_numeric(df["Will_Return"], errors="coerce")

    # Clean string columns
    string_cols = ["Donor_ID", "Name", "Hospital", "Blood_Type", "Contact", "Status", "Age_Group", "Gender"]
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()

    # Drop rows with critical missing values
    required_cols = [
        "Donor_ID",
        "Hospital",
        "Blood_Type",
        "Last_Donation_Date",
        "Total_Donations",
        "Days_Since_Last",
        "Status",
        "Age_Group",
        "Gender",
        "Will_Return"
    ]
    df = df.dropna(subset=required_cols)

    return df


def create_features(df):
    print("Creating donor system features...")

    # Reminder due date = 4 months after last donation
    df["Reminder_Due_Date"] = df["Last_Donation_Date"] + pd.DateOffset(months=4)

    # Reminder eligibility based on 120 days
    df["Eligible_By_Time"] = df["Days_Since_Last"] >= 120

    # Frequent donor flag
    df["Frequent_Donor"] = df["Total_Donations"] >= 5

    # Rare blood type flag
    rare_types = ["O-", "A-", "B-", "AB-"]
    df["Rare_Blood_Type"] = df["Blood_Type"].isin(rare_types)

    # Valid contact flag
    df["Has_Valid_Contact"] = df["Contact"].notna() & (df["Contact"].astype(str).str.strip() != "")

    # Active donor status flag
    df["Active_Status"] = df["Status"].str.lower().isin(["active", "eligible", "available", "yes"])

    return df


def save_processed_data(df):
    print("Saving processed donor dataset...")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Saved to:", OUTPUT_PATH)


def main():
    df = load_data()
    df = clean_data(df)
    df = create_features(df)
    save_processed_data(df)
    print("\nProcessed donor data preview:")
    print(df.head())


if __name__ == "__main__":
    main()