import pandas as pd

DATA_PATH = "data/processed_donors.csv"
OUTPUT_PATH = "outputs/eligible_donors.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def filter_eligible_donors(df):
    eligible_df = df[
        (df["Eligible_By_Time"] == True) &
        (df["Active_Status"] == True) &
        (df["Has_Valid_Contact"] == True)
    ].copy()

    return eligible_df


def save_output(df):
    df.to_csv(OUTPUT_PATH, index=False)
    print("Eligible donors saved to:", OUTPUT_PATH)
    print("Eligible donor count:", len(df))


def main():
    df = load_data()
    eligible_df = filter_eligible_donors(df)
    save_output(eligible_df)
    print("\nEligible donor preview:")
    print(eligible_df.head())


if __name__ == "__main__":
    main()