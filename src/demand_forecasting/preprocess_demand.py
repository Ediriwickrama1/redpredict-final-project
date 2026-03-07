import pandas as pd
import os

DATA_PATH = "data/nbts_demand.csv"
OUTPUT_PATH = "data/processed_demand.csv"


def load_data():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    return df


def clean_data(df):

    print("Cleaning data...")

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.drop_duplicates()

    df = df.sort_values("Date")

    return df


def create_time_features(df):

    print("Creating time features...")

    df["day_of_week"] = df["Date"].dt.dayofweek

    df["week_of_year"] = df["Date"].dt.isocalendar().week

    df["quarter"] = df["Date"].dt.quarter

    return df


def save_processed_data(df):

    print("Saving processed dataset...")

    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved to:", OUTPUT_PATH)


def main():

    df = load_data()

    df = clean_data(df)

    df = create_time_features(df)

    save_processed_data(df)

    print(df.head())


if __name__ == "__main__":
    main()