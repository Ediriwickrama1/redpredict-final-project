import pandas as pd

PREDICTION_PATH = "outputs/donor_predictions.csv"
ELIGIBLE_PATH = "outputs/eligible_donors.csv"
OUTPUT_PATH = "outputs/priority_donors.csv"


def load_data():
    predictions_df = pd.read_csv(PREDICTION_PATH)
    eligible_df = pd.read_csv(ELIGIBLE_PATH)
    return predictions_df, eligible_df


def rank_donors(predictions_df, eligible_df):
    merged_df = pd.merge(
        eligible_df,
        predictions_df[["Donor_ID", "Return_Probability"]],
        on="Donor_ID",
        how="inner"
    )

    merged_df["Priority_Score"] = (
        0.60 * merged_df["Return_Probability"] +
        0.20 * merged_df["Rare_Blood_Type"].astype(int) +
        0.20 * merged_df["Frequent_Donor"].astype(int)
    )

    ranked_df = merged_df.sort_values("Priority_Score", ascending=False)

    return ranked_df


def save_output(df):
    df.to_csv(OUTPUT_PATH, index=False)
    print("Priority donor list saved to:", OUTPUT_PATH)
    print("Priority donor count:", len(df))


def main():
    predictions_df, eligible_df = load_data()
    ranked_df = rank_donors(predictions_df, eligible_df)
    save_output(ranked_df)
    print("\nPriority donor preview:")
    print(ranked_df[["Donor_ID", "Blood_Type", "Return_Probability", "Priority_Score"]].head())


if __name__ == "__main__":
    main()