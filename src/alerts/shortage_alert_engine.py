import os
import pandas as pd

FORECAST_FOLDER = "outputs"
PRIORITY_DONORS_PATH = "outputs/priority_donors.csv"
OUTPUT_PATH = "outputs/shortage_alerts.csv"


def load_priority_donors():
    if os.path.exists(PRIORITY_DONORS_PATH):
        return pd.read_csv(PRIORITY_DONORS_PATH)
    return pd.DataFrame()


def collect_forecast_files():
    files = []
    for f in os.listdir(FORECAST_FOLDER):
        if f.endswith("_forecast.csv"):
            files.append(os.path.join(FORECAST_FOLDER, f))
    return files


def detect_shortages(threshold=10):
    alerts = []

    donor_df = load_priority_donors()

    forecast_files = collect_forecast_files()

    if len(forecast_files) == 0:
        print("No forecast files found.")
        return

    for file_path in forecast_files:
        df = pd.read_csv(file_path)

        if "Forecast_Units" not in df.columns:
            continue

        avg_forecast = df["Forecast_Units"].mean()

        blood_bank = df["Blood_Bank"].iloc[0] if "Blood_Bank" in df.columns else "Unknown"
        blood_type = df["Blood_Type"].iloc[0] if "Blood_Type" in df.columns else "Unknown"

        if avg_forecast >= threshold:
            matched_donors = 0

            if not donor_df.empty and "blood_type" in donor_df.columns:
                matched = donor_df[donor_df["blood_type"] == blood_type]
                matched_donors = len(matched)

            alerts.append({
                "Blood_Bank": blood_bank,
                "Blood_Type": blood_type,
                "Average_Forecast_Units": round(avg_forecast, 2),
                "Shortage_Alert": "YES",
                "Matched_Priority_Donors": matched_donors
            })

    alert_df = pd.DataFrame(alerts)

    alert_df.to_csv(OUTPUT_PATH, index=False)

    print("\nShortage alerts saved to:", OUTPUT_PATH)
    print("Total alerts:", len(alert_df))


if __name__ == "__main__":
    detect_shortages()