import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

BASE_XAI_DIR = "outputs/xai"
os.makedirs(BASE_XAI_DIR, exist_ok=True)


def normalize_demand_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    required_cols = [
        "cluster", "blood_bank", "blood_type", "rcc_demand_units",
        "is_weekend", "is_public_holiday", "is_dengue_peak", "is_external_disaster", "date"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required demand column: {col}")

    df = df.sort_values(["cluster", "blood_bank", "blood_type", "date"]).reset_index(drop=True)

    group_cols = ["blood_bank", "blood_type"]
    grouper = df.groupby(group_cols)["rcc_demand_units"]

    df["lag_1_day"] = grouper.shift(1)
    df["lag_7_day"] = grouper.shift(7)
    df["rolling_mean_7"] = grouper.transform(lambda x: x.rolling(window=7, min_periods=1).mean())

    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter

    df = df.fillna(0)
    return df


def get_same_cluster_transfer_recommendations(df, selected_cluster, selected_bank, blood_type, top_n=3):
    filtered = df[
        (df["cluster"] == selected_cluster) &
        (df["blood_bank"] != selected_bank) &
        (df["blood_type"] == blood_type)
    ].copy()

    if filtered.empty:
        return []

    recent = (
        filtered.sort_values("date")
        .groupby("blood_bank")
        .tail(7)
        .groupby("blood_bank")["rcc_demand_units"]
        .mean()
        .sort_values()
    )

    return recent.head(top_n).index.tolist()


def generate_narrative_explanation(bank_df, predicted_demand, current_stock, best_model):
    latest = bank_df.iloc[-1]
    recent_actual = float(latest["rcc_demand_units"])
    weekly_avg = float(latest["rolling_mean_7"])

    reasons = []

    if recent_actual > weekly_avg * 1.05:
        reasons.append(
            f"Recent demand is above the 7-day average ({recent_actual:.0f} vs {weekly_avg:.0f}), which pushes the forecast upward."
        )
    elif recent_actual < weekly_avg * 0.95:
        reasons.append(
            f"Recent demand is below the 7-day average ({recent_actual:.0f} vs {weekly_avg:.0f}), which reduces the forecast."
        )
    else:
        reasons.append(
            f"Recent demand is close to the weekly average ({recent_actual:.0f} vs {weekly_avg:.0f}), indicating a stable pattern."
        )

    if int(latest["is_public_holiday"]) == 1:
        reasons.append("A public holiday flag is present, which can influence routine demand patterns.")
    if int(latest["is_weekend"]) == 1:
        reasons.append("Weekend conditions are present, which may reduce routine blood usage.")
    if int(latest["is_dengue_peak"]) == 1:
        reasons.append("A dengue peak indicator is active, which can increase blood demand.")
    if int(latest["is_external_disaster"]) == 1:
        reasons.append("An external disaster flag is active, which can sharply increase emergency demand.")

    gap = predicted_demand - current_stock
    if gap > 0:
        reasons.append(f"The predicted demand exceeds current stock by {gap} units, indicating a shortage risk.")
    else:
        reasons.append(f"The current stock is sufficient for the predicted demand with a margin of {abs(gap)} units.")

    reasons.append(f"The selected forecasting model for this case is {best_model}.")

    return reasons


def generate_runtime_xai(bank_df):
    feature_cols = [
        "rolling_mean_7",
        "lag_1_day",
        "lag_7_day",
        "month",
        "day_of_week",
        "is_weekend",
        "is_public_holiday",
        "is_dengue_peak",
        "is_external_disaster"
    ]

    usable = bank_df.copy()

    if len(usable) < 10:
        return None, None

    X = usable[feature_cols].copy()
    y = usable["rcc_demand_units"].copy()

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)

    sample_X = X.tail(min(50, len(X)))
    shap_values = explainer.shap_values(sample_X)

    global_bar_path = os.path.join(BASE_XAI_DIR, "runtime_demand_global_bar.png")
    local_bar_path = os.path.join(BASE_XAI_DIR, "runtime_demand_local_bar.png")

    plt.figure()
    shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(global_bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    one_row = X.tail(1)
    one_shap = explainer.shap_values(one_row)

    local_df = pd.DataFrame({
        "Feature": one_row.columns,
        "Impact": one_shap[0]
    }).sort_values("Impact", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.barh(local_df["Feature"], local_df["Impact"])
    plt.gca().invert_yaxis()
    plt.xlabel("Local impact on prediction")
    plt.title("Local Factor Impact for Latest Forecast Context")
    plt.tight_layout()
    plt.savefig(local_bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    return global_bar_path, local_bar_path