import os
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

sys.path.append(os.path.dirname(__file__))

from arima_model import load_data as arima_load_data
from lstm_model import prepare_series as prepare_lstm_series, train_lstm

OUTPUT_PATH = "outputs/forecast_metrics.csv"


def normalize_df(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["blood_bank", "blood_type", "date"]).reset_index(drop=True)
    return df


def compute_arima_metrics(df, blood_bank, blood_type):
    filtered = df[
        (df["blood_bank"] == blood_bank) &
        (df["blood_type"] == blood_type)
    ].copy()

    series = filtered.sort_values("date")["rcc_demand_units"]

    if len(series) < 20:
        return {
            "Model": "ARIMA",
            "RMSE": None,
            "MAE": None,
            "MAPE": None,
            "Accuracy": None
        }

    train_size = int(len(series) * 0.8)
    train = series.iloc[:train_size]
    test = series.iloc[train_size:]

    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))

    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)

    test_safe = test.replace(0, np.nan)
    mape = (np.abs((test - forecast) / test_safe)).mean() * 100
    accuracy = max(0, 100 - mape) if pd.notna(mape) else None

    return {
        "Model": "ARIMA",
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "Accuracy": float(accuracy)
    }


def compute_lstm_metrics(raw_df, blood_bank, blood_type):
    try:
        lstm_df = prepare_lstm_series(raw_df, blood_bank, blood_type)

        if len(lstm_df) < 20:
            return {
                "Model": "LSTM",
                "RMSE": None,
                "MAE": None,
                "MAPE": None,
                "Accuracy": None
            }

        result = train_lstm(lstm_df)

        # expected patterns:
        # (model, predictions, actual, metrics_dict)
        # (model, predictions, actual, rmse_scalar)
        if len(result) >= 4:
            _, predictions, actual, metrics_or_rmse = result

            predictions = np.array(predictions).flatten()
            actual = np.array(actual).flatten()

            if isinstance(metrics_or_rmse, dict):
                rmse = metrics_or_rmse.get("RMSE")
                mae = metrics_or_rmse.get("MAE")
                mape = metrics_or_rmse.get("MAPE")
            else:
                rmse = float(metrics_or_rmse)
                mae = float(mean_absolute_error(actual, predictions))
                actual_safe = np.where(actual == 0, np.nan, actual)
                mape = float(np.nanmean(np.abs((actual - predictions) / actual_safe)) * 100)

            accuracy = max(0, 100 - mape) if pd.notna(mape) else None

            return {
                "Model": "LSTM",
                "RMSE": float(rmse) if rmse is not None else None,
                "MAE": float(mae) if mae is not None else None,
                "MAPE": float(mape) if mape is not None else None,
                "Accuracy": float(accuracy) if accuracy is not None else None
            }

    except Exception as e:
        print(f"LSTM metric generation failed: {e}")

    return {
        "Model": "LSTM",
        "RMSE": None,
        "MAE": None,
        "MAPE": None,
        "Accuracy": None
    }


def select_best_model(arima_metrics, lstm_metrics):
    if arima_metrics["RMSE"] is None and lstm_metrics["RMSE"] is None:
        return "Unavailable"
    if arima_metrics["RMSE"] is None:
        return "LSTM"
    if lstm_metrics["RMSE"] is None:
        return "ARIMA"
    return "LSTM" if lstm_metrics["RMSE"] < arima_metrics["RMSE"] else "ARIMA"


def run_model_comparison(blood_bank, blood_type, save=True):
    raw_df = arima_load_data()
    normalized_df = normalize_df(raw_df)

    arima_metrics = compute_arima_metrics(normalized_df, blood_bank, blood_type)
    lstm_metrics = compute_lstm_metrics(raw_df, blood_bank, blood_type)

    best_model = select_best_model(arima_metrics, lstm_metrics)

    result_df = pd.DataFrame([
        {
            "Blood_Bank": blood_bank,
            "Blood_Type": blood_type,
            "Model": arima_metrics["Model"],
            "RMSE": arima_metrics["RMSE"],
            "MAE": arima_metrics["MAE"],
            "MAPE": arima_metrics["MAPE"],
            "Accuracy": arima_metrics["Accuracy"],
            "Best_Model": best_model
        },
        {
            "Blood_Bank": blood_bank,
            "Blood_Type": blood_type,
            "Model": lstm_metrics["Model"],
            "RMSE": lstm_metrics["RMSE"],
            "MAE": lstm_metrics["MAE"],
            "MAPE": lstm_metrics["MAPE"],
            "Accuracy": lstm_metrics["Accuracy"],
            "Best_Model": best_model
        }
    ])

    if save:
        result_df.to_csv(OUTPUT_PATH, index=False)
        print("Forecast metrics saved to:", OUTPUT_PATH)

    return result_df, best_model


if __name__ == "__main__":
    df, best = run_model_comparison("Accident Service", "O+")
    print(df)
    print("Best Model:", best)