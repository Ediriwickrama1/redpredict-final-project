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

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.sort_values(["blood_bank", "blood_type", "date"]).reset_index(drop=True)
    return df


def safe_float(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return None


def compute_accuracy_from_mape(mape):
    mape = safe_float(mape)
    if mape is None:
        return None
    return max(0.0, 100.0 - mape)


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

    try:
        train_size = int(len(series) * 0.8)
        train = series.iloc[:train_size]
        test = series.iloc[train_size:]

        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))

        test = pd.to_numeric(test, errors="coerce")
        forecast = pd.to_numeric(forecast, errors="coerce")

        valid_mask = test.notna() & forecast.notna()
        test = test[valid_mask]
        forecast = forecast[valid_mask]

        if len(test) == 0:
            raise ValueError("ARIMA test/forecast contains no valid numeric pairs.")

        rmse = np.sqrt(mean_squared_error(test, forecast))
        mae = mean_absolute_error(test, forecast)

        test_safe = test.replace(0, np.nan)
        mape = (np.abs((test - forecast) / test_safe)).mean() * 100

        return {
            "Model": "ARIMA",
            "RMSE": safe_float(rmse),
            "MAE": safe_float(mae),
            "MAPE": safe_float(mape),
            "Accuracy": compute_accuracy_from_mape(mape)
        }

    except Exception as e:
        print(f"ARIMA metric generation failed: {e}")
        return {
            "Model": "ARIMA",
            "RMSE": None,
            "MAE": None,
            "MAPE": None,
            "Accuracy": None
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

        if not isinstance(result, (tuple, list)) or len(result) < 4:
            raise ValueError("Unexpected LSTM return format.")

        _, predictions, actual, metrics_or_rmse = result

        predictions = pd.to_numeric(pd.Series(np.array(predictions).flatten()), errors="coerce")
        actual = pd.to_numeric(pd.Series(np.array(actual).flatten()), errors="coerce")

        valid_mask = predictions.notna() & actual.notna()
        predictions = predictions[valid_mask]
        actual = actual[valid_mask]

        if len(actual) == 0:
            raise ValueError("LSTM actual/prediction contains no valid numeric pairs.")

        if isinstance(metrics_or_rmse, dict):
            rmse = safe_float(metrics_or_rmse.get("RMSE"))
            mae = safe_float(metrics_or_rmse.get("MAE"))
            mape = safe_float(metrics_or_rmse.get("MAPE"))

            if rmse is None:
                rmse = np.sqrt(mean_squared_error(actual, predictions))
            if mae is None:
                mae = mean_absolute_error(actual, predictions)
            if mape is None:
                actual_safe = actual.replace(0, np.nan)
                mape = (np.abs((actual - predictions) / actual_safe)).mean() * 100
        else:
            rmse = safe_float(metrics_or_rmse)
            if rmse is None:
                rmse = np.sqrt(mean_squared_error(actual, predictions))
            mae = mean_absolute_error(actual, predictions)
            actual_safe = actual.replace(0, np.nan)
            mape = (np.abs((actual - predictions) / actual_safe)).mean() * 100

        return {
            "Model": "LSTM",
            "RMSE": safe_float(rmse),
            "MAE": safe_float(mae),
            "MAPE": safe_float(mape),
            "Accuracy": compute_accuracy_from_mape(mape)
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
    arima_rmse = safe_float(arima_metrics.get("RMSE"))
    lstm_rmse = safe_float(lstm_metrics.get("RMSE"))

    if arima_rmse is None and lstm_rmse is None:
        return "ARIMA"
    if arima_rmse is None:
        return "LSTM"
    if lstm_rmse is None:
        return "ARIMA"

    return "LSTM" if lstm_rmse < arima_rmse else "ARIMA"


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