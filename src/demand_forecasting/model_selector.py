import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from arima_model import load_data as load_arima_data, prepare_series as prepare_arima_series, train_arima
from lstm_model import load_data as load_lstm_data, prepare_series as prepare_lstm_series, train_lstm

OUTPUT_PATH = "outputs/forecast_metrics.csv"


def select_best_model(arima_metrics, lstm_metrics):
    if lstm_metrics["RMSE"] < arima_metrics["RMSE"]:
        return "LSTM"
    return "ARIMA"


def save_metrics(arima_metrics, lstm_metrics, blood_bank, blood_type, best_model, arima_conf_int=None):
    rows = [
        {
            "Blood_Bank":       blood_bank,
            "Blood_Type":       blood_type,
            "Model":            arima_metrics["Model"],
            "RMSE":             arima_metrics["RMSE"],
            "MAE":              arima_metrics["MAE"],
            "MAPE":             arima_metrics["MAPE"],
            "CI_Lower_First":   arima_conf_int.iloc[0, 0] if arima_conf_int is not None else None,
            "CI_Upper_First":   arima_conf_int.iloc[0, 1] if arima_conf_int is not None else None,
            "Best_Model":       best_model
        },
        {
            "Blood_Bank":       blood_bank,
            "Blood_Type":       blood_type,
            "Model":            lstm_metrics["Model"],
            "RMSE":             lstm_metrics["RMSE"],
            "MAE":              lstm_metrics["MAE"],
            "MAPE":             lstm_metrics["MAPE"],
            "CI_Lower_First":   None,
            "CI_Upper_First":   None,
            "Best_Model":       best_model
        }
    ]

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print("\nForecast metrics saved to:", OUTPUT_PATH)


def main():
    blood_bank = "Accident Service"
    blood_type = "O+"

    # ARIMA
    df_arima     = load_arima_data()
    arima_series = prepare_arima_series(df_arima, blood_bank, blood_type)
    _, arima_conf_int, arima_metrics = train_arima(arima_series)

    # LSTM
    df_lstm        = load_lstm_data()
    lstm_series_df = prepare_lstm_series(df_lstm, blood_bank, blood_type)

    if len(lstm_series_df) < 20:
        print("Not enough data for LSTM. Best model = ARIMA")
        best_model = "ARIMA"
        save_metrics(
            arima_metrics,
            {"Model": "LSTM", "RMSE": None, "MAE": None, "MAPE": None},
            blood_bank,
            blood_type,
            best_model,
            arima_conf_int=arima_conf_int
        )
        return

    _, _, _, lstm_metrics = train_lstm(lstm_series_df)

    best_model = select_best_model(arima_metrics, lstm_metrics)

    print("\nModel Comparison Result")
    print("-----------------------")
    print("Blood Bank:    ", blood_bank)
    print("Blood Type:    ", blood_type)
    print("ARIMA Metrics: ", arima_metrics)
    print("LSTM Metrics:  ", lstm_metrics)
    print("Best Model:    ", best_model)

    save_metrics(arima_metrics, lstm_metrics, blood_bank, blood_type, best_model, arima_conf_int=arima_conf_int)


if __name__ == "__main__":
    main()