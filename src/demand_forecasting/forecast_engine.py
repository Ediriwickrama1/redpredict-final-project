import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__))

from arima_model import load_data, prepare_series, train_arima
from lstm_model import prepare_series as prepare_lstm_series, train_lstm

OUTPUT_FOLDER = "outputs"


def select_best_model(arima_metrics, lstm_metrics):
    if lstm_metrics["RMSE"] < arima_metrics["RMSE"]:
        return "LSTM"
    return "ARIMA"


def save_forecast(blood_bank, blood_type, forecast_values, model_used, conf_int=None):
    safe_bank = blood_bank.replace("/", "-").replace(" ", "_")
    safe_type = blood_type.replace("+", "pos").replace("-", "neg")

    file_name = f"{safe_bank}_{safe_type}_forecast.csv"
    file_path = os.path.join(OUTPUT_FOLDER, file_name)

    if hasattr(forecast_values, "flatten"):
        forecast_values = forecast_values.flatten()

    forecast_df = pd.DataFrame({
        "Forecast_Units": forecast_values
    })

    if conf_int is not None and len(conf_int) == len(forecast_df):
        forecast_df["Lower_Bound"] = conf_int.iloc[:, 0].values
        forecast_df["Upper_Bound"] = conf_int.iloc[:, 1].values
    else:
        forecast_df["Lower_Bound"] = None
        forecast_df["Upper_Bound"] = None

    forecast_df["Model_Used"] = model_used
    forecast_df["Blood_Bank"] = blood_bank
    forecast_df["Blood_Type"] = blood_type

    forecast_df.to_csv(file_path, index=False)
    print(f"\nForecast saved to: {file_path}")


def generate_forecast(blood_bank, blood_type):
    df = load_data()

    # ARIMA
    arima_series = prepare_series(df, blood_bank, blood_type)

    if len(arima_series) < 10:
        print("Not enough data for forecasting with this blood bank and blood type.")
        return

    arima_forecast, arima_conf_int, arima_metrics = train_arima(arima_series)

    # LSTM
    lstm_series_df = prepare_lstm_series(df, blood_bank, blood_type)

    if len(lstm_series_df) < 20:
        print("Not enough data for LSTM. Using ARIMA.")
        best_model = "ARIMA"
        best_forecast = arima_forecast
        best_conf_int = arima_conf_int
        lstm_metrics = {"RMSE": None, "MAE": None, "MAPE": None}
    else:
        _, lstm_predictions, _, lstm_metrics = train_lstm(lstm_series_df)

        best_model = select_best_model(arima_metrics, lstm_metrics)

        if best_model == "LSTM":
            best_forecast = lstm_predictions
            best_conf_int = None
        else:
            best_forecast = arima_forecast
            best_conf_int = arima_conf_int

    print("\nModel Comparison")
    print("----------------")
    print("Blood Bank:", blood_bank)
    print("Blood Type:", blood_type)
    print("ARIMA Metrics:", arima_metrics)
    print("LSTM Metrics:", lstm_metrics)
    print("Best Model:", best_model)

    save_forecast(blood_bank, blood_type, best_forecast, best_model, best_conf_int)


def main():
    blood_bank = input("Enter Blood Bank: ").strip()
    blood_type = input("Enter Blood Type: ").strip()

    generate_forecast(blood_bank, blood_type)


if __name__ == "__main__":
    main()