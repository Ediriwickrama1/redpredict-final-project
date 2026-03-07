import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__))

from arima_model import load_data, prepare_series, train_arima
from lstm_model import prepare_series as prepare_lstm_series, train_lstm


OUTPUT_FOLDER = "outputs"


def select_best_model(arima_rmse, lstm_rmse):
    if lstm_rmse < arima_rmse:
        return "LSTM"
    return "ARIMA"


def save_forecast(blood_bank, blood_type, forecast_values, model_used):
    safe_bank = blood_bank.replace("/", "-").replace(" ", "_")
    safe_type = blood_type.replace("+", "pos").replace("-", "neg")

    file_name = f"{safe_bank}_{safe_type}_forecast.csv"
    file_path = os.path.join(OUTPUT_FOLDER, file_name)

    if hasattr(forecast_values, "flatten"):
        forecast_values = forecast_values.flatten()

    forecast_df = pd.DataFrame({
        "Forecast_Units": forecast_values
    })
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

    arima_forecast, arima_rmse = train_arima(arima_series)

    # LSTM
    lstm_series_df = prepare_lstm_series(df, blood_bank, blood_type)

    if len(lstm_series_df) < 20:
        print("Not enough data for LSTM. Using ARIMA.")
        best_model = "ARIMA"
        best_forecast = arima_forecast
        lstm_rmse = None
    else:
        _, lstm_predictions, _, lstm_rmse = train_lstm(lstm_series_df)

        best_model = select_best_model(arima_rmse, lstm_rmse)

        if best_model == "LSTM":
            best_forecast = lstm_predictions
        else:
            best_forecast = arima_forecast

    print("\nModel Comparison")
    print("----------------")
    print("Blood Bank:", blood_bank)
    print("Blood Type:", blood_type)
    print("ARIMA RMSE:", arima_rmse)
    print("LSTM RMSE:", lstm_rmse)
    print("Best Model:", best_model)

    save_forecast(blood_bank, blood_type, best_forecast, best_model)


def main():
    blood_bank = input("Enter Blood Bank: ").strip()
    blood_type = input("Enter Blood Type: ").strip()

    generate_forecast(blood_bank, blood_type)


if __name__ == "__main__":
    main()