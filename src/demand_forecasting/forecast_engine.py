import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))

from arima_model import load_data, prepare_series, train_arima
from lstm_model import prepare_series as prepare_lstm_series, train_lstm

OUTPUT_FOLDER = "outputs"


def select_best_model(arima_rmse, lstm_rmse):
    if lstm_rmse < arima_rmse:
        return "LSTM"
    return "ARIMA"


def generate_forecast(blood_bank, blood_type):

    df = load_data()

    # ARIMA
    series = prepare_series(df, blood_bank, blood_type)

    arima_forecast, arima_rmse = train_arima(series)

    # LSTM
    lstm_series_df = prepare_lstm_series(df, blood_bank, blood_type)

    if len(lstm_series_df) < 20:
        best_model = "ARIMA"
        forecast_values = arima_forecast

    else:
        _, lstm_predictions, _, lstm_rmse = train_lstm(lstm_series_df)

        best_model = select_best_model(arima_rmse, lstm_rmse)

        if best_model == "LSTM":
            forecast_values = lstm_predictions
        else:
            forecast_values = arima_forecast

    save_forecast(blood_bank, blood_type, forecast_values, best_model)


def save_forecast(blood_bank, blood_type, forecast_values, model):

    file_name = f"{blood_bank}_{blood_type}_forecast.csv"

    file_path = os.path.join(OUTPUT_FOLDER, file_name)

    df = pd.DataFrame({
        "Forecast_Units": forecast_values.flatten()
    })

    df["Model_Used"] = model

    df.to_csv(file_path, index=False)

    print("\nForecast saved to:", file_path)


def main():

    blood_bank = input("Enter Blood Bank: ")
    blood_type = input("Enter Blood Type: ")

    generate_forecast(blood_bank, blood_type)


if __name__ == "__main__":
    main()