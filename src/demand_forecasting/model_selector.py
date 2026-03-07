import pandas as pd
from arima_model import load_data as load_arima_data, prepare_series as prepare_arima_series, train_arima
from lstm_model import load_data as load_lstm_data, prepare_series as prepare_lstm_series, train_lstm

def select_best_model(arima_rmse, lstm_rmse):
    if lstm_rmse < arima_rmse:
        return "LSTM"
    else:
        return "ARIMA"


def main():
    blood_bank = "Accident Service"
    blood_type = "O+"

    # ARIMA
    df_arima = load_arima_data()
    arima_series = prepare_arima_series(df_arima, blood_bank, blood_type)
    _, arima_rmse = train_arima(arima_series)

    # LSTM
    df_lstm = load_lstm_data()
    lstm_series_df = prepare_lstm_series(df_lstm, blood_bank, blood_type)

    if len(lstm_series_df) < 20:
        print("Not enough data for LSTM. Best model = ARIMA")
        return

    _, _, _, lstm_rmse = train_lstm(lstm_series_df)

    best_model = select_best_model(arima_rmse, lstm_rmse)

    print("\nModel Comparison Result")
    print("-----------------------")
    print("Blood Bank:", blood_bank)
    print("Blood Type:", blood_type)
    print("ARIMA RMSE:", arima_rmse)
    print("LSTM RMSE:", lstm_rmse)
    print("Best Model:", best_model)


if __name__ == "__main__":
    main()