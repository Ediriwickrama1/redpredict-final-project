import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


DATA_PATH = "data/processed_demand.csv"


def load_data():

    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    return df


def prepare_series(df, blood_bank, blood_type):

    filtered = df[
        (df["Blood_Bank"] == blood_bank) &
        (df["Blood_Type"] == blood_type)
    ]

    series = filtered.groupby("Date")["RCC_Demand_Units"].sum()

    return series


def train_arima(series):

    train_size = int(len(series) * 0.8)

    train = series[:train_size]
    test  = series[train_size:]

    model     = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()

    forecast_result = model_fit.get_forecast(steps=len(test))
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae  = mean_absolute_error(test, forecast)

    # avoid divide-by-zero in MAPE
    test_safe = test.replace(0, np.nan)
    mape = (np.abs((test - forecast) / test_safe)).mean() * 100

    print("ARIMA RMSE:", rmse)
    print("ARIMA MAE: ", mae)
    print("ARIMA MAPE:", mape)

    metrics = {
        "Model": "ARIMA",
        "RMSE":  rmse,
        "MAE":   mae,
        "MAPE":  mape
    }

    return forecast, conf_int, metrics


def main():

    df = load_data()

    blood_bank = "Accident Service"
    blood_type = "O+"

    series = prepare_series(df, blood_bank, blood_type)

    forecast, conf_int, metrics = train_arima(series)

    print("Forecast sample:")
    print(forecast.head())

    print("\nConfidence Intervals sample:")
    print(conf_int.head())

    print("\nMetrics:")
    print(metrics)


if __name__ == "__main__":
    main()