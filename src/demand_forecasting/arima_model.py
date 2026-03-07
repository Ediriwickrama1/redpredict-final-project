import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
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
    test = series[train_size:]

    model = ARIMA(train, order=(5,1,0))

    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(test))

    rmse = np.sqrt(mean_squared_error(test, forecast))

    print("ARIMA RMSE:", rmse)

    return forecast


def main():

    df = load_data()

    blood_bank = "Accident Service"
    blood_type = "O+"

    series = prepare_series(df, blood_bank, blood_type)

    forecast = train_arima(series)

    print("Forecast sample:")
    print(forecast.head())


if __name__ == "__main__":
    main()