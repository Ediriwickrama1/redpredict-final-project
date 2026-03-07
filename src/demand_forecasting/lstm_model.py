import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math

DATA_PATH = "data/processed_demand.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def prepare_series(df, blood_bank, blood_type):
    filtered = df[
        (df["Blood_Bank"] == blood_bank) &
        (df["Blood_Type"] == blood_type)
    ].copy()

    series = filtered.groupby("Date")["RCC_Demand_Units"].sum().reset_index()
    return series


def create_sequences(data, seq_length=7):
    x = []
    y = []

    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    return np.array(x), np.array(y)


def train_lstm(series_df):
    values = series_df["RCC_Demand_Units"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)

    seq_length = 7
    x, y = create_sequences(scaled_values, seq_length)

    split_index = int(len(x) * 0.8)
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(seq_length, 1)))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train, y_train, epochs=20, batch_size=16, verbose=1)

    predictions = model.predict(x_test)

    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = math.sqrt(mean_squared_error(y_test_actual, predictions))
    print("LSTM RMSE:", rmse)

    return model, predictions, y_test_actual, rmse


def main():
    df = load_data()

    blood_bank = "Accident Service"
    blood_type = "O+"

    series_df = prepare_series(df, blood_bank, blood_type)

    if len(series_df) < 20:
        print("Not enough data for LSTM training.")
        return

    model, predictions, actual, rmse = train_lstm(series_df)

    print("Prediction sample:")
    print(predictions[:5])


if __name__ == "__main__":
    main()