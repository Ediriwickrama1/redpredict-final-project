import pandas as pd

# Example values (you can replace later)
arima_rmse = 6.73
lstm_rmse = 1.98

data = {
    "Scenario": ["Before System (ARIMA / Manual)", "After System (LSTM + AI)"],
    "RMSE": [arima_rmse, lstm_rmse]
}

df = pd.DataFrame(data)
df.to_csv("outputs/system_comparison.csv", index=False)

print("System comparison saved.")