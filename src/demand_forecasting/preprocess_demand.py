import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data(df):

    # convert date column
    df['Date'] = pd.to_datetime(df['Date'])

    # sort by date
    df = df.sort_values('Date')

    # remove duplicates
    df = df.drop_duplicates()

    return df

def create_features(df):

    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek

    return df

def preprocess_pipeline(file_path):

    df = load_data(file_path)

    df = clean_data(df)

    df = create_features(df)

    return df


if __name__ == "__main__":

    file_path = "data/nbts_demand.csv"

    df = preprocess_pipeline(file_path)

    print(df.head())