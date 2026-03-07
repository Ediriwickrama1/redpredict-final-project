import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/processed_donors.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def prepare_features(df):

    features = [
        "Blood_Type",
        "Hospital",
        "Total_Donations",
        "Days_Since_Last",
        "Status",
        "Age_Group",
        "Gender",
        "Frequent_Donor",
        "Rare_Blood_Type"
    ]

    X = df[features]
    y = df["Will_Return"]

    # Encode categorical columns
    categorical_cols = [
        "Blood_Type",
        "Hospital",
        "Status",
        "Age_Group",
        "Gender"
    ]

    encoder = LabelEncoder()

    for col in categorical_cols:
        X[col] = encoder.fit_transform(X[col])

    return X, y


def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return model


def main():

    df = load_data()

    X, y = prepare_features(df)

    model = train_model(X, y)


if __name__ == "__main__":
    main()