import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/processed_donors.csv"
OUTPUT_PATH = "outputs/donor_predictions.csv"


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

    X = df[features].copy()
    y = df["Will_Return"].copy()

    categorical_cols = [
        "Blood_Type",
        "Hospital",
        "Status",
        "Age_Group",
        "Gender"
    ]

    encoders = {}

    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
        encoders[col] = encoder

    return X, y, df


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


def generate_prediction_output(model, X, original_df):
    probabilities = model.predict_proba(X)[:, 1]

    output_df = original_df.copy()
    output_df["Return_Probability"] = probabilities

    output_df.to_csv(OUTPUT_PATH, index=False)
    print("\nPrediction output saved to:", OUTPUT_PATH)

    return output_df


def main():
    df = load_data()
    X, y, original_df = prepare_features(df)
    model = train_model(X, y)
    prediction_df = generate_prediction_output(model, X, original_df)
    print("\nPrediction preview:")
    print(prediction_df[["Donor_ID", "Blood_Type", "Return_Probability"]].head())


if __name__ == "__main__":
    main()