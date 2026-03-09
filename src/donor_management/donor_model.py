import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/processed_donors.csv"
OUTPUT_PATH = "outputs/donor_predictions.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def prepare_features(df):
    features = [
        "blood_type",
        "hospital",
        "total_donations",
        "days_since_last",
        "status",
        "age_group",
        "gender",
        "frequent_donor",
        "rare_blood_type"
    ]

    X = df[features].copy()
    y = df["will_return"].copy()

    categorical_cols = [
        "blood_type",
        "hospital",
        "status",
        "age_group",
        "gender"
    ]

    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])

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
    output_df["return_probability"] = probabilities

    output_df.to_csv(OUTPUT_PATH, index=False)
    print("\nPrediction output saved to:", OUTPUT_PATH)

    return output_df


def main():
    df = load_data()
    print("Loaded processed donors shape:", df.shape)

    X, y, original_df = prepare_features(df)
    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)

    if len(X) == 0:
        print("No data available for training. Check processed_donors.csv and preprocessing step.")
        return

    model = train_model(X, y)
    prediction_df = generate_prediction_output(model, X, original_df)

    print("\nPrediction preview:")
    print(prediction_df[["donor_id", "blood_type", "return_probability"]].head())


if __name__ == "__main__":
    main()