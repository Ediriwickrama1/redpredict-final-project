import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
import matplotlib.pyplot as plt


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

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # --- Metrics ---
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_prob)

    print("Accuracy: ",  accuracy)
    print("Precision:", precision)
    print("Recall:   ", recall)
    print("F1 Score: ", f1)
    print("AUC:      ", auc)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --- Save metrics to CSV ---
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
        "Value":  [accuracy, precision, recall, f1, auc]
    })
    metrics_df.to_csv("outputs/donor_metrics.csv", index=False)
    print("Metrics saved to: outputs/donor_metrics.csv")

    # --- Save confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved to: outputs/confusion_matrix.png")

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