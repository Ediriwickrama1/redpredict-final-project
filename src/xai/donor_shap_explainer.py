import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/processed_donors.csv"
OUTPUT_DIR = "outputs/xai"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower()
    return df


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

    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))
        encoders[col] = encoder

    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_train, X_test


def generate_global_shap(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "donor_feature_importance.png"), bbox_inches="tight")
    plt.close()

    return explainer, shap_values


def generate_local_shap(explainer, X_test):
    one_row = X_test.iloc[[0]]
    shap_values_local = explainer(one_row)

    plt.figure()
    shap.plots.waterfall(shap_values_local[0], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "donor_local_explanation.png"), bbox_inches="tight")
    plt.close()

    return one_row, shap_values_local


def main():
    print("Loading donor data...")
    df = load_data()

    print("Preparing features...")
    X, y = prepare_features(df)

    print("Training donor model for SHAP...")
    model, X_train, X_test = train_model(X, y)

    print("Generating global SHAP explanation...")
    explainer, shap_values = generate_global_shap(model, X_train)

    print("Generating local SHAP explanation...")
    one_row, shap_values_local = generate_local_shap(explainer, X_test)

    print("Saved:")
    print("-", os.path.join(OUTPUT_DIR, "donor_feature_importance.png"))
    print("-", os.path.join(OUTPUT_DIR, "donor_local_explanation.png"))

    print("\nExample donor row used for local explanation:")
    print(one_row)


if __name__ == "__main__":
    main()