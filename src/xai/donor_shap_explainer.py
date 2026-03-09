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

    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))

    # Convert boolean columns to int
    X["frequent_donor"] = X["frequent_donor"].astype(int)
    X["rare_blood_type"] = X["rare_blood_type"].astype(int)

    # Force all columns to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop rows with any remaining NaN values
    valid_index = X.dropna().index
    X = X.loc[valid_index].copy()
    y = y.loc[valid_index].copy()

    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_train, X_test


def generate_global_shap(model, X_train):
    sample_X = X_train.sample(n=min(300, len(X_train)), random_state=42).copy()

    # Ensure pure numeric dtypes
    sample_X = sample_X.astype(float)

    explainer = shap.Explainer(model, sample_X)
    shap_values = explainer(sample_X)

    # Bar plot
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "donor_feature_importance.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Beeswarm/summary plot
    plt.figure()
    shap.summary_plot(
        shap_values.values.astype(float),
        sample_X.astype(float),
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "donor_feature_summary.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    return explainer


def generate_local_shap(explainer, X_test):
    one_row = X_test.iloc[[0]].copy().astype(float)
    shap_values_local = explainer(one_row)

    plt.figure()
    shap.plots.waterfall(shap_values_local[0], show=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "donor_local_explanation.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    return one_row


def main():
    print("Loading donor data...")
    df = load_data()

    print("Preparing donor features...")
    X, y = prepare_features(df)

    print("Feature dtypes:")
    print(X.dtypes)

    print("Training donor explainer model...")
    model, X_train, X_test = train_model(X, y)

    print("Generating global SHAP explanation...")
    explainer = generate_global_shap(model, X_train)

    print("Generating local SHAP explanation...")
    one_row = generate_local_shap(explainer, X_test)

    print("Saved:")
    print("-", os.path.join(OUTPUT_DIR, "donor_feature_importance.png"))
    print("-", os.path.join(OUTPUT_DIR, "donor_feature_summary.png"))
    print("-", os.path.join(OUTPUT_DIR, "donor_local_explanation.png"))

    print("\nExample donor row used for local explanation:")
    print(one_row)


if __name__ == "__main__":
    main()