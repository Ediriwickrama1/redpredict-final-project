import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/processed_demand.csv"
OUTPUT_DIR = "outputs/xai"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower()
    return df


def prepare_features(df):
    feature_cols = [
        "blood_bank",
        "blood_type",
        "month",
        "day",
        "day_of_week",
        "week_of_year",
        "quarter",
        "is_weekend",
        "is_public_holiday",
        "is_dengue_peak",
        "is_external_disaster"
    ]

    X = df[feature_cols].copy()
    y = df["rcc_demand_units"].copy()

    categorical_cols = ["blood_bank", "blood_type"]

    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))

    return X, y


def train_explainer_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_train, X_test


def generate_global_shap(model, X_train):
    sample_X = X_train.sample(n=min(1000, len(X_train)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_X)

    plt.figure()
    shap.summary_plot(shap_values, sample_X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "demand_feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "demand_feature_importance_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return explainer, sample_X


def generate_local_shap(explainer, X_test):
    one_row = X_test.iloc[[0]]
    shap_values_local = explainer.shap_values(one_row)

    plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_values_local[0],
        one_row.iloc[0],
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "demand_local_explanation.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return one_row


def main():
    print("Loading demand data...")
    df = load_data()

    print("Preparing demand features...")
    X, y = prepare_features(df)

    print("Training demand explainer model...")
    model, X_train, X_test = train_explainer_model(X, y)

    print("Generating global SHAP explanation...")
    explainer, sample_X = generate_global_shap(model, X_train)

    print("Generating local SHAP explanation...")
    one_row = generate_local_shap(explainer, X_test)

    print("Saved:")
    print("-", os.path.join(OUTPUT_DIR, "demand_feature_importance.png"))
    print("-", os.path.join(OUTPUT_DIR, "demand_feature_importance_bar.png"))
    print("-", os.path.join(OUTPUT_DIR, "demand_local_explanation.png"))

    print("\nExample demand row used for local explanation:")
    print(one_row)


if __name__ == "__main__":
    main()