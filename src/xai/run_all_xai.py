import os

print("Running donor SHAP explainer...")
os.system("python src/xai/donor_shap_explainer.py")

print("\nRunning demand SHAP explainer...")
os.system("python src/xai/demand_shap_explainer.py")

print("\nAll XAI outputs generated in outputs/xai/")