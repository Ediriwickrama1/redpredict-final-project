import os

print("Preparing full demo...")

print("1. Preprocessing donor data from MySQL...")
os.system("python src/donor_management/preprocess_donors.py")

print("2. Generating donor reminders...")
os.system("python src/donor_management/reminder_engine.py")

print("3. Generating donor SHAP explanations...")
os.system("python src/xai/donor_shap_explainer.py")

print("4. Generating demand SHAP explanations...")
os.system("python src/xai/demand_shap_explainer.py")

print("5. Generating shortage alerts...")
os.system("python src/alerts/shortage_alert_engine.py")

print("6. Performance logs updated automatically during execution.")

print("\nDone. Now launch the dashboard with:")
print("streamlit run src/dashboard/app.py")