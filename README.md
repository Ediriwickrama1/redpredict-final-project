# RedPredict: AI-Driven Blood Supply Management System

RedPredict is an intelligent decision support system developed for blood transfusion services to improve blood demand forecasting and donor management. The system combines time-series forecasting, donor classification, reminder generation, explainable AI, and dashboard-based monitoring into one unified Streamlit application.

## Project Overview

Blood transfusion services face two major operational problems:

- inaccurate blood demand forecasting, which can lead to shortages and expired blood waste
- inefficient donor management, which makes it difficult to identify reliable and rare blood donors

RedPredict addresses these challenges through two core modules:

1. **Demand Forecasting Module**
   - forecasts blood demand for specific blood banks and blood types
   - uses ARIMA and LSTM models
   - provides shortage alerts, inventory gap calculations, recommendations, and explainable outputs

2. **Donor Management Module**
   - predicts donor return behavior
   - identifies and prioritizes eligible donors
   - generates reminders for donors after configurable intervals
   - logs donor communications and supports role-based dashboard access

The system also integrates **Explainable AI (XAI)** using SHAP-based visualizations to improve trust and transparency in predictions.

---

## Main Features

### Demand Forecasting
- dynamic filtering by **cluster**, **blood bank**, and **blood type**
- demand forecasting using **ARIMA** and **LSTM**
- forecasting metrics: **RMSE, MAE, MAPE, Accuracy**
- shortage alerts and inventory gap calculations
- same-cluster transfer recommendations
- historical vs forecast visualizations
- narrative explanation for forecast reasoning
- factor impact plots for explainability

### Donor Management
- donor return prediction using a classification model
- donor prioritization using return probability, rare blood type, and donation history
- donor eligibility filtering
- configurable reminder generation
- donor communication logging
- role-based login system

### Explainable AI
- SHAP-based explanation outputs for donor prediction
- demand-side factor impact explanations
- visual explanation charts integrated into dashboard

### Dashboards
- unified Streamlit dashboard
- demand forecast dashboard
- donor management dashboard
- reminder dashboard
- communication log dashboard
- performance dashboard
- model performance dashboard
- XAI dashboard

---

## Tech Stack

- **Frontend / Dashboard**: Streamlit
- **Programming Language**: Python
- **Forecasting Models**: ARIMA, LSTM
- **Classification Model**: Logistic Regression
- **Database**: MySQL (XAMPP for local development)
- **Explainability**: SHAP
- **Visualization**: Plotly, Matplotlib
- **Version Control**: Git, GitHub

---

## Project Structure

```text
redpredict-final-project/
├── data/
├── docs/
├── models/
├── notebooks/
├── outputs/
├── src/
│   ├── alerts/
│   ├── auth/
│   ├── dashboard/
│   ├── database/
│   ├── demand_forecasting/
│   ├── donor_management/
│   ├── evaluation/
│   ├── utils/
│   └── xai/
├── tests/
├── .gitignore
├── requirements.txt
└── run_full_demo.py

How to Run the Project Locally

1. Clone the repository
git clone <your-github-repo-url>
cd redpredict-final-project
2. Create and activate virtual environment
Windows
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Start MySQL using XAMPP

Make sure:

XAMPP is installed
MySQL is running
your local database is available as:
blood_donor_system
5. Run the full preparation pipeline
python run_full_demo.py

This runs the main preprocessing and output generation steps required before launching the dashboard.

6. Launch the Streamlit app
streamlit run src/dashboard/app.py
Local Database Setup

The system uses a local MySQL database during development.

Expected database:

blood_donor_system

Key tables:

donors
donor_communications
system_users

The local MySQL connection is configured in:

src/database/mysql_connection.py
Demo Login Accounts

For local testing, the system supports role-based access.

Typical roles:

Blood Bank Manager
Donor Coordinator
Hospital Staff

These users are stored in the system_users table.

Evaluation Metrics
Forecasting Module
RMSE
MAE
MAPE
Accuracy
Donor Classification Module
Accuracy
Precision
Recall
F1 Score
AUC-ROC
Confusion Matrix
ROC Curve
Explainable AI

RedPredict integrates explainable AI to improve trust in healthcare settings.

The system provides:

narrative explanations for demand forecasts
SHAP-based feature importance outputs
local and global factor impact visualizations

These outputs help users understand why the model made a specific prediction.

Current Scope

This project is a working prototype focused on:

decision support
forecasting
donor prioritization
reminder generation
explainable AI
dashboard-based monitoring
Not included in current scope
financial transaction management
blood transportation logistics
real-time hospital system integration
production-grade encryption and cloud authentication
live SMS/email delivery infrastructure
Future Improvements

Possible future extensions include:

cloud-hosted deployment
password hashing and stronger authentication
real-time SMS and email reminders
integration with hospital information systems
multilingual donor communication
live inventory integration across sites
Author

Samadhi Lochana Ediriwickrama
Final Year Individual Project
BSc (Hons) Computer Science

License

This project is developed for academic purposes.