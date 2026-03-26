import os
import sys
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from demand_forecasting.arima_model import load_data
from demand_forecasting.forecast_engine import generate_forecast
from demand_forecasting.model_selector import run_model_comparison
from demand_forecasting.demand_dashboard_utils import (
    normalize_demand_df,
    get_same_cluster_transfer_recommendations,
    generate_narrative_explanation,
    generate_runtime_xai
)

st.title("Demand Forecast Dashboard")
st.write("Unified forecasting, validation, alerting, recommendation, and explanation interface for NBTS managers.")


def safe_metric(value, decimals=3, suffix=""):
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass

    try:
        return f"{round(float(value), decimals)}{suffix}"
    except Exception:
        return "N/A"


# =========================================================
# 1. LOAD DATA AND DYNAMIC FILTERS
# =========================================================
raw_df = load_data()
df = normalize_demand_df(raw_df)

clusters = sorted(df["cluster"].dropna().unique().tolist())
selected_cluster = st.sidebar.selectbox("Select Cluster", clusters)

banks = sorted(
    df[df["cluster"] == selected_cluster]["blood_bank"].dropna().unique().tolist()
)
selected_bank = st.sidebar.selectbox("Select Blood Bank", banks)

blood_types = sorted(
    df[
        (df["cluster"] == selected_cluster) &
        (df["blood_bank"] == selected_bank)
    ]["blood_type"].dropna().unique().tolist()
)
selected_blood_type = st.sidebar.selectbox("Select Blood Type", blood_types)

current_stock = st.sidebar.number_input("Current Stock", min_value=0, value=50)
run_btn = st.sidebar.button("Run Forecast Analysis", type="primary")

# =========================================================
# 2. RUN FORECAST
# =========================================================
if run_btn:
    with st.spinner("Generating forecast and validation metrics..."):
        try:
            generate_forecast(selected_bank, selected_blood_type)

            try:
                metrics_df, best_model = run_model_comparison(selected_bank, selected_blood_type, save=True)
            except Exception as metric_error:
                st.warning(f"Model comparison failed, continuing with forecast output only. Details: {metric_error}")
                metrics_df = pd.DataFrame()
                best_model = "ARIMA"

            safe_bank = selected_bank.replace("/", "-").replace(" ", "_")
            safe_type = selected_blood_type.replace("+", "pos").replace("-", "neg")
            forecast_path = f"outputs/{safe_bank}_{safe_type}_forecast.csv"

            if not os.path.exists(forecast_path):
                st.error("Forecast file was not generated.")
                st.stop()

            forecast_df = pd.read_csv(forecast_path)

            forecast_values = pd.to_numeric(forecast_df["Forecast_Units"], errors="coerce")
            if forecast_values.notna().sum() == 0:
                st.error("Forecast contains no valid numeric values.")
                st.write("Debug forecast data:", forecast_df.head())
                st.stop()

            predicted_demand = int(round(forecast_values.mean()))
            gap = predicted_demand - int(current_stock)

            selected_df = df[
                (df["cluster"] == selected_cluster) &
                (df["blood_bank"] == selected_bank) &
                (df["blood_type"] == selected_blood_type)
            ].copy()

            # =========================================================
            # 3. SUMMARY + ALERTS + GAP
            # =========================================================
            st.subheader(f"Forecast Report: {selected_bank} | {selected_blood_type}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Demand", f"{predicted_demand} Units")
            c2.metric("Current Stock", f"{int(current_stock)} Units")

            if gap > 0:
                c3.metric("Status", "SHORTAGE", delta=f"-{gap} Units", delta_color="inverse")
                st.error(f"Shortage Notification: predicted demand exceeds current stock by {gap} units.")
            else:
                c3.metric("Status", "SAFE", delta=f"+{abs(gap)} Units")
                st.success("No immediate shortage risk detected for the selected forecast.")

            st.metric("Inventory Gap", gap)

            # =========================================================
            # 4. OPERATIONAL GUIDANCE
            # =========================================================
            recommendations = get_same_cluster_transfer_recommendations(
                df, selected_cluster, selected_bank, selected_blood_type
            )

            st.subheader("Operational Guidance and Recommendations")
            if gap > 0 and recommendations:
                st.info(
                    "Suggested transfer review within the same cluster: " +
                    ", ".join(recommendations)
                )
            elif gap > 0:
                st.warning("No same-cluster transfer candidates were identified from the current historical view.")
            else:
                st.info("No transfer recommendation is required because the current stock is sufficient.")

            # =========================================================
            # 5. FORECAST VISUALIZATION
            # =========================================================
            st.subheader("Demand Forecast Visualizations")

            history_df = selected_df.tail(30)[["date", "rcc_demand_units"]].copy()
            history_df.columns = ["Date", "Actual_Demand"]

            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history_df["Date"],
                    y=history_df["Actual_Demand"],
                    mode="lines+markers",
                    name="Historical Demand"
                ))

                if len(history_df) > 0:
                    last_date = history_df["Date"].iloc[-1]
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=len(forecast_df),
                        freq="D"
                    )
                else:
                    future_dates = list(range(len(forecast_df)))

                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast_values,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(dash="dot")
                ))

                if "Lower_Bound" in forecast_df.columns and "Upper_Bound" in forecast_df.columns:
                    lower_vals = pd.to_numeric(forecast_df["Lower_Bound"], errors="coerce")
                    upper_vals = pd.to_numeric(forecast_df["Upper_Bound"], errors="coerce")

                    if lower_vals.notna().any() and upper_vals.notna().any():
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=upper_vals,
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=lower_vals,
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            name="Confidence Interval"
                        ))

                fig.update_layout(
                    title="Historical Demand and Forecast",
                    xaxis_title="Date",
                    yaxis_title="Blood Units",
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                line_df = pd.DataFrame({
                    "Date": history_df["Date"],
                    "Historical Demand": history_df["Actual_Demand"]
                }).set_index("Date")
                st.line_chart(line_df)

            st.subheader("Forecast Output Table")
            st.dataframe(forecast_df, use_container_width=True)

            # =========================================================
            # 6. MODEL PERFORMANCE VALIDATION
            # =========================================================
            st.subheader("Model Performance Validation")

            if not metrics_df.empty:
                st.dataframe(metrics_df, use_container_width=True)

                valid_rows = metrics_df[metrics_df["RMSE"].notna()].copy()

                if not valid_rows.empty:
                    best_row = valid_rows.sort_values("RMSE").iloc[0]

                    v1, v2, v3, v4 = st.columns(4)
                    v1.metric("RMSE", safe_metric(best_row.get("RMSE")))
                    v2.metric("MAE", safe_metric(best_row.get("MAE")))
                    v3.metric("MAPE", safe_metric(best_row.get("MAPE"), suffix="%"))
                    v4.metric("Accuracy", safe_metric(best_row.get("Accuracy"), decimals=2, suffix="%"))
                else:
                    st.info("No valid numeric model metrics are available for this selection.")
            else:
                st.info("Validation metrics are not available for this selection.")

            # =========================================================
            # 7. NARRATIVE EXPLANATION
            # =========================================================
            st.subheader("Narrative Explanation")
            reasons = generate_narrative_explanation(
                selected_df,
                predicted_demand=predicted_demand,
                current_stock=int(current_stock),
                best_model=best_model
            )

            for idx, reason in enumerate(reasons, start=1):
                st.write(f"{idx}. {reason}")

            # =========================================================
            # 8. XAI OUTPUTS
            # =========================================================
            st.subheader("Explainable AI (XAI) Outputs")

            global_xai_path, local_xai_path = generate_runtime_xai(selected_df)

            existing_global = "outputs/xai/demand_feature_importance_bar.png"
            existing_local = "outputs/xai/demand_local_explanation.png"

            if os.path.exists(existing_global):
                st.image(existing_global, caption="Global SHAP Factor Impact Plot")
            elif global_xai_path and os.path.exists(global_xai_path):
                st.image(global_xai_path, caption="Runtime Global Factor Impact Plot")
            else:
                st.info("Global factor impact plot is not available yet.")

            if local_xai_path and os.path.exists(local_xai_path):
                st.image(local_xai_path, caption="Local Factor Impact Plot for Latest Forecast Context")
            elif os.path.exists(existing_local):
                st.image(existing_local, caption="Existing Local SHAP Explanation")
            else:
                st.info("Local factor impact plot is not available yet.")

            st.success("The selected forecast, shortage status, metrics, recommendations, and explanations are now shown in one unified dashboard.")

        except Exception as e:
            st.error(f"Error running demand forecast dashboard: {e}")
else:
    st.info("Select a cluster, blood bank, blood type, and current stock, then click 'Run Forecast Analysis'.")