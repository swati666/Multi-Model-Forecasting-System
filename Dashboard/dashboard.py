import streamlit as st
import pandas as pd
import joblib
import requests
import plotly.express as px




# PAGE CONFIG

st.set_page_config(page_title="Sales Forecasting Dashboard",page_icon="📈",layout="wide")

# LOAD REGISTRIES


@st.cache_resource
def load_registries():

    metrics_registry = joblib.load(
        "metrics_registry.pkl"
    )

    return metrics_registry


metrics_registry = load_registries()

# FASTAPI URL

# API_URL = "http://localhost:8080"

API_URL = "https://multi-model-forecasting-system.onrender.com"


# HEADER

st.title("📈 Multi-State Sales Forecasting Dashboard")

st.markdown("""Interactive forecasting platform with automatic best-model selection.

**Models Benchmarked**

- XGBoost
- LSTM
- SARIMA
- Prophet

Forecast horizon: **8 Weeks**
"""

)

st.markdown("---")



# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title(
    "Forecast Controls"
)

states = sorted(
    list(metrics_registry.keys())
)

selected_state = st.sidebar.selectbox(
    "Select State",
    states
)

if "generate" not in st.session_state:
    st.session_state.generate = False

if st.sidebar.button(
    "🚀 Generate Forecast",
    use_container_width=True
):
    st.session_state.generate = True

if not st.session_state.generate:
    st.info(
        """
        Select a state and click Generate Forecast
        """
    )
    st.stop()
# API REQUEST


with st.spinner("Generating 8-week forecast..."):

    try:

        response = requests.get(
            f"{API_URL}/forecast/{selected_state}",
            timeout=60
        )

        if response.status_code != 200:

            st.error(
                "Forecast API returned an error."
            )

            st.stop()

        forecast_response = (
            response.json()
        )

    except requests.exceptions.ConnectionError:

        st.error(
            """
            FastAPI server is not running.

            Start backend first:

            uvicorn app:app --reload
            """
        )

        st.stop()

    except requests.exceptions.Timeout:

        st.error(
            "Forecast request timed out."
        )

        st.stop()

    except Exception as e:

        st.error(
            f"Unexpected error: {e}"
        )

        st.stop()

# EXTRACT RESPONSE

best_model = forecast_response["best_model"]

forecast_df = pd.DataFrame(forecast_response["forecast"])

metrics_df = metrics_registry[selected_state]

model_metrics = metrics_df[metrics_df["Model"]== best_model].iloc[0]

# PAGE TITLE

st.subheader(f"📍 Forecast for {selected_state}")

st.caption(f"Automatically selected model: {best_model}")

st.markdown("---")

# KPI CARDS


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        label="🏆 Best Model",
        value=best_model
    )

with col2:

    st.metric(
        label="RMSE",
        value=f"{model_metrics['RMSE']:,.0f}"
    )

with col3:

    st.metric(
        label="MAE",
        value=f"{model_metrics['MAE']:,.0f}"
    )

with col4:

    st.metric(
        label="MAPE",
        value=f"{model_metrics['MAPE']:.2f}%"
    )

st.caption(f"{best_model} achieved the lowest MAPE for {selected_state}")

st.markdown("---")

# FORECAST CHART

st.subheader("📅 8-Week Sales Forecast")

forecast_df["date"] = pd.to_datetime(forecast_df["date"])

fig = px.line(forecast_df,x="date",y="forecast",markers=True,title=f"{selected_state} Forecast Trend")

fig.update_layout(xaxis_title="Forecast Week",yaxis_title="Sales",hovermode="x unified")

st.plotly_chart(fig,use_container_width=True)

# FORECAST TABLE


st.subheader("Forecast Data")

display_df = forecast_df.copy()

display_df.columns = ["Date","Forecast"]

display_df["Forecast"] = (display_df["Forecast"].round(0).astype(int))

st.dataframe(display_df,use_container_width=True)

st.markdown("---")

# MODEL COMPARISON TABLE


st.subheader("📊 Model Benchmark Comparison")

comparison_df = metrics_df.sort_values(by="MAPE")

st.dataframe(comparison_df,use_container_width=True)


# MAPE COMPARISON CHART

metric_fig = px.bar(comparison_df,x="MAPE",y="Model",orientation="h",title="Model Comparison by MAPE")

st.plotly_chart(metric_fig,use_container_width=True)

st.markdown("---")

# FOOTER



st.caption("""Built with FastAPI • Streamlit • XGBoost • LSTM • SARIMA • Prophet""")

