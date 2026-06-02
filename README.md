# 📈 Multi-Model Sales Forecasting System

End-to-End Time Series Forecasting Platform using XGBoost, LSTM, SARIMA, and Prophet with Automatic Model Selection, FastAPI Deployment, Docker Containerization, and Interactive Streamlit Dashboard.

---

# 🚀 Project Overview

Accurate sales forecasting is critical for inventory planning, demand estimation, budgeting, and business decision-making.

Different geographical regions often exhibit different sales patterns due to:

* Seasonal variations
* Economic trends
* Local demand fluctuations
* Holiday effects
* State-specific sales behavior

Instead of relying on a single forecasting algorithm, this project builds a production-ready forecasting system that:

✅ Trains multiple forecasting models independently for each state

✅ Compares model performance using forecasting metrics

✅ Automatically selects the best-performing model

✅ Generates future sales forecasts

✅ Serves predictions through REST APIs

✅ Visualizes forecasts through an interactive dashboard

---

# 📊 Dataset

The dataset contains weekly sales records across 43 U.S. states from 2019 to 2023.

### Dataset Characteristics

* Multi-state sales history
* Weekly granularity
* Missing historical periods
* State-specific trends
* Seasonal sales patterns
* Holiday-driven demand fluctuations

---

# 🎯 Business Problem

Forecast the next 8 weeks of sales for every state using historical sales data.

The system should:

* Handle missing dates
* Handle missing sales values
* Capture trends and seasonality
* Compare multiple forecasting techniques
* Select the best model automatically
* Serve predictions through APIs
* Provide dashboard-based visualization

---

# ⚙️ Project Pipeline

### 1. Data Cleaning

* Missing timeline reconstruction
* Missing value imputation
* Date standardization

### 2. Feature Engineering

* Lag features
* Rolling statistics
* Calendar features
* Holiday indicators
* Cyclical encoding

### 3. Model Development

* XGBoost
* LSTM
* SARIMA
* Prophet

### 4. Model Evaluation

Each model is evaluated independently for every state using:

* RMSE
* MAE
* MAPE

### 5. Automatic Model Selection

The best-performing model is selected dynamically for each state.

### 6. Forecast Generation

Recursive forecasting is used to predict future sales for:

* 8-week forecast horizon

### 7. Model Deployment

* FastAPI Backend
* Docker Containerization
* Render Deployment

### 8. Visualization

Interactive Streamlit Dashboard

---

# 🧠 Feature Engineering

### Lag Features

* lag_1
* lag_4
* lag_12

### Rolling Statistics

* rolling_mean_4
* rolling_std_4

### Calendar Features

* day_of_week
* month
* quarter
* week_of_year
* year

### Holiday Features

Holiday indicators generated using:

```python
holidays.US()
```

### Cyclical Encoding

* month_sin
* month_cos

---

# 🤖 Forecasting Models

## 1️⃣ XGBoost

Gradient Boosting Regression model trained on engineered time-series features.

### Advantages

* Handles non-linear relationships
* Fast inference
* Strong tabular performance

---

## 2️⃣ LSTM

Long Short-Term Memory neural network designed for sequential forecasting.

### Advantages

* Captures temporal dependencies
* Learns long-term patterns
* Deep learning approach

---

## 3️⃣ SARIMA

Seasonal AutoRegressive Integrated Moving Average model.

### Advantages

* Statistical forecasting
* Explicit seasonality modeling
* Strong baseline model

---

## 4️⃣ Prophet

Meta's forecasting framework.

### Advantages

* Trend decomposition
* Holiday effects
* Seasonality modeling

---

# 🏆 Automatic Model Selection

Instead of assuming one model works best globally, the system:

1. Trains all four forecasting models
2. Evaluates each model independently
3. Compares forecasting metrics
4. Selects the best model for each state

This creates a dynamic forecasting engine capable of adapting to different regional sales behaviors.

---

# 📏 Evaluation Metrics

### RMSE

Root Mean Squared Error

Measures forecasting error magnitude.

### MAE

Mean Absolute Error

Measures average prediction error.

### MAPE

Mean Absolute Percentage Error

Measures percentage forecasting error.

Lower values indicate better forecasting performance.

---

# 🌐 FastAPI Backend

The forecasting engine is exposed through REST APIs.

## Available States

```http
GET /states
```

Returns all supported states.

---

## Forecast Endpoint

```http
GET /forecast/{state}
```

Returns:

* Best model
* 8-week forecast
* Forecast dates

---

## Best Model Endpoint

```http
GET /best-model/{state}
```

Returns the selected model for a given state.

---

## Metrics Endpoint

```http
GET /metrics/{state}
```

Returns model evaluation metrics.

---

# 📊 Interactive Dashboard

The Streamlit dashboard allows users to:

### State Selection

Select any supported state.

### Forecast Visualization

View 8-week sales forecasts.

### KPI Monitoring

Display:

* Best Model
* RMSE
* MAE
* MAPE

### Forecast Tables

Inspect forecasted values.

### Model Benchmarking

Compare:

* XGBoost
* LSTM
* SARIMA
* Prophet

for each state.

---

# 🐳 Docker Deployment

Both FastAPI and Streamlit applications were containerized using Docker.

### Benefits

* Environment consistency
* Easy deployment
* Scalability
* Platform independence

---

# ☁️ Deployment

## FastAPI

Deployed on Render

API Documentation:

https://multi-model-forecasting-system.onrender.com/docs

---

## Streamlit Dashboard

Deployed on Render

Dashboard:

https://multi-state-sales-forecasting-system.onrender.com/

---

# 🛠️ Tech Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Deep Learning

* TensorFlow
* Keras

### Time-Series Forecasting

* Prophet
* SARIMA (Statsmodels)

### Deployment

* FastAPI
* Streamlit
* Docker
* Render

### Visualization

* Plotly

### Model Serialization

* Joblib

---

# 📂 Project Structure

```text
Forecast_Project/
│
├── app.py
├── requirements.txt
├── Dockerfile
│
├── models/
│   ├── best_model_registry.json
│   ├── metrics_registry.pkl
│   ├── statewise_xgb_models
│   ├── statewise_lstm_models
│   └── prophet_models
│
├── Dashboard/
│   ├── dashboard.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── df_final.csv
│
└── notebooks/
```

---

# 🎓 Key Learnings

* Time Series Forecasting
* Feature Engineering for Temporal Data
* Recursive Forecasting
* Model Benchmarking
* Automatic Model Selection
* FastAPI Development
* Streamlit Dashboards
* Docker Containerization
* Cloud Deployment
* End-to-End Machine Learning Systems

---

# 🔮 Future Improvements

* Automated model retraining
* MLflow experiment tracking
* Ensemble forecasting
* CI/CD pipeline
* Cloud storage integration
* Forecast monitoring system
* Drift detection
* Real-time forecasting

---

# 👩‍💻 Author

**Swati Yadav**

Aspiring Data Scientist | Machine Learning Engineer

GitHub:

https://github.com/swati666
