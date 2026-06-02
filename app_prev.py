from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
import json
import holidays

from tensorflow.keras.models import load_model


app = FastAPI(
    title="Sales Forecast API"
)


# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
df_final = pd.read_csv("df_final.csv")
df_final['Date'] = pd.to_datetime(df_final['Date'])

xgb_model = joblib.load("models/xgb_model.pkl")

with open("models/best_model_registry.json", "r") as f:
    best_model_registry = json.load(f)

us_holidays = holidays.US()


# -----------------------------
# XGB FEATURE CREATION
# -----------------------------
def create_xgb_features(df):

    df = df.copy()

    df['lag_1'] = df['Total'].shift(1)
    df['lag_4'] = df['Total'].shift(4)
    df['lag_12'] = df['Total'].shift(12)

    df['rolling_mean_4'] = df['Total'].rolling(4).mean()
    df['rolling_std_4'] = df['Total'].rolling(4).std()

    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['year'] = df['Date'].dt.year

    df['month_sin'] = np.sin(
        2 * np.pi * df['month'] / 12
    )

    df['month_cos'] = np.cos(
        2 * np.pi * df['month'] / 12
    )

    df['is_holiday'] = (
        df['Date']
        .isin(us_holidays)
        .astype(int)
    )

    return df


FEATURES = [
    'lag_1',
    'lag_4',
    'lag_12',
    'rolling_mean_4',
    'rolling_std_4',
    'day_of_week',
    'month',
    'quarter',
    'week_of_year',
    'year',
    'month_sin',
    'month_cos',
    'is_holiday',
    'state_encoded'
]


# -----------------------------
# XGB FORECAST
# -----------------------------
def forecast_xgb(state, steps=8):

    state_df = df_final[
        df_final['State'] == state
    ].copy()

    forecasts = []

    for _ in range(steps):

        next_date = state_df['Date'].max() + pd.Timedelta(weeks=1)

        new_row = pd.DataFrame({
            'Date': [next_date],
            'State': [state],
            'Total': [np.nan]
        })

        state_df = pd.concat(
            [state_df, new_row],
            ignore_index=True
        )

        state_df['state_encoded'] = (
            df_final[df_final['State'] == state]['state_encoded'].iloc[0]
        )

        state_df = create_xgb_features(state_df)

        latest = state_df.iloc[[-1]]

        pred = xgb_model.predict(
            latest[FEATURES]
        )[0]

        state_df.loc[
            state_df.index[-1],
            'Total'
        ] = pred

        forecasts.append({
            "date": str(next_date.date()),
            "forecast": float(pred)
        })

    return forecasts


# -----------------------------
# LSTM FORECAST
# -----------------------------
def forecast_lstm(state, steps=8):

    safe_name = state.lower().replace(" ", "_")

    model = load_model(
        f"models/{safe_name}_lstm.keras"
    )

    scaler = joblib.load(
        f"models/{safe_name}_scaler.pkl"
    )

    state_df = df_final[
        df_final['State'] == state
    ].copy()

    series = state_df['Total'].values.reshape(-1, 1)

    scaled = scaler.transform(series)

    seq = scaled[-12:]

    forecasts = []

    last_date = state_df['Date'].max()

    for i in range(steps):

        pred_scaled = model.predict(
            seq.reshape(1, 12, 1),
            verbose=0
        )

        pred = scaler.inverse_transform(
            pred_scaled
        )[0][0]

        next_date = last_date + pd.Timedelta(weeks=i+1)

        forecasts.append({
            "date": str(next_date.date()),
            "forecast": float(pred)
        })

        seq = np.vstack([
            seq[1:],
            pred_scaled
        ])

    return forecasts


# -----------------------------
# API ENDPOINT
# -----------------------------
@app.get("/forecast/{state}")
def forecast(state: str):

    if state not in best_model_registry:
        raise HTTPException(
            status_code=404,
            detail="State not found"
        )

    best_model = best_model_registry[state]

    if best_model == "XGB":
        forecast_data = forecast_xgb(state)

    elif best_model == "LSTM":
        forecast_data = forecast_lstm(state)

    else:
        raise HTTPException(
            status_code=500,
            detail="Unsupported model"
        )

    return {
        "state": state,
        "best_model": best_model,
        "forecast_horizon_weeks": 8,
        "forecast": forecast_data
    }