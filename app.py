
from fastapi import FastAPI, HTTPException

import pandas as pd
import numpy as np
import joblib
import json
import holidays

from tensorflow.keras.models import load_model

from prophet.serialize import model_from_json


# ==================================================
# APP
# ==================================================

app = FastAPI(
    title="Sales Forecast API"
)


# ==================================================
# LOAD DATA
# ==================================================

df_final = pd.read_csv("df_final.csv")

df_final["Date"] = pd.to_datetime(
    df_final["Date"]
)

with open(
    "models/best_model_registry.json",
    "r"
) as f:

    best_model_registry = json.load(f)

us_holidays = holidays.US()

SEQ_LENGTH = 12


# ==================================================
# XGB FEATURES
# ==================================================

FEATURES = [
    "lag_1",
    "lag_4",
    "lag_12",
    "rolling_mean_4",
    "rolling_std_4",
    "day_of_week",
    "month",
    "quarter",
    "week_of_year",
    "year",
    "month_sin",
    "month_cos",
    "is_holiday"
]


# ==================================================
# FEATURE CREATION
# ==================================================

def create_xgb_features(df):

    df = df.copy()

    df["lag_1"] = (
        df["Total"].shift(1)
    )

    df["lag_4"] = (
        df["Total"].shift(4)
    )

    df["lag_12"] = (
        df["Total"].shift(12)
    )

    df["rolling_mean_4"] = (
        df["Total"]
        .rolling(4)
        .mean()
    )

    df["rolling_std_4"] = (
        df["Total"]
        .rolling(4)
        .std()
    )

    df["day_of_week"] = (
        df["Date"]
        .dt.dayofweek
    )

    df["month"] = (
        df["Date"]
        .dt.month
    )

    df["quarter"] = (
        df["Date"]
        .dt.quarter
    )

    df["week_of_year"] = (
        df["Date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["year"] = (
        df["Date"]
        .dt.year
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    df["is_holiday"] = (
        df["Date"]
        .isin(us_holidays)
        .astype(int)
    )

    return df


# ==================================================
# LSTM FORECAST
# ==================================================

def forecast_lstm(
    state,
    steps=8
):

    safe_name = (
        state.lower()
        .replace(" ", "_")
    )

    model = load_model(
        f"models/{safe_name}_lstm.keras"
    )

    scaler = joblib.load(
        f"models/{safe_name}_scaler.pkl"
    )

    state_df = df_final[
        df_final["State"] == state
    ].copy()

    series = (
        state_df["Total"]
        .values
        .reshape(-1, 1)
    )

    scaled_series = scaler.transform(
        series
    )

    sequence = scaled_series[
        -SEQ_LENGTH:
    ]

    forecasts = []

    last_date = (
        state_df["Date"]
        .max()
    )

    for i in range(steps):

        pred_scaled = model.predict(
            sequence.reshape(
                1,
                SEQ_LENGTH,
                1
            ),
            verbose=0
        )

        pred = scaler.inverse_transform(
            pred_scaled
        )[0][0]

        next_date = (
            last_date
            + pd.Timedelta(
                weeks=i + 1
            )
        )

        forecasts.append({
            "date": str(
                next_date.date()
            ),
            "forecast": float(pred)
        })

        sequence = np.vstack([
            sequence[1:],
            pred_scaled
        ])

    return forecasts


# ==================================================
# XGB FORECAST
# ==================================================

def forecast_xgb(
    state,
    steps=8
):

    safe_name = (
        state.lower()
        .replace(" ", "_")
    )

    model = joblib.load(
        f"models/{safe_name}_xgb.pkl"
    )

    state_df = df_final[
        df_final["State"] == state
    ].copy()

    state_df = (
        state_df
        .sort_values("Date")
        .copy()
    )

    forecasts = []

    for _ in range(steps):

        next_date = (
            state_df["Date"].max()
            + pd.Timedelta(
                weeks=1
            )
        )

        new_row = pd.DataFrame({
            "Date": [next_date],
            "State": [state],
            "Total": [np.nan]
        })

        state_df = pd.concat(
            [state_df, new_row],
            ignore_index=True
        )

        state_df = create_xgb_features(
            state_df
        )

        latest_row = (
            state_df
            .iloc[[-1]]
        )

        prediction = model.predict(
            latest_row[FEATURES]
        )[0]

        state_df.loc[
            state_df.index[-1],
            "Total"
        ] = prediction

        forecasts.append({
            "date": str(
                next_date.date()
            ),
            "forecast": float(
                prediction
            )
        })

    return forecasts


# ==================================================
# PROPHET FORECAST
# ==================================================

def forecast_prophet(
    state,
    steps=8
):

    safe_name = (
        state.lower()
        .replace(" ", "_")
    )

    with open(
        f"models/{safe_name}_prophet.json",
        "r"
    ) as fin:

        model = model_from_json(
            fin.read()
        )

    future = (
        model
        .make_future_dataframe(
            periods=steps,
            freq="W"
        )
    )

    forecast = model.predict(
        future
    )

    future_part = (
        forecast
        .tail(steps)
    )

    forecasts = []

    for _, row in future_part.iterrows():

        forecasts.append({
            "date": str(
                row["ds"].date()
            ),
            "forecast": float(
                row["yhat"]
            )
        })

    return forecasts


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/")
def home():

    return {
        "message": "Sales Forecast API Running"
    }


# ==================================================
# FORECAST ENDPOINT
# ==================================================

@app.get("/forecast/{state}")
def forecast(
    state: str
):

    if state not in best_model_registry:

        raise HTTPException(
            status_code=404,
            detail="State not found"
        )

    best_model = (
        best_model_registry[state]
    )

    if best_model == "LSTM":

        forecast_data = (
            forecast_lstm(
                state
            )
        )

    elif best_model == "XGB":

        forecast_data = (
            forecast_xgb(
                state
            )
        )

    elif best_model == "PROPHET":

        forecast_data = (
            forecast_prophet(
                state
            )
        )

    else:

        raise HTTPException(
            status_code=500,
            detail=f"Unsupported model: {best_model}"
        )

    return {
        "state": state,
        "best_model": best_model,
        "forecast_horizon_weeks": 8,
        "forecast": forecast_data
    }
