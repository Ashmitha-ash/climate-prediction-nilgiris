"""
Standalone Model Trainer for Nilgiris Spatio-Temporal Climate Prediction
Author: Ashmitha A
"""
import os
import argparse
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NILGIRIS_LOCATIONS = {
    "Coonoor": {"lat": 11.3536, "lon": 76.7937},
    "Ooty": {"lat": 11.4102, "lon": 76.6950},
    "Kotagiri": {"lat": 11.4268, "lon": 76.8838},
    "Gudalur": {"lat": 11.4167, "lon": 76.6833},
    "Wellington": {"lat": 11.3653, "lon": 76.7869}
}

SEQUENCE_LENGTH = 72

def train_model_for_location(location_name):
    if location_name not in NILGIRIS_LOCATIONS:
        print(f"Error: Location '{location_name}' not recognized. Available: {list(NILGIRIS_LOCATIONS.keys())}")
        return

    coords = NILGIRIS_LOCATIONS[location_name]
    lat, lon = coords["lat"], coords["lon"]
    model_filename = os.path.join(BASE_DIR, f"{location_name.lower()}_model.txt")
    scaler_filename = os.path.join(BASE_DIR, f"{location_name.lower()}_scaler.gz")

    print(f"📥 Fetching historical weather data for {location_name} ({lat}, {lon})...")
    api_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2022-01-01",
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "hourly": "temperature_2m"
    }

    response = requests.get(api_url, params=params, timeout=60)
    response.raise_for_status()

    df = pd.DataFrame(response.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    dataframe = df[["temperature_2m"]].dropna()

    print(f"📊 Training dataset prepared: {len(dataframe)} hourly records.")

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataframe)

    # Sequence generation (72 hours context)
    X, y = [], []
    for i in range(len(scaled_data) - SEQUENCE_LENGTH):
        X.append(scaled_data[i:(i + SEQUENCE_LENGTH)].flatten())
        y.append(scaled_data[i + SEQUENCE_LENGTH])

    X, y = np.array(X), np.array(y).ravel()

    # Train LightGBM Regressor
    print("🚀 Training LightGBM model...")
    lgb_params = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "verbose": -1
    }
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X, y, eval_set=[(X, y)], eval_metric="mae", callbacks=[lgb.early_stopping(10, verbose=False)])

    # Save Booster and Scaler
    model.booster_.save_model(model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"✅ Saved model to: {model_filename}")
    print(f"✅ Saved scaler to: {scaler_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train climate prediction model for a Nilgiris location.")
    parser.add_argument("--location", default="Ooty", choices=list(NILGIRIS_LOCATIONS.keys()), help="Nilgiris location")
    args = parser.parse_args()

    train_model_for_location(args.location)
