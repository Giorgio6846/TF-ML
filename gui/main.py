
"""
Streamlit App for 3-Day Coin Price Forecasts
------------------------------------------------
This app loads trained LSTM models for selected coins, performs inference for the next 72 hours, and compares the predictions to real price data fetched from CoinGecko (using only the requests library, with caching).

Key Functions:
- load_selected_coins: Loads the list of coins to forecast.
- load_scaler_for_coin: Loads the MinMaxScaler for each coin.
- load_model_for_coin: Loads the trained PyTorch Lightning model for each coin.
- get_last_sequence: Prepares the last sequence of features for inference from local JSON data.
- forecast_and_unscale: Runs iterative forecasting and inverse-scales the predictions.
- fetch_real_prices: Fetches and processes real hourly prices from CoinGecko, with disk caching.

The UI displays a summary table of daily means and a detailed hourly comparison chart for each coin.
"""
# app.py


import os
import sys
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import torch
import requests
import time

from dotenv import load_dotenv
from datetime import datetime

CACHE_DIR = "data/price_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

print("▶ Working dir:", os.getcwd())
print("▶ This file:", __file__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- your own modules; update these imports to your paths ---
from src.trainer import LitTrainer
from src.utils import process_coingecko_hourly

load_dotenv()

# —————————————————————————————————————————————
# 1) Load selected coins from JSON
# —————————————————————————————————————————————

@st.cache_data
def load_selected_coins(path: str = "data/selected_coins.json") -> list[str]:
    """
    Load the list of selected coins from a JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)


COINGECKO_API_KEY = os.getenv("coingecko_API")
COINS = load_selected_coins()

# —————————————————————————————————————————————
# 2) Load a scaler for each coin
# —————————————————————————————————————————————

@st.cache_resource
def load_scaler_for_coin(name: str, scaled_root: str = "data/scaled") -> joblib:
    """
    Load the MinMaxScaler for a given coin.
    """
    path = os.path.join(scaled_root, f"info_{name}.joblib")
    return joblib.load(path)

# —————————————————————————————————————————————
# 3) Load a model for each coin
# —————————————————————————————————————————————

@st.cache_resource
def load_model_for_coin(
    name: str,
    exports_root: str = "models/exports",
    hidden_size: int = 76,
    num_layers: int = 2,
    lr: float = 0.005094100053762949,
    dropout: float = 0.4197932446614159
) -> LitTrainer:
    """
    Load the trained PyTorch Lightning model for a given coin.
    """
    model = LitTrainer(
        hidden_size=hidden_size,
        num_layers=num_layers,
        lr=lr,
        dropout=dropout
    )
    pth_path = os.path.join(exports_root, name, "best_model.pth")
    state_dict = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# —————————————————————————————————————————————
# 4) Prepare the last sequence of features
# —————————————————————————————————————————————

# —————————————————————————————————————————————
# 4) Prepare the last sequence of features (from local data file, not priceDataframe)
# —————————————————————————————————————————————



@st.cache_data
def get_last_sequence(name: str, seq_len: int = 72) -> torch.Tensor:
    """
    Prepare the last sequence of features for a given coin from local JSON data.
    Uses the features required by the model: price, hour_cos, hour_sin, dayweek_cos, dayweek_sin.
    Returns a tensor of shape (1, seq_len, 5).
    """
    jsonPath = f"data/info_{name}.json"
    with open(jsonPath, "r") as f:
        data = json.load(f)
    # Try to get cap and volume if present, else fill with nan
    price_list = data["prices"]
    cap_list = data.get("market_caps", None)
    vol_list = data.get("total_volumes", None)
    df = pd.DataFrame({
        "price_timestamp": [item[0] for item in price_list],
        "price": [item[1] for item in price_list],
        "cap": [item[1] for item in cap_list] if cap_list is not None else [np.nan]*len(price_list),
        "volume": [item[1] for item in vol_list] if vol_list is not None else [np.nan]*len(price_list)
    })
    df["price_timestamp"] = pd.to_datetime(df["price_timestamp"], unit="ms")
    df["hour"] = df["price_timestamp"].dt.hour
    df["dayweek"] = df["price_timestamp"].dt.dayofweek
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["dayweek_cos"] = np.cos(2 * np.pi * df["hour"] / 7)
    df["dayweek_sin"] = np.sin(2 * np.pi * df["hour"] / 7)
    features = df[[
        "price", "cap", "volume", "hour_cos", "hour_sin", "dayweek_cos", "dayweek_sin"
    ]].values
    seq = features[-seq_len:]
    return torch.tensor(seq[np.newaxis, ...], dtype=torch.float32)

# —————————————————————————————————————————————
# 5) Iterative forecasting + inverse-scaling
# —————————————————————————————————————————————



def forecast_and_unscale(
    model: LitTrainer,
    scaler: joblib,
    X0: torch.Tensor,
    steps: int = 72
) -> np.ndarray:
    """
    Perform iterative forecasting for a given model and input sequence, then inverse-scale the predictions.
    Returns a numpy array of real-valued price predictions.
    """
    norm_preds = []
    X = X0.clone()
    cap_preds = []
    vol_preds = []
    for _ in range(steps):
        with torch.no_grad():
            y_pred = model.model(X).cpu().numpy()
            # y_pred shape: (1, output_size) or (1, N)
            if y_pred.ndim == 2:
                y_price = y_pred[0, 0]
                y_cap = y_pred[0, 1] if y_pred.shape[1] > 1 else np.nan
                y_vol = y_pred[0, 2] if y_pred.shape[1] > 2 else np.nan
            elif y_pred.ndim == 1:
                y_price = y_pred[0]
                y_cap = y_pred[1] if len(y_pred) > 1 else np.nan
                y_vol = y_pred[2] if len(y_pred) > 2 else np.nan
            else:
                raise ValueError(f"Unexpected prediction shape: {y_pred.shape}")
        norm_preds.append(y_price)
        cap_preds.append(y_cap)
        vol_preds.append(y_vol)
        # roll the window: drop oldest, append new_row
        new_row = np.zeros(X.shape[2], dtype=float)
        new_row[0] = y_price
        new_row[1] = y_cap
        new_row[2] = y_vol
        arr = np.concatenate([X.numpy()[0,1:], new_row[None]], axis=0)
        X = torch.tensor(arr[None], dtype=torch.float32)

    norm_preds = np.array(norm_preds)  # shape (steps,)
    cap_preds = np.array(cap_preds)
    vol_preds = np.array(vol_preds)
    # build dummy for scaler (expects 3 cols: price, cap, volume)
    dummy = np.zeros((len(norm_preds), 3))
    dummy[:, 0] = norm_preds
    dummy[:, 1] = cap_preds
    dummy[:, 2] = vol_preds
    real_preds = scaler.inverse_transform(dummy)
    return real_preds[:, 0], real_preds[:, 1], real_preds[:, 2]


# —————————————————————————————————————————————
# 6) Fetch actual prices from CoinGecko (requests only, cache to disk)
# —————————————————————————————————————————————


def fetch_real_prices(name: str, days: int = 3) -> np.ndarray:
    """
    Fetch price data for a coin from CoinGecko (using requests), cache to disk, and process for model compatibility.
    Uses automatic granularity: 1 day = 5-minutely, 2-90 days = hourly, >90 days = daily.
    Returns a numpy array of the last 72 prices (hourly if days <= 90, else daily).
    """
    now = datetime.utcnow()
    hour_stamp = now.strftime("%Y%m%dT%H")
    cache_file = os.path.join(CACHE_DIR, f"{name}_{days}_{hour_stamp}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            prices = json.load(f)
    else:
        url = f"https://api.coingecko.com/api/v3/coins/{name}/market_chart"
        params = {"vs_currency": "usd", "days": days}  # leave interval empty for automatic granularity
        headers = {"Accept": "application/json"}
        if COINGECKO_API_KEY:
            headers["X-CoinGecko-Api-Key"] = COINGECKO_API_KEY
        time.sleep(0.2)  # Slowdown to respect 30 requests/minute limit (200ms)
        resp = requests.get(url, params=params, headers=headers)
        resp.raise_for_status()
        prices = resp.json().get("prices", [])
        with open(cache_file, "w") as f:
            json.dump(prices, f)

    # Use the new utility to process the data for model compatibility
    df = process_coingecko_hourly(prices[-72:])
    return df["price"].values

# —————————————————————————————————————————————
# 7) Build Streamlit UI (ModelTraining-style inference and comparison)
# —————————————————————————————————————————————
st.set_page_config(page_title="3-Day Coin Forecasts", layout="wide")
st.title("3-Day Price Forecasts vs. CoinGecko (requests only)")

# Load all resources once
models  = {c: load_model_for_coin(c)  for c in COINS}
scalers = {c: load_scaler_for_coin(c) for c in COINS}


st.subheader("Forecast Summary for Selected Coins")
rows = []
for coin in COINS:
    X0 = get_last_sequence(coin)
    preds, caps, vols = forecast_and_unscale(models[coin], scalers[coin], X0, steps=72)
    actual = fetch_real_prices(coin, days=3)
    # For now, actual only has price. If you want to fetch actual cap/vol, you need to update fetch_real_prices.
    pred_daily = preds.reshape(3,24).mean(axis=1)
    cap_daily = caps.reshape(3,24).mean(axis=1)
    vol_daily = vols.reshape(3,24).mean(axis=1)
    actual_daily = actual.reshape(3,24).mean(axis=1) if len(actual) == 72 else [np.nan]*3

    # Percentage difference: (pred - actual) / actual * 100
    diff_daily = [((p - a) / a * 100) if (not np.isnan(a) and a != 0) else np.nan for p, a in zip(pred_daily, actual_daily)]
    rows.append([coin, *pred_daily, *cap_daily, *vol_daily, *actual_daily, *diff_daily])

summary_df = pd.DataFrame(
    rows,
    columns=[
        "Coin",
        "Pred Price Day+1","Pred Price Day+2","Pred Price Day+3",
        "Pred Cap Day+1","Pred Cap Day+2","Pred Cap Day+3",
        "Pred Vol Day+1","Pred Vol Day+2","Pred Vol Day+3",
        "Actual Price Day+1","Actual Price Day+2","Actual Price Day+3",
        "Diff Price Day+1","Diff Price Day+2","Diff Price Day+3"
    ]
)
st.dataframe(summary_df, use_container_width=True)

st.subheader("Detailed Prediction vs. Actual (Hourly)")
col1, col2 = st.columns([1, 2])

with col1:
    selected = st.selectbox("Select a coin", COINS)

with col2:
    preds, caps, vols = forecast_and_unscale(
        models[selected],
        scalers[selected],
        get_last_sequence(selected),
        steps=72
    )
    actual = fetch_real_prices(selected, days=3)
    # Build DataFrame for hourly comparison
    if len(actual) == 72:
        # Percentage difference: (pred - actual) / actual * 100
        diff = np.where(actual != 0, (preds - actual) / actual * 100, np.nan)
    else:
        diff = [np.nan]*72
    comp_df = pd.DataFrame({
        "Predicted Price": preds,
        "Predicted Cap": caps,
        "Predicted Volume": vols,
        "Actual Price": actual if len(actual) == 72 else [np.nan]*72,
    })
    diff_df = pd.DataFrame({
        "Diff Price (%)": diff
    })
    st.line_chart(comp_df, use_container_width=True)
    st.markdown("**Percentage Difference (Price, Predicted vs. Actual)**")
    st.line_chart(diff_df, use_container_width=True)
