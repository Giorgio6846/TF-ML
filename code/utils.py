import torch
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def priceDataframe(name):
    jsonPath = f"../data/info_{name}.json"
    with open(jsonPath, "r") as f:
        data = json.load(f) 
        
    df = pd.DataFrame({
        "price_timestamp": [item[0] for item in data["prices"]],
        "price": [item[1] for item in data["prices"]],
        "market_caps_timestamp": [item[0] for item in data["market_caps"]],
        "market_caps": [item[1] for item in data["market_caps"]],
        "volume_timestamp": [item[0] for item in data["total_volumes"]],
        "volume": [item[1] for item in data["total_volumes"]]
    })
    
    df["price_timestamp"] = pd.to_datetime(df["price_timestamp"], unit="ms")
    #df["market_caps_timestamp"] = pd.to_datetime(df["market_caps_timestamp"], unit="ms")
    #df["volume_timestamp"] = pd.to_datetime(df["volume_timestamp"], unit="ms")

    df["hour"] = df["price_timestamp"].dt.hour
    #df["market_caps_timestamp_hour"] = df["market_caps_timestamp"].dt.hour
    #df["volume_timestamp_hour"] = df["volume_timestamp"].dt.hour

    df["dayweek"] = df["price_timestamp"].dt.dayofweek
    #df["market_caps_timestamp_dayweek"] = df["market_caps_timestamp"].dt.dayofweek
    #df["volume_timestamp_dayweek"] = df["volume_timestamp"].dt.dayofweek

    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)

    df["dayweek_cos"] = np.cos(2 * np.pi * df["hour"] / 7)
    df["dayweek_sin"] = np.sin(2 * np.pi * df["hour"] / 7)

    df.drop(columns=["price_timestamp", "market_caps_timestamp", "volume_timestamp"], inplace=True)
    df.drop(columns=["hour", "dayweek"], inplace=True)
    
    scaler = MinMaxScaler()
    df[["price", "market_caps", "volume"]] = scaler.fit_transform(df[["price", "market_caps", "volume"]])
    
    return df

def splitData(name, sequence_length=72, prediction_offset=120):
    df = priceDataframe(name)

    feature_cols = ["price", "market_caps", "volume", 
                    "hour_cos", "hour_sin", 
                    "dayweek_cos", "dayweek_sin"]

    data = df[feature_cols].values

    X, y = [], []
    for i in range(len(data) - sequence_length - prediction_offset):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length + prediction_offset - 1])

    X = np.array(X)
    y = np.array(y)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)   
    
    return X_tensor, y_tensor