import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from datetime import timedelta

np.random.seed(42)

# ============================================================
# CONFIG
# ============================================================
PAST_STEPS = 30
FORECAST_DAYS = 7

MODEL_PATH = "models/gru_daily_load.h5"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH = "telangana_demand_weather_final.csv"

# ============================================================
# LOAD MODEL & SCALER
# ============================================================
gru = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

_default_feature_cols = [
    "demand",
    "temperature",
    "humidity",
    "rain",
    "cloud",
    "windspeed",
    "dayofweek",
    "is_weekend",
    "is_holiday",
]

feature_cols = list(getattr(scaler, "feature_names_in_", _default_feature_cols))
load_feature = "load" if "load" in feature_cols else "demand"
load_idx = feature_cols.index(load_feature)

print("✅ GRU model & scaler loaded")
print("Feature order:", feature_cols)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.sort_values("timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)

# ============================================================
# TIME FEATURES
# ============================================================
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
df["is_holiday"] = 0

# ============================================================
# FEATURE NAME ALIGNMENT
# ============================================================
pairs = [
    ("demand", "load"),
    ("rain", "rainfall"),
    ("cloud", "cloud_cover"),
    ("windspeed", "wind_speed"),
]

rename_map = {}
for a, b in pairs:
    if b in feature_cols and a in df.columns:
        rename_map[a] = b
    elif a in feature_cols and b in df.columns:
        rename_map[b] = a

if rename_map:
    df.rename(columns=rename_map, inplace=True)

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

load_col = load_feature

# ============================================================
# SCALE DATA
# ============================================================
scaled_df = pd.DataFrame(
    scaler.transform(df[feature_cols]),
    columns=feature_cols
)

last_sequence = scaled_df.values[-PAST_STEPS:].reshape(
    1, PAST_STEPS, len(feature_cols)
)

# ============================================================
# STATISTICAL ANCHORS
# ============================================================
BASELINE_LOAD = df[load_col].rolling(30).mean().iloc[-1]
HIST_MIN = df[load_col].quantile(0.05)
HIST_MAX = df[load_col].quantile(0.95)

# ============================================================
# 🔮 NEXT DAY PREDICTION
# ============================================================
def predict_next_day():
    pred_scaled = gru.predict(last_sequence, verbose=0)[0, 0]

    helper = np.zeros((1, len(feature_cols)))
    helper[0, load_idx] = pred_scaled
    pred_load = scaler.inverse_transform(helper)[0, load_idx]

    pred_load = 0.7 * pred_load + 0.3 * BASELINE_LOAD
    pred_load *= 1 + np.random.normal(0, 0.015)
    return float(np.clip(pred_load, HIST_MIN, HIST_MAX))

# ============================================================
# 🔁 NEXT 7 DAYS ROLLING FORECAST
# ============================================================
def predict_next_7_days():
    preds = []
    seq = last_sequence.copy()
    current_date = df["timestamp"].iloc[-1] + timedelta(days=1)

    last_real_load = df[load_col].iloc[-1]

    weather_cols = ["temperature", "humidity", "rain", "cloud", "windspeed"]
    base_weather = df.iloc[-1][weather_cols]

    for i in range(FORECAST_DAYS):

        pred_scaled = gru.predict(seq, verbose=0)[0, 0]
        helper = np.zeros((1, len(feature_cols)))
        helper[0, load_idx] = pred_scaled
        pred_load = scaler.inverse_transform(helper)[0, load_idx]

        # Momentum
        pred_load += 0.4 * (pred_load - last_real_load)

        # Weather drift
        weather = base_weather.copy()
        weather["temperature"] += np.random.normal(0, 1.2)
        weather["humidity"] += np.random.normal(0, 3.5)

        pred_load *= (
            1 + 0.02 * (weather["temperature"] - 30)
        ) * (
            1 + 0.01 * (weather["humidity"] - 60)
        )

        # Mean reversion
        deviation = abs(pred_load - BASELINE_LOAD) / BASELINE_LOAD
        pred_load = (1 - min(deviation, 0.3)) * pred_load + min(deviation, 0.3) * BASELINE_LOAD

        # Weekly seasonality
        pred_load *= 1 + 0.04 * np.sin(2 * np.pi * i / 7)

        # Noise + bounds
        pred_load += np.random.normal(0, 0.012 * pred_load)
        pred_load = np.clip(pred_load, HIST_MIN, HIST_MAX)

        preds.append(float(pred_load))
        last_real_load = pred_load

        # Update sequence
        next_row = seq[0, -1].copy()
        helper[0, load_idx] = pred_load
        next_row[load_idx] = scaler.transform(helper)[0, load_idx]

        for col in weather_cols:
            idx = feature_cols.index(col)
            helper = np.zeros((1, len(feature_cols)))
            helper[0, idx] = weather[col]
            next_row[idx] = scaler.transform(helper)[0, idx]

        dow = current_date.dayofweek
        next_row[feature_cols.index("dayofweek")] = dow
        next_row[feature_cols.index("is_weekend")] = int(dow >= 5)
        next_row[feature_cols.index("is_holiday")] = 0

        seq = np.concatenate([seq[:, 1:, :], next_row.reshape(1, 1, -1)], axis=1)
        current_date += timedelta(days=1)

    return np.array(preds)

# ============================================================
# 📊 BACKTEST ACCURACY
# ============================================================
def get_backtest_accuracy(last_days=7):
    timestamps, actual, predicted = [], [], []

    for i in range(len(df) - last_days, len(df)):
        seq = scaled_df.values[i - PAST_STEPS:i].reshape(1, PAST_STEPS, len(feature_cols))
        pred_scaled = gru.predict(seq, verbose=0)[0, 0]

        helper = np.zeros((1, len(feature_cols)))
        helper[0, load_idx] = pred_scaled
        pred_load = scaler.inverse_transform(helper)[0, load_idx]

        timestamps.append(df["timestamp"].iloc[i].isoformat())
        actual.append(df[load_col].iloc[i])
        predicted.append(pred_load)

    a, p = np.array(actual), np.array(predicted)
    mape = np.mean(np.abs((a - p) / a)) * 100

    return {
        "timestamps": timestamps,
        "actual": actual,
        "predicted": predicted,
        "metrics": {
            "mae": float(np.mean(np.abs(a - p))),
            "rmse": float(np.sqrt(np.mean((a - p) ** 2))),
            "mape": float(mape),
            "accuracy_pct": float(100 - mape),
        },
    }

# ============================================================
# 🌦️ WEATHER IMPACT ANALYSIS
# ============================================================
def get_weather_impact():
    weather_cols = ["temperature", "humidity", "rain", "cloud", "windspeed"]

    scatter, corr = {}, {}
    for col in weather_cols:
        x = df[col].astype(float)
        y = df[load_col].astype(float)
        mask = ~x.isna() & ~y.isna()

        scatter[col] = {
            "x": x[mask].tolist(),
            "y": y[mask].tolist(),
        }

        corr[col] = float(np.corrcoef(x[mask], y[mask])[0, 1])

    return {
        "scatter": scatter,
        "correlation": corr,
        "columns": weather_cols,
        "demand_column": load_col,
    }
