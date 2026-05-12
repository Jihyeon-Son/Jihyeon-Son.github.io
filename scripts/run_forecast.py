import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tensorflow as tf


# ============================================================
# Paths
# ============================================================

MODEL_JSON = Path("model/eflux_model.json")
MODEL_WEIGHTS = Path("model/eflux_model.h5")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FORECAST_JSON = DATA_DIR / "forecast.json"
OUTPUT_INPUT_CSV = DATA_DIR / "input_latest_3days.csv"
OUTPUT_INPUT_JSON = DATA_DIR / "latest_inputs.json"


# ============================================================
# Data URLs
# ============================================================

URLS = {
    "mag": "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json",
    "plasma": "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json",
    "electron": "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-3-day.json",
    "kp": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "dst": "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/presentmonth/index.html",
}


# ============================================================
# Normalization constants
# ============================================================

NORM = {
    "B": (0.7, 32.5),
    "Bz": (-20.8, 24.0),
    "T": (4069.0, 988971.0),
    "N": (0.1, 137.2),
    "V": (240.0, 775.0),
    "Kp24": (0.0, 141.9),
    "Dst": (-155.0, 43.0),
}


# ============================================================
# Utility
# ============================================================

def utc_now_floor_hour():
    now = datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0)


def fetch_json(url):
    r = requests.get(url, timeout=40)
    r.raise_for_status()
    return r.json()


def fetch_text(url):
    r = requests.get(url, timeout=40)
    r.raise_for_status()
    return r.text


def minmax(x, vmin, vmax):
    x = np.asarray(x, dtype=float)
    return (x - vmin) / (vmax - vmin)


def swpc_table_to_df(data):
    header = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=header)

    time_col = "time_tag" if "time_tag" in df.columns else df.columns[0]
    df["time"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])

    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# Load data
# ============================================================

def load_solar_wind_mag():
    data = fetch_json(URLS["mag"])
    df = swpc_table_to_df(data)

    rename_map = {
        "bt": "B",
        "bz_gsm": "Bz",
    }

    df = df.rename(columns=rename_map)

    needed = ["time", "B", "Bz"]
    return df[[c for c in needed if c in df.columns]]


def load_solar_wind_plasma():
    data = fetch_json(URLS["plasma"])
    df = swpc_table_to_df(data)

    rename_map = {
        "temperature": "T",
        "density": "N",
        "speed": "V",
    }

    df = df.rename(columns=rename_map)

    needed = ["time", "T", "N", "V"]
    return df[[c for c in needed if c in df.columns]]


def load_electron_flux():
    data = fetch_json(URLS["electron"])
    df = pd.DataFrame(data)

    if "time_tag" not in df.columns:
        raise ValueError("electron flux JSON does not contain time_tag")

    df["time"] = pd.to_datetime(df["time_tag"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])

    # Pick 2 MeV channel as flexibly as possible.
    # Common NOAA fields include energy, flux, satellite, time_tag.
    energy_cols = [c for c in df.columns if c.lower() in ["energy", "channel"]]

    if energy_cols:
        energy_col = energy_cols[0]
        e = df[energy_col].astype(str).str.lower()

        mask = (
            e.str.contains("2", regex=False, na=False)
            & (
                e.str.contains("mev", regex=False, na=False)
                | e.str.contains(">=", regex=False, na=False)
                | e.str.contains(">", regex=False, na=False)
            )
        )

        if mask.sum() > 0:
            df = df[mask]

    flux_col = None
    for candidate in ["flux", "electron_flux", "value"]:
        if candidate in df.columns:
            flux_col = candidate
            break

    if flux_col is None:
        numeric_candidates = []
        for c in df.columns:
            if c not in ["time", "time_tag", "satellite", "energy", "channel"]:
                numeric_candidates.append(c)
        if not numeric_candidates:
            raise ValueError("Could not find electron flux column")
        flux_col = numeric_candidates[0]

    df["electron_flux"] = pd.to_numeric(df[flux_col], errors="coerce")
    df = df.dropna(subset=["electron_flux"])

    return df[["time", "electron_flux"]]


def load_kp():
    data = fetch_json(URLS["kp"])
    df = swpc_table_to_df(data)

    kp_col = None
    for candidate in ["kp_index", "estimated_kp", "Kp", "kp"]:
        if candidate in df.columns:
            kp_col = candidate
            break

    if kp_col is None:
        numeric_cols = [c for c in df.columns if c != "time"]
        if not numeric_cols:
            raise ValueError("Could not find Kp column")
        kp_col = numeric_cols[0]

    df["Kp"] = pd.to_numeric(df[kp_col], errors="coerce")
    df = df.dropna(subset=["Kp"])

    return df[["time", "Kp"]]


def parse_kyoto_dst_presentmonth(html, reference_time):

    year = reference_time.year
    month = reference_time.month

    text = re.sub(r"<[^>]+>", " ", html)

    lines = text.splitlines()

    records = []

    for line in lines:

        # DAY line만
        m = re.match(r"^\s*(\d{1,2})\s+", line)

        if not m:
            continue

        day = int(m.group(1))

        # day 이후 문자열
        rest = line[m.end():]

        # fixed width parsing
        # Kyoto Dst는 대략 4-char width
        chunks = []

        width = 4

        for i in range(0, len(rest), width):
            chunk = rest[i:i+width].strip()
            chunks.append(chunk)

        values = []

        for chunk in chunks[:24]:

            if chunk == "":
                values.append(np.nan)
                continue

            # pure missing sentinel
            if chunk == "9999":
                values.append(np.nan)
                continue

            try:
                val = float(chunk)

                # impossible dst
                if abs(val) > 1000:
                    val = np.nan

                values.append(val)

            except Exception:
                values.append(np.nan)

        for hour, val in enumerate(values):

            try:
                t = datetime(
                    year,
                    month,
                    day,
                    hour,
                    tzinfo=timezone.utc
                )

            except ValueError:
                continue

            records.append({
                "time": t,
                "Dst": val
            })

    return pd.DataFrame(records)

def load_dst(reference_time):
    try:
        html = fetch_text(URLS["dst"])
        df = parse_kyoto_dst_presentmonth(html, reference_time)
        if not df.empty:
            return df[["time", "Dst"]]
    except Exception as exc:
        print(f"[WARN] Failed to load Kyoto Dst: {exc}")

    return pd.DataFrame(columns=["time", "Dst"])
    
# ============================================================
# Hourly preprocessing
# ============================================================

def hourly_mean(df, value_cols):
    if df.empty:
        return pd.DataFrame(columns=["time"] + value_cols)

    out = (
        df.sort_values("time")
        .set_index("time")
        .resample("1h")
        .mean(numeric_only=True)
        .reset_index()
    )

    keep = ["time"] + [c for c in value_cols if c in out.columns]
    return out[keep]


def build_hourly_input_dataframe():
    """
    Makes recent 72-hour input dataframe.

    If forecast_time is 2026-05-03 23:00 UTC,
    input range is 2026-05-01 00:00 UTC through 2026-05-03 23:00 UTC.

    For Kp24, we additionally need 23 hours before the model input start,
    so Kp is loaded on a 96-hour grid first.
    """

    forecast_time = utc_now_floor_hour()
    input_start = forecast_time - timedelta(hours=71)
    
    kp_start = input_start - timedelta(hours=23)

    # Kp24 계산을 위해 96시간 grid 생성
    hourly_grid = pd.DataFrame({
        "time": pd.date_range(
            kp_start,
            forecast_time,
            freq="1h",
            tz="UTC",
        )
    })

    mag = hourly_mean(load_solar_wind_mag(), ["B", "Bz"])
    plasma = hourly_mean(load_solar_wind_plasma(), ["T", "N", "V"])
    electron = hourly_mean(load_electron_flux(), ["electron_flux"])
    kp = hourly_mean(load_kp(), ["Kp"])
    dst = hourly_mean(load_dst(forecast_time), ["Dst"])

    df = hourly_grid.copy()

    for source in [mag, plasma, electron, kp, dst]:
        if source.empty:
            continue
        df = pd.merge(df, source, on="time", how="left")

    required_cols = ["B", "Bz", "T", "N", "V", "electron_flux", "Kp", "Dst"]

    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Missing-data handling
    df[required_cols] = df[required_cols].interpolate(limit_direction="both")
    df[required_cols] = df[required_cols].ffill().bfill()

    # If a whole column is still NaN, use conservative fallback.
    fallback = {
        "B": 5.0,
        "Bz": 0.0,
        "T": 100000.0,
        "N": 5.0,
        "V": 400.0,
        "electron_flux": 1e4,
        "Kp": 1.0,
        "Dst": 0.0,
    }

    for col, value in fallback.items():
        df[col] = df[col].fillna(value)

    # Electron flux must be positive before log10
    df["electron_flux"] = df["electron_flux"].clip(lower=1e-30)

    # Kp rolling 24-hour sum.
    # At each hourly timestamp, Kp24 means sum over previous 24 hourly Kp values including current hour.
    df["Kp24"] = (
        df["Kp"]
        .rolling(window=24, min_periods=24)
        .sum()
    )
    df["Kp24"] = df["Kp24"].interpolate(limit_direction="both").ffill().bfill()
    df_model = df[df["time"] >= input_start].copy()
    
    if len(df_model) != 72:
        raise ValueError(f"Expected 72 rows for model input, got {len(df_model)}")
        
    return df_model, forecast_time


# ============================================================
# Model handling
# ============================================================

def load_model():
    with open(MODEL_JSON, "r", encoding="utf-8") as f:
        model_json = f.read()

    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(str(MODEL_WEIGHTS))

    return model


def make_model_inputs(df):
    """
    Model input assumed from your saved model traceback:

    input_1: 504 = 72 hours x 7 variables
             [B, Bz, T, N, V, Kp24, Dst], flattened

    input_2: 72 = electron flux history
             log10(electron_flux) / 7
    """

    if len(df) != 72:
        raise ValueError(f"Input dataframe must have 72 rows, got {len(df)}")

    B = minmax(df["B"].values, *NORM["B"])
    Bz = minmax(df["Bz"].values, *NORM["Bz"])
    T = minmax(df["T"].values, *NORM["T"])
    N = minmax(df["N"].values, *NORM["N"])
    V = minmax(df["V"].values, *NORM["V"])
    Kp24 = minmax(df["Kp24"].values, *NORM["Kp24"])
    Dst = minmax(df["Dst"].values, *NORM["Dst"])

    features = np.stack(
        [B, Bz, T, N, V, Kp24, Dst],
        axis=1,
    ).astype(np.float32)

    x_features = features.reshape(1, 504)

    raw_flux = df["electron_flux"].astype(float).values
    raw_flux = np.clip(raw_flux, 4, None)
    log_flux = np.log10(raw_flux)
    x_eflux = (log_flux / 7.0).astype(np.float32)
    x_eflux = x_eflux.reshape(1, 72)

    return [x_features, x_eflux]


def run_prediction(model, model_inputs):
    pred = model.predict(model_inputs, verbose=0)

    if isinstance(pred, list):
        pred = pred[0]

    pred = np.asarray(pred).reshape(-1)

    # Model output is log10(flux) / 7
    pred_flux = 10 ** (pred * 7.0)

    return pred, pred_flux


# ============================================================
# Save outputs
# ============================================================

def save_latest_inputs(df, forecast_time):
    df.to_csv(OUTPUT_INPUT_CSV, index=False)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_base_time_utc": forecast_time.isoformat(),
        "cadence": "1 hour",
        "columns": list(df.columns),
        "data": [
            {
                c: (
                    row[c].isoformat()
                    if c == "time"
                    else None if pd.isna(row[c])
                    else float(row[c])
                )
                for c in df.columns
            }
            for _, row in df.iterrows()
        ],
    }

    with open(OUTPUT_INPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_forecast_json(df, forecast_time, pred_norm, pred_flux):
    forecast_times = [
        forecast_time + timedelta(hours=i + 1)
        for i in range(len(pred_flux))
    ]

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_base_time_utc": forecast_time.isoformat(),
        "input_start_utc": df["time"].iloc[0].isoformat(),
        "input_end_utc": df["time"].iloc[-1].isoformat(),
        "target_start_utc": forecast_times[0].isoformat(),
        "target_end_utc": forecast_times[-1].isoformat(),
        "status": "Forecast is current.",
        "model_note": "Input uses 72 hourly values. Electron flux input is log10(flux)/7. Kp input is rolling 24-hour sum. Output is restored by 10**(prediction*7).",
        "history": [
            {
                "time": row["time"].isoformat(),
                "electron_flux": float(row["electron_flux"]),
                "B": float(row["B"]),
                "Bz": float(row["Bz"]),
                "T": float(row["T"]),
                "N": float(row["N"]),
                "V": float(row["V"]),
                "Kp": float(row["Kp"]),
                "Kp24": float(row["Kp24"]),
                "Dst": float(row["Dst"]),
            }
            for _, row in df.iterrows()
        ],
        "forecast": [
            {
                "time": t.isoformat(),
                "electron_flux_pred_norm": float(y_norm),
                "electron_flux_pred": float(y_flux),
            }
            for t, y_norm, y_flux in zip(forecast_times, pred_norm, pred_flux)
        ],
    }

    with open(OUTPUT_FORECAST_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ============================================================
# Main
# ============================================================

def main():
    print("[INFO] Building latest 72-hour input dataframe...")
    df, forecast_time = build_hourly_input_dataframe()

    print("[INFO] Saving latest input files...")
    save_latest_inputs(df, forecast_time)

    print("[INFO] Loading model...")
    model = load_model()

    print("[INFO] Making model inputs...")
    model_inputs = make_model_inputs(df)

    print("[INFO] Running prediction...")
    pred_norm, pred_flux = run_prediction(model, model_inputs)

    print("[INFO] Saving forecast JSON...")
    save_forecast_json(df, forecast_time, pred_norm, pred_flux)

    print("[DONE]")
    print(f"Forecast base time: {forecast_time.isoformat()}")
    print(f"Input rows: {len(df)}")
    print(f"Forecast length: {len(pred_flux)}")
    print(df.tail())
    print(pred_flux[:5])


if __name__ == "__main__":
    main()
