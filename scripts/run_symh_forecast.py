import io
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

from symh_models import TiDE

MODEL_PATH = Path("model/symh_model_best.pth")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SOLAR_WIND_HISTORY = DATA_DIR / "symh_solarwind_30min_history.csv"
SYM_H_LOCAL = DATA_DIR / "symh_observed.csv"
OUTPUT_JSON = DATA_DIR / "symh_forecast.json"
OUTPUT_INPUT = DATA_DIR / "symh_input_latest_12hours.csv"

URLS = {
    "mag": "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json",
    "wind": "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json",
}

FEATURES = ["Bmag", "By", "Bz", "V", "N", "P", "Ey", "sym-h"]
MIN_VAL = np.array([
    8.91666667e-01, -4.03233333e01, -5.13550000e01, 2.56780000e02,
    1.60000000e-01, 4.00000000e-02, -3.09383333e01, -4.75333333e02,
], dtype=np.float32)
MAX_VAL = np.array([
    68.475, 26.88833333, 38.43833333, 1089.08,
    64.945, 65.4, 31.79166667, 106.0,
], dtype=np.float32)

CADENCE = "30min"
INPUT_BINS = 24
FORECAST_BINS = 24
TRIGGER_BZ = -3.0
TRIGGER_BINS = 12  # 6 hours at 30-minute cadence


def fetch_json(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def parse_bool(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def rtsw_to_df(data):
    df = pd.DataFrame(data)
    if df.empty or "time_tag" not in df.columns:
        raise ValueError("Unexpected NOAA RTSW JSON format")

    df["time"] = pd.to_datetime(df["time_tag"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])

    if "active" in df.columns:
        active = parse_bool(df["active"])
        if active.any():
            df = df[active]

    return df.sort_values("time")


def load_current_solar_wind():
    mag = rtsw_to_df(fetch_json(URLS["mag"]))
    wind = rtsw_to_df(fetch_json(URLS["wind"]))

    mag_map = {"bt": "Bmag", "by_gsm": "By", "bz_gsm": "Bz"}
    wind_map = {
        "proton_speed": "V",
        "proton_density": "N",
        "speed": "V",
        "density": "N",
    }
    mag = mag.rename(columns=mag_map)
    wind = wind.rename(columns=wind_map)

    for col in ["Bmag", "By", "Bz"]:
        mag[col] = pd.to_numeric(mag.get(col), errors="coerce")
    for col in ["V", "N"]:
        wind[col] = pd.to_numeric(wind.get(col), errors="coerce")

    mag30 = (
        mag.set_index("time")[["Bmag", "By", "Bz"]]
        .resample(CADENCE, label="left", closed="left")
        .mean()
    )
    wind30 = (
        wind.set_index("time")[["V", "N"]]
        .resample(CADENCE, label="left", closed="left")
        .mean()
    )

    df = mag30.join(wind30, how="outer").reset_index()
    df["P"] = 1.6726e-6 * df["N"] * np.square(df["V"])
    df["Ey"] = -df["V"] * df["Bz"] * 1e-3
    return df


def update_solar_wind_history(new_df):
    if SOLAR_WIND_HISTORY.exists():
        old = pd.read_csv(SOLAR_WIND_HISTORY)
        old["time"] = pd.to_datetime(old["time"], utc=True, errors="coerce")
    else:
        old = pd.DataFrame(columns=new_df.columns)

    merged = pd.concat([old, new_df], ignore_index=True)
    merged["time"] = pd.to_datetime(merged["time"], utc=True, errors="coerce")
    merged = merged.dropna(subset=["time"])
    merged = merged.sort_values("time").drop_duplicates("time", keep="last")

    cutoff = datetime.now(timezone.utc) - timedelta(days=4)
    merged = merged[merged["time"] >= cutoff]
    merged.to_csv(SOLAR_WIND_HISTORY, index=False)
    return merged


def load_symh_observations():
    """
    Load observed SYM-H from either:
      1. SYM_H_URL environment variable (CSV or JSON), or
      2. data/symh_observed.csv in the repository.

    Required columns: time and sym-h (SYM-H/symh are also accepted).
    The source must provide near-real-time observed SYM-H; Dst is not substituted.
    """
    url = os.environ.get("SYM_H_URL", "").strip()

    if url:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "json" in content_type or url.lower().endswith(".json"):
            raw = response.json()
            if isinstance(raw, dict):
                raw = raw.get("data", raw.get("records", raw))
            df = pd.DataFrame(raw)
        else:
            df = pd.read_csv(io.StringIO(response.text))
    elif SYM_H_LOCAL.exists():
        df = pd.read_csv(SYM_H_LOCAL)
    else:
        return pd.DataFrame(columns=["time", "sym-h"])

    lower_map = {str(c).lower().strip(): c for c in df.columns}
    time_col = next((lower_map[x] for x in ["time", "time_tag", "datetime", "date"] if x in lower_map), None)
    value_col = next((lower_map[x] for x in ["sym-h", "sym_h", "symh"] if x in lower_map), None)
    if time_col is None or value_col is None:
        raise ValueError("SYM-H source must contain time and sym-h columns")

    out = pd.DataFrame({
        "time": pd.to_datetime(df[time_col], utc=True, errors="coerce"),
        "sym-h": pd.to_numeric(df[value_col], errors="coerce"),
    }).dropna()

    return (
        out.set_index("time")[["sym-h"]]
        .resample(CADENCE, label="left", closed="left")
        .mean()
        .reset_index()
    )


def latest_complete_bin():
    now = pd.Timestamp.now(tz="UTC")
    return now.floor(CADENCE) - pd.Timedelta(CADENCE)


def build_model_input():
    sw = update_solar_wind_history(load_current_solar_wind())
    symh = load_symh_observations()
    end = latest_complete_bin()
    start = end - pd.Timedelta(minutes=30 * (INPUT_BINS - 1))
    grid = pd.DataFrame({"time": pd.date_range(start, end, freq=CADENCE, tz="UTC")})

    df = grid.merge(sw, on="time", how="left").merge(symh, on="time", how="left")
    input_cols = FEATURES

    availability = {col: int(df[col].notna().sum()) if col in df else 0 for col in input_cols}
    missing_cols = [col for col in input_cols if col not in df or df[col].notna().sum() == 0]

    if missing_cols:
        return None, end, availability, f"Missing data source: {', '.join(missing_cols)}"

    # Interpolate only short internal gaps; do not fabricate an entirely absent variable.
    df[input_cols] = df[input_cols].interpolate(limit=2, limit_direction="both").ffill().bfill()
    if df[input_cols].isna().any().any():
        return None, end, availability, "Recent input contains unresolved missing values."

    return df, end, availability, None


def trigger_is_active(df):
    recent = df["Bz"].iloc[-TRIGGER_BINS:]
    return len(recent) == TRIGGER_BINS and bool((recent <= TRIGGER_BZ).all())


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    args = checkpoint.get("args") if isinstance(checkpoint, dict) else None

    model = TiDE(
        inp_len=getattr(args, "inp_len", 24),
        horizon=getattr(args, "horizon", 24),
        inp_dim=getattr(args, "inp_dim", 8),
        mlp_hidden=getattr(args, "mlp_hidden", 256),
        n_blocks=getattr(args, "n_blocks", 4),
        dropout=getattr(args, "dropout", 0.5),
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(df):
    values = df[FEATURES].to_numpy(dtype=np.float32)
    diff = MAX_VAL - MIN_VAL
    diff[diff == 0] = 1e-8
    normalized = (values - MIN_VAL) / diff
    x = torch.from_numpy(normalized).unsqueeze(0)

    with torch.no_grad():
        pred_norm = load_model()(x).cpu().numpy().reshape(-1)

    symh_min = MIN_VAL[-1]
    symh_max = MAX_VAL[-1]
    pred = pred_norm * (symh_max - symh_min) + symh_min
    return pred_norm, pred


def save_payload(status, base_time, availability, df=None, pred_norm=None, pred=None, message=""):
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_base_time_utc": base_time.isoformat(),
        "status": status,
        "message": message,
        "trigger": {
            "definition": "Bz <= -3 nT continuously for the latest 6 hours",
            "threshold_nt": TRIGGER_BZ,
            "required_bins": TRIGGER_BINS,
            "cadence": CADENCE,
        },
        "data_availability": availability,
        "history": [],
        "forecast": [],
    }

    if df is not None:
        payload["history"] = [
            {"time": row["time"].isoformat(), **{c: float(row[c]) for c in FEATURES}}
            for _, row in df.iterrows()
        ]
        df.to_csv(OUTPUT_INPUT, index=False)

    if pred is not None:
        times = [base_time + timedelta(minutes=30 * (i + 1)) for i in range(len(pred))]
        payload["forecast"] = [
            {"time": t.isoformat(), "sym_h_pred_norm": float(a), "sym_h_pred": float(b)}
            for t, a, b in zip(times, pred_norm, pred)
        ]

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    print("[INFO] Loading real-time solar-wind and SYM-H observations...")
    df, base_time, availability, error = build_model_input()

    if error:
        save_payload(
            "waiting_for_data", base_time, availability,
            message=error + " Configure SYM_H_URL or update data/symh_observed.csv."
        )
        print(f"[WAIT] {error}")
        return

    if not trigger_is_active(df):
        recent_min = float(df["Bz"].iloc[-TRIGGER_BINS:].min())
        save_payload(
            "standby", base_time, availability, df=df,
            message=(
                "Forecast is on standby because Bz has not remained <= -3 nT "
                f"for the latest 6 hours. Recent 6-hour minimum Bz: {recent_min:.2f} nT."
            ),
        )
        print("[STANDBY] Southward-Bz trigger is not active.")
        return

    pred_norm, pred = predict(df)
    save_payload(
        "forecast_active", base_time, availability, df=df,
        pred_norm=pred_norm, pred=pred,
        message="Strong southward-Bz trigger is active; 12-hour SYM-H forecast generated.",
    )
    print("[DONE] SYM-H forecast generated.")


if __name__ == "__main__":
    main()
