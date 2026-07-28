import json
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


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
SOLAR_WIND_HISTORY_CSV = DATA_DIR / "solar_wind_hourly_history.csv"


# ============================================================
# Data URLs
# ============================================================

URLS = {
    "mag": "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json",
    "wind": "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json",
    "electron": "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-3-day.json",
    "kp": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "dst": "https://services.swpc.noaa.gov/products/kyoto-dst.json",
}


# ============================================================
# Configuration
# ============================================================

MODEL_INPUT_HOURS = 72
MAX_INTERPOLATED_SOLAR_WIND_GAP_HOURS = 3
REQUEST_TIMEOUT_SECONDS = 40
REQUEST_ATTEMPTS = 3

SOLAR_WIND_COLS = ["B", "Bz", "T", "N", "V"]


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
# HTTP session
# ============================================================

HTTP = requests.Session()
HTTP.headers.update(
    {
        "User-Agent": (
            "Jihyeon-Son.github.io forecast dashboard "
            "(https://github.com/Jihyeon-Son/Jihyeon-Son.github.io)"
        )
    }
)


# ============================================================
# Utility
# ============================================================


def utc_now_floor_hour():
    """Return the last fully completed UTC hourly bin."""
    now = datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)


def fetch_json(url):
    last_error = None

    for attempt in range(1, REQUEST_ATTEMPTS + 1):
        try:
            response = HTTP.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            print(
                f"[WARN] Request failed ({attempt}/{REQUEST_ATTEMPTS}) "
                f"for {url}: {exc}"
            )
            if attempt < REQUEST_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"Failed to fetch JSON from {url}: {last_error}")


def minmax(x, vmin, vmax):
    x = np.asarray(x, dtype=float)
    return (x - vmin) / (vmax - vmin)


def atomic_write_json(path, payload):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    tmp_path.replace(path)


def atomic_write_csv(path, df):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def swpc_payload_to_df(data):
    """Parse both SWPC list-of-objects and legacy table-style JSON."""
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("SWPC response is not a non-empty JSON list")

    if isinstance(data[0], dict):
        df = pd.DataFrame(data)
    elif isinstance(data[0], list):
        header = data[0]
        rows = data[1:]
        df = pd.DataFrame(rows, columns=header)
    else:
        raise ValueError("Unsupported SWPC JSON structure")

    if df.empty:
        raise ValueError("SWPC response contains no rows")

    time_col = "time_tag" if "time_tag" in df.columns else df.columns[0]
    df["time"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).copy()

    if df.empty:
        raise ValueError("SWPC response contains no valid timestamps")

    return df


def first_existing_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def numeric_series(df, column):
    values = pd.to_numeric(df[column], errors="coerce")
    return values.mask(values <= -9990)


def parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def select_active_rtsw_rows(df, label):
    """Match the old products by retaining the spacecraft active at each time."""
    out = df.copy()

    if "active" in out.columns:
        active_mask = out["active"].map(parse_bool)
        if active_mask.any():
            out = out.loc[active_mask].copy()
        else:
            print(f"[WARN] {label}: no rows are marked active; using all rows")

    sort_columns = ["time"]
    ascending = [True]

    if "overall_quality" in out.columns:
        out["_quality"] = numeric_series(out, "overall_quality")
        sort_columns.append("_quality")
        ascending.append(True)

    if "sample_size" in out.columns:
        out["_sample_size"] = numeric_series(out, "sample_size")
        sort_columns.append("_sample_size")
        ascending.append(False)

    out = (
        out.sort_values(sort_columns, ascending=ascending, na_position="last")
        .drop_duplicates(subset=["time"], keep="first")
        .sort_values("time")
    )

    if "source" in out.columns:
        sources = sorted(out["source"].dropna().astype(str).unique().tolist())
        if sources:
            print(f"[INFO] {label} active source(s): {', '.join(sources)}")

    return out


def longest_false_run(mask):
    longest = 0
    current = 0

    for value in mask:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)

    return longest


# ============================================================
# Load data
# ============================================================


def load_solar_wind_mag():
    df = swpc_payload_to_df(fetch_json(URLS["mag"]))
    df = select_active_rtsw_rows(df, "RTSW magnetometer")

    b_col = first_existing_column(df, ["bt", "B"])
    bz_col = first_existing_column(df, ["bz_gsm", "Bz"])

    if b_col is None or bz_col is None:
        raise ValueError(
            "RTSW magnetometer JSON does not contain both bt and bz_gsm"
        )

    out = pd.DataFrame(
        {
            "time": df["time"],
            "B": numeric_series(df, b_col),
            "Bz": numeric_series(df, bz_col),
        }
    )

    return out.dropna(subset=["B", "Bz"], how="all")


def load_solar_wind_wind():
    df = swpc_payload_to_df(fetch_json(URLS["wind"]))
    df = select_active_rtsw_rows(df, "RTSW solar wind")

    temperature_col = first_existing_column(
        df, ["proton_temperature", "temperature", "T"]
    )
    density_col = first_existing_column(df, ["proton_density", "density", "N"])
    speed_col = first_existing_column(df, ["proton_speed", "speed", "V"])

    if temperature_col is None or density_col is None or speed_col is None:
        raise ValueError(
            "RTSW wind JSON does not contain proton_temperature, "
            "proton_density, and proton_speed"
        )

    out = pd.DataFrame(
        {
            "time": df["time"],
            "T": numeric_series(df, temperature_col),
            "N": numeric_series(df, density_col),
            "V": numeric_series(df, speed_col),
        }
    )

    return out.dropna(subset=["T", "N", "V"], how="all")


def load_electron_flux():
    data = fetch_json(URLS["electron"])
    df = swpc_payload_to_df(data)

    energy_col = first_existing_column(df, ["energy", "channel"])
    if energy_col is not None:
        energy = df[energy_col].astype(str).str.lower()
        mask = energy.str.contains("2", regex=False, na=False) & energy.str.contains(
            "mev", regex=False, na=False
        )
        if mask.any():
            df = df.loc[mask].copy()

    flux_col = first_existing_column(df, ["flux", "electron_flux", "value"])

    if flux_col is None:
        excluded = {
            "time",
            "time_tag",
            "satellite",
            "energy",
            "channel",
            "active",
            "source",
        }
        numeric_candidates = []
        for column in df.columns:
            if column in excluded:
                continue
            converted = pd.to_numeric(df[column], errors="coerce")
            if converted.notna().any():
                numeric_candidates.append(column)

        if not numeric_candidates:
            raise ValueError("Could not find electron flux column")
        flux_col = numeric_candidates[0]

    df["electron_flux"] = pd.to_numeric(df[flux_col], errors="coerce")
    df = df.dropna(subset=["electron_flux"])

    return (
        df[["time", "electron_flux"]]
        .groupby("time", as_index=False)
        .mean(numeric_only=True)
        .sort_values("time")
    )


def load_kp():
    df = swpc_payload_to_df(fetch_json(URLS["kp"]))

    kp_col = first_existing_column(
        df, ["Kp", "kp", "kp_index", "estimated_kp"]
    )
    if kp_col is None:
        raise ValueError("Could not find Kp column")

    df["Kp"] = pd.to_numeric(df[kp_col], errors="coerce")
    df = df.dropna(subset=["Kp"])

    return (
        df[["time", "Kp"]]
        .drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")
    )


def load_dst():
    df = swpc_payload_to_df(fetch_json(URLS["dst"]))

    dst_col = first_existing_column(df, ["dst", "Dst"])
    if dst_col is None:
        raise ValueError("Could not find Dst column")

    df["Dst"] = pd.to_numeric(df[dst_col], errors="coerce")
    df.loc[df["Dst"].abs() > 1000, "Dst"] = np.nan
    df = df.dropna(subset=["Dst"])

    return (
        df[["time", "Dst"]]
        .drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")
    )


# ============================================================
# Solar-wind history
# ============================================================


def hourly_mean(df, value_cols):
    present_cols = [column for column in value_cols if column in df.columns]

    if df.empty or not present_cols:
        return pd.DataFrame(columns=["time"] + value_cols)

    out = (
        df.sort_values("time")
        .set_index("time")[present_cols]
        .resample("1h")
        .mean()
        .reset_index()
    )

    out = out.dropna(subset=present_cols, how="all")
    return out[["time"] + present_cols]


def build_new_solar_wind_hourly(wall_forecast_time):
    mag = hourly_mean(load_solar_wind_mag(), ["B", "Bz"])
    wind = hourly_mean(load_solar_wind_wind(), ["T", "N", "V"])

    new_data = pd.merge(mag, wind, on="time", how="outer")

    for column in SOLAR_WIND_COLS:
        if column not in new_data.columns:
            new_data[column] = np.nan
        new_data[column] = pd.to_numeric(new_data[column], errors="coerce")

    new_data = new_data[new_data["time"] <= wall_forecast_time].copy()
    new_data = new_data.dropna(subset=SOLAR_WIND_COLS, how="any")
    new_data = (
        new_data[["time"] + SOLAR_WIND_COLS]
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
    )

    if new_data.empty:
        raise RuntimeError(
            "The new RTSW files did not contain a complete hourly solar-wind bin"
        )

    return new_data


def read_solar_wind_history():
    if not SOLAR_WIND_HISTORY_CSV.exists():
        return pd.DataFrame(columns=["time"] + SOLAR_WIND_COLS)

    try:
        history = pd.read_csv(SOLAR_WIND_HISTORY_CSV)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["time"] + SOLAR_WIND_COLS)

    if "time" not in history.columns:
        print("[WARN] Existing solar-wind history has no time column; resetting it")
        return pd.DataFrame(columns=["time"] + SOLAR_WIND_COLS)

    history["time"] = pd.to_datetime(history["time"], utc=True, errors="coerce")
    history = history.dropna(subset=["time"]).copy()

    for column in SOLAR_WIND_COLS:
        if column not in history.columns:
            history[column] = np.nan
        history[column] = pd.to_numeric(history[column], errors="coerce")

    return history[["time"] + SOLAR_WIND_COLS]


def update_solar_wind_history(new_data, wall_forecast_time):
    old_data = read_solar_wind_history()

    old_data = old_data.copy()
    new_data = new_data.copy()
    old_data["_priority"] = 0
    new_data["_priority"] = 1

    frames = [frame for frame in [old_data, new_data] if not frame.empty]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["time"] <= wall_forecast_time].copy()

    # Prefer newly downloaded values for an hour while retaining old history.
    combined = (
        combined.sort_values(["time", "_priority"], kind="stable")
        .groupby("time", as_index=False)[SOLAR_WIND_COLS]
        .last()
    )
    combined = combined.dropna(subset=SOLAR_WIND_COLS, how="any")

    if combined.empty:
        raise RuntimeError("Solar-wind history is empty after merging new data")

    forecast_time = combined["time"].max()
    retention_start = forecast_time - timedelta(hours=MODEL_INPUT_HOURS - 1)

    combined = combined[
        (combined["time"] >= retention_start)
        & (combined["time"] <= forecast_time)
    ].copy()
    combined = combined.sort_values("time").reset_index(drop=True)

    atomic_write_csv(SOLAR_WIND_HISTORY_CSV, combined)

    return combined, forecast_time


def prepare_solar_wind_window(history, forecast_time):
    input_start = forecast_time - timedelta(hours=MODEL_INPUT_HOURS - 1)
    grid = pd.DataFrame(
        {
            "time": pd.date_range(
                start=input_start,
                end=forecast_time,
                freq="1h",
            )
        }
    )

    window = pd.merge(grid, history, on="time", how="left")
    complete_mask = window[SOLAR_WIND_COLS].notna().all(axis=1)

    available_hours = int(complete_mask.sum())
    max_missing_run = longest_false_run(complete_mask.tolist())
    edges_present = bool(complete_mask.iloc[0] and complete_mask.iloc[-1])

    ready = (
        edges_present
        and max_missing_run <= MAX_INTERPOLATED_SOLAR_WIND_GAP_HOURS
    )

    if ready:
        window[SOLAR_WIND_COLS] = window[SOLAR_WIND_COLS].interpolate(
            method="linear",
            limit_direction="both",
        )
        ready = not window[SOLAR_WIND_COLS].isna().any().any()

    metrics = {
        "ready": bool(ready),
        "available_hours": available_hours,
        "required_hours": MODEL_INPUT_HOURS,
        "max_missing_run_hours": int(max_missing_run),
        "history_start_utc": (
            history["time"].min().isoformat() if not history.empty else None
        ),
        "history_end_utc": (
            history["time"].max().isoformat() if not history.empty else None
        ),
    }

    return window, metrics


def collect_solar_wind_history():
    wall_forecast_time = utc_now_floor_hour()

    print(f"[INFO] Last completed wall-clock hour: {wall_forecast_time.isoformat()}")
    print("[INFO] Fetching current NOAA RTSW one-day files...")

    new_data = build_new_solar_wind_hourly(wall_forecast_time)
    print(
        "[INFO] Complete hourly bins in current RTSW download: "
        f"{len(new_data)}"
    )

    history, forecast_time = update_solar_wind_history(
        new_data,
        wall_forecast_time,
    )
    window, metrics = prepare_solar_wind_window(history, forecast_time)

    print(
        "[INFO] Solar-wind history coverage: "
        f"{metrics['available_hours']}/{metrics['required_hours']} hours"
    )
    print(f"[INFO] History file: {SOLAR_WIND_HISTORY_CSV}")

    return window, forecast_time, metrics


# ============================================================
# Hourly model input preprocessing
# ============================================================


def build_hourly_input_dataframe(solar_wind_window, forecast_time):
    input_start = forecast_time - timedelta(hours=MODEL_INPUT_HOURS - 1)
    kp_start = input_start - timedelta(hours=23)

    hourly_grid = pd.DataFrame(
        {
            "time": pd.date_range(
                start=kp_start,
                end=forecast_time,
                freq="1h",
            )
        }
    )

    electron = hourly_mean(load_electron_flux(), ["electron_flux"])
    kp = hourly_mean(load_kp(), ["Kp"])
    dst = hourly_mean(load_dst(), ["Dst"])

    df = hourly_grid.copy()

    for source in [solar_wind_window, electron, kp, dst]:
        if source.empty:
            continue
        df = pd.merge(df, source, on="time", how="left")

    required_cols = [
        "B",
        "Bz",
        "T",
        "N",
        "V",
        "electron_flux",
        "Kp",
        "Dst",
    ]

    for column in required_cols:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Preserve the original missing-data policy for non-RTSW products.
    df[required_cols] = df[required_cols].interpolate(limit_direction="both")
    df[required_cols] = df[required_cols].ffill().bfill()

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

    for column, value in fallback.items():
        df[column] = df[column].fillna(value)

    df["electron_flux"] = df["electron_flux"].clip(lower=1e-30)

    df["Kp24"] = df["Kp"].rolling(window=24, min_periods=24).sum()
    df["Kp24"] = df["Kp24"].interpolate(limit_direction="both").ffill().bfill()

    df_model = df[
        (df["time"] >= input_start) & (df["time"] <= forecast_time)
    ].copy()
    df_model = df_model.reset_index(drop=True)

    if len(df_model) != MODEL_INPUT_HOURS:
        raise ValueError(
            f"Expected {MODEL_INPUT_HOURS} rows for model input, "
            f"got {len(df_model)}"
        )

    numeric_check_cols = required_cols + ["Kp24"]
    if not np.isfinite(df_model[numeric_check_cols].to_numpy(dtype=float)).all():
        raise ValueError("Model input contains non-finite values")

    return df_model


# ============================================================
# Model handling
# ============================================================


def load_model():
    # TensorFlow is imported only when 72 hours have accumulated.
    import tensorflow as tf

    with open(MODEL_JSON, "r", encoding="utf-8") as f:
        model_json = f.read()

    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(str(MODEL_WEIGHTS))

    return model


def make_model_inputs(df):
    if len(df) != MODEL_INPUT_HOURS:
        raise ValueError(
            f"Input dataframe must have {MODEL_INPUT_HOURS} rows, got {len(df)}"
        )

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
    x_eflux = (log_flux / 7.0).astype(np.float32).reshape(1, 72)

    return [x_features, x_eflux]


def run_prediction(model, model_inputs):
    pred = model.predict(model_inputs, verbose=0)

    if isinstance(pred, list):
        pred = pred[0]

    pred = np.asarray(pred).reshape(-1)
    pred_flux = 10 ** (pred * 7.0)

    return pred, pred_flux


# ============================================================
# Save outputs
# ============================================================


def save_latest_inputs(df, forecast_time):
    atomic_write_csv(OUTPUT_INPUT_CSV, df)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_base_time_utc": forecast_time.isoformat(),
        "cadence": "1 hour",
        "columns": list(df.columns),
        "data": [
            {
                column: (
                    row[column].isoformat()
                    if column == "time"
                    else None
                    if pd.isna(row[column])
                    else float(row[column])
                )
                for column in df.columns
            }
            for _, row in df.iterrows()
        ],
    }

    atomic_write_json(OUTPUT_INPUT_JSON, payload)


def save_forecast_json(df, forecast_time, pred_norm, pred_flux):
    forecast_times = [
        forecast_time + timedelta(hours=i + 1) for i in range(len(pred_flux))
    ]

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_base_time_utc": forecast_time.isoformat(),
        "input_start_utc": df["time"].iloc[0].isoformat(),
        "input_end_utc": df["time"].iloc[-1].isoformat(),
        "target_start_utc": forecast_times[0].isoformat(),
        "target_end_utc": forecast_times[-1].isoformat(),
        "status": "Forecast is current.",
        "model_note": (
            "Input uses 72 hourly values. Electron flux input is "
            "log10(flux)/7. Kp input is a rolling 24-hour sum. "
            "Output is restored by 10**(prediction*7). Solar-wind "
            "history is retained locally from NOAA RTSW one-day files."
        ),
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
                "time": forecast_time_item.isoformat(),
                "electron_flux_pred_norm": float(y_norm),
                "electron_flux_pred": float(y_flux),
            }
            for forecast_time_item, y_norm, y_flux in zip(
                forecast_times,
                pred_norm,
                pred_flux,
            )
        ],
    }

    atomic_write_json(OUTPUT_FORECAST_JSON, payload)


def save_waiting_forecast_json(forecast_time, metrics):
    available = metrics["available_hours"]
    required = metrics["required_hours"]

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_base_time_utc": forecast_time.isoformat(),
        "input_start_utc": None,
        "input_end_utc": metrics["history_end_utc"],
        "target_start_utc": None,
        "target_end_utc": None,
        "status": (
            "Collecting NOAA RTSW solar-wind history: "
            f"{available}/{required} complete hourly bins available. "
            "Forecast will resume automatically."
        ),
        "model_note": (
            "NOAA retired the legacy 3-day and 7-day solar-wind files. "
            "This dashboard now retains hourly values from the RTSW "
            "one-day files until a 72-hour input window is available."
        ),
        "solar_wind_history": metrics,
        "history": [],
        "forecast": [],
    }

    atomic_write_json(OUTPUT_FORECAST_JSON, payload)


def save_error_forecast_json(exc):
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_base_time_utc": None,
        "input_start_utc": None,
        "input_end_utc": None,
        "target_start_utc": None,
        "target_end_utc": None,
        "status": f"Forecast update failed: {type(exc).__name__}: {exc}",
        "history": [],
        "forecast": [],
    }

    atomic_write_json(OUTPUT_FORECAST_JSON, payload)


# ============================================================
# Main
# ============================================================


def main():
    print("[INFO] Updating retained 72-hour solar-wind history...")
    solar_wind_window, forecast_time, metrics = collect_solar_wind_history()

    if not metrics["ready"]:
        print(
            "[WAIT] Solar-wind history is not ready: "
            f"{metrics['available_hours']}/{metrics['required_hours']} hours, "
            f"maximum missing run {metrics['max_missing_run_hours']} hours."
        )
        save_waiting_forecast_json(forecast_time, metrics)
        print("[DONE] History saved. Prediction was intentionally skipped.")
        return

    print("[INFO] Building latest 72-hour model input dataframe...")
    df = build_hourly_input_dataframe(solar_wind_window, forecast_time)

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
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        try:
            save_error_forecast_json(error)
        except Exception:
            traceback.print_exc()
        raise
