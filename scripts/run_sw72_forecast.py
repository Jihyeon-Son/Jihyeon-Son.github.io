"""Live 72-hour hourly solar-wind speed forecast (ACE-72 model, FDL leaderboard `ace_v1`).

Runs from the repository root (same convention as scripts/run_forecast.py).

Issue time t0 = latest 00/06/12/18 UTC mark (override with --t0 for testing).
Inputs (identical to leaderboard/predict_submission.py):
  - past solar wind: ACE SWEPAM hourly speed with time_tag <= t0-2h (causal), 6h-gap
    interpolation inside the slice, 20 trailing 6-h bin means; all-missing ->
    persistence of the last valid speed within 10 days, else climatology 420 km/s
  - images: 10 frames at t0-114h .. t0-6h (12h step), AIA 211 & 193, from the JSOC
    near-real-time synoptic series (aia.lev1_nrt2, 1024px), preprocessed with the chain
    validated for the 2026 replay:
      DN/scale (193:/4, 211:/1) -> clip(10, 4096) -> /disk-median -> resize 512
      -> log10 -> per-image min-max uint8 -> vertical flip -> (64px grayscale at inference)
    Frames are cached as 512px PNGs under data/sw72/aia/<wave>/ so each run only
    downloads the frames it does not have yet; frames older than 8 days are pruned.
  - model: model/sw72_ace_v1_arch.json + model/sw72_ace_v1_weights.h5 (Keras, TF 2.10)
Outputs (data/sw72/):
  forecast.json          current forecast + observed history + provenance (dashboard input)
  forecast_archive.json  rolling list of past forecasts (last 21 days) for verification plots
  aia/<wave>/*.png       frame cache, aia/manifest.csv (frame time, url, raw disk median)
"""
import argparse
import csv
import io
import json
import re
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from PIL import Image, ImageOps

# ============================================================
# Paths
# ============================================================

MODEL_ARCH = Path("model/sw72_ace_v1_arch.json")
MODEL_WEIGHTS = Path("model/sw72_ace_v1_weights.h5")

DATA_DIR = Path("data/sw72")
AIA_DIR = DATA_DIR / "aia"
MANIFEST_CSV = AIA_DIR / "manifest.csv"
OUTPUT_JSON = DATA_DIR / "forecast.json"
ARCHIVE_JSON = DATA_DIR / "forecast_archive.json"

# ============================================================
# Data URLs
# ============================================================

ACE_MONTHLY_URL = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/{ym}_ace_swepam_1h.txt"
ACE_SWPC_JSON_URL = "https://services.swpc.noaa.gov/json/ace/swepam/ace_swepam_1h.json"
JSOC_NRT_URL = ("https://jsoc1.stanford.edu/data/aia/synoptic/nrt/{y}/{m:02d}/{d:02d}/H{h:02d}00/"
                "AIA{y}{m:02d}{d:02d}_{h:02d}{mi:02d}00_{wave}.fits")
JSOC_SYN_URL = ("https://jsoc1.stanford.edu/data/aia/synoptic/{y}/{m:02d}/{d:02d}/H{h:02d}00/"
                "AIA{y}{m:02d}{d:02d}_{h:02d}{mi:02d}_{wave}.fits")

# ============================================================
# Model / preprocessing constants (from the leaderboard pipeline)
# ============================================================

INPUT_SEQ_SW = 20          # 6-h bins of past speed
IMG_SEQ = 10               # frames per sample
IMG_STEP_H = 12
IMG_LAST_OFFSET_H = 6      # newest frame at t0-6h
IMG_SIZE = 64
SPEED_SCALE = 1000.0
CLIMATOLOGY_KMS = 420.0
GAP_INTERP_LIMIT_H = 6
MAX_IMG_OFFSET_H = 12.0    # fall back to earlier frames up to this far before nominal
WAVES = ["0211", "0193"]   # channel order of the image tensor
DN_SCALE = {"0193": 4.0, "0211": 1.0}
AEC_LOW_FRACTION = 0.4     # skip frame if raw disk median < 40% of running median (AEC short exposure)
CACHE_KEEP_DAYS = 8
ARCHIVE_KEEP_DAYS = 21
HISTORY_DAYS = 5
PRED_CLIP = (200.0, 1100.0)

REQUEST_TIMEOUT_SECONDS = 90
REQUEST_ATTEMPTS = 3

HTTP = requests.Session()
HTTP.headers.update({
    "User-Agent": ("Jihyeon-Son.github.io solar-wind forecast dashboard "
                   "(https://github.com/Jihyeon-Son/Jihyeon-Son.github.io)")
})


# ============================================================
# Utilities
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def latest_issue_time(now=None):
    """Most recent 00/06/12/18 UTC mark at or before now."""
    now = now or utc_now()
    h = (now.hour // 6) * 6
    return pd.Timestamp(now.replace(hour=h, minute=0, second=0, microsecond=0)).tz_convert(None)


def fetch_bytes(url, ok_404=False):
    last = None
    for attempt in range(1, REQUEST_ATTEMPTS + 1):
        try:
            r = HTTP.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            if r.status_code == 404 and ok_404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException as exc:
            last = exc
            print(f"[WARN] request failed ({attempt}/{REQUEST_ATTEMPTS}) {url}: {exc}")
            if attempt < REQUEST_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def atomic_write_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    tmp.replace(path)


def iso(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def none_if_nan(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


# ============================================================
# ACE hourly speed
# ============================================================

def parse_ace_monthly(text):
    rows = {}
    for line in text.splitlines():
        if line.startswith((":", "#")) or not line.strip():
            continue
        p = line.split()
        if len(p) < 9:
            continue
        t = pd.Timestamp(f"{p[0]}-{p[1]}-{p[2]} {p[3][:2]}:{p[3][2:]}")
        status, speed = int(p[6]), float(p[8])
        rows[t] = speed if status == 0 and speed > -999 else np.nan
    return pd.Series(rows, dtype=float).sort_index()


def load_ace_hourly(t0):
    """Raw (un-interpolated) ACE SWEPAM hourly speed covering [t0-HISTORY..t0]; NaN = gap."""
    months = sorted({(t0 - pd.Timedelta(days=k)).strftime("%Y%m") for k in (0, 7, 14)})
    parts, sources = [], []
    for ym in months:
        raw = fetch_bytes(ACE_MONTHLY_URL.format(ym=ym), ok_404=True)
        if raw is None:
            print(f"[WARN] ACE monthly file {ym} not found")
            continue
        parts.append(parse_ace_monthly(raw.decode("ascii", "ignore")))
        sources.append(f"ace_monthly:{ym}")
    if not parts:
        print("[WARN] no monthly ACE files; falling back to SWPC JSON")
        data = json.loads(fetch_bytes(ACE_SWPC_JSON_URL))
        s = pd.Series({pd.Timestamp(r["time_tag"]): (float(r["speed"]) if int(r.get("dsflag", 0)) == 0 else np.nan)
                       for r in data}, dtype=float).sort_index()
        parts.append(s)
        sources.append("swpc_json")
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    full = pd.date_range(s.index[0], max(s.index[-1], t0), freq="h")
    return s.reindex(full), sources


def past_bins_causal(raw, issue):
    """20 trailing 6-h bin means from hours with time_tag in [issue-121h, issue-2h] (causal)."""
    lo = issue - pd.Timedelta(hours=121 + 24)
    hi = issue - pd.Timedelta(hours=2)
    s = raw.loc[lo:hi].copy()
    s = s.interpolate(method="linear", limit=GAP_INTERP_LIMIT_H, limit_area="inside")
    hours = pd.date_range(issue - pd.Timedelta(hours=121), hi, freq="h")
    v = s.reindex(hours).values.reshape(INPUT_SEQ_SW, 6)
    with np.errstate(invalid="ignore"):
        bins = np.nanmean(v, axis=1)
    n_valid_hours = int(np.isfinite(v).sum())
    fallback = None
    if np.isnan(bins).all():
        prev = raw.loc[issue - pd.Timedelta(days=10):hi].dropna()
        fill = prev.iloc[-1] if len(prev) else CLIMATOLOGY_KMS
        bins = np.full(INPUT_SEQ_SW, fill)
        fallback = "persistence" if len(prev) else "climatology"
    else:
        bins = pd.Series(bins).interpolate(limit_area="inside").ffill().bfill().values
    bin_times = [hours[6 * k + 5] for k in range(INPUT_SEQ_SW)]      # bin end hour
    return bins.astype(np.float32), bin_times, n_valid_hours, fallback


# ============================================================
# AIA frames
# ============================================================

def preprocess_frame(data, hdr, wave):
    """Validated 2026-replay chain -> (512px uint8 PIL image, raw disk median)."""
    d = data.astype(np.float64) / DN_SCALE[wave]
    n = d.shape[0]
    rsun = hdr.get("R_SUN", hdr.get("RSUN_OBS", 970) / abs(hdr.get("CDELT1", 2.4)))
    X = np.arange(n)[:, None]
    Y = np.arange(n)[None, :]
    disk = np.sqrt((X - n / 2.0) ** 2 + (Y - n / 2.0) ** 2) < rsun
    med = float(np.median(d[disk]))
    d = d.clip(10, 2 ** 12) / np.median(d.clip(10, 2 ** 12)[disk])
    img = np.array(Image.fromarray(d).resize((512, 512), Image.BILINEAR))
    img = np.log10(np.maximum(img, 1e-6))
    u8 = ((img - img.min()) / (img.max() - img.min()) * 255.0).astype(np.uint8)
    return ImageOps.flip(Image.fromarray(u8)), med


def read_manifest():
    if not MANIFEST_CSV.exists():
        return []
    with open(MANIFEST_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows):
    rows = sorted(rows, key=lambda r: (r["wave"], r["frame_time"]))
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["wave", "frame_time", "url", "raw_disk_median", "fetched_utc"])
        w.writeheader()
        w.writerows(rows)


def frame_path(wave, t):
    return AIA_DIR / wave / f"AIA{pd.Timestamp(t).strftime('%Y%m%d_%H%M%S')}_{wave}.png"


def frame_time_from_path(p):
    m = re.search(r"AIA(\d{8})_(\d{6})", p.name)
    return pd.Timestamp(f"{m.group(1)} {m.group(2)[:2]}:{m.group(2)[2:4]}:{m.group(2)[4:]}")


def try_fetch_frame(t, wave, ref_median):
    """Download + preprocess the frame at hour t (NRT first, then definitive synoptic).
    Returns (PIL image, url, raw median) or None. Skips AEC short-exposure files."""
    y, m, d, h = t.year, t.month, t.day, t.hour
    candidates = [JSOC_NRT_URL.format(y=y, m=m, d=d, h=h, mi=mi, wave=wave) for mi in (0, 3, 6, 9, 12)]
    candidates += [JSOC_SYN_URL.format(y=y, m=m, d=d, h=h, mi=mi, wave=wave) for mi in (0, 2, 4, 6, 8, 10)]
    skipped = None
    for url in candidates:
        try:
            raw = fetch_bytes(url, ok_404=True)
        except RuntimeError as exc:
            print(f"[WARN] {exc}")
            continue
        if raw is None:
            continue
        try:
            hdul = fits.open(io.BytesIO(raw))
            hdul.verify("silentfix")
            hdu = hdul[1] if len(hdul) > 1 and hdul[1].data is not None else hdul[0]
            img, med = preprocess_frame(hdu.data, dict(hdu.header), wave)
        except Exception as exc:  # corrupt / partial file -> next candidate
            print(f"[WARN] cannot read {url}: {exc}")
            continue
        if ref_median is not None and med < AEC_LOW_FRACTION * ref_median:
            print(f"[INFO] {wave} {t}: low exposure (median {med:.1f} vs ref {ref_median:.1f}), trying next file")
            skipped = (img, url, med)
            continue
        return img, url, med
    return skipped   # better a short-exposure frame than none


def ensure_frames(nominal_times):
    """Make sure the cache holds a frame at (or up to MAX_IMG_OFFSET_H before) every nominal time.
    Returns {wave: [(used_time, path, offset_h), ...]} in nominal order."""
    manifest = read_manifest()
    used = {}
    for wave in WAVES:
        (AIA_DIR / wave).mkdir(parents=True, exist_ok=True)
        meds = [float(r["raw_disk_median"]) for r in manifest if r["wave"] == wave and r["raw_disk_median"]]
        used[wave] = []
        for nominal in nominal_times:
            chosen = None
            for back in range(0, int(MAX_IMG_OFFSET_H) + 1):
                t = nominal - pd.Timedelta(hours=back)
                p = frame_path(wave, t)
                if p.exists():
                    chosen = (t, p, float(back))
                    break
                ref = float(np.median(meds[-20:])) if meds else None
                got = try_fetch_frame(t, wave, ref)
                if got is None:
                    continue
                img, url, med = got
                img.save(p)
                meds.append(med)
                manifest.append({"wave": wave, "frame_time": iso(t), "url": url,
                                 "raw_disk_median": f"{med:.3f}", "fetched_utc": iso(utc_now())})
                chosen = (t, p, float(back))
                print(f"[INFO] fetched {wave} {t} (offset {back}h) from {url}")
                break
            if chosen is None:
                raise RuntimeError(f"no AIA {wave} frame within {MAX_IMG_OFFSET_H}h before {nominal}")
            used[wave].append(chosen)
    write_manifest(manifest)
    return used


def prune_cache(t0):
    cutoff = t0 - pd.Timedelta(days=CACHE_KEEP_DAYS)
    removed = 0
    for wave in WAVES:
        for p in (AIA_DIR / wave).glob("*.png"):
            if frame_time_from_path(p) < cutoff:
                p.unlink()
                removed += 1
    manifest = [r for r in read_manifest() if pd.Timestamp(r["frame_time"]).tz_convert(None) >= cutoff]
    write_manifest(manifest)
    if removed:
        print(f"[INFO] pruned {removed} cached frames older than {cutoff}")


def load_frame_64(path):
    return np.array(Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float16) / np.float16(255.0)


# ============================================================
# Model
# ============================================================

def load_model():
    import tensorflow as tf
    with open(MODEL_ARCH, encoding="utf-8") as f:
        model = tf.keras.models.model_from_json(f.read())
    model.load_weights(str(MODEL_WEIGHTS))
    return model


# ============================================================
# Outputs
# ============================================================

def load_archive():
    if not ARCHIVE_JSON.exists():
        return []
    try:
        with open(ARCHIVE_JSON, encoding="utf-8") as f:
            return json.load(f).get("forecasts", [])
    except (OSError, ValueError):
        return []


def update_archive(entry, t0):
    arch = [a for a in load_archive() if a["t0"] != entry["t0"]]
    arch.append(entry)
    cutoff = t0 - pd.Timedelta(days=ARCHIVE_KEEP_DAYS)
    arch = sorted([a for a in arch if pd.Timestamp(a["t0"]).tz_convert(None) >= cutoff], key=lambda a: a["t0"])
    atomic_write_json(ARCHIVE_JSON, {"updated_utc": iso(utc_now()), "forecasts": arch})
    return arch


def write_failure(t0, message):
    payload = {}
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            payload = {}
    payload.update({
        "last_attempt_utc": iso(utc_now()),
        "last_attempt_t0_utc": iso(t0),
        "last_attempt_status": f"FAILED: {message}",
    })
    if "status" not in payload:
        payload["status"] = f"No forecast available yet ({message})."
    atomic_write_json(OUTPUT_JSON, payload)


# ============================================================
# Main
# ============================================================

def run(t0):
    print(f"[INFO] issue time t0 = {t0} UTC")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- solar wind ---
    raw, sw_sources = load_ace_hourly(t0)
    sw_bins, bin_times, n_valid, sw_fallback = past_bins_causal(raw, t0)
    last_valid = raw.loc[:t0 - pd.Timedelta(hours=2)].dropna()
    last_ace_time = last_valid.index[-1] if len(last_valid) else None
    print(f"[INFO] ACE: {n_valid}/120 valid hours in input window, last valid {last_ace_time}, "
          f"fallback={sw_fallback or 'none'}")

    # --- images ---
    nominal = [t0 - pd.Timedelta(hours=IMG_LAST_OFFSET_H + IMG_STEP_H * (IMG_SEQ - 1 - k)) for k in range(IMG_SEQ)]
    used = ensure_frames(nominal)
    img = np.zeros((1, IMG_SEQ, IMG_SIZE, IMG_SIZE, len(WAVES)), dtype=np.float16)
    frames_out = []
    for k, nom in enumerate(nominal):
        rec = {"nominal_time": iso(nom)}
        for c, wave in enumerate(WAVES):
            t, p, off = used[wave][k]
            img[0, k, :, :, c] = load_frame_64(p)
            rec[f"aia{wave}_time"] = iso(t)
            rec[f"aia{wave}_path"] = p.as_posix()
            rec[f"aia{wave}_offset_h"] = off
        frames_out.append(rec)
    max_off = max(r[f"aia{w}_offset_h"] for r in frames_out for w in WAVES)
    print(f"[INFO] images ready, max offset from nominal {max_off:.0f}h")

    # --- inference ---
    model = load_model()
    sw_in = (sw_bins / SPEED_SCALE).reshape(1, INPUT_SEQ_SW).astype(np.float32)
    pred = model.predict([sw_in, img], verbose=0)[0] * SPEED_SCALE
    if not np.isfinite(pred).all():
        raise RuntimeError("non-finite prediction")
    n_clip = int(((pred < PRED_CLIP[0]) | (pred > PRED_CLIP[1])).sum())
    pred = pred.clip(*PRED_CLIP)
    target_times = pd.date_range(t0 + pd.Timedelta(hours=1), periods=72, freq="h")
    print(f"[INFO] prediction range [{pred.min():.0f}, {pred.max():.0f}] km/s, +1h {pred[0]:.0f}, "
          f"+72h {pred[-1]:.0f}, clipped {n_clip}")

    # --- outputs ---
    hist_idx = pd.date_range(t0 - pd.Timedelta(days=HISTORY_DAYS), raw.index[-1], freq="h")
    hist = raw.reindex(hist_idx)
    history = [{"time": iso(t), "speed": none_if_nan(v)} for t, v in hist.items()]

    forecast = [{"time": iso(t), "speed_pred": round(float(v), 1)} for t, v in zip(target_times, pred)]
    entry = {"t0": iso(t0), "created_utc": iso(utc_now()),
             "speed_pred": [round(float(v), 1) for v in pred], "sw_fallback": sw_fallback or ""}
    archive = update_archive(entry, t0)
    previous = [a for a in archive if pd.Timestamp(a["t0"]).tz_convert(None) <= t0 - pd.Timedelta(hours=24)]
    prev_forecast = None
    if previous:
        a = previous[-1]
        pt0 = pd.Timestamp(a["t0"]).tz_convert(None)
        prev_forecast = {"t0": a["t0"], "forecast": [
            {"time": iso(pt0 + pd.Timedelta(hours=h + 1)), "speed_pred": v} for h, v in enumerate(a["speed_pred"])]}

    notes = []
    if sw_fallback:
        notes.append(f"Past solar-wind input unavailable; used {sw_fallback} fill.")
    if max_off > 0:
        notes.append(f"Some AIA frames were up to {max_off:.0f} h earlier than nominal.")
    if n_clip:
        notes.append(f"{n_clip} predicted values clipped to [{PRED_CLIP[0]:.0f}, {PRED_CLIP[1]:.0f}] km/s.")

    payload = {
        "created_utc": iso(utc_now()),
        "forecast_base_time_utc": iso(t0),
        "input_sw_start_utc": iso(t0 - pd.Timedelta(hours=121)),
        "input_sw_end_utc": iso(t0 - pd.Timedelta(hours=2)),
        "target_start_utc": iso(target_times[0]),
        "target_end_utc": iso(target_times[-1]),
        "status": "Forecast is current.",
        "message": " ".join(notes) if notes else "All inputs nominal.",
        "model_name": "ACE-72 (FDL 2026 solar-wind challenge, ace_v1 inception)",
        "model_note": ("Inputs: 20 six-hour means of ACE SWEPAM speed up to t0-2h and 10 AIA 211/193 "
                       "synoptic frames at t0-114h..t0-6h (12 h step). Output: hourly speed for t0+1h..t0+72h."),
        "solar_wind_sources": sw_sources,
        "last_ace_time_utc": iso(last_ace_time) if last_ace_time is not None else None,
        "sw_valid_hours": n_valid,
        "sw_fallback": sw_fallback or "",
        "max_image_offset_h": max_off,
        "input_bins": [{"time": iso(t), "speed": round(float(v), 1)} for t, v in zip(bin_times, sw_bins)],
        "latest_images": {f"aia{w}": used[w][-1][1].as_posix() for w in WAVES},
        "latest_image_times": {f"aia{w}": iso(used[w][-1][0]) for w in WAVES},
        "frames": frames_out,
        "history": history,
        "forecast": forecast,
        "previous_forecast": prev_forecast,
        "last_attempt_utc": iso(utc_now()),
        "last_attempt_t0_utc": iso(t0),
        "last_attempt_status": "OK",
    }
    atomic_write_json(OUTPUT_JSON, payload)
    prune_cache(t0)
    print(f"[INFO] wrote {OUTPUT_JSON} ({len(forecast)} hourly values)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", default=None, help="issue time override, e.g. 2026-09-08T00:00 (UTC)")
    args = ap.parse_args()
    t0 = pd.Timestamp(args.t0) if args.t0 else latest_issue_time()
    assert t0.hour % 6 == 0 and t0.minute == 0, "t0 must be on the 00/06/12/18 UTC grid"
    try:
        run(t0)
    except Exception as exc:
        traceback.print_exc()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        write_failure(t0, f"{type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
