# =============================================================================
# SARIP 2026 — IIT Kanpur
# src/preprocessing.py
# Week 1: Data Download, Preprocessing, and Gap Masking Pipeline
#
# Author : Shri Harsan M | RA2512052010014 | SRM IST
# Project: Robustness and Interpretability of a Hybrid CNN-LSTM Model
#          for Exoplanet Transit Detection
#
# Run    : conda activate sarip && python src/preprocessing.py
# =============================================================================

import numpy as np
import pandas as pd
import lightkurve as lk
import requests
import warnings
import logging
from io import StringIO
from pathlib import Path
from tqdm import tqdm

# ── Reproducibility (set at top of EVERY script) ─────────────────────────────
np.random.seed(42)

# ── Suppress noisy but harmless warnings ─────────────────────────────────────
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# ── Project Paths (never hardcoded) ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"

# Create directories if they don't exist
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ── Fixed Constants — DO NOT CHANGE ──────────────────────────────────────────
N_BINS       = 201             # All light curves binned to this length
PERIOD_COL   = "koi_period"    # Orbital period (days)
EPOCH_COL    = "koi_time0bk"   # Transit epoch (BKJD)
DURATION_COL = "koi_duration"  # Transit duration (hours)
LABEL_COL    = "koi_disposition"

# ── Label Encoding ────────────────────────────────────────────────────────────
LABEL_MAP = {
    "CONFIRMED"    : 1,
    "FALSE POSITIVE": 0,
    "CANDIDATE"    : 2    # Excluded in binary classification runs
}


# =============================================================================
# 1. DOWNLOAD KOI CUMULATIVE TABLE
# =============================================================================
def download_koi_table() -> pd.DataFrame:
    """
    Download the NASA KOI cumulative table via direct CSV API.
    Uses requests + pandas — no astroquery TAP needed.
    Saves to data/raw/koi_table.csv and returns a clean DataFrame.
    """
    logging.info("Downloading KOI cumulative table from NASA Exoplanet Archive...")

    url = (
        "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
        "?table=cumulative"
        "&select=kepid,kepoi_name,koi_disposition,"
        "koi_period,koi_time0bk,koi_duration,koi_depth"
        "&format=csv"
    )

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Download failed: {e}")
        raise

    # NASA IPAC format uses '#' comment lines at the top — skip them
    koi = pd.read_csv(StringIO(response.text), comment='#')

    logging.info(f"Raw download: {len(koi)} rows, columns: {list(koi.columns)}")

    # Drop rows missing the columns we need for phase-folding
    koi = koi.dropna(subset=[PERIOD_COL, EPOCH_COL, DURATION_COL])

    # Keep only the three disposition classes we care about
    koi = koi[koi[LABEL_COL].isin(["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"])]
    koi = koi.reset_index(drop=True)

    # Save raw table
    save_path = DATA_RAW / "koi_table.csv"
    koi.to_csv(save_path, index=False)

    logging.info(f"KOI table saved → {save_path}  ({len(koi)} rows)")
    logging.info(f"Class distribution:\n{koi[LABEL_COL].value_counts().to_string()}")

    return koi


# =============================================================================
# 2. DOWNLOAD ONE LIGHT CURVE VIA LIGHTKURVE
# =============================================================================
def fetch_light_curve(kepid: int) -> lk.LightCurve | None:
    """
    Download Kepler PDCSAP long-cadence light curve for a single KIC ID.
    Stitches all quarters into one continuous light curve.
    Returns None on failure (missing data, network error, etc.).
    """
    try:
        search = lk.search_lightcurve(
            f"KIC {kepid}",
            mission="Kepler",
            cadence="long",
            author="Kepler"
        )
        if len(search) == 0:
            return None

        # ── KEY CHANGE: first 3 quarters only ────────────────────────────
        # Full 17-quarter download = ~60s per star = hours for 9500 stars
        # 3 quarters = ~10s per star, ~8 min for 50 curves test run
        quarter_limit = min(3, len(search))
        lc_collection = search[:quarter_limit].download_all()

        if lc_collection is None or len(lc_collection) == 0:
            return None

        lc = lc_collection.stitch()
        lc = lc.remove_nans().remove_outliers(sigma=5)
        return lc

    except Exception as e:
        logging.debug(f"KIC {kepid} — fetch failed: {e}")
        return None


# =============================================================================
# 3. PHASE-FOLD → NORMALIZE → BIN TO 201 POINTS
# =============================================================================
def phase_fold_and_bin(
    lc       : lk.LightCurve,
    period   : float,
    epoch    : float,
    duration_hours: float,
    n_bins   : int = N_BINS
) -> np.ndarray | None:
    """
    Phase-fold the light curve on the transit period, normalize flux to [0,1],
    bin to exactly n_bins points via interpolation.

    Returns a float32 array of shape (n_bins,), or None if the curve has
    insufficient data.

    Parameters
    ----------
    lc            : lightkurve LightCurve object
    period        : orbital period in days
    epoch         : transit epoch in BKJD
    duration_hours: transit duration in hours
    n_bins        : number of bins (fixed at 201 for this project)
    """
    try:
        # Phase-fold around the transit
        folded = lc.fold(period=period, epoch_time=epoch)

        # Restrict to ±2× transit duration around phase centre
        duration_days = duration_hours / 24.0
        half_window   = max(2.0 * duration_days / period, 0.1)

        mask   = np.abs(folded.phase.value) < half_window
        folded = folded[mask]

        if len(folded) < 20:
            return None   # Too sparse to be useful

        # Extract and normalize flux to [0, 1]
        flux  = folded.flux.value.astype(np.float32)
        f_min = flux.min()
        f_max = flux.max()

        if (f_max - f_min) < 1e-10:
            return None   # Flat/degenerate curve

        flux_norm = (flux - f_min) / (f_max - f_min)

        # Sort by phase and interpolate to fixed n_bins grid
        phase_vals  = folded.phase.value
        sort_idx    = np.argsort(phase_vals)
        phase_sorted = phase_vals[sort_idx]
        flux_sorted  = flux_norm[sort_idx]

        bin_grid    = np.linspace(phase_sorted[0], phase_sorted[-1], n_bins)
        flux_binned = np.interp(bin_grid, phase_sorted, flux_sorted)

        return flux_binned.astype(np.float32)

    except Exception as e:
        logging.debug(f"Phase-fold error: {e}")
        return None


# =============================================================================
# 4. RANDOM GAP MASKING (10–15% of time steps → 0)
# =============================================================================
def apply_gap_masking(
    curve         : np.ndarray,
    mask_fraction : float = None,
    rng           : np.random.Generator = None
) -> np.ndarray:
    """
    Randomly zero out 10–15% of time steps to simulate telemetry gaps.
    Uses a fixed RNG for reproducibility.

    Parameters
    ----------
    curve         : 1D float32 array of shape (n_bins,)
    mask_fraction : if None, sampled uniformly from [0.10, 0.15]
    rng           : numpy Generator (seeded externally for reproducibility)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if mask_fraction is None:
        mask_fraction = rng.uniform(0.10, 0.15)

    n_mask   = int(len(curve) * mask_fraction)
    mask_idx = rng.choice(len(curve), size=n_mask, replace=False)

    curve_masked          = curve.copy()
    curve_masked[mask_idx] = 0.0
    return curve_masked


# =============================================================================
# 5. FULL PIPELINE: DOWNLOAD → PREPROCESS → SAVE
# =============================================================================
def build_processed_dataset(
    koi       : pd.DataFrame,
    max_curves: int  = None,
    save      : bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each KOI in the table:
      1. Download Kepler light curve via lightkurve
      2. Phase-fold + normalize + bin to 201 points
      3. Apply random gap masking (10–15%)
      4. Encode label to integer
      5. Collect into X (N, 201) and y (N,) arrays

    Parameters
    ----------
    koi        : cleaned KOI DataFrame from download_koi_table()
    max_curves : limit for test runs (set None for full ~9500 run)
    save       : if True, saves X, y, and metadata to data/processed/

    Returns
    -------
    X : float32 array of shape (N, 201)
    y : int8 array of shape (N,)
    """
    rng    = np.random.default_rng(42)   # single RNG for reproducibility
    subset = koi if max_curves is None else koi.head(max_curves)

    X_list, y_list, meta_list = [], [], []
    skipped = 0

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Processing KOIs"):
        kepid    = int(row["kepid"])
        period   = float(row[PERIOD_COL])
        epoch    = float(row[EPOCH_COL])
        duration = float(row[DURATION_COL])
        label    = row[LABEL_COL]

        # Skip CANDIDATE for binary classification (keep for now, filter at train time)
        encoded_label = LABEL_MAP.get(label, -1)
        if encoded_label == -1:
            skipped += 1
            continue

        # Download
        lc = fetch_light_curve(kepid)
        if lc is None:
            skipped += 1
            continue

        # Phase-fold + bin
        curve = phase_fold_and_bin(lc, period, epoch, duration)
        if curve is None:
            skipped += 1
            continue

        # Gap masking
        curve = apply_gap_masking(curve, rng=rng)

        X_list.append(curve)
        y_list.append(encoded_label)
        meta_list.append({
            "kepid"      : kepid,
            "kepoi_name" : row.get("kepoi_name", ""),
            "label"      : label,
            "label_int"  : encoded_label,
            "period"     : period,
            "duration_hr": duration,
        })

    X = np.array(X_list, dtype=np.float32)   # (N, 201)
    y = np.array(y_list, dtype=np.int8)       # (N,)

    # Summary
    logging.info(f"\n{'='*50}")
    logging.info(f"Dataset built:   {X.shape[0]} curves  |  {skipped} skipped")
    logging.info(f"Array shapes:    X={X.shape}  y={y.shape}")

    unique, counts = np.unique(y, return_counts=True)
    label_names    = {v: k for k, v in LABEL_MAP.items()}
    for u, c in zip(unique, counts):
        logging.info(f"  Class {u} ({label_names.get(u, '?')}): {c} samples")
    logging.info(f"{'='*50}\n")

    if save:
        np.save(DATA_PROC / "X_processed.npy", X)
        np.save(DATA_PROC / "y_processed.npy", y)
        pd.DataFrame(meta_list).to_csv(DATA_PROC / "metadata.csv", index=False)
        logging.info(f"Saved → {DATA_PROC}/")
        logging.info(f"  X_processed.npy  {X.nbytes / 1e6:.2f} MB")
        logging.info(f"  y_processed.npy  {y.nbytes / 1e3:.2f} KB")
        logging.info(f"  metadata.csv")

    return X, y


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":

    logging.info("="*50)
    logging.info("SARIP 2026 — Preprocessing Pipeline")
    logging.info(f"Project root : {PROJECT_ROOT}")
    logging.info(f"Output dir   : {DATA_PROC}")
    logging.info("="*50)

    # Step 1: Download KOI table
    koi = download_koi_table()

    # Step 2: Build processed dataset
    # ─────────────────────────────────────────────────────────────────────────
    # ⚠️  FIRST RUN: keep max_curves=50 to verify the pipeline end-to-end.
    #     Takes ~10–20 min on CPU (NASA API response time dominates).
    #     Once confirmed working, remove the cap for the full ~9500 run overnight.
    # ─────────────────────────────────────────────────────────────────────────
    X, y = build_processed_dataset(koi, max_curves=50)

    print(f"\n✅ Preprocessing complete.")
    print(f"   X shape : {X.shape}   (curves × time-steps)")
    print(f"   y shape : {y.shape}")
    print(f"   Saved to: {DATA_PROC}")
    print(f"\n   Next step: run src/noise_injection.py to generate all 4 SNR versions.")
