# =============================================================================
# SARIP 2026 — IIT Kanpur
# src/download_full.py
# Full dataset downloader — multiprocessing, incremental, resumable
#
# FIX v2: ThreadPoolExecutor → ProcessPoolExecutor
#   Corrupted 0-byte FITS files send SIGBUS which kills threads but only
#   crashes the worker *process* in multiprocessing — main process survives.
#
# Run: conda activate sarip && python src/download_full.py
# =============================================================================

import numpy as np
import pandas as pd
import warnings
import logging
import time
import sys
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROC      = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = DATA_PROC / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
N_WORKERS        = 4    # ProcessPoolExecutor workers (each is a separate process)
CHECKPOINT_EVERY = 100  # Save partial results every N successful curves


# =============================================================================
# WORKER FUNCTION — runs in a separate process
# Must be importable at module level for multiprocessing to pickle it
# Accepts a plain dict (picklable) instead of a pandas Series
# =============================================================================
def process_one(row_dict: dict) -> dict | None:
    """
    Worker function: download + preprocess one KOI.
    Runs in its own process — a SIGBUS/crash here does NOT kill the main process.
    Accepts a plain dict so it's safely picklable across processes.
    """
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import lightkurve as lk

    # Each worker imports preprocessing from src/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preprocessing import (
        phase_fold_and_bin, apply_gap_masking,
        LABEL_MAP, PERIOD_COL, EPOCH_COL, DURATION_COL, LABEL_COL
    )

    try:
        kepid    = int(row_dict["kepid"])
        period   = float(row_dict[PERIOD_COL])
        epoch    = float(row_dict[EPOCH_COL])
        duration = float(row_dict[DURATION_COL])
        label    = row_dict[LABEL_COL]

        encoded_label = LABEL_MAP.get(label, -1)
        if encoded_label == -1:
            return None

        # Download — 3 quarters only
        search = lk.search_lightcurve(
            f"KIC {kepid}", mission="Kepler",
            cadence="long", author="Kepler"
        )
        if len(search) == 0:
            return None

        quarter_limit = min(3, len(search))
        lc_collection = search[:quarter_limit].download_all()
        if lc_collection is None or len(lc_collection) == 0:
            return None

        lc = lc_collection.stitch().remove_nans().remove_outliers(sigma=5)

        curve = phase_fold_and_bin(lc, period, epoch, duration)
        if curve is None:
            return None

        rng   = np.random.default_rng(kepid)   # seeded by kepid — deterministic
        curve = apply_gap_masking(curve, rng=rng)

        return {
            "curve"      : curve,
            "label"      : encoded_label,
            "kepid"      : kepid,
            "kepoi_name" : row_dict.get("kepoi_name", ""),
            "label_str"  : label,
            "period"     : period,
            "duration_hr": duration,
        }

    except Exception:
        return None   # Silently skip — keeps the pool alive


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================
def save_checkpoint(results: list, chunk_id: int):
    curves = np.array([r["curve"] for r in results], dtype=np.float32)
    labels = np.array([r["label"] for r in results], dtype=np.int8)
    meta   = pd.DataFrame([{k: v for k, v in r.items() if k != "curve"}
                            for r in results])
    path   = CHECKPOINT_DIR / f"chunk_{chunk_id:04d}.npz"
    np.savez(path, X=curves, y=labels)
    meta.to_csv(CHECKPOINT_DIR / f"chunk_{chunk_id:04d}_meta.csv", index=False)
    logging.info(f"  ✅ Checkpoint {chunk_id:04d} saved — {len(results)} curves")


def merge_checkpoints() -> tuple:
    logging.info("Merging all checkpoints...")
    X_parts, y_parts, meta_parts = [], [], []
    for f in sorted(CHECKPOINT_DIR.glob("chunk_*.npz")):
        data = np.load(f)
        X_parts.append(data["X"])
        y_parts.append(data["y"])
        meta_path = f.with_name(f.stem + "_meta.csv")
        if meta_path.exists():
            meta_parts.append(pd.read_csv(meta_path))
    X    = np.concatenate(X_parts, axis=0).astype(np.float32)
    y    = np.concatenate(y_parts, axis=0).astype(np.int8)
    meta = pd.concat(meta_parts, ignore_index=True) if meta_parts else pd.DataFrame()
    return X, y, meta


# =============================================================================
# MAIN
# =============================================================================
def run_full_download():
    koi_path = DATA_RAW / "koi_table.csv"
    if not koi_path.exists():
        logging.error(f"KOI table not found: {koi_path}")
        return

    koi = pd.read_csv(koi_path)
    logging.info(f"Loaded KOI table: {len(koi)} rows")

    # ── Resume: skip already-processed kepids ────────────────────────────────
    processed_ids = set()
    for f in CHECKPOINT_DIR.glob("chunk_*_meta.csv"):
        df = pd.read_csv(f)
        processed_ids.update(df["kepid"].astype(int).tolist())

    if processed_ids:
        koi = koi[~koi["kepid"].astype(int).isin(processed_ids)]
        logging.info(f"Resuming — {len(processed_ids)} already saved, "
                     f"{len(koi)} remaining")

    # Convert to list of plain dicts — required for pickling across processes
    rows     = [row.to_dict() for _, row in koi.iterrows()]
    results  = []
    chunk_id = len(list(CHECKPOINT_DIR.glob("chunk_*.npz")))
    done     = 0
    skipped  = 0
    start    = time.time()

    logging.info(f"Starting: {len(rows)} KOIs | {N_WORKERS} worker processes\n")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_one, row): row for row in rows}

        with tqdm(total=len(rows), desc="Downloading KOIs") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception:
                    result = None   # Worker process crashed — skip safely

                if result is not None:
                    results.append(result)
                    done += 1
                else:
                    skipped += 1

                pbar.update(1)
                pbar.set_postfix(done=done, skipped=skipped)

                if len(results) >= CHECKPOINT_EVERY:
                    save_checkpoint(results, chunk_id)
                    results  = []
                    chunk_id += 1

    if results:
        save_checkpoint(results, chunk_id)

    X, y, meta = merge_checkpoints()

    np.save(DATA_PROC / "X_processed.npy", X)
    np.save(DATA_PROC / "y_processed.npy", y)
    meta.to_csv(DATA_PROC / "metadata.csv", index=False)

    elapsed = (time.time() - start) / 3600
    logging.info(f"\n{'='*55}")
    logging.info(f"COMPLETE — {X.shape[0]} curves | {skipped} skipped")
    logging.info(f"X shape: {X.shape} | Time: {elapsed:.2f} hrs")
    label_names = {v: k for k, v in
                   {"CONFIRMED": 1, "FALSE POSITIVE": 0, "CANDIDATE": 2}.items()}
    for u, c in zip(*np.unique(y, return_counts=True)):
        logging.info(f"  Class {u} ({label_names.get(u,'?')}): {c}")
    logging.info(f"Saved → {DATA_PROC}/")
    logging.info(f"{'='*55}")


if __name__ == "__main__":
    run_full_download()
