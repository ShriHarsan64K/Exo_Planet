# =============================================================================
# SARIP 2026 — IIT Kanpur
# src/train.py
# Training Loop — All Models × All SNR Levels
#
# Author : Shri Harsan M | RA2512052010014 | SRM IST
# Week   : 3 (baselines), 5–6 (CNN-LSTM + ablation)
#
# Usage:
#   # Run all baseline models across all SNR levels (Week 3):
#   python src/train.py --run baselines
#
#   # Run CNN-LSTM + ablation (Week 5-6):
#   python src/train.py --run ablation
#
#   # Run a single model on a single SNR (for testing):
#   python src/train.py --model cnn --snr snr_high
# =============================================================================

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import logging
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score,
                              classification_report)
from sklearn.preprocessing import label_binarize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import (configure_gpu, get_model, focal_loss, F1Score,
                    SKLEARN_MODELS, KERAS_MODELS,
                    BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE)

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_NOISY   = PROJECT_ROOT / "data" / "noisy"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Fixed SNR Levels ──────────────────────────────────────────────────────────
SNR_DIRS = {
    "clean"       : DATA_PROC,          # No noise — original processed data
    "snr_high"    : DATA_NOISY / "snr_high",
    "snr_medium"  : DATA_NOISY / "snr_medium",
    "snr_low"     : DATA_NOISY / "snr_low",
    "snr_very_low": DATA_NOISY / "snr_very_low",
}

# ── Run groups ────────────────────────────────────────────────────────────────
BASELINE_MODELS = ["lr", "rf", "cnn"]
ABLATION_MODELS = ["cnn", "lstm", "cnn_lstm"]


# =============================================================================
# DATA LOADING — one SNR version at a time (never load all simultaneously)
# =============================================================================
def load_snr_data(snr_name: str) -> tuple:
    """
    Load X and y for one SNR level.
    Filters out CANDIDATE class (label=2) — binary classification only.
    Reshapes X to (N, 201, 1) for Keras models.

    Returns
    -------
    X_flat  : (N, 201)    float32 — for sklearn models
    X_3d    : (N, 201, 1) float32 — for Keras models
    y       : (N,)        int     — binary: 0=FALSE POSITIVE, 1=CONFIRMED
    """
    data_dir = SNR_DIRS[snr_name]

    # Handle clean vs noisy directory structure
    x_path = data_dir / ("X_processed.npy" if snr_name == "clean" else "X.npy")
    y_path = data_dir / ("y_processed.npy" if snr_name == "clean" else "y.npy")

    if not x_path.exists():
        raise FileNotFoundError(f"Data not found: {x_path}")

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int8)

    # Binary filter: keep only CONFIRMED (1) and FALSE POSITIVE (0)
    binary_mask = y != 2
    X = X[binary_mask]
    y = y[binary_mask]

    logging.info(f"  Loaded [{snr_name}]: X={X.shape}  "
                 f"FP={int((y==0).sum())}  CONF={int((y==1).sum())}")

    X_flat = X                              # (N, 201)
    X_3d   = X[:, :, np.newaxis]           # (N, 201, 1) for Keras

    return X_flat, X_3d, y.astype(np.int32)


# =============================================================================
# TRAIN-VAL-TEST SPLIT — stratified, fixed seed
# =============================================================================
def split_data(X_flat, X_3d, y):
    """
    Stratified 70/15/15 split.
    Returns flat and 3D versions for sklearn and Keras respectively.
    """
    # First split: train (70%) vs temp (30%)
    (Xf_train, Xf_temp,
     X3_train, X3_temp,
     y_train, y_temp) = train_test_split(
        X_flat, X_3d, y,
        test_size=0.30, stratify=y, random_state=42
    )

    # Second split: val (15%) vs test (15%) from temp
    (Xf_val, Xf_test,
     X3_val, X3_test,
     y_val, y_test) = train_test_split(
        Xf_temp, X3_temp, y_temp,
        test_size=0.50, stratify=y_temp, random_state=42
    )

    logging.info(f"  Split: train={len(y_train)}  "
                 f"val={len(y_val)}  test={len(y_test)}")
    return (Xf_train, Xf_val, Xf_test,
            X3_train, X3_val, X3_test,
            y_train, y_val, y_test)


# =============================================================================
# COMPUTE CLASS WEIGHTS — for imbalanced training
# =============================================================================
def compute_class_weights(y_train: np.ndarray) -> dict:
    """Compute balanced class weights from training labels."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    cw = dict(zip(classes.tolist(), weights.tolist()))
    logging.info(f"  Class weights: {cw}")
    return cw


# =============================================================================
# KERAS CALLBACKS
# =============================================================================
def get_callbacks(model_name: str, snr_name: str) -> list:
    """
    Return the standard callback set (locked per project spec).
    Saves best model checkpoint to models/ directory.
    """
    ckpt_dir = MODELS_DIR / ("baselines" if model_name in BASELINE_MODELS
                              else "cnn_lstm")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{model_name}_{snr_name}.keras"

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor           = 'val_roc_auc',
            patience          = PATIENCE,
            restore_best_weights = True,
            mode              = 'max',
            verbose           = 1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath          = str(ckpt_path),
            monitor           = 'val_roc_auc',
            save_best_only    = True,
            mode              = 'max',
            verbose           = 0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor           = 'val_roc_auc',
            factor            = 0.5,
            patience          = 3,
            mode              = 'max',
            verbose           = 1,
            min_lr            = 1e-6
        ),
    ]


# =============================================================================
# EVALUATE MODEL — compute all required metrics
# =============================================================================
def evaluate_model(
    model_name : str,
    snr_name   : str,
    y_true     : np.ndarray,
    y_pred_prob: np.ndarray,
    elapsed_s  : float
) -> dict:
    """
    Compute all 5 required metrics + per-class breakdown.
    Returns a flat dict ready for CSV saving.
    """
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # ── Macro metrics ─────────────────────────────────────────────────────────
    results = {
        "model"    : model_name,
        "snr"      : snr_name,
        "precision": round(precision_score(y_true, y_pred,
                                           average='macro', zero_division=0), 4),
        "recall"   : round(recall_score(y_true, y_pred,
                                        average='macro', zero_division=0), 4),
        "f1"       : round(f1_score(y_true, y_pred,
                                    average='macro', zero_division=0), 4),
        "roc_auc"  : round(roc_auc_score(y_true, y_pred_prob), 4),
        "pr_auc"   : round(average_precision_score(y_true, y_pred_prob), 4),
        "time_s"   : round(elapsed_s, 1),
    }

    # ── Per-class metrics ─────────────────────────────────────────────────────
    for cls, cls_name in [(0, "fp"), (1, "conf")]:
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        results[f"precision_{cls_name}"] = round(
            precision_score(y_true, y_pred, labels=[cls],
                            average='macro', zero_division=0), 4)
        results[f"recall_{cls_name}"]    = round(
            recall_score(y_true, y_pred, labels=[cls],
                         average='macro', zero_division=0), 4)
        results[f"f1_{cls_name}"]        = round(
            f1_score(y_true, y_pred, labels=[cls],
                     average='macro', zero_division=0), 4)

    return results


# =============================================================================
# TRAIN ONE SKLEARN MODEL
# =============================================================================
def train_sklearn(
    model_name: str,
    snr_name  : str,
    Xf_train  : np.ndarray,
    Xf_test   : np.ndarray,
    y_train   : np.ndarray,
    y_test    : np.ndarray
) -> dict:
    """Train LR or RF. Returns metrics dict."""
    logging.info(f"  Training {model_name.upper()} on {snr_name}...")
    model = get_model(model_name)
    t0    = time.time()
    model.fit(Xf_train, y_train)
    elapsed = time.time() - t0

    y_prob = model.predict_proba(Xf_test)[:, 1]
    metrics = evaluate_model(model_name, snr_name, y_test, y_prob, elapsed)

    # Save sklearn model
    import joblib
    save_dir = MODELS_DIR / "baselines"
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_dir / f"{model_name}_{snr_name}.pkl")

    logging.info(f"  F1={metrics['f1']}  ROC-AUC={metrics['roc_auc']}  "
                 f"PR-AUC={metrics['pr_auc']}")
    return metrics


# =============================================================================
# TRAIN ONE KERAS MODEL
# =============================================================================
def train_keras(
    model_name: str,
    snr_name  : str,
    X3_train  : np.ndarray,
    X3_val    : np.ndarray,
    X3_test   : np.ndarray,
    y_train   : np.ndarray,
    y_val     : np.ndarray,
    y_test    : np.ndarray,
    class_weights: dict
) -> dict:
    """Train CNN-only, LSTM-only, or CNN-LSTM. Returns metrics dict."""
    logging.info(f"  Training {model_name.upper()} on {snr_name}...")
    model     = get_model(model_name)
    callbacks = get_callbacks(model_name, snr_name)
    t0        = time.time()

    history = model.fit(
        X3_train, y_train,
        validation_data = (X3_val, y_val),
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        class_weight    = class_weights,
        callbacks       = callbacks,
        verbose         = 1
    )

    elapsed = time.time() - t0
    y_prob  = model.predict(X3_test, batch_size=BATCH_SIZE, verbose=0).flatten()
    metrics = evaluate_model(model_name, snr_name, y_test, y_prob, elapsed)

    # Record best epoch
    best_epoch = int(np.argmax(history.history.get('val_f1', [0]))) + 1
    metrics['best_epoch'] = best_epoch

    logging.info(f"  F1={metrics['f1']}  ROC-AUC={metrics['roc_auc']}  "
                 f"PR-AUC={metrics['pr_auc']}  best_epoch={best_epoch}")
    return metrics


# =============================================================================
# SAVE RESULTS TO CSV — called after EVERY run
# =============================================================================
def append_results(metrics: dict, csv_name: str = "all_results.csv"):
    """Append one row to the master results CSV. Never overwrites."""
    csv_path = RESULTS_DIR / csv_name
    df_new   = pd.DataFrame([metrics])

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(csv_path, index=False)
    logging.info(f"  ✅ Results saved → {csv_path}  "
                 f"(total rows: {len(df_out)})")


# =============================================================================
# RUN ONE MODEL × ONE SNR
# =============================================================================
def run_one(model_name: str, snr_name: str):
    """Full pipeline: load → split → train → evaluate → save."""
    logging.info(f"\n{'='*55}")
    logging.info(f"  Model: {model_name.upper()}  |  SNR: {snr_name}")
    logging.info(f"{'='*55}")

    X_flat, X_3d, y = load_snr_data(snr_name)
    (Xf_train, Xf_val, Xf_test,
     X3_train, X3_val, X3_test,
     y_train, y_val, y_test) = split_data(X_flat, X_3d, y)

    class_weights = compute_class_weights(y_train)

    if model_name in SKLEARN_MODELS:
        metrics = train_sklearn(model_name, snr_name,
                                Xf_train, Xf_test, y_train, y_test)
    else:
        metrics = train_keras(model_name, snr_name,
                              X3_train, X3_val, X3_test,
                              y_train, y_val, y_test, class_weights)

    append_results(metrics)

    # Force memory cleanup between runs — critical for 6GB VRAM
    tf.keras.backend.clear_session()
    import gc; gc.collect()

    return metrics


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    configure_gpu()

    parser = argparse.ArgumentParser(description="SARIP 2026 — Training Script")
    parser.add_argument("--run",   type=str, default=None,
                        choices=["baselines", "ablation"],
                        help="Run a full group of models across all SNR levels")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model to train: lr, rf, cnn, lstm, cnn_lstm")
    parser.add_argument("--snr",   type=str, default=None,
                        choices=list(SNR_DIRS.keys()),
                        help="Single SNR level to train on")
    args = parser.parse_args()

    all_metrics = []

    # ── Mode 1: Run a full group ───────────────────────────────────────────────
    if args.run == "baselines":
        model_list = BASELINE_MODELS
        logging.info(f"\nRunning BASELINES: {model_list}")
        logging.info(f"SNR levels: {list(SNR_DIRS.keys())}")
        logging.info(f"Total runs: {len(model_list) * len(SNR_DIRS)}\n")

        for snr_name in SNR_DIRS:
            for model_name in model_list:
                m = run_one(model_name, snr_name)
                all_metrics.append(m)

    elif args.run == "ablation":
        model_list = ABLATION_MODELS
        logging.info(f"\nRunning ABLATION: {model_list}")
        logging.info(f"Total runs: {len(model_list) * len(SNR_DIRS)}\n")

        for snr_name in SNR_DIRS:
            for model_name in model_list:
                m = run_one(model_name, snr_name)
                all_metrics.append(m)

    # ── Mode 2: Single run ────────────────────────────────────────────────────
    elif args.model and args.snr:
        m = run_one(args.model, args.snr)
        all_metrics.append(m)

    else:
        parser.print_help()

    # ── Final summary ──────────────────────────────────────────────────────────
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        print(f"\n{'='*55}")
        print("TRAINING COMPLETE — Results Summary")
        print(f"{'='*55}")
        print(df[["model", "snr", "f1", "roc_auc", "pr_auc"]].to_string(index=False))
        print(f"\nFull results saved → {RESULTS_DIR}/all_results.csv")
