# =============================================================================
# SARIP 2026 — IIT Kanpur
# src/gradcam.py
# Week 7–8: Grad-CAM Interpretability Analysis
#
# Author : Shri Harsan M | RA2512052010014 | SRM IST
# Project: Robustness and Interpretability of a Hybrid CNN-LSTM Model
#
# What this does:
#   1. Loads the saved CNN-LSTM model (best SNR-high checkpoint)
#   2. Computes Grad-CAM activations for the last Conv1D layer ("conv2")
#   3. Overlays activation heatmaps on light curves
#   4. Verifies whether highlighted regions align with transit dip
#   5. Compares heatmaps across 4 SNR levels (same curve)
#   6. Documents failure cases at very low SNR
#
# Run: conda activate sarip && python src/gradcam.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import warnings
import logging
from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── GPU config ────────────────────────────────────────────────────────────────
def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logging.info(f"GPU ready: {gpus[0].name}")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
DATA_NOISY   = PROJECT_ROOT / "data" / "noisy"
MODELS_DIR   = PROJECT_ROOT / "models" / "cnn_lstm"
GRADCAM_DIR  = PROJECT_ROOT / "results" / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

# ── Fixed constants ───────────────────────────────────────────────────────────
GRADCAM_LAYER = "conv2"    # Last Conv1D before LSTM — Grad-CAM target
SNR_LEVELS = {
    "clean"       : (DATA_PROC / "X_processed.npy",   "Clean",    "steelblue"),
    "snr_high"    : (DATA_NOISY / "snr_high" / "X.npy",  "25 dB",  "#2ecc71"),
    "snr_medium"  : (DATA_NOISY / "snr_medium" / "X.npy", "15 dB", "#f39c12"),
    "snr_low"     : (DATA_NOISY / "snr_low" / "X.npy",   "8 dB",   "#e67e22"),
    "snr_very_low": (DATA_NOISY / "snr_very_low" / "X.npy", "3 dB", "#e74c3c"),
}
LABEL_NAMES = {0: "FALSE POSITIVE", 1: "CONFIRMED", 2: "CANDIDATE"}


# =============================================================================
# 1. LOAD MODEL
# =============================================================================
def load_cnn_lstm(snr_name: str = "snr_high") -> tf.keras.Model:
    """
    Load the saved CNN-LSTM checkpoint.
    Tries snr_high first (best performing), falls back to clean.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from models import focal_loss, F1Score, configure_gpu as cfg_gpu

    candidates = [snr_name, "snr_high", "clean", "snr_medium"]
    for candidate in candidates:
        path = MODELS_DIR / f"cnn_lstm_{candidate}.keras"
        if path.exists():
            logging.info(f"Loading model: {path}")
            model = tf.keras.models.load_model(
                path,
                custom_objects={
                    "focal_loss_fn": focal_loss(gamma=2.0, alpha=0.25),
                    "F1Score": F1Score
                }
            )
            logging.info(f"Model loaded. Layers: "
                         f"{[l.name for l in model.layers]}")
            return model, candidate

    raise FileNotFoundError(
        f"No CNN-LSTM checkpoint found in {MODELS_DIR}. "
        f"Run train.py --model cnn_lstm first."
    )


# =============================================================================
# 2. GRAD-CAM CORE — 1D implementation using GradientTape
# =============================================================================
def compute_gradcam_1d(
    model      : tf.keras.Model,
    curve      : np.ndarray,
    layer_name : str = GRADCAM_LAYER,
) -> tuple[np.ndarray, float]:
    """
    Compute Grad-CAM activation map for a single 1D light curve.

    For a 1D CNN:
        Input  : (1, 201, 1)
        conv2 output : (1, ~50, 128)   ← Grad-CAM target
        CAM    : (50,) → upsample → (201,)

    Returns
    -------
    cam     : float32 array (201,) normalized to [0,1]
    pred    : float — probability of CONFIRMED class
    """
    # Build sub-model outputting (conv layer, final prediction)
    grad_model = tf.keras.Model(
        inputs  = model.input,
        outputs = [model.get_layer(layer_name).output, model.output]
    )

    x = tf.cast(curve[np.newaxis, :, np.newaxis], tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x)
        conv_output, prediction = grad_model(x, training=False)
        # For binary classification: score for CONFIRMED class
        score = prediction[:, 0]

    # Gradient of the CONFIRMED score w.r.t. conv layer output
    grads = tape.gradient(score, conv_output)   # (1, T, C)

    # Global average pooling over time and batch → importance per filter
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))   # (C,)

    # Weight conv outputs by importance
    conv_out = conv_output[0]                             # (T, C)
    cam = conv_out @ pooled_grads[..., tf.newaxis]        # (T, 1)
    cam = tf.squeeze(cam).numpy()                         # (T,)

    # ReLU — only positive contributions matter
    cam = np.maximum(cam, 0)

    # Upsample from conv spatial dim → original 201 bins
    t_cam = len(cam)
    cam_upsampled = np.interp(
        np.linspace(0, 1, 201),
        np.linspace(0, 1, t_cam),
        cam
    )

    # Normalize to [0, 1]
    if cam_upsampled.max() > 1e-8:
        cam_upsampled = cam_upsampled / cam_upsampled.max()

    return cam_upsampled.astype(np.float32), float(prediction[0, 0])


# =============================================================================
# 3. PLOT 1 — SINGLE CURVE: LIGHT CURVE + CAM OVERLAY
# =============================================================================
def plot_gradcam_single(
    curve       : np.ndarray,
    cam         : np.ndarray,
    pred        : float,
    true_label  : int,
    title       : str,
    save_path   : Path,
    snr_label   : str = "Clean",
    color       : str = "steelblue"
):
    """
    Two-panel plot: top = light curve with CAM heatmap overlay,
    bottom = raw CAM activation.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5),
                                    gridspec_kw={'height_ratios': [3, 1]})

    bins = np.arange(201)
    pred_label = "CONFIRMED" if pred >= 0.5 else "FALSE POS"
    correct    = "✓" if (pred >= 0.5) == (true_label == 1) else "✗"

    # ── Top: light curve + heatmap ────────────────────────────────────────────
    ax1.plot(bins, curve, color=color, linewidth=1.0, zorder=2, label="Flux")

    # Heatmap: colour-fill under the curve weighted by CAM
    for i in range(len(bins) - 1):
        intensity = (cam[i] + cam[i+1]) / 2
        ax1.axvspan(bins[i], bins[i+1],
                    alpha=float(intensity) * 0.6,
                    color='red', zorder=1)

    ax1.set_ylabel("Normalised Flux", fontsize=10)
    ax1.set_title(
        f"{title}  |  SNR: {snr_label}  |  "
        f"True: {LABEL_NAMES.get(true_label,'?')}  |  "
        f"Pred: {pred_label} ({pred:.2f}) {correct}",
        fontsize=10, fontweight='bold'
    )
    ax1.set_xlim(0, 200)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # ── Bottom: raw CAM signal ────────────────────────────────────────────────
    ax2.fill_between(bins, cam, alpha=0.7, color='red', label='Grad-CAM')
    ax2.plot(bins, cam, color='darkred', linewidth=0.8)
    ax2.set_xlabel("Phase Bin", fontsize=10)
    ax2.set_ylabel("CAM", fontsize=10)
    ax2.set_xlim(0, 200)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.2)

    # Mark the expected transit centre (bin 100 = phase 0 after folding)
    for ax in [ax1, ax2]:
        ax.axvline(100, color='black', linestyle='--',
                   alpha=0.5, linewidth=1, label='Transit centre')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved: {save_path.name}")


# =============================================================================
# 4. PLOT 2 — CROSS-SNR COMPARISON (same curve, all 4 SNR levels)
# =============================================================================
def plot_gradcam_snr_comparison(
    model       : tf.keras.Model,
    curve_idx   : int,
    y_true      : int,
    save_path   : Path
):
    """
    5-column figure: clean | 25dB | 15dB | 8dB | 3dB.
    Each column: flux curve + CAM overlay.
    Shows how model attention degrades with increasing noise.
    """
    fig, axes = plt.subplots(2, 5, figsize=(22, 5),
                              gridspec_kw={'height_ratios': [3, 1]},
                              sharey='row')

    bins = np.arange(201)

    for col, (snr_key, (x_path, label, color)) in enumerate(SNR_LEVELS.items()):
        X = np.load(x_path)
        curve = X[curve_idx]

        cam, pred = compute_gradcam_1d(model, curve)

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        # Flux
        ax_top.plot(bins, curve, color=color, linewidth=0.9, zorder=2)
        # CAM overlay
        for i in range(200):
            intensity = (cam[i] + cam[i+1]) / 2
            ax_top.axvspan(i, i+1,
                           alpha=float(intensity) * 0.55,
                           color='red', zorder=1)

        # Transit centre
        ax_top.axvline(100, color='black', linestyle='--',
                       alpha=0.4, linewidth=1)
        ax_top.set_title(
            f"{label}\nPred: {'CONF' if pred>=0.5 else 'FP'} ({pred:.2f})",
            fontsize=9, fontweight='bold'
        )
        ax_top.set_xlim(0, 200)
        ax_top.grid(True, alpha=0.2)
        if col == 0:
            ax_top.set_ylabel("Normalised Flux", fontsize=9)

        # CAM bar
        ax_bot.fill_between(bins, cam, alpha=0.7, color='red')
        ax_bot.axvline(100, color='black', linestyle='--',
                       alpha=0.4, linewidth=1)
        ax_bot.set_xlim(0, 200)
        ax_bot.set_ylim(0, 1.05)
        ax_bot.set_xlabel("Phase Bin", fontsize=8)
        if col == 0:
            ax_bot.set_ylabel("CAM", fontsize=9)
        ax_bot.grid(True, alpha=0.2)

    plt.suptitle(
        f"Grad-CAM Across SNR Levels — Curve {curve_idx}  "
        f"(True: {LABEL_NAMES.get(y_true,'?')})\n"
        f"SARIP 2026 | Shri Harsan M | SRM IST | IIT Kanpur",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved: {save_path.name}")


# =============================================================================
# 5. ALIGNMENT ANALYSIS — Does CAM peak near transit centre (bin 100)?
# =============================================================================
def analyse_cam_alignment(
    model  : tf.keras.Model,
    X      : np.ndarray,
    y      : np.ndarray,
    n_samples : int = 100,
    snr_name  : str = "clean"
) -> pd.DataFrame:
    """
    For n_samples curves, compute CAM and check if the peak aligns
    with the expected transit centre (bins 85–115 = ±15 bins around bin 100).

    Returns a DataFrame with per-curve alignment statistics.
    """
    rng = np.random.default_rng(42)

    # Sample confirmed planets and false positives
    conf_idx = np.where(y == 1)[0]
    fp_idx   = np.where(y == 0)[0]
    n_each   = min(n_samples // 2, len(conf_idx), len(fp_idx))

    conf_sample = rng.choice(conf_idx, n_each, replace=False)
    fp_sample   = rng.choice(fp_idx,   n_each, replace=False)
    all_idx     = np.concatenate([conf_sample, fp_sample])

    records = []
    for idx in all_idx:
        curve = X[idx]
        cam, pred = compute_gradcam_1d(model, curve)

        cam_peak_bin  = int(np.argmax(cam))
        peak_in_window = 80 <= cam_peak_bin <= 120   # ±20 bins of centre
        cam_centre_mean = float(cam[80:121].mean())  # mean CAM in transit window
        cam_outside_mean= float(np.concatenate([cam[:80], cam[121:]]).mean())

        records.append({
            "curve_idx"       : int(idx),
            "true_label"      : int(y[idx]),
            "pred_prob"       : float(pred),
            "pred_label"      : int(pred >= 0.5),
            "correct"         : int((pred >= 0.5) == (y[idx] == 1)),
            "cam_peak_bin"    : cam_peak_bin,
            "peak_in_window"  : int(peak_in_window),
            "cam_centre_mean" : cam_centre_mean,
            "cam_outside_mean": cam_outside_mean,
            "cam_centre_ratio": cam_centre_mean / (cam_outside_mean + 1e-8),
            "snr"             : snr_name,
        })

    df = pd.DataFrame(records)
    return df


# =============================================================================
# 6. FAILURE CASE ANALYSIS
# =============================================================================
def find_failure_cases(
    model : tf.keras.Model,
    X     : np.ndarray,
    y     : np.ndarray,
    n_cases : int = 5
) -> list[int]:
    """
    Find curves where:
    - True label = CONFIRMED but model predicts FALSE POSITIVE (false negative)
    - CAM peak is far from transit centre (misaligned attention)
    """
    conf_idx  = np.where(y == 1)[0]
    failures  = []

    for idx in conf_idx:
        cam, pred = compute_gradcam_1d(model, X[idx])
        cam_peak  = np.argmax(cam)
        is_fn     = pred < 0.5                     # false negative
        misaligned= cam_peak < 70 or cam_peak > 130 # peak far from transit

        if is_fn or misaligned:
            failures.append({
                "idx"      : int(idx),
                "pred"     : float(pred),
                "cam_peak" : int(cam_peak),
                "is_fn"    : bool(is_fn),
                "misaligned": bool(misaligned),
            })
        if len(failures) >= n_cases:
            break

    return failures


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_gradcam_analysis():
    configure_gpu()

    logging.info("="*55)
    logging.info("SARIP 2026 — Grad-CAM Interpretability Analysis")
    logging.info("="*55)

    # ── Load model ─────────────────────────────────────────────────────────────
    model, model_snr = load_cnn_lstm("snr_high")
    logging.info(f"Using model trained on: {model_snr}")

    # ── Load clean data for primary analysis ───────────────────────────────────
    X_clean = np.load(DATA_PROC / "X_processed.npy")
    y       = np.load(DATA_PROC / "y_processed.npy").astype(np.int32)

    # Binary filter
    binary_mask = y != 2
    X_clean = X_clean[binary_mask]
    y       = y[binary_mask]

    logging.info(f"Clean data: X={X_clean.shape}  "
                 f"CONF={int((y==1).sum())}  FP={int((y==0).sum())}")

    # ── Select representative curves ───────────────────────────────────────────
    rng       = np.random.default_rng(42)
    conf_idx  = np.where(y == 1)[0]
    fp_idx    = np.where(y == 0)[0]

    # Pick the confirmed curve with median variance (not too clean, not too noisy)
    conf_vars = X_clean[conf_idx].var(axis=1)
    median_conf_idx = conf_idx[np.argsort(conf_vars)[len(conf_vars)//2]]
    sample_fp_idx   = fp_idx[rng.integers(0, len(fp_idx))]

    logging.info(f"Selected CONFIRMED curve: idx={median_conf_idx}")
    logging.info(f"Selected FALSE POSITIVE:  idx={sample_fp_idx}")

    # ── Figure 1: Single CONFIRMED curve (clean, all layers) ─────────────────
    logging.info("\n[1/5] Generating single-curve Grad-CAM (CONFIRMED)...")
    cam_conf, pred_conf = compute_gradcam_1d(model, X_clean[median_conf_idx])
    plot_gradcam_single(
        curve       = X_clean[median_conf_idx],
        cam         = cam_conf,
        pred        = pred_conf,
        true_label  = int(y[median_conf_idx]),
        title       = f"Grad-CAM — Confirmed Planet (idx {median_conf_idx})",
        save_path   = GRADCAM_DIR / "gradcam_confirmed_clean.png",
        snr_label   = "Clean",
        color       = "steelblue"
    )

    # ── Figure 2: Single FALSE POSITIVE curve ────────────────────────────────
    logging.info("[2/5] Generating single-curve Grad-CAM (FALSE POSITIVE)...")
    cam_fp, pred_fp = compute_gradcam_1d(model, X_clean[sample_fp_idx])
    plot_gradcam_single(
        curve       = X_clean[sample_fp_idx],
        cam         = cam_fp,
        pred        = pred_fp,
        true_label  = int(y[sample_fp_idx]),
        title       = f"Grad-CAM — False Positive (idx {sample_fp_idx})",
        save_path   = GRADCAM_DIR / "gradcam_false_positive_clean.png",
        snr_label   = "Clean",
        color       = "#e74c3c"
    )

    # ── Figure 3: Cross-SNR comparison (same CONFIRMED curve) ────────────────
    logging.info("[3/5] Generating cross-SNR Grad-CAM comparison...")
    plot_gradcam_snr_comparison(
        model     = model,
        curve_idx = median_conf_idx,
        y_true    = int(y[median_conf_idx]),
        save_path = GRADCAM_DIR / "gradcam_snr_comparison.png"
    )

    # ── Figure 4: Alignment analysis ─────────────────────────────────────────
    logging.info("[4/5] Running CAM alignment analysis (100 curves)...")
    alignment_df = analyse_cam_alignment(
        model, X_clean, y, n_samples=100, snr_name="clean"
    )
    alignment_df.to_csv(GRADCAM_DIR / "cam_alignment_clean.csv", index=False)

    # Print alignment summary
    conf_df = alignment_df[alignment_df["true_label"] == 1]
    fp_df   = alignment_df[alignment_df["true_label"] == 0]
    logging.info(f"\n── Alignment Summary (clean data) ──")
    logging.info(f"CONFIRMED:     {conf_df['peak_in_window'].mean()*100:.1f}% "
                 f"peaks within ±20 bins of transit centre")
    logging.info(f"FALSE POS:     {fp_df['peak_in_window'].mean()*100:.1f}% "
                 f"peaks within ±20 bins of transit centre")
    logging.info(f"CONFIRMED CAM centre/outside ratio: "
                 f"{conf_df['cam_centre_ratio'].mean():.2f}")
    logging.info(f"FALSE POS  CAM centre/outside ratio: "
                 f"{fp_df['cam_centre_ratio'].mean():.2f}")

    # ── Figure 5: Failure cases ───────────────────────────────────────────────
    logging.info("\n[5/5] Identifying failure cases...")
    failures = find_failure_cases(model, X_clean, y, n_cases=4)

    if failures:
        fig, axes = plt.subplots(2, len(failures),
                                  figsize=(5*len(failures), 6),
                                  gridspec_kw={'height_ratios': [3, 1]})
        if len(failures) == 1:
            axes = axes.reshape(2, 1)

        for col, case in enumerate(failures):
            idx   = case["idx"]
            curve = X_clean[idx]
            cam, pred = compute_gradcam_1d(model, curve)

            ax_top = axes[0, col]
            ax_bot = axes[1, col]
            bins   = np.arange(201)

            ax_top.plot(bins, curve, color='steelblue', linewidth=0.9)
            for i in range(200):
                ax_top.axvspan(i, i+1,
                               alpha=float((cam[i]+cam[i+1])/2)*0.55,
                               color='red')
            ax_top.axvline(100, color='black', linestyle='--',
                           alpha=0.4, linewidth=1)

            reason = []
            if case["is_fn"]:     reason.append("False Negative")
            if case["misaligned"]:reason.append("Misaligned CAM")
            ax_top.set_title(
                f"idx={idx}\n{', '.join(reason)}\nPred={pred:.2f}",
                fontsize=8, color='darkred'
            )
            ax_top.set_xlim(0, 200)
            ax_top.grid(True, alpha=0.2)

            ax_bot.fill_between(bins, cam, alpha=0.7, color='red')
            ax_bot.axvline(100, color='black', linestyle='--', alpha=0.4)
            ax_bot.set_xlim(0, 200)
            ax_bot.set_ylim(0, 1.05)
            ax_bot.set_xlabel("Phase Bin", fontsize=8)
            ax_bot.grid(True, alpha=0.2)

        plt.suptitle(
            "Grad-CAM Failure Cases — CONFIRMED Planets Misclassified or Misaligned\n"
            "SARIP 2026 | Shri Harsan M",
            fontsize=11, y=1.02
        )
        plt.tight_layout()
        plt.savefig(GRADCAM_DIR / "gradcam_failure_cases.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"Failure cases saved: {len(failures)} cases")

        # Save failure case metadata
        pd.DataFrame(failures).to_csv(
            GRADCAM_DIR / "failure_cases.csv", index=False
        )
    else:
        logging.info("No failure cases found in sampled curves.")

    # ── Final Summary ──────────────────────────────────────────────────────────
    logging.info(f"\n{'='*55}")
    logging.info("GRAD-CAM ANALYSIS COMPLETE")
    logging.info(f"Outputs saved → {GRADCAM_DIR}/")
    logging.info(f"  gradcam_confirmed_clean.png   ← Figure: CONFIRMED transit")
    logging.info(f"  gradcam_false_positive_clean.png ← Figure: False positive")
    logging.info(f"  gradcam_snr_comparison.png    ← Figure: Cross-SNR")
    logging.info(f"  gradcam_failure_cases.png     ← Figure: Failures")
    logging.info(f"  cam_alignment_clean.csv       ← Alignment statistics")
    logging.info(f"{'='*55}")

    return alignment_df


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    alignment_df = run_gradcam_analysis()

    # Quick statistics for the report
    print(f"\n── Report Statistics ──────────────────────────────")
    conf = alignment_df[alignment_df["true_label"] == 1]
    fp   = alignment_df[alignment_df["true_label"] == 0]
    print(f"CONFIRMED planets — CAM aligned: "
          f"{conf['peak_in_window'].mean()*100:.1f}%")
    print(f"FALSE POSITIVES   — CAM aligned: "
          f"{fp['peak_in_window'].mean()*100:.1f}%")
    print(f"Model accuracy on sample:        "
          f"{alignment_df['correct'].mean()*100:.1f}%")
    print(f"\nAll Grad-CAM figures saved to: results/gradcam/")
