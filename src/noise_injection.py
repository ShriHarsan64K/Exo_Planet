# =============================================================================
# SARIP 2026 — IIT Kanpur
# src/noise_injection.py
# Week 2: Synthetic Noise Injection at 4 Fixed SNR Levels
#
# Author : Shri Harsan M | RA2512052010014 | SRM IST
#
# SNR Levels (LOCKED — do not change):
#   High     25 dB  → Clean Kepler pipeline
#   Medium   15 dB  → Moderate instrument noise
#   Low       8 dB  → Ground-based / early processing
#   Very Low  3 dB  → Severely degraded / raw data
#
# Run: conda activate sarip && python src/noise_injection.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
DATA_NOISY   = PROJECT_ROOT / "data" / "noisy"

# ── Fixed SNR Levels — DO NOT CHANGE ─────────────────────────────────────────
SNR_LEVELS = {
    "snr_high"     : 25,   # dB
    "snr_medium"   : 15,   # dB
    "snr_low"      :  8,   # dB
    "snr_very_low" :  3,   # dB
}


# =============================================================================
# CORE: ADD GAUSSIAN NOISE AT A TARGET SNR
# =============================================================================
def add_gaussian_noise(
    X          : np.ndarray,
    snr_db     : float,
    rng        : np.random.Generator
) -> np.ndarray:
    """
    Add Gaussian noise to each light curve to achieve a target SNR in dB.

    SNR (dB) = 10 * log10(signal_power / noise_power)
    → noise_std = sqrt(signal_power / 10^(snr_db/10))

    Signal power is computed per-curve so noise scales with each
    individual curve's amplitude — physically correct.

    Parameters
    ----------
    X      : float32 array of shape (N, 201)
    snr_db : target SNR in decibels
    rng    : seeded numpy Generator

    Returns
    -------
    X_noisy : float32 array of shape (N, 201), clipped to [0, 1]
    """
    X_noisy = np.zeros_like(X, dtype=np.float32)

    for i in range(len(X)):
        curve = X[i]

        # Signal power = mean of squared values
        signal_power = np.mean(curve ** 2)

        # If curve is flat (all zeros after masking), use a small default
        if signal_power < 1e-10:
            signal_power = 1e-6

        # Compute required noise standard deviation
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise_std   = np.sqrt(noise_power)

        # Generate and add noise
        noise            = rng.normal(loc=0.0, scale=noise_std, size=curve.shape)
        X_noisy[i]       = np.clip(curve + noise, 0.0, 1.0).astype(np.float32)

    return X_noisy


# =============================================================================
# SAVE ONE SNR VERSION
# =============================================================================
def save_snr_version(
    X_noisy  : np.ndarray,
    y        : np.ndarray,
    snr_name : str,
    snr_db   : float
):
    """Save one noisy version to data/noisy/<snr_name>/."""
    out_dir = DATA_NOISY / snr_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X.npy", X_noisy)
    np.save(out_dir / "y.npy", y)

    # Save a small summary CSV
    summary = pd.DataFrame({
        "snr_name"  : [snr_name],
        "snr_db"    : [snr_db],
        "n_curves"  : [len(X_noisy)],
        "n_class_0" : [int((y == 0).sum())],
        "n_class_1" : [int((y == 1).sum())],
        "n_class_2" : [int((y == 2).sum())],
        "x_mean"    : [float(X_noisy.mean())],
        "x_std"     : [float(X_noisy.std())],
    })
    summary.to_csv(out_dir / "summary.csv", index=False)

    logging.info(f"  Saved → {out_dir}/  "
                 f"[X={X_noisy.shape}, mean={X_noisy.mean():.4f}, "
                 f"std={X_noisy.std():.4f}]")


# =============================================================================
# VISUALISE: PLOT ONE CURVE AT ALL 4 SNR LEVELS
# =============================================================================
def plot_snr_comparison(
    X_clean      : np.ndarray,
    noisy_dict   : dict,
    curve_idx    : int = 0,
    save_path    : Path = None
):
    """
    Plot one light curve at all 4 SNR levels side by side.
    Useful for sanity-checking noise injection visually.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 3), sharey=True)

    # Clean
    axes[0].plot(X_clean[curve_idx], color="steelblue", linewidth=0.8)
    axes[0].set_title("Clean (processed)", fontsize=10)
    axes[0].set_xlabel("Phase bin")
    axes[0].set_ylabel("Normalised flux")

    # Each SNR level
    for ax, (snr_name, (X_noisy, snr_db)) in zip(axes[1:], noisy_dict.items()):
        ax.plot(X_noisy[curve_idx], color="coral", linewidth=0.8, alpha=0.8)
        ax.set_title(f"{snr_name.replace('snr_','').replace('_',' ').title()}\n"
                     f"({snr_db} dB)", fontsize=10)
        ax.set_xlabel("Phase bin")

    plt.suptitle(f"Noise Injection — Curve {curve_idx}", fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        logging.info(f"  Plot saved → {save_path}")

    plt.show()
    plt.close()


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_noise_injection():
    # ── Load processed dataset ────────────────────────────────────────────────
    X_path = DATA_PROC / "X_processed.npy"
    y_path = DATA_PROC / "y_processed.npy"

    if not X_path.exists():
        logging.error(f"X_processed.npy not found at {X_path}")
        logging.error("Run preprocessing.py / download_full.py first.")
        return

    X = np.load(X_path)   # (6916, 201)
    y = np.load(y_path)   # (6916,)

    logging.info(f"Loaded: X={X.shape}  y={y.shape}")
    logging.info(f"X range: [{X.min():.4f}, {X.max():.4f}]")
    logging.info(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    logging.info("")

    rng        = np.random.default_rng(42)
    noisy_dict = {}   # for plotting

    # ── Inject noise at each SNR level ────────────────────────────────────────
    logging.info("Injecting noise at all 4 SNR levels...")
    logging.info("=" * 55)

    for snr_name, snr_db in SNR_LEVELS.items():
        logging.info(f"\n[{snr_name.upper()}]  SNR = {snr_db} dB")

        X_noisy = add_gaussian_noise(X, snr_db, rng)
        save_snr_version(X_noisy, y, snr_name, snr_db)
        noisy_dict[snr_name] = (X_noisy, snr_db)

    logging.info("\n" + "=" * 55)
    logging.info("All 4 SNR versions saved.\n")

    # ── Sanity check: verify files exist and sizes match ─────────────────────
    logging.info("Verifying saved files...")
    all_ok = True
    for snr_name in SNR_LEVELS:
        x_file = DATA_NOISY / snr_name / "X.npy"
        y_file = DATA_NOISY / snr_name / "y.npy"
        if x_file.exists() and y_file.exists():
            x_check = np.load(x_file)
            logging.info(f"  ✅ {snr_name}: X={x_check.shape}")
        else:
            logging.error(f"  ❌ {snr_name}: files missing!")
            all_ok = False

    # ── Visualise one CONFIRMED transit at all SNR levels ────────────────────
    confirmed_idx = np.where(y == 1)[0][0]   # first confirmed planet
    plot_snr_comparison(
        X_clean    = X,
        noisy_dict = noisy_dict,
        curve_idx  = confirmed_idx,
        save_path  = PROJECT_ROOT / "results" / "gradcam" / "snr_comparison.png"
    )

    if all_ok:
        logging.info("\n✅ noise_injection.py complete.")
        logging.info("   Next step: run src/models.py (Week 3 — baselines)")
    else:
        logging.warning("\n⚠️  Some files missing — rerun noise_injection.py")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    logging.info("=" * 55)
    logging.info("SARIP 2026 — Noise Injection Pipeline")
    logging.info(f"Input  : {DATA_PROC}")
    logging.info(f"Output : {DATA_NOISY}")
    logging.info("=" * 55)
    run_noise_injection()
