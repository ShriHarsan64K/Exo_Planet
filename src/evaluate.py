# =============================================================================
# SARIP 2026 — IIT Kanpur
# src/evaluate.py
# Evaluation, Metrics Reporting, and Degradation Curve Plotting
#
# Author : Shri Harsan M | RA2512052010014 | SRM IST
# Week   : 3–6 (called after every training run)
#
# Usage:
#   # Print full results table from saved CSV:
#   python src/evaluate.py --report
#
#   # Plot performance degradation curves:
#   python src/evaluate.py --plot
#
#   # Both:
#   python src/evaluate.py --report --plot
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results" / "metrics"
PLOTS_DIR    = PROJECT_ROOT / "results" / "gradcam"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── SNR order for x-axis (high → very low) ───────────────────────────────────
SNR_ORDER = ["clean", "snr_high", "snr_medium", "snr_low", "snr_very_low"]
SNR_LABELS = {
    "clean"       : "Clean",
    "snr_high"    : "25 dB\n(High)",
    "snr_medium"  : "15 dB\n(Medium)",
    "snr_low"     : "8 dB\n(Low)",
    "snr_very_low": "3 dB\n(Very Low)",
}
SNR_DB = {
    "clean": 40, "snr_high": 25, "snr_medium": 15,
    "snr_low": 8, "snr_very_low": 3
}

# ── Model display config ──────────────────────────────────────────────────────
MODEL_STYLE = {
    "lr"      : {"label": "Logistic Regression", "color": "#7f8c8d",
                 "ls": "--",  "marker": "o"},
    "rf"      : {"label": "Random Forest",        "color": "#95a5a6",
                 "ls": "--",  "marker": "s"},
    "cnn"     : {"label": "CNN-only",             "color": "#3498db",
                 "ls": "-.",  "marker": "^"},
    "lstm"    : {"label": "LSTM-only",            "color": "#e67e22",
                 "ls": "-.",  "marker": "D"},
    "cnn_lstm": {"label": "CNN-LSTM (Proposed)",  "color": "#e74c3c",
                 "ls": "-",   "marker": "*"},
}


# =============================================================================
# LOAD RESULTS CSV
# =============================================================================
def load_results(csv_name: str = "all_results.csv") -> pd.DataFrame:
    csv_path = RESULTS_DIR / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {csv_path}\n"
            f"Run train.py first to generate results."
        )
    df = pd.read_csv(csv_path)
    # Enforce SNR order
    df["snr"] = pd.Categorical(df["snr"], categories=SNR_ORDER, ordered=True)
    df = df.sort_values(["model", "snr"]).reset_index(drop=True)
    return df


# =============================================================================
# PRINT FULL METRICS TABLE
# =============================================================================
def print_metrics_table(df: pd.DataFrame):
    """Print a formatted table of all results."""
    print(f"\n{'='*80}")
    print("SARIP 2026 — Full Results Table")
    print(f"{'='*80}")

    # ── Macro metrics pivot ───────────────────────────────────────────────────
    for metric in ["f1", "roc_auc", "pr_auc"]:
        pivot = df.pivot_table(
            index="model", columns="snr",
            values=metric, aggfunc="first"
        )
        # Reorder columns
        cols = [c for c in SNR_ORDER if c in pivot.columns]
        pivot = pivot[cols]
        pivot.columns = [SNR_LABELS[c].replace('\n', ' ') for c in cols]

        print(f"\n── {metric.upper()} ──")
        print(pivot.round(4).to_string())

    # ── Per-class F1 for CONFIRMED class ─────────────────────────────────────
    if "f1_conf" in df.columns:
        pivot_conf = df.pivot_table(
            index="model", columns="snr",
            values="f1_conf", aggfunc="first"
        )
        cols = [c for c in SNR_ORDER if c in pivot_conf.columns]
        pivot_conf = pivot_conf[cols]
        pivot_conf.columns = [SNR_LABELS[c].replace('\n', ' ') for c in cols]

        print(f"\n── F1 (CONFIRMED class only — most important) ──")
        print(pivot_conf.round(4).to_string())

    print(f"\n{'='*80}")

    # Save formatted table to CSV
    summary_path = RESULTS_DIR / "summary_table.csv"
    df.to_csv(summary_path, index=False)
    logging.info(f"Full table saved → {summary_path}")


# =============================================================================
# PLOT 1: PERFORMANCE DEGRADATION CURVES
# F1-Score and ROC-AUC vs SNR Level (THE central figure of the project)
# =============================================================================
def plot_degradation_curves(df: pd.DataFrame, models_to_plot: list = None):
    """
    The core research output: performance vs SNR level per model.
    This is Figure 2 of the research report.
    """
    if models_to_plot is None:
        models_to_plot = df["model"].unique().tolist()

    snr_present  = [s for s in SNR_ORDER if s in df["snr"].values]
    x_labels     = [SNR_LABELS[s] for s in snr_present]
    x_vals       = [SNR_DB[s] for s in snr_present]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric, ax, title in [
        ("f1",      axes[0], "F1-Score (Macro)"),
        ("roc_auc", axes[1], "ROC-AUC"),
    ]:
        for model_name in models_to_plot:
            style  = MODEL_STYLE.get(model_name, {})
            subset = df[df["model"] == model_name].copy()
            subset = subset.sort_values("snr")

            snr_vals = [s for s in snr_present if s in subset["snr"].values]
            y_vals   = [subset[subset["snr"] == s][metric].values[0]
                        for s in snr_vals]
            x_plot   = [SNR_DB[s] for s in snr_vals]

            ax.plot(
                x_plot, y_vals,
                label   = style.get("label", model_name),
                color   = style.get("color", "black"),
                linestyle = style.get("ls", "-"),
                marker  = style.get("marker", "o"),
                markersize = 7,
                linewidth  = 2.0
            )

        ax.set_xlabel("SNR Level (dB)", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"{title} vs SNR Level", fontsize=12, fontweight='bold')
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.invert_xaxis()   # High SNR → Low SNR (left to right)
        ax.legend(fontsize=9, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Mark the "cliff" — where CNN-LSTM advantage is largest
        ax.axvline(x=8, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(8.5, 0.05, '← Robustness\nfrontier', fontsize=8,
                color='gray', alpha=0.7)

    plt.suptitle(
        "Performance Degradation vs SNR Level\n"
        "",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    save_path = PLOTS_DIR / "degradation_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logging.info(f"Degradation curves saved → {save_path}")
    plt.show()
    plt.close()


# =============================================================================
# PLOT 2: HEATMAP — All Metrics × All Models × All SNR Levels
# =============================================================================
def plot_heatmap(df: pd.DataFrame, metric: str = "f1"):
    """
    Heatmap of one metric across all model × SNR combinations.
    Useful for spotting patterns at a glance.
    """
    snr_present = [s for s in SNR_ORDER if s in df["snr"].values]

    pivot = df.pivot_table(
        index="model", columns="snr",
        values=metric, aggfunc="first"
    )
    cols = [c for c in snr_present if c in pivot.columns]
    pivot = pivot[cols]
    pivot.columns = [SNR_LABELS[c].replace('\n', ' ') for c in cols]
    pivot.index   = [MODEL_STYLE.get(m, {}).get("label", m)
                     for m in pivot.index]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".3f",
        cmap="RdYlGn", vmin=0.4, vmax=1.0,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 10}
    )
    ax.set_title(f"{metric.upper()} — All Models × All SNR Levels\n"
                 f"Exoplanet Transit Detection",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("SNR Level", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)
    plt.tight_layout()

    save_path = PLOTS_DIR / f"heatmap_{metric}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logging.info(f"Heatmap saved → {save_path}")
    plt.show()
    plt.close()


# =============================================================================
# PLOT 3: BAR CHART — Model Comparison at a Single SNR Level
# =============================================================================
def plot_model_comparison(df: pd.DataFrame, snr_name: str = "snr_high"):
    """Bar chart comparing all models at one SNR level."""
    subset  = df[df["snr"] == snr_name].copy()
    metrics = ["f1", "roc_auc", "pr_auc"]

    x     = np.arange(len(subset))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = subset[metric].values
        bars = ax.bar(x + i * width, vals, width,
                      label=metric.upper().replace("_", "-"),
                      color=color, alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [MODEL_STYLE.get(m, {}).get("label", m) for m in subset["model"]],
        rotation=15, ha='right'
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Model Comparison at SNR = "
                 f"{SNR_LABELS.get(snr_name, snr_name).replace(chr(10), ' ')}\n"
                 f"Exoplanet Transit Detection",
                 fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    save_path = PLOTS_DIR / f"comparison_{snr_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logging.info(f"Comparison bar chart saved → {save_path}")
    plt.show()
    plt.close()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SARIP 2026 — Evaluation")
    parser.add_argument("--report", action="store_true",
                        help="Print full metrics table")
    parser.add_argument("--plot",   action="store_true",
                        help="Generate all plots")
    parser.add_argument("--snr",    type=str, default="snr_high",
                        help="SNR level for bar chart comparison")
    args = parser.parse_args()

    df = load_results()
    logging.info(f"Loaded {len(df)} result rows for "
                 f"{df['model'].nunique()} models × "
                 f"{df['snr'].nunique()} SNR levels")

    if args.report:
        print_metrics_table(df)

    if args.plot:
        plot_degradation_curves(df)
        for metric in ["f1", "roc_auc"]:
            plot_heatmap(df, metric)
        plot_model_comparison(df, snr_name=args.snr)

    if not args.report and not args.plot:
        print_metrics_table(df)
        plot_degradation_curves(df)
