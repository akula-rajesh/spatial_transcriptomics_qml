#!/usr/bin/env python3
"""
Training Visualization Report Generator
========================================
Merges multiple results.json files from consecutive training runs
(e.g. run1 → resume → run2 → resume → run3) into a single
continuous training curve and generates a full HTML + PNG report.

Usage
-----
  # Pass individual results.json paths
  python scripts/visualize_training.py \\
      results/experiment_20260319_220858/20260319_220858/results.json \\
      results/experiment_20260320_113554/20260320_113554/results.json \\
      results/experiment_20260320_152035/20260320_152035/results.json

  # Or pass experiment run directories (results.json is found automatically)
  python scripts/visualize_training.py \\
      results/experiment_20260319_220858/20260319_220858 \\
      results/experiment_20260320_113554/20260320_113554

  # Save to a specific output directory
  python scripts/visualize_training.py run1/results.json run2/results.json \\
      --output reports/combined_training

  # Show interactive plots instead of saving
  python scripts/visualize_training.py run1/results.json --show
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# ── Optional heavy imports — fail gracefully with a clear message ──────────────
try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required.  pip install numpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (works on servers too)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    print("WARNING: matplotlib not found — PNG plots will be skipped.  pip install matplotlib")

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def find_results_json(path: str) -> Path:
    """Accept either a results.json file OR the directory that contains it."""
    p = Path(path)
    if p.is_file() and p.suffix == ".json":
        return p
    if p.is_dir():
        candidate = p / "results.json"
        if candidate.exists():
            return candidate
        # Search one level deeper (experiment_xxx/TIMESTAMP/results.json)
        for child in sorted(p.iterdir()):
            if child.is_dir():
                candidate = child / "results.json"
                if candidate.exists():
                    return candidate
    raise FileNotFoundError(f"Could not find results.json at or inside: {path}")


def load_run(results_file: Path) -> Dict[str, Any]:
    """Load and validate a single results.json."""
    with open(results_file) as f:
        data = json.load(f)

    # Normalise: always expect data["training"][<model>] structure
    if "training" not in data:
        raise ValueError(f"No 'training' key in {results_file}")

    data["_source_file"] = str(results_file)
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Merging logic
# ──────────────────────────────────────────────────────────────────────────────

def merge_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Concatenate multiple training runs into one continuous timeline.

    For each model found across runs, epoch indices are shifted so that
    run2 starts where run1 ended, giving a single unbroken curve.

    Returns
    -------
    dict keyed by model_name →
        {
          train_losses, val_losses,
          train_metrics, val_metrics,   # per-epoch dicts
          best_epoch, best_val_loss,
          run_boundaries,               # [0, 30, 60, …]  epoch indices where new run starts
          run_labels,                   # ["Run 1 (exp_…)", …]
          total_epochs,
          training_time_s,
        }
    """
    merged: Dict[str, Any] = {}

    for run_idx, run in enumerate(runs):
        source  = Path(run["_source_file"]).parent.name  # timestamp folder name
        exp     = Path(run["_source_file"]).parent.parent.name

        for model_name, model_data in run.get("training", {}).items():
            if not isinstance(model_data, dict):
                continue

            if model_name not in merged:
                merged[model_name] = {
                    "train_losses":     [],
                    "val_losses":       [],
                    "train_metrics":    [],
                    "val_metrics":      [],
                    "run_boundaries":   [0],
                    "run_labels":       [],
                    "training_time_s":  0.0,
                    "best_val_loss":    float("inf"),
                    "best_epoch":       0,
                    "total_epochs":     0,
                }

            m = merged[model_name]
            offset = len(m["train_losses"])   # epochs already accumulated

            # Append epoch arrays
            tl = model_data.get("train_losses", [])
            vl = model_data.get("val_losses",   [])
            tm = model_data.get("train_metrics", [])
            vm = model_data.get("val_metrics",   [])

            m["train_losses"].extend(tl)
            m["val_losses"].extend(vl)
            m["train_metrics"].extend(tm)
            m["val_metrics"].extend(vm)

            # Track global best
            best_epoch_local = model_data.get("best_epoch", 0)
            best_loss_local  = model_data.get("best_val_loss", float("inf"))
            if best_loss_local < m["best_val_loss"]:
                m["best_val_loss"] = best_loss_local
                m["best_epoch"]    = offset + best_epoch_local

            # Accumulate training time
            m["training_time_s"] += model_data.get("training_time_s", 0.0)

            # Boundary marker for vertical lines on plots
            if offset > 0:
                m["run_boundaries"].append(offset)

            label = f"Run {run_idx + 1} ({exp})"
            m["run_labels"].append(label)


    # Final total_epochs pass
    for model_name, m in merged.items():
        m["total_epochs"] = len(m["train_losses"])

    return merged


def extract_metric_series(epoch_dicts: List[Dict], key: str) -> List[float]:
    """Extract a named metric from a list of per-epoch metric dicts."""
    return [float(d.get(key, float("nan"))) for d in epoch_dicts if isinstance(d, dict)]


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    "train":       "#2196F3",   # blue
    "val":         "#F44336",   # red
    "best":        "#4CAF50",   # green
    "boundary":    "#9C27B0",   # purple
    "amae":        "#FF9800",   # orange
    "armse":       "#009688",   # teal
    "acc":         "#E91E63",   # pink
    "train_light": "#BBDEFB",
    "val_light":   "#FFCDD2",
}

METRIC_META = {
    "amae":                   ("aMAE",   "↓ lower is better", COLORS["amae"]),
    "armse":                  ("aRMSE",  "↓ lower is better", COLORS["armse"]),
    "correlation_coefficient":("aCC",    "↑ higher is better", COLORS["acc"]),
}


def _draw_run_boundaries(ax, boundaries: List[int], labels: List[str]):
    """Draw vertical dashed lines at run boundaries with a label."""
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)  # x=data, y=axes (0-1)
    for i, boundary in enumerate(boundaries):
        if boundary == 0:
            continue
        ax.axvline(x=boundary, color=COLORS["boundary"], linestyle="--",
                   linewidth=1.2, alpha=0.7)
        label = labels[i] if i < len(labels) else f"Run {i+1}"
        ax.text(boundary + 0.3, 0.97, label,
                transform=trans,
                fontsize=7, color=COLORS["boundary"], va="top", rotation=90, alpha=0.8)


def plot_loss_curve(ax, model_name: str, m: Dict) -> None:
    """Plot train + val loss with run boundaries and best-epoch marker."""
    epochs = list(range(len(m["train_losses"])))

    ax.plot(epochs, m["train_losses"], color=COLORS["train"],
            linewidth=1.8, label="Train Loss", zorder=3)
    ax.plot(epochs, m["val_losses"],   color=COLORS["val"],
            linewidth=1.8, label="Val Loss",   zorder=3)

    # Best epoch marker
    be = m["best_epoch"]
    if 0 <= be < len(m["val_losses"]):
        ax.axvline(x=be, color=COLORS["best"], linestyle=":", linewidth=1.5,
                   label=f"Best epoch ({be+1})", zorder=2)
        ax.scatter([be], [m["val_losses"][be]], color=COLORS["best"],
                   s=60, zorder=4)

    # Run boundary lines
    _draw_run_boundaries(ax, m["run_boundaries"], m["run_labels"])

    ax.set_title(f"{model_name} — Loss Curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch (continuous across all runs)")
    ax.set_ylabel("MSE Loss")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)


def plot_metric_curve(ax, model_name: str, m: Dict, metric_key: str) -> None:
    """Plot a single metric (aMAE / aRMSE / aCC) for train and val."""
    label_short, direction, color = METRIC_META.get(
        metric_key, (metric_key, "", "#607D8B")
    )

    train_series = extract_metric_series(m["train_metrics"], metric_key)
    val_series   = extract_metric_series(m["val_metrics"],   metric_key)

    epochs_t = list(range(len(train_series)))
    epochs_v = list(range(len(val_series)))

    if train_series:
        ax.plot(epochs_t, train_series, color=color, linewidth=1.8,
                linestyle="-",  label=f"Train {label_short}", zorder=3)
    if val_series:
        ax.plot(epochs_v, val_series,   color=color, linewidth=1.8,
                linestyle="--", label=f"Val {label_short}",   zorder=3, alpha=0.8)

    _draw_run_boundaries(ax, m["run_boundaries"], m["run_labels"])

    ax.set_title(f"{model_name} — {label_short}  ({direction})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch (continuous across all runs)")
    ax.set_ylabel(label_short)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)


def plot_metric_summary_bar(ax, merged: Dict[str, Dict]) -> None:
    """Bar chart comparing final val metrics across models."""
    metric_keys  = ["amae", "armse", "correlation_coefficient"]
    metric_labels = ["aMAE ↓", "aRMSE ↓", "aCC ↑"]
    colors = [COLORS["amae"], COLORS["armse"], COLORS["acc"]]

    model_names = list(merged.keys())
    x = np.arange(len(metric_keys))
    width = 0.8 / max(len(model_names), 1)

    for i, model_name in enumerate(model_names):
        m = merged[model_name]
        # Use last val_metrics entry for final values
        final = m["val_metrics"][-1] if m["val_metrics"] else {}
        values = [float(final.get(k, 0)) for k in metric_keys]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name,
                      color=colors, alpha=0.75, edgecolor="white")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_title("Final Validation Metrics", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def plot_training_time(ax, merged: Dict[str, Dict], runs: List[Dict]) -> None:
    """Stacked bar showing training time per run per model."""
    model_names = list(merged.keys())

    for model_idx, model_name in enumerate(model_names):
        run_times = []
        run_labels = []
        for run_idx, run in enumerate(runs):
            model_data = run.get("training", {}).get(model_name, {})
            t = model_data.get("training_time_s", 0)
            run_times.append(t / 3600)   # convert to hours
            exp = Path(run["_source_file"]).parent.parent.name
            run_labels.append(f"Run {run_idx+1}")

        x = np.arange(len(run_times))
        ax.bar(x + model_idx * 0.35, run_times, 0.35,
               label=model_name, alpha=0.8)

    n_runs = len(runs)
    ax.set_xticks(np.arange(n_runs) + (len(model_names) - 1) * 0.175)
    ax.set_xticklabels([f"Run {i+1}" for i in range(n_runs)])
    ax.set_title("Training Time per Run", fontsize=11, fontweight="bold")
    ax.set_ylabel("Hours")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


# ──────────────────────────────────────────────────────────────────────────────
# HTML report
# ──────────────────────────────────────────────────────────────────────────────

def build_html_report(
    merged: Dict[str, Dict],
    runs: List[Dict],
    png_paths: List[Path],
    output_dir: Path,
) -> Path:
    """Generate a self-contained HTML report with embedded images."""

    import base64

    def img_tag(path: Path) -> str:
        if not path.exists():
            return f"<p>Image not found: {path}</p>"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return (f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:100%;border-radius:6px;margin:8px 0">')

    rows_html = ""
    for model_name, m in merged.items():
        best_val  = m.get("best_val_loss", float("nan"))
        best_ep   = m.get("best_epoch", 0) + 1
        total_ep  = m.get("total_epochs", 0)
        total_hrs = m.get("training_time_s", 0) / 3600

        # Final validation metrics from last epoch
        final = m["val_metrics"][-1] if m["val_metrics"] else {}
        amae  = final.get("amae",                   float("nan"))
        armse = final.get("armse",                  float("nan"))
        acc   = final.get("correlation_coefficient", float("nan"))

        rows_html += f"""
        <tr>
          <td><b>{model_name}</b></td>
          <td>{total_ep}</td>
          <td>{best_ep}</td>
          <td>{best_val:.6f}</td>
          <td>{amae:.6f}</td>
          <td>{armse:.6f}</td>
          <td>{acc:.6f}</td>
          <td>{total_hrs:.2f}h</td>
        </tr>"""

    run_rows = ""
    for i, run in enumerate(runs):
        src  = run.get("_source_file", "?")
        exp  = Path(src).parent.parent.name
        ts   = Path(src).parent.name
        stat = run.get("status", run.get("metadata", {}).get("status", "?"))

        for model_name, model_data in run.get("training", {}).items():
            ep = model_data.get("total_epochs", len(model_data.get("train_losses", [])))
            bvl = model_data.get("best_val_loss", float("nan"))
            hrs = model_data.get("training_time_s", 0) / 3600
            run_rows += f"""
            <tr>
              <td>Run {i+1}</td>
              <td>{model_name}</td>
              <td>{exp}</td>
              <td>{ts}</td>
              <td>{ep}</td>
              <td>{bvl:.6f}</td>
              <td>{hrs:.2f}h</td>
              <td>{stat}</td>
            </tr>"""

    plots_html = "".join(img_tag(p) for p in png_paths if p.exists())

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Training Visualization Report</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px;
            background: #f5f5f5; color: #333; }}
    h1   {{ color: #1565C0; border-bottom: 3px solid #1565C0; padding-bottom: 8px; }}
    h2   {{ color: #1976D2; margin-top: 32px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0;
             background: white; border-radius: 8px; overflow: hidden;
             box-shadow: 0 2px 6px rgba(0,0,0,.1); }}
    th   {{ background: #1565C0; color: white; padding: 10px 14px; text-align: left; }}
    td   {{ padding: 8px 14px; border-bottom: 1px solid #e0e0e0; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #E3F2FD; }}
    .badge {{ display:inline-block; padding:2px 8px; border-radius:12px;
              font-size:.8em; font-weight:bold; }}
    .badge-ok   {{ background:#C8E6C9; color:#2E7D32; }}
    .badge-fail {{ background:#FFCDD2; color:#C62828; }}
    .meta {{ color:#666; font-size:.9em; margin-bottom:24px; }}
    .section {{ background:white; border-radius:8px; padding:20px;
                margin:16px 0; box-shadow:0 2px 6px rgba(0,0,0,.1); }}
  </style>
</head>
<body>
<h1>📊 Training Visualization Report</h1>
<p class="meta">Generated: {generated} &nbsp;|&nbsp; Runs merged: {len(runs)}</p>

<div class="section">
  <h2>📋 Combined Model Summary</h2>
  <table>
    <tr>
      <th>Model</th><th>Total Epochs</th><th>Best Epoch</th>
      <th>Best Val Loss</th><th>Final aMAE</th><th>Final aRMSE</th>
      <th>Final aCC</th><th>Total Time</th>
    </tr>
    {rows_html}
  </table>
</div>

<div class="section">
  <h2>🔄 Individual Runs</h2>
  <table>
    <tr>
      <th>#</th><th>Model</th><th>Experiment</th><th>Timestamp</th>
      <th>Epochs</th><th>Best Val Loss</th><th>Time</th><th>Status</th>
    </tr>
    {run_rows}
  </table>
</div>

<div class="section">
  <h2>📈 Training Curves</h2>
  <p style="color:#666;font-size:.9em">
    Purple dashed lines mark resume boundaries (where a new run started from checkpoint).
    Green dotted line marks the global best epoch.
  </p>
  {plots_html}
</div>

</body>
</html>"""

    out = output_dir / "training_report.html"
    out.write_text(html, encoding="utf-8")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Summary table (terminal)
# ──────────────────────────────────────────────────────────────────────────────

def print_summary_table(merged: Dict[str, Dict], runs: List[Dict]) -> None:
    """Print a concise ASCII summary to stdout."""
    sep = "─" * 90
    print(f"\n{sep}")
    print(f"  MERGED TRAINING SUMMARY  ({len(runs)} run(s))")
    print(sep)
    fmt = "{:<30} {:>8} {:>10} {:>12} {:>10} {:>10} {:>10}"
    print(fmt.format("Model", "Epochs", "BestEpoch", "BestValLoss",
                     "FinalAMAE", "FinalARMSE", "FinalACC"))
    print(sep)

    for model_name, m in merged.items():
        final = m["val_metrics"][-1] if m["val_metrics"] else {}
        print(fmt.format(
            model_name[:29],
            m["total_epochs"],
            m["best_epoch"] + 1,
            f"{m['best_val_loss']:.6f}",
            f"{final.get('amae', float('nan')):.6f}",
            f"{final.get('armse', float('nan')):.6f}",
            f"{final.get('correlation_coefficient', float('nan')):.6f}",
        ))

    print(sep)
    print(f"\nRun details:")
    for i, run in enumerate(runs):
        src = run.get("_source_file", "?")
        for model_name, model_data in run.get("training", {}).items():
            ep  = len(model_data.get("train_losses", []))
            bvl = model_data.get("best_val_loss", float("nan"))
            hrs = model_data.get("training_time_s", 0) / 3600
            print(f"  Run {i+1}  {model_name:<28}  {ep:>3} epochs  "
                  f"best_val={bvl:.6f}  time={hrs:.2f}h  [{src}]")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple training run results.json files and generate visualization report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="Paths to results.json files or experiment run directories (in chronological order)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for report files (default: reports/<timestamp>)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive matplotlib windows instead of saving PNGs",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML report generation",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG resolution (default: 150)",
    )

    args = parser.parse_args()

    # ── Resolve and load result files ─────────────────────────────────────────
    print(f"\nLoading {len(args.results)} result file(s)...")
    runs: List[Dict] = []
    for path_str in args.results:
        try:
            rfile = find_results_json(path_str)
            run   = load_run(rfile)
            runs.append(run)
            print(f"  ✓ {rfile}")
        except (FileNotFoundError, ValueError) as e:
            print(f"  ✗ {path_str}: {e}")
            sys.exit(1)

    if not runs:
        print("No valid result files found. Exiting.")
        sys.exit(1)

    # ── Merge ─────────────────────────────────────────────────────────────────
    print(f"\nMerging {len(runs)} run(s)...")
    merged = merge_runs(runs)

    # ── Terminal summary ───────────────────────────────────────────────────────
    print_summary_table(merged, runs)

    # ── Output directory ───────────────────────────────────────────────────────
    if args.output:
        output_dir = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("reports") / f"training_report_{ts}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if not MATPLOTLIB_OK:
        print("Skipping plots (matplotlib not available).")
        return

    if args.show:
        matplotlib.use("TkAgg")  # switch to interactive

    png_paths: List[Path] = []

    for model_name, m in merged.items():
        n_metrics = len(METRIC_META)
        fig = plt.figure(figsize=(16, 5 * (2 + n_metrics)), constrained_layout=True)
        fig.suptitle(
            f"Training Report — {model_name}\n"
            f"{len(runs)} run(s) merged  |  "
            f"{m['total_epochs']} total epochs  |  "
            f"Best val loss: {m['best_val_loss']:.6f} (epoch {m['best_epoch']+1})",
            fontsize=13, fontweight="bold",
        )

        gs = gridspec.GridSpec(2 + n_metrics, 1, figure=fig)

        # ── Row 0: Loss curve ──────────────────────────────────────────────────
        ax_loss = fig.add_subplot(gs[0])
        plot_loss_curve(ax_loss, model_name, m)

        # ── Row 1: Summary bar chart ───────────────────────────────────────────
        ax_bar = fig.add_subplot(gs[1])
        plot_metric_summary_bar(ax_bar, merged)

        # ── Row 2+: Per-metric curves ──────────────────────────────────────────
        for row_i, (metric_key, _) in enumerate(METRIC_META.items()):
            ax_m = fig.add_subplot(gs[2 + row_i])
            plot_metric_curve(ax_m, model_name, m, metric_key)

        # Save or show
        safe_name = model_name.replace("/", "_").replace(" ", "_")
        if args.show:
            plt.show()
        else:
            png_path = output_dir / f"{safe_name}_training_curves.png"
            fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            png_paths.append(png_path)
            print(f"  Saved: {png_path}")

    # ── Training time overview plot ────────────────────────────────────────────
    if len(runs) > 1 and not args.show:
        fig_t, ax_t = plt.subplots(figsize=(max(6, len(runs) * 2), 4))
        plot_training_time(ax_t, merged, runs)
        fig_t.tight_layout()
        tp = output_dir / "training_time_breakdown.png"
        fig_t.savefig(tp, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig_t)
        png_paths.append(tp)
        print(f"  Saved: {tp}")

    # ── HTML report ────────────────────────────────────────────────────────────
    if not args.no_html and not args.show:
        html_path = build_html_report(merged, runs, png_paths, output_dir)
        print(f"\n✅ HTML report: {html_path}")

    print(f"\n✅ Done. All outputs saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
