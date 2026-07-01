#!/usr/bin/env python3
"""
Compare Time-in-Band (TiB) between two series (two CSV files), each containing
goal (setpoint) and value (measured).

Examples
--------
Two 2-column CSVs without headers (goal,value):
  python tib_compare.py --csv-a a.csv --csv-b b.csv --goal-col 1 --value-col 2 \
    --bands "in:0.05,near:0.10,out:inf" --mode percent --out tib_compare.pdf

With headers:
  python tib_compare.py --csv-a a.csv --csv-b b.csv --goal-name goal --value-name value \
    --bands "in:0.05,near:0.10,out:inf" --mode percent --out tib_compare.pdf

Semicolon delimiter:
  python tib_compare.py --csv-a a.csv --csv-b b.csv --delimiter ";" \
    --goal-col 1 --value-col 2 --bands "in:0.05,near:0.10,out:inf" --out tib.pdf

Optional time-weighting (each file must have time column):
  python tib_compare.py --csv-a a.csv --csv-b b.csv --t-col 1 --goal-col 2 --value-col 3 \
    --bands "in:0.05,near:0.10,out:inf" --mode percent --out tib_compare.pdf
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Band:
    name: str
    threshold: float  # max allowed metric for this band; inf allowed


def parse_bands(spec: str) -> List[Band]:
    """
    Parse "name:thr,name:thr,..." where thr is float or 'inf'.
    Thresholds must be non-decreasing; last should typically be inf.
    """
    bands: List[Band] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid band part '{part}'. Expected name:threshold.")
        name, thr_s = part.split(":", 1)
        name = name.strip()
        thr_s = thr_s.strip().lower()
        if thr_s in ("inf", "+inf", "infinity"):
            thr = float("inf")
        else:
            thr = float(thr_s)
            if thr < 0:
                raise ValueError(f"Threshold must be >= 0, got {thr} for band '{name}'.")
        bands.append(Band(name=name, threshold=thr))

    if not bands:
        raise ValueError("No bands parsed. Provide --bands like 'in:0.05,near:0.10,out:inf'.")

    for i in range(1, len(bands)):
        if bands[i].threshold < bands[i - 1].threshold:
            raise ValueError("Band thresholds must be non-decreasing (increasing tolerance).")

    if not math.isinf(bands[-1].threshold):
        raise ValueError("Last band should usually be inf (e.g., out:inf) to catch all samples.")

    return bands


def one_based_to_idx(col: int) -> int:
    if col <= 0:
        raise ValueError("Column indices are 1-based and must be >= 1.")
    return col - 1


def load_columns(
        csv_path: Path,
        delimiter: str,
        goal_col: Optional[int],
        value_col: Optional[int],
        goal_name: Optional[str],
        value_name: Optional[str],
        t_col: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    df = pd.read_csv(csv_path, sep=delimiter)

    # goal
    if goal_name is not None:
        if goal_name not in df.columns:
            raise ValueError(f"[{csv_path.name}] Column '{goal_name}' not found in headers: {list(df.columns)}")
        goal = pd.to_numeric(df[goal_name], errors="coerce").to_numpy()
    else:
        if goal_col is None:
            raise ValueError(f"[{csv_path.name}] Provide either --goal-name or --goal-col.")
        g_idx = one_based_to_idx(goal_col)
        if g_idx >= df.shape[1]:
            raise ValueError(f"[{csv_path.name}] --goal-col {goal_col} out of range (has {df.shape[1]} columns).")
        goal = pd.to_numeric(df.iloc[:, g_idx], errors="coerce").to_numpy()

    # value
    if value_name is not None:
        if value_name not in df.columns:
            raise ValueError(f"[{csv_path.name}] Column '{value_name}' not found in headers: {list(df.columns)}")
        value = pd.to_numeric(df[value_name], errors="coerce").to_numpy()
    else:
        if value_col is None:
            raise ValueError(f"[{csv_path.name}] Provide either --value-name or --value-col.")
        v_idx = one_based_to_idx(value_col)
        if v_idx >= df.shape[1]:
            raise ValueError(f"[{csv_path.name}] --value-col {value_col} out of range (has {df.shape[1]} columns).")
        value = pd.to_numeric(df.iloc[:, v_idx], errors="coerce").to_numpy()

    if np.isnan(goal).any() or np.isnan(value).any():
        raise ValueError(f"[{csv_path.name}] Goal or value contains NaNs / non-numeric entries after parsing.")

    # optional time
    t = None
    if t_col is not None:
        t_idx = one_based_to_idx(t_col)
        if t_idx >= df.shape[1]:
            raise ValueError(f"[{csv_path.name}] --t-col {t_col} out of range (has {df.shape[1]} columns).")
        t = pd.to_numeric(df.iloc[:, t_idx], errors="coerce").to_numpy()
        if np.isnan(t).any():
            raise ValueError(f"[{csv_path.name}] Time column contains NaNs / non-numeric entries after parsing.")
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"[{csv_path.name}] Time column must be strictly increasing for time-weighting.")

    if len(goal) != len(value):
        raise ValueError(f"[{csv_path.name}] Goal and value must have same length.")

    return goal.astype(float), value.astype(float), (t.astype(float) if t is not None else None)


def compute_weights(t: Optional[np.ndarray], n: int) -> np.ndarray:
    """
    If time is provided, compute sample weights via dt; else weight=1 per sample.
    """
    if t is None or n <= 1:
        return np.ones(n, dtype=float)
    dt = np.diff(t)
    tail = float(np.median(dt)) if dt.size > 0 else 1.0
    return np.concatenate([dt, np.array([tail], dtype=float)])


def time_in_band(
        goal: np.ndarray,
        value: np.ndarray,
        bands: List[Band],
        mode: str,
        weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (totals, perc) for each band.
    """
    err = np.abs(value - goal)
    if mode == "absolute":
        metric = err
    elif mode == "percent":
        denom = np.maximum(np.abs(goal), 1e-12)
        metric = err / denom
    else:
        raise ValueError("mode must be 'percent' or 'absolute'.")

    totals = np.zeros(len(bands), dtype=float)
    for i, m in enumerate(metric):
        for j, b in enumerate(bands):
            if m <= b.threshold:
                totals[j] += weights[i]
                break

    total = float(np.sum(weights))
    perc = (totals / total) * 100.0 if total > 0 else np.zeros_like(totals)
    return totals, perc


def plot_comparison(
        band_names: List[str],
        perc_a: np.ndarray,
        perc_b: np.ndarray,
        label_a: str,
        label_b: str,
        title: str,
        out_pdf: Path,
        out_png: Optional[Path],
        show_values: bool,
):
    x = np.arange(len(band_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    bars_a = ax.bar(x - width / 2, perc_a, width=width, label=label_a)
    bars_b = ax.bar(x + width / 2, perc_b, width=width, label=label_b)

    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.set_ylabel("Time (%)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.legend()

    ymax = max(float(np.max(perc_a)) if perc_a.size else 0.0, float(np.max(perc_b)) if perc_b.size else 0.0)
    ax.set_ylim(0, max(100.0, ymax * 1.15))

    if show_values:
        def annotate(bars):
            for r in bars:
                h = r.get_height()
                ax.text(
                    r.get_x() + r.get_width() / 2,
                    h,
                    f"{h:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    )
        annotate(bars_a)
        annotate(bars_b)

    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf")
    if out_png is not None:
        fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Compare Time-in-Band between two CSV series (goal,value).")
    ap.add_argument("--csv-a", required=True, help="CSV path for series A.")
    ap.add_argument("--csv-b", required=True, help="CSV path for series B.")
    ap.add_argument("--delimiter", default=",", help="CSV delimiter (default ',').")

    # Column selection (applies to BOTH files)
    ap.add_argument("--goal-col", type=int, default=None, help="1-based index of goal column.")
    ap.add_argument("--value-col", type=int, default=None, help="1-based index of value column.")
    ap.add_argument("--goal-name", default=None, help="Header name of goal column.")
    ap.add_argument("--value-name", default=None, help="Header name of value column.")
    ap.add_argument("--t-col", type=int, default=None, help="1-based index of time column (optional, both files).")

    ap.add_argument("--mode", choices=["percent", "absolute"], default="percent",
                    help="Threshold interpretation: percent (fraction of goal) or absolute units.")
    ap.add_argument("--bands", default="in:0.20,near:0.25,out:inf",
                    help="Band spec increasing tolerance: name:thr,... last usually inf.")
    ap.add_argument("--label-a", default="Series A", help="Legend label for series A.")
    ap.add_argument("--label-b", default="Series B", help="Legend label for series B.")
    ap.add_argument("--title", default="Time-in-Band Comparison", help="Figure title.")
    ap.add_argument("--out", required=True, help="Output PDF filename (e.g., tib_compare.pdf).")
    ap.add_argument("--png", default=None, help="Optional PNG filename.")
    ap.add_argument("--no-values", action="store_true", help="Disable value labels on bars.")

    args = ap.parse_args()

    csv_a = Path(args.csv_a)
    csv_b = Path(args.csv_b)
    if not csv_a.exists():
        raise FileNotFoundError(f"CSV A not found: {csv_a}")
    if not csv_b.exists():
        raise FileNotFoundError(f"CSV B not found: {csv_b}")

    bands = parse_bands(args.bands)
    band_names = [b.name for b in bands]

    goal_a, value_a, t_a = load_columns(csv_a, args.delimiter, args.goal_col, args.value_col,
                                        args.goal_name, args.value_name, args.t_col)
    goal_b, value_b, t_b = load_columns(csv_b, args.delimiter, args.goal_col, args.value_col,
                                        args.goal_name, args.value_name, args.t_col)

    w_a = compute_weights(t_a, len(goal_a))
    w_b = compute_weights(t_b, len(goal_b))

    _, perc_a = time_in_band(goal_a, value_a, bands, args.mode, w_a)
    _, perc_b = time_in_band(goal_b, value_b, bands, args.mode, w_b)

    out_pdf = Path(args.out)
    out_png = Path(args.png) if args.png else None

    plot_comparison(
        band_names=band_names,
        perc_a=perc_a,
        perc_b=perc_b,
        label_a=args.label_a,
        label_b=args.label_b,
        title=args.title,
        out_pdf=out_pdf,
        out_png=out_png,
        show_values=not args.no_values,
    )

    print(f"Saved: {out_pdf}")
    if out_png:
        print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
