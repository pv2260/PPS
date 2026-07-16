"""
behavior_qc.py  -  descriptive behavioral QC for the PPS and Hit/Miss tasks,
styled with ppsanalysis.style.

What changed relative to quick_behavior_plots:
  1. Trial counts are rendered as styled tables (style.table), not bar plots.
     One wide table per task: subjects on rows, speed x category on columns,
     plus a row total column and an "all subjects" summary row.
  2. PPS RT is shown as a single T vs VT contrast figure, split by speed,
     with per-subject median points over group-median bars.
  3. Hit/Miss accuracy is shown across hit / near-hit / near-miss / miss,
     split by speed, with a chance line and per-subject points.

Counts and points are computed before exclusions (no usable == True filter),
so this is a descriptive QC layer.

Use from the notebook after building `t`:

    from ppsanalysis import behavior_qc as qc
    import ppsanalysis.style as style
    style.apply()

    qc.table_pps_trial_counts(t.pps_trials)          # displays a styled table
    qc.table_collision_trial_counts(t.collision_trials)

    qc.plot_pps_rt_contrast(t.pps_trials, save_dir="figures/quick_qc")
    qc.plot_collision_accuracy(t.collision_trials, save_dir="figures/quick_qc")

    # or run everything at once (displays tables, saves figures):
    res = qc.run(t, save_dir="figures/quick_qc")
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# style import that works whether style.py sits inside the package or on the path
try:
    from . import style
except Exception:
    try:
        import ppsanalysis.style as style
    except Exception:
        import style

try:
    from IPython.display import display
except Exception:
    display = print


# preferred display / ordering
PPS_CONDITIONS = ["T", "VT", "V"]
SPEED_ORDER = ["slow", "fast"]
TRIAL_TYPE_PREF = ["hit", "near_hit", "near_miss", "miss"]


# -------------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------------

def _norm(s) -> str:
    """Normalize a label: split camelCase, lowercase, unify separators."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(s).strip())
    return s.lower().replace(" ", "_").replace("-", "_")


def _pretty(key: str) -> str:
    return key.replace("_", " ").title()


def _ensure_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _sorted_nonmissing(series: pd.Series) -> list:
    return sorted(series.dropna().unique().tolist())


def _order_by_pref(present: Sequence, pref: Sequence[str]) -> list:
    pref_norm = [_norm(p) for p in pref]

    def key(v):
        n = _norm(v)
        return (pref_norm.index(n) if n in pref_norm else len(pref_norm), str(v))

    return sorted(present, key=key)


def _savefig(save_dir: str, name: str, show: bool = True) -> None:
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, f"{name}.png")
    pdf_path = os.path.join(save_dir, f"{name}.pdf")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    print(f"saved: {png_path}")
    if show:
        plt.show()
    else:
        plt.close()


# -------------------------------------------------------------------------
# Count tables
# -------------------------------------------------------------------------

def _wide_counts(
    df: pd.DataFrame,
    cat_col: str,
    cat_keys: Sequence[str],
    speeds: Sequence[str],
) -> pd.DataFrame:
    """Subjects on rows, one column per (speed, category), plus row total and
    an all-subjects summary row. Column keys are normalized."""
    d = df.copy()
    d["_cat"] = d[cat_col].map(_norm)
    d["_spd"] = d["speed"].map(_norm)
    d["_sub"] = d["subject"].astype("object")

    subjects = _sorted_nonmissing(d["subject"])
    counts = d.groupby(["_sub", "_spd", "_cat"]).size().reset_index(name="n")

    col_index = pd.MultiIndex.from_tuples(
        [(sp, c) for sp in speeds for c in cat_keys]
    )
    wide = counts.pivot_table(
        index="_sub", columns=["_spd", "_cat"], values="n", fill_value=0
    ).reindex(index=subjects, columns=col_index, fill_value=0)

    wide.columns = [f"{_pretty(sp)}  {_pretty(c).upper() if len(c) <= 2 else _pretty(c)}"
                    for sp, c in wide.columns]
    wide = wide.astype(int)
    wide["total"] = wide.sum(axis=1)

    if len(subjects) > 1:
        wide.loc["all"] = wide.sum(axis=0)

    wide.index.name = "subject"
    return wide.reset_index()


def pps_trial_counts_frame(pps: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_cols(pps, ["subject", "speed", "sensory_condition"])
    speeds = _order_by_pref(_sorted_nonmissing(df["speed"]), SPEED_ORDER)
    cat_keys = [c for c in ["t", "vt", "v"]]  # normalized T, VT, V
    return _wide_counts(df, "sensory_condition", cat_keys, [_norm(s) for s in speeds])


def collision_trial_counts_frame(collision: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_cols(collision, ["subject", "speed", "trial_type"])
    speeds = _order_by_pref(_sorted_nonmissing(df["speed"]), SPEED_ORDER)
    present = [_norm(v) for v in _sorted_nonmissing(df["trial_type"])]
    cat_keys = _order_by_pref(present, TRIAL_TYPE_PREF)
    return _wide_counts(df, "trial_type", cat_keys, [_norm(s) for s in speeds])


def table_pps_trial_counts(pps: pd.DataFrame, show: bool = True):
    """Styled PPS trial-count table (subjects x speed x modality)."""
    frame = pps_trial_counts_frame(pps)
    sty = style.table(
        frame,
        caption="PPS trial counts by modality and speed (before exclusions)",
        bar="total",
        bar_color=style.PATIENT,
        precision=0,
    )
    if show:
        display(sty)
    return sty


def table_collision_trial_counts(collision: pd.DataFrame, show: bool = True):
    """Styled Hit/Miss trial-count table (subjects x speed x trial type)."""
    frame = collision_trial_counts_frame(collision)
    sty = style.table(
        frame,
        caption="Hit/Miss trial counts by trial type and speed (before exclusions)",
        bar="total",
        bar_color=style.PATIENT,
        precision=0,
    )
    if show:
        display(sty)
    return sty


# -------------------------------------------------------------------------
# Grouped bars with per-subject points (shared by both figures)
# -------------------------------------------------------------------------

def _grouped_bars(
    ax,
    df: pd.DataFrame,
    cat_col: str,
    sub_col: str,
    value_col: str,
    cat_keys: Sequence[str],
    cat_labels: Sequence[str],
    sub_keys: Sequence[str],
    sub_labels: Sequence[str],
    colors: dict,
    agg: str = "median",
    scale: float = 1.0,
    annotate_n: bool = True,
    point_size: int = 34,
    jitter_width: float = 0.05,
    bar_alpha: float = 0.9,
    seed: int = 1,
):
    """Grouped bars (one group per cat_key, one bar per sub_key) with jittered
    per-subject points. Matching is done on normalized helper columns so raw
    label spellings do not matter."""
    d = df.copy()
    d["_cat"] = d[cat_col].map(_norm)
    d["_sub"] = d[sub_col].map(_norm)
    d["_val"] = pd.to_numeric(d[value_col], errors="coerce")

    reducer = np.mean if agg == "mean" else np.median
    n_sub = max(len(sub_keys), 1)
    group_w = 0.8
    bar_w = group_w / n_sub
    x = np.arange(len(cat_keys), dtype=float)

    for j, (sk, slabel) in enumerate(zip(sub_keys, sub_labels)):
        offset = (j - (n_sub - 1) / 2) * bar_w
        color = colors.get(sk, style.CONTROL)

        heights, ns = [], []
        for ck in cat_keys:
            vals = d.loc[d["_cat"].eq(ck) & d["_sub"].eq(sk), "_val"].dropna()
            heights.append(reducer(vals) * scale if len(vals) else np.nan)
            ns.append(len(vals))
        heights = np.asarray(heights, dtype=float)

        ax.bar(
            x + offset, np.nan_to_num(heights), width=bar_w * 0.9,
            color=color, alpha=bar_alpha, edgecolor="none", zorder=1,
            label=slabel,
        )

        # per-subject points (one dot per subject per cell)
        for k, ck in enumerate(cat_keys):
            cell = d[d["_cat"].eq(ck) & d["_sub"].eq(sk)]
            if "subject" in cell.columns and cell["subject"].notna().any():
                per = cell.groupby("subject")["_val"].apply(
                    lambda s: reducer(s.dropna()) if s.notna().any() else np.nan
                ).dropna()
                pts = per.to_numpy(dtype=float) * scale
            else:
                pts = cell["_val"].dropna().to_numpy(dtype=float) * scale
            if len(pts):
                jit = style.jitter(len(pts), jitter_width, seed + j * 10 + k)
                ax.scatter(
                    x[k] + offset + jit, pts, s=point_size,
                    color=style.shade(color, -0.22), alpha=0.9,
                    edgecolor="none", zorder=3,
                )

        if annotate_n:
            top = np.nanmax(heights) if np.isfinite(np.nanmax(heights)) else 0.0
            pad = 0.02 * (top if top else 1.0)
            for k, (h, nk) in enumerate(zip(heights, ns)):
                y = (h if np.isfinite(h) else 0.0) + pad
                ax.annotate(
                    f"n={nk}", (x[k] + offset, y),
                    ha="center", va="bottom", fontsize=7, color="#8A8A8A",
                    rotation=0,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    style.clean(ax)
    return ax


# -------------------------------------------------------------------------
# PPS RT contrast: T vs VT
# -------------------------------------------------------------------------

def plot_pps_rt_contrast(
    pps: pd.DataFrame,
    rt_col: str = "rt_ms",
    rt_min: Optional[float] = None,
    rt_max: Optional[float] = None,
    save_dir: str = "figures/quick_qc",
    show: bool = True,
    filename: str = "pps_rt_contrast_T_vs_VT",
) -> pd.DataFrame:
    """PPS RT contrast between T and VT, split by speed.

    Bars are group median RT; dots are per-subject median RT. Optional dashed
    lines mark the RT exclusion window used downstream.
    """
    style.apply()
    df = _ensure_cols(pps, ["subject", "speed", "sensory_condition", rt_col]).copy()
    df[rt_col] = pd.to_numeric(df[rt_col], errors="coerce")
    df = df[df["sensory_condition"].map(_norm).isin(["t", "vt"]) & df[rt_col].notna()]

    speeds = _order_by_pref(_sorted_nonmissing(df["speed"]), SPEED_ORDER)
    sub_keys = [_norm(s) for s in speeds]
    colors = {_norm(k): v for k, v in style.SPEED.items()}

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _grouped_bars(
        ax, df,
        cat_col="sensory_condition", sub_col="speed", value_col=rt_col,
        cat_keys=["t", "vt"], cat_labels=["T", "VT"],
        sub_keys=sub_keys, sub_labels=[_pretty(s) for s in speeds],
        colors=colors, agg="median",
    )

    if rt_min is not None:
        ax.axhline(rt_min, linestyle="--", linewidth=1, color="#B0B0B0")
    if rt_max is not None:
        ax.axhline(rt_max, linestyle="--", linewidth=1, color="#B0B0B0")

    ax.set_ylabel("RT from tactile event (ms)")
    ax.set_title("PPS RT: T vs VT by speed")
    ax.legend(title="Speed")

    _savefig(save_dir, filename, show=show)

    summary = (
        df.assign(_cat=df["sensory_condition"].map(_norm),
                  _spd=df["speed"].map(_norm))
        .groupby(["subject", "_spd", "_cat"])[rt_col]
        .agg(n="size", median_rt="median", mean_rt="mean")
        .reset_index()
        .rename(columns={"_spd": "speed", "_cat": "condition"})
        .sort_values(["subject", "speed", "condition"])
    )
    return summary


# -------------------------------------------------------------------------
# Hit/Miss accuracy by trial type
# -------------------------------------------------------------------------

def plot_collision_accuracy(
    collision: pd.DataFrame,
    acc_col: str = "accuracy",
    save_dir: str = "figures/quick_qc",
    show: bool = True,
    filename: str = "collision_accuracy_by_type",
    trial_type_order: Optional[Sequence[str]] = None,
    chance: Optional[float] = 50.0,
) -> pd.DataFrame:
    """Hit/Miss accuracy across hit / near-hit / near-miss / miss, split by speed.

    Bars are group mean accuracy (%); dots are per-subject mean accuracy (%).
    A dashed chance line is drawn at `chance` (set None to hide).
    """
    style.apply()
    df = _ensure_cols(collision, ["subject", "speed", "trial_type", acc_col]).copy()
    df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")
    df = df[df[acc_col].notna()]

    present = [_norm(v) for v in _sorted_nonmissing(df["trial_type"])]
    if trial_type_order is not None:
        cat_keys = [_norm(v) for v in trial_type_order if _norm(v) in present]
        cat_keys += [c for c in _order_by_pref(present, TRIAL_TYPE_PREF) if c not in cat_keys]
    else:
        cat_keys = _order_by_pref(present, TRIAL_TYPE_PREF)
    cat_labels = [_pretty(c) for c in cat_keys]

    speeds = _order_by_pref(_sorted_nonmissing(df["speed"]), SPEED_ORDER)
    sub_keys = [_norm(s) for s in speeds]
    colors = {_norm(k): v for k, v in style.SPEED.items()}

    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(cat_keys) + 2), 4.2))
    _grouped_bars(
        ax, df,
        cat_col="trial_type", sub_col="speed", value_col=acc_col,
        cat_keys=cat_keys, cat_labels=cat_labels,
        sub_keys=sub_keys, sub_labels=[_pretty(s) for s in speeds],
        colors=colors, agg="mean", scale=100.0,
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108)
    if chance is not None:
        ax.axhline(chance, linestyle="--", linewidth=1, color="#B0B0B0")
        ax.annotate("chance", (ax.get_xlim()[1], chance), ha="right", va="bottom",
                    fontsize=8, color="#8A8A8A")
    ax.set_title("Hit/Miss accuracy by trial type and speed")
    ax.legend(title="Speed")

    _savefig(save_dir, filename, show=show)

    summary = (
        df.assign(_type=df["trial_type"].map(_norm),
                  _spd=df["speed"].map(_norm))
        .groupby(["subject", "_spd", "_type"])[acc_col]
        .agg(n="size", accuracy="mean")
        .reset_index()
        .assign(accuracy_percent=lambda x: 100 * x["accuracy"])
        .rename(columns={"_spd": "speed", "_type": "trial_type"})
        .sort_values(["subject", "speed", "trial_type"])
    )
    return summary


# -------------------------------------------------------------------------
# Master call
# -------------------------------------------------------------------------

def run(
    t,
    rt_min: Optional[float] = None,
    rt_max: Optional[float] = None,
    save_dir: str = "figures/quick_qc",
    show: bool = True,
) -> dict:
    """Display the two count tables and save the two contrast figures.

    Returns a dict with the styled tables, the underlying count frames, and the
    RT / accuracy per-subject summaries.
    """
    style.apply()
    pps = t.pps_trials.copy()
    collision = t.collision_trials.copy()
    results = {}

    print("=== PPS trial counts ===")
    results["pps_counts_frame"] = pps_trial_counts_frame(pps)
    results["pps_counts_table"] = table_pps_trial_counts(pps, show=show)

    print("\n=== Hit/Miss trial counts ===")
    results["collision_counts_frame"] = collision_trial_counts_frame(collision)
    results["collision_counts_table"] = table_collision_trial_counts(collision, show=show)

    print("\n=== PPS RT contrast (T vs VT) ===")
    results["pps_rt_summary"] = plot_pps_rt_contrast(
        pps, rt_min=rt_min, rt_max=rt_max, save_dir=save_dir, show=show
    )

    print("\n=== Hit/Miss accuracy by trial type ===")
    results["collision_accuracy_summary"] = plot_collision_accuracy(
        collision, save_dir=save_dir, show=show
    )

    return results