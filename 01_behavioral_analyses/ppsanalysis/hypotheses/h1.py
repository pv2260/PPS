"""H1: facilitation should increase as the stimulus approaches the body.

This module is designed to be called one step at a time from a notebook.
It also owns the shared building blocks that H1a reuses:

    DISTANCE_M_BY_RANK / rank_to_metres   distance mapping
    subject_profile / group_summary       collapsing to subject and group level
    profile_grid                          estimator x correction sweep
    _format_distance_axis                 distance axis in ranks or metres
    correction_grid                       shared 2 x 3 panel layout

WHICH TRIALS ENTER
------------------
``get_trials`` is the only door into this module, and it applies two rules:

    usable only   Trials excluded by QC never reach a cell estimate. If the
                  table carries a ``usable`` column it is honoured. If not,
                  the absolute RT window from config is applied and a warning
                  is printed, because a table without that column has not been
                  through QC and the window alone cannot reconstruct the
                  no-response or interruption rules.

No cell is ever excluded for being small. A median of two observations is
their mean, which is a usable estimate and simply a less precise one;
discarding real trials to avoid reporting that imprecision is the worse
trade. Thin cells are reported instead, and ``rt_cells`` carries ``n_T`` and
``n_VT`` so a facilitation value can always be traced back to the trial
counts behind it.

THE ESTIMATOR IN THIN CELLS
---------------------------
At one or two observations the median is not just imprecise, it is a poor
use of the data: with two values it is their mean anyway, and with one it is
that value. ``thin_cell_n`` therefore switches the cell estimator to the mean
at or below that count, and leaves the preregistered median in place
everywhere else. The switch changes nothing where it does not apply, is
reported by ``rt_cells`` in ``estimator_T`` and ``estimator_VT``, and never
removes a trial.

Typical use
-----------
from ppsanalysis.analyses import h1_facilitation as h1

trials = h1.get_trials(t)

h1.rt_distribution_table(trials)
h1.plot_rt_distributions(trials)

h1.compare_estimators(trials)
h1.plot_mean_vs_median_facilitation(trials)

h1.baseline_contribution(trials)

profile = h1.facilitation_profile(
    trials,
    estimator="median",
    correction="delay_matched",
)
h1.plot_rt_profiles(profile)
h1.plot_facilitation_profiles(profile)
h1.plot_group_facilitation(profile)

nearfar = h1.nearfar_table(profile)
h1.plot_nearfar_estimation(nearfar)
h1.plot_nearfar_pairs(nearfar)
h1.group_test(nearfar)

halfmax = h1.half_max_boundary(profile)
h1.plot_halfmax_boundaries(profile, halfmax)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .. import figures

import style
from .. import config
from ..stats_utils import bootstrap_mean_ci, wilcoxon_one_sided

style.apply()


NEAR = [1, 2]
FAR = [6, 7]
SPEEDS = ["slow", "fast"]
CONDITIONS = ["T", "VT"]
CORRECTIONS = [
    "delay_matched",
    "expectancy_model",
    "overall_tactile",
    "fastest_tactile",
    "noel_endpoints",
    "fastest_n_tactile",
]

# PPS condition colors
VT = "#19d3c5"
T = "#838689"
V = "#e5e5e5"

# Trials pooled for the fastest-n baseline. Taking the N fastest tactile
# trials of roughly 28 selects a low quantile, so this baseline sits below
# the others and facilitation shrinks accordingly.
FASTEST_N = 4

# Cells at or below this many trials are summarised by the mean rather than
# the median. Not an exclusion: nothing is ever removed for being small.
THIN_CELL_N = 2

# Threshold used only for reporting which cells are thin.
MIN_CELL_N = 3

CELL_KEYS = [
    "subject",
    "session",
    "speed",
    "sensory_condition",
    "position_rank",
]

CORRECTION_LABELS = {
    "delay_matched": "Delay-matched",
    "expectancy_model": "Modelled expectancy",
    "overall_tactile": "Overall tactile",
    "fastest_tactile": "Fastest tactile",
    "noel_endpoints": "Fastest of D1 and D7 (Noel et al.)",
    "fastest_n_tactile": f"Fastest {FASTEST_N} tactile trials",
}

CURVE_LABELS = {
    "T": "Tactile-only RT",
    "VT": "Visuo-tactile RT",
    "tactile_baseline_ms": "Tactile baseline used",
    "facilitation_ms": "Facilitation (baseline - VT)",
}

# Physical distance corresponding to each position rank.
# Rank 1 is closest to the body.
DISTANCE_M_BY_RANK = {
    1: 0.250,
    2: 0.583,
    3: 0.917,
    4: 1.250,
    5: 1.583,
    6: 1.917,
    7: 2.250,
}

# style exports the muted grey privately; alias it so a typo cannot take a
# whole figure down at draw time.
MUTED = getattr(style, "_MUTED", "gray")


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def _cell_keys(trials):
    return [column for column in CELL_KEYS if column in trials.columns]


def get_trials(t, min_cell_n=None, verbose=True):
    """Return usable young-control T and VT trials.

    Parameters
    ----------
    t
        Analysis object carrying ``t.pps_young``. Pass the QC-filtered copy.
    min_cell_n : int or None, default=None
        Off by default: no cell is excluded for being small. Setting it
        removes cells below that many trials, which is a real exclusion of
        real data and belongs in a sensitivity analysis. Thin cells are
        reported either way.
    verbose : bool, default=True
        Print what was excluded and which cells are thin.
    """
    trials = t.pps_young.copy()
    trials["trial_index"] = trials.index
    n_in = len(trials)

    if "usable" in trials.columns:
        trials = trials[trials["usable"].fillna(False).astype(bool)]
    else:
        rt_min = getattr(config, "RT_MIN_MS", 100)
        rt_max = getattr(config, "RT_MAX_MS", 900)
        before = len(trials)
        trials = trials[
            trials["rt_ms"].between(rt_min, rt_max, inclusive="neither")
        ]

        if verbose:
            print(
                "WARNING: the trial table has no 'usable' column, so h1 "
                f"applied only the absolute RT window ({rt_min:g} to "
                f"{rt_max:g} ms), removing {before - len(trials)} trial(s). "
                "This cannot reconstruct the no-response or interruption "
                "rules. Pass the QC-filtered tables."
            )

    trials = trials[
        trials["sensory_condition"].isin(CONDITIONS)
        & trials["rt_ms"].notna()
        & trials["position_rank"].notna()
    ].copy()

    n_after_exclusions = len(trials)

    keys = _cell_keys(trials)
    sizes = trials.groupby(keys, dropna=False)["rt_ms"].transform("size")

    # Reported, not removed.
    thin = sizes <= THIN_CELL_N

    if verbose and thin.any():
        listing = (
            trials.loc[thin]
            .groupby(keys, dropna=False, as_index=False)["rt_ms"]
            .size()
            .rename(columns={"size": "n_trials"})
            .sort_values(keys)
        )
        print(
            f"{len(listing)} cell(s) hold {THIN_CELL_N} trials or fewer and "
            f"will be summarised by the mean rather than the median. "
            f"No trial is removed:"
        )
        print("  " + listing.to_string(index=False).replace("\n", "\n  "))

    if min_cell_n:
        removed = sizes < min_cell_n

        if verbose and removed.any():
            print(
                f"SENSITIVITY: excluding {int(removed.sum())} trial(s) in "
                f"cells below n = {min_cell_n}. This is not the "
                f"preregistered analysis."
            )

        trials = trials.loc[~removed]

    if verbose:
        print(
            f"Trials entering H1: {len(trials)} of {n_in} "
            f"({n_in - n_after_exclusions} excluded by QC)"
        )

    return trials.reset_index(drop=True)

def facilitation_components(
    trials,
    estimator="median",
    correction="delay_matched",
    thin_cell_n=THIN_CELL_N,
    expectancy_degree=1,
):
    """
    Show exactly which trials contribute to every facilitation value.

    Returns one row per subject x speed x distance containing:
        - T trial indices and RTs
        - VT trial indices and RTs
        - T estimate
        - VT estimate
        - tactile baseline
        - facilitation
        - trial counts
    """

    if "trial_index" not in trials.columns:
        raise ValueError(
            "trials has no 'trial_index' column. "
            "Add trials['trial_index'] = trials.index "
            "inside get_trials() before reset_index()."
        )

    # ---------------------------------------------------------
    # Calculate the values exactly as H1 does
    # ---------------------------------------------------------

    cells = rt_cells(
        trials,
        estimator=estimator,
        correction=correction,
        thin_cell_n=thin_cell_n,
        expectancy_degree=expectancy_degree,
    )

    # ---------------------------------------------------------
    # Collect raw T and VT trials for each cell
    # ---------------------------------------------------------

    keys = [
        "subject",
        "session",
        "speed",
        "position_rank",
    ]

    raw_rows = []

    for key, one in trials.groupby(keys, dropna=False):

        record = dict(zip(keys, key))

        for condition in ["T", "VT"]:

            c = (
                one[
                    one["sensory_condition"].eq(condition)
                ]
                .sort_values("trial_index")
            )

            record[f"{condition}_indices"] = (
                c["trial_index"].astype(int).tolist()
            )

            record[f"{condition}_RTs"] = (
                c["rt_ms"]
                .round(1)
                .tolist()
            )

        raw_rows.append(record)

    raw = pd.DataFrame(raw_rows)

    # ---------------------------------------------------------
    # Merge raw trials with calculated facilitation values
    # ---------------------------------------------------------

    out = cells.merge(
        raw,
        on=keys,
        how="left",
    )

    columns = [
        "subject",
        "session",
        "speed",
        "position_rank",

        "T_indices",
        "T_RTs",
        "n_T",
        "T",

        "VT_indices",
        "VT_RTs",
        "n_VT",
        "VT",

        "tactile_baseline_ms",
        "facilitation_ms",
        "correction",
    ]

    # Include estimator columns if available
    for col in ["estimator_T", "estimator_VT"]:
        if col in out.columns:
            columns.insert(
                columns.index("tactile_baseline_ms"),
                col,
            )

    out = out[
        [c for c in columns if c in out.columns]
    ].copy()

    for col in [
        "T",
        "VT",
        "tactile_baseline_ms",
        "facilitation_ms",
    ]:
        if col in out.columns:
            out[col] = out[col].round(1)

    out["position_rank"] = (
        out["position_rank"].astype(int)
    )

    return (
        out
        .sort_values(
            [
                "subject",
                "speed",
                "position_rank",
            ]
        )
        .reset_index(drop=True)
    )


def rank_to_metres(rank):
    """Convert a continuous position rank to physical distance in metres."""
    ranks = np.asarray(sorted(DISTANCE_M_BY_RANK), dtype=float)
    distances = np.asarray(
        [DISTANCE_M_BY_RANK[int(r)] for r in ranks],
        dtype=float,
    )

    values = np.asarray(rank, dtype=float)
    converted = np.interp(values, ranks, distances)

    if np.ndim(rank) == 0:
        return float(converted)

    return converted


def _save(fig, save_path):
    if save_path is None:
        return

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")


def _as_key_list(by):
    """Normalise a grouping argument to a list of column names."""
    if by is None:
        return []

    if isinstance(by, str):
        return [by]

    return list(by)


# ---------------------------------------------------------------------
# Shared axis and layout helpers
# ---------------------------------------------------------------------

def _format_distance_axis(ax, unit="rank", ranks=None):
    """
    Format the distance axis.

    unit="rank"
        Ticks at every rank, labelled D1 to D7.

    unit="m"
        Ticks at every rank, labelled with the corresponding distance in
        metres. ``ranks`` restricts the ticks to the ranks actually tested.
    """
    if ranks is None:
        tick_ranks = sorted(DISTANCE_M_BY_RANK)
    else:
        tick_ranks = [int(r) for r in sorted(ranks)]

    ax.set_xticks(tick_ranks)

    if unit == "m":
        ax.set_xticklabels(
            [f"{DISTANCE_M_BY_RANK[r]:.3f}" for r in tick_ranks]
        )
        ax.set_xlabel("Distance from body (m)")
    else:
        ax.set_xticklabels([f"D{r}" for r in tick_ranks])
        ax.set_xlabel("Distance (D1 = closest)")

    style.clean(ax)


def correction_grid(
    estimators=("mean", "median"),
    corrections=CORRECTIONS,
    figsize=(15.5, 7.0),
    ylabel="{estimator}\nGroup-average facilitation (ms)",
    sharey=True,
):
    """
    Create the shared estimator x correction panel grid.

    Rows are estimators, columns are tactile corrections. Titles, row labels,
    and the zero reference line are applied here; the caller draws the data.

    Returns
    -------
    fig, axes : matplotlib Figure and 2D array of Axes
    """
    fig, axes = plt.subplots(
        len(estimators),
        len(corrections),
        figsize=figsize,
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )

    for row, estimator in enumerate(estimators):
        for col, correction in enumerate(corrections):
            ax = axes[row, col]

            ax.axhline(
                0,
                linestyle="--",
                linewidth=0.8,
                color="gray",
                zorder=0,
            )

            if row == 0:
                ax.set_title(CORRECTION_LABELS[correction])

            if col == 0:
                ax.set_ylabel(
                    ylabel.format(estimator=estimator.capitalize())
                )

    return fig, axes


# ---------------------------------------------------------------------
# 1. Inspect reaction-time distributions
# ---------------------------------------------------------------------

def rt_distribution_table(trials):
    """Summarise RT distributions before choosing mean or median."""
    table = (
        trials
        .groupby(["subject", "speed", "sensory_condition"])["rt_ms"]
        .agg(
            n="size",
            mean="mean",
            median="median",
            sd="std",
            minimum="min",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            maximum="max",
            skew="skew",
        )
        .reset_index()
    )

    table["mean_minus_median"] = table["mean"] - table["median"]
    table["iqr"] = table["q3"] - table["q1"]

    return table.round(1)


def cell_counts(trials):
    """Usable trials in every cell, so a thin cell is never a surprise."""
    keys = _cell_keys(trials)

    return (
        trials
        .groupby(keys, dropna=False, as_index=False)["rt_ms"]
        .size()
        .rename(columns={"size": "n_trials"})
        .sort_values(keys)
        .reset_index(drop=True)
    )


def plot_rt_distributions(trials, save_path=None):
    """
    Plot every usable RT.

    Coloured horizontal line = mean.
    Black dashed horizontal line = median.
    """
    subjects = sorted(trials["subject"].dropna().unique())

    fig, axes = plt.subplots(
        len(subjects),
        2,
        figsize=(11, 3.8 * len(subjects)),
        squeeze=False,
        sharey=True,
    )

    categories = [
        ("slow", "T"),
        ("slow", "VT"),
        ("fast", "T"),
        ("fast", "VT"),
    ]

    for row, subject in enumerate(subjects):
        subject_trials = trials[trials["subject"] == subject]

        for col, (label, positions) in enumerate([
            ("Far: D6-D7", FAR),
            ("Near: D1-D2", NEAR),
        ]):
            ax = axes[row, col]
            selected = subject_trials[
                subject_trials["position_rank"].isin(positions)
            ]

            for x, (speed, condition) in enumerate(categories, start=1):
                values = selected.loc[
                    (selected["speed"] == speed)
                    & (selected["sensory_condition"] == condition),
                    "rt_ms",
                ].to_numpy()

                if len(values) == 0:
                    continue

                jitter = np.random.default_rng(x).uniform(
                    -0.08, 0.08, len(values)
                )

                ax.scatter(
                    x + jitter,
                    values,
                    s=18,
                    alpha=0.35,
                    color=style.SPEED[speed],
                    marker="o" if condition == "T" else "x",
                )

                ax.plot(
                    [x - 0.20, x + 0.20],
                    [np.mean(values), np.mean(values)],
                    color=style.SPEED[speed],
                    linewidth=2,
                )

                ax.plot(
                    [x - 0.20, x + 0.20],
                    [np.median(values), np.median(values)],
                    color=style.INK,
                    linewidth=2,
                    linestyle="--",
                )

            ax.set_xticks(range(1, 5))
            ax.set_xticklabels(
                ["slow T", "slow VT", "fast T", "fast VT"],
                rotation=25,
                ha="right",
            )
            ax.set_title(f"{subject}: {label}")
            ax.set_ylabel("RT (ms)")
            style.clean(ax)

    fig.suptitle(
        "RT distributions\n"
        "colour = mean; black dashed = median",
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, save_path)

    return fig


# ---------------------------------------------------------------------
# 2. Calculate T and VT RTs and facilitation
# ---------------------------------------------------------------------

def expectancy_baseline(trials, degree=1, interaction=True):
    """Tactile-only RT as a function of onset delay, fitted per participant.

    The delay-matched correction rests on a claim about temporal expectancy:
    that tactile RT varies with the delay between trial onset and the tactile
    event. That is a claim about a *function of delay*, and a function of
    delay does not need one free parameter per distance. Estimating it as
    fourteen unrelated cell means, each from four trials, spends far more
    parameters than the claim requires and carries the sampling noise of
    every one of them into the corrected score.

    This estimates the same function directly. For each participant,

        RT_T = b0 + b1 * delay + b2 * speed + b3 * (delay x speed)

    is fitted by least squares on all of that participant's tactile-only
    trials at once, and the fitted value read off at each distance and speed
    to give the matched baseline.

    The expectancy-control principle is unchanged: the baseline still varies
    with delay, the two speed contexts are still kept apart, and the
    correction is still applied at every distance. Only the estimator
    changes, from fourteen medians of four observations to four coefficients
    from roughly fifty.

    Parameters
    ----------
    degree : int, default=1
        Polynomial order in delay. Reports disagree about the form of this
        function, describing it as monotonic (Kandula et al., 2017),
        logarithmic (Hobeika et al., 2020) and bell-shaped (Bertoni et al.,
        2022), so ``degree=2`` is the check for curvature rather than an
        assumption of it.
    interaction : bool, default=True
        Include the delay-by-speed term, so the expectancy function may
        differ between speed contexts rather than being assumed to.

    Returns
    -------
    fitted : pandas.DataFrame
        Predicted baseline per subject, session, speed and position, with
        the standard error of the fitted value.
    coefficients : pandas.DataFrame
        One row per participant: the fitted terms with their standard
        errors, the residual SD and R-squared. ``delay`` is the expectancy
        ramp in milliseconds per distance step at the slow speed;
        ``delay_x_fast`` is how much steeper it is at the fast speed.
    """
    tactile = trials[trials["sensory_condition"] == "T"]

    if tactile.empty:
        return pd.DataFrame(), pd.DataFrame()

    keys = [c for c in ["subject", "session"] if c in tactile.columns]

    fitted_rows, coefficient_rows = [], []

    for key, one in tactile.groupby(keys, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)

        x = pd.to_numeric(one["position_rank"], errors="coerce").to_numpy(float)
        y = one["rt_ms"].to_numpy(dtype=float)
        fast = (one["speed"] == "fast").to_numpy(dtype=float)

        ok = np.isfinite(x) & np.isfinite(y)
        x, y, fast = x[ok], y[ok], fast[ok]

        if len(x) < 6 or len(np.unique(x)) < 3:
            continue

        # Delay is centred so the intercept reads as RT at a delay that was
        # actually sampled rather than extrapolated to rank zero, and so the
        # speed term is the difference between contexts at that delay.
        centre = float(np.mean(x))
        x_c = x - centre

        columns, names = [np.ones_like(x_c)], ["intercept"]

        for power in range(1, int(degree) + 1):
            columns.append(x_c ** power)
            names.append("delay" if power == 1 else f"delay_{power}")

        if len(np.unique(fast)) > 1:
            columns.append(fast)
            names.append("speed_fast")

            if interaction:
                columns.append(x_c * fast)
                names.append("delay_x_fast")

        design = np.column_stack(columns)

        try:
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        residual = y - design @ beta
        residual_df = len(y) - design.shape[1]

        if residual_df <= 0:
            continue

        residual_sd = float(np.sqrt(np.sum(residual ** 2) / residual_df))

        try:
            covariance = np.linalg.inv(design.T @ design) * residual_sd ** 2
            errors = np.sqrt(np.diag(covariance))
        except np.linalg.LinAlgError:
            covariance, errors = None, np.full(len(beta), np.nan)

        total = float(np.sum((y - y.mean()) ** 2))

        record = dict(zip(keys, key_values))
        record["n_trials"] = int(len(y))

        for name, value, error in zip(names, beta, errors):
            record[name] = float(value)
            record[f"se_{name}"] = float(error)

        record["residual_sd_ms"] = residual_sd
        record["r2"] = (
            1.0 - float(np.sum(residual ** 2)) / total if total > 0 else np.nan
        )
        coefficient_rows.append(record)

        for speed in one["speed"].dropna().unique():
            is_fast = 1.0 if speed == "fast" else 0.0

            for rank in sorted(one["position_rank"].dropna().unique()):
                point = [1.0]

                for power in range(1, int(degree) + 1):
                    point.append((rank - centre) ** power)

                if "speed_fast" in names:
                    point.append(is_fast)

                    if interaction:
                        point.append((rank - centre) * is_fast)

                point = np.asarray(point, dtype=float)

                fitted_rows.append({
                    **dict(zip(keys, key_values)),
                    "speed": speed,
                    "position_rank": rank,
                    "tactile_baseline_ms": float(point @ beta),
                    "baseline_se_ms": (
                        float(np.sqrt(point @ covariance @ point))
                        if covariance is not None else np.nan
                    ),
                    "n_trials_fit": int(len(y)),
                })

    return pd.DataFrame(fitted_rows), pd.DataFrame(coefficient_rows)


def rt_cells(
    trials,
    estimator="median",
    correction="delay_matched",
    thin_cell_n=THIN_CELL_N,
    expectancy_degree=1,
):
    """
    Return T, VT, and corrected facilitation for every experimental cell.

    ``n_T`` and ``n_VT`` are carried alongside, so a facilitation value can
    always be traced back to how many trials produced it. ``estimator_T`` and
    ``estimator_VT`` record which statistic each cell actually used.

    ``thin_cell_n`` switches the cell estimator to the mean at or below that
    many trials. Nothing is excluded; the switch exists because a median of
    one or two observations wastes them. Set it to None to use ``estimator``
    everywhere.

    Parameters
    ----------
    trials : pandas.DataFrame
        Usable tactile-only (T) and visuo-tactile (VT) trials, from
        ``get_trials``.
    estimator : {"mean", "median"}
        Statistic used to summarise RTs within each experimental condition.
    correction : {
        "delay_matched",
        "expectancy_model",
        "overall_tactile",
        "fastest_tactile",
        "fastest_n_tactile",
    }
        Tactile baseline used to correct each VT condition:

        ``delay_matched``
            Compare each VT condition with tactile-only trials delivered at
            the same position and therefore the same onset delay:
            T(subject, session, speed, distance) - VT(...). Fourteen
            independent cell estimates per participant.

        ``expectancy_model``
            The same delay-dependence, estimated as a function of delay
            rather than as fourteen independent cell means. Tactile RT is
            regressed on delay, speed and their interaction using all of a
            participant's tactile-only trials, and the fitted value is read
            off at each distance. Keeps the expectancy control; changes only
            the estimator.

        ``overall_tactile``
            Compare every VT condition with the participant's tactile-only
            RT averaged across all tactile onset positions, separately for
            session and speed.

        ``fastest_tactile``
            Compare every VT condition with the fastest tactile-only
            position, separately for participant, session, and speed. This
            is the most conservative correction because VT must outperform
            the participant's best unimodal tactile condition.

        ``fastest_n_tactile``
            The median of the ``FASTEST_N`` fastest tactile-only trials,
            pooled across distance, per participant, session and speed.
            The same conservative floor as ``fastest_tactile``, resting on
            several trials rather than on one cell.

    Notes
    -----
    Corrected scores are always expressed as tactile baseline minus VT RT.
    Positive values therefore indicate multisensory facilitation.
    """
    if estimator not in {"mean", "median"}:
        raise ValueError("estimator must be 'mean' or 'median'")

    if correction not in CORRECTIONS:
        raise ValueError(
            "correction must be one of: "
            + ", ".join(CORRECTIONS)
        )

    group_keys = [
        "subject",
        "session",
        "speed",
        "position_rank",
        "sensory_condition",
    ]

    # The preregistered estimator everywhere, except in cells at or below
    # thin_cell_n, where the median is a poor use of one or two observations:
    # with two it is their mean anyway, and with one it is that value. The
    # switch never removes a trial and never applies to a full cell.
    def _estimate(values):
        values = values.dropna()

        if len(values) == 0:
            return np.nan

        if thin_cell_n and len(values) <= thin_cell_n:
            return float(values.mean())

        return float(values.mean() if estimator == "mean" else values.median())

    cells = (
        trials
        .groupby(group_keys)["rt_ms"]
        .agg(_estimate)
        .unstack("sensory_condition")
        .reset_index()
    )

    counts = (
        trials
        .groupby(group_keys)["rt_ms"]
        .size()
        .unstack("sensory_condition")
        .rename(columns={"T": "n_T", "VT": "n_VT"})
        .reset_index()
    )

    for condition in CONDITIONS:
        if condition not in cells.columns:
            cells[condition] = np.nan

    for column in ("n_T", "n_VT"):
        if column not in counts.columns:
            counts[column] = 0

    cells = cells.merge(
        counts[group_keys[:-1] + ["n_T", "n_VT"]],
        on=group_keys[:-1],
        how="left",
    )
    cells[["n_T", "n_VT"]] = cells[["n_T", "n_VT"]].fillna(0).astype(int)

    for condition in CONDITIONS:
        cells[f"estimator_{condition}"] = np.where(
            thin_cell_n and cells[f"n_{condition}"].le(thin_cell_n),
            "mean (thin cell)",
            estimator,
        )

    baseline_keys = ["subject", "session", "speed"]

    if correction == "expectancy_model":
        # The delay-dependence is kept, but estimated as a function of delay
        # from every tactile trial rather than independently at each
        # distance. See ``expectancy_baseline``.
        fitted, _ = expectancy_baseline(trials, degree=expectancy_degree)

        if fitted.empty:
            cells["tactile_baseline_ms"] = np.nan
            cells["n_baseline"] = 0
        else:
            merge_on = [
                column
                for column in ["subject", "session", "speed", "position_rank"]
                if column in cells.columns and column in fitted.columns
            ]
            cells = cells.merge(
                fitted[merge_on + ["tactile_baseline_ms", "baseline_se_ms",
                                   "n_trials_fit"]],
                on=merge_on,
                how="left",
            )
            cells["n_baseline"] = cells["n_trials_fit"]

    elif correction == "delay_matched":
        # Delay-matched correction (Pfeiffer et al.; Masson et al.):
        # each VT RT is compared with tactile-only trials occurring at the
        # same distance/onset delay. This controls for temporal preparation
        # and foreperiod effects across the looming trajectory.
        cells["tactile_baseline_ms"] = cells["T"]
        cells["n_baseline"] = cells["n_T"]

    elif correction == "fastest_n_tactile":
        # The median of the N fastest tactile trials in the session, pooled
        # across distance. Like fastest_tactile this assumes no expectancy
        # ramp, but the floor rests on N trials rather than on whichever
        # single cell happened to come out lowest, so it does not inherit
        # the sampling noise of one four-trial cell.
        tactile = trials.loc[trials["sensory_condition"] == "T"]

        baseline = (
            tactile
            .groupby(baseline_keys, as_index=False)
            .agg(
                tactile_baseline_ms=(
                    "rt_ms",
                    lambda s: float(s.nsmallest(FASTEST_N).median()),
                ),
                n_baseline=(
                    "rt_ms",
                    lambda s: int(min(len(s), FASTEST_N)),
                ),
            )
        )

        cells = cells.merge(
            baseline,
            on=baseline_keys,
            how="left",
        )

    else:
        # A single unimodal baseline per participant, session, and speed.
        #
        # overall_tactile  averages the position-specific tactile estimates.
        # fastest_tactile  takes the participant's best tactile position
        #                  (Serino et al.; Noel et al.), which is the most
        #                  conservative reference.
        aggregation = (
            "mean" if correction == "overall_tactile" else "min"
        )

        # noel_endpoints reproduces the correction used by Noel et al.
        # exactly: they sampled tactile-only trials at two delays, the
        # temporal equivalents of the nearest and farthest positions, and
        # subtracted the faster of the two. Restricting the candidates to
        # those two positions matters, because a minimum taken over all
        # seven can only be lower, and therefore reports more apparent
        # facilitation than their method would.
        candidates = cells

        if correction == "noel_endpoints":
            ranks = cells["position_rank"].dropna()
            endpoints = [ranks.min(), ranks.max()]
            candidates = cells[cells["position_rank"].isin(endpoints)]

        baseline = (
            candidates
            .groupby(baseline_keys, as_index=False)
            .agg(
                tactile_baseline_ms=("T", aggregation),
                n_baseline=("n_T", "sum"),
            )
        )

        cells = cells.merge(
            baseline,
            on=baseline_keys,
            how="left",
        )

    cells["correction"] = correction
    cells["facilitation_ms"] = (
        cells["tactile_baseline_ms"] - cells["VT"]
    )

    return cells


def facilitation_profile(
    trials,
    estimator="median",
    correction="delay_matched",
    thin_cell_n=THIN_CELL_N,
    expectancy_degree=1,
):
    """Average sessions to obtain one profile per subject and speed.

    Trial counts are summed rather than averaged, so ``n_T`` and ``n_VT``
    still report how many trials stand behind each point.
    """
    cells = rt_cells(
        trials,
        estimator=estimator,
        correction=correction,
        thin_cell_n=thin_cell_n,
        expectancy_degree=expectancy_degree,
    )

    values = ["T", "VT", "tactile_baseline_ms", "facilitation_ms"]
    keys = ["subject", "speed", "position_rank"]

    profile = (
        cells
        .groupby(keys, as_index=False)[values]
        .mean()
    )

    counts = (
        cells
        .groupby(keys, as_index=False)[["n_T", "n_VT"]]
        .sum()
    )

    return (
        profile
        .merge(counts, on=keys, how="left")
        .assign(correction=correction)
    )


def subject_profile(profile, by=None, value="facilitation_ms"):
    """
    Collapse to one value per subject and distance.

    This is the single place where speed (and, if requested, any additional
    key) is averaged within a participant. Averaging within participants
    before averaging across them keeps every participant equally weighted
    regardless of trial counts.

    Parameters
    ----------
    profile : pandas.DataFrame
        Output of ``facilitation_profile`` or ``profile_grid``.
    by : str or sequence of str, optional
        Extra keys to retain, for example ``"speed"`` or
        ``["estimator", "correction"]``.
    """
    keys = ["subject"] + _as_key_list(by) + ["position_rank"]

    return (
        profile
        .dropna(subset=["position_rank", value])
        .groupby(keys, as_index=False)[value]
        .mean()
        .sort_values(keys)
        .reset_index(drop=True)
    )


def group_summary(individual, by=None, value="facilitation_ms"):
    """
    Average participants and return mean, SD, n, and SEM per distance.

    Expects one row per subject and distance, that is, the output of
    ``subject_profile``.
    """
    keys = _as_key_list(by) + ["position_rank"]

    group = (
        individual
        .groupby(keys, as_index=False)[value]
        .agg(
            group_mean="mean",
            group_sd="std",
            n="count",
        )
    )

    group["group_sem"] = group["group_sd"] / np.sqrt(group["n"])

    return group.sort_values(keys).reset_index(drop=True)


def profile_grid(
    trials,
    estimators=("mean", "median"),
    corrections=CORRECTIONS,
    collapse_speed=False,
):
    """
    Build participant profiles for every estimator x correction combination.

    Returns one tidy frame with columns ``estimator``, ``correction``,
    ``speed``, ``subject``, ``position_rank``, ``facilitation_ms``.

    When ``collapse_speed`` is True, slow and fast are averaged within each
    participant first and ``speed`` is set to ``"collapsed"``.
    """
    frames = []

    for estimator in estimators:
        for correction in corrections:
            one = facilitation_profile(
                trials,
                estimator=estimator,
                correction=correction,
            ).assign(estimator=estimator, correction=correction)

            if collapse_speed:
                one = subject_profile(
                    one,
                    by=["estimator", "correction"],
                ).assign(speed="collapsed")

            frames.append(one)

    return pd.concat(frames, ignore_index=True)


def compare_estimators(trials, correction="delay_matched"):
    """Compare the near-far result using means and medians."""
    renames = {
        "mean": {
            "near_facilitation": "near_mean_estimator",
            "far_facilitation": "far_mean_estimator",
            "nearfar_pps": "nearfar_mean_estimator",
        },
        "median": {
            "near_facilitation": "near_median_estimator",
            "far_facilitation": "far_median_estimator",
            "nearfar_pps": "nearfar_median_estimator",
        },
    }

    tables = {
        estimator: nearfar_table(
            facilitation_profile(
                trials,
                estimator=estimator,
                correction=correction,
            )
        ).rename(columns=renames[estimator])
        for estimator in ("mean", "median")
    }

    comparison = tables["mean"].merge(
        tables["median"],
        on="subject",
        how="outer",
    )

    comparison["mean_minus_median"] = (
        comparison["nearfar_mean_estimator"]
        - comparison["nearfar_median_estimator"]
    )

    return comparison


# ---------------------------------------------------------------------
# 2b. Which curve is driving the facilitation shape?
# ---------------------------------------------------------------------
#
# Facilitation is baseline minus VT, so its shape across distance can come
# from either term. With four tactile trials per cell the delay-matched
# baseline carries substantial sampling noise, and that noise passes into
# every point of the facilitation curve. This decomposition puts a number
# on how much of the curve is the baseline and how much is VT.
#
# Because variance is additive:
#
#     Var(facilitation) = Var(baseline) + Var(VT) - 2 Cov(baseline, VT)
#
# the three shares sum to 1 and can be read directly as proportions.
#
#     share_baseline near 1, r_facilitation_baseline near +1
#         the curve is inherited from the tactile baseline
#     share_vt near 1, r_facilitation_vt near -1
#         the curve is the visuo-tactile profile, inverted
#
# Sanity check: run with correction="fastest_tactile". The baseline is a
# constant there, so share_vt must be exactly 1 and r_facilitation_vt
# exactly -1. Anything else means the profile was built wrongly.


def _safe_correlation(a, b):
    """Pearson correlation, returning NaN rather than raising on a flat vector."""
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def baseline_contribution(
    trials,
    estimator="median",
    correction="delay_matched",
    by_speed=True,
):
    """
    Split the variance of the facilitation curve into its two sources.

    Parameters
    ----------
    trials : pandas.DataFrame
        Usable T and VT trials, normally from ``get_trials(t)``.
    estimator : {"mean", "median"}
    correction : one of ``CORRECTIONS``
    by_speed : bool, default=True
        Keep slow and fast separate. If False, the two per-speed profiles
        are averaged within each participant first, which is the same
        two-stage collapse ``subject_profile`` uses.

    Returns
    -------
    per_subject : pandas.DataFrame
        One row per participant (and speed). Spread columns are in ms, so
        they can be read against the fitted sigmoid amplitude directly.
    summary : pandas.DataFrame
        Means across participants, per speed.
    """
    if estimator not in {"mean", "median"}:
        raise ValueError("estimator must be 'mean' or 'median'")

    if correction not in CORRECTIONS:
        raise ValueError(
            "correction must be one of: " + ", ".join(CORRECTIONS)
        )

    values = ["tactile_baseline_ms", "VT", "facilitation_ms"]

    profile = facilitation_profile(
        trials,
        estimator=estimator,
        correction=correction,
    )

    keys = ["subject"] + (["speed"] if by_speed else [])

    if not by_speed:
        profile = (
            profile
            .groupby(["subject", "position_rank"], as_index=False)[values]
            .mean()
        )

    rows = []

    for key, one in profile.groupby(keys):
        one = one.sort_values("position_rank").dropna(subset=values)

        if len(one) < 3:
            continue

        baseline = one["tactile_baseline_ms"].to_numpy(dtype=float)
        vt = one["VT"].to_numpy(dtype=float)
        facilitation = one["facilitation_ms"].to_numpy(dtype=float)

        variance_facilitation = float(np.var(facilitation, ddof=1))
        variance_baseline = float(np.var(baseline, ddof=1))
        variance_vt = float(np.var(vt, ddof=1))
        covariance = float(np.cov(baseline, vt, ddof=1)[0, 1])

        if variance_facilitation > 0:
            share_baseline = variance_baseline / variance_facilitation
            share_vt = variance_vt / variance_facilitation
            share_covariance = -2.0 * covariance / variance_facilitation
        else:
            share_baseline = np.nan
            share_vt = np.nan
            share_covariance = np.nan

        record = dict(
            zip(keys, key if isinstance(key, tuple) else (key,))
        )

        record.update({
            "n_distances": int(len(one)),
            "range_baseline_ms": float(np.ptp(baseline)),
            "range_vt_ms": float(np.ptp(vt)),
            "range_facilitation_ms": float(np.ptp(facilitation)),
            "r_facilitation_baseline": _safe_correlation(
                facilitation, baseline
            ),
            "r_facilitation_vt": _safe_correlation(facilitation, vt),
            "share_baseline": share_baseline,
            "share_vt": share_vt,
            "share_covariance": share_covariance,
        })

        rows.append(record)

    per_subject = pd.DataFrame(rows)

    if per_subject.empty:
        return per_subject, pd.DataFrame()

    aggregated = {
        column: (column, "mean")
        for column in [
            "range_baseline_ms",
            "range_vt_ms",
            "range_facilitation_ms",
            "r_facilitation_baseline",
            "r_facilitation_vt",
            "share_baseline",
            "share_vt",
            "share_covariance",
        ]
    }
    aggregated["n_subjects"] = ("subject", "count")

    if by_speed:
        summary = (
            per_subject
            .groupby("speed", as_index=False)
            .agg(**aggregated)
        )
    else:
        # DataFrame.agg does not take named-aggregation tuples, so group on
        # a constant key and drop it again.
        summary = (
            per_subject
            .assign(_all="all")
            .groupby("_all", as_index=False)
            .agg(**aggregated)
            .drop(columns="_all")
        )

    summary = summary.assign(estimator=estimator, correction=correction)

    return per_subject.round(3), summary.round(3)


# ---------------------------------------------------------------------
# 3. Plot the RT components
# ---------------------------------------------------------------------

def _per_subject_panels(profile, figsize_per_row=4.0):
    """Create one stacked axis per subject and return (fig, axes, subjects)."""
    subjects = sorted(profile["subject"].unique())

    fig, axes = plt.subplots(
        len(subjects),
        1,
        figsize=(7, figsize_per_row * len(subjects)),
        squeeze=False,
    )

    return fig, axes.ravel(), subjects


def plot_rt_profiles(profile, estimator="median", save_path=None):
    """Plot T and VT RTs before subtracting them."""
    fig, axes, subjects = _per_subject_panels(profile)

    for ax, subject in zip(axes, subjects):
        one = profile[profile["subject"] == subject]

        for speed in SPEEDS:
            line = one[one["speed"] == speed].sort_values("position_rank")

            for condition in CONDITIONS:
                ax.plot(
                    line["position_rank"],
                    line[condition],
                    color=style.SPEED[speed],
                    linestyle="-" if condition == "T" else "--",
                    marker="o",
                    markerfacecolor=(
                        style.SPEED[speed]
                        if condition == "T"
                        else "white"
                    ),
                    label=f"{speed} {condition}",
                )

        _format_distance_axis(ax)
        ax.set_ylabel(f"{estimator.capitalize()} RT (ms)")
        ax.set_title(str(subject))
        ax.legend()

    fig.suptitle(
        f"T and VT RT profiles using the {estimator}",
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, save_path)

    return fig


def plot_facilitation_profiles(
    trials,
    estimator="median",
    correction="delay_matched",
    by_speed=False,
    subjects=None,
    ncols=3,
    figsize_per_panel=(3.4, 2.7),
    sharey=True,
    save_path=None,
    expectancy_degree=1,
):
    """
    Plot facilitation-by-distance for individual participants.

    Parameters
    ----------
    trials : pandas.DataFrame
        Usable T and VT trials returned by get_trials().

    estimator : {"mean", "median"}
        RT estimator.

    correction : {
        "delay_matched",
        "expectancy_model",
        "overall_tactile",
        "fastest_tactile",
        "fastest_n_tactile",
    }
        Tactile baseline used to calculate facilitation.

    by_speed : bool, default=False
        False:
            average slow and fast within participant before plotting.
        True:
            show slow and fast separately.

    subjects : sequence or None
        Participants to plot. None plots all.

    ncols : int
        Number of columns in the participant grid.

    figsize_per_panel : tuple
        Approximate size of each participant panel.

    sharey : bool
        Share y-axis across participants.

    Returns
    -------
    fig, individual
    """

    if correction not in CORRECTIONS:
        raise ValueError(
            "correction must be one of: "
            + ", ".join(CORRECTIONS)
        )

    # ---------------------------------------------------------
    # Calculate facilitation using requested baseline
    # ---------------------------------------------------------
    profile = facilitation_profile(
        trials,
        estimator=estimator,
        correction=correction,
        expectancy_degree=expectancy_degree,
    )

    if subjects is None:
        subjects = sorted(
            profile["subject"].dropna().unique()
        )
    else:
        subjects = list(subjects)
        profile = profile[
            profile["subject"].isin(subjects)
        ].copy()

    # ---------------------------------------------------------
    # Collapse speed if requested
    # ---------------------------------------------------------

    if by_speed:
        individual = profile.copy()
    else:
        individual = subject_profile(
            profile,
            value="facilitation_ms",
        )

    # ---------------------------------------------------------
    # Compact grid
    # ---------------------------------------------------------

    n_subjects = len(subjects)

    ncols = min(ncols, n_subjects)
    nrows = int(np.ceil(n_subjects / ncols))

    width = figsize_per_panel[0] * ncols
    height = figsize_per_panel[1] * nrows

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width, height),
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )

    axes = axes.ravel()

    # ---------------------------------------------------------
    # Plot participants
    # ---------------------------------------------------------

    for ax, subject in zip(axes, subjects):

        one = individual[
            individual["subject"] == subject
        ]

        if by_speed:

            for speed in SPEEDS:

                line = (
                    one[
                        one["speed"] == speed
                    ]
                    .sort_values("position_rank")
                )

                if line.empty:
                    continue

                ax.plot(
                    line["position_rank"],
                    line["facilitation_ms"],
                    color=style.SPEED[speed],
                    marker="o",
                    markersize=4,
                    linewidth=1.6,
                    label=speed.capitalize(),
                )

        else:

            line = one.sort_values(
                "position_rank"
            )

            ax.plot(
                line["position_rank"],
                line["facilitation_ms"],
                color=style.INK,
                marker="o",
                markersize=4,
                linewidth=1.6,
            )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=0.7,
            color="gray",
            zorder=0,
        )

        _format_distance_axis(ax)

        ax.set_title(
            str(subject),
            fontsize=10,
        )

        # Don't repeat axis labels in every panel
        ax.set_xlabel("")
        ax.set_ylabel("")

    # ---------------------------------------------------------
    # Remove unused axes
    # ---------------------------------------------------------

    for ax in axes[n_subjects:]:
        ax.remove()

    # Shared labels
    fig.supxlabel(
        "Distance (D1 = closest)"
    )

    fig.supylabel(
        "Facilitation, baseline - VT (ms)"
    )

    correction_label = CORRECTION_LABELS.get(
        correction,
        correction,
    )

    fig.suptitle(
        f"Individual facilitation profiles — "
        f"{correction_label} baseline",
        y=1.01,
    )

    # Single speed legend
    if by_speed and n_subjects:
        handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            title="Approach speed",
            frameon=False,
            loc="upper right",
        )

    fig.tight_layout()

    _save(
        fig,
        save_path,
    )

    return fig, individual


def plot_correction_comparison(
    trials,
    collapse_speed=False,
    show_sem=False,
    save_path=None,
):
    """
    Compare three tactile corrections under mean- and median-based RT summaries.

    Rows are estimators, columns are corrections. Each plotted point is
    calculated in two stages: one facilitation value per participant, speed,
    and distance, then an average across participants. Every participant
    therefore contributes equally regardless of trial count.

    Parameters
    ----------
    trials : pandas.DataFrame
        Usable T and VT trials, normally returned by ``get_trials(t)``.
    collapse_speed : bool, default=False
        If True, average slow and fast within each participant first and plot
        a single speed-collapsed profile.
    show_sem : bool, default=False
        Shade +/- 1 SEM across participants.
    save_path : str or pathlib.Path, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    group_profiles : pandas.DataFrame
        Group mean, SD, SEM, and sample size for every plotted point.
    """
    estimators = ("mean", "median")

    participant_profiles = profile_grid(
        trials,
        estimators=estimators,
        corrections=CORRECTIONS,
        collapse_speed=collapse_speed,
    )

    group_profiles = group_summary(
        participant_profiles,
        by=["estimator", "correction", "speed"],
    )

    fig, axes = correction_grid(estimators=estimators)

    speeds = ["collapsed"] if collapse_speed else SPEEDS

    for row, estimator in enumerate(estimators):
        for col, correction in enumerate(CORRECTIONS):
            ax = axes[row, col]

            panel = group_profiles[
                (group_profiles["estimator"] == estimator)
                & (group_profiles["correction"] == correction)
            ]

            for speed in speeds:
                line = panel[panel["speed"] == speed]

                if line.empty:
                    continue

                if collapse_speed:
                    colour = style.INK
                    label = "Average across speeds"
                else:
                    colour = style.SPEED[speed]
                    label = speed.capitalize()

                ax.plot(
                    line["position_rank"],
                    line["group_mean"],
                    color=colour,
                    marker="o",
                    markersize=4.5,
                    linewidth=2.0,
                    label=label,
                    zorder=3,
                )

                if show_sem:
                    ax.fill_between(
                        line["position_rank"],
                        line["group_mean"] - line["group_sem"],
                        line["group_mean"] + line["group_sem"],
                        color=colour,
                        alpha=0.14,
                        linewidth=0,
                        zorder=2,
                    )

            _format_distance_axis(ax)

            # Only the bottom row needs x-axis labels.
            if row == 0:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            # One compact legend is sufficient because speed coding is shared.
            if row == 0 and col == len(CORRECTIONS) - 1:
                ax.legend(frameon=False, title="Speed")

    fig.suptitle(
        "Facilitation under alternative tactile corrections",
        x=0.07,
        ha="left",
        y=0.995,
        color=style.INK,
        fontweight="regular",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, save_path)

    return fig, group_profiles


def curve_summary(
    profile,
    values=("T", "VT", "tactile_baseline_ms", "facilitation_ms"),
    by=None,
):
    """
    Subject-level and group-level summaries for several value columns at once.

    Each column is collapsed the same two-stage way as ``subject_profile``
    followed by ``group_summary``, so the RT curves and the facilitation
    curve are averaged identically and can be read against each other.

    Returns
    -------
    individual, group : pandas.DataFrame
        Long format with a ``curve`` column naming the quantity, and the
        value in ``value`` (individual) or ``group_mean`` (group).
    """
    individual_frames = []
    group_frames = []

    for value in values:
        if value not in profile.columns:
            continue

        one = subject_profile(profile, by=by, value=value)
        individual_frames.append(
            one.rename(columns={value: "value"}).assign(curve=value)
        )

        group_frames.append(
            group_summary(one, by=by, value=value).assign(curve=value)
        )

    if not individual_frames:
        return pd.DataFrame(), pd.DataFrame()

    return (
        pd.concat(individual_frames, ignore_index=True),
        pd.concat(group_frames, ignore_index=True),
    )


CURVE_STYLE = {
    "T": {"colour": "SLOW", "linestyle": "--", "marker": "s"},
    "VT": {"colour": "PATIENT", "linestyle": "-", "marker": "o"},
    "tactile_baseline_ms": {"colour": "INK", "linestyle": ":", "marker": None},
    "facilitation_ms": {"colour": "INK", "linestyle": "-", "marker": "o"},
}


def _curve_colour(curve):
    return getattr(style, CURVE_STYLE[curve]["colour"])


def plot_rt_and_facilitation(
    trials,
    corrections=("delay_matched", "fastest_tactile"),
    estimator="median",
    by_speed=False,
    show_sem=True,
    show_individuals=False,
    save_path=None,
):
    """
    Raw tactile RT, raw visuo-tactile RT, and facilitation, side by side.

    Layout
    ------
    Columns are corrections. The top row shows the reaction times the
    correction operates on; the bottom row shows the facilitation it
    produces. The two rows share an x-axis but not a y-axis, because RTs run
    in the hundreds of milliseconds and facilitation in the tens.

    Why the baseline curve is drawn
    -------------------------------
    The tactile-only and visuo-tactile RT curves are identical across
    corrections: only the baseline subtracted from VT changes. Under
    ``delay_matched`` the baseline is T at the same distance, so it lies
    exactly on the T curve and is not drawn separately. Under
    ``fastest_tactile`` it is a single value per participant, session and
    speed, so it appears as a flat line. Reading the flat baseline against
    the sloping T curve shows how much of the resulting facilitation profile
    is the tactile foreperiod effect rather than multisensory facilitation.

    Parameters
    ----------
    by_speed : bool, default=False
        If True, slow is drawn at reduced opacity alongside fast. Colour
        continues to encode the curve, not the speed.
    show_individuals : bool, default=False
        Thin per-participant lines behind the facilitation group mean.

    Returns
    -------
    fig : matplotlib.figure.Figure
    group_curves : pandas.DataFrame
        Every plotted point, in long format with a ``correction`` column.
    """
    corrections = list(corrections)

    for correction in corrections:
        if correction not in CORRECTIONS:
            raise ValueError(
                "correction must be one of: " + ", ".join(CORRECTIONS)
            )

    by = "speed" if by_speed else None
    speeds = SPEEDS if by_speed else [None]

    rows = [
        ("T", "VT", "tactile_baseline_ms"),
        ("facilitation_ms",),
    ]

    fig, axes = plt.subplots(
        2,
        len(corrections),
        figsize=(5.6 * len(corrections), 7.2),
        sharex=True,
        squeeze=False,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )

    # Share y within each row so the columns are directly comparable.
    for row in range(2):
        for col in range(1, len(corrections)):
            axes[row, col].sharey(axes[row, 0])

    collected = []

    for col, correction in enumerate(corrections):
        profile = facilitation_profile(
            trials,
            estimator=estimator,
            correction=correction,
        )

        individual, group = curve_summary(profile, by=by)
        collected.append(group.assign(correction=correction))

        # Under delay-matched the baseline is the T curve itself.
        baseline_is_T = correction == "delay_matched"

        for row, curves in enumerate(rows):
            ax = axes[row, col]

            for curve in curves:
                if curve == "tactile_baseline_ms" and baseline_is_T:
                    continue

                curve_style = CURVE_STYLE[curve]
                colour = _curve_colour(curve)

                for speed in speeds:
                    line = group[group["curve"] == curve]

                    if speed is not None:
                        line = line[line["speed"] == speed]

                    line = line.sort_values("position_rank")

                    if line.empty:
                        continue

                    alpha = 0.45 if speed == "slow" else 1.0

                    label = CURVE_LABELS[curve] + (
                        f" ({speed})" if speed is not None else ""
                    )

                    if show_sem and line["group_sem"].notna().any():
                        ax.fill_between(
                            line["position_rank"],
                            line["group_mean"] - line["group_sem"],
                            line["group_mean"] + line["group_sem"],
                            color=colour,
                            alpha=0.13 * alpha,
                            linewidth=0,
                            zorder=2,
                        )

                    ax.plot(
                        line["position_rank"],
                        line["group_mean"],
                        color=colour,
                        linestyle=curve_style["linestyle"],
                        marker=curve_style["marker"],
                        markersize=5,
                        markerfacecolor=(
                            "white" if curve == "VT" else colour
                        ),
                        linewidth=2.2,
                        alpha=alpha,
                        label=label,
                        zorder=4,
                    )

            if row == 1:
                if show_individuals:
                    thin = individual[
                        individual["curve"] == "facilitation_ms"
                    ]

                    for _, one in thin.groupby("subject"):
                        one = one.sort_values("position_rank")

                        ax.plot(
                            one["position_rank"],
                            one["value"],
                            color=style.CONTROL,
                            linewidth=0.9,
                            alpha=0.45,
                            zorder=1,
                            label="_nolegend_",
                        )

                ax.axhline(
                    0,
                    linestyle="--",
                    linewidth=0.8,
                    color="gray",
                    zorder=0,
                )

            _format_distance_axis(ax)

            if row == 0:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
                ax.set_title(CORRECTION_LABELS[correction])

            if col == 0:
                ax.set_ylabel(
                    f"{estimator.capitalize()} RT (ms)"
                    if row == 0
                    else "Facilitation, baseline - VT (ms)"
                )

            if col == len(corrections) - 1:
                ax.legend(frameon=False, fontsize=8)

        if baseline_is_T:
            axes[0, col].text(
                0.03,
                0.05,
                "Baseline = the tactile curve",
                transform=axes[0, col].transAxes,
                fontsize=8,
                color=MUTED,
            )

    fig.suptitle(
        f"Reaction times and the facilitation they produce ({estimator})",
        x=0.02,
        ha="left",
        y=1.0,
        color=style.INK,
    )

    fig.tight_layout()
    _save(fig, save_path)

    return fig, pd.concat(collected, ignore_index=True)


def plot_rt_and_facilitation_by_subject(
    trials,
    corrections=("delay_matched", "fastest_tactile"),
    estimator="median",
    subjects=None,
    show_baseline=True,
    save_path=None,
):
    """
    One panel per participant, with all three curves in the panel.

    Layout
    ------
    Rows are corrections, columns are participants. Each panel carries the
    tactile-only and visuo-tactile RTs on the left axis and the facilitation
    they produce on the right axis, because RTs run in the hundreds of
    milliseconds and facilitation in the tens; on a single scale the
    facilitation curve would be a flat line.

    Both axes are shared across every panel, so a participant whose
    facilitation curve looks large really is large relative to the others.
    Speed is averaged within each participant first.

    Under ``fastest_tactile`` the baseline is a single value per participant,
    so it is drawn as a flat dotted line on the left axis. The gap between
    that flat line and the sloping tactile curve is the foreperiod effect
    that the delay-matched correction removes.

    Returns
    -------
    fig : matplotlib.figure.Figure
    curves : pandas.DataFrame
        Every plotted point, long format, with ``correction``, ``subject``,
        ``curve``, ``position_rank`` and ``value``.
    """
    corrections = list(corrections)

    for correction in corrections:
        if correction not in CORRECTIONS:
            raise ValueError(
                "correction must be one of: " + ", ".join(CORRECTIONS)
            )

    if subjects is None:
        subjects = sorted(trials["subject"].dropna().unique())
    else:
        subjects = list(subjects)

    fig, axes = plt.subplots(
        len(corrections),
        len(subjects),
        figsize=(2.3 * len(subjects), 3.3 * len(corrections)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    twins = np.empty(axes.shape, dtype=object)
    collected = []

    for row, correction in enumerate(corrections):
        profile = facilitation_profile(
            trials,
            estimator=estimator,
            correction=correction,
        )

        individual, _ = curve_summary(profile)
        collected.append(individual.assign(correction=correction))

        baseline_is_T = correction == "delay_matched"

        for col, subject in enumerate(subjects):
            ax = axes[row, col]
            twin = ax.twinx()
            twins[row, col] = twin

            one = individual[individual["subject"] == subject]

            # Left axis: the two raw RT curves, plus the baseline when it is
            # not simply the tactile curve.
            drawn_curves = ["T", "VT"]

            if show_baseline and not baseline_is_T:
                drawn_curves.append("tactile_baseline_ms")

            for curve in drawn_curves:
                line = (
                    one[one["curve"] == curve]
                    .sort_values("position_rank")
                )

                if line.empty:
                    continue

                curve_style = CURVE_STYLE[curve]
                colour = (
                    MUTED
                    if curve == "tactile_baseline_ms"
                    else _curve_colour(curve)
                )

                ax.plot(
                    line["position_rank"],
                    line["value"],
                    color=colour,
                    linestyle=curve_style["linestyle"],
                    marker=curve_style["marker"],
                    markersize=3.5,
                    markerfacecolor="white" if curve == "VT" else colour,
                    linewidth=1.6,
                    label=CURVE_LABELS[curve],
                    zorder=3,
                )

            # Right axis: facilitation.
            line = (
                one[one["curve"] == "facilitation_ms"]
                .sort_values("position_rank")
            )

            twin.axhline(
                0,
                linestyle="--",
                linewidth=0.7,
                color="gray",
                zorder=0,
            )

            if not line.empty:
                twin.plot(
                    line["position_rank"],
                    line["value"],
                    color=style.INK,
                    linestyle="-",
                    marker="o",
                    markersize=3.5,
                    linewidth=2.0,
                    label=CURVE_LABELS["facilitation_ms"],
                    zorder=4,
                )

            _format_distance_axis(ax)

            # clean() removes the right spine, which the twin axis needs.
            twin.spines["top"].set_visible(False)
            twin.spines["right"].set_visible(True)

            if row == 0:
                ax.set_title(str(subject), fontsize=9)

            if row < len(corrections) - 1:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            if col == 0:
                ax.set_ylabel(
                    f"{CORRECTION_LABELS[correction]}\n"
                    f"{estimator.capitalize()} RT (ms)"
                )
            else:
                ax.tick_params(labelleft=False)

            if col == len(subjects) - 1:
                twin.set_ylabel(
                    "Facilitation (ms)",
                    color=style.INK,
                )
            else:
                twin.tick_params(labelright=False)

    # Share the facilitation axis across every panel too, so panel heights
    # are comparable in both quantities. sharey() copies the current limits
    # of the reference axis, so set the range from every panel's data first,
    # otherwise the reference panel silently clips the others.
    reference = twins[0, 0]

    all_facilitation = pd.concat(collected, ignore_index=True)
    all_facilitation = all_facilitation.loc[
        all_facilitation["curve"] == "facilitation_ms", "value"
    ].dropna()

    if len(all_facilitation):
        low = float(all_facilitation.min())
        high = float(all_facilitation.max())
        pad = 0.08 * (high - low if high > low else 1.0)
        reference.set_ylim(low - pad, high + pad)

    for row in range(len(corrections)):
        for col in range(len(subjects)):
            if twins[row, col] is not reference:
                twins[row, col].sharey(reference)

    # Same guard on the RT axis, which tight markers can otherwise touch.
    axes[0, 0].margins(y=0.08)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    twin_handles, twin_labels = twins[0, 0].get_legend_handles_labels()

    if len(corrections) > 1:
        extra_handles, extra_labels = axes[1, 0].get_legend_handles_labels()

        for handle, label in zip(extra_handles, extra_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    fig.legend(
        handles + twin_handles,
        labels + twin_labels,
        loc="upper center",
        ncol=len(labels + twin_labels),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.06),
    )

    fig.suptitle(
        "Reaction times and facilitation, per participant "
        f"({estimator}, speeds averaged)",
        x=0.02,
        ha="left",
        y=1.10,
        color=style.INK,
    )

    fig.tight_layout()
    _save(fig, save_path)

    return fig, pd.concat(collected, ignore_index=True)


def plot_mean_vs_median_facilitation(
    trials,
    correction="delay_matched",
    save_path=None,
):
    """
    Compare mean-based and median-based facilitation.

    For each participant:
        top row:
            mean and median facilitation profiles, separated by speed
        bottom row:
            mean-minus-median difference at each distance

    Returns
    -------
    fig : matplotlib.figure.Figure
    comparison : pandas.DataFrame
        One row per subject, speed, and distance.
    """
    profiles = {}

    for estimator in ("mean", "median"):
        profiles[estimator] = (
            facilitation_profile(
                trials,
                estimator=estimator,
                correction=correction,
            )
            [["subject", "speed", "position_rank", "T", "VT", "facilitation_ms"]]
            .rename(
                columns={
                    "T": f"{estimator}_T",
                    "VT": f"{estimator}_VT",
                    "facilitation_ms": f"{estimator}_facilitation",
                }
            )
        )

    keys = [
        "subject",
        "speed",
        "position_rank",
    ]

    comparison = profiles["mean"].merge(
        profiles["median"],
        on=keys,
        how="outer",
    )

    comparison["mean_minus_median"] = (
        comparison["mean_facilitation"]
        - comparison["median_facilitation"]
    )

    subjects = sorted(
        comparison["subject"].dropna().unique()
    )

    n_subjects = len(subjects)

    fig, axes = plt.subplots(
        nrows=2 * n_subjects,
        ncols=2,
        figsize=(12, 6.5 * n_subjects),
        squeeze=False,
        sharex="col",
    )

    for subject_index, subject in enumerate(subjects):
        subject_rows = comparison[
            comparison["subject"] == subject
        ]

        profile_row = subject_index * 2
        difference_row = profile_row + 1

        for speed_index, speed in enumerate(SPEEDS):
            speed_rows = (
                subject_rows[
                    subject_rows["speed"] == speed
                ]
                .sort_values("position_rank")
            )

            profile_ax = axes[profile_row, speed_index]
            difference_ax = axes[difference_row, speed_index]

            profile_ax.plot(
                speed_rows["position_rank"],
                speed_rows["mean_facilitation"],
                marker="o",
                linewidth=2,
                linestyle="-",
                color=style.SPEED[speed],
                label="Mean-based",
            )

            profile_ax.plot(
                speed_rows["position_rank"],
                speed_rows["median_facilitation"],
                marker="s",
                markersize=6,
                linewidth=2,
                linestyle="--",
                color=style.INK,
                label="Median-based",
            )

            # Connect mean and median estimates at each distance
            for _, row in speed_rows.iterrows():
                if (
                    pd.notna(row["mean_facilitation"])
                    and pd.notna(row["median_facilitation"])
                ):
                    profile_ax.plot(
                        [row["position_rank"], row["position_rank"]],
                        [
                            row["mean_facilitation"],
                            row["median_facilitation"],
                        ],
                        color=style.INK,
                        alpha=0.25,
                        linewidth=1,
                    )

            profile_ax.axhline(
                0,
                linestyle=":",
                linewidth=0.8,
                color="gray",
            )

            _format_distance_axis(profile_ax)
            profile_ax.set_ylabel("Facilitation, T - VT (ms)")
            profile_ax.set_title(
                f"{subject}, {speed.capitalize()} speed"
            )
            profile_ax.legend(frameon=False, fontsize=9)

            difference_ax.axhline(
                0,
                linestyle="--",
                linewidth=0.8,
                color="gray",
            )

            difference_ax.plot(
                speed_rows["position_rank"],
                speed_rows["mean_minus_median"],
                marker="o",
                linewidth=2,
                color=style.SPEED[speed],
            )

            _format_distance_axis(difference_ax)
            difference_ax.set_ylabel(
                "Mean minus median\nfacilitation (ms)"
            )
            difference_ax.set_title(
                f"{speed.capitalize()}: estimator difference"
            )

    fig.suptitle(
        "Mean-based versus median-based facilitation",
        y=1.005,
    )

    fig.tight_layout()
    _save(fig, save_path)

    return fig, comparison


def group_facilitation_table(profile, by=None):
    """
    Collapse across speed and return:
    1. one facilitation value per subject and distance
    2. the group mean, SD, n, and SEM at each distance
    """
    required = {
        "subject",
        "position_rank",
        "facilitation_ms",
    }

    missing = required.difference(profile.columns)

    if missing:
        raise ValueError(
            "profile is missing required columns: "
            + ", ".join(sorted(missing))
        )

    individual = subject_profile(profile, by=by)
    group = group_summary(individual, by=by)

    return individual, group


def plot_group_facilitation(
    profile,
    estimator="median",
    by_speed=False,
    show_individuals=True,
    show_sem=False,
    save_path=None,
):
    """
    Plot the group facilitation profile.

    Thin lines show individual participants and the thick line shows the
    group mean. The shaded band shows +/- 1 SEM where available.

    Positive facilitation means faster VT than T:
        facilitation = T - VT

    Parameters
    ----------
    by_speed : bool, default=False
        If False, average slow and fast within each participant and plot one
        curve. If True, plot slow and fast separately using ``style.SPEED``.

    Returns
    -------
    fig, individual, group
    """
    by = "speed" if by_speed else None

    individual, group = group_facilitation_table(profile, by=by)

    fig, ax = plt.subplots(figsize=(7.5, 5) if by_speed else (7, 4.5))

    tracks = (
        [(speed, style.SPEED[speed], speed.capitalize()) for speed in SPEEDS]
        if by_speed
        else [(None, style.INK, "Group mean")]
    )

    for key, colour, label in tracks:
        if key is None:
            track_individual = individual
            track_group = group
        else:
            track_individual = individual[individual["speed"] == key]
            track_group = group[group["speed"] == key]

        if show_individuals:
            for _, rows in track_individual.groupby("subject"):
                rows = rows.sort_values("position_rank")

                ax.plot(
                    rows["position_rank"],
                    rows["facilitation_ms"],
                    color=colour if by_speed else style.CONTROL,
                    marker=None if by_speed else "o",
                    markersize=3,
                    linewidth=1,
                    alpha=0.20 if by_speed else 0.45,
                )

        track_group = track_group.sort_values("position_rank")

        if (
            show_sem
            and track_group["group_sem"].notna().any()
            and (track_group["n"] >= 2).any()
        ):
            ax.fill_between(
                track_group["position_rank"],
                track_group["group_mean"] - track_group["group_sem"],
                track_group["group_mean"] + track_group["group_sem"],
                color=colour,
                alpha=0.18 if by_speed else 0.12,
                linewidth=0,
            )

        ax.plot(
            track_group["position_rank"],
            track_group["group_mean"],
            marker="o",
            linewidth=2.5,
            color=colour,
            label=label,
            zorder=5,
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=0.8,
        color="gray",
    )

    _format_distance_axis(ax)
    ax.set_ylabel("Facilitation, T - VT (ms)")

    if by_speed:
        ax.set_title(
            f"Group facilitation by speed ({estimator}-based)"
        )
        ax.legend(title="Approach speed", frameon=False)
    else:
        ax.set_title(
            f"Group facilitation by distance using the {estimator}"
        )
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    _save(fig, save_path)

    return fig, individual, group


def plot_group_facilitation_by_speed(
    profile,
    estimator="mean",
    show_individuals=False,
    show_sem=True,
    save_path=None,
):
    """
    Deprecated alias for ``plot_group_facilitation(by_speed=True)``.

    Kept so existing notebook cells keep working. Note that the returned
    ``group`` frame now uses the shared column names ``group_mean``,
    ``group_sd``, ``n``, ``group_sem``.
    """
    return plot_group_facilitation(
        profile,
        estimator=estimator,
        by_speed=True,
        show_individuals=show_individuals,
        show_sem=show_sem,
        save_path=save_path,
    )


# ---------------------------------------------------------------------
# 4. Near-far H1 test
# ---------------------------------------------------------------------

def nearfar_table(profile):
    """Calculate near-minus-far facilitation per participant."""
    by_position = subject_profile(profile)

    rows = []

    for subject, group in by_position.groupby("subject"):
        near = group.loc[
            group["position_rank"].isin(NEAR),
            "facilitation_ms",
        ].mean()

        far = group.loc[
            group["position_rank"].isin(FAR),
            "facilitation_ms",
        ].mean()

        rows.append({
            "subject": subject,
            "near_facilitation": near,
            "far_facilitation": far,
            "nearfar_pps": near - far,
        })

    return pd.DataFrame(rows)


def group_test(nearfar):
    """Bootstrap and one-sided Wilcoxon test of near-far values."""
    values = nearfar["nearfar_pps"].dropna().to_numpy()
    n = len(values)

    if n < 2:
        return {
            "n": n,
            "mean": values.mean() if n else np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "wilcoxon_W": np.nan,
            "wilcoxon_p": np.nan,
            "supported": False,
        }

    boot = bootstrap_mean_ci(
        values,
        n_boot=config.N_BOOT,
        ci=95,
        seed=config.RANDOM_SEED,
    )

    wilcoxon = wilcoxon_one_sided(
        values,
        alternative="greater",
    )

    return {
        "n": n,
        "mean": values.mean(),
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "wilcoxon_W": wilcoxon["W"],
        "wilcoxon_p": wilcoxon["p"],
        "supported": bool(boot["ci_low"] > 0),
    }


# ---------------------------------------------------------------------
# 4b. Near-far index as a figure, organised by design
# ---------------------------------------------------------------------
#
# ``nearfar_table`` collapses speed and returns one value per participant,
# which is what the group test needs but hides two things a reader wants:
# whether the index survives at both speeds, and whether the three designs
# behave differently. These functions keep both.


def nearfar_by_subject(
    profile,
    by_speed=True,
    design_map=None,
):
    """Near-minus-far facilitation per participant, optionally per speed.

    ``nearfar_table`` averages the speeds first. This does not, so a
    participant who shows the effect at one speed and not the other is
    visible rather than averaged away.

    Parameters
    ----------
    profile : pandas.DataFrame
        Output of ``facilitation_profile``.
    by_speed : bool, default=True
        Keep slow and fast apart. If False the two are averaged within each
        participant first, reproducing ``nearfar_table``.
    design_map : dict or None
        Subject to design label. Adds a ``design`` column for grouping.

    Returns
    -------
    pandas.DataFrame
        subject, [speed], near_facilitation, far_facilitation, nearfar_pps,
        n_near, n_far, [design].
    """
    keys = ["subject"] + (["speed"] if by_speed else [])

    individual = subject_profile(
        profile,
        by="speed" if by_speed else None,
    )

    rows = []

    for key, one in individual.groupby(keys):
        key_values = key if isinstance(key, tuple) else (key,)

        near = one.loc[one["position_rank"].isin(NEAR), "facilitation_ms"]
        far = one.loc[one["position_rank"].isin(FAR), "facilitation_ms"]

        record = dict(zip(keys, key_values))
        record.update({
            "near_facilitation": near.mean(),
            "far_facilitation": far.mean(),
            "nearfar_pps": near.mean() - far.mean(),
            "n_near": int(near.notna().sum()),
            "n_far": int(far.notna().sum()),
        })

        rows.append(record)

    out = pd.DataFrame(rows)

    if design_map is not None and len(out):
        out["design"] = out["subject"].map(design_map)

    return out.sort_values(keys).reset_index(drop=True)


def plot_nearfar_pairs(
    nearfar,
    *,
    group_by=None,
    speed=None,
    near_color="#19d3c5",
    far_color="#bbe2dc",
    point_size=36,
    line_alpha=0.30,
    line_width=0.9,
    jitter_width=0.025,
    seed=0,
):
    """
    Plot Near and Far facilitation as paired subject-level points.

    Near is positioned close to the y-axis, Far to the right.
    Each subject's Near and Far values are connected by a thin line.

    Parameters
    ----------
    nearfar : DataFrame
        Output of nearfar_by_subject(...).

    group_by : str or None
        Optional grouping column, e.g. "design".

    speed : {"slow", "fast"} or None
        If provided, plot only that speed.

    near_color, far_color : matplotlib colors
        Colors for Near and Far points.

    Returns
    -------
    matplotlib.figure.Figure
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import style

    df = nearfar.copy()

    # ---------------------------------------------------------
    # Optional speed selection
    # ---------------------------------------------------------

    if speed is not None:
        if "speed" not in df.columns:
            raise ValueError(
                "nearfar has no 'speed' column, so speed cannot be selected."
            )

        df = df.loc[
            df["speed"] == speed
        ].copy()

    # ---------------------------------------------------------
    # Groups / panels
    # ---------------------------------------------------------

    if group_by is None:
        groups = [("All", df)]
    else:
        groups = list(
            df.groupby(
                group_by,
                dropna=False,
            )
        )

    n_groups = len(groups)

    fig, axes = plt.subplots(
        1,
        n_groups,
        figsize=(
            max(3.0 * n_groups, 3.4),
            3.5,
        ),
        squeeze=False,
        sharey=True,
    )

    axes = axes.ravel()

    rng = np.random.default_rng(seed)

    # Put Near quite close to the y-axis
    x_near = 0.18
    x_far = 0.82

    for ax, (group_name, one) in zip(
        axes,
        groups,
    ):

        paired = one[
            [
                "subject",
                "near_facilitation",
                "far_facilitation",
            ]
        ].dropna(
            subset=[
                "near_facilitation",
                "far_facilitation",
            ]
        )

        # -----------------------------------------------------
        # Connecting lines first, behind points
        # -----------------------------------------------------

        for _, row in paired.iterrows():

            ax.plot(
                [x_near, x_far],
                [
                    row["near_facilitation"],
                    row["far_facilitation"],
                ],
                color=style.MUTED,
                alpha=line_alpha,
                linewidth=line_width,
                zorder=1,
            )

        # -----------------------------------------------------
        # Slight x jitter so overlapping subjects remain visible
        # -----------------------------------------------------

        n = len(paired)

        near_x = (
            np.full(n, x_near)
            + rng.uniform(
                -jitter_width,
                jitter_width,
                size=n,
            )
        )

        far_x = (
            np.full(n, x_far)
            + rng.uniform(
                -jitter_width,
                jitter_width,
                size=n,
            )
        )

        # -----------------------------------------------------
        # Near points
        # -----------------------------------------------------

        ax.scatter(
            near_x,
            paired["near_facilitation"],
            s=point_size,
            facecolor=near_color,
            edgecolors="none",
            linewidths=0,
            alpha=0.85,
            zorder=3,
        )

        # -----------------------------------------------------
        # Far points
        # -----------------------------------------------------

        ax.scatter(
            far_x,
            paired["far_facilitation"],
            s=point_size,
            facecolor=far_color,
            edgecolors="none",
            linewidths=0,
            alpha=0.85,
            zorder=3,
        )

        # -----------------------------------------------------
        # Reference line
        # -----------------------------------------------------

        ax.axhline(
            0,
            color=style.RULE,
            linewidth=0.8,
            zorder=0,
        )

        # -----------------------------------------------------
        # Axis formatting
        # -----------------------------------------------------

        ax.set_xlim(
            0.0,
            1.0,
        )

        ax.set_xticks(
            [x_near, x_far]
        )

        ax.set_xticklabels(
            ["Near", "Far"]
        )

        ax.set_ylabel(
            "Facilitation (ms)"
        )

        if group_by is not None:
            ax.set_title(
                str(group_name)
            )
        elif speed is not None:
            ax.set_title(
                str(speed).capitalize()
            )

        style.clean(ax)

    style.finish_figure(fig)

    return fig

def plot_nearfar_estimation(
    nearfar,
    *,
    group_by=None,
    speed=None,
    near_color=None,
    far_color=None,
    box_alpha=0.22,
    point_alpha=0.65,
    point_size=32,
    jitter_width=0.07,
    sharey=True,
    save_path=None,
):
    """
    Plot the Near and Far facilitation distributions as separate box/point plots.

    This is the group-level companion to ``plot_nearfar_pairs``:
    - Near and Far are shown as separate distributions.
    - box border, whiskers, caps and median use the full category colour.
    - box fill uses a light alpha of the same colour.
    - individual observations are superimposed without marker borders.

    Parameters
    ----------
    nearfar : pandas.DataFrame
        Output of ``nearfar_by_subject``.

    group_by : str or None
        Optional panel grouping column, e.g. ``"design"``.

    speed : {"slow", "fast"} or None
        If supplied, restrict to one speed.

    near_color, far_color : matplotlib colors or None
        Defaults to ``style.NEAR`` / ``style.FAR`` when available, otherwise
        the project turquoise values.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = nearfar.copy()

    if speed is not None:
        if "speed" not in df.columns:
            raise ValueError(
                "nearfar has no 'speed' column, so speed cannot be selected."
            )
        df = df.loc[df["speed"].eq(speed)].copy()

    near_color = (
        near_color
        if near_color is not None
        else getattr(style, "NEAR", "#19d3c5")
    )
    far_color = (
        far_color
        if far_color is not None
        else getattr(style, "FAR", "#bbe2dc")
    )

    if group_by is None:
        groups = [("All", df)]
    else:
        if group_by not in df.columns:
            raise KeyError(
                f"{group_by!r} is not a column in nearfar. "
                f"Available columns: {list(df.columns)}"
            )
        groups = list(df.groupby(group_by, dropna=False))

    if not groups:
        raise ValueError("No data available to plot.")

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(max(3.2 * len(groups), 3.6), 3.5),
        sharey=sharey,
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, (group_name, one) in zip(axes, groups):
        near = pd.to_numeric(
            one["near_facilitation"], errors="coerce"
        ).dropna().to_numpy(float)

        far = pd.to_numeric(
            one["far_facilitation"], errors="coerce"
        ).dropna().to_numpy(float)

        # Use the central style helper when available so these boxes match
        # every other boxplot in the project.
        if hasattr(style, "boxplot_points"):
            style.boxplot_points(
                ax,
                data=[near, far],
                positions=[0, 1],
                labels=["Near", "Far"],
                colors=[near_color, far_color],
                box_alpha=box_alpha,
                point_alpha=point_alpha,
                point_size=point_size,
                jitter_width=jitter_width,
                showfliers=False,
            )
        else:
            bp = ax.boxplot(
                [near, far],
                positions=[0, 1],
                widths=0.62,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
            )

            for i, color in enumerate([near_color, far_color]):
                bp["boxes"][i].set_facecolor(
                    plt.matplotlib.colors.to_rgba(color, box_alpha)
                )
                bp["boxes"][i].set_edgecolor(color)
                bp["boxes"][i].set_linewidth(1.25)

                for artist in bp["whiskers"][2 * i:2 * i + 2]:
                    artist.set_color(color)
                    artist.set_linewidth(1.25)

                for artist in bp["caps"][2 * i:2 * i + 2]:
                    artist.set_color(color)
                    artist.set_linewidth(1.25)

                bp["medians"][i].set_color(color)
                bp["medians"][i].set_linewidth(1.6)

            rng = np.random.default_rng(0)
            for i, (values, color) in enumerate(
                zip([near, far], [near_color, far_color])
            ):
                if len(values):
                    x = i + rng.uniform(
                        -jitter_width, jitter_width, size=len(values)
                    )
                    ax.scatter(
                        x,
                        values,
                        s=point_size,
                        facecolor=color,
                        edgecolors="none",
                        linewidths=0,
                        alpha=point_alpha,
                        zorder=3,
                    )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Near", "Far"])

        ax.axhline(
            0,
            color=getattr(style, "RULE", "#E6E6E6"),
            linewidth=0.8,
            zorder=0,
        )

        if group_by is not None:
            ax.set_title(str(group_name))
        elif speed is not None:
            ax.set_title(str(speed).capitalize())

        ax.set_ylabel("")
        style.clean(ax)

    axes[0].set_ylabel("Facilitation (ms)")

    if speed is None and "speed" in df.columns and df["speed"].nunique() > 1:
        fig.suptitle("Near vs far facilitation — speeds pooled")
    elif speed is not None:
        fig.suptitle(f"Near vs far facilitation — {speed}")
    else:
        fig.suptitle("Near vs far facilitation")

    if hasattr(style, "finish_figure"):
        style.finish_figure(fig)
    else:
        fig.tight_layout()

    _save(fig, save_path)
    return fig


def plot_facilitation_by_design(
    profile,
    design_map,
    by_speed=True,
    show_mean=True,
    sharey=True,
    save_path=None,
):
    """
    Individual facilitation profiles, one panel per design.

    Individual participants are shown as faint trajectories.
    The design mean is shown prominently in the speed color.
    """

    data = profile.copy()
    data["design"] = data["subject"].map(design_map)

    designs = sorted(
        data["design"].dropna().unique(),
        key=str,
    )

    n_designs = len(designs)

    fig, axes = plt.subplots(
        1,
        n_designs,
        figsize=(
            max(3.7 * n_designs, 4.5),
            3.8,
        ),
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )

    axes = axes[0]

    # ---------------------------------------------------------
    # Speeds / plotting tracks
    # ---------------------------------------------------------

    tracks = (
        SPEEDS
        if by_speed
        else [None]
    )

    for ax, design in zip(
        axes,
        designs,
    ):

        one = data.loc[
            data["design"] == design
        ].copy()

        subjects = sorted(
            one["subject"]
            .dropna()
            .unique()
        )

        # -----------------------------------------------------
        # Zero reference
        # -----------------------------------------------------

        ax.axhline(
            0,
            color=style.RULE,
            linewidth=0.9,
            zorder=0,
        )

        # -----------------------------------------------------
        # Draw each speed separately
        # -----------------------------------------------------

        for speed in tracks:

            selected = (
                one
                if speed is None
                else one.loc[
                    one["speed"] == speed
                ]
            )

            if selected.empty:
                continue

            colour = (
                style.INK
                if speed is None
                else style.SPEED[speed]
            )

            # -------------------------------------------------
            # Individual participant profiles
            # -------------------------------------------------

            for subject in subjects:

                line = (
                    selected.loc[
                        selected["subject"]
                        == subject
                    ]
                    .sort_values(
                        "position_rank"
                    )
                )

                if line.empty:
                    continue

                ax.plot(
                    line["position_rank"],
                    line["facilitation_ms"],
                    color=colour,
                    linewidth=0.9,
                    alpha=0.20,
                    zorder=1,
                )

                ax.scatter(
                    line["position_rank"],
                    line["facilitation_ms"],
                    s=16,
                    facecolor=colour,
                    edgecolors="none",
                    linewidths=0,
                    alpha=0.28,
                    zorder=2,
                )

            # -------------------------------------------------
            # Design mean
            # -------------------------------------------------

            if show_mean:

                mean = (
                    selected
                    .groupby(
                        "position_rank",
                        as_index=False,
                    )["facilitation_ms"]
                    .mean()
                    .sort_values(
                        "position_rank"
                    )
                )

                ax.plot(
                    mean["position_rank"],
                    mean["facilitation_ms"],
                    color=colour,
                    linewidth=2.6,
                    zorder=4,
                    label=(
                        speed.capitalize()
                        if speed is not None
                        else "Mean"
                    ),
                )

                ax.scatter(
                    mean["position_rank"],
                    mean["facilitation_ms"],
                    s=38,
                    facecolor=colour,
                    edgecolors="none",
                    linewidths=0,
                    alpha=1.0,
                    zorder=5,
                )

        # -----------------------------------------------------
        # Axis formatting
        # -----------------------------------------------------

        _format_distance_axis(ax)

        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.set_title(
            f"{design}",
            fontsize=10,
        )

        # Participant count as secondary annotation
        ax.text(
            0.02,
            0.98,
            f"n = {len(subjects)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=style.MUTED,
        )

        style.clean(ax)

    # ---------------------------------------------------------
    # Shared labels
    # ---------------------------------------------------------

    axes[0].set_ylabel(
        "Facilitation (ms)"
    )

    fig.supxlabel(
        "Distance (D1 = closest)"
    )

    # ---------------------------------------------------------
    # One legend only
    # ---------------------------------------------------------

    if by_speed:

        handles, labels = (
            axes[-1]
            .get_legend_handles_labels()
        )

        if handles:
            axes[-1].legend(
                handles,
                labels,
                title="Speed",
                frameon=False,
                fontsize=8,
                title_fontsize=8,
                loc="best",
            )

    fig.suptitle(
        "Facilitation profiles by design",
        fontsize=11,
    )

    if hasattr(style, "finish_figure"):
        style.finish_figure(fig)
    else:
        fig.tight_layout()

    _save(
        fig,
        save_path,
    )

    return fig



# ---------------------------------------------------------------------
# 4c. Why some participants cannot carry a boundary estimate
# ---------------------------------------------------------------------

def subject_variability(trials, profile=None, design_map=None):
    """Per-participant noise, against the signal it has to reveal.

    The columns that decide whether a boundary is estimable at all:

        cell_sd_T / cell_sd_VT    within-cell spread, in ms
        se_T / se_VT              standard error of a cell estimate
        n_T / n_VT                median trials per cell
        profile_range_ms          how far the facilitation profile travels
        signal_to_noise           that range divided by the tactile SE

    A signal-to-noise near 1 means the whole profile is the size of the
    error bar on one of its points, so no fit to it can be trusted.
    """
    keys = _cell_keys(trials)

    cells = (
        trials
        .groupby(keys, dropna=False)["rt_ms"]
        .agg(n="size", sd="std")
        .reset_index()
    )

    cells["se"] = cells["sd"] / np.sqrt(cells["n"])

    rows = []

    for subject, one in cells.groupby("subject"):
        record = {"subject": subject}

        for condition in CONDITIONS:
            selected = one.loc[one["sensory_condition"] == condition]

            record[f"n_{condition}"] = float(selected["n"].median())
            record[f"cell_sd_{condition}"] = float(selected["sd"].mean())
            record[f"se_{condition}"] = float(selected["se"].mean())

        rows.append(record)

    out = pd.DataFrame(rows)

    if profile is not None:
        span = (
            profile
            .dropna(subset=["facilitation_ms"])
            .groupby("subject")["facilitation_ms"]
            .agg(lambda values: float(np.ptp(values)))
            .rename("profile_range_ms")
            .reset_index()
        )
        out = out.merge(span, on="subject", how="left")
        out["signal_to_noise"] = out["profile_range_ms"] / out["se_T"]

    if design_map is not None:
        out["design"] = out["subject"].map(design_map)

    columns = (
        ["design"] if "design" in out.columns else []
    ) + [c for c in out.columns if c != "design"]

    return out[columns].round(2)


def plot_subject_variability(
    variability,
    reference=None,
    save_path=None,
):
    """Two panels: the noise per participant, and the signal against it.

    Left, the standard error of a cell estimate in each condition. Right,
    the facilitation profile's range divided by that error. Participants
    below the dashed line on the right have no profile larger than the
    uncertainty on a single point of it.

    ``reference`` draws a horizontal line on the right panel; 2 is a
    reasonable floor for attempting a fit.
    """
    data = variability.dropna(subset=["se_T"]).copy()
    data = data.sort_values("se_T")

    has_snr = "signal_to_noise" in data.columns
    n_panels = 2 if has_snr else 1

    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.6 * n_panels, 4.6), squeeze=False,
    )
    axes = axes[0]

    x = np.arange(len(data))

    # --- panel A: the noise floor --------------------------------
    ax = axes[0]

    for condition, offset in [("T", -0.18), ("VT", 0.18)]:
        column = f"se_{condition}"

        if column not in data.columns:
            continue

        ax.bar(
            x + offset,
            data[column],
            width=0.34,
            color=_curve_colour(condition),
            alpha=0.85,
            label=CURVE_LABELS[condition],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(data["subject"], rotation=45, ha="right")
    ax.set_ylabel("Standard error of a cell estimate (ms)")
    ax.set_title("Noise per participant")
    ax.legend(frameon=False, fontsize=8)
    style.clean(ax)

    # --- panel B: signal against noise ---------------------------
    if has_snr:
        ax = axes[1]

        colours = [
            style.CONTROL if value >= (reference or 2)
            else style.SLOW
            for value in data["signal_to_noise"]
        ]

        ax.bar(x, data["signal_to_noise"], width=0.6, color=colours,
               edgecolor=style.INK, linewidth=0.6)

        if reference:
            ax.axhline(
                reference,
                linestyle="--",
                linewidth=1.0,
                color="gray",
            )
            ax.text(
                len(data) - 0.5,
                reference,
                f" {reference}",
                va="bottom",
                ha="right",
                fontsize=8,
                color=MUTED,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(data["subject"], rotation=45, ha="right")
        ax.set_ylabel("Profile range / tactile SE")
        ax.set_title("Signal against its own noise")
        style.clean(ax)

    fig.tight_layout()
    _save(fig, save_path)

    return fig


# ---------------------------------------------------------------------
# 5. Half-maximum boundary, the preregistered sigmoid fallback
# ---------------------------------------------------------------------

def half_max_boundary(profile, value_col="facilitation_ms"):
    """
    Estimate the PPS boundary directly from the observed facilitation profile.

    Preregistered fallback, used where the fitted sigmoid centre is not
    estimable with usable precision:

        far plateau  = mean(D7, D6)
        near plateau = mean(D1, D2)
        half maximum = midpoint between the two plateaus

    Starting at D7 and walking inward to D1, the first adjacent pair whose
    values bracket the half maximum defines the boundary, which is linearly
    interpolated between them.

    Returns
    -------
    pandas.DataFrame
        One row per subject and speed with the plateaus, the half level, the
        interpolated boundary in ranks and metres, the bracketing distances,
        and a ``halfmax_ok`` flag with the reason where it failed.
    """
    required = {"subject", "speed", "position_rank", value_col}
    missing = required.difference(profile.columns)

    if missing:
        raise ValueError(
            "profile is missing required columns: "
            + ", ".join(sorted(missing))
        )

    plateau_positions = {1, 2, 6, 7}
    inward = [7, 6, 5, 4, 3, 2, 1]

    rows = []

    for (subject, speed), one in profile.groupby(["subject", "speed"]):
        values = (
            one[["position_rank", value_col]]
            .dropna()
            .sort_values("position_rank")
            .set_index("position_rank")[value_col]
        )

        record = {
            "subject": subject,
            "speed": speed,
            "far_plateau": np.nan,
            "near_plateau": np.nan,
            "half_level": np.nan,
            "x_half_rank": np.nan,
            "x_half_m": np.nan,
            "bracket_outer": np.nan,
            "bracket_inner": np.nan,
            "halfmax_ok": False,
            "reason": "",
        }

        if not plateau_positions.issubset(set(values.index)):
            record["reason"] = "missing plateau distance"
            rows.append(record)
            continue

        far_plateau = float(np.mean([values.loc[7], values.loc[6]]))
        near_plateau = float(np.mean([values.loc[1], values.loc[2]]))
        half_level = (far_plateau + near_plateau) / 2.0

        record.update({
            "far_plateau": far_plateau,
            "near_plateau": near_plateau,
            "half_level": half_level,
        })

        for outer, inner in zip(inward[:-1], inward[1:]):
            if outer not in values.index or inner not in values.index:
                continue

            y_outer = float(values.loc[outer])
            y_inner = float(values.loc[inner])

            if not (
                min(y_outer, y_inner) <= half_level <= max(y_outer, y_inner)
            ):
                continue

            if np.isclose(y_inner, y_outer):
                # Flat segment sitting exactly at the half level.
                x_half = (outer + inner) / 2.0
            else:
                x_half = outer + (half_level - y_outer) / (
                    y_inner - y_outer
                ) * (inner - outer)

            record.update({
                "x_half_rank": x_half,
                "x_half_m": rank_to_metres(x_half),
                "bracket_outer": outer,
                "bracket_inner": inner,
                "halfmax_ok": True,
                "reason": "ok",
            })
            break

        if not record["halfmax_ok"]:
            record["reason"] = "half level not bracketed"

        rows.append(record)

    return pd.DataFrame(rows)


def delta_pps(halfmax):
    """Fast minus slow boundary per participant, from the half-max estimates.

    Only participants with a usable boundary at both speeds contribute.
    """
    usable = halfmax[halfmax["halfmax_ok"]]

    wide = usable.pivot(
        index="subject", columns="speed", values="x_half_m"
    )

    for speed in SPEEDS:
        if speed not in wide.columns:
            wide[speed] = np.nan

    wide["delta_pps_m"] = wide["fast"] - wide["slow"]

    return wide.reset_index()


def plot_halfmax_boundaries(profile, halfmax, save_path=None):
    """
    Plot observed facilitation profiles with the half-maximum boundaries.

    For each participant and speed: the observed profile, the far plateau
    (D7 and D6), the near plateau (D1 and D2), the half level between them,
    and the interpolated boundary where the profile first crosses it walking
    inward from D7.
    """
    subjects = sorted(profile["subject"].dropna().unique())

    fig, axes = plt.subplots(
        nrows=len(subjects),
        ncols=1,
        figsize=(7.5, 3.8 * len(subjects)),
        squeeze=False,
        sharex=True,
    )
    axes = axes.ravel()

    for ax, subject in zip(axes, subjects):
        subject_profile_rows = profile[profile["subject"].eq(subject)]
        subject_halfmax = halfmax[halfmax["subject"].eq(subject)]

        for speed in SPEEDS:
            one = (
                subject_profile_rows[subject_profile_rows["speed"].eq(speed)]
                .sort_values("position_rank")
            )
            boundary = subject_halfmax[subject_halfmax["speed"].eq(speed)]

            if one.empty or boundary.empty:
                continue

            boundary = boundary.iloc[0]
            colour = style.SPEED[speed]

            ax.plot(
                one["position_rank"],
                one["facilitation_ms"],
                marker="o",
                linewidth=2.0,
                color=colour,
                label=f"{speed.capitalize()} facilitation",
                zorder=4,
            )

            if not np.isfinite(boundary["half_level"]):
                continue

            # The two plateaus that define the half level.
            ax.plot(
                [6, 7],
                [boundary["far_plateau"]] * 2,
                color=colour,
                linewidth=1.3,
                linestyle=":",
                alpha=0.7,
            )
            ax.plot(
                [1, 2],
                [boundary["near_plateau"]] * 2,
                color=colour,
                linewidth=1.3,
                linestyle=":",
                alpha=0.7,
            )
            ax.axhline(
                boundary["half_level"],
                color=colour,
                linewidth=1.0,
                linestyle="--",
                alpha=0.55,
            )

            if boundary["halfmax_ok"]:
                ax.axvline(
                    boundary["x_half_rank"],
                    color=colour,
                    linewidth=1.8,
                    linestyle="--",
                    alpha=0.9,
                )
                ax.scatter(
                    boundary["x_half_rank"],
                    boundary["half_level"],
                    s=50,
                    color=colour,
                    edgecolor="white",
                    linewidth=1.0,
                    zorder=6,
                )
                ax.annotate(
                    f"{speed}: {boundary['x_half_m']:.2f} m",
                    (boundary["x_half_rank"], boundary["half_level"]),
                    xytext=(6, 8),
                    textcoords="offset points",
                    fontsize=8,
                    color=colour,
                )
            else:
                ax.annotate(
                    f"{speed}: {boundary['reason']}",
                    (7, boundary["half_level"]),
                    xytext=(-6, 8),
                    textcoords="offset points",
                    ha="right",
                    fontsize=7.5,
                    color=MUTED,
                )

        ax.axhline(
            0,
            color=MUTED,
            linewidth=0.8,
            linestyle=":",
            zorder=0,
        )

        _format_distance_axis(ax, unit="m")
        ax.set_ylabel("Facilitation, T - VT (ms)")
        ax.set_title(str(subject))
        ax.legend(frameon=False, fontsize=8, ncol=2)
        style.clean(ax)

    fig.suptitle(
        "Observed half-maximum PPS boundaries",
        x=0.02,
        ha="left",
        y=1.0,
        color=style.INK,
    )

    fig.tight_layout()
    _save(fig, save_path)

    return fig


# ----------------------------------------------------------------------
# Near against far, as a paired comparison
# ----------------------------------------------------------------------

# The same colours plot_nearfar_estimation uses, so the two figures of
# the same quantity read alike.
NEARFAR_NEAR_COLOUR = getattr(style, "NEAR", "#19d3c5")
NEARFAR_FAR_COLOUR = getattr(style, "FAR", "#bbe2dc")


def _stars(p):
    """Conventional significance markers, or 'ns'."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def plot_nearfar_boxes(profile, exclude=None, subjects=None, n_boot=None,
                       save_path=None, verbose=True):
    """Near against far facilitation, with the paired test.

    The near-far index is a difference score, so its group test is a
    one-sample test against zero. This draws the two terms it is built
    from, which shows what the index cannot: whether a participant with a
    small value has little facilitation anywhere or a great deal at both
    distances.

    Participants are joined by a line because the test is paired. Boxes
    alone would invite a reader to compare two distributions that are not
    independent, and would hide whether a small group difference comes
    from small individual differences or from large ones cancelling.

    The asterisk comes from the same one-sided Wilcoxon that
    ``group_test`` applies to the index, so the figure and the reported
    statistic refer to one analysis rather than two.
    """
    table = nearfar_table(profile)

    if subjects is not None:
        table = table[table["subject"].isin(list(subjects))]

    if exclude:
        dropped = sorted(set(exclude) & set(table["subject"]))
        table = table[~table["subject"].isin(list(exclude))]

        if verbose and dropped:
            print(f"Excluded: {', '.join(dropped)}")

        missing = sorted(set(exclude) - set(dropped))

        if verbose and missing:
            print(f"  not present: {', '.join(missing)}")

    table = table.dropna(subset=["near_facilitation", "far_facilitation"])
    table = table.sort_values("subject").reset_index(drop=True)

    if len(table) < 3:
        raise ValueError(f"only {len(table)} participants after filtering")

    near = table["near_facilitation"].to_numpy(dtype=float)
    far = table["far_facilitation"].to_numpy(dtype=float)
    differences = table["nearfar_pps"].to_numpy(dtype=float)

    boot = bootstrap_mean_ci(
        differences, n_boot=n_boot or config.N_BOOT, ci=95,
        seed=config.RANDOM_SEED,
    )
    test = wilcoxon_one_sided(differences, alternative="greater")

    fig, ax = plt.subplots(figsize=(3.8, 3.6))

    # Drawn with the same helper as plot_nearfar_estimation, so the box
    # fill, border weight and point styling match rather than being
    # reimplemented and drifting apart.
    style.boxplot_points(
        ax,
        data=[near, far],
        positions=[0, 1],
        labels=["Near\n(D1, D2)", "Far\n(D6, D7)"],
        colors=[NEARFAR_NEAR_COLOUR, NEARFAR_FAR_COLOUR],
        box_alpha=0.22,
        point_alpha=0.75,
        point_size=32,
        jitter_width=0.06,
        showfliers=False,
    )

    # One line per participant, drawn over the boxes. The test is on the
    # paired differences, and boxes alone would invite a reader to
    # compare two distributions that are not independent.
    for _, row in table.iterrows():
        ax.plot(
            [0, 1], [row["near_facilitation"], row["far_facilitation"]],
            color=style.MUTED, linewidth=0.8, alpha=0.45, zorder=2,
        )

    top = max(near.max(), far.max())
    bottom = min(near.min(), far.min())
    span = top - bottom if top > bottom else 1.0

    bar = top + 0.10 * span
    ax.plot(
        [0, 0, 1, 1],
        [bar, bar + 0.03 * span, bar + 0.03 * span, bar],
        color=style.INK, linewidth=1.0,
    )
    ax.text(0.5, bar + 0.05 * span, _stars(test["p"]),
            ha="center", va="bottom", fontsize=11, color=style.INK)

    ax.axhline(0, linestyle="--", linewidth=0.8, color=style.MUTED,
               zorder=0)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(bottom - 0.12 * span, bar + 0.16 * span)
    ax.set_ylabel("Facilitation (ms)")
    ax.set_title("Near vs far facilitation, speeds pooled",
                 loc="left", fontsize=9.5)
    style.clean(ax)

    if verbose:
        print(table.round(1).to_string(index=False))
        print()
        print(f"  n                = {len(table)}")
        print(f"  mean near-far    = {boot['mean']:.1f} ms")
        print(f"  bootstrap 95% CI = [{boot['ci_low']:.1f}, "
              f"{boot['ci_high']:.1f}]")
        print(f"  Wilcoxon         = W {test['W']}, p {test['p']:.4f} "
              f"({_stars(test['p'])})")

    fig.tight_layout()
    _save(fig, save_path)

    return fig, table, {
        "n": len(table), "mean": boot["mean"],
        "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
        "W": test["W"], "p": test["p"],
    }