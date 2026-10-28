"""Trial selection, exclusion accounting, and descriptive analysis.

Two questions, in order. Which trials are usable, and what do the usable
trials look like.

One baseline, one delay variable
--------------------------------
Every corrected value in this module and in the hypothesis modules comes from
the same two choices, and neither is made here:

    the baseline     ``baseline=`` on facilitation_table and its callers.
                     ``expectancy_model`` fits tactile RT against delay and
                     reads the fitted value at each cell; ``delay_matched``
                     uses that cell's own tactile median; ``smooth`` fits per
                     speed separately and is a sensitivity analysis.

    the delay        ``h1.DELAY_COLUMN``, set once in the notebook. Both
                     ``expectancy_model`` and ``smooth`` read it, and
                     ``expectancy_model`` delegates the fit itself to
                     ``h1.expectancy_baseline``, so a QC figure and a
                     hypothesis result cannot describe different baselines.

Stage argument
--------------
Every summary and plot takes ``stage``:

    "before"   all T and VT trials that produced a reaction time, with no RT
               window, no interruption filter, and no dropped positions
    "after"    usable trials only
    "both"     draw or tabulate the two side by side (plots only)

"before" still requires a reaction time to exist, because a trial with no
response has no RT to plot. Responses never made appear in the exclusion
counts instead.

Group averages are always formed in two stages: summarise within each
participant, then average participants, so every participant counts equally
regardless of how many usable trials they contributed.

Typical use
-----------
    t_qc, pps = qc.prepare_analysis_tables(t, subjects=..., rt_min=..., rt_max=...)

    qc.plot_inspection(pps)          # look before deciding anything
    qc.coverage(pps)                 # trials per cell
    qc.thin_cells(pps)               # cells resting on few trials
    qc.exclusion_table(pps)          # what the rules removed, per subject
    qc.plot_trial_distribution(pps)  # RT distribution of every cell

    qc.plot_profile_legibility(pps, baseline="expectancy_model")
    qc.describe(pps, baseline="expectancy_model")

    qc.plot_subject_deepdive(pps, "P005")     # one participant, four views
    qc.flag_trials(pps)                       # candidates for manual review
    qc.exclude_trials(pps, candidates, note="why")

The descriptive sequence runs in a fixed order, because each step is a
precondition for reading the next:

    1. coverage        trials per cell, and which cells are too thin
    2. precision       the standard error of each cell estimate and of their
                       difference. This is the noise floor: no distance
                       structure smaller than it can be resolved
    3. shape           skew and mean-minus-median, the evidence for or
                       against the median as the cell estimator
    4. profiles        RT and facilitation across distance, split by speed
    5. decomposition   how much of the facilitation profile's shape comes
                       from the tactile baseline rather than from VT
"""

from pathlib import Path

import copy

from . import config

import matplotlib.colors as mcolors

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy import stats

from . import figures

from .config import N_BOOT, RANDOM_SEED   # or however config is reached

from .stats_utils import bootstrap_mean_ci

import style

from .config import DROP_POSITIONS, RT_MAX_MS, RT_MIN_MS

from . import expectancy as ex


def _display_name(subject):
    """Figure label for a participant. Unlisted ids pass through unchanged."""
    if hasattr(style, "display_name"):
        return style.display_name(subject)

    return {
        "PT001": "Patient",
        "PA002": "Age-matched control",
        "PA001": "Adult control",
    }.get(str(subject), str(subject))

def _display_names(subjects):
    """Vectorised form, for axis tick labels."""
    return [_display_name(s) for s in subjects]

TACTILE_CONDITIONS = ["T", "VT"]

CONDITION_LABELS = {"T": "Tactile-only", "VT": "Visuo-tactile"}

CONDITION_COLOUR = {"T": style.T, "VT": style.VT}

SPEEDS = ["slow", "fast"]

POSITIONS = range(1, 8)

NEAR = [1, 2]

FAR = [6, 7]

STAGES = ("before", "after")

STAGE_STYLE = {
    "before": {
        "label": "Before exclusions",
        "colour": style.CONTROL,
        "linestyle": "--",
        "marker": "s",
        "alpha": 0.75,
    },
    "after": {
        "label": "After exclusions",
        "colour": style.INK,
        "linestyle": "-",
        "marker": "o",
        "alpha": 1.0,
    },
}

MIN_CELL_N = 3

EXCLUDE_THIN_CELLS = None

MIN_CROSSOVER_AMPLITUDE_MS = 20.0

DISTANCE_M_BY_RANK = {
    1: 0.250, 2: 0.583, 3: 0.917, 4: 1.250,
    5: 1.583, 6: 1.917, 7: 2.250,
}

def rank_to_metres(rank):
    """Convert a continuous position rank to metres by linear interpolation."""
    ranks = np.asarray(sorted(DISTANCE_M_BY_RANK), dtype=float)
    metres = np.asarray([DISTANCE_M_BY_RANK[int(r)] for r in ranks], dtype=float)

    converted = np.interp(np.asarray(rank, dtype=float), ranks, metres)
    return float(converted) if np.ndim(rank) == 0 else converted

REASON_ORDER = [
    "usable",
    "no_response",
    "interrupted",
    "missing_rt",
    "too_fast",
    "too_slow",
    "dropped_position",
    "thin_cell",
]

N_BOOTSTRAP = 2000

BOOTSTRAP_SEED = 20250811

style.apply()

def _as_list(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return [value]
    return list(value)

def _select_subjects(data, subjects=None):
    """Restrict a trial table to the named participants.

    Accepts a single name or a list. Raises rather than silently returning an
    empty frame when a name is not present, because a typo in a participant
    code otherwise looks exactly like a participant with no usable trials.
    """
    if subjects is None:
        return data

    wanted = _as_list(subjects)
    present = set(data["subject"].dropna().unique())
    missing = [name for name in wanted if name not in present]

    if missing:
        raise KeyError(
            f"not in this table: {missing}. Present: {sorted(present)}"
        )

    return data[data["subject"].isin(wanted)]

def _drop_subjects(data, drop=None):
    """Remove named participants from a trial table.

    The mirror of ``_select_subjects``, and equally strict: a name that is not
    present is raised rather than ignored, because a typo in a participant code
    otherwise looks exactly like a participant who was already gone.
    """
    if drop is None:
        return data

    unwanted = _as_list(drop)
    present = set(data["subject"].dropna().unique())
    missing = [name for name in unwanted if name not in present]

    if missing:
        raise KeyError(
            f"not in this table: {missing}. Present: {sorted(present)}"
        )

    return data[~data["subject"].isin(unwanted)]

def _cell_keys(data):
    """The cell definition used throughout this module."""
    return [
        column
        for column in [
            "subject",
            "session",
            "speed",
            "sensory_condition",
            "position_rank",
        ]
        if column in data.columns
    ]

def _check_stage(stage, allow_both=True):
    allowed = set(STAGES) | ({"both"} if allow_both else set())

    if stage not in allowed:
        raise ValueError("stage must be one of: " + ", ".join(sorted(allowed)))

def _stages_to_draw(stage):
    _check_stage(stage)
    return list(STAGES) if stage == "both" else [stage]

def _ranks(pps):
    ranks = (
        pd.to_numeric(pps.get("position_rank"), errors="coerce")
        .dropna()
        .unique()
    )

    if len(ranks) == 0:
        return list(POSITIONS)

    return [int(rank) for rank in sorted(ranks)]

def _trial_order_column(data):
    return next(
        (
            column
            for column in ["global_trial", "trial_number", "trial", "trial_index"]
            if column in data.columns
        ),
        None,
    )

def _save(fig, save_path):
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")

def _distance_axis(ax, ranks=None):
    ticks = list(POSITIONS) if ranks is None else list(ranks)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"D{int(rank)}" for rank in ticks])
    ax.set_xlabel("Distance (D1 = closest)")

def _finish(ax, title, ylabel, ranks=None):
    _distance_axis(ax, ranks=ranks)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color=style._RULE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    style.clean(ax)

def _safe_correlation(a, b):
    """Pearson correlation returning NaN rather than raising on a flat vector."""
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])

def _rt_range(pps, xlim="robust", pad=0.03):
    """Choose a plotted RT range.

    "robust"      Tukey fence on the pre-exclusion data, widened to show the
                  RT window. A handful of corrupt trials otherwise stretch
                  the axis by an order of magnitude.
    "tight"       1st to 99th percentile of the usable data, ignoring the
                  window. Use when the quantity of interest is small
                  relative to the RT range.
    "full"        the complete observed range.
    (low, high)   an explicit range in ms.
    """
    if isinstance(xlim, (tuple, list)):
        return float(xlim[0]), float(xlim[1])

    values = trials_at_stage(pps, stage="before")["rt_ms"].dropna()

    if values.empty:
        return float(RT_MIN_MS), float(RT_MAX_MS)

    if xlim == "full":
        low, high = float(values.min()), float(values.max())

    elif xlim == "tight":
        kept = trials_at_stage(pps, stage="after")["rt_ms"].dropna()
        source = kept if not kept.empty else values
        low, high = np.percentile(source, [1, 99])
        low = max(float(low), 0.0)

    elif xlim == "robust":
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1

        # Asymmetric fence: RT is right-skewed, so a wide upper fence keeps
        # the slow tail that the RT window exists to cut.
        low = max(float(values.min()), q1 - 3 * iqr)
        high = min(float(values.max()), q3 + 6 * iqr)

        low = min(low, float(RT_MIN_MS))

        if (values > RT_MAX_MS).any():
            high = max(high, float(RT_MAX_MS))

        # RT cannot be negative; anything below zero is a data fault.
        low = max(low, min(0.0, float(RT_MIN_MS)))

    else:
        raise ValueError(
            "xlim must be 'robust', 'tight', 'full', or a (low, high) pair"
        )

    span = high - low if high > low else 1.0
    return low - pad * span, high + pad * span

def select_pps(t, subjects=None, sessions=None):
    """Return the requested PPS trials from the Tables object."""
    pps = t.pps_trials.copy()

    subjects = _as_list(subjects)
    sessions = _as_list(sessions)

    if subjects is not None:
        pps = pps[pps["subject"].isin(subjects)]

    if sessions is not None:
        pps = pps[pps["session"].isin(sessions)]

    if pps.empty:
        print("No PPS trials matched this selection.")

    return pps

def repair_rt(pps):
    """Repair old files where rt_ms holds an absolute timestamp.

    The preferred correction is response_time_ms - vibrotactile_onset_ms.
    """
    pps = pps.copy()

    for column in [
        "rt_ms",
        "reaction_time_ms",
        "response_time_ms",
        "vibrotactile_onset_ms",
    ]:
        if column in pps:
            pps[column] = pd.to_numeric(pps[column], errors="coerce")

    if "rt_ms" not in pps:
        pps["rt_ms"] = np.nan

    bad_rt = pps["rt_ms"].isna() | (pps["rt_ms"].abs() > 10_000)
    tactile = pps["sensory_condition"].isin(TACTILE_CONDITIONS)

    if {"response_time_ms", "vibrotactile_onset_ms"}.issubset(pps.columns):
        reconstructed = pps["response_time_ms"] - pps["vibrotactile_onset_ms"]
        use = tactile & bad_rt & reconstructed.between(-5_000, 5_000)
        pps.loc[use, "rt_ms"] = reconstructed[use]
        print(f"Repaired {use.sum()} RTs from timestamps.")

    if "reaction_time_ms" in pps:
        use = (
            tactile
            & pps["rt_ms"].isna()
            & pps["reaction_time_ms"].between(-5_000, 5_000)
        )
        pps.loc[use, "rt_ms"] = pps.loc[use, "reaction_time_ms"]
        print(f"Repaired {use.sum()} RTs from reaction_time_ms.")

    pps.loc[pps["sensory_condition"] == "V", "rt_ms"] = np.nan
    return pps

def mark_usable(
    pps,
    rt_min=RT_MIN_MS,
    rt_max=RT_MAX_MS,
    drop_positions=DROP_POSITIONS,
    min_cell_n=EXCLUDE_THIN_CELLS,
):
    """Add ``usable`` and ``exclusion_reason`` columns without dropping rows.

    A trial is usable when it is a tactile trial, a response was made, the
    trial was not interrupted, the RT falls inside the absolute window, and
    the position is not excluded by configuration. These are the
    preregistered rules and, by default, nothing else is applied.

    ``min_cell_n`` is off by default and should stay off for the primary
    analysis. Setting it marks every trial in a cell holding fewer than that
    many survivors as ``thin_cell``, which is a real exclusion of real data
    and belongs in a sensitivity analysis rather than in the main path. Thin
    cells are reported either way, by ``thin_cells`` and ``coverage``.

    This function only marks. No row is removed here; downstream functions
    select ``usable == True`` when ``stage="after"``.
    """
    pps = pps.copy()
    drop_positions = [] if drop_positions is None else list(drop_positions)

    tactile = pps["sensory_condition"].isin(TACTILE_CONDITIONS)
    valid_rt = pps["rt_ms"].between(rt_min, rt_max, inclusive="neither")
    dropped = pps["position_rank"].isin(drop_positions)

    if "response_made" in pps:
        responded = pps["response_made"].fillna(False).astype(bool)
    else:
        responded = pps["rt_ms"].notna()

    if "trial_interrupted" in pps:
        uninterrupted = (
            pd.to_numeric(pps["trial_interrupted"], errors="coerce")
            .fillna(0)
            .eq(0)
        )
    else:
        uninterrupted = pd.Series(True, index=pps.index)

    preliminary = tactile & responded & uninterrupted & valid_rt & ~dropped

    # Stage 2. Thin cells, evaluated on the trials that survived stage 1.
    thin = pd.Series(False, index=pps.index)

    if min_cell_n:
        keys = _cell_keys(pps)

        if keys:
            eligible = pps.loc[preliminary]

            if not eligible.empty:
                sizes = eligible.groupby(keys, dropna=False)["rt_ms"].transform("size")
                thin.loc[eligible.index] = sizes < min_cell_n

    pps["usable"] = preliminary & ~thin
    pps["exclusion_reason"] = "usable"

    pps.loc[~tactile, "exclusion_reason"] = "visual_only"
    pps.loc[tactile & ~responded, "exclusion_reason"] = "no_response"
    pps.loc[tactile & responded & ~uninterrupted, "exclusion_reason"] = "interrupted"
    pps.loc[
        tactile & responded & uninterrupted & pps["rt_ms"].isna(),
        "exclusion_reason",
    ] = "missing_rt"
    pps.loc[
        tactile & responded & uninterrupted & (pps["rt_ms"] <= rt_min),
        "exclusion_reason",
    ] = "too_fast"
    pps.loc[
        tactile & responded & uninterrupted & (pps["rt_ms"] >= rt_max),
        "exclusion_reason",
    ] = "too_slow"
    pps.loc[
        tactile & responded & uninterrupted & valid_rt & dropped,
        "exclusion_reason",
    ] = "dropped_position"
    pps.loc[thin, "exclusion_reason"] = "thin_cell"

    return pps

def prepare(
    t,
    subjects=None,
    sessions=None,
    repair=False,
    rt_min=RT_MIN_MS,
    rt_max=RT_MAX_MS,
    drop_positions=DROP_POSITIONS,
    min_cell_n=EXCLUDE_THIN_CELLS,
):
    """Select and mark PPS trials once, at the top of the notebook.

    No row is ever dropped: exclusions are represented by ``usable`` and
    ``exclusion_reason``, so before and after can always be compared.
    """
    pps = select_pps(t, subjects=subjects, sessions=sessions)

    if repair:
        pps = repair_rt(pps)

    return mark_usable(
        pps,
        rt_min=rt_min,
        rt_max=rt_max,
        drop_positions=drop_positions,
        min_cell_n=min_cell_n,
    )

def prepare_analysis_tables(t, **kwargs):
    """Return an analysis-ready copy of ``t`` plus the fully marked table.

    ``t`` is never modified. In the returned copy, ``pps_trials`` holds only
    usable trials. The second return value keeps every row and all QC
    diagnostics, and is what every function in this module expects.
    """
    pps_marked = prepare(t, **kwargs)

    t_clean = copy.deepcopy(t)
    t_clean.pps_trials = pps_marked.loc[pps_marked["usable"]].copy()

    return t_clean, pps_marked

def exclusion_rules(
    rt_min=RT_MIN_MS,
    rt_max=RT_MAX_MS,
    drop_positions=DROP_POSITIONS,
    min_cell_n=EXCLUDE_THIN_CELLS,
):
    """The exclusion rules in words, for display before any filtering."""
    drop_positions = [] if drop_positions is None else list(drop_positions)
    dropped_text = (
        ", ".join(f"D{int(rank)}" for rank in drop_positions)
        if drop_positions
        else "None"
    )

    return pd.DataFrame([
        {
            "reason": "visual_only",
            "criterion": "sensory_condition == 'V'",
            "why": "Visual-only trials have no tactile RT.",
        },
        {
            "reason": "no_response",
            "criterion": "response_made == False",
            "why": "No reaction time exists when no response was made.",
        },
        {
            "reason": "interrupted",
            "criterion": "trial_interrupted != 0",
            "why": "The trial did not complete under the intended conditions.",
        },
        {
            "reason": "missing_rt",
            "criterion": "rt_ms is missing",
            "why": "The dependent variable cannot be analysed without an RT.",
        },
        {
            "reason": "too_fast",
            "criterion": f"rt_ms <= {rt_min:g} ms",
            "why": "At or below the preregistered lower bound; treated as anticipatory.",
        },
        {
            "reason": "too_slow",
            "criterion": f"rt_ms >= {rt_max:g} ms",
            "why": "At or above the preregistered upper bound; treated as excessively delayed.",
        },
        {
            "reason": "dropped_position",
            "criterion": f"position_rank in {dropped_text}",
            "why": "Excluded by analysis configuration, not by observed RT.",
        },
        {
            "reason": "thin_cell",
            "criterion": (
                f"fewer than {min_cell_n} surviving trials in the cell"
                if min_cell_n else "not applied"
            ),
            "why": (
                "Off by default. Thin cells are reported by thin_cells and "
                "coverage and given a larger standard error by "
                "cell_precision, rather than being excluded."
            ),
        },
    ])

def trials_at_stage(pps, stage="after", subjects=None):
    """T and VT trials with a reaction time, before or after exclusions.

    ``subjects`` restricts the result to the named participants, which is how
    a patient and their matched control are pulled out for comparison.
    """
    _check_stage(stage, allow_both=False)

    data = pps[
        pps["sensory_condition"].isin(TACTILE_CONDITIONS) & pps["rt_ms"].notna()
    ]

    if stage == "after":
        data = data[data["usable"]]

    return _select_subjects(data.copy(), subjects)

def usable_trials(pps):
    """Usable T and VT trials only."""
    return trials_at_stage(pps, stage="after")

def exclusion_table(pps, per_subject=True):
    """Trials by exclusion reason, per subject or overall.

    Visual-only trials are dropped first: they can never be usable, so
    including them would make the retention percentage meaningless.
    """
    tactile = pps[pps["sensory_condition"].isin(TACTILE_CONDITIONS)]

    if per_subject:
        counts = (
            pd.crosstab(tactile["subject"], tactile["exclusion_reason"])
            .reindex(columns=REASON_ORDER, fill_value=0)
            .reset_index()
        )
    else:
        counts = (
            tactile["exclusion_reason"]
            .value_counts()
            .reindex(REASON_ORDER, fill_value=0)
            .to_frame()
            .T
            .reset_index(drop=True)
        )

    reasons = [column for column in REASON_ORDER if column in counts.columns]

    counts["n_tactile"] = counts[reasons].sum(axis=1)
    counts["n_usable"] = counts["usable"]
    counts["percent_kept"] = (
        100 * counts["n_usable"] / counts["n_tactile"]
    ).round(1)

    return counts

def exclusion_impact(pps, statistic="median"):
    """What the exclusions actually changed, per subject.

    Reports trials lost, the RT shift in each condition, and the near-minus-far
    facilitation before and after. A large ``nearfar_change`` in a subject who
    lost few trials is the case worth inspecting: a small number of trials was
    carrying the effect.
    """
    impact = exclusion_table(pps)[
        ["subject", "n_tactile", "n_usable", "percent_kept"]
    ]

    frames = [impact]

    for stage in STAGES:
        data = trials_at_stage(pps, stage=stage)

        rt = (
            data
            .groupby(["subject", "sensory_condition"], as_index=False)["rt_ms"]
            .agg(statistic)
            .pivot(index="subject", columns="sensory_condition", values="rt_ms")
            .rename(columns={
                condition: f"rt_{condition}_{stage}"
                for condition in TACTILE_CONDITIONS
            })
            .reset_index()
        )

        near_far = (
            nearfar(pps, statistic=statistic, stage=stage, by_speed=False)
            [["subject", "nearfar_pps"]]
            .rename(columns={"nearfar_pps": f"nearfar_{stage}"})
        )

        frames.extend([rt, near_far])

    out = frames[0]

    for frame in frames[1:]:
        out = out.merge(frame, on="subject", how="outer")

    for condition in TACTILE_CONDITIONS:
        before, after = f"rt_{condition}_before", f"rt_{condition}_after"
        if before in out and after in out:
            out[f"rt_{condition}_change"] = out[after] - out[before]

    out["nearfar_change"] = out["nearfar_after"] - out["nearfar_before"]

    return out.round(1)

def cell_counts(pps, stage="after"):
    """Usable trial count in every cell."""
    data = trials_at_stage(pps, stage=stage)
    keys = _cell_keys(data)

    return (
        data
        .groupby(keys, as_index=False)["rt_ms"]
        .size()
        .rename(columns={"size": "n_trials"})
        .sort_values(keys)
        .reset_index(drop=True)
    )

def thin_cells(pps, minimum=MIN_CELL_N, stage="after"):
    """Cells left with fewer than ``minimum`` usable trials.

    Worth checking after any change to the exclusion rules: the tactile-only
    baseline is the precision-limiting term, so a cell that drops to one or
    two trials makes its facilitation value unreliable.
    """
    counts = cell_counts(pps, stage=stage)
    return (
        counts[counts["n_trials"] < minimum]
        .sort_values(["subject", "sensory_condition", "position_rank"])
        .reset_index(drop=True)
    )

def rt_profile(pps, statistic="median", stage="after", by_speed=True):
    """Subject-level and group-level RT profiles across distance."""
    data = trials_at_stage(pps, stage=stage)

    keys = (
        ["subject"]
        + (["speed"] if by_speed else [])
        + ["sensory_condition", "position_rank"]
    )

    subject = (
        data
        .groupby(keys, as_index=False)["rt_ms"]
        .agg(statistic)
        .sort_values(keys)
        .reset_index(drop=True)
    )

    group = (
        subject
        .groupby(keys[1:], as_index=False)["rt_ms"]
        .agg(group_mean="mean", group_sd="std", n="count")
    )
    group["group_sem"] = group["group_sd"] / np.sqrt(group["n"])

    return subject, group.sort_values(keys[1:]).reset_index(drop=True)

def smooth_baseline(pps, stage="after", degree=1, delay_column=None):
    """Tactile-only RT as a smooth function of delay, fitted per speed context.

    The preregistered baseline estimates the tactile-only cell separately at
    each distance, which at four trials per cell carries more sampling noise
    than the facilitation effect it is subtracted from. This keeps the
    delay-dependence that Hobeika and Kandula establish, but stops estimating
    it seven times over: within each subject, session and speed context, all
    tactile-only trials are fitted against delay with a polynomial of the
    given degree, and the fitted value replaces the cell estimate.

    Two parameters instead of seven independent medians, fitted on roughly 28
    trials instead of 4. Every trial contributes, including those in cells too
    thin to summarise on their own, so no cell has to be dropped.

    Speed contexts stay separate, because a tactile-only cell is not
    interchangeable across them even where the delays coincide numerically.

    Returns one row per subject, session, speed and position with the fitted
    baseline, the trials behind the fit, and its residual SD.
    """
    data = trials_at_stage(pps, stage=stage)
    data = data[data["sensory_condition"] == "T"]

    if data.empty:
        return pd.DataFrame()

    from .hypotheses import h1

    # One delay definition for the whole project. See h1.DELAY_COLUMN.
    delay = h1.DELAY_COLUMN if delay_column is None else delay_column

    keys = [
        column
        for column in ["subject", "session", "speed"]
        if column in data.columns
    ]

    rows = []

    for key, one in data.groupby(keys, dropna=False):
        x = pd.to_numeric(one[delay], errors="coerce").to_numpy(dtype=float)
        y = one["rt_ms"].to_numpy(dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]

        distinct = len(np.unique(x))

        if distinct < 2:
            continue

        # Never fit more freely than the delays can support.
        order = int(min(degree, distinct - 1))
        coefficients = np.polyfit(x, y, order)
        fitted = np.polyval(coefficients, x)

        # Standard error of the fitted value, which is what replaces the
        # standard error of the cell estimate once the baseline is modelled.
        design = np.vander(x, order + 1)
        residual_df = len(y) - (order + 1)

        if residual_df > 0:
            residual_sd = float(
                np.sqrt(np.sum((y - fitted) ** 2) / residual_df)
            )
            try:
                covariance = np.linalg.inv(design.T @ design)
            except np.linalg.LinAlgError:
                covariance = np.linalg.pinv(design.T @ design)
        else:
            residual_sd = np.nan
            covariance = None

        key_tuple = key if isinstance(key, tuple) else (key,)

        for rank in sorted(one["position_rank"].dropna().unique()):
            at_rank = one[one["position_rank"] == rank]
            delay_value = float(
                pd.to_numeric(at_rank[delay], errors="coerce").mean()
            )

            if covariance is None or not np.isfinite(residual_sd):
                baseline_se = np.nan
            else:
                point = np.vander(np.array([delay_value]), order + 1)[0]
                baseline_se = float(
                    residual_sd * np.sqrt(point @ covariance @ point)
                )

            record = dict(zip(keys, key_tuple))
            record.update({
                "position_rank": rank,
                "baseline_smooth_ms": float(np.polyval(coefficients, delay_value)),
                "baseline_se_ms": baseline_se,
                "n_trials_fit": int(len(y)),
                "n_trials_cell": int(at_rank["rt_ms"].notna().sum()),
                "fit_degree": order,
                "fit_residual_sd_ms": residual_sd,
                "delay_column": delay,
            })
            rows.append(record)

    return pd.DataFrame(rows)


def expectancy_model(
    pps,
    stage="after",
    delay_column=None,
    interaction=True,
    degree=1,
):
    """Tactile-only RT as a function of delay, fitted per participant.

    A thin wrapper around ``expectancy.expectancy_baseline``, which is where
    the fit lives. The QC reports and the hypothesis results therefore
    describe the same baseline, fitted against the same variable.
    """
    from . import expectancy as ex

    trials = trials_at_stage(pps, stage=stage)

    if trials.empty:
        return pd.DataFrame(), pd.DataFrame()

    fitted, coefficients = ex.expectancy_baseline(
        trials,
        degree=degree,
        interaction=interaction,
        delay_column=delay_column,
    )

    if fitted.empty:
        return fitted, coefficients

    return (
        fitted.rename(columns={"tactile_baseline_ms": "baseline_model_ms"}),
        coefficients,
    )




def facilitation_table(
    pps,
    statistic="median",
    stage="after",
    baseline_statistic=None,
    degree=1,
    delay_column=None,
):
    """T minus VT at every subject, session, speed and position.

    Two baselines are carried side by side, so they can be reported against
    each other rather than chosen blind:

        facilitation_ms         T(distance) - VT(distance), the preregistered
                                delay-matched correction. Each VT cell is
                                referenced to the tactile-only cell at the
                                same position and therefore the same delay.
        facilitation_smooth_ms  the same, with the baseline read off a fit of
                                tactile-only RT against delay rather than
                                from that one cell. See ``smooth_baseline``.

    ``baseline_statistic`` summarises the tactile cells separately from the
    visuo-tactile ones. At four trials the median is the noisier of the two
    estimators, so ``baseline_statistic="mean"`` with ``statistic="median"``
    is a defensible pairing and is named in the preregistration as a
    sensitivity analysis. Leave it None to use the same estimator for both.

    Positive values mean VT was faster than T.
    """
    data = trials_at_stage(pps, stage=stage)
    baseline_statistic = statistic if baseline_statistic is None else baseline_statistic

    keys = ["subject", "session", "speed", "position_rank", "sensory_condition"]

    summary = (
        data
        .groupby(keys)["rt_ms"]
        .agg(statistic)
        .unstack("sensory_condition")
        .reset_index()
    )

    if baseline_statistic != statistic:
        tactile = (
            data[data["sensory_condition"] == "T"]
            .groupby(keys[:-1])["rt_ms"]
            .agg(baseline_statistic)
            .rename("T_alt")
            .reset_index()
        )
        summary = summary.merge(tactile, on=keys[:-1], how="left")
        summary["T"] = summary["T_alt"]
        summary = summary.drop(columns="T_alt")

    for condition in TACTILE_CONDITIONS:
        if condition not in summary:
            summary[condition] = np.nan

    summary["facilitation_ms"] = summary["T"] - summary["VT"]

    smooth = smooth_baseline(
        pps, stage=stage, degree=degree, delay_column=delay_column
    )

    if smooth.empty:
        summary["baseline_smooth_ms"] = np.nan
        summary["facilitation_smooth_ms"] = np.nan
        return summary

    merge_on = [
        column
        for column in ["subject", "session", "speed", "position_rank"]
        if column in summary.columns and column in smooth.columns
    ]

    summary = summary.merge(
        smooth[
            merge_on
            + ["baseline_smooth_ms", "baseline_se_ms", "n_trials_fit", "fit_degree"]
        ],
        on=merge_on,
        how="left",
    )

    summary["facilitation_smooth_ms"] = (
        summary["baseline_smooth_ms"] - summary["VT"]
    )

    # The expectancy model: one function of delay per participant, fitted on
    # all their tactile trials at once, evaluated at each matched delay.
    modelled, _ = expectancy_model(
        pps, stage=stage, delay_column=delay_column, degree=degree
    )

    if modelled.empty:
        summary["baseline_model_ms"] = np.nan
        summary["facilitation_model_ms"] = np.nan
        return summary

    model_keys = [
        column
        for column in ["subject", "speed", "position_rank"]
        if column in summary.columns and column in modelled.columns
    ]

    summary = summary.merge(
        modelled[model_keys + ["baseline_model_ms", "baseline_se_ms"]].rename(
            columns={"baseline_se_ms": "baseline_model_se_ms"}
        ),
        on=model_keys,
        how="left",
    )

    summary["facilitation_model_ms"] = (
        summary["baseline_model_ms"] - summary["VT"]
    )

    return summary

BASELINES = {
    # The preregistered correction: the tactile median at that exact
    # distance and speed, from its own four trials.
    "delay_matched": "facilitation_ms",
    # The same delay-dependence estimated as a function rather than as
    # independent cell means, fitted per participant and speed.
    "smooth": "facilitation_smooth_ms",
    # As "smooth", but fitted across both speeds at once with a
    # delay-by-speed term, so the expectancy function is estimated from every
    # tactile trial the participant produced.
    "expectancy_model": "facilitation_model_ms",
}

def _facilitation_column(baseline):
    if baseline not in BASELINES:
        raise ValueError(
            "baseline must be one of: " + ", ".join(sorted(BASELINES))
        )
    return BASELINES[baseline]

def facilitation_profile(
    pps,
    statistic="median",
    stage="after",
    by_speed=True,
    baseline="delay_matched",
    **kwargs,
):
    """Subject-level and group-level facilitation profiles.

    ``baseline="delay_matched"`` uses the preregistered per-cell correction,
    ``baseline="smooth"`` the fitted one. ``baseline_statistic`` passes
    through to ``facilitation_table``. Either way the returned column is
    named ``facilitation_ms``, so everything downstream is unchanged.
    """
    column = _facilitation_column(baseline)
    table = facilitation_table(pps, statistic=statistic, stage=stage, **kwargs)

    keys = ["subject"] + (["speed"] if by_speed else []) + ["position_rank"]

    subject = (
        table
        .dropna(subset=[column])
        .groupby(keys, as_index=False)[column]
        .mean()
        .rename(columns={column: "facilitation_ms"})
        .sort_values(keys)
        .reset_index(drop=True)
    )

    group = (
        subject
        .groupby(keys[1:], as_index=False)["facilitation_ms"]
        .agg(group_mean="mean", group_sd="std", n="count")
    )
    group["group_sem"] = group["group_sd"] / np.sqrt(group["n"])

    return subject, group.sort_values(keys[1:]).reset_index(drop=True)

def nearfar(
    pps, statistic="median", stage="after", by_speed=True,
    baseline="delay_matched",
):
    """Near-minus-far facilitation per subject.

    ``by_speed=True`` keeps the speed contexts separate, which is the form
    the hypotheses are stated in. ``by_speed=False`` collapses them.
    """
    subject, _ = facilitation_profile(
        pps, statistic=statistic, stage=stage, by_speed=by_speed,
        baseline=baseline,
    )

    keys = ["subject"] + (["speed"] if by_speed else [])
    rows = []

    for key, one in subject.groupby(keys):
        key_tuple = key if isinstance(key, tuple) else (key,)

        record = dict(zip(keys, key_tuple))
        record.update({
            "near_facilitation": one.loc[
                one["position_rank"].isin(NEAR), "facilitation_ms"
            ].mean(),
            "far_facilitation": one.loc[
                one["position_rank"].isin(FAR), "facilitation_ms"
            ].mean(),
        })
        record["nearfar_pps"] = (
            record["near_facilitation"] - record["far_facilitation"]
        )
        rows.append(record)

    return pd.DataFrame(rows).round(1)

def coverage(pps, stage="after", minimum=MIN_CELL_N):
    """Step 1. How well each condition is sampled, per condition and speed."""
    counts = cell_counts(pps, stage=stage)

    if counts.empty:
        return pd.DataFrame()

    expected = counts["subject"].nunique() * counts["position_rank"].nunique()

    summary = (
        counts
        .groupby(["sensory_condition", "speed"], as_index=False)
        .agg(
            cells_present=("n_trials", "size"),
            trials_total=("n_trials", "sum"),
            median_n=("n_trials", "median"),
            minimum_n=("n_trials", "min"),
            maximum_n=("n_trials", "max"),
        )
    )

    thin = (
        counts[counts["n_trials"] < minimum]
        .groupby(["sensory_condition", "speed"], as_index=False)
        .size()
        .rename(columns={"size": "cells_thin"})
    )

    summary = summary.merge(thin, on=["sensory_condition", "speed"], how="left")
    summary["cells_thin"] = summary["cells_thin"].fillna(0).astype(int)
    summary["cells_expected"] = expected
    summary["cells_missing"] = expected - summary["cells_present"]

    ordered = [
        "sensory_condition", "speed",
        "cells_expected", "cells_present", "cells_missing", "cells_thin",
        "trials_total", "median_n", "minimum_n", "maximum_n",
    ]
    return (
        summary[ordered]
        .sort_values(["sensory_condition", "speed"])
        .reset_index(drop=True)
    )

def missing_cells(pps, stage="after"):
    """Cells the design expected that hold no usable trial at all."""
    counts = cell_counts(pps, stage=stage)

    if counts.empty:
        return pd.DataFrame()

    index = pd.MultiIndex.from_product(
        [
            sorted(counts["subject"].unique()),
            sorted(counts["speed"].dropna().unique()),
            TACTILE_CONDITIONS,
            sorted(counts["position_rank"].unique()),
        ],
        names=["subject", "speed", "sensory_condition", "position_rank"],
    )

    present = counts.set_index(
        ["subject", "speed", "sensory_condition", "position_rank"]
    ).index

    return (
        index.difference(present)
        .to_frame(index=False)
        .sort_values(["subject", "sensory_condition", "speed", "position_rank"])
        .reset_index(drop=True)
    )

def _estimator_se(values, statistic, rng, n_boot=N_BOOTSTRAP):
    """Standard error of the cell estimator.

    The mean has a closed form. The median does not, and its sampling
    variability at n = 4 is exactly the quantity in question, so it is
    bootstrapped rather than approximated.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size

    if n < 2:
        return np.nan

    if statistic == "mean":
        return float(values.std(ddof=1) / np.sqrt(n))

    draws = rng.choice(values, size=(n_boot, n), replace=True)
    return float(np.median(draws, axis=1).std(ddof=1))

def cell_precision(pps, statistic="median", stage="after", n_boot=N_BOOTSTRAP):
    """Estimate, SD, and standard error of every cell. One row per cell.

    This replaces a plain per-cell descriptive table: the standard error is
    what determines whether a cell estimate can be read at all.
    """
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")

    data = trials_at_stage(pps, stage=stage)
    keys = _cell_keys(data)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    rows = []

    for key, cell in data.groupby(keys, dropna=False):
        values = cell["rt_ms"].to_numpy(dtype=float)
        key_tuple = key if isinstance(key, tuple) else (key,)

        record = dict(zip(keys, key_tuple))
        record.update({
            "n_trials": int(np.isfinite(values).sum()),
            "estimate_ms": float(
                np.median(values) if statistic == "median" else np.mean(values)
            ),
            "sd_ms": float(np.nanstd(values, ddof=1)) if values.size > 1 else np.nan,
            "se_ms": _estimator_se(values, statistic, rng, n_boot=n_boot),
        })
        rows.append(record)

    return pd.DataFrame(rows)

def precision(
    pps, statistic="median", stage="after", n_boot=N_BOOTSTRAP,
    baseline="delay_matched",
):
    """Step 2. The noise floor of the facilitation measure, per speed and distance.

    ``se_facilitation_ms`` is the square root of se_T squared plus se_VT
    squared, the two terms being independent samples. Under
    ``baseline="smooth"`` the tactile term is the standard error of the
    fitted value rather than of the cell, which is the whole point of
    modelling it. It is the smallest
    distance-dependent change a participant's facilitation curve could be
    expected to resolve. Compare it against ``group_facilitation_range_ms``,
    the peak-to-trough range of the group profile at that speed: where the
    floor is the larger of the two, the visible shape of the curve is not
    evidence of distance structure.
    """
    cells = cell_precision(pps, statistic=statistic, stage=stage, n_boot=n_boot)

    if cells.empty:
        return pd.DataFrame()

    by_cell = (
        cells
        .groupby(["speed", "position_rank", "sensory_condition"], as_index=False)
        .agg(mean_n=("n_trials", "mean"), mean_se_ms=("se_ms", "mean"))
    )

    wide = by_cell.pivot(
        index=["speed", "position_rank"],
        columns="sensory_condition",
        values=["mean_n", "mean_se_ms"],
    )
    wide.columns = [f"{metric}_{condition}" for metric, condition in wide.columns]
    wide = wide.reset_index()

    for condition in TACTILE_CONDITIONS:
        for metric in ["mean_n", "mean_se_ms"]:
            if f"{metric}_{condition}" not in wide:
                wide[f"{metric}_{condition}"] = np.nan

    if baseline in ("smooth", "expectancy_model"):
        if baseline == "smooth":
            fitted = smooth_baseline(pps, stage=stage)
        else:
            fitted, _ = expectancy_model(pps, stage=stage)

        if not fitted.empty and "baseline_se_ms" in fitted:
            fitted_se = (
                fitted
                .groupby(["speed", "position_rank"], as_index=False)["baseline_se_ms"]
                .mean()
                .rename(columns={"baseline_se_ms": "mean_se_ms_T"})
            )
            wide = wide.drop(columns="mean_se_ms_T").merge(
                fitted_se, on=["speed", "position_rank"], how="left"
            )

    wide["se_facilitation_ms"] = np.sqrt(
        wide["mean_se_ms_T"] ** 2 + wide["mean_se_ms_VT"] ** 2
    )

    _, group = facilitation_profile(
        pps, statistic=statistic, stage=stage, by_speed=True, baseline=baseline,
    )

    if not group.empty:
        spread = (
            group
            .groupby("speed", as_index=False)["group_mean"]
            .agg(group_facilitation_range_ms=lambda v: float(np.ptp(v)))
        )
        wide = wide.merge(spread, on="speed", how="left")

        wide = wide.merge(
            group[["speed", "position_rank", "group_mean", "group_sem"]].rename(
                columns={
                    "group_mean": "group_facilitation_ms",
                    "group_sem": "group_sem_ms",
                }
            ),
            on=["speed", "position_rank"],
            how="left",
        )

    ordered = [
        "speed", "position_rank",
        "mean_n_T", "mean_n_VT",
        "mean_se_ms_T", "mean_se_ms_VT", "se_facilitation_ms",
        "group_facilitation_ms", "group_sem_ms",
        "group_facilitation_range_ms",
    ]
    ordered = [column for column in ordered if column in wide]

    return (
        wide[ordered]
        .sort_values(["speed", "position_rank"])
        .reset_index(drop=True)
        .round(1)
    )

def shape(pps, by="sensory_condition", stage="after"):
    """Step 3. Skew and mean-minus-median, the evidence behind the estimator.

    A positive ``mean_minus_median`` of the order of tens of milliseconds is
    the right skew that motivates the median. Where it is small, the mean is
    the better estimator in thin cells, in which the median is markedly
    noisier.
    """
    data = trials_at_stage(pps, stage=stage)
    by = [by] if isinstance(by, str) else (list(by) if by else [])

    def summarise(values):
        values = pd.to_numeric(values, errors="coerce").dropna()

        if values.empty:
            return pd.Series(dtype=float)

        q1, q3 = values.quantile(0.25), values.quantile(0.75)

        return pd.Series({
            "n": values.size,
            "mean": values.mean(),
            "median": values.median(),
            "sd": values.std(ddof=1),
            "iqr": q3 - q1,
            "skew": values.skew(),
            "mean_minus_median": values.mean() - values.median(),
        })

    if not by:
        return summarise(data["rt_ms"]).to_frame().T.round(1)

    return (
        data
        .groupby(by, dropna=False)["rt_ms"]
        .apply(summarise)
        .unstack()
        .reset_index()
        .round(1)
    )

def profiles(pps, statistic="median", stage="after", baseline="delay_matched"):
    """Step 4. Group RT and facilitation across distance, split by speed.

    One long table with a row per speed, distance and quantity, so RT and
    facilitation can be read against each other without reconciling two
    shapes. Speeds stay separate: a step present at one speed and absent at
    the other is diluted by averaging them.
    """
    _, rt_group = rt_profile(pps, statistic=statistic, stage=stage, by_speed=True)
    _, fac_group = facilitation_profile(
        pps, statistic=statistic, stage=stage, by_speed=True, baseline=baseline,
    )

    rt_group = rt_group.rename(columns={"sensory_condition": "quantity"})
    fac_group = fac_group.copy()
    fac_group["quantity"] = "facilitation"

    columns = [
        "speed", "quantity", "position_rank",
        "group_mean", "group_sd", "n", "group_sem",
    ]

    return (
        pd.concat(
            [rt_group.reindex(columns=columns), fac_group.reindex(columns=columns)],
            ignore_index=True,
        )
        .sort_values(["quantity", "speed", "position_rank"])
        .reset_index(drop=True)
        .round(1)
    )

def describe(
    pps, statistic="median", stage="after", n_boot=N_BOOTSTRAP,
    baseline="delay_matched",
):
    """Run the descriptive sequence and return every table, in order.

    ``baseline="smooth"`` runs the whole sequence against the fitted
    baseline instead of the per-cell one, so the two can be compared step by
    step rather than only at the final profile.
    """
    _, decomposition = baseline_contribution(
        pps, statistic=statistic, stage=stage, by_speed=True, baseline=baseline,
    )

    return {
        "coverage": coverage(pps, stage=stage),
        "precision": precision(
            pps, statistic=statistic, stage=stage, n_boot=n_boot,
            baseline=baseline,
        ),
        "shape": shape(pps, by="sensory_condition", stage=stage),
        "profiles": profiles(
            pps, statistic=statistic, stage=stage, baseline=baseline
        ),
        "decomposition": decomposition,
        "missing_cells": missing_cells(pps, stage=stage),
        "nearfar": nearfar(
            pps, statistic=statistic, stage=stage, by_speed=True,
            baseline=baseline,
        ),
        "settings": {
            "statistic": statistic, "stage": stage,
            "n_boot": n_boot, "baseline": baseline,
        },
    }

def print_report(report):
    """Compact console summary of the six steps, in order."""
    settings = report.get("settings", {})
    print(
        f"Descriptive report  (statistic = {settings.get('statistic')}, "
        f"stage = {settings.get('stage')}, "
        f"baseline = {settings.get('baseline', 'delay_matched')})"
    )

    table = report.get("coverage")
    if table is not None and not table.empty:
        print("\n1. Coverage")
        for _, row in table.iterrows():
            print(
                f"   {row['sensory_condition']:>2} {row['speed']:<5} "
                f"median n = {row['median_n']:.0f}, "
                f"min = {row['minimum_n']:.0f}, "
                f"thin cells = {row['cells_thin']}, "
                f"missing = {row['cells_missing']}"
            )

    table = report.get("precision")
    if table is not None and not table.empty:
        print("\n2. Precision (the noise floor)")
        for speed, one in table.groupby("speed"):
            floor = one["se_facilitation_ms"].median()
            span = one.get("group_facilitation_range_ms")
            span = float(span.iloc[0]) if span is not None and len(span) else np.nan
            verdict = (
                "below the floor" if np.isfinite(span) and span < floor
                else "above the floor"
            )
            print(
                f"   {speed:<5} median SE of facilitation = {floor:.1f} ms; "
                f"group profile range = {span:.1f} ms ({verdict})"
            )

    table = report.get("shape")
    if table is not None and not table.empty:
        print("\n3. Distribution shape")
        for _, row in table.iterrows():
            print(
                f"   {row['sensory_condition']:>2} n = {row['n']:.0f}, "
                f"skew = {row['skew']:.2f}, "
                f"mean - median = {row['mean_minus_median']:.1f} ms"
            )

    table = report.get("profiles")
    if table is not None and not table.empty:
        print("\n4. Profiles across distance")
        for (quantity, speed), one in table.groupby(["quantity", "speed"]):
            one = one.sort_values("position_rank")
            peak = one.loc[one["group_mean"].idxmax(), "position_rank"]
            trough = one.loc[one["group_mean"].idxmin(), "position_rank"]
            print(
                f"   {quantity:<12} {speed:<5} range = "
                f"{float(np.ptp(one['group_mean'])):.1f} ms "
                f"(lowest at D{int(trough)}, highest at D{int(peak)})"
            )

    table = report.get("decomposition")
    if table is not None and not table.empty:
        print("\n5. Share of the facilitation profile's shape")
        for _, row in table.iterrows():
            print(
                f"   {row.get('speed', 'all'):<5} baseline "
                f"{row['share_baseline']:.2f}, VT {row['share_vt']:.2f}, "
                f"covariance {row['share_covariance']:.2f}"
            )

def loo_consistency(
    pps,
    statistic="median",
    stage="after",
    conditions=None,
    speeds=None,
):
    """Does each condition's distance profile have the same shape in every subject?

    Correlates each subject's centred profile against the mean of the others.

    high mean r   the profile is shared, so it is a within-trial time effect
                  and subtracting it per distance is correct
    r near zero   the profile is not shared, so it is carrying subject-specific
                  noise into the facilitation curve at full weight

    Profiles are centred within subject, so this compares shape and ignores
    overall RT level. Speeds stay separate: a T cell is not interchangeable
    across speed contexts even where the SOAs coincide numerically.
    """
    conditions = TACTILE_CONDITIONS if conditions is None else list(conditions)
    speeds = SPEEDS if speeds is None else list(speeds)

    data = trials_at_stage(pps, stage=stage)
    rows = []

    for condition in conditions:
        for speed in speeds:
            cell = data[
                (data["sensory_condition"] == condition)
                & (data["speed"] == speed)
            ]

            if cell.empty:
                continue

            values = (
                cell.groupby(["subject", "position_rank"])["rt_ms"]
                .agg(statistic)
                .unstack("position_rank")
                .sort_index(axis=1)
            )
            counts = (
                cell.groupby(["subject", "position_rank"])["rt_ms"]
                .size()
                .unstack("position_rank")
                .sort_index(axis=1)
            )

            complete = values.dropna(axis=0, how="any")

            if len(complete) < 3:
                continue

            centred = complete.sub(complete.mean(axis=1), axis=0)

            for subject in centred.index:
                own = centred.loc[subject].to_numpy(dtype=float)
                others = (
                    centred.drop(index=subject).mean(axis=0).to_numpy(dtype=float)
                )

                cells = counts.loc[subject].dropna() if subject in counts.index else []

                rows.append({
                    "sensory_condition": condition,
                    "speed": speed,
                    "subject": subject,
                    "r": _safe_correlation(own, others),
                    "range_ms": float(np.ptp(complete.loc[subject])),
                    "min_cell_trials": int(cells.min()) if len(cells) else np.nan,
                    "n_distances": int(centred.shape[1]),
                    "n_subjects": int(len(centred)),
                })

    per_subject = pd.DataFrame(rows)

    if per_subject.empty:
        return per_subject, pd.DataFrame()

    summary = (
        per_subject
        .groupby(["sensory_condition", "speed"], as_index=False)
        .agg(
            mean_r=("r", "mean"),
            sd_r=("r", "std"),
            n_subjects=("subject", "count"),
            median_range_ms=("range_ms", "median"),
            median_min_cell=("min_cell_trials", "median"),
        )
    )

    return per_subject.round(3), summary.round(3)

def baseline_contribution(
    pps, statistic="median", stage="after", by_speed=True,
    baseline="delay_matched",
):
    """Which curve is driving the shape of the facilitation profile.

    Facilitation is baseline minus VT, so its shape across distance can come
    from either term. Because variance is additive,

        Var(facilitation) = Var(baseline) + Var(VT) - 2 Cov(baseline, VT)

    the three shares sum to 1 and read directly as proportions.

    share_baseline near 1, r_facilitation_baseline near +1
        the curve is inherited from the tactile baseline
    share_vt near 1, r_facilitation_vt near -1
        the curve is the visuo-tactile profile, inverted
    """
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")

    facilitation_column = _facilitation_column(baseline)
    baseline_column = {
        "delay_matched": "T",
        "smooth": "baseline_smooth_ms",
        "expectancy_model": "baseline_model_ms",
    }[baseline]

    # Built once: facilitation_table refits every smooth baseline, so calling
    # it twice doubles the work for nothing.
    table = facilitation_table(pps, statistic=statistic, stage=stage)

    keep = [
        column
        for column in ["subject", "session", "speed", "position_rank"]
        if column in table.columns
    ]

    table = table[keep + [baseline_column, "VT", facilitation_column]].rename(
        columns={
            baseline_column: "tactile_baseline_ms",
            facilitation_column: "facilitation_ms",
        }
    )
    values = ["tactile_baseline_ms", "VT", "facilitation_ms"]
    keys = ["subject"] + (["speed"] if by_speed else [])

    if not by_speed:
        table = (
            table
            .groupby(["subject", "position_rank"], as_index=False)[values]
            .mean()
        )

    rows = []

    for key, one in table.groupby(keys):
        one = one.sort_values("position_rank").dropna(subset=values)

        if len(one) < 3:
            continue

        # Named tactile_ms rather than baseline, which is the argument.
        tactile_ms = one["tactile_baseline_ms"].to_numpy(dtype=float)
        vt = one["VT"].to_numpy(dtype=float)
        facilitation = one["facilitation_ms"].to_numpy(dtype=float)

        total = float(np.var(facilitation, ddof=1))
        usable_variance = total > 0

        key_tuple = key if isinstance(key, tuple) else (key,)
        record = dict(zip(keys, key_tuple))
        record.update({
            "n_distances": int(len(one)),
            "range_baseline_ms": float(np.ptp(tactile_ms)),
            "range_vt_ms": float(np.ptp(vt)),
            "range_facilitation_ms": float(np.ptp(facilitation)),
            "r_facilitation_baseline": _safe_correlation(facilitation, tactile_ms),
            "r_facilitation_vt": _safe_correlation(facilitation, vt),
            "share_baseline": (
                np.var(tactile_ms, ddof=1) / total if usable_variance else np.nan
            ),
            "share_vt": np.var(vt, ddof=1) / total if usable_variance else np.nan,
            "share_covariance": (
                -2.0 * np.cov(tactile_ms, vt, ddof=1)[0, 1] / total
                if usable_variance else np.nan
            ),
        })
        rows.append(record)

    per_subject = pd.DataFrame(rows)

    if per_subject.empty:
        return per_subject, pd.DataFrame()

    metrics = [column for column in per_subject.columns if column not in keys]

    if by_speed:
        summary = (
            per_subject
            .groupby("speed", as_index=False)
            .agg(
                **{metric: (metric, "mean") for metric in metrics},
                n_subjects=("subject", "count"),
            )
        )
    else:
        summary = pd.DataFrame({
            **{metric: [per_subject[metric].mean()] for metric in metrics},
            "n_subjects": [per_subject["subject"].nunique()],
        })

    summary["statistic"] = statistic
    summary["stage"] = stage
    return per_subject.round(3), summary.round(3)

def plot_inspection(
    pps,
    statistic="median",
    subjects=None,
    drop=None,
    whisker=1.5,
    line_width=2.0,
    marker_size=None,
    save_path=None,
):
    """Look at the data before deciding anything about exclusions.

    Two rows, before and after, sharing scales column by column so the two
    can be read against each other rather than side by side.

        A. Composition. What proportion of trials each condition contributed,
           labelled with the count. The design as delivered, not as planned.
        B. RT density, T against VT on one axis. Overlaid rather than in two
           panels, because the question is whether the distributions differ,
           not what each looks like alone.
        C. RT by condition. The line is the median, the filled circle the
           mean. Their separation is the skew.
        D. RT by distance and condition. Where a distance effect would first
           be visible in the raw trials.
        E, F. Group RT across distance, speeds collapsed, with the cells
           summarised by median and by mean. Unlike C and D, which pool
           trials across participants, these summarise within each
           participant first and then average participants, so a
           participant with more usable trials does not count for more.
           Read against D: where they disagree, the pooled picture is being
           driven by trial counts rather than by participants. Read E
           against F: where those disagree, the profile depends on the
           estimator, which at four tactile trials per cell is a real risk.
        G, H. The same two, split by speed. A profile present at one speed
           and absent at the other is diluted in E and F, so this is where
           to look for it.

    No error bands are drawn on E through H. At seven participants the
    standard error is wide enough to cover any shape, and overlapping bands
    across four panels obscure the profiles they are meant to qualify. The
    precision question is answered numerically by ``precision``, not
    visually here.

    Blue is visuo-tactile throughout, grey tactile-only, from the project
    style module.

    Every trial in the row is drawn, and the RT axis spans the full range
    observed before exclusions. The before row therefore shows the extreme
    trials where they actually fell, and the after row shows the same panel
    without them. The difference between the two rows is the effect of the
    exclusions; nothing is annotated or marked to make that point.
    """
    pps = _drop_subjects(_select_subjects(pps, subjects), drop)

    ranks = _ranks(pps)

    fig = plt.figure(figsize=(27, 7.6))
    grid = fig.add_gridspec(
        2, 8,
        width_ratios=[0.6, 0.9, 0.45, 1.8, 1.15, 1.15, 1.5, 1.5],
        hspace=0.34,
        wspace=0.38,
    )

    axes = {
        stage: [fig.add_subplot(grid[row, column]) for column in range(8)]
        for row, stage in enumerate(STAGES)
    }

    all_conditions = [
        condition
        for condition in ["V", "T", "VT"]
        if condition in pps["sensory_condition"].unique()
    ]
    composition_colour = dict(CONDITION_COLOUR)
    composition_colour.setdefault("V", style._MUTED)

    # One RT range for every panel and both rows, spanning every observed
    # trial rather than the exclusion window, so nothing the rules remove is
    # hidden and the two rows stay directly comparable.
    limits = _observed_range(pps)

    rng = np.random.default_rng(11)

    for stage in STAGES:
        (
            ax_composition, ax_density, ax_condition, ax_distance,
        ) = axes[stage][:4]
        group_axes = axes[stage][4:]
        data = trials_at_stage(pps, stage=stage)

        # --- A. Composition ------------------------------------------
        source = (
            pps if stage == "before"
            else pps[pps["usable"] | (pps["sensory_condition"] == "V")]
        )

        counts = (
            source["sensory_condition"]
            .value_counts()
            .reindex(all_conditions, fill_value=0)
        )
        total = int(counts.sum())
        percent = 100 * counts / total if total else counts * np.nan

        bars = ax_composition.bar(
            np.arange(len(all_conditions)),
            percent.to_numpy(dtype=float),
            width=0.62,
            color=[composition_colour[c] for c in all_conditions],
            linewidth=0,
        )

        for bar, condition in zip(bars, all_conditions):
            ax_composition.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.0f}%\n{int(counts[condition])}",
                ha="center", va="bottom", fontsize=7.5, color=style.INK,
                linespacing=1.25,
            )

        ax_composition.set_xticks(np.arange(len(all_conditions)))
        ax_composition.set_xticklabels(all_conditions)
        ax_composition.set_ylabel("% of trials")
        ax_composition.set_ylim(0, 78)

        # --- B. RT density -------------------------------------------
        for condition in TACTILE_CONDITIONS:
            values = (
                data.loc[data["sensory_condition"] == condition, "rt_ms"]
                .dropna().to_numpy(dtype=float)
            )

            if values.size < 3 or np.std(values) == 0:
                continue

            colour = CONDITION_COLOUR[condition]
            kde = stats.gaussian_kde(values)
            x = np.linspace(limits[0], limits[1], 400)

            ax_density.fill_between(
                x, 0, kde(x), color=colour, alpha=0.18, linewidth=0
            )
            ax_density.plot(
                x, kde(x), color=colour, linewidth=1.4, alpha=0.85,
                label=f"{CONDITION_LABELS[condition]} (n = {values.size:,})",
            )
            ax_density.axvline(
                float(np.median(values)), color=colour,
                linestyle="--", linewidth=1.0, alpha=0.7,
            )

        ax_density.set_xlim(limits)
        ax_density.set_xlabel("Reaction time (ms)")
        ax_density.set_ylabel("Density")
        ax_density.legend(frameon=False, fontsize=7.5, loc="upper right")

        # --- C. RT by condition --------------------------------------
        for position, condition in enumerate(TACTILE_CONDITIONS, start=1):
            values = (
                data.loc[data["sensory_condition"] == condition, "rt_ms"]
                .dropna().to_numpy(dtype=float)
            )

            if not values.size:
                continue

            colour = CONDITION_COLOUR[condition]
            _boxplot(ax_condition, values, position, colour, whisker=whisker)

            _scatter_trials(ax_condition, values, position, colour, rng, spread=0.05)

            ax_condition.scatter(
                position,
                values.mean(),
                marker="o",
                s=30,
                facecolors=colour,
                edgecolors="none",
                linewidths=0,
                zorder=6,
            )

        ax_condition.set_xticks([1, 2])
        ax_condition.set_xticklabels(TACTILE_CONDITIONS)
        ax_condition.set_ylim(limits)
        ax_condition.set_ylabel("RT (ms)")

        # --- D. RT by distance and condition -------------------------
        offsets = {"T": -0.19, "VT": 0.19}

        for condition in TACTILE_CONDITIONS:
            colour = CONDITION_COLOUR[condition]

            for rank in ranks:
                values = (
                    data.loc[
                        data["position_rank"].eq(rank)
                        & data["sensory_condition"].eq(condition),
                        "rt_ms",
                    ].dropna().to_numpy(dtype=float)
                )

                if not values.size:
                    continue

                centre = rank + offsets[condition]
                _boxplot(
                    ax_distance, values, centre, colour,
                    width=0.31, whisker=whisker,
                )

                _scatter_trials(
                    ax_distance, values, centre, colour, rng,
                    spread=0.028, size=5,
                )

            ax_distance.plot(
                [], [], marker="s", linestyle="none", color=colour,
                label=CONDITION_LABELS[condition],
            )

        ax_distance.set_xlim(min(ranks) - 0.6, max(ranks) + 0.6)
        ax_distance.set_ylim(limits)
        ax_distance.set_ylabel("RT (ms)")
        ax_distance.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
        _distance_axis(ax_distance, ranks)

        # --- E to H. Group RT profiles ------------------------------
        panels = [
            ("median", False),
            ("mean", False),
            ("median", True),
            ("mean", True),
        ]

        for ax, (estimator, split_by_speed) in zip(group_axes, panels):
            _, group = rt_profile(
                pps, statistic=estimator, stage=stage, by_speed=split_by_speed
            )

            for condition in TACTILE_CONDITIONS:
                colour = CONDITION_COLOUR[condition]

                # Colour carries the condition; opacity and marker size carry
                # the speed, so the two never compete for the same channel.
                variants = (
                    [("slow", 0.45, 4), ("fast", 1.0, 6)]
                    if split_by_speed else [(None, 1.0, 5)]
                )

                for speed, alpha, size in variants:
                    line = group[group["sensory_condition"] == condition]

                    if speed is not None:
                        line = line[line["speed"] == speed]

                    line = line.sort_values("position_rank")

                    if line.empty:
                        continue

                    ax.plot(
                        line["position_rank"], line["group_mean"],
                        color=colour, alpha=alpha,
                        marker="s" if condition == "T" else "o",
                        linestyle="--" if condition == "T" else "-",
                        markerfacecolor="white" if condition == "T" else colour,
                        markersize=marker_size or size,
                        linewidth=line_width,
                        label=(
                            f"{condition}, {speed}" if speed
                            else CONDITION_LABELS[condition]
                        ),
                    )

            ax.set_ylabel(f"Group {estimator} RT (ms)")
            ax.legend(
                frameon=False, fontsize=7 if split_by_speed else 8,
                ncol=2 if split_by_speed else 1, loc="best",
            )
            _distance_axis(ax, ranks)

    titles = [
        "A. Trial composition",
        "B. RT density by condition",
        "C. RT by condition",
        "D. RT by distance and condition",
        "E. Group median by distance",
        "F. Group mean by distance",
        "G. Group median, by speed",
        "H. Group mean, by speed",
    ]
    for ax, title in zip(axes["before"], titles):
        ax.set_title(title, loc="left", pad=8, fontsize=9.5)

    for stage in STAGES:
        axes[stage][0].set_ylabel(
            f"{STAGE_STYLE[stage]['label']}\n\n% of trials"
        )

    # Density is held common across the two rows so a change caused by the
    # exclusions reads as movement rather than as a redrawn axis. The two
    # group-median panels share one range with each other as well as across
    # rows, since they show the same quantity at two levels of detail.
    low = min(axes[stage][1].get_ylim()[0] for stage in STAGES)
    high = max(axes[stage][1].get_ylim()[1] for stage in STAGES)
    for stage in STAGES:
        axes[stage][1].set_ylim(low, high)

    group_columns = (4, 5, 6, 7)
    group_low = min(
        axes[stage][column].get_ylim()[0]
        for stage in STAGES for column in group_columns
    )
    group_high = max(
        axes[stage][column].get_ylim()[1]
        for stage in STAGES for column in group_columns
    )
    for stage in STAGES:
        for column in group_columns:
            axes[stage][column].set_ylim(group_low, group_high)

    for stage in STAGES:
        for ax in axes[stage]:
            ax.grid(axis="y", color=style._RULE, linewidth=0.6, alpha=0.7)
            ax.set_axisbelow(True)
            style.clean(ax)

    fig.suptitle(
        "Reaction-time data before and after exclusions",
        x=0.015, ha="left", y=0.995, color=style.INK,
    )
    _save(fig, save_path)

    return fig

def _scatter_trials(ax, values, centre, colour, rng, spread=0.05, size=10):
    """Overlay raw trials using the category hue, without marker borders."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if not values.size:
        return

    ax.scatter(
        centre + rng.normal(0, spread, values.size),
        values,
        s=size,
        facecolors=colour,
        edgecolors="none",
        linewidths=0,
        alpha=0.55,
        zorder=3,
    )

def _observed_range(pps, pad=0.04):
    """RT range covering every observed trial before exclusions.

    Deliberately not the whisker range and not the exclusion window. The
    point of the before row is to show the trials the rules are about to
    remove, so every one of them has to be inside the axis. Compressing the
    boxes somewhat is the price of that, and it is the right price here.
    """
    values = trials_at_stage(pps, stage="before")["rt_ms"].dropna()

    if values.empty:
        return float(RT_MIN_MS), float(RT_MAX_MS)

    low, high = float(values.min()), float(values.max())
    span = high - low if high > low else 1.0
    return max(0.0, low - pad * span), high + pad * span

def _darker(colour, factor=0.62):
    """A darker version of a colour, for the mean marker inside a box."""
    red, green, blue = mcolors.to_rgb(colour)
    return (red * factor, green * factor, blue * factor)

def _boxplot(ax, values, position, colour, width=0.44, whisker=1.5):
    """One category-coloured boxplot with translucent fill and opaque outline."""
    box = ax.boxplot(
        [values],
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        whis=whisker,
    )
    style.style_boxplot(
        box,
        [colour],
        box_alpha=0.22,
        linewidth=1.25,
    )
    return box

def _distribution_shape(
    ax, values, position, colour, width=0.44, whisker=1.5, kind="both",
):
    """One cell drawn as a box, a violin, or a violin with the box inside it.

    The violin carries the shape and the box the quartiles, so "both" is the
    default: at these cell sizes the kernel can suggest a bimodality that three
    or four observations cannot support, and the box says where the middle
    actually is.
    """
    if kind not in {"box", "violin", "both"}:
        raise ValueError("distribution must be 'box', 'violin', or 'both'")

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if not values.size:
        return

    if kind in {"violin", "both"} and values.size >= 2 and np.std(values) > 0:
        parts = ax.violinplot(
            [values],
            positions=[position],
            widths=width * 1.6,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for body in parts["bodies"]:
            body.set_facecolor(colour)
            body.set_edgecolor(colour)
            body.set_alpha(0.18)
            body.set_linewidth(0)

    if kind in {"box", "both"}:
        _boxplot(
            ax, values, position, colour,
            width=width * (0.42 if kind == "both" else 1.0),
            whisker=whisker,
        )

    elif kind == "violin":
        # No box, so the median still needs marking.
        ax.plot(
            [position - width * 0.35, position + width * 0.35],
            [float(np.median(values))] * 2,
            color=_darker(colour), linewidth=2.0,
            solid_capstyle="butt", zorder=5,
        )

def plot_exclusions(pps, statistic="median", save_path=None):
    """What the exclusions did: retention, RT profile, facilitation profile.

    Panels B and C are the ones that matter. If the two curves lie on top of
    each other, the exclusions removed noise without moving the effect. If
    they diverge, the rule is shaping the result and belongs in the
    sensitivity analyses.
    """
    ranks = _ranks(pps)

    fig, (ax_counts, ax_rt, ax_fac) = plt.subplots(1, 3, figsize=(15.5, 4.6))

    counts = exclusion_table(pps).set_index("subject")
    reasons = [
        column
        for column in REASON_ORDER
        if column in counts.columns and counts[column].sum() > 0
    ]

    bottom = np.zeros(len(counts))
    x = np.arange(len(counts))

    for index, reason in enumerate(reasons):
        colour = (
            style.CONTROL if reason == "usable"
            else style.shade(style.INK, 0.25 + 0.5 * index / max(len(reasons) - 1, 1))
        )
        values = counts[reason].to_numpy(dtype=float)
        ax_counts.bar(
            x, values, bottom=bottom, color=colour,
            edgecolor="white", linewidth=0.5, label=reason.replace("_", " "),
        )
        bottom += values

    for index, (_, row) in enumerate(counts.iterrows()):
        ax_counts.text(
            index, row["n_tactile"], f"{row['percent_kept']:.0f}%",
            ha="center", va="bottom", fontsize=8, color=style.INK,
        )

    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(
        _display_names(counts.index), rotation=25, ha="right",
    )
    ax_counts.set_ylabel("Tactile trials")
    ax_counts.set_title("A. Trials kept and lost (label = % kept)", loc="left")
    ax_counts.legend(frameon=False, fontsize=8)
    style.clean(ax_counts)

    for stage in STAGES:
        stage_style = STAGE_STYLE[stage]

        _, group = rt_profile(pps, statistic=statistic, stage=stage, by_speed=False)

        for condition in TACTILE_CONDITIONS:
            line = group[
                group["sensory_condition"] == condition
            ].sort_values("position_rank")

            if line.empty:
                continue

            ax_rt.plot(
                line["position_rank"], line["group_mean"],
                color=stage_style["colour"],
                linestyle="-" if condition == "T" else "--",
                marker=stage_style["marker"], markersize=4.5,
                markerfacecolor=(
                    stage_style["colour"] if condition == "T" else "white"
                ),
                linewidth=2.0, alpha=stage_style["alpha"],
                label=f"{stage_style['label']}, {CONDITION_LABELS[condition]}",
            )

        _, group = facilitation_profile(
            pps, statistic=statistic, stage=stage, by_speed=False
        )
        group = group.sort_values("position_rank")

        if group.empty:
            continue

        ax_fac.fill_between(
            group["position_rank"],
            group["group_mean"] - group["group_sem"],
            group["group_mean"] + group["group_sem"],
            color=stage_style["colour"], alpha=0.15, linewidth=0,
        )
        ax_fac.plot(
            group["position_rank"], group["group_mean"],
            color=stage_style["colour"], linestyle=stage_style["linestyle"],
            marker=stage_style["marker"], markersize=4.5, linewidth=2.2,
            label=stage_style["label"],
        )

    _finish(
        ax_rt,
        f"B. {statistic.capitalize()} RT before and after",
        f"{statistic.capitalize()} RT (ms)",
        ranks=ranks,
    )
    ax_rt.legend(frameon=False, fontsize=8)

    ax_fac.axhline(0, linestyle=":", linewidth=0.8, color=style._MUTED)
    _finish(ax_fac, "C. Facilitation before and after", "T - VT (ms)", ranks=ranks)
    ax_fac.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Effect of the trial exclusions",
        x=0.02, ha="left", y=1.02, color=style.INK,
    )
    fig.tight_layout()
    _save(fig, save_path)

    return fig, exclusion_impact(pps, statistic=statistic)

def plot_rt_distributions(
    pps,
    by=None,
    stage="both",
    bins=None,
    xlim=None,
    show_limits=True,
    save_path=None,
):
    """Reaction-time histograms at whatever level of detail is asked for.

    by=None                                     one panel
    by="sensory_condition"                      one panel per condition
    by=["sensory_condition", "position_rank"]   a rows x columns grid

    ``stage="both"`` overlays the two with a dashed median line each, so the
    effect of the RT window is visible in the distribution itself. Trials
    outside the plotted range are counted in each panel, never dropped
    silently.
    """
    stages = _stages_to_draw(stage)
    by = [by] if isinstance(by, str) else (list(by) if by else [])

    if len(by) > 2:
        raise ValueError("by accepts at most two columns")

    reference = trials_at_stage(pps, stage="before")

    if len(by) == 0:
        row_values, col_values = [None], [None]
        figsize, default_bins, default_xlim = (7.5, 4.5), 30, "robust"
    elif len(by) == 1:
        row_values = [None]
        col_values = sorted(reference[by[0]].dropna().unique())
        figsize = (5.0 * len(col_values), 4.2)
        default_bins, default_xlim = 25, "robust"
    else:
        row_values = sorted(reference[by[0]].dropna().unique())
        col_values = sorted(reference[by[1]].dropna().unique())
        figsize = (2.0 * len(col_values), 2.9 * len(row_values))
        default_bins, default_xlim = 14, "tight"

    bins = default_bins if bins is None else bins
    xlim = default_xlim if xlim is None else xlim
    compact = len(by) == 2

    # Conditions have very different trial counts, so sharing y across rows
    # would flatten the smaller one. Share within a row instead.
    fig, axes = plt.subplots(
        len(row_values), len(col_values), figsize=figsize,
        sharex=True, sharey="row" if compact else True, squeeze=False,
    )

    limits = _rt_range(pps, xlim=xlim)
    edges = np.linspace(limits[0], limits[1], int(bins) + 1)

    for row, row_value in enumerate(row_values):
        for col, col_value in enumerate(col_values):
            ax = axes[row, col]
            drawn = {}

            for one_stage in stages:
                data = trials_at_stage(pps, stage=one_stage)
                keep = pd.Series(True, index=data.index)

                if len(by) == 1:
                    keep &= data[by[0]] == col_value
                elif len(by) == 2:
                    keep &= data[by[0]] == row_value
                    keep &= data[by[1]] == col_value

                values = data.loc[keep, "rt_ms"].dropna()
                drawn[one_stage] = values

                if values.empty:
                    continue

                stage_style = STAGE_STYLE[one_stage]
                ax.hist(
                    values, bins=edges, color=stage_style["colour"],
                    edgecolor="white", linewidth=0.5,
                    alpha=0.55 if len(stages) > 1 else 0.95,
                    label=stage_style["label"] if len(stages) > 1 else None,
                )
                ax.axvline(
                    float(values.median()), color=stage_style["colour"],
                    linewidth=2.0, linestyle="--", zorder=5,
                )

            if show_limits:
                for limit in (RT_MIN_MS, RT_MAX_MS):
                    if limits[0] < limit < limits[1]:
                        ax.axvline(
                            limit, color=style._MUTED, linestyle=":",
                            linewidth=1.0, zorder=1,
                        )

            _annotate_histogram(ax, drawn, stages, limits, compact=compact)
            ax.set_xlim(limits)

            if len(by) == 1:
                ax.set_title(_axis_label(by[0], col_value), loc="left")
            elif len(by) == 2:
                if row == 0:
                    ax.set_title(_axis_label(by[1], col_value), loc="left")
                if col == 0:
                    ax.set_ylabel(
                        f"{_axis_label(by[0], row_value)}\nTrial count"
                    )

            if row == len(row_values) - 1:
                ax.set_xlabel("Reaction time (ms)")

            style.clean(ax)

    if len(by) < 2:
        axes[0, 0].set_ylabel("Trial count")

    for row in range(len(row_values)):
        top = max(ax.get_ylim()[1] for ax in axes[row])
        axes[row, 0].set_ylim(0, top * 1.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper right", ncol=len(labels),
            frameon=False, fontsize=8, bbox_to_anchor=(1.0, 1.02),
        )

    title = "Reaction-time distributions"
    if by:
        title += " by " + " and ".join(c.replace("_", " ") for c in by)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    _save(fig, save_path)

    return fig, shape(pps, by=by, stage=stages[-1])

def _axis_label(column, value):
    if column == "sensory_condition":
        return CONDITION_LABELS.get(value, str(value))
    if column == "position_rank":
        return f"D{int(value)}"
    return f"{column.replace('_', ' ')} = {value}"

def _annotate_histogram(ax, drawn, stages, limits, compact=False):
    lines = []
    off_scale = 0

    for stage in stages:
        values = drawn.get(stage)

        if values is None or values.empty:
            continue

        off_scale = max(
            off_scale,
            int(((values < limits[0]) | (values > limits[1])).sum()),
        )

        if compact:
            lines.append(f"n={len(values):,}, M={values.median():.0f}")
        elif len(stages) > 1:
            lines.append(f"{stage}: n={len(values):,}, M={values.median():.0f}")
        else:
            lines.append(f"n = {len(values):,}")
            lines.append(f"mean - median = {values.mean() - values.median():.1f}")
            lines.append(f"skew = {values.skew():.2f}")

    if not lines:
        ax.text(
            0.5, 0.5, "No trials", transform=ax.transAxes,
            ha="center", va="center", fontsize=8, color=style._MUTED,
        )
        return

    if off_scale:
        lines.append(f"{off_scale} trial(s) off scale")

    ax.text(
        0.03, 0.97, "\n".join(lines), transform=ax.transAxes,
        ha="left", va="top", fontsize=6.5 if compact else 7.5, color=style.INK,
    )

def plot_rt_boxplots(
    pps,
    by="position_rank",
    split_condition=True,
    stage="after",
    show_observations=True,
    ylim="robust",
    save_path=None,
):
    """RT boxplots grouped by subject or by distance.

    Diamonds mark means; the box centre line is the median. Individual
    observations are overlaid by default.
    """
    _check_stage(stage, allow_both=False)

    if by not in {"subject", "position_rank"}:
        raise ValueError("by must be 'subject' or 'position_rank'")

    data = trials_at_stage(pps, stage=stage)
    groups = sorted(data[by].dropna().unique())

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(groups)), 4.8))
    rng = np.random.default_rng(123)

    conditions = TACTILE_CONDITIONS if split_condition else [None]
    offsets = {"T": -0.17, "VT": 0.17, None: 0.0}
    width = 0.28 if split_condition else 0.5

    for condition in conditions:
        colour = CONDITION_COLOUR.get(condition, style.CONTROL)
        positions = np.arange(len(groups), dtype=float) + offsets[condition]

        arrays = []
        for group in groups:
            keep = data[by] == group
            if condition is not None:
                keep &= data["sensory_condition"] == condition
            arrays.append(data.loc[keep, "rt_ms"].dropna().to_numpy(dtype=float))

        box = ax.boxplot(
            arrays,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        style.style_boxplot(
            box,
            [colour] * len(arrays),
            box_alpha=0.22,
            linewidth=1.25,
        )

        if show_observations:
            for centre, values in zip(positions, arrays):
                if not len(values):
                    continue
                ax.scatter(
                    centre + rng.normal(0, 0.024, len(values)),
                    values,
                    s=12,
                    facecolors=colour,
                    edgecolors="none",
                    linewidths=0,
                    alpha=0.55,
                    zorder=3,
                )

        ax.scatter(
            positions,
            [np.mean(v) if len(v) else np.nan for v in arrays],
            marker="D",
            s=26,
            facecolors=colour,
            edgecolors="none",
            linewidths=0,
            zorder=4,
            label=(
                f"{CONDITION_LABELS[condition]}: diamond = mean"
                if condition is not None else "Diamond = mean"
            ),
        )

    if stage == "after":
        for limit in (RT_MIN_MS, RT_MAX_MS):
            ax.axhline(limit, linestyle=":", linewidth=0.8, color=style._MUTED)

    ax.set_xticks(np.arange(len(groups)))

    if by == "position_rank":
        ax.set_xticklabels([f"D{int(group)}" for group in groups])
        ax.set_xlabel("Distance (D1 = closest)")
    else:
        ax.set_xticklabels(groups, rotation=25, ha="right")
        ax.set_xlabel("Subject")

    ax.set_ylim(_rt_range(pps, xlim=ylim))
    ax.set_ylabel("Reaction time (ms)")
    ax.set_title(
        f"RT distributions by {by.replace('_', ' ')} "
        f"({STAGE_STYLE[stage]['label'].lower()})",
        loc="left",
    )
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.grid(axis="y", color=style._RULE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    style.clean(ax)
    fig.tight_layout()
    _save(fig, save_path)

    return fig, data

def plot_profiles(
    pps,
    kind="rt",
    statistic="median",
    stage="after",
    by_speed=True,
    baseline="delay_matched",
    individual=False,
    save_path=None,
):
    """RT or facilitation across distance.

    kind="rt"            T and VT profiles
    kind="facilitation"  T minus VT
    by_speed=True        one line per speed, which is the form the hypotheses
                         are stated in. Averaging the speeds dilutes a step
                         present at one speed and absent at the other
    individual=True      one panel per subject
    """
    if kind not in {"rt", "facilitation"}:
        raise ValueError("kind must be 'rt' or 'facilitation'")

    ranks = _ranks(pps)

    if kind == "rt":
        subject, group = rt_profile(
            pps, statistic=statistic, stage=stage, by_speed=by_speed
        )
        value, ylabel = "rt_ms", f"{statistic.capitalize()} RT (ms)"
        title = "Reaction time by distance"
    else:
        subject, group = facilitation_profile(
            pps, statistic=statistic, stage=stage, by_speed=by_speed,
            baseline=baseline,
        )
        value, ylabel = "facilitation_ms", "T - VT (ms)"
        title = "Multisensory facilitation"

    speeds = SPEEDS if by_speed else [None]

    if individual:
        subjects = sorted(subject["subject"].dropna().unique())

        fig, axes = plt.subplots(
            len(subjects), 1,
            figsize=(7, 3.4 * max(len(subjects), 1)), squeeze=False,
        )

        for ax, one_subject in zip(axes.ravel(), subjects):
            one = subject[subject["subject"] == one_subject]

            for speed in speeds:
                rows = one if speed is None else one[one["speed"] == speed]
                colour = style.SPEED[speed] if speed else style.INK

                if kind == "rt":
                    for condition in TACTILE_CONDITIONS:
                        line = rows[
                            rows["sensory_condition"] == condition
                        ].sort_values("position_rank")
                        ax.plot(
                            line["position_rank"], line[value], marker="o",
                            linestyle="-" if condition == "T" else "--",
                            color=colour,
                            markerfacecolor=colour if condition == "T" else "white",
                            label=f"{speed or 'all'} {condition}",
                        )
                else:
                    line = rows.sort_values("position_rank")
                    ax.plot(
                        line["position_rank"], line[value],
                        marker="o", color=colour, label=speed or "all speeds",
                    )

            if kind == "facilitation":
                ax.axhline(0, linestyle=":", linewidth=0.8, color=style._MUTED)

            _finish(ax, str(one_subject), ylabel, ranks=ranks)
            ax.legend(frameon=False, fontsize=8)

        fig.suptitle(
            f"Individual {kind} profiles "
            f"({STAGE_STYLE[stage]['label'].lower()})",
            x=0.02, ha="left", y=1.005, color=style.INK,
        )
        fig.tight_layout()
        _save(fig, save_path)
        return fig, subject

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for speed in speeds:
        rows = group if speed is None else group[group["speed"] == speed]
        colour = style.SPEED[speed] if speed else style.INK

        if kind == "rt":
            for condition in TACTILE_CONDITIONS:
                line = rows[
                    rows["sensory_condition"] == condition
                ].sort_values("position_rank")

                if line.empty:
                    continue

                ax.fill_between(
                    line["position_rank"],
                    line["group_mean"] - line["group_sem"],
                    line["group_mean"] + line["group_sem"],
                    color=colour, alpha=0.10, linewidth=0,
                )
                ax.plot(
                    line["position_rank"], line["group_mean"], marker="o",
                    linestyle="-" if condition == "T" else "--", color=colour,
                    markerfacecolor=colour if condition == "T" else "white",
                    linewidth=2.0, label=f"{speed or 'all'} {condition}",
                )
        else:
            line = rows.sort_values("position_rank")

            if line.empty:
                continue

            ax.fill_between(
                line["position_rank"],
                line["group_mean"] - line["group_sem"],
                line["group_mean"] + line["group_sem"],
                color=colour, alpha=0.15, linewidth=0,
            )
            ax.plot(
                line["position_rank"], line["group_mean"], marker="o",
                color=colour, linewidth=2.2,
                label=(speed or "all speeds").capitalize(),
            )

    if kind == "facilitation":
        ax.axhline(0, linestyle=":", linewidth=0.8, color=style._MUTED)

    _finish(ax, title, ylabel, ranks=ranks)
    ax.legend(frameon=False, fontsize=8, title="Speed" if by_speed else None)
    fig.tight_layout()
    _save(fig, save_path)

    return fig, group

def _fade(colour, amount):
    """Mix a colour toward white. ``amount`` 0 leaves it, 1 makes it white."""
    red, green, blue = mcolors.to_rgb(colour)
    amount = float(np.clip(amount, 0.0, 1.0))
    return (
        red + (1.0 - red) * amount,
        green + (1.0 - green) * amount,
        blue + (1.0 - blue) * amount,
    )

def _order_fraction(data, within=None):
    """Position of each trial in the session, scaled to 0 at the start and 1 at
    the end. ``within`` scales inside each group instead, which is what to use
    when the two speeds were interleaved and their trial numbers interleave
    with them."""
    order = _trial_order_column(data)

    if order is None:
        position = pd.Series(np.arange(len(data), dtype=float), index=data.index)
    else:
        position = pd.to_numeric(data[order], errors="coerce")

    if within:
        grouped = position.groupby([data[column] for column in within])
        low = grouped.transform("min")
        high = grouped.transform("max")
    else:
        low, high = position.min(), position.max()

    span = high - low
    return (position - low) / span.replace(0, np.nan) if within else (
        (position - low) / (span if span else np.nan)
    )

def plot_subject_deepdive(
    pps,
    subject,
    stage="after",
    statistic="median",
    distribution="both",
    whisker=1.5,
    show_mean_line=False,
    save_path=None,
):
    """One participant, four views of where their reaction times came from.

        A. RT by condition, every trial drawn. The line is the median and the
           filled circle the mean.
        B. The same, split by speed. A participant whose facilitation looks
           wrong at one speed only shows it here before any profile is formed.
        C. RT by block and speed. This is the drift panel: a participant who
           slowed, or who settled after a bad first block, moves the block
           medians rather than producing individual outliers.
        D. RT by distance and condition, one row per speed. Points are shaded
           by when in the session they were delivered, pale early and solid
           late, so a cell built mostly from late trials is visible as a dark
           cluster sitting away from its neighbours.

    Shading in C and D runs within each speed, because the two were
    interleaved and a raw trial number would otherwise make one speed look
    uniformly earlier than the other.

    Returns the figure and a table of the cell medians behind panel D.
    """
    data = trials_at_stage(pps, stage=stage, subjects=[subject])

    if data.empty:
        print(f"No usable trials for {subject} at stage '{stage}'.")
        return None, pd.DataFrame()

    data = data.copy()
    data["order_fraction"] = _order_fraction(data, within=["speed"])

    ranks = _ranks(pps)
    limits = _observed_range(_select_subjects(pps, [subject]))
    rng = np.random.default_rng(53)

    block = _block_column(data)
    if block is None:
        data["_block"] = pd.qcut(
            data["order_fraction"], 4, labels=False, duplicates="drop"
        ) + 1
        block = "_block"

    fig = plt.figure(figsize=(15.5, 9.0))
    grid = fig.add_gridspec(
        2, 3, width_ratios=[0.75, 1.25, 1.6], hspace=0.34, wspace=0.30
    )

    ax_condition = fig.add_subplot(grid[0, 0])
    ax_speed = fig.add_subplot(grid[0, 1])
    ax_blocks = fig.add_subplot(grid[1, :2])
    ax_distance = [fig.add_subplot(grid[row, 2]) for row in range(2)]

    # --- A. RT by condition ------------------------------------------
    for position, condition in enumerate(TACTILE_CONDITIONS, start=1):
        values = (
            data.loc[data["sensory_condition"] == condition, "rt_ms"]
            .dropna().to_numpy(dtype=float)
        )

        if not values.size:
            continue

        colour = CONDITION_COLOUR[condition]
        _distribution_shape(
            ax_condition, values, position, colour,
            whisker=whisker, kind=distribution,
        )
        _scatter_trials(ax_condition, values, position, colour, rng, spread=0.05)
        ax_condition.scatter(
            position, values.mean(), marker="o", s=30,
            facecolors=_darker(colour), edgecolors="none", zorder=6,
        )

    ax_condition.set_xticks([1, 2])
    ax_condition.set_xticklabels(TACTILE_CONDITIONS)
    ax_condition.set_ylim(limits)
    ax_condition.set_ylabel("RT (ms)")
    ax_condition.set_title("A. By condition", loc="left", fontsize=9.5)

    # --- B. RT by condition and speed --------------------------------
    position = 0
    ticks, labels = [], []

    for condition in TACTILE_CONDITIONS:
        for speed in SPEEDS:
            position += 1
            values = (
                data.loc[
                    data["sensory_condition"].eq(condition)
                    & data["speed"].eq(speed),
                    "rt_ms",
                ].dropna().to_numpy(dtype=float)
            )

            ticks.append(position)
            labels.append(f"{condition}\n{speed}")

            if not values.size:
                continue

            colour = style.SPEED[speed]
            _distribution_shape(
                ax_speed, values, position, colour,
                whisker=whisker, kind=distribution,
            )
            _scatter_trials(ax_speed, values, position, colour, rng, spread=0.05)
            ax_speed.scatter(
                position, values.mean(), marker="o", s=28,
                facecolors=_darker(colour), edgecolors="none", zorder=6,
            )

        position += 0.6

    ax_speed.set_xticks(ticks)
    ax_speed.set_xticklabels(labels, fontsize=8)
    ax_speed.set_ylim(limits)
    ax_speed.set_ylabel("RT (ms)")
    ax_speed.set_title("B. By condition and speed", loc="left", fontsize=9.5)

    # --- C. RT by block and speed ------------------------------------
    blocks = sorted(data[block].dropna().unique())
    offsets = {"slow": -0.19, "fast": 0.19}

    for speed in SPEEDS:
        colour = style.SPEED[speed]

        for index, one_block in enumerate(blocks):
            cell = data[data[block].eq(one_block) & data["speed"].eq(speed)]
            values = cell["rt_ms"].dropna().to_numpy(dtype=float)

            if not values.size:
                continue

            centre = index + offsets[speed]
            _distribution_shape(
                ax_blocks, values, centre, colour,
                width=0.30, whisker=whisker, kind=distribution,
            )

            # Shade each trial by when it fell in that speed's sequence.
            for _, trial in cell.iterrows():
                if not np.isfinite(trial["rt_ms"]):
                    continue

                ax_blocks.scatter(
                    centre + rng.normal(0, 0.035),
                    trial["rt_ms"],
                    s=16,
                    facecolors=_fade(colour, 0.75 * (1 - trial["order_fraction"])),
                    edgecolors="none",
                    alpha=0.85,
                    zorder=3,
                )

        ax_blocks.plot(
            [], [], marker="s", linestyle="none", color=colour,
            label=speed.capitalize(),
        )

    ax_blocks.set_xticks(np.arange(len(blocks)))
    ax_blocks.set_xticklabels([str(one_block) for one_block in blocks])
    ax_blocks.set_xlabel("Block")
    ax_blocks.set_ylim(limits)
    ax_blocks.set_ylabel("RT (ms)")
    ax_blocks.set_title(
        "C. By block (pale = early in that speed, solid = late)",
        loc="left", fontsize=9.5,
    )
    ax_blocks.legend(frameon=False, fontsize=8, ncol=2, title="Speed")

    # --- D. RT by distance, one row per speed ------------------------
    condition_offsets = {"T": -0.19, "VT": 0.19}
    cells = []

    for ax, speed in zip(ax_distance, SPEEDS):
        summaries = {}

        for condition in TACTILE_CONDITIONS:
            colour = CONDITION_COLOUR[condition]

            for rank in ranks:
                cell = data[
                    data["position_rank"].eq(rank)
                    & data["sensory_condition"].eq(condition)
                    & data["speed"].eq(speed)
                ]
                values = cell["rt_ms"].dropna().to_numpy(dtype=float)

                cells.append({
                    "speed": speed,
                    "sensory_condition": condition,
                    "position_rank": rank,
                    "n_trials": int(values.size),
                    "estimate_ms": (
                        float(np.median(values)) if statistic == "median"
                        else float(np.mean(values))
                    ) if values.size else np.nan,
                    "mean_order_fraction": float(cell["order_fraction"].mean())
                    if len(cell) else np.nan,
                })

                if not values.size:
                    continue

                centre = rank + condition_offsets[condition]

                for _, trial in cell.iterrows():
                    if not np.isfinite(trial["rt_ms"]):
                        continue

                    ax.scatter(
                        centre + rng.normal(0, 0.045),
                        trial["rt_ms"],
                        s=22,
                        facecolors=_fade(
                            colour, 0.75 * (1 - trial["order_fraction"])
                        ),
                        edgecolors="none",
                        alpha=0.9,
                        zorder=3,
                    )

                summaries.setdefault(
                    condition, {"median": [], "mean": []}
                )
                summaries[condition]["median"].append(
                    (centre, float(np.median(values)))
                )
                summaries[condition]["mean"].append(
                    (centre, float(np.mean(values)))
                )

            ax.plot(
                [], [], marker="s", linestyle="none", color=colour,
                label=CONDITION_LABELS[condition],
            )

        # Joined across distances, so the delay dependence reads as a
        # trajectory rather than seven separate marks. Both lines take
        # the condition's own colour and are told apart by style, since
        # the condition is the more important distinction: a tactile-only
        # and a visuo-tactile cell at the same distance are different
        # quantities, whereas the median and mean are two summaries of
        # one. Where they separate, a single slow response is carrying
        # the mean.
        # Drawn in the same style as the profile panels of
        # plot_inspection, so a reader moving between the two figures
        # reads the same encoding: dashed with open squares for
        # tactile-only, solid with filled circles for visuo-tactile.
        for condition, traces in summaries.items():
            colour_line = CONDITION_COLOUR[condition]

            traces_to_draw = [("median", 1.0)]

            if show_mean_line:
                traces_to_draw.append(("mean", 0.45))

            for name, alpha_line in traces_to_draw:
                points = sorted(traces[name])

                if len(points) < 2:
                    continue

                ax.plot(
                    [x for x, _ in points], [y for _, y in points],
                    color=colour_line, alpha=alpha_line,
                    marker="s" if condition == "T" else "o",
                    linestyle="--" if condition == "T" else "-",
                    markerfacecolor=(
                        "white" if condition == "T" else colour_line
                    ),
                    markersize=5, linewidth=3.0, zorder=6,
                )

        ax.set_xlim(min(ranks) - 0.6, max(ranks) + 0.6)
        ax.set_ylim(limits)
        _finish(
            ax,
            f"{'D' if speed == SPEEDS[0] else 'E'}. {speed.capitalize()}, "
            f"bar = cell median",
            "RT (ms)",
            ranks=ranks,
        )

        if show_mean_line:
            # The mean is the same colour and style, drawn faint, so the
            # legend names the distinction the eye cannot take from the
            # line itself.
            ax.plot([], [], color=style._MUTED, linewidth=3.0,
                    label="Median")
            ax.plot([], [], color=style._MUTED, linewidth=3.0, alpha=0.45,
                    label="Mean")

        ax.legend(frameon=False, fontsize=7.5, ncol=2)

    for ax in [ax_condition, ax_speed, ax_blocks]:
        ax.grid(axis="y", color=style._RULE, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        style.clean(ax)

    fig.suptitle(
        f"{_display_name(subject)}: where the reaction times came from "
        f"({STAGE_STYLE[stage]['label'].lower()})",
        x=0.015, ha="left", y=0.995, color=style.INK,
    )
    _save(fig, save_path)

    return fig, pd.DataFrame(cells).round(1)

def trial_level_model(pps, stage="after", statistic=None):
    """The distance by speed interaction on facilitation, fitted on trials.

    The preregistration names this as the floor that needs no boundary
    estimate, and it is the one test in the analysis that never forms a
    four-trial cell. Reaction time is modelled directly with condition,
    distance and speed as fixed effects and a random intercept per
    participant, so the tactile-only trials enter as data rather than as a
    subtracted estimate and their sampling noise never propagates.

    The terms that carry the hypotheses:

        condition x distance          facilitation varies with distance
        condition x distance x speed  and does so differently by speed

    ``statistic`` is accepted and ignored, so the call signature matches the
    rest of the module; this model uses trials, not cell estimates.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError(
            "trial_level_model needs statsmodels: pip install statsmodels"
        )

    data = trials_at_stage(pps, stage=stage).copy()

    data = data.dropna(
        subset=["rt_ms", "position_rank", "speed", "sensory_condition"]
    )
    data["condition"] = pd.Categorical(
        data["sensory_condition"], categories=TACTILE_CONDITIONS
    )
    data["speed_c"] = pd.Categorical(data["speed"], categories=SPEEDS)

    # Distance is centred so that the condition main effect reads as overall
    # facilitation rather than as facilitation extrapolated to D0, and so the
    # interaction terms stay on a comparable scale.
    distance = pd.to_numeric(data["position_rank"], errors="coerce")
    data["distance_c"] = distance - distance.mean()

    model = smf.mixedlm(
        "rt_ms ~ condition * distance_c * speed_c",
        data=data,
        groups=data["subject"],
    )

    # lbfgs is fast but goes singular on designs this small; fall back rather
    # than fail.
    fit = None
    for method in ["lbfgs", "bfgs", "powell"]:
        try:
            fit = model.fit(reml=True, method=method)
            break
        except np.linalg.LinAlgError:
            continue

    if fit is None:
        raise np.linalg.LinAlgError(
            "the mixed model did not converge under any optimiser"
        )

    table = pd.DataFrame({
        "term": fit.params.index,
        "estimate": fit.params.to_numpy(),
        "se": fit.bse.to_numpy(),
        "z": fit.tvalues.to_numpy(),
        "p": fit.pvalues.to_numpy(),
    })
    table = table[~table["term"].str.contains("Group Var")].reset_index(drop=True)
    table.attrs["n_trials"] = int(len(data))
    table.attrs["n_subjects"] = int(data["subject"].nunique())

    return fit, table.round(4)

def crossover_boundary(
    pps,
    statistic="median",
    stage="after",
    quantity="facilitation",
    **kwargs,
):
    """Half-maximum crossover distance, the preregistered sigmoid fallback.

    ``quantity`` chooses what profile the boundary is read from:

        "facilitation"  the preregistered index, T - VT at each distance.
                        Both terms enter, so the tactile sampling noise
                        propagates into the boundary.
        "VT"            the visuo-tactile profile alone. Fifteen trials per
                        cell instead of four, and no subtraction, so the
                        boundary does not inherit the baseline's noise. This
                        is the quantity Noel et al. fit the sigmoid to.

    Under "VT" the profile runs the other way, faster near the body rather
    than more facilitation, so the plateaus are swapped before the crossing
    is found. The returned boundary is on the same distance scale either
    way and the two are directly comparable.

    Where the fitted sigmoid centre is not estimable with usable precision,
    the preregistration falls back to reading the boundary straight off the
    observed profile: take the far plateau as the mean of D7 and D6, the near
    plateau as the mean of D1 and D2, set the half-maximum H midway between
    them, and interpolate the first inward crossing of H.

    Returns one row per subject and speed with the plateaus, H, the crossover
    in rank and in metres, and ``crossover_ok``. A crossing can be
    interpolated from any profile including a flat one, so ``crossover_ok``
    additionally requires the two plateaus to differ by at least
    ``MIN_CROSSOVER_AMPLITUDE_MS``; below that the boundary is being read off
    a line with no step in it. The second return value is fast minus slow per
    subject in metres, over the participants whose crossings are usable at
    both speeds.
    """
    if quantity == "facilitation":
        subject, _ = facilitation_profile(
            pps, statistic=statistic, stage=stage, by_speed=True, **kwargs
        )
        value = "facilitation_ms"

    elif quantity in TACTILE_CONDITIONS:
        subject, _ = rt_profile(
            pps, statistic=statistic, stage=stage, by_speed=True
        )
        subject = subject[subject["sensory_condition"] == quantity]

        # RT falls toward the body where facilitation rises, so the sign is
        # flipped to keep "near plateau high, far plateau low" true for both
        # and the crossing logic identical.
        subject = subject.assign(value=-subject["rt_ms"])
        value = "value"

    else:
        raise ValueError(
            "quantity must be 'facilitation' or one of: "
            + ", ".join(TACTILE_CONDITIONS)
        )

    rows = []

    for (subj, speed), one in subject.groupby(["subject", "speed"]):
        one = one.sort_values("position_rank")
        profile = one.set_index("position_rank")[value]

        near = profile.reindex([1, 2]).mean()
        far = profile.reindex([7, 6]).mean()

        if not np.isfinite(near) or not np.isfinite(far):
            continue

        half = (near + far) / 2.0

        # Walk inward from the far end and interpolate the first crossing.
        ranks = sorted(profile.dropna().index, reverse=True)
        crossover = np.nan

        for outer, inner in zip(ranks, ranks[1:]):
            y_outer, y_inner = profile[outer], profile[inner]

            if (y_outer - half) * (y_inner - half) <= 0 and y_outer != y_inner:
                crossover = inner + (half - y_inner) * (outer - inner) / (
                    y_outer - y_inner
                )
                break

        # Sign back to milliseconds for the report. Under "VT" the profile
        # was negated so the crossing logic could be shared; the plateaus are
        # reported as the reaction times they are.
        sign = -1.0 if quantity in TACTILE_CONDITIONS else 1.0

        rows.append({
            "subject": subj,
            "speed": speed,
            "near_plateau_ms": sign * near,
            "far_plateau_ms": sign * far,
            "half_maximum_ms": sign * half,
            "amplitude_ms": abs(near - far),
            "crossover_rank": crossover,
            "crossover_m": (
                rank_to_metres(crossover) if np.isfinite(crossover) else np.nan
            ),
            "quantity": quantity,
        })

    boundaries = pd.DataFrame(rows)

    if boundaries.empty:
        return boundaries, pd.DataFrame()

    # A crossing is only meaningful when the two plateaus actually differ.
    # Where the amplitude is inside the noise a crossing can still be
    # interpolated, but it is reading a boundary off a flat line.
    boundaries["crossover_ok"] = (
        boundaries["crossover_rank"].notna()
        & (boundaries["amplitude_ms"] >= MIN_CROSSOVER_AMPLITUDE_MS)
    )

    usable = boundaries[boundaries["crossover_ok"]]

    wide = usable.pivot(
        index="subject", columns="speed", values="crossover_m"
    )

    for speed in SPEEDS:
        if speed not in wide.columns:
            wide[speed] = np.nan

    wide["delta_pps_m"] = wide["fast"] - wide["slow"]

    return boundaries.round(2), wide.reset_index().round(2)

def profile_legibility(
    pps,
    statistic="median",
    stage="after",
    baseline="delay_matched",
    n_boot=N_BOOTSTRAP,
    by_speed=False,
):
    """Can each participant's facilitation profile support a shape at all?

    Fitting a four-parameter sigmoid to seven points will always return
    something. This asks the prior question: is there anything in the profile
    for a model to find, or is the participant's curve indistinguishable from
    noise around a flat line?

    Three quantities answer it, and they are independent of any model.

        range_over_floor    the peak-to-trough range of the profile divided
                            by the standard error of a single facilitation
                            point. Below about 2 the whole curve fits inside
                            its own error bars.

        reversals           how many times the profile changes direction
                            across the seven distances. A sigmoid has one
                            monotone run, so at most one reversal. Four or
                            five reversals on seven points is alternation,
                            which is what noise looks like.

        rho                 Spearman correlation with distance. A profile
                            with a boundary is monotone in the region the
                            boundary separates, so |rho| should be
                            substantial even when the shape is not linear.

    ``fittable`` is True only when the range clears twice the floor and the
    profile reverses at most twice. It is a screen, not a test: a participant
    who fails it cannot support a sigmoid, but passing does not establish a
    boundary.
    """
    subject, _ = facilitation_profile(
        pps, statistic=statistic, stage=stage, by_speed=by_speed,
        baseline=baseline,
    )

    if subject.empty:
        return pd.DataFrame()

    # Standard error of a single facilitation point, per subject, from the
    # two cells that produced it.
    cells = cell_precision(pps, statistic=statistic, stage=stage, n_boot=n_boot)

    keys = ["subject"] + (["speed"] if by_speed else []) + ["position_rank"]

    wide = (
        cells
        .groupby(
            ["subject", "speed", "position_rank", "sensory_condition"],
            as_index=False,
        )["se_ms"]
        .mean()
        .pivot(
            index=["subject", "speed", "position_rank"],
            columns="sensory_condition",
            values="se_ms",
        )
        .reset_index()
    )

    for condition in TACTILE_CONDITIONS:
        if condition not in wide:
            wide[condition] = np.nan

    wide["se_point_ms"] = np.sqrt(wide["T"] ** 2 + wide["VT"] ** 2)

    if by_speed:
        floors = wide[["subject", "speed", "position_rank", "se_point_ms"]]
    else:
        # Averaging the two speeds halves the standard error of the average,
        # so the floor of the collapsed profile is not the floor of a cell.
        floors = (
            wide
            .groupby(["subject", "position_rank"], as_index=False)["se_point_ms"]
            .agg(lambda values: float(
                np.sqrt(np.nansum(np.square(values))) / max(len(values), 1)
            ))
        )

    rows = []

    for key, one in subject.groupby(keys[:-1]):
        one = one.sort_values("position_rank").dropna(subset=["facilitation_ms"])
        values = one["facilitation_ms"].to_numpy(dtype=float)
        ranks = one["position_rank"].to_numpy(dtype=float)

        if len(values) < 3:
            continue

        key_tuple = key if isinstance(key, tuple) else (key,)
        record = dict(zip(keys[:-1], key_tuple))

        # Direction changes across the profile.
        steps = np.sign(np.diff(values))
        steps = steps[steps != 0]
        reversals = int(np.sum(steps[1:] != steps[:-1])) if len(steps) > 1 else 0

        if len(values) >= 3 and np.std(values) > 0:
            rho, p_value = stats.spearmanr(ranks, values)
        else:
            rho, p_value = np.nan, np.nan

        matching = floors
        for column, value in record.items():
            matching = matching[matching[column] == value]

        floor = float(matching["se_point_ms"].median()) if len(matching) else np.nan
        span = float(np.ptp(values))

        record.update({
            "n_distances": int(len(values)),
            "range_ms": span,
            "floor_ms": floor,
            "range_over_floor": span / floor if floor and np.isfinite(floor) else np.nan,
            "reversals": reversals,
            "max_reversals": max(len(values) - 2, 0),
            "rho": rho,
            "rho_p": p_value,
            "mean_ms": float(np.mean(values)),
        })

        record["fittable"] = bool(
            np.isfinite(record["range_over_floor"])
            and record["range_over_floor"] >= 2.0
            and reversals <= 2
        )

        rows.append(record)

    return pd.DataFrame(rows).round(2)

def plot_profile_legibility(
    pps,
    statistic="median",
    stage="after",
    baseline="fastest_tactile",
    n_boot=N_BOOTSTRAP,
    n_columns=4,
    save_path=None,
):
    """One panel per participant: the profile against its own error bars.

    Read this before fitting anything. The band is plus and minus one
    standard error of a single facilitation point, computed from the two
    cells that produced it. Where the whole profile fits inside the band,
    there is no shape for a sigmoid to find, and any curve the optimiser
    returns is fitting noise. The annotation gives the peak-to-trough range
    against that floor, and the number of direction changes: a sigmoid is
    monotone, so more than two reversals on seven points rules it out
    whatever the fit reports.

    Participants that fail the screen are titled in muted grey.
    """
    ranks = _ranks(pps)

    subject, group = facilitation_profile(
        pps, statistic=statistic, stage=stage, by_speed=False, baseline=baseline,
    )
    legibility = profile_legibility(
        pps, statistic=statistic, stage=stage, baseline=baseline,
        n_boot=n_boot, by_speed=False,
    )

    subjects = sorted(subject["subject"].dropna().unique())
    n_panels = len(subjects) + 1
    n_rows = int(np.ceil(n_panels / n_columns))

    fig, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(3.6 * n_columns, 3.0 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )
    flat_axes = axes.ravel()

    for ax, name in zip(flat_axes, subjects):
        one = subject[subject["subject"] == name].sort_values("position_rank")
        row = legibility[legibility["subject"] == name]
        floor = float(row["floor_ms"].iloc[0]) if len(row) else np.nan
        fittable = bool(row["fittable"].iloc[0]) if len(row) else False

        if np.isfinite(floor):
            ax.fill_between(
                one["position_rank"],
                one["facilitation_ms"] - floor,
                one["facilitation_ms"] + floor,
                color=style._MUTED, alpha=0.18, linewidth=0,
            )

        ax.plot(
            one["position_rank"], one["facilitation_ms"],
            color=style.INK if fittable else style._MUTED,
            marker="o", markersize=4.5, linewidth=2.0,
        )
        ax.axhline(0, linestyle=":", linewidth=0.9, color=style._MUTED)

        if len(row):
            row = row.iloc[0]
            ax.text(
                0.03, 0.97,
                f"range {row['range_ms']:.0f} ms\n"
                f"floor {row['floor_ms']:.0f} ms\n"
                f"ratio {row['range_over_floor']:.1f}\n"
                f"reversals {int(row['reversals'])}/{int(row['max_reversals'])}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7, color=style.INK, linespacing=1.35,
            )

        ax.set_title(
            str(name),
            loc="left",
            color=style.INK if fittable else style._MUTED,
            fontsize=9.5,
        )
        _distance_axis(ax, ranks)
        ax.grid(axis="y", color=style._RULE, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        style.clean(ax)

    # Final panel: the group profile on the same scale, for reference.
    ax = flat_axes[len(subjects)]
    group = group.sort_values("position_rank")

    ax.fill_between(
        group["position_rank"],
        group["group_mean"] - group["group_sem"],
        group["group_mean"] + group["group_sem"],
        color=style.CONTROL, alpha=0.20, linewidth=0,
    )
    ax.plot(
        group["position_rank"], group["group_mean"],
        color=style.CONTROL, marker="o", markersize=5, linewidth=2.4,
    )
    ax.axhline(0, linestyle=":", linewidth=0.9, color=style._MUTED)

    n_fittable = int(legibility["fittable"].sum()) if len(legibility) else 0
    ax.text(
        0.03, 0.97,
        f"{n_fittable} of {len(subjects)} profiles\nclear the screen",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7.5, color=style.INK, linespacing=1.35,
    )
    ax.set_title("Group (band = SEM)", loc="left", fontsize=9.5)
    _distance_axis(ax, ranks)
    ax.grid(axis="y", color=style._RULE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    style.clean(ax)

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    for row_index in range(n_rows):
        axes[row_index, 0].set_ylabel("T - VT (ms)")

    fig.suptitle(
        "Is there a shape to fit? Facilitation against its own error bars",
        x=0.02, ha="left", y=1.0, color=style.INK,
    )
    fig.tight_layout()
    _save(fig, save_path)

    return fig, legibility

MANUAL_REASON = "manual_review"

if MANUAL_REASON not in REASON_ORDER:
    REASON_ORDER = REASON_ORDER + [MANUAL_REASON]

def _block_column(data):
    return next(
        (
            column
            for column in ["block", "block_number", "block_index"]
            if column in data.columns
        ),
        None,
    )

def trial_diagnostics(pps, stage="after", whisker=1.5):
    """Every trial with the context needed to judge it.

    Adds, per trial:

        cell_n              trials in its own cell
        cell_median_ms      the estimate that trial contributes to
        deviation_ms        trial minus that estimate
        robust_z            deviation divided by the cell MAD, scaled to be
                            comparable with an SD. The MAD is used rather
                            than the SD because at four trials a single
                            extreme value inflates the SD enough to hide
                            itself
        influence_ms        how far the cell estimate moves if this trial is
                            left out. This is the number that matters: a
                            trial can look extreme and change nothing, or
                            look ordinary and carry the cell
        beyond_fence        outside the 1.5 IQR fence of its distance x
                            condition cell, pooled across participants
        session_rt_median   the participant's own median in that session, so
                            a slow trial can be read against their own pace
        block_rt_median     the same for its block, which is where a bad
                            stretch shows up
    """
    data = trials_at_stage(pps, stage=stage).copy()

    if data.empty:
        return data

    keys = _cell_keys(data)
    grouped = data.groupby(keys, dropna=False)["rt_ms"]

    data["cell_n"] = grouped.transform("size")
    data["cell_median_ms"] = grouped.transform("median")
    data["deviation_ms"] = data["rt_ms"] - data["cell_median_ms"]

    # MAD, scaled so that a robust_z reads on the same scale as an SD-based z
    # for normally distributed data.
    mad = grouped.transform(
        lambda values: float(np.median(np.abs(values - np.median(values))))
    )
    data["cell_mad_ms"] = mad
    data["robust_z"] = np.where(
        mad > 0, data["deviation_ms"] / (1.4826 * mad), np.nan
    )

    # Leave-one-out influence on the cell estimate.
    def _influence(values):
        values = values.to_numpy(dtype=float)
        full = np.median(values)

        if len(values) < 2:
            return pd.Series(np.zeros(len(values)), index=values and None)

        return pd.Series(
            [np.median(np.delete(values, i)) - full for i in range(len(values))]
        )

    influence = []
    for _, cell in data.groupby(keys, dropna=False, sort=False):
        values = cell["rt_ms"].to_numpy(dtype=float)
        full = float(np.median(values))

        if len(values) < 2:
            influence.append(pd.Series(np.zeros(len(values)), index=cell.index))
        else:
            influence.append(pd.Series(
                [float(np.median(np.delete(values, i))) - full
                 for i in range(len(values))],
                index=cell.index,
            ))

    data["influence_ms"] = pd.concat(influence).reindex(data.index)
    data["abs_influence_ms"] = data["influence_ms"].abs()

    # Pooled fence, the same rule the boxplots draw.
    fence_keys = ["position_rank", "sensory_condition"]
    q1 = data.groupby(fence_keys)["rt_ms"].transform(lambda v: v.quantile(0.25))
    q3 = data.groupby(fence_keys)["rt_ms"].transform(lambda v: v.quantile(0.75))
    iqr = q3 - q1
    data["lower_fence_ms"] = q1 - whisker * iqr
    data["upper_fence_ms"] = q3 + whisker * iqr
    data["beyond_fence"] = (
        (data["rt_ms"] < data["lower_fence_ms"])
        | (data["rt_ms"] > data["upper_fence_ms"])
    )

    # The participant's own pace, for context.
    if "session" in data.columns:
        data["session_rt_median"] = data.groupby(
            ["subject", "session"]
        )["rt_ms"].transform("median")

    block = _block_column(data)

    if block is not None:
        group = ["subject", block]
        if "session" in data.columns:
            group = ["subject", "session", block]

        data["block_rt_median"] = data.groupby(group)["rt_ms"].transform("median")
        data["block_trial"] = data.groupby(group).cumcount() + 1

    preferred = [
        column
        for column in [
            "subject", "session", block, "block_trial",
            _trial_order_column(data), "speed", "sensory_condition",
            "position_rank", "rt_ms", "cell_n", "cell_median_ms",
            "deviation_ms", "robust_z", "influence_ms", "abs_influence_ms",
            "beyond_fence", "block_rt_median", "session_rt_median",
        ]
        if column is not None and column in data.columns
    ]
    remaining = [column for column in data.columns if column not in preferred]

    return data[preferred + remaining]

def flag_trials(
    pps,
    stage="after",
    min_abs_z=3.0,
    min_influence_ms=40.0,
    min_cell_n_for_influence=5,
    subjects=None,
    whisker=1.5,
):
    """Candidates for review, ranked by how much they move their own cell.

    A trial is a candidate when it is far from its cell in robust terms, when
    it falls outside the pooled fence for its distance and condition, or when
    removing it would move the cell estimate by more than
    ``min_influence_ms``. The three catch different things and a trial only
    has to meet one.

    The influence criterion is applied only in cells of at least
    ``min_cell_n_for_influence`` trials. Below that it is uninformative: in a
    four-trial cell the median sits between the second and third value, so
    dropping any one of the four moves it by construction, and the criterion
    would flag entire cells rather than individual trials.

    These are candidates, not exclusions. Look at them in
    ``plot_subject_trials`` before deciding, and use ``exclude_trials`` to
    record whatever you decide.
    """
    data = trial_diagnostics(pps, stage=stage, whisker=whisker)

    if data.empty:
        return data

    if subjects is not None:
        data = data[data["subject"].isin(_as_list(subjects))]

    extreme = data["robust_z"].abs() >= min_abs_z
    influential = (
        (data["abs_influence_ms"] >= min_influence_ms)
        & (data["cell_n"] >= min_cell_n_for_influence)
    )
    outside = data["beyond_fence"]

    flagged = data[extreme | influential | outside].copy()

    flagged["flag_reason"] = np.select(
        [extreme[flagged.index] & influential[flagged.index],
         extreme[flagged.index],
         influential[flagged.index]],
        ["extreme and influential", "extreme within cell", "moves the cell"],
        default="outside pooled fence",
    )

    # The index is the trial label in ``pps``, which is what
    # ``exclude_trials`` needs, so it is deliberately preserved rather than
    # reset.
    return flagged.sort_values("abs_influence_ms", ascending=False)

def exclude_trials(pps, trials, reason=MANUAL_REASON, note=None):
    """Mark named trials unusable, and record that the decision was manual.

    ``trials`` is either a list of index labels from ``pps``, or a frame with
    those labels as its index, which is what ``flag_trials`` returns. Only
    the named trials change. The reason is written as ``manual_review`` and
    never as one of the preregistered reasons, so a hand-made exclusion can
    always be counted separately and reversed.

    Returns a new frame; ``pps`` is not modified.
    """
    if isinstance(trials, pd.DataFrame):
        labels = list(trials.index)
    else:
        labels = list(_as_list(trials) or [])

    updated = pps.copy()
    missing = [label for label in labels if label not in updated.index]

    if missing:
        raise KeyError(
            f"{len(missing)} trial label(s) are not in this table: {missing[:5]}"
        )

    if "manual_note" not in updated.columns:
        updated["manual_note"] = ""

    updated.loc[labels, "usable"] = False
    updated.loc[labels, "exclusion_reason"] = reason

    if note:
        updated.loc[labels, "manual_note"] = note

    print(
        f"Excluded {len(labels)} trial(s) as '{reason}'"
        + (f": {note}" if note else "")
    )
    print(
        f"Usable now {int(updated['usable'].sum())}, "
        f"was {int(pps['usable'].sum())}."
    )

    return updated

def cell_trials(pps, subject, positions=None, stage="after", conditions=None):
    """Every trial in the named cells, laid out so a cell can be read directly.

    One row per cell, the individual RTs in the order they were delivered,
    and the estimate they produce. This is the view for the question "what is
    that cell actually made of", which the pooled plots cannot answer: a cell
    estimate is a summary of three or four numbers, and at that size the only
    way to judge it is to see them.

    ``trials`` holds the raw values, ``estimate_ms`` the median they produce,
    and ``spread_ms`` their range, which at four trials says more about the
    estimate's reliability than a standard deviation does.
    """
    data = trials_at_stage(pps, stage=stage, subjects=[subject])

    if positions is not None:
        data = data[data["position_rank"].isin(_as_list(positions))]

    if conditions is not None:
        data = data[data["sensory_condition"].isin(_as_list(conditions))]

    if data.empty:
        return pd.DataFrame()

    order = _trial_order_column(data)
    block = _block_column(data)

    keys = [
        column
        for column in ["speed", "sensory_condition", "position_rank"]
        if column in data.columns
    ]

    rows = []

    for key, cell in data.groupby(keys, dropna=False):
        cell = cell.sort_values(order) if order else cell
        values = cell["rt_ms"].to_numpy(dtype=float)

        record = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        record.update({
            "n": int(len(values)),
            "trials": np.round(values, 1).tolist(),
            "estimate_ms": float(np.median(values)),
            "spread_ms": float(np.ptp(values)) if len(values) > 1 else 0.0,
        })

        if block is not None:
            record["blocks"] = cell[block].tolist()

        rows.append(record)

    return (
        pd.DataFrame(rows)
        .sort_values(["sensory_condition", "speed", "position_rank"])
        .reset_index(drop=True)
    )

def plot_cell_trials(
    pps,
    subject,
    positions=None,
    stage="after",
    save_path=None,
):
    """Every trial at each distance, with the cell estimate marked.

    Two panels, tactile-only and visuo-tactile, distance along the x axis.
    Each trial is a point; the horizontal bar is the median that cell
    contributes to the facilitation profile. Where the points at one distance
    sit far from their neighbours, the estimate there is carried by however
    few trials produced it, and the bar shows how far that pulled it.

    This is the view to use when one distance looks wrong in a profile: it
    shows whether the cell holds one deviant trial or is uniformly displaced.
    """
    ranks = _ranks(pps)

    if positions is not None:
        ranks = [rank for rank in ranks if rank in _as_list(positions)]

    data = trials_at_stage(pps, stage=stage, subjects=[subject])
    data = data[data["position_rank"].isin(ranks)]

    if data.empty:
        print(f"No usable trials for {subject}.")
        return None, pd.DataFrame()

    fig, axes = plt.subplots(
        1, len(TACTILE_CONDITIONS),
        figsize=(6.4 * len(TACTILE_CONDITIONS), 4.4),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    rng = np.random.default_rng(31)
    offsets = {"slow": -0.16, "fast": 0.16}

    for ax, condition in zip(axes, TACTILE_CONDITIONS):
        colour = CONDITION_COLOUR[condition]

        for speed, alpha, marker in [("slow", 0.45, "s"), ("fast", 1.0, "o")]:
            for rank in ranks:
                values = (
                    data.loc[
                        data["position_rank"].eq(rank)
                        & data["sensory_condition"].eq(condition)
                        & data["speed"].eq(speed),
                        "rt_ms",
                    ].dropna().to_numpy(dtype=float)
                )

                if not values.size:
                    continue

                centre = rank + offsets[speed]

                ax.scatter(
                    centre + rng.normal(0, 0.035, values.size), values,
                    s=30, color=colour, alpha=0.35 * alpha,
                    marker=marker, linewidth=0, zorder=2,
                )

                # The estimate this cell contributes, and its trial count.
                ax.plot(
                    [centre - 0.11, centre + 0.11],
                    [np.median(values)] * 2,
                    color=_darker(colour), linewidth=2.4, alpha=alpha,
                    solid_capstyle="butt", zorder=4,
                )
                ax.annotate(
                    f"{len(values)}",
                    (centre, np.median(values)),
                    xytext=(0, -11), textcoords="offset points",
                    ha="center", fontsize=6.5, color=style._MUTED,
                )

            ax.plot(
                [], [], marker=marker, linestyle="none", color=colour,
                alpha=alpha, label=speed.capitalize(),
            )

        _finish(
            ax, CONDITION_LABELS[condition], "RT (ms)", ranks=ranks,
        )
        ax.legend(frameon=False, fontsize=8, title="Speed")

    fig.suptitle(
        f"{subject}: every trial per cell "
        "(bar = the median that cell contributes, number = trials behind it)",
        x=0.02, ha="left", y=1.01, color=style.INK,
    )
    fig.tight_layout()
    _save(fig, save_path)

    return fig, cell_trials(pps, subject, positions=positions, stage=stage)

def plot_subject_trials(
    pps,
    subject,
    stage="after",
    min_abs_z=3.0,
    min_influence_ms=40.0,
    min_cell_n_for_influence=5,
    label_flagged=True,
    save_path=None,
):
    """One participant's trials, block by block, with the candidates ringed.

    Each panel is one block, trials in the order they were delivered. Grey is
    tactile-only, blue visuo-tactile. The horizontal line is that block's
    median RT, so a block that ran slow is visible as a whole rather than as
    a set of individual outliers. Ringed points are the candidates from
    ``flag_trials``, labelled with their index so they can be passed straight
    to ``exclude_trials``.

    The point of the layout is that a bad stretch and a bad trial look
    different: one shifts the block line, the other sits above it.
    """
    data = trial_diagnostics(pps, stage=stage)
    data = data[data["subject"] == subject]

    if data.empty:
        print(f"No usable trials for {subject}.")
        return None, pd.DataFrame()

    block = _block_column(data)
    order = _trial_order_column(data)

    if block is None:
        data = data.assign(_block="all")
        block = "_block"

    blocks = sorted(data[block].dropna().unique())
    sessions = sorted(data["session"].dropna().unique()) if "session" in data else [None]

    flagged = flag_trials(
        pps, stage=stage, min_abs_z=min_abs_z,
        min_influence_ms=min_influence_ms,
        min_cell_n_for_influence=min_cell_n_for_influence,
        subjects=[subject],
    )
    flagged_labels = set(flagged.index) & set(data.index)

    fig, axes = plt.subplots(
        len(sessions), len(blocks),
        figsize=(3.7 * len(blocks), 3.2 * len(sessions)),
        sharey=True, squeeze=False,
    )

    for row, session in enumerate(sessions):
        for column, one_block in enumerate(blocks):
            ax = axes[row, column]

            panel = data[data[block] == one_block]
            if session is not None:
                panel = panel[panel["session"] == session]

            if panel.empty:
                ax.set_visible(False)
                continue

            x = panel[order] if order in panel else panel.get(
                "block_trial", pd.Series(range(len(panel)), index=panel.index)
            )

            for condition in TACTILE_CONDITIONS:
                rows = panel[panel["sensory_condition"] == condition]

                if rows.empty:
                    continue

                ax.scatter(
                    x.loc[rows.index], rows["rt_ms"],
                    s=26, color=CONDITION_COLOUR[condition],
                    marker="s" if condition == "T" else "o",
                    alpha=0.75, linewidth=0,
                    label=CONDITION_LABELS[condition],
                )

            ax.axhline(
                float(panel["rt_ms"].median()),
                color=style.INK, linestyle="--", linewidth=1.2, alpha=0.7,
            )

            ringed = panel.loc[panel.index.intersection(flagged_labels)]

            if not ringed.empty:
                ax.scatter(
                    x.loc[ringed.index], ringed["rt_ms"],
                    s=110, facecolors="none", edgecolors=style.INK,
                    linewidths=1.3, zorder=5,
                )

                if label_flagged:
                    for label, row_values in ringed.iterrows():
                        ax.annotate(
                            str(label),
                            (x.loc[label], row_values["rt_ms"]),
                            xytext=(5, 5), textcoords="offset points",
                            fontsize=6.5, color=style.INK,
                        )

            title = f"Block {one_block}"
            if session is not None and len(sessions) > 1:
                title = f"Session {session}, block {one_block}"

            ax.set_title(title, loc="left", fontsize=9.5)
            ax.set_xlabel("Trial order")
            ax.grid(axis="y", color=style._RULE, linewidth=0.6, alpha=0.7)
            ax.set_axisbelow(True)
            style.clean(ax)

        axes[row, 0].set_ylabel("RT (ms)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            dict(zip(labels, handles)).values(),
            dict(zip(labels, handles)).keys(),
            frameon=False, ncol=2, loc="upper right",
            bbox_to_anchor=(1.0, 1.03), fontsize=8,
        )

    fig.suptitle(
        f"{subject}: every trial, in the order it was delivered "
        f"(dashed = block median, ringed = candidate for review)",
        x=0.02, ha="left", y=1.02, color=style.INK,
    )
    fig.tight_layout()
    _save(fig, save_path)

    return fig, flagged

def compare_retention(pps, subjects=None):
    """Did the two participants lose trials at the same rate, and for what?

    Run this before reading any RT difference. A patient who lost twice as
    many trials as their control is not comparable to that control on RT
    alone, whatever the RTs say, and the reason for the loss usually says
    which. Missed responses point at attention or motor difficulty; slow
    responses at the RT window catching a genuinely slower participant.
    """
    tactile = _select_subjects(
        pps[pps["sensory_condition"].isin(TACTILE_CONDITIONS)], subjects
    )

    counts = (
        pd.crosstab(tactile["subject"], tactile["exclusion_reason"])
        .reindex(columns=REASON_ORDER, fill_value=0)
    )

    reasons = [column for column in REASON_ORDER if column in counts.columns]
    counts["n_tactile"] = counts[reasons].sum(axis=1)
    counts["percent_kept"] = (
        100 * counts["usable"] / counts["n_tactile"]
    ).round(1)

    # Which reasons actually fired, so the table stays readable.
    fired = [
        reason for reason in reasons
        if reason != "usable" and counts[reason].sum() > 0
    ]

    return counts[["n_tactile", "usable"] + fired + ["percent_kept"]].reset_index()

def compare_cells(pps, subjects=None, stage="after", minimum=MIN_CELL_N):
    """Usable trials per cell for each participant, side by side.

    A cell that is thin in one participant and full in the other is the case
    to know about: their two estimates are not equally precise, so the
    difference between them carries that asymmetry.
    """
    counts = cell_counts(pps, stage=stage)
    counts = _select_subjects(counts, subjects)

    if counts.empty:
        return pd.DataFrame()

    wide = counts.pivot_table(
        index=["speed", "sensory_condition", "position_rank"],
        columns="subject",
        values="n_trials",
        aggfunc="sum",
    ).reset_index()

    people = [
        column for column in wide.columns
        if column not in {"speed", "sensory_condition", "position_rank"}
    ]

    wide["min_n"] = wide[people].min(axis=1)
    wide["thin_in_either"] = wide["min_n"] < minimum

    return wide.sort_values(
        ["sensory_condition", "speed", "position_rank"]
    ).reset_index(drop=True)

def plot_pair_inspection(
    pps,
    subjects,
    labels=None,
    statistic="median",
    whisker=1.5,
    save_path=None,
):
    """Two participants, before and after exclusions, on shared scales.

    Rows are participant by stage, ordered so each person's before and after
    sit adjacent: the change from exclusions is read between neighbouring
    rows, and the two people are compared across the block. Every panel
    shares one RT axis, so a difference in height is a difference in reaction
    time and not a redrawn axis.

        A. Trial composition, with the count and the percentage kept.
        B. RT by condition, every trial drawn, the line the median and the
           filled circle the mean.
        C. RT by distance and condition.
        D. Facilitation across distance, one line per speed.

    Read the composition panel first. If the two participants kept different
    proportions of their trials, the RT panels are comparing two differently
    filtered datasets, and that has to be said before anything else is read
    from them.
    """
    people = _as_list(subjects)

    if len(people) < 2:
        raise ValueError("plot_pair_inspection needs at least two participants")

    labels = list(labels) if labels else list(people)
    ranks = _ranks(pps)
    limits = _observed_range(_select_subjects(pps, people))

    # Person-major, so each person's before and after sit adjacent and the
    # change is read without scanning past the other participant.
    rows = [(stage, person) for person in people for stage in STAGES]

    fig, axes = plt.subplots(
        len(rows), 4,
        figsize=(17, 3.1 * len(rows)),
        squeeze=False,
    )

    rng = np.random.default_rng(19)

    for row, (stage, person) in enumerate(rows):
        ax_composition, ax_condition, ax_distance, ax_fac = axes[row]
        data = trials_at_stage(pps, stage=stage, subjects=[person])
        label = labels[people.index(person)]

        # --- A. Composition ------------------------------------------
        source = _select_subjects(pps, [person])
        if stage == "after":
            source = source[
                source["usable"] | (source["sensory_condition"] == "V")
            ]

        present = [
            condition for condition in ["V", "T", "VT"]
            if condition in source["sensory_condition"].unique()
        ]
        counts = source["sensory_condition"].value_counts().reindex(
            present, fill_value=0
        )
        colours = dict(CONDITION_COLOUR)
        colours.setdefault("V", style._MUTED)

        bars = ax_composition.bar(
            np.arange(len(present)), counts.to_numpy(dtype=float),
            width=0.62, color=[colours[c] for c in present], linewidth=0,
        )
        for bar, condition in zip(bars, present):
            ax_composition.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{int(counts[condition])}",
                ha="center", va="bottom", fontsize=7.5, color=style.INK,
            )

        ax_composition.set_xticks(np.arange(len(present)))
        ax_composition.set_xticklabels(present)
        ax_composition.set_ylabel(
            f"{label}\n{STAGE_STYLE[stage]['label']}\n\nTrials"
        )

        # --- B. RT by condition --------------------------------------
        for position, condition in enumerate(TACTILE_CONDITIONS, start=1):
            values = (
                data.loc[data["sensory_condition"] == condition, "rt_ms"]
                .dropna().to_numpy(dtype=float)
            )

            if not values.size:
                continue

            colour = CONDITION_COLOUR[condition]
            _boxplot(ax_condition, values, position, colour, whisker=whisker)
            _scatter_trials(
                ax_condition, values, position, colour, rng, spread=0.05, size=8
            )
            ax_condition.scatter(
                position,
                values.mean(),
                marker="o",
                s=30,
                facecolors=colour,
                edgecolors="none",
                linewidths=0,
                zorder=6,
            )

        ax_condition.set_xticks([1, 2])
        ax_condition.set_xticklabels(TACTILE_CONDITIONS)
        ax_condition.set_ylim(limits)
        ax_condition.set_ylabel("RT (ms)")

        # --- C. RT by distance and condition -------------------------
        offsets = {"T": -0.19, "VT": 0.19}

        for condition in TACTILE_CONDITIONS:
            colour = CONDITION_COLOUR[condition]

            for rank in ranks:
                values = (
                    data.loc[
                        data["position_rank"].eq(rank)
                        & data["sensory_condition"].eq(condition),
                        "rt_ms",
                    ].dropna().to_numpy(dtype=float)
                )

                if not values.size:
                    continue

                centre = rank + offsets[condition]
                _boxplot(
                    ax_distance, values, centre, colour,
                    width=0.31, whisker=whisker,
                )
                _scatter_trials(
                    ax_distance, values, centre, colour, rng,
                    spread=0.03, size=7,
                )

            ax_distance.plot(
                [], [], marker="s", linestyle="none", color=colour,
                label=CONDITION_LABELS[condition],
            )

        ax_distance.set_xlim(min(ranks) - 0.6, max(ranks) + 0.6)
        ax_distance.set_ylim(limits)
        ax_distance.set_ylabel("RT (ms)")

        if row == 0:
            ax_distance.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")

        _distance_axis(ax_distance, ranks)

        # --- D. Facilitation -----------------------------------------
        one_person = _select_subjects(pps, [person])
        subject_profile, _ = facilitation_profile(
            one_person, statistic=statistic, stage=stage, by_speed=True
        )

        for speed in SPEEDS:
            line = subject_profile[
                subject_profile["speed"] == speed
            ].sort_values("position_rank")

            if line.empty:
                continue

            ax_fac.plot(
                line["position_rank"], line["facilitation_ms"],
                color=style.SPEED[speed], marker="o", markersize=5,
                linewidth=2.2, label=speed.capitalize(),
            )

        ax_fac.axhline(0, linestyle=":", linewidth=1.0, color=style._MUTED)
        ax_fac.set_ylabel("T - VT (ms)")

        if row == 0:
            ax_fac.legend(frameon=False, fontsize=8, title="Speed")

        _distance_axis(ax_fac, ranks)

        for ax in axes[row]:
            ax.grid(axis="y", color=style._RULE, linewidth=0.6, alpha=0.7)
            ax.set_axisbelow(True)
            style.clean(ax)

    titles = [
        "A. Trial composition",
        "B. RT by condition",
        "C. RT by distance and condition",
        "D. Facilitation by distance",
    ]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, loc="left", pad=8, fontsize=9.5)

    # Facilitation and composition are the panels whose range is not fixed in
    # advance; hold each common across every row so a difference between the
    # two people reads as a difference and not as a redrawn axis.
    for column in (0, 3):
        low = min(axes[row, column].get_ylim()[0] for row in range(len(rows)))
        high = max(axes[row, column].get_ylim()[1] for row in range(len(rows)))
        for row in range(len(rows)):
            axes[row, column].set_ylim(low, high)

    fig.suptitle(
        "Two participants before and after exclusions: "
        + " against ".join(labels),
        x=0.02, ha="left", y=0.998, color=style.INK,
    )
    fig.tight_layout()
    _save(fig, save_path)

    return fig

def pair_report(pps, subjects, labels=None, statistic="median", n_boot=N_BOOTSTRAP):
    """Everything needed to judge whether the pair is comparable.

    Returns the retention table, the per-cell counts, the two facilitation
    profiles and their difference, and the precision of each. The difference
    is the quantity of interest and the precision is what says whether it can
    be read; both are returned so neither is quoted without the other.
    """
    people = _as_list(subjects)
    labels = list(labels) if labels else list(people)

    subset = _select_subjects(pps, people)

    profiles_by_person = {}
    for person in people:
        one, _ = facilitation_profile(
            _select_subjects(pps, [person]), statistic=statistic,
            stage="after", by_speed=True,
        )
        profiles_by_person[person] = one.set_index(["speed", "position_rank"])[
            "facilitation_ms"
        ]

    difference = pd.DataFrame(profiles_by_person)
    difference.columns = labels

    if len(labels) == 2:
        difference["difference"] = difference[labels[0]] - difference[labels[1]]

    return {
        "retention": compare_retention(pps, subjects=people),
        "cells": compare_cells(pps, subjects=people),
        "facilitation": difference.round(1).reset_index(),
        "precision": {
            label: precision(
                _select_subjects(pps, [person]), statistic=statistic,
                n_boot=n_boot,
            )
            for person, label in zip(people, labels)
        },
        "subjects": people,
        "labels": labels,
    }

def plot_baselines(
    pps,
    statistic="median",
    stage="after",
    degree=1,
    save_path=None,
):
    """What each tactile baseline actually is, and what it does downstream.

    Four ways of turning tactile-only trials into a baseline, drawn on the
    same axes so the difference between them is visible rather than
    described:

        per-cell        the tactile median at that distance, on its own.
                        Seven independent estimates from four trials each.
                        Follows the data exactly, including its noise.
        fitted          tactile RT regressed on distance, using every tactile
                        trial in the speed context. Two parameters from 28
                        observations. Keeps the delay-dependence, drops the
                        per-cell sampling noise with it.
        mean            one value per participant and speed: the average
                        across distances. Flat by construction, so any
                        delay-dependence is discarded rather than modelled.
        fastest         the fastest distance's estimate. Also flat, and the
                        most conservative, since visuo-tactile must beat the
                        participant's best unimodal condition.

    Top row: the tactile-only profile with the four baselines on it. Where
    the fitted line and the per-cell points diverge is the sampling noise the
    fit removes; where the fitted line and the flat lines diverge is the
    delay effect the flat baselines discard.

    Bottom row: the facilitation profile each baseline produces. All four use
    the same visuo-tactile medians, so every difference between the curves
    comes from the baseline alone.
    """
    ranks = _ranks(pps)

    fig, axes = plt.subplots(
        2, len(SPEEDS),
        figsize=(6.6 * len(SPEEDS), 8.0),
        sharex=True, squeeze=False,
    )

    fitted = smooth_baseline(pps, stage=stage, degree=degree)

    styles = {
        "per-cell": {"colour": style.SLOW, "linestyle": "--", "marker": "s"},
        "fitted": {"colour": style.PATIENT, "linestyle": "-", "marker": None},
        "mean": {"colour": style.FAST, "linestyle": ":", "marker": None},
        "fastest": {"colour": style._MUTED, "linestyle": "-.", "marker": None},
    }

    for column, speed in enumerate(SPEEDS):
        ax_rt, ax_fac = axes[0, column], axes[1, column]

        # --- The tactile-only profile, and the four baselines -----------
        _, tactile_group = rt_profile(
            pps, statistic=statistic, stage=stage, by_speed=True
        )
        line = tactile_group[
            (tactile_group["sensory_condition"] == "T")
            & (tactile_group["speed"] == speed)
        ].sort_values("position_rank")

        if not line.empty:
            ax_rt.plot(
                line["position_rank"], line["group_mean"],
                color=styles["per-cell"]["colour"],
                linestyle=styles["per-cell"]["linestyle"],
                marker="s", markerfacecolor="white", markersize=6,
                linewidth=2.0, label="Per-cell (delay-matched)", zorder=4,
            )

        if not fitted.empty:
            curve = (
                fitted[fitted["speed"] == speed]
                .groupby("position_rank", as_index=False)["baseline_smooth_ms"]
                .mean()
                .sort_values("position_rank")
            )
            if not curve.empty:
                ax_rt.plot(
                    curve["position_rank"], curve["baseline_smooth_ms"],
                    color=styles["fitted"]["colour"], linewidth=2.6,
                    label="Fitted against distance", zorder=5,
                )

        # The two flat baselines, computed the same way the corrections do.
        cells = facilitation_table(pps, statistic=statistic, stage=stage)
        cells = cells[cells["speed"] == speed]

        for name, aggregation in [("mean", "mean"), ("fastest", "min")]:
            per_subject = cells.groupby("subject")["T"].agg(aggregation)
            level = float(per_subject.mean())

            ax_rt.axhline(
                level, color=styles[name]["colour"],
                linestyle=styles[name]["linestyle"], linewidth=2.0,
                label=(
                    "Mean across distances" if name == "mean"
                    else "Fastest distance"
                ),
                zorder=3,
            )

        _finish(
            ax_rt, f"{speed.capitalize()}: tactile-only and its baselines",
            f"{statistic.capitalize()} tactile RT (ms)", ranks=ranks,
        )
        ax_rt.legend(frameon=False, fontsize=8)

        # --- The facilitation each baseline produces --------------------
        for name, baseline, correction in [
            ("per-cell", "delay_matched", None),
            ("fitted", "smooth", None),
        ]:
            _, group = facilitation_profile(
                pps, statistic=statistic, stage=stage, by_speed=True,
                baseline=baseline,
            )
            one = group[group["speed"] == speed].sort_values("position_rank")

            if one.empty:
                continue

            ax_fac.plot(
                one["position_rank"], one["group_mean"],
                color=styles[name]["colour"],
                linestyle=styles[name]["linestyle"],
                marker=styles[name]["marker"], markersize=6,
                markerfacecolor="white" if name == "per-cell" else None,
                linewidth=2.6 if name == "fitted" else 2.0,
                label=(
                    "Per-cell baseline" if name == "per-cell"
                    else "Fitted baseline"
                ),
                zorder=5 if name == "fitted" else 4,
            )

        # The flat baselines produce facilitation curves that are the VT
        # profile inverted, shifted by a constant. Drawn for comparison.
        for name, aggregation in [("mean", "mean"), ("fastest", "min")]:
            values = cells.copy()
            level = values.groupby("subject")["T"].transform(aggregation)
            values["facilitation"] = level - values["VT"]

            per_subject = values.groupby(
                ["subject", "position_rank"], as_index=False
            )["facilitation"].mean()
            one = (
                per_subject
                .groupby("position_rank", as_index=False)["facilitation"]
                .mean()
                .sort_values("position_rank")
            )

            ax_fac.plot(
                one["position_rank"], one["facilitation"],
                color=styles[name]["colour"],
                linestyle=styles[name]["linestyle"], linewidth=2.0,
                label=(
                    "Mean baseline" if name == "mean" else "Fastest baseline"
                ),
                zorder=3,
            )

        ax_fac.axhline(0, linestyle=":", linewidth=1.0, color=style._MUTED)
        _finish(
            ax_fac, f"{speed.capitalize()}: the facilitation each one gives",
            "T - VT (ms)", ranks=ranks,
        )
        ax_fac.legend(frameon=False, fontsize=8)

    # The two rows carry different quantities, so each is shared across
    # speeds only, not down the column.
    for row in range(2):
        low = min(axes[row, c].get_ylim()[0] for c in range(len(SPEEDS)))
        high = max(axes[row, c].get_ylim()[1] for c in range(len(SPEEDS)))
        for c in range(len(SPEEDS)):
            axes[row, c].set_ylim(low, high)

    fig.suptitle(
        "Four tactile baselines, and the facilitation each produces",
        x=0.02, ha="left", y=1.0, color=style.INK,
    )
    fig.tight_layout()
    _save(fig, save_path)

    return fig

COLLISION_SEQUENCE_ORDER = ["slow-slow", "slow-fast", "fast-fast", "fast-slow"]

COLLISION_TRIAL_TYPE_ORDER = ["clear_hit", "near_hit", "near_miss", "clear_miss"]

COLLISION_NEAR_TYPES = {"near_hit", "near_miss"}

def _collision_subject_column(data):
    """Return the participant-ID column used by a collision trial table."""
    for column in ["subject", "subject_id"]:
        if column in data.columns:
            return column
    raise KeyError("Collision table needs a 'subject' or 'subject_id' column.")

def _collision_response_present(series):
    """True when a HIT/MISS response is actually present."""
    text = series.astype("string").str.strip()
    return (
        series.notna()
        & text.ne("")
        & text.str.lower().ne("nan")
        & text.str.lower().ne("none")
    )

def collision_prepare(data):
    """Add collision-task descriptive/QC columns without dropping trials.

    The function preserves every original row and derives:

    ``responded``
        A participant response exists.
    ``correct``
        Participant response matches ``correct_response``. This is recomputed
        from the response strings rather than trusting the exported accuracy
        column blindly.
    ``near_boundary``
        ``near_hit`` or ``near_miss`` trial.
    ``hit_response``
        HIT=1, MISS=0, missing response=NaN.
    ``response_repetition``
        Current participant response equals the participant's immediately
        preceding response within the same subject/session.
    ``perseverative_opportunity``
        Repeating the preceding participant response would conflict with the
        CURRENT correct response.
    ``perseverative_error``
        On such an opportunity, the participant actually repeated the old
        response and was therefore wrong.

    ``response_repetition`` and ``perseverative_error`` are deliberately kept
    separate. Repeating HIT twice is not perseveration when HIT is correct on
    both trials.
    """
    required = {"participant_response", "correct_response", "trial_type"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(f"Collision table is missing required columns: {missing}")

    out = data.copy()
    subject = _collision_subject_column(out)

    out["responded"] = _collision_response_present(out["participant_response"])
    out["near_boundary"] = out["trial_type"].isin(COLLISION_NEAR_TYPES)

    response = out["participant_response"].astype("string").str.upper()
    correct_response = out["correct_response"].astype("string").str.upper()
    out["correct"] = out["responded"] & response.eq(correct_response)

    if "accuracy" in out.columns:
        exported_accuracy = pd.to_numeric(out["accuracy"], errors="coerce")
        out["accuracy_disagrees"] = (
            exported_accuracy.notna()
            & out["responded"]
            & exported_accuracy.ne(out["correct"].astype(int))
        )
    else:
        out["accuracy_disagrees"] = False

    out["hit_response"] = np.where(
        out["responded"], response.eq("HIT").astype(float), np.nan
    )

    if "reaction_time_ms" in out.columns:
        out["reaction_time_ms"] = pd.to_numeric(
            out["reaction_time_ms"], errors="coerce"
        )
    else:
        out["reaction_time_ms"] = np.nan

    # Work in real trial order before computing lagged responses.
    order_columns = [
        column for column in
        [subject, "session_number", "block_number", "trial_number"]
        if column in out.columns
    ]
    original_index = out.index
    ordered = out.sort_values(order_columns).copy()

    lag_groups = [subject]
    if "session_number" in ordered.columns:
        lag_groups.append("session_number")

    ordered["previous_participant_response"] = (
        ordered.groupby(lag_groups, sort=False)["participant_response"].shift(1)
    )
    ordered["previous_correct_response"] = (
        ordered.groupby(lag_groups, sort=False)["correct_response"].shift(1)
    )

    if "transition_status" in ordered.columns:
        is_start = ordered["transition_status"].astype("string").str.lower().eq("start")
    else:
        is_start = pd.Series(False, index=ordered.index)

    has_previous = ordered["previous_participant_response"].notna() & ~is_start
    current_text = ordered["participant_response"].astype("string").str.upper()
    previous_text = (
        ordered["previous_participant_response"].astype("string").str.upper()
    )
    correct_text = ordered["correct_response"].astype("string").str.upper()

    ordered["response_repetition"] = np.where(
        has_previous & ordered["responded"],
        current_text.eq(previous_text),
        np.nan,
    )

    # Repeating the preceding RESPONSE, not merely the preceding correct answer,
    # would now be inappropriate.
    ordered["perseverative_opportunity"] = np.where(
        has_previous & ordered["responded"],
        correct_text.ne(previous_text),
        np.nan,
    )

    ordered["perseverative_error"] = np.where(
        ordered["perseverative_opportunity"].eq(True),
        current_text.eq(previous_text) & ~ordered["correct"],
        np.nan,
    )

    # Restore the input row order so adding QC columns does not surprise callers.
    return ordered.reindex(original_index)

def collision_response_table(data, near_only=False):
    """Response/no-response counts and response rate per participant."""
    d = collision_prepare(data)
    subject = _collision_subject_column(d)

    if near_only:
        d = d[d["near_boundary"]]

    table = (
        d.groupby(subject, as_index=False)
        .agg(
            n_trials=("responded", "size"),
            n_response=("responded", "sum"),
        )
    )
    table["n_no_response"] = table["n_trials"] - table["n_response"]
    table["response_rate"] = table["n_response"] / table["n_trials"]
    return table

def collision_confusion_table(data, near_only=False):
    """Hits, misses, false alarms and correct rejections per participant."""
    d = collision_prepare(data)
    subject = _collision_subject_column(d)

    if near_only:
        d = d[d["near_boundary"]]

    d = d[d["responded"]].copy()
    participant = d["participant_response"].astype("string").str.upper()
    correct = d["correct_response"].astype("string").str.upper()

    d["outcome"] = np.select(
        [
            correct.eq("HIT") & participant.eq("HIT"),
            correct.eq("HIT") & participant.eq("MISS"),
            correct.eq("MISS") & participant.eq("HIT"),
            correct.eq("MISS") & participant.eq("MISS"),
        ],
        ["hit", "miss", "false_alarm", "correct_rejection"],
        default="other",
    )

    columns = ["hit", "miss", "false_alarm", "correct_rejection", "other"]
    table = (
        pd.crosstab(d[subject], d["outcome"])
        .reindex(columns=columns, fill_value=0)
        .reset_index()
    )
    table["n_answered"] = table[columns].sum(axis=1)
    table["accuracy"] = (
        table["hit"] + table["correct_rejection"]
    ) / table["n_answered"]
    return table

def collision_accuracy_table(data):
    """Collision accuracy and confusion counts per participant.

    Accuracy is computed explicitly as

        (hits + correct rejections)
        -----------------------------------------------
        hits + misses + false alarms + correct rejections

    so the table shows exactly which response outcomes contribute to the
    performance measure. The same quantities are reported for near-boundary
    trials.
    """
    overall = collision_confusion_table(data, near_only=False).copy()
    near = collision_confusion_table(data, near_only=True).copy()

    if overall.empty:
        return overall

    subject = _collision_subject_column(overall)

    overall = overall.rename(columns={
        "hit": "hits",
        "miss": "misses",
        "false_alarm": "false_alarms",
        "correct_rejection": "correct_rejections",
    })

    near = near.rename(columns={
        "hit": "near_hits",
        "miss": "near_misses",
        "false_alarm": "near_false_alarms",
        "correct_rejection": "near_correct_rejections",
        "n_answered": "n_near_boundary",
        "accuracy": "near_boundary_accuracy",
        "other": "near_other",
    })

    keep_near = [
        subject,
        "near_hits",
        "near_misses",
        "near_false_alarms",
        "near_correct_rejections",
        "n_near_boundary",
        "near_boundary_accuracy",
    ]
    keep_near = [column for column in keep_near if column in near.columns]

    out = overall.merge(near[keep_near], on=subject, how="left")

    # Keep response completeness visible too.
    prepared = collision_prepare(data)
    response = (
        prepared.groupby(subject, as_index=False)
        .agg(
            n_trials=("responded", "size"),
            n_answered_check=("responded", "sum"),
            response_rate=("responded", "mean"),
        )
    )
    response["n_no_response"] = response["n_trials"] - response["n_answered_check"]

    out = out.merge(
        response[[subject, "n_trials", "n_no_response", "response_rate"]],
        on=subject,
        how="left",
    )

    ordered = [
        subject,
        "n_trials",
        "n_answered",
        "n_no_response",
        "response_rate",
        "hits",
        "misses",
        "false_alarms",
        "correct_rejections",
        "accuracy",
        "n_near_boundary",
        "near_hits",
        "near_misses",
        "near_false_alarms",
        "near_correct_rejections",
        "near_boundary_accuracy",
    ]
    return out[[column for column in ordered if column in out.columns]]

def collision_rt_table(data, by=("current_speed",), near_only=False):
    """Reaction-time descriptives for answered collision trials.

    No RT exclusion is applied here. This is an inspection table, so extreme
    values remain visible until a collision-specific RT rule is explicitly
    chosen.
    """
    d = collision_prepare(data)
    d = d[d["responded"] & d["reaction_time_ms"].notna()]

    if near_only:
        d = d[d["near_boundary"]]

    by = list(by) if isinstance(by, (list, tuple)) else [by]

    return (
        d.groupby(by, as_index=False, dropna=False)["reaction_time_ms"]
        .agg(
            n="count",
            mean_ms="mean",
            median_ms="median",
            sd_ms="std",
            q1_ms=lambda values: values.quantile(0.25),
            q3_ms=lambda values: values.quantile(0.75),
            min_ms="min",
            max_ms="max",
        )
        .round(1)
    )

def collision_sequence_table(data, near_only=False):
    """Participant-level descriptives by previous-current speed sequence.

    The four rows of interest are slow-slow, slow-fast, fast-fast and
    fast-slow. Explicit start trials are omitted because they have no prior
    speed context.
    """
    d = collision_prepare(data)
    subject = _collision_subject_column(d)

    if near_only:
        d = d[d["near_boundary"]]

    if "speed_sequence" not in d.columns:
        raise KeyError("Collision table needs 'speed_sequence' for this summary.")

    d = d[d["speed_sequence"].isin(COLLISION_SEQUENCE_ORDER)]
    rows = []

    for (participant, sequence), one in d.groupby(
        [subject, "speed_sequence"], sort=False
    ):
        answered = one[one["responded"]]
        rt = answered["reaction_time_ms"].dropna()
        opportunities = answered[
            answered["perseverative_opportunity"].eq(True)
        ]

        rows.append({
            subject: participant,
            "speed_sequence": sequence,
            "n_trials": len(one),
            "n_answered": len(answered),
            "response_rate": one["responded"].mean(),
            "accuracy": answered["correct"].mean() if len(answered) else np.nan,
            "hit_rate": (
                answered["hit_response"].mean() if len(answered) else np.nan
            ),
            "median_rt_ms": rt.median() if len(rt) else np.nan,
            "mean_rt_ms": rt.mean() if len(rt) else np.nan,
            "response_repetition_rate": (
                answered["response_repetition"].mean()
                if answered["response_repetition"].notna().any() else np.nan
            ),
            "n_perseverative_opportunities": len(opportunities),
            "perseverative_error_rate": (
                opportunities["perseverative_error"].mean()
                if len(opportunities) else np.nan
            ),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["speed_sequence"] = pd.Categorical(
        out["speed_sequence"],
        categories=COLLISION_SEQUENCE_ORDER,
        ordered=True,
    )
    return out.sort_values([subject, "speed_sequence"]).reset_index(drop=True)

def collision_describe(data):
    """Return the main collision descriptive tables in one dictionary."""
    return {
        "accuracy": collision_accuracy_table(data),
        "confusion_all": collision_confusion_table(data, near_only=False),
        "confusion_near": collision_confusion_table(data, near_only=True),
        "responses_all": collision_response_table(data, near_only=False),
        "responses_near": collision_response_table(data, near_only=True),
        "rt_by_speed": collision_rt_table(data, by=("current_speed",)),
        "rt_by_sequence": collision_rt_table(data, by=("speed_sequence",)),
        "sequence_all": collision_sequence_table(data, near_only=False),
        "sequence_near": collision_sequence_table(data, near_only=True),
    }

COLLISION_RT_MIN_MS = 150

COLLISION_RT_MAX_MS = 3000

MIN_PSE_TRIALS = 25

MIN_PSE_LEVELS = 4

def collision_rt_window(
    data,
    rt_min=COLLISION_RT_MIN_MS,
    rt_max=COLLISION_RT_MAX_MS,
    verbose=True,
):
    """Apply an absolute RT window to collision trials.

    The PPS trials go through ``prepare_analysis_tables``, which applies a
    window; the collision trials never did. Without one the reaction-time
    distribution runs to tens of seconds and no plot of it is readable.
    """
    d = collision_prepare(data)
    d["rt_ms"] = pd.to_numeric(d["rt_ms"], errors="coerce")

    outside = d["responded"] & ~d["rt_ms"].between(
        rt_min, rt_max, inclusive="neither"
    )

    if verbose and outside.any():
        print(
            f"collision RT window {rt_min}-{rt_max} ms: "
            f"{int(outside.sum())} of {int(d['responded'].sum())} "
            f"responses outside "
            f"({100 * outside.sum() / max(d['responded'].sum(), 1):.1f}%)"
        )

    out = d.copy()
    out.loc[outside, "rt_ms"] = np.nan
    out["rt_usable"] = d["responded"] & ~outside

    return out

# Median and mean are drawn in their own colours rather than in a shade of
# the condition colour: the two summaries answer different questions at
# four trials per cell, and a reader has to be able to tell which line is
# which without matching hues.
#
# Named for the figure that uses them. Every other panel in this module
# takes its summary colour from the condition, so an unqualified name
# would read as a module-wide default and invite exactly the confusion
# it is meant to prevent.
DISTRIBUTION_MEDIAN_COLOUR = "#000000"
DISTRIBUTION_MEAN_COLOUR = "#00bf63"

# Later trials get their own colour rather than a lower alpha. Alpha
# distinguishes overlapping points poorly once a cell holds more than a
# handful, which is exactly the case the ordering is meant to reveal.
# One late colour per condition, since a single one for both would break
# the grey-and-teal pairing the rest of the panel uses to separate them.
LATE_TRIAL_COLOUR = {"T": "#716c6a", "VT": "#14877f"}

def cell_medians(pps, stage="after", subjects=None, drop=None,
                 statistic="median"):
    """One value per participant and cell, and the group value over those.

    The two-stage summary the group analyses use: summarise within each
    participant, then across participants, so a participant who contributed
    more trials does not count for more.

    Returns (per_subject, group). ``per_subject`` has one row per subject,
    speed, condition and distance; ``group`` has the median and spread across
    participants, plus how many contributed.
    """
    data = _drop_subjects(
        trials_at_stage(pps, stage=stage, subjects=subjects), drop
    )

    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    keys = ["speed", "sensory_condition", "position_rank"]

    by_speed = (
        data.groupby(["subject"] + keys, as_index=False)
        .agg(value_ms=("rt_ms", statistic), n_trials=("rt_ms", "size"))
    )

    # The pooled panel summarises each participant's trials across both
    # speeds, rather than averaging their two speed medians, so it stays a
    # median of trials rather than a median of medians.
    pooled = (
        data.groupby(["subject"] + keys[1:], as_index=False)
        .agg(value_ms=("rt_ms", statistic), n_trials=("rt_ms", "size"))
        .assign(speed="both")
    )

    per_subject = pd.concat(
        [by_speed, pooled[["subject"] + keys + ["value_ms", "n_trials"]]],
        ignore_index=True,
    )

    group = (
        per_subject.groupby(keys, as_index=False)
        .agg(
            group_median=("value_ms", "median"),
            group_mean=("value_ms", "mean"),
            group_sd=("value_ms", "std"),
            n_subjects=("subject", "nunique"),
            n_trials=("n_trials", "sum"),
        )
    )
    group["group_sem"] = group["group_sd"] / np.sqrt(group["n_subjects"])

    return per_subject, group.sort_values(keys).reset_index(drop=True)


def plot_trial_distribution(
    pps,
    stage="after",
    subjects=None,
    drop=None,
    statistic="median",
    show_points=True,
    jitter=0.045,
    save_path=None,
):
    """Participant medians per cell, and the group value formed over them.

    Three panels: slow, fast, and the two pooled. At each distance the
    tactile-only and visuo-tactile cells are drawn side by side. Every point
    is one participant's median for that cell, the violin is the distribution
    of those participant values, and the bars are the median and mean across
    participants.

    The point of drawing it this way is that the points are exactly what the
    group line was computed from. A group value resting on nine participants
    who disagree looks different here from one they agree on, and the trial
    counts underneath say how many observations sat behind each participant
    value.

    Read the number under each cell as participants, not trials: it is how
    many contributed a median, and a cell missing from one participant shows
    up as a smaller count rather than as a gap.
    """
    ranks = _ranks(pps)

    per_subject, group = cell_medians(
        pps, stage=stage, subjects=subjects, drop=drop, statistic=statistic,
    )

    if per_subject.empty:
        print("No usable trials for this selection.")
        return None, pd.DataFrame()

    panels = SPEEDS + ["both"]
    offsets = {"T": -0.20, "VT": 0.20}
    rng = np.random.default_rng(41)

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.6 * len(panels), 4.6),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    limits = _observed_range(_drop_subjects(_select_subjects(pps, subjects), drop))

    for ax, speed in zip(axes, panels):
        at_speed = per_subject[per_subject["speed"] == speed]

        for condition in TACTILE_CONDITIONS:
            colour = CONDITION_COLOUR[condition]
            cells = at_speed[at_speed["sensory_condition"] == condition]

            for rank in ranks:
                values = (
                    cells.loc[cells["position_rank"].eq(rank), "value_ms"]
                    .dropna().to_numpy(dtype=float)
                )
                centre = rank + offsets[condition]

                if not values.size:
                    continue

                if values.size >= 2 and np.std(values) > 0:
                    parts = ax.violinplot(
                        [values], positions=[centre], widths=0.34,
                        showmeans=False, showmedians=False, showextrema=False,
                    )

                    for body in parts["bodies"]:
                        body.set_facecolor(colour)
                        body.set_edgecolor(colour)
                        body.set_alpha(0.18)
                        body.set_linewidth(0)

                if show_points:
                    _scatter_trials(
                        ax, values, centre, colour, rng,
                        spread=jitter, size=22,
                    )

                # The median across participants, and the mean beside it, so
                # a skewed set of participant values is visible as the gap
                # between the two rather than having to be described.
                ax.plot(
                    [centre - 0.13, centre + 0.13],
                    [float(np.median(values))] * 2,
                    color=_darker(colour), linewidth=2.6,
                    solid_capstyle="butt", zorder=5,
                )
                ax.plot(
                    [centre - 0.09, centre + 0.09],
                    [float(np.mean(values))] * 2,
                    color=_darker(colour), linewidth=1.2, linestyle=":",
                    solid_capstyle="butt", zorder=5,
                )

                ax.annotate(
                    f"{values.size}",
                    (centre, limits[0]),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=6.5, color=style._MUTED,
                )

            ax.plot(
                [], [], marker="s", linestyle="none", color=colour,
                label=CONDITION_LABELS[condition],
            )

        title = "Both speeds" if speed == "both" else speed.capitalize()
        _finish(ax, title, f"Participant {statistic} RT (ms)", ranks=ranks)
        ax.set_xlim(min(ranks) - 0.6, max(ranks) + 0.6)
        ax.set_ylim(limits)

    axes[0].legend(frameon=False, ncol=2, fontsize=8, loc="upper left")

    n_subjects = int(per_subject["subject"].nunique())
    fig.suptitle(
        f"Participant {statistic}s per distance, and the group value over "
        f"them ({n_subjects} participants; solid bar = median across "
        "participants, dotted = mean; number = participants in the cell)",
        x=0.02, ha="left", y=1.01, color=style.INK,
    )
    fig.tight_layout()
    _save(fig, save_path)

    return fig, group