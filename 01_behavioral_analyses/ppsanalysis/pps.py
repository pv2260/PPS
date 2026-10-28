"""PPS (Task 1) derived measures.

facilitation -> near_far_index -> sigmoid_boundary_by_subject -> Delta_PPS

Primary PPS correction
----------------------
Facilitation is computed with the preregistered delay- and speed-matched tactile
baseline:

    FAC(d, s) = median[T(d, s)] - median[VT(d, s)]

Boundary estimation
-------------------
A sigmoid x_c is used only when the transition is identifiable in the sampled
range and trial-level bootstrap localises x_c with acceptable precision.
Otherwise, the observed half-maximum crossover is used as the preregistered
fallback.

For Delta_PPS, the same estimator is used at both speeds:
    - sigmoid at both speeds if BOTH sigmoid boundaries are estimable
    - otherwise half-maximum at both speeds
This avoids subtracting two different kinds of boundary estimator.

Matching H2
-----------
``delta_pps_as_h2`` computes Delta_PPS exactly as ``h2.run_by_speed`` does:
facilitation from ``h1.facilitation_profile`` under a named correction, then
the H1a boundary sigmoid at each speed. ``bootstrap_pps_delta_one_session``
uses it when given ``correction``, so its draws describe the same quantity as
the observed Delta_PPS that H5 prints first. Without ``correction`` it keeps
the delay-matched behaviour described above.
"""

import numpy as np
import pandas as pd

from .config import SLOW_LABEL, FAST_LABEL, N_BOOT, RANDOM_SEED
from .stats_utils import safe_sigmoid_fit


# ---------------------------------------------------------------------
# Default post-preregistration operational criteria for an estimable x_c
# ---------------------------------------------------------------------

DEFAULT_MIN_BOOT_SUCCESS = 0.80
DEFAULT_MAX_BOOT_CI_WIDTH = 2.0       # distance ranks
DEFAULT_MIN_TRANSITION_WIDTH = 1.0    # ranks; narrower = unresolved step
DEFAULT_MAX_TRANSITION_WIDTH = None   # if None, use sampled x-range


def compute_facilitation(pps_df):
    """Preregistered delay-matched facilitation.

    Median RT is estimated separately within each
    subject x session x speed x position x condition cell, then

        facilitation = T - VT

    Because T and VT are matched on speed and position, this is the
    preregistered delay-matched tactile correction.

    The output keeps `position`, `position_rank`, and optionally `distance_m`.
    """
    df = pps_df.loc[pps_df["usable"]].copy()
    df = df[df["sensory_condition"].isin(["T", "VT"])]

    index_cols = [
        "subject",
        "session",
        "group",
        "cohort",
        "speed",
        "position",
        "position_rank",
    ]

    med = (
        df.groupby(
            index_cols + ["sensory_condition"],
            as_index=False,
            dropna=False,
        )["rt_ms"]
        .median()
    )

    wide = (
        med.pivot(
            index=index_cols,
            columns="sensory_condition",
            values="rt_ms",
        )
        .reset_index()
    )

    wide.columns.name = None

    if "T" not in wide.columns:
        wide["T"] = np.nan
    if "VT" not in wide.columns:
        wide["VT"] = np.nan

    wide["facilitation_ms"] = wide["T"] - wide["VT"]

    # Attach distance_m after the pivot so an all-NaN distance column cannot
    # make valid position rows disappear.
    if "distance_m" in df.columns and df["distance_m"].notna().any():
        dist_map = (
            df[
                [
                    "subject",
                    "session",
                    "position",
                    "position_rank",
                    "distance_m",
                ]
            ]
            .dropna(subset=["distance_m"])
            .drop_duplicates(
                subset=["subject", "session", "position", "position_rank"]
            )
        )

        wide = wide.merge(
            dist_map,
            on=["subject", "session", "position", "position_rank"],
            how="left",
        )
    else:
        wide["distance_m"] = np.nan

    return wide


def near_far_index(fac_df, near_n=2, far_n=2, axis_col="position_rank"):
    """Near-far PPS index per subject/session.

    With position_rank:
        rank 1 = nearest
        largest rank = farthest
    """
    rows = []
    columns = [
        "subject",
        "session",
        "group",
        "cohort",
        "near_mean",
        "far_mean",
        "nearfar_pps",
    ]

    for keys, g in fac_df.groupby(
        ["subject", "session", "group", "cohort"],
        dropna=False,
    ):
        by_axis = (
            g.dropna(subset=[axis_col, "facilitation_ms"])
            .groupby(axis_col, as_index=False)["facilitation_ms"]
            .mean()
            .sort_values(axis_col)
        )

        if by_axis.empty:
            continue

        near = by_axis.head(near_n)["facilitation_ms"].mean()
        far = by_axis.tail(far_n)["facilitation_ms"].mean()

        rows.append(
            {
                "subject": keys[0],
                "session": keys[1],
                "group": keys[2],
                "cohort": keys[3],
                "near_mean": near,
                "far_mean": far,
                "nearfar_pps": near - far,
            }
        )

    return pd.DataFrame(rows, columns=columns)


# =====================================================================
# Boundary helpers
# =====================================================================

def _transition_width_10_90(k):
    """Width in x units between 10% and 90% of a logistic transition."""
    if k is None or not np.isfinite(k) or k == 0:
        return np.inf
    return float(2.0 * np.log(9.0) / abs(k))


def _fit_geometry(
    fit,
    x_min,
    x_max,
    min_transition_width=1.0,
    max_transition_width=None,
):
    """Evaluate whether a fitted sigmoid contains a usable PPS transition.

    Steep transitions are flagged but do not invalidate x_c.
    Shallow transitions can invalidate x_c because the transition is not
    contained within the sampled range.
    """
    out = {
        "sigmoid_ok": False,
        "x_c": np.nan,
        "k": np.nan,
        "low": np.nan,
        "high": np.nan,
        "x_c_in_range": False,
        "direction_ok": False,
        "transition_width_10_90": np.nan,
        "transition_too_steep": False,
        "transition_too_shallow": False,
        "transition_width_ok": False,
        "geometry_ok": False,
    }

    if not fit or not fit.get("ok", False):
        return out

    x_c = fit.get("x_c", np.nan)
    k = fit.get("k", np.nan)
    low = fit.get("low", np.nan)
    high = fit.get("high", np.nan)

    out.update({
        "sigmoid_ok": True,
        "x_c": x_c,
        "k": k,
        "low": low,
        "high": high,
    })

    if not (np.isfinite(x_c) and np.isfinite(k)):
        return out

    x_c_in_range = bool(x_min < x_c < x_max)

    width = _transition_width_10_90(k)

    if max_transition_width is None:
        max_transition_width = float(x_max - x_min)

    # With:
    # y = low + (high-low)/(1 + exp(k*(x-x_c)))
    #
    # position_rank increases away from the body.
    # Facilitation should therefore decrease as rank increases.
    if np.isfinite(low) and np.isfinite(high):
        direction_ok = bool((high - low) * k > 0)
    else:
        direction_ok = False

    too_steep = bool(
        np.isfinite(width)
        and width < min_transition_width
    )

    too_shallow = bool(
        not np.isfinite(width)
        or width > max_transition_width
    )

    # Steep is only a warning.
    # Too shallow makes xc geometrically uninformative.
    transition_width_ok = not too_shallow

    geometry_ok = bool(
        x_c_in_range
        and direction_ok
        and transition_width_ok
    )

    out.update({
        "x_c_in_range": x_c_in_range,
        "direction_ok": direction_ok,
        "transition_width_10_90": width,
        "transition_too_steep": too_steep,
        "transition_too_shallow": too_shallow,
        "transition_width_ok": transition_width_ok,
        "geometry_ok": geometry_ok,
    })

    return out

def _halfmax_boundary(profile, axis_col="position_rank"):
    """Observed half-maximum crossover.

    Far plateau  = mean of two farthest observed positions.
    Near plateau = mean of two nearest observed positions.
    Half level   = midpoint of those plateaus.

    Starting at the far end, walk inward and linearly interpolate the first
    pair of observed points that straddles the half-maximum level.
    """
    one = (
        profile[[axis_col, "facilitation_ms"]]
        .dropna()
        .groupby(axis_col, as_index=False)["facilitation_ms"]
        .mean()
        .sort_values(axis_col)
    )

    if len(one) < 4:
        return {
            "halfmax": np.nan,
            "halfmax_level": np.nan,
            "near_plateau": np.nan,
            "far_plateau": np.nan,
            "halfmax_ok": False,
        }

    near_plateau = float(one.head(2)["facilitation_ms"].mean())
    far_plateau = float(one.tail(2)["facilitation_ms"].mean())
    half_level = 0.5 * (near_plateau + far_plateau)

    # Walk D7 -> D1 (largest rank -> smallest rank).
    inward = one.sort_values(axis_col, ascending=False).reset_index(drop=True)

    boundary = np.nan

    for i in range(len(inward) - 1):
        x1 = float(inward.loc[i, axis_col])
        y1 = float(inward.loc[i, "facilitation_ms"])
        x2 = float(inward.loc[i + 1, axis_col])
        y2 = float(inward.loc[i + 1, "facilitation_ms"])

        a = y1 - half_level
        b = y2 - half_level

        if a == 0:
            boundary = x1
            break

        if b == 0:
            boundary = x2
            break

        if a * b < 0:
            if y2 == y1:
                boundary = 0.5 * (x1 + x2)
            else:
                boundary = x1 + (half_level - y1) * (x2 - x1) / (y2 - y1)
            break

    return {
        "halfmax": float(boundary) if np.isfinite(boundary) else np.nan,
        "halfmax_level": half_level,
        "near_plateau": near_plateau,
        "far_plateau": far_plateau,
        "halfmax_ok": bool(np.isfinite(boundary)),
    }


def _bootstrap_xc(
    raw_trials,
    *,
    speed=None,
    axis_col="position_rank",
    n_boot=1000,
    seed=RANDOM_SEED,
    min_transition_width=DEFAULT_MIN_TRANSITION_WIDTH,
    max_transition_width=DEFAULT_MAX_TRANSITION_WIDTH,
):
    """Fast trial-level bootstrap distribution of geometrically valid x_c.

    Trials are resampled within speed x position x condition cells. Instead of
    rebuilding a full trial DataFrame on every resample, this function stores
    the RT arrays once and resamples their medians directly. This is equivalent
    for the median delay-matched FAC profile and is much faster.
    """
    df = raw_trials.loc[raw_trials["usable"]].copy()
    df = df[
        df["sensory_condition"].isin(["T", "VT"])
        & df["rt_ms"].notna()
    ]

    if speed is not None:
        df = df[df["speed"].eq(speed)]

    if df.empty:
        return {
            "bootstrap_n": int(n_boot),
            "bootstrap_n_valid": 0,
            "bootstrap_success_rate": 0.0,
            "bootstrap_x_c_median": np.nan,
            "bootstrap_x_c_ci_low": np.nan,
            "bootstrap_x_c_ci_high": np.nan,
            "bootstrap_x_c_ci_width": np.nan,
        }

    # Store one RT array per speed x rank x condition cell.
    cell_arrays = {}
    for key, g in df.groupby(
        ["speed", axis_col, "sensory_condition"],
        dropna=False,
        sort=False,
    ):
        values = g["rt_ms"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            cell_arrays[key] = values

    rng = np.random.default_rng(seed)
    xcs = []

    speeds = [speed] if speed is not None else sorted(df["speed"].dropna().unique())

    for _ in range(int(n_boot)):
        fac_rows = []

        for this_speed in speeds:
            ranks = sorted(
                {
                    key[1]
                    for key in cell_arrays
                    if key[0] == this_speed
                }
            )

            for rank in ranks:
                t = cell_arrays.get((this_speed, rank, "T"))
                vt = cell_arrays.get((this_speed, rank, "VT"))

                if t is None or vt is None or len(t) == 0 or len(vt) == 0:
                    continue

                t_med = float(np.median(
                    t[rng.integers(0, len(t), size=len(t))]
                ))
                vt_med = float(np.median(
                    vt[rng.integers(0, len(vt), size=len(vt))]
                ))

                fac_rows.append((this_speed, float(rank), t_med - vt_med))

        if not fac_rows:
            continue

        boot_fac = pd.DataFrame(
            fac_rows,
            columns=["speed", axis_col, "facilitation_ms"],
        )

        # H1a/speed-averaged boundary averages the two speed-specific FAC values
        # only after each speed x position FAC has been formed.
        profile = (
            boot_fac
            .groupby(axis_col, as_index=False)["facilitation_ms"]
            .mean()
            .sort_values(axis_col)
        )

        if len(profile) < 5:
            continue

        x = profile[axis_col].to_numpy(dtype=float)
        y = profile["facilitation_ms"].to_numpy(dtype=float)

        fit = safe_sigmoid_fit(x, y)

        geom = _fit_geometry(
            fit,
            x_min=float(np.min(x)),
            x_max=float(np.max(x)),
            min_transition_width=min_transition_width,
            max_transition_width=max_transition_width,
        )

        if geom["geometry_ok"]:
            xcs.append(float(geom["x_c"]))

    xcs = np.asarray(xcs, dtype=float)
    n_valid = int(len(xcs))

    if n_valid == 0:
        return {
            "bootstrap_n": int(n_boot),
            "bootstrap_n_valid": 0,
            "bootstrap_success_rate": 0.0,
            "bootstrap_x_c_median": np.nan,
            "bootstrap_x_c_ci_low": np.nan,
            "bootstrap_x_c_ci_high": np.nan,
            "bootstrap_x_c_ci_width": np.nan,
        }

    lo, hi = np.percentile(xcs, [2.5, 97.5])

    return {
        "bootstrap_n": int(n_boot),
        "bootstrap_n_valid": n_valid,
        "bootstrap_success_rate": n_valid / float(n_boot),
        "bootstrap_x_c_median": float(np.median(xcs)),
        "bootstrap_x_c_ci_low": float(lo),
        "bootstrap_x_c_ci_high": float(hi),
        "bootstrap_x_c_ci_width": float(hi - lo),
    }

def _estimate_one_boundary(
    profile,
    *,
    raw_trials=None,
    speed=None,
    axis_col="position_rank",
    n_boot=1000,
    seed=RANDOM_SEED,
    min_boot_success=DEFAULT_MIN_BOOT_SUCCESS,
    max_boot_ci_width=DEFAULT_MAX_BOOT_CI_WIDTH,
    min_transition_width=DEFAULT_MIN_TRANSITION_WIDTH,
    max_transition_width=DEFAULT_MAX_TRANSITION_WIDTH,
):
    """Fit one profile and decide whether sigmoid x_c is estimable."""
    one = (
        profile.dropna(subset=[axis_col, "facilitation_ms"])
        .groupby(axis_col, as_index=False)["facilitation_ms"]
        .mean()
        .sort_values(axis_col)
    )

    result = {
        "n_distances": int(len(one)),
        "sigmoid_ok": False,
        "x_c": np.nan,
        "k": np.nan,
        "low": np.nan,
        "high": np.nan,
        "x_c_in_range": False,
        "direction_ok": False,
        "transition_width_10_90": np.nan,
        "transition_too_steep": False,
        "transition_too_shallow": False,
        "transition_width_ok": False,
        "geometry_ok": False,
        "bootstrap_n": int(n_boot),
        "bootstrap_n_valid": 0,
        "bootstrap_success_rate": np.nan,
        "bootstrap_x_c_median": np.nan,
        "bootstrap_x_c_ci_low": np.nan,
        "bootstrap_x_c_ci_high": np.nan,
        "bootstrap_x_c_ci_width": np.nan,
        "bootstrap_precise": False,
        "sigmoid_estimable": False,
    }

    result.update(_halfmax_boundary(one, axis_col=axis_col))

    if len(one) < 5:
        return result

    x = one[axis_col].to_numpy(dtype=float)
    y = one["facilitation_ms"].to_numpy(dtype=float)

    x_min = float(np.min(x))
    x_max = float(np.max(x))

    fit = safe_sigmoid_fit(x, y)

    geom = _fit_geometry(
        fit,
        x_min=x_min,
        x_max=x_max,
        min_transition_width=min_transition_width,
        max_transition_width=max_transition_width,
    )
    result.update(geom)

    # A raw aggregated profile can tell us geometry, but NOT trial-level
    # precision. If raw trials are unavailable, the fit is deliberately not
    # promoted to "estimable".
    if raw_trials is None or int(n_boot) <= 0:
        return result

    boot = _bootstrap_xc(
        raw_trials,
        speed=speed,
        axis_col=axis_col,
        n_boot=n_boot,
        seed=seed,
        min_transition_width=min_transition_width,
        max_transition_width=max_transition_width,
    )
    result.update(boot)

    bootstrap_precise = bool(
        boot["bootstrap_success_rate"] >= min_boot_success
        and np.isfinite(boot["bootstrap_x_c_ci_width"])
        and boot["bootstrap_x_c_ci_width"] <= max_boot_ci_width
    )

    result["bootstrap_precise"] = bootstrap_precise
    result["sigmoid_estimable"] = bool(
        result["geometry_ok"] and bootstrap_precise
    )

    return result


def _prefix(d, suffix):
    """Return a dictionary with `suffix` appended to every key."""
    return {f"{key}_{suffix}": value for key, value in d.items()}


def sigmoid_boundary_by_subject(
    fac_df,
    split_speed=False,
    axis_col="position_rank",
    *,
    raw_trials=None,
    n_boot=1000,
    seed=RANDOM_SEED,
    min_boot_success=DEFAULT_MIN_BOOT_SUCCESS,
    max_boot_ci_width=DEFAULT_MAX_BOOT_CI_WIDTH,
    min_transition_width=DEFAULT_MIN_TRANSITION_WIDTH,
    max_transition_width=DEFAULT_MAX_TRANSITION_WIDTH,
):
    """Estimate PPS boundary per subject/session.

    Parameters
    ----------
    fac_df
        Delay-matched facilitation table from compute_facilitation().
    split_speed
        False: one speed-averaged boundary.
        True: slow and fast boundaries plus Delta_PPS.
    raw_trials
        Trial-level PPS table. Required for trial-level bootstrap precision.
        If omitted, sigmoid geometry and half-max are still returned, but a
        sigmoid is not labelled "estimable".
    n_boot
        Number of trial-level bootstrap resamples used to localise x_c.

    Returns
    -------
    DataFrame
        When split_speed=True, raw sigmoid diagnostics are returned separately
        for slow/fast. The FINAL boundaries use the same estimator at both
        speeds:

            both sigmoid estimable -> sigmoid / sigmoid
            otherwise             -> halfmax / halfmax

        delta_pps = boundary_fast - boundary_slow
    """
    rows = []
    group_cols = ["subject", "session", "group", "cohort"]

    for group_i, (keys, g) in enumerate(
        fac_df.groupby(group_cols, dropna=False)
    ):
        base = {
            "subject": keys[0],
            "session": keys[1],
            "group": keys[2],
            "cohort": keys[3],
        }

        raw_one = None
        if raw_trials is not None:
            mask = pd.Series(True, index=raw_trials.index)

            for col, value in zip(group_cols, keys):
                if pd.isna(value):
                    mask &= raw_trials[col].isna()
                else:
                    mask &= raw_trials[col].eq(value)

            raw_one = raw_trials.loc[mask].copy()

        # Stable but distinct RNG stream per subject/session.
        subject_seed = int(seed) + 100003 * group_i

        if not split_speed:
            profile = (
                g.dropna(subset=[axis_col, "facilitation_ms"])
                .groupby(axis_col, as_index=False)["facilitation_ms"]
                .mean()
            )

            est = _estimate_one_boundary(
                profile,
                raw_trials=raw_one,
                speed=None,
                axis_col=axis_col,
                n_boot=n_boot,
                seed=subject_seed,
                min_boot_success=min_boot_success,
                max_boot_ci_width=max_boot_ci_width,
                min_transition_width=min_transition_width,
                max_transition_width=max_transition_width,
            )

            if est["sigmoid_estimable"]:
                boundary_method = "sigmoid"
                boundary = est["x_c"]
            elif est["halfmax_ok"]:
                boundary_method = "halfmax"
                boundary = est["halfmax"]
            else:
                boundary_method = "not_estimable"
                boundary = np.nan

            rows.append(
                {
                    **base,
                    **est,
                    "boundary_method": boundary_method,
                    "boundary": boundary,
                }
            )

        else:
            est_by_speed = {}

            for j, speed in enumerate([SLOW_LABEL, FAST_LABEL]):
                profile = g[
                    g["speed"].eq(speed)
                ].dropna(subset=[axis_col, "facilitation_ms"])

                est_by_speed[speed] = _estimate_one_boundary(
                    profile,
                    raw_trials=raw_one,
                    speed=speed,
                    axis_col=axis_col,
                    n_boot=n_boot,
                    seed=subject_seed + 1009 * (j + 1),
                    min_boot_success=min_boot_success,
                    max_boot_ci_width=max_boot_ci_width,
                    min_transition_width=min_transition_width,
                    max_transition_width=max_transition_width,
                )

            slow = est_by_speed[SLOW_LABEL]
            fast = est_by_speed[FAST_LABEL]

            both_sigmoid = bool(
                slow["sigmoid_estimable"]
                and fast["sigmoid_estimable"]
            )

            both_halfmax = bool(
                slow["halfmax_ok"]
                and fast["halfmax_ok"]
            )

            if both_sigmoid:
                boundary_method = "sigmoid"
                boundary_slow = slow["x_c"]
                boundary_fast = fast["x_c"]

            elif both_halfmax:
                boundary_method = "halfmax"
                boundary_slow = slow["halfmax"]
                boundary_fast = fast["halfmax"]

            else:
                boundary_method = "not_estimable"
                boundary_slow = np.nan
                boundary_fast = np.nan

            delta_pps = (
                boundary_fast - boundary_slow
                if np.isfinite(boundary_fast) and np.isfinite(boundary_slow)
                else np.nan
            )

            rows.append(
                {
                    **base,
                    **_prefix(slow, SLOW_LABEL),
                    **_prefix(fast, FAST_LABEL),
                    "boundary_method": boundary_method,
                    "boundary_slow": boundary_slow,
                    "boundary_fast": boundary_fast,
                    "delta_pps": delta_pps,
                }
            )

    return pd.DataFrame(rows)


def delta_pps_as_h2(trials, correction, expectancy_degree=1):
    """Boundaries and Delta_PPS for one participant, computed as H2 does.

    Returns (boundary_slow, boundary_fast, delta_pps), each NaN where a
    speed could not be fitted. A failure inside the fit returns NaNs rather
    than raising, so one bad resample cannot stop a loop of thousands.
    """
    # Imported here rather than at the top: the hypothesis modules import
    # this module, so a top-level import would be circular.
    from .hypotheses import h2

    try:
        _, boundaries = h2.boundaries_by_speed(
            trials,
            corrections=(correction,),
            verbose=False,
            expectancy_degree=expectancy_degree,
        )
    except Exception:
        return np.nan, np.nan, np.nan

    if boundaries.empty:
        return np.nan, np.nan, np.nan

    row = boundaries.iloc[0]

    return (
        float(row.get(SLOW_LABEL, np.nan)),
        float(row.get(FAST_LABEL, np.nan)),
        float(row.get("delta_pps", np.nan)),
    )


def bootstrap_pps_delta_one_session(
    df,
    n_boot=N_BOOT,
    seed=RANDOM_SEED,
    *,
    correction=None,
    expectancy_degree=1,
    axis_col="position_rank",
    min_boot_success=DEFAULT_MIN_BOOT_SUCCESS,
    max_boot_ci_width=DEFAULT_MAX_BOOT_CI_WIDTH,
    min_transition_width=DEFAULT_MIN_TRANSITION_WIDTH,
    max_transition_width=DEFAULT_MAX_TRANSITION_WIDTH,
):
    """Bootstrap Delta_PPS for one subject/session.

    This outer bootstrap estimates uncertainty in the FINAL Delta_PPS estimator.
    Trials are resampled within speed x position x condition cells.

    With ``correction`` set, each resample's Delta_PPS is computed by
    ``delta_pps_as_h2``, the same estimator H5 reports as the observed value.

    With ``correction=None``, each resample rebuilds delay-matched facilitation
    and applies the rule below. To avoid a nested bootstrap explosion, the
    resampled profile uses the geometric sigmoid checks; if both
    speed-specific sigmoids fail those checks, half-max is used for both
    speeds.
    """
    rng = np.random.default_rng(seed)

    original = df.loc[df["usable"]].copy()
    original = original[
        original["sensory_condition"].isin(["T", "VT"])
        & original["speed"].isin([SLOW_LABEL, FAST_LABEL])
    ]

    if original.empty:
        return pd.DataFrame()

    cell_cols = ["speed", "position", axis_col, "sensory_condition"]
    cell_cols = [c for c in cell_cols if c in original.columns]

    groups = [
        g.copy()
        for _, g in original.groupby(cell_cols, dropna=False)
    ]

    rows = []

    for i in range(int(n_boot)):
        parts = []

        for cell in groups:
            take = rng.integers(0, len(cell), size=len(cell))
            parts.append(cell.iloc[take].copy())

        boot_df = pd.concat(parts, ignore_index=True)

        if correction is not None:
            b_slow, b_fast, delta = delta_pps_as_h2(
                boot_df, correction, expectancy_degree,
            )
            rows.append(
                {
                    "boot": i,
                    "boundary_method": f"h2 ({correction})",
                    "boundary_slow": b_slow,
                    "boundary_fast": b_fast,
                    "delta_pps": delta,
                }
            )
            continue

        fac = compute_facilitation(boot_df)

        speed_estimates = {}

        for speed in [SLOW_LABEL, FAST_LABEL]:
            profile = fac[
                fac["speed"].eq(speed)
            ].dropna(subset=[axis_col, "facilitation_ms"])

            one = (
                profile.groupby(
                    axis_col,
                    as_index=False,
                )["facilitation_ms"]
                .mean()
                .sort_values(axis_col)
            )

            half = _halfmax_boundary(one, axis_col=axis_col)

            if len(one) >= 5:
                x = one[axis_col].to_numpy(dtype=float)
                y = one["facilitation_ms"].to_numpy(dtype=float)

                fit = safe_sigmoid_fit(x, y)
                geom = _fit_geometry(
                    fit,
                    x_min=float(np.min(x)),
                    x_max=float(np.max(x)),
                    min_transition_width=min_transition_width,
                    max_transition_width=max_transition_width,
                )
            else:
                geom = {
                    "geometry_ok": False,
                    "x_c": np.nan,
                }

            speed_estimates[speed] = {
                "x_c": geom.get("x_c", np.nan),
                "sigmoid_geometry_ok": bool(
                    geom.get("geometry_ok", False)
                ),
                "halfmax": half["halfmax"],
                "halfmax_ok": half["halfmax_ok"],
            }

        slow = speed_estimates[SLOW_LABEL]
        fast = speed_estimates[FAST_LABEL]

        if (
            slow["sigmoid_geometry_ok"]
            and fast["sigmoid_geometry_ok"]
        ):
            method = "sigmoid"
            b_slow = slow["x_c"]
            b_fast = fast["x_c"]

        elif slow["halfmax_ok"] and fast["halfmax_ok"]:
            method = "halfmax"
            b_slow = slow["halfmax"]
            b_fast = fast["halfmax"]

        else:
            method = "not_estimable"
            b_slow = np.nan
            b_fast = np.nan

        delta = (
            b_fast - b_slow
            if np.isfinite(b_fast) and np.isfinite(b_slow)
            else np.nan
        )

        rows.append(
            {
                "boot": i,
                "boundary_method": method,
                "boundary_slow": b_slow,
                "boundary_fast": b_fast,
                "delta_pps": delta,
            }
        )

    return pd.DataFrame(rows)