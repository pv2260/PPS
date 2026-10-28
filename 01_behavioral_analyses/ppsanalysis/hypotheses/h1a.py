"""
H1a. The facilitation profile is sigmoidal, not flat and not a straight line.

H1 showed whether facilitation is stronger near the body. But "stronger near
the body" could look like several different shapes:

    flat     : facilitation is the same everywhere
    linear   : facilitation increases steadily as the stimulus gets closer
    sigmoid  : facilitation is low far away, changes around a boundary, then
               levels off close to the body

The sigmoid shape is the one that implies a boundary. A straight line does not:
it says the effect changes gradually, with no special distance. H1a therefore
asks whether the sigmoid fits better than both the flat and linear alternatives.

HOW WE COMPARE THE THREE SHAPES
-------------------------------
All three models are fitted to each participant's facilitation profile and
compared by AIC, which penalises the more flexible model for its extra
parameters.

    LOWER AIC = BETTER

WINNING ON AIC IS NOT ENOUGH
----------------------------
A curve can have the lowest AIC and still not be a boundary. Three things have
to hold before a fitted sigmoid describes one:

    k_ok           the transition runs in the predicted direction (more
                   facilitation near the body) and is wide enough to be
                   resolved by distances sampled one rank apart. An enormous
                   |k| is a step function: the optimiser has drawn a vertical
                   wall between two adjacent points, usually to reach a single
                   deviant value.

    x_c_in_range   the fitted boundary lies inside the distances tested.

    x_c_precise    the boundary has a usable standard error. When k is very
                   large the likelihood is flat in x_c across the whole gap
                   between two ranks, so se_x_c collapses towards zero or
                   explodes; both are the same pathology and both are
                   rejected.

Two counts are therefore reported and they answer different questions:

    "n_preferred"
        Sigmoid has the numerically lowest AIC. No structural check applied.
        This is the AIC comparison on its own.

    "n_supported"  (column: sigmoid_lower_aic)
        Sigmoid wins on AIC AND clears all three checks. This is the count
        that licenses any claim about a boundary, and it is what the H1a
        verdict reports.

WHICH TRIALS ENTER
------------------
H1a fits cell estimates, so it is only as good as the cells. Two rules are
applied before anything is fitted, and both are reported:

    usable only     Trials excluded by QC never reach the profile. If the
                    table carries a ``usable`` column it is honoured; if not,
                    the absolute RT window from config is applied and a
                    warning is printed, because a table without that column
                    has not been through QC and a single unexcluded trial can
                    carry an entire fitted sigmoid.

    min_cell_n      A cell with fewer trials than this contributes no
                    estimate. A median of one trial is that trial, and at the
                    tactile-only sampling density such a cell can produce the
                    largest facilitation value in the dataset.

``trial_accounting`` in the results reports how many trials and cells each
rule removed, so a profile can never be read without knowing what it rests on.

A NOTE ON POWER
---------------
Each participant contributes seven distances. The sigmoid has four parameters
plus the residual variance, the line three, the flat model two. On seven points
the sigmoid must reduce the residual sum of squares by roughly 44 per cent
relative to the line, and 58 per cent relative to flat, simply to break even on
AIC. AICc, which is the criterion seven points actually warrant, adds a further
52 points to the sigmoid; ``preferred_aicc`` applies the identical rule to it.
Where AIC and AICc disagree, the sigmoid won on a criterion that does not hold
at this profile length. A small count of participants meeting the criterion is
weak evidence about shape either way, and ``fit_diagnostics`` reports RSS and
R-squared so a failed fit can be told apart from a fit that simply lost.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import style

from .. import config
from .. import figures
from . import h1 as h1
from ..stats_utils import (
    sigmoid,
    fit_sigmoid_aic,
    compare_models_per_participant,
)

style.apply()

# The distance mapping and the rank-to-metre conversion live in h1 so that both
# modules format the distance axis identically. Re-exported here because
# existing notebook cells import them from h1a.
DISTANCE_M_BY_RANK = h1.DISTANCE_M_BY_RANK
rank_to_metres = h1.rank_to_metres

# Grey for tactile-only, blue for visuo-tactile, as everywhere else in the
# project. Panel A previously coloured by participant-versus-group, which left
# the condition carried only by line style.
CONDITION_COLOUR = {"T": style.SLOW, "VT": style.PATIENT}

# Facilitation is T - VT, so it is neither condition. Giving it one of
# their colours would read as a condition curve; it gets its own.
FACILITATION_COLOUR = getattr(style, "FACILITATION", style.INK)

# Columns produced by compare_models_per_participant, plus the subject key.
COMPARISON_COLUMNS = [
    "subject",
    "aic_flat",
    "aic_linear",
    "aic_sigmoid",
    "aic_delta_vs_flat",
    "aic_delta_vs_linear",
    "aicc_flat",
    "aicc_linear",
    "aicc_sigmoid",
    "aicc_delta_vs_flat",
    "aicc_delta_vs_linear",
    "rss_flat",
    "rss_linear",
    "rss_sigmoid",
    "r2_sigmoid",
    "x_c",
    "se_x_c",
    "k",
    "se_k",
    "amplitude_ms",
    "sigmoid_ok",
    "x_c_in_range",
    "x_c_precise",
    "k_ok",
    "k_at_bound",
    # Boundary-identifiability diagnostics used by H1a.  The legacy Hessian
    # SE fields above are retained for transparency but are no longer used as
    # the primary precision criterion.
    "transition_width_10_90",
    "transition_direction_ok",
    "transition_width_ok",
    "bootstrap_n",
    "bootstrap_n_valid",
    "bootstrap_success_rate",
    "bootstrap_x_c_median",
    "bootstrap_x_c_ci_low",
    "bootstrap_x_c_ci_high",
    "bootstrap_x_c_ci_width",
    "bootstrap_precise",
    "boundary_estimable",
    "halfmax_rank",
    "halfmax_m",
    "boundary_method",
    "boundary_rank",
    "boundary_m",
    "preferred",
    "preferred_aicc",
    "low",
    "high",
]

# Columns worth reading in the printed table. The full table is returned in
# results["table"]; printing all twenty-six columns makes none of them legible.
REPORT_COLUMNS = [
    "subject",
    "aic_flat",
    "aic_linear",
    "aic_sigmoid",
    "aic_delta_vs_linear",
    "aicc_delta_vs_linear",
    "r2_sigmoid",
    "x_c",
    "se_x_c",
    "k",
    "amplitude_ms",
    "k_ok",
    "k_at_bound",
    "transition_width_10_90",
    "transition_width_ok",
    "bootstrap_success_rate",
    "bootstrap_x_c_ci_width",
    "bootstrap_precise",
    "boundary_estimable",
    "boundary_method",
    "boundary_rank",
    "preferred",
    "preferred_aicc",
    "sigmoid_lower_aic",
]


CELL_KEYS = ["subject", "session", "speed", "sensory_condition", "position_rank"]


def _cell_keys(trials):
    return [column for column in CELL_KEYS if column in trials.columns]


def prepare_trials(trials, min_cell_n=3, verbose=True):
    """Apply the exclusions before any cell estimate is formed.

    Returns the usable trials and an accounting of what was removed. Both the
    RT window and the thin-cell rule act here rather than downstream, because
    once a cell estimate exists there is no way to tell how many trials it
    rests on.
    """
    accounting = {
        "n_trials_in": int(len(trials)),
        "used_usable_column": "usable" in trials.columns,
    }

    kept = trials.copy()

    if "usable" in kept.columns:
        kept = kept[kept["usable"].fillna(False).astype(bool)]
    else:
        rt_min = getattr(config, "RT_MIN_MS", 100)
        rt_max = getattr(config, "RT_MAX_MS", 900)
        kept = kept[kept["rt_ms"].between(rt_min, rt_max, inclusive="neither")]

        if verbose:
            print(
                "WARNING: no 'usable' column in the trial table, so H1a "
                f"applied the RT window ({rt_min:g} to {rt_max:g} ms) itself. "
                "Pass the QC-filtered tables to get the full exclusions."
            )
            print()

    kept = kept[kept["rt_ms"].notna()]
    accounting["n_trials_excluded"] = accounting["n_trials_in"] - len(kept)

    # Thin cells. Counted before removal so the report can name them.
    keys = _cell_keys(kept)
    sizes = kept.groupby(keys, dropna=False)["rt_ms"].transform("size")
    thin = sizes < min_cell_n

    accounting["n_cells_before"] = int(
        kept.groupby(keys, dropna=False).ngroups
    )
    accounting["thin_cells"] = (
        kept.loc[thin]
        .groupby(keys, dropna=False, as_index=False)["rt_ms"]
        .size()
        .rename(columns={"size": "n_trials"})
        .sort_values(keys)
        .reset_index(drop=True)
    )
    accounting["n_cells_thin"] = int(len(accounting["thin_cells"]))
    accounting["n_trials_in_thin_cells"] = int(thin.sum())

    kept = kept.loc[~thin]

    accounting["n_trials_used"] = int(len(kept))
    accounting["n_cells_used"] = int(
        kept.groupby(keys, dropna=False).ngroups if len(kept) else 0
    )
    accounting["min_cell_n"] = int(min_cell_n)

    return kept.reset_index(drop=True), accounting


def _preferred_from_aic(comparison):
    """
    Return the model with the lowest finite AIC.

    This intentionally does NOT apply an AIC margin, slope-direction rule,
    boundary-range rule, or precision rule. Those are reported separately and
    enter the H1a verdict through ``sigmoid_lower_aic``, so that the AIC
    comparison and the structural checks stay visibly distinct.
    """
    candidates = {
        "flat": comparison.get("aic_flat", np.nan),
        "linear": comparison.get("aic_linear", np.nan),
        "sigmoid": comparison.get("aic_sigmoid", np.nan),
    }

    finite = {
        name: float(value)
        for name, value in candidates.items()
        if value is not None and np.isfinite(value)
    }

    if not finite:
        return "fit_failed"

    return min(finite, key=finite.get)



def _fit_boundary_sigmoid(
    x,
    y,
    *,
    x_min=None,
    x_max=None,
    k_direction="positive",
    k_min=0.05,
    k_max=None,
    n_anchor=2,
):
    """Fit a sigmoid for boundary estimation, with the plateaus fixed.

    Seven sampled positions cannot support four free sigmoid parameters. When
    the low plateau, the high plateau, the midpoint and the slope are all
    estimated together they trade against one another: the optimiser moves a
    plateau outward and the midpoint follows, arriving at the edge of the
    admissible range without the residuals getting much worse. That is what
    produced midpoints pinned at the range bounds and standard errors in the
    tens of thousands.

    Here the plateaus are read from the data rather than fitted. The near
    plateau is the mean of the ``n_anchor`` nearest positions and the far
    plateau the mean of the ``n_anchor`` farthest, leaving only the midpoint
    and the slope to estimate. Five degrees of freedom remain on seven points
    instead of three, and the midpoint no longer runs to the bound.

    The assumption this buys is that the profile really is flat at both ends.
    Where it is still changing at the far edge, the far anchor is not a
    plateau, and the returned ``amplitude_ms`` will be small relative to the
    profile's own range. The caller should treat that as the signal that the
    transition has not been captured, in the same way a bound-pinned midpoint
    was treated under the four-parameter fit.

    ``k_direction`` states which way the transition should run: positive for
    facilitation, which is larger near the body, and negative for raw
    visuo-tactile reaction time, which is smaller near the body. A fit running
    the wrong way is reported with ``ok`` False rather than silently returned.

    The return contract matches the previous four-parameter helper, so the
    callers in this module need no change.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    failed = {
        "ok": False,
        "low": np.nan,
        "high": np.nan,
        "x_c": np.nan,
        "se_x_c": np.nan,
        "k": np.nan,
        "se_k": np.nan,
        "amplitude_ms": np.nan,
        "rss": np.nan,
        "k_at_bound": False,
    }

    if len(x) < 5 or len(np.unique(x)) < 4:
        return failed

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_lo = float(np.nanmin(x)) if x_min is None else float(x_min)
    x_hi = float(np.nanmax(x)) if x_max is None else float(x_max)

    # One position in from each edge, so a "boundary" cannot be reported at
    # the very edge of the sampled space.
    xc_lo = x_lo + 1.0
    xc_hi = x_hi - 1.0

    if not np.isfinite(xc_lo) or not np.isfinite(xc_hi) or xc_lo >= xc_hi:
        return failed

    n_anchor = int(max(1, min(n_anchor, len(x) // 2 - 1)))
    near_plateau = float(np.mean(y[:n_anchor]))
    far_plateau = float(np.mean(y[-n_anchor:]))
    amplitude = far_plateau - near_plateau

    if not np.isfinite(amplitude) or amplitude == 0.0:
        return failed

    # The exponent sign matches ``stats_utils.sigmoid``, which this module's
    # plotting evaluates the fitted parameters through. Fitting with the
    # opposite sign and plotting with this one draws the curve mirrored, so
    # the two must agree.
    def _curve(x_value, x_c, k):
        return near_plateau + amplitude / (1.0 + np.exp(k * (x_value - x_c)))

    # Default to the project-wide bound rather than an arbitrary large
    # number, so a direct call behaves the same way as the calls this module
    # makes internally. Those pass config.K_MAX explicitly, and a helper that
    # silently used a different bound would make the two disagree.
    slope_bound = (
        float(getattr(config, "K_MAX", 10.0)) if k_max is None
        else float(k_max)
    )
    bounds = ([xc_lo, -slope_bound], [xc_hi, slope_bound])

    best = None
    best_rss = np.inf

    for x_c0 in np.linspace(xc_lo, xc_hi, 5):
        for k0 in (0.5, 2.0, 5.0):
            for sign in (1.0, -1.0):
                try:
                    popt, pcov = curve_fit(
                        _curve,
                        x,
                        y,
                        p0=[float(x_c0), sign * float(k0)],
                        bounds=bounds,
                        maxfev=5000,
                    )
                except Exception:
                    continue

                rss = float(np.sum((y - _curve(x, *popt)) ** 2))

                if np.isfinite(rss) and rss < best_rss:
                    best = (popt, pcov)
                    best_rss = rss

    if best is None:
        return failed

    popt, pcov = best
    x_c = float(popt[0])
    k_raw = float(popt[1])

    if pcov is not None and np.all(np.isfinite(pcov)):
        errors = np.sqrt(np.diag(pcov))
        se_x_c = float(errors[0])
        se_k = float(errors[1])
    else:
        se_x_c = np.nan
        se_k = np.nan

    # Canonicalise the slope so its sign describes the direction of the
    # transition in the profile, not the sign the optimiser happened to use.
    # Positive means the profile is larger near rank 1 and falls with
    # distance, which is the facilitation case; negative means it is smaller
    # near rank 1 and rises, which is raw visuo-tactile reaction time. This
    # matches the convention the callers in this module already use.
    #
    # The multiplier follows from the exponent sign above: with
    # exp(+k(x - x_c)) the raw slope is negative for a profile that starts
    # high and falls, so the signed amplitude has to flip it back.
    k = k_raw * float(np.sign(far_plateau - near_plateau))

    at_bound = bool(
        np.isfinite(k_raw)
        and abs(abs(k_raw) - slope_bound) < 1e-6
    )

    # A fit can be rejected for running the wrong way or for being too
    # shallow to locate, and neither is a statement about how precisely the
    # midpoint was estimated. The standard errors are therefore carried
    # through rather than left as NaN: a caller comparing a rejected fit with
    # an accepted one should see the same fields for both, and a NaN reads as
    # "the optimiser produced nothing", which is not what happened.
    def _rejected():
        return {
            **failed,
            "low": near_plateau,
            "high": far_plateau,
            "near_plateau_ms": near_plateau,
            "far_plateau_ms": far_plateau,
            "x_c": x_c,
            "se_x_c": se_x_c,
            "k": k,
            "se_k": se_k,
            "amplitude_ms": abs(float(amplitude)),
            "rss": best_rss,
            "k_at_bound": at_bound,
        }

    if k_direction == "positive" and not k > 0:
        return _rejected()

    if k_direction == "negative" and not k < 0:
        return _rejected()

    if np.isfinite(k) and abs(k) < float(k_min):
        return _rejected()

    # ``low`` and ``high`` are reported in the sense ``stats_utils.sigmoid``
    # expects, not by position: that function returns ``high`` as x goes to
    # minus infinity and ``low`` as x goes to plus infinity when k is
    # positive, and the reverse when k is negative. Naming them by position
    # instead would draw every curve mirrored, because the plotting evaluates
    # the fitted parameters straight through that equation.
    #
    # ``near_plateau_ms`` and ``far_plateau_ms`` are carried alongside for
    # anyone who wants the plateaus by position without decoding the sign.
    if k > 0:
        low, high = far_plateau, near_plateau
    else:
        low, high = near_plateau, far_plateau

    return {
        "ok": True,
        "low": low,
        "high": high,
        "near_plateau_ms": near_plateau,
        "far_plateau_ms": far_plateau,
        "x_c": x_c,
        "se_x_c": se_x_c,
        "k": k,
        "se_k": se_k,
        "amplitude_ms": abs(float(amplitude)),
        "rss": best_rss,
        "k_at_bound": at_bound,
    }
def _transition_width_10_90(k):
    """Width, in rank units, over which a logistic goes from 10% to 90%."""
    if not np.isfinite(k) or k == 0:
        return np.nan
    return float(2.0 * np.log(9.0) / abs(k))


def _observed_halfmax(profile, value_column="facilitation_ms"):
    """Observed half-maximum crossover, walking inward from D7 to D1.

    The far plateau is the mean of D7/D6 and the near plateau the mean of
    D1/D2.  The first adjacent pair encountered when walking inward that
    straddles the half-dynamic-range level is linearly interpolated.

    ``value_column`` names the quantity being bounded. Facilitation is a
    difference between two conditions; a reaction time is not. They are
    different quantities and the column is stated rather than assumed, so
    that a reaction-time profile cannot be read as though it were a
    facilitation profile.
    """
    one = (
        profile[["position_rank", value_column]]
        .dropna()
        .sort_values("position_rank", ascending=False)
        .drop_duplicates("position_rank")
    )
    values = dict(
        zip(
            one["position_rank"].astype(float),
            one[value_column].astype(float),
        )
    )
    if not all(float(d) in values for d in (1, 2, 6, 7)):
        return np.nan

    far = np.mean([values[7.0], values[6.0]])
    near = np.mean([values[1.0], values[2.0]])
    half = 0.5 * (far + near)

    x = one["position_rank"].to_numpy(float)
    y = one[value_column].to_numpy(float)
    z = y - half

    for i in range(len(x) - 1):
        if z[i] == 0:
            return float(x[i])
        if z[i] * z[i + 1] <= 0:
            if y[i + 1] == y[i]:
                return float((x[i] + x[i + 1]) / 2.0)
            return float(x[i] + (half - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))

    if len(x) and z[-1] == 0:
        return float(x[-1])
    return np.nan


def _bootstrap_subject_boundary(
    subject_trials,
    estimator,
    correction,
    x_min,
    x_max,
    n_boot=100,
    seed=20260823,
    min_transition_width=1.0,
    max_transition_width=None,
    target="facilitation",
):
    """Trial-level bootstrap of x_c for one participant.

    Trials are resampled WITHIN the cells that define the correction
    (session/speed/condition/position where present), so the experimental
    structure and each cell's sample size are preserved.

    Each resample is put through ``h1.facilitation_profile``, so what is
    bootstrapped is a facilitation profile. Calling this for a
    reaction-time fit would attach an interval computed on one quantity
    to a fit of another, which is why ``target`` is checked rather than
    ignored.
    """
    if target != "facilitation":
        raise NotImplementedError(
            "This bootstrap resamples facilitation profiles. A "
            "reaction-time boundary needs its own resampler; reusing "
            "this one would report an interval computed on a different "
            "quantity from the fit it is attached to."
        )

    if n_boot is None or int(n_boot) <= 0 or len(subject_trials) == 0:
        return {
            "bootstrap_n": 0,
            "bootstrap_n_valid": 0,
            "bootstrap_success_rate": np.nan,
            "bootstrap_x_c_median": np.nan,
            "bootstrap_x_c_ci_low": np.nan,
            "bootstrap_x_c_ci_high": np.nan,
            "bootstrap_x_c_ci_width": np.nan,
        }

    n_boot = int(n_boot)
    rng = np.random.default_rng(seed)
    cell_keys = [c for c in ["session", "speed", "sensory_condition", "position_rank"] if c in subject_trials.columns]
    groups = [g.reset_index(drop=True) for _, g in subject_trials.groupby(cell_keys, dropna=False)]
    valid_xc = []
    max_width = float(x_max - x_min) if max_transition_width is None else float(max_transition_width)

    for _ in range(n_boot):
        pieces = []
        for g in groups:
            take = rng.integers(0, len(g), size=len(g))
            pieces.append(g.iloc[take])
        boot_trials = pd.concat(pieces, ignore_index=True)

        try:
            prof_speed = h1.facilitation_profile(
                boot_trials,
                estimator=estimator,
                correction=correction,
            )
            prof = h1.subject_profile(prof_speed).dropna(subset=["facilitation_ms"])
            if len(prof) < 5:
                continue
            fit = _fit_boundary_sigmoid(
                prof["position_rank"].to_numpy(float),
                prof["facilitation_ms"].to_numpy(float),
                x_min=x_min,
                x_max=x_max,
                k_direction="positive",
                k_max=getattr(config, "K_MAX", 10.0),
            )
        except Exception:
            continue

        xc = fit.get("x_c", np.nan)
        k = fit.get("k", np.nan)
        width = _transition_width_10_90(k)
        # H1a's sigmoid convention expects positive k for facilitation because
        # rank 1 is near and rank 7 is far.
        geometry_ok = (
            fit.get("ok", False)
            and np.isfinite(xc)
            and x_min <= xc <= x_max
            and np.isfinite(k)
            and k > 0
            and not fit.get("k_at_bound", False)
            and np.isfinite(width)
            and min_transition_width <= width <= max_width
        )
        if geometry_ok:
            valid_xc.append(float(xc))

    if valid_xc:
        arr = np.asarray(valid_xc, dtype=float)
        lo, hi = np.quantile(arr, [0.025, 0.975])
        med = np.median(arr)
        width = hi - lo
    else:
        lo = hi = med = width = np.nan

    return {
        "bootstrap_n": n_boot,
        "bootstrap_n_valid": int(len(valid_xc)),
        "bootstrap_success_rate": float(len(valid_xc) / n_boot),
        "bootstrap_x_c_median": float(med) if np.isfinite(med) else np.nan,
        "bootstrap_x_c_ci_low": float(lo) if np.isfinite(lo) else np.nan,
        "bootstrap_x_c_ci_high": float(hi) if np.isfinite(hi) else np.nan,
        "bootstrap_x_c_ci_width": float(width) if np.isfinite(width) else np.nan,
    }


def _add_boundary_diagnostics(
    comparison,
    profile,
    subject_trials=None,
    estimator="median",
    correction="delay_matched",
    x_min=1.0,
    x_max=7.0,
    n_boot=500,
    bootstrap_seed=20260823,
    min_boot_success=0.80,
    max_boot_ci_width=2.0,
    min_transition_width=1.0,
    max_transition_width=None,
    value_column="facilitation_ms",
):
    """Add pre-specified, interpretable boundary-estimability diagnostics.

    ``value_column`` is the quantity the profile holds. The half-maximum
    fallback reads it directly, so it has to be named: a reaction-time
    profile and a facilitation profile are not interchangeable inputs.
    """
    out = dict(comparison)
    k = out.get("k", np.nan)
    xc = out.get("x_c", np.nan)
    width = _transition_width_10_90(k)
    max_width = float(x_max - x_min) if max_transition_width is None else float(max_transition_width)

    out["transition_width_10_90"] = width
    k_max = float(getattr(config, "K_MAX", 10.0))
    k_at_bound = bool(
        out.get("k_at_bound", False)
        or (np.isfinite(k) and np.isclose(abs(k), k_max, rtol=0.0, atol=0.02))
    )
    # The comparison is on the absolute slope. A fit that ran to the negative
    # end of the bound is pinned in exactly the same way as one that ran to
    # the positive end, and comparing the signed value would recognise only
    # half of them. That matters because the transition-width check below
    # excludes bound-pinned fits: a fit sitting on the bound has a width of
    # 2*ln(9)/K_MAX by construction, so its width says nothing about the data
    # and would otherwise pass or fail purely according to where the bound
    # was set.
    out["k_at_bound"] = bool(
        np.isfinite(k)
        and np.isclose(
            abs(k),
            abs(float(getattr(config, "K_MAX", 10.0))),
            rtol=0.0,
            atol=0.02,
        )
    )
    out["transition_direction_ok"] = bool(
        np.isfinite(k)
        and k > 0
    )
    out["transition_width_ok"] = bool(
        np.isfinite(width)
        and min_transition_width <= width <= max_width
        and not out["k_at_bound"]
    )
    out["x_c_in_range"] = bool(np.isfinite(xc) and x_min <= xc <= x_max)

    # Recompute legacy flags from the bounded boundary fit so they are not
    # inherited from the unconstrained AIC fit.
    se_xc = out.get("se_x_c", np.nan)
    out["k_ok"] = bool(
        out["transition_direction_ok"] and out["transition_width_ok"]
    )
    out["x_c_precise"] = bool(
        np.isfinite(se_xc)
        and se_xc > 0
        and se_xc <= 1.5
        and out["x_c_in_range"]
        and out["k_ok"]
    )

    if subject_trials is not None:
        boot = _bootstrap_subject_boundary(
            subject_trials,
            estimator=estimator,
            correction=correction,
            x_min=x_min,
            x_max=x_max,
            n_boot=n_boot,
            seed=bootstrap_seed,
            min_transition_width=min_transition_width,
            max_transition_width=max_width,
        )
    else:
        boot = {
            "bootstrap_n": 0,
            "bootstrap_n_valid": 0,
            "bootstrap_success_rate": np.nan,
            "bootstrap_x_c_median": np.nan,
            "bootstrap_x_c_ci_low": np.nan,
            "bootstrap_x_c_ci_high": np.nan,
            "bootstrap_x_c_ci_width": np.nan,
        }
    out.update(boot)

    out["bootstrap_precise"] = bool(
        np.isfinite(out["bootstrap_success_rate"])
        and out["bootstrap_success_rate"] >= min_boot_success
        and np.isfinite(out["bootstrap_x_c_ci_width"])
        and out["bootstrap_x_c_ci_width"] <= max_boot_ci_width
    )

    out["boundary_estimable"] = bool(
        out.get("sigmoid_ok", False)
        and out["transition_direction_ok"]
        and out["transition_width_ok"]
        and out["x_c_in_range"]
        and out["bootstrap_precise"]
    )

    half = _observed_halfmax(profile, value_column=value_column)
    out["halfmax_rank"] = half
    out["halfmax_m"] = rank_to_metres(half) if np.isfinite(half) else np.nan

    # This boundary field is descriptive in H1a. H1a's SHAPE verdict below
    # still additionally requires the sigmoid to win the preregistered AIC
    # comparison. Downstream speed-specific analyses should use the same helper
    # and apply the same method to both speeds within a participant.
    if out["boundary_estimable"]:
        out["boundary_method"] = "sigmoid"
        out["boundary_rank"] = float(xc)
    elif np.isfinite(half):
        out["boundary_method"] = "halfmax"
        out["boundary_rank"] = float(half)
    else:
        out["boundary_method"] = "not_estimable"
        out["boundary_rank"] = np.nan
    out["boundary_m"] = (
        rank_to_metres(out["boundary_rank"])
        if np.isfinite(out["boundary_rank"])
        else np.nan
    )
    return out


def _boundary_is_usable(comparison):
    """True when the sigmoid boundary clears the new identifiability checks."""
    return bool(comparison.get("boundary_estimable", False))


def _boundary_failure_reasons(comparison):
    """Plain-language list of which new boundary checks failed."""
    reasons = []
    if not comparison.get("sigmoid_ok", False):
        return ["the sigmoid did not converge"]

    k = comparison.get("k", np.nan)
    width = comparison.get("transition_width_10_90", np.nan)
    xc = comparison.get("x_c", np.nan)

    if not comparison.get("transition_direction_ok", False):
        reasons.append(f"the transition runs in the wrong direction (k = {k:.3g})")
    if not comparison.get("transition_width_ok", False):
        reasons.append(
            f"the 10–90% transition is not resolved by the sampled range "
            f"(width = {width:.2f} ranks)"
            if np.isfinite(width)
            else "the transition width could not be evaluated"
        )
    if not comparison.get("x_c_in_range", False):
        reasons.append(f"the boundary falls outside the sampled range (x_c = {xc:.2f})")
    if not comparison.get("bootstrap_precise", False):
        rate = comparison.get("bootstrap_success_rate", np.nan)
        ciw = comparison.get("bootstrap_x_c_ci_width", np.nan)
        reasons.append(
            "the boundary is not bootstrap-localised "
            f"(valid = {rate:.0%}, 95% CI width = {ciw:.2f} ranks)"
            if np.isfinite(rate) and np.isfinite(ciw)
            else "the boundary could not be bootstrap-localised"
        )
    return reasons


def run(
    t,
    estimator="median",
    correction="delay_matched",
    exclude=None,
    min_cell_n=3,
    n_boot=1, # CHANGE!
    bootstrap_seed=20260823,
    min_boot_success=0.80,
    max_boot_ci_width=2.0,
    min_transition_width=1.0,
    max_transition_width=None,
    plot=True,
    verbose=True,
):
    """
    Run H1a.

    Parameters
    ----------
    t
        Analysis object containing ``t.pps_young``.

    estimator : {"mean", "median"}, default="median"
        RT summary used to build each participant's facilitation profile.
        ``median`` is the preregistered choice.

    correction : {"delay_matched", "overall_tactile", "fastest_tactile"},
        default="delay_matched"
        Tactile baseline correction used before fitting the models.
        ``delay_matched`` is the preregistered choice; the other methods are
        supplementary sensitivity analyses.

    min_cell_n : int, default=3
        Cells with fewer usable trials than this contribute no estimate. A
        median of one or two trials is not a summary of anything.

    n_boot : int, default=500
        Trial-level bootstrap resamples per participant for x_c localisation.
        Use 2000 or more for the final reported analysis.

    min_boot_success : float, default=.80
        Minimum proportion of bootstrap resamples yielding a geometrically
        valid sigmoid boundary.

    max_boot_ci_width : float, default=2.0
        Maximum 95% bootstrap CI width for x_c, in distance-rank units.

    min_transition_width : float, default=1.0
        Minimum resolvable 10–90% sigmoid transition width in rank units.
        Narrower transitions are effectively steps between sampled distances.

    max_transition_width : float or None
        Maximum 10–90% transition width. None uses the sampled rank span, so
        a transition broader than the experiment is treated as too shallow.

    plot : bool, default=True
        Whether to create and display the diagnostic figure.

    verbose : bool, default=True
        Whether to print the model-comparison report.

    Returns
    -------
    dict
        Model-comparison results, descriptive profiles, and optionally
        the generated figure.
    """

    if estimator not in {"mean", "median"}:
        raise ValueError("estimator must be 'mean' or 'median'")

    if correction not in h1.CORRECTIONS:
        raise ValueError(
            "correction must be one of: " + ", ".join(h1.CORRECTIONS)
        )

    results = {
        "estimator": estimator,
        "correction": correction,
        "is_preregistered": (
            estimator == "median"
            and correction == "delay_matched"
        ),
    }

    # ==================================================================
    # Step 1. Build the RT and facilitation profiles
    # ==================================================================
    #
    # H1a consumes the H1 calculation directly so that the estimator and the
    # baseline correction stay selectable without duplicating the correction
    # formulas. H1 returns one profile per subject and speed; H1a averages
    # over speed because H1a tests the overall shape.
    #
    # The raw RT profiles are descriptive only. They are plotted so that the
    # underlying T and VT reaction times can be inspected alongside the
    # facilitation values derived from them.

    trials_in = h1.get_trials(t)

    # Excluded here, before anything is fitted, so the raw profiles, the
    # facilitation values and the sigmoid all describe the same sample.
    # Filtering downstream would leave the figure showing participants
    # who contributed to none of the numbers beside it.
    if exclude:
        dropped = sorted(set(exclude) & set(trials_in["subject"]))
        trials_in = trials_in[~trials_in["subject"].isin(list(exclude))]

        if verbose and dropped:
            print(f"Excluded from H1a: {', '.join(dropped)}")

        missing = sorted(set(exclude) - set(dropped))

        if verbose and missing:
            print(f"  not present: {', '.join(missing)}")

    raw_rt_trials, accounting = prepare_trials(
        trials_in, min_cell_n=min_cell_n, verbose=verbose
    )
    results["trial_accounting"] = accounting
    results["excluded"] = sorted(exclude) if exclude else []

    if verbose:
        print(
            f"Trials entering H1a: {accounting['n_trials_used']} of "
            f"{accounting['n_trials_in']} "
            f"({accounting['n_trials_excluded']} excluded by QC, "
            f"{accounting['n_trials_in_thin_cells']} in cells below "
            f"n = {min_cell_n})"
        )
        print(
            f"Cells entering H1a: {accounting['n_cells_used']} of "
            f"{accounting['n_cells_before']} "
            f"({accounting['n_cells_thin']} too thin to summarise)"
        )
        print()

    raw_rt_per_subject = (
        raw_rt_trials
        .groupby(
            ["subject", "sensory_condition", "position_rank"],
            as_index=False,
        )["rt_ms"]
        .agg(estimator)
        .sort_values(["subject", "sensory_condition", "position_rank"])
    )

    # Average participants only after each participant's own profile exists,
    # so participants contribute equally despite unequal trial counts.
    raw_rt_group = (
        raw_rt_per_subject
        .groupby(["sensory_condition", "position_rank"], as_index=False)["rt_ms"]
        .mean()
        .sort_values(["sensory_condition", "position_rank"])
    )

    results["raw_rt_per_subject"] = raw_rt_per_subject
    results["raw_rt_group"] = raw_rt_group

    profile_by_speed = h1.facilitation_profile(
        raw_rt_trials,
        estimator=estimator,
        correction=correction,
    )

    profile_per_subject = h1.subject_profile(profile_by_speed)

    # A distance with no facilitation value cannot be fitted. Dropping it here
    # rather than inside the fit keeps the reported profile and the fitted
    # profile identical.
    n_profile_points = len(profile_per_subject)
    profile_per_subject = profile_per_subject.dropna(
        subset=["facilitation_ms"]
    ).reset_index(drop=True)

    accounting["n_profile_points_dropped"] = int(
        n_profile_points - len(profile_per_subject)
    )
    accounting["points_per_subject"] = (
        profile_per_subject
        .groupby("subject", as_index=False)["facilitation_ms"]
        .size()
        .rename(columns={"size": "n_distances"})
    )

    results["profile_by_speed"] = profile_by_speed
    results["profile_per_subject"] = profile_per_subject

    if verbose and accounting["n_profile_points_dropped"]:
        print(
            f"WARNING: {accounting['n_profile_points_dropped']} distance(s) "
            "had no facilitation value and were dropped from the profiles. "
            "Participants with fewer than five distances cannot support a "
            "sigmoid fit at all."
        )
        print(
            accounting["points_per_subject"]
            .query("n_distances < 7")
            .to_string(index=False)
        )
        print()

    # ==================================================================
    # Step 2. Determine the range of distance ranks actually fitted
    # ==================================================================
    #
    # The range is taken from the profile that is fitted, not from the full
    # trial table, so that the x_c-inside-range check refers to the same
    # distances the models saw.

    fitted_ranks = profile_per_subject["position_rank"].dropna()

    if len(fitted_ranks) > 0:
        x_min = float(fitted_ranks.min())
        x_max = float(fitted_ranks.max())
    else:
        x_min = 1.0
        x_max = 7.0

    results["x_min"] = x_min
    results["x_max"] = x_max

    if verbose:
        print(
            f"Distance ranks entering the fit: "
            f"{x_min:.0f} to {x_max:.0f}"
        )
        print()

    # ==================================================================
    # Step 3. Fit flat, linear, and sigmoid models per participant
    # ==================================================================

    rows = []

    for subject, one_subject in profile_per_subject.groupby("subject"):

        one_subject = one_subject.sort_values("position_rank")

        comparison = compare_models_per_participant(
            one_subject["position_rank"].to_numpy(),
            one_subject["facilitation_ms"].to_numpy(),
            x_min=x_min,
            x_max=x_max,
            aic_margin=config.AIC_MARGIN,
            k_direction="positive",
        )

        # compare_models_per_participant applies the structural checks when it
        # fills "preferred". H1a keeps the AIC comparison and the checks
        # visibly separate, so "preferred" here is the AIC winner alone and
        # the checks enter through sigmoid_lower_aic below.
        comparison["preferred"] = _preferred_from_aic(comparison)

        # AIC/AICc above remain the preregistered model comparison. Boundary
        # parameters are estimated separately with a bounded multi-start fit so
        # an effectively vertical step cannot masquerade as a precise boundary.
        boundary_fit = _fit_boundary_sigmoid(
            one_subject["position_rank"].to_numpy(float),
            one_subject["facilitation_ms"].to_numpy(float),
            x_min=x_min,
            x_max=x_max,
            k_direction="positive",
            k_max=getattr(config, "K_MAX", 10.0),
        )

        comparison["low"] = boundary_fit.get("low", np.nan)
        comparison["high"] = boundary_fit.get("high", np.nan)
        comparison["x_c"] = boundary_fit.get("x_c", np.nan)
        comparison["se_x_c"] = boundary_fit.get("se_x_c", np.nan)
        comparison["k"] = boundary_fit.get("k", np.nan)
        comparison["se_k"] = boundary_fit.get("se_k", np.nan)
        comparison["amplitude_ms"] = boundary_fit.get("amplitude_ms", np.nan)
        comparison["k_at_bound"] = boundary_fit.get("k_at_bound", False)

        comparison = _add_boundary_diagnostics(
            comparison,
            one_subject,
            subject_trials=raw_rt_trials[raw_rt_trials["subject"] == subject],
            estimator=estimator,
            correction=correction,
            x_min=x_min,
            x_max=x_max,
            n_boot=n_boot,
            bootstrap_seed=bootstrap_seed + len(rows),
            min_boot_success=min_boot_success,
            max_boot_ci_width=max_boot_ci_width,
            min_transition_width=min_transition_width,
            max_transition_width=max_transition_width,
        )

        rows.append({"subject": subject, **comparison})

    # Explicit columns ensure that an empty table still has the expected
    # structure and does not crash later code.
    table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)

    # A fit that did not converge is recorded as NA rather than False, so that
    # a failed fit is never silently counted as evidence against the sigmoid.
    fitted = (
        table["sigmoid_ok"].fillna(False).astype(bool)
        & table["aic_sigmoid"].notna()
        & table["aic_flat"].notna()
        & table["aic_linear"].notna()
    )

    aic_winner = (
    table["preferred"]
    == "sigmoid"
    )

    # ----------------------------------------------------------
    # H1a SHAPE:
    # sigmoid has the lowest AIC
    # ----------------------------------------------------------

    table["sigmoid_lower_aic"] = np.where(
        fitted,
        aic_winner,
        np.nan,
    )

    # ----------------------------------------------------------
    # BOUNDARY ESTIMABILITY:
    # separate from model-shape preference
    # ----------------------------------------------------------

    table["sigmoid_boundary_estimable"] = np.where(
        fitted,
        table["boundary_estimable"]
        .fillna(False)
        .astype(bool),
        np.nan,
    )

    results["table"] = table
    results["n_fit_failed"] = int(
        (~fitted).sum()
    )

    results["table"] = table
    results["n_fit_failed"] = int((~fitted).sum())

    # ==================================================================
    # Step 4. Stop cleanly if no participant has usable data
    # ==================================================================

    if len(table) == 0:

        print("H1a: no participant has usable facilitation data.")
        print()
        print("Check that:")
        print("  - young_subjects is not empty")
        print(
            "  - VT and T trials exist at the same position "
            "within each subject/session"
        )
        print("  - rt_ms is computable for both T and VT trials")

        results["group_average"] = pd.DataFrame(
            columns=["position_rank", "facilitation_ms"]
        )
        results["group_summary"] = pd.DataFrame()
        results["group_comparison"] = None
        results["group_sigmoid_fit"] = None
        results["group_x_c_m"] = np.nan
        results["group_boundary_usable"] = False
        results["n_supported"] = 0
        results["n_total"] = 0
        results["n_preferred"] = 0
        results["n_identifiable"] = 0
        results["n_estimable"] = 0

        return results

    # ==================================================================
    # Step 5. Calculate and model the group-average facilitation profile
    # ==================================================================
    #
    # The group profile is descriptive. The preregistered inference is based
    # on the participant-level model comparisons above.

    group_stats = h1.group_summary(profile_per_subject)

    group_average = (
        group_stats
        .rename(columns={"group_mean": "facilitation_ms"})
        [["position_rank", "facilitation_ms"]]
    )

    results["group_summary"] = group_stats
    results["group_average"] = group_average

    enough_points = len(group_average) >= 5

    if enough_points:
        group_comparison = compare_models_per_participant(
            group_average["position_rank"].to_numpy(),
            group_average["facilitation_ms"].to_numpy(),
            x_min=x_min,
            x_max=x_max,
            aic_margin=config.AIC_MARGIN,
            k_direction="positive",
        )

        # Same separation as the participant-level analysis.
        group_comparison["preferred"] = _preferred_from_aic(group_comparison)

        # AIC/AICc remain those of the preregistered comparison. The curve used
        # for boundary estimation and plotting is the same stable bounded fit
        # used at participant level and in the bootstrap.
        group_sigmoid_fit = _fit_boundary_sigmoid(
            group_average["position_rank"].to_numpy(float),
            group_average["facilitation_ms"].to_numpy(float),
            x_min=x_min,
            x_max=x_max,
            k_direction="positive",
            k_max=getattr(config, "K_MAX", 10.0),
        )
        group_comparison["low"] = group_sigmoid_fit.get("low", np.nan)
        group_comparison["high"] = group_sigmoid_fit.get("high", np.nan)
        group_comparison["x_c"] = group_sigmoid_fit.get("x_c", np.nan)
        group_comparison["se_x_c"] = group_sigmoid_fit.get("se_x_c", np.nan)
        group_comparison["k"] = group_sigmoid_fit.get("k", np.nan)
        group_comparison["se_k"] = group_sigmoid_fit.get("se_k", np.nan)
        group_comparison["amplitude_ms"] = group_sigmoid_fit.get("amplitude_ms", np.nan)
        group_comparison["k_at_bound"] = group_sigmoid_fit.get("k_at_bound", False)
    else:
        group_comparison = {
            key: np.nan for key in COMPARISON_COLUMNS if key != "subject"
        }
        group_comparison["preferred"] = "insufficient data"
        group_sigmoid_fit = {"ok": False}

    # The group profile is descriptive.  We apply the geometry/half-max logic
    # but do not use it for participant-level inference.  A trial bootstrap of
    # the group would mix within- and between-participant uncertainty, so the
    # report does not label a group x_c as inferentially estimable.
    if enough_points:
        group_comparison = _add_boundary_diagnostics(
            group_comparison,
            group_average,
            subject_trials=None,
            estimator=estimator,
            correction=correction,
            x_min=x_min,
            x_max=x_max,
            n_boot=0,
            min_boot_success=min_boot_success,
            max_boot_ci_width=max_boot_ci_width,
            min_transition_width=min_transition_width,
            max_transition_width=max_transition_width,
        )

    results["group_comparison"] = group_comparison
    results["group_sigmoid_fit"] = group_sigmoid_fit

    # Group curve is descriptive, so its display check is geometry-only. The
    # participant-level H1a verdict still uses bootstrap localisation.
    group_boundary_usable = bool(
        group_comparison.get("sigmoid_ok", False)
        and group_comparison.get("transition_direction_ok", False)
        and group_comparison.get("transition_width_ok", False)
        and group_comparison.get("x_c_in_range", False)
    )
    results["group_boundary_usable"] = group_boundary_usable
    results["group_boundary_reasons"] = (
        [] if group_boundary_usable else _boundary_failure_reasons(group_comparison)
    )

    if (
        group_sigmoid_fit is not None
        and group_sigmoid_fit.get("ok", False)
        and np.isfinite(group_sigmoid_fit.get("x_c", np.nan))
    ):
        results["group_x_c_m"] = rank_to_metres(group_sigmoid_fit["x_c"])
    else:
        results["group_x_c_m"] = np.nan

    # ==================================================================
    # Step 6. Count participant-level support
    # ==================================================================

    results["n_total"] = len(table)

    # H1a shape support:
    # sigmoid is the lowest-AIC model. This is the same criterion the
    # reaction-time fits use, so the two are comparable.
    results["n_supported"] = int(
        (
            table["sigmoid_lower_aic"]
            == 1
        ).sum()
    )
    results["criterion_supported"] = "sigmoid has the lowest AIC"

    # Boundary estimability:
    # separate question from whether sigmoid wins AIC
    results["n_estimable"] = int(
        (
            table["sigmoid_boundary_estimable"]
            == 1
        ).sum()
    )

    results["n_preferred"] = int(
        aic_winner.sum()
    )

    results["n_identifiable"] = int(
        (
            table["transition_direction_ok"]
            .fillna(False)
            .astype(bool)

            & table["transition_width_ok"]
            .fillna(False)
            .astype(bool)

            & table["x_c_in_range"]
            .fillna(False)
            .astype(bool)
        ).sum()
    )
    # ==================================================================
    # Step 7. Print and plot
    # ==================================================================

    if verbose:
        report(results)

    if plot:
        fig = make_figure(results)
        figures.save(fig, "h1a_sigmoid_shape")
        plt.show()
        results["figure"] = fig

        individual = plot_individual_fits(results)
        figures.save(individual, "h1a_individual_fits")
        plt.show()
        results["figure_individual"] = individual

    return results


def _format_value(value):
    """Format numeric and text values safely."""
    try:
        if pd.isna(value):
            return "nan"
    except TypeError:
        pass

    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def report(results):
    """Print the H1a model-comparison results."""

    table = results["table"]

    analysis_type = (
        "preregistered" if results["is_preregistered"]
        else "supplementary"
    )

    print(
        f"H1a ({analysis_type}): {results['estimator']} RTs, "
        f"{results['correction']} correction"
    )
    print("H1a: model comparison for each participant")
    print("  lower AIC = better fit")
    print(
        "  aic_delta_vs_linear < 0 means the sigmoid beats the line "
        "after its parameter penalty"
    )
    print(
        "  transition width + bootstrap localisation are the structural checks: "
        "a curve can win on AIC and still not describe an estimable boundary"
    )
    print()

    printable = [
        column for column in REPORT_COLUMNS if column in table.columns
    ]
    print(table[printable].round(2).to_string(index=False))
    print()
    print(
        "  (the full comparison, including AICc and every standard error, "
        "is in results['table'])"
    )
    print()

    accounting = results.get("trial_accounting") or {}

    if accounting:
        print("Trials and cells behind these fits")
        print(
            f"  trials: {accounting.get('n_trials_used')} used, "
            f"{accounting.get('n_trials_excluded')} excluded by QC, "
            f"{accounting.get('n_trials_in_thin_cells')} in cells below "
            f"n = {accounting.get('min_cell_n')}"
        )
        print(
            f"  cells:  {accounting.get('n_cells_used')} used, "
            f"{accounting.get('n_cells_thin')} too thin"
        )

        thin = accounting.get("thin_cells")
        if thin is not None and len(thin):
            print("  cells excluded for being too thin:")
            print("    " + thin.to_string(index=False).replace("\n", "\n    "))

        if not accounting.get("used_usable_column", True):
            print(
                "  NOTE: the trial table had no 'usable' column, so only the "
                "absolute RT window was applied here."
            )
        print()

    if results["n_fit_failed"]:
        print(
            f"WARNING: the sigmoid failed to fit in "
            f"{results['n_fit_failed']} participant(s). "
            "Those rows are NA, not False."
        )
        print()

    print("H1a: group-averaged profile (descriptive)")
    print()

    print(
        pd.Series(results["group_comparison"])
        .map(_format_value)
        .to_string()
    )

    group_x_c_m = results.get("group_x_c_m", np.nan)
    group_comparison = results.get("group_comparison") or {}

    print()

    if np.isfinite(group_x_c_m) and results.get("group_boundary_usable", False):
        print(
            f"Group sigmoid boundary: {group_x_c_m:.3f} m from the body "
            f"(x_c = {group_comparison.get('x_c', np.nan):.2f} ranks, "
            f"SE = {group_comparison.get('se_x_c', np.nan):.2f})"
        )
    else:
        print("Group sigmoid boundary is not reportable:")
        for reason in results.get("group_boundary_reasons", []):
            print(f"  - {reason}")
        print(
            "  The preregistered fallback applies: read the boundary as the "
            "half-maximum crossover interpolated from the observed profile."
        )

    print()

    n_total = results["n_total"]

    print("H1a verdict (preregistered criterion):")
    print(
        f"  sigmoid wins on AIC AND describes an estimable boundary "
        f"in {results['n_supported']} of {n_total} participants"
    )
    print()

    print("Where that count comes from:")
    print(
        f"  sigmoid has the lowest AIC in "
        f"{results['n_preferred']} of {n_total}"
    )
    print(
        f"  transition direction/width and x_c range are geometrically usable "
        f"in {results['n_identifiable']} of {n_total}"
    )
    print(
        f"  boundary localised by the bootstrap criterion in "
        f"{results['n_estimable']} of {n_total}"
    )

    if results["n_preferred"] > results["n_supported"]:
        print()
        print(
            f"  {results['n_preferred'] - results['n_supported']} "
            "participant(s) won on AIC with a curve that does not describe a "
            "boundary. A transition can fail because it is too steep to locate "
            "between sampled ranks, too shallow to complete within the sampled "
            "range, outside the range, or unstable under bootstrap resampling."
        )


def _plot_sigmoid_curve(
    ax, fit, x_min, x_max, label="Fitted sigmoid",
    colour=None, **kwargs,
):
    """
    Draw a fitted sigmoid from its parameters. Returns True if drawn.

    The curve is evaluated from the same equation used in the fit:
        y = sigmoid(x, low, high, x_c, k)

    ``colour`` defaults to the facilitation colour, since that is what
    these panels fit. Pass it explicitly when the curve describes a
    reaction time instead, so the curve matches the data under it.
    """
    colour = FACILITATION_COLOUR if colour is None else colour
    required = ("low", "high", "x_c", "k")

    if fit is None or not fit.get("ok", False):
        return False

    if not all(name in fit and np.isfinite(fit[name]) for name in required):
        return False

    smooth_x = np.linspace(x_min, x_max, 300)
    smooth_y = sigmoid(
        smooth_x,
        fit["low"],
        fit["high"],
        fit["x_c"],
        fit["k"],
    )

    ax.plot(
        smooth_x,
        smooth_y,
        linestyle="--",
        color=colour,
        label=label,
        **kwargs,
    )

    ax.axvline(
        fit["x_c"],
        linestyle=":",
        linewidth=1.1,
        color=colour,
        alpha=0.8,
        label="_nolegend_",
        zorder=2,
    )

    return True


# A participant whose profile runs against the group is worth finding in
# the raw traces, since the corrected profile cannot show whether the
# reversal was already in the reaction times. Named here rather than
# passed through every call, so the figure and the results text can point
# at the same person without an argument threaded through four functions.
HIGHLIGHT_SUBJECT = None
HIGHLIGHT_COLOUR = "#d1495b"


def _observed_label(results):
    """What the thick line is, given the estimator and the sample.

    "Observed group mean" was wrong twice over: the estimator is a
    parameter and is usually the median, and with a single participant
    there is no group to average over.
    """
    estimator = results.get("estimator", "median")

    profiles = results.get("profile_per_subject")
    n_subjects = (
        profiles["subject"].nunique()
        if profiles is not None and "subject" in profiles
        else 1
    )

    return f"Observed {estimator}" + (" (group)" if n_subjects > 1 else "")


def make_figure(results):
    """
    Build the H1a diagnostic figure.

    Panel A
        Raw RTs at each distance for T and VT trials. Grey is tactile-only,
        blue visuo-tactile; faint thin lines are individual participants,
        thick lines the group averages.

    Panel B
        Observed facilitation profile and the fitted group sigmoid, the
        latter evaluated directly from the sigmoid equation. The curve is
        drawn only when the fit describes an estimable boundary; otherwise
        the panel says why not, rather than showing a step function as
        though it were a boundary.
    """

    raw_rt_per_subject = results["raw_rt_per_subject"]
    raw_rt_group = results["raw_rt_group"]

    profile_per_subject = results["profile_per_subject"]
    group_average = results["group_average"]
    group_sigmoid_fit = results["group_sigmoid_fit"]

    estimator_label = results["estimator"].capitalize()
    correction_label = results["correction"].replace("_", " ")

    fig, axes = figures.new_figure(
        n_panels=2,
        width=getattr(figures, "TWO_COLUMN", 7.2),
        height=3.6,
    )

    ax_rt, ax_fac = axes[0], axes[1]

    # ==================================================================
    # Panel A. Raw RT profiles
    # ==================================================================

    trial_styles = {
        "T": {"linestyle": "--", "marker": "s", "label": "Tactile-only"},
        "VT": {"linestyle": "-", "marker": "o", "label": "Visuo-tactile"},
    }

    # Colour carries the condition; participants are separated from the group
    # average by opacity and line weight rather than by a second colour.
    for frame, is_group in (
        (raw_rt_per_subject, False),
        (raw_rt_group, True),
    ):
        keys = (
            ["sensory_condition"]
            if is_group
            else ["subject", "sensory_condition"]
        )

        for key, one_profile in frame.groupby(keys):
            # groupby with a list of keys yields a tuple; the sensory
            # condition is always the last element.
            if not isinstance(key, tuple):
                key = (key,)

            sensory_condition = key[-1]

            if sensory_condition not in trial_styles:
                continue

            plot_style = trial_styles[sensory_condition]
            one_profile = one_profile.sort_values("position_rank")

            # The highlighted participant keeps the condition's line style,
            # so their tactile and visuo-tactile traces stay
            # distinguishable from each other, and takes a separate colour
            # so both stand out from the rest of the sample.
            highlighted = (
                not is_group
                and HIGHLIGHT_SUBJECT is not None
                and key[0] == HIGHLIGHT_SUBJECT
            )

            if highlighted:
                colour = HIGHLIGHT_COLOUR
                width, alpha, size, order = 1.8, 1.0, 4, 5
                label = f"{HIGHLIGHT_SUBJECT}, {plot_style['label'].lower()}"
            else:
                colour = CONDITION_COLOUR[sensory_condition]
                width = 2.3 if is_group else 0.9
                alpha = 1.0 if is_group else 0.30
                size = 5 if is_group else 3
                order = 10 if is_group else 1
                label = plot_style["label"] if is_group else "_nolegend_"

            ax_rt.plot(
                one_profile["position_rank"],
                one_profile["rt_ms"],
                color=colour,
                linestyle=plot_style["linestyle"],
                marker=plot_style["marker"],
                markersize=size,
                linewidth=width,
                alpha=alpha,
                label=label,
                zorder=order,
            )

    ax_rt.set_ylabel(f"{estimator_label} reaction time (ms)")
    ax_rt.set_title("A. Raw reaction-time profiles")
    ax_rt.legend(frameon=False)

    # ==================================================================
    # Panel B. Facilitation and fitted sigmoid
    # ==================================================================

    for _, one_subject in profile_per_subject.groupby("subject"):
        one_subject = one_subject.sort_values("position_rank")

        ax_fac.plot(
            one_subject["position_rank"],
            one_subject["facilitation_ms"],
            marker="o",
            markersize=3,
            linestyle="-",
            color=FACILITATION_COLOUR,
            alpha=0.30,
            linewidth=0.9,
            label="_nolegend_",
            zorder=1,
        )

    ax_fac.plot(
        group_average["position_rank"],
        group_average["facilitation_ms"],
        marker="o",
        markersize=5,
        linestyle="-",
        color=FACILITATION_COLOUR,
        linewidth=2.0,
        label=_observed_label(results),
        zorder=10,
    )

    if results.get("group_boundary_usable", False):
        boundary_m = rank_to_metres(group_sigmoid_fit.get("x_c", np.nan))

        _plot_sigmoid_curve(
            ax_fac,
            group_sigmoid_fit,
            results["x_min"],
            results["x_max"],
            label=f"Fitted sigmoid ($x_c$ = {boundary_m:.2f} m)",
            linewidth=2.2,
            zorder=9,
        )
    else:
        # Drawing a step function here would read as a boundary. Say why the
        # curve is absent instead.
        ax_fac.text(
            0.03,
            0.03,
            "Sigmoid not shown:\nno estimable boundary",
            transform=ax_fac.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            color=style._MUTED,
        )

    ax_fac.axhline(
        0,
        linestyle="--",
        linewidth=0.8,
        color="gray",
        zorder=0,
    )

    ax_fac.set_ylabel(f"{estimator_label} facilitation (ms)")
    ax_fac.set_title(f"B. Facilitation and sigmoid ({correction_label})")
    ax_fac.legend(frameon=False)

    # ==================================================================
    # Shared x-axis formatting, in metres
    # ==================================================================

    tested_ranks = sorted(
        profile_per_subject["position_rank"].dropna().unique()
    )

    for ax in (ax_rt, ax_fac):
        h1._format_distance_axis(ax, unit="m", ranks=tested_ranks or None)

    fig.tight_layout()

    return fig


def plot_individual_fits(results, n_columns=4, save_path=None):
    """One panel per participant: their own profile and their own fit.

    The main figure shows every participant as a thin line behind the group
    mean, which is enough to see the spread but not enough to see whose
    profile did what. This gives each participant their own panel, with the
    model comparison printed beside their curve, so a count like "sigmoid in
    2 of 7" can be traced to which two and on what basis.

    A fitted sigmoid is drawn only where it passes the structural checks. A
    curve that won on AIC while failing them is a step function, and drawing
    it would suggest a boundary the data do not support; the panel says which
    check failed instead.

    Participants whose fit is not usable are titled in muted grey.
    """
    profile = results["profile_per_subject"]
    table = results["table"]

    subjects = sorted(profile["subject"].dropna().unique())
    n_panels = len(subjects) + 1
    n_rows = int(np.ceil(n_panels / n_columns))

    fig, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(3.7 * n_columns, 3.2 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )
    flat_axes = axes.ravel()

    tested = sorted(profile["position_rank"].dropna().unique())

    for ax, name in zip(flat_axes, subjects):
        one = profile[profile["subject"] == name].sort_values("position_rank")
        row = table[table["subject"] == name]

        usable = bool(row["sigmoid_lower_aic"].iloc[0] == 1) if len(row) else False

        ax.plot(
            one["position_rank"], one["facilitation_ms"],
            color=FACILITATION_COLOUR if usable else style._MUTED,
            marker="o", markersize=4.5, linewidth=2.0, zorder=4,
        )
        ax.axhline(0, linestyle="--", linewidth=0.8, color="gray", zorder=0)

        if len(row):
            row = row.iloc[0]

            # Refit rather than storing every curve: the comparison table
            # keeps the parameters, and evaluating the equation from them is
            # cheaper than carrying seven fit objects through the results.
            fit = {
                "ok": bool(row.get("sigmoid_ok", False)),
                "low": row.get("low", np.nan),
                "high": row.get("high", np.nan),
                "x_c": row.get("x_c", np.nan),
                "k": row.get("k", np.nan),
            }

            checks_pass = bool(row.get("boundary_estimable", False))

            if checks_pass:
                _plot_sigmoid_curve(
                    ax, fit, results["x_min"], results["x_max"],
                    label="_nolegend_", linewidth=1.8, zorder=3,
                )

            annotation = [
                f"AIC: {row.get('preferred', '')}",
                f"AICc: {row.get('preferred_aicc', '')}",
                f"$R^2$ = {row.get('r2_sigmoid', np.nan):.2f}",
            ]

            if not checks_pass:
                if not row.get("transition_direction_ok", False):

                    annotation.append(
                        f"k = {row.get('k', np.nan):.2f}, wrong direction"
                    )

                elif row.get("k_at_bound", False):

                    annotation.append(
                        f"k = {row.get('k', np.nan):.2f}, transition too steep"
                    )

                elif not row.get("transition_width_ok", False):

                    annotation.append(
                        f"transition width = "
                        f"{row.get('transition_width_10_90', np.nan):.2f} ranks"
                    )   

            ax.text(
                0.98, 0.03, "\n".join(annotation),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color=style.INK, linespacing=1.35,
            )

        ax.set_title(
            str(name), loc="left", fontsize=9.5,
            color=style.INK if usable else style._MUTED,
        )
        h1._format_distance_axis(ax, unit="m", ranks=tested or None)

    # Final panel: the group profile, on the same scale.
    ax = flat_axes[len(subjects)]
    group = results["group_average"].sort_values("position_rank")

    ax.plot(
        group["position_rank"], group["facilitation_ms"],
        color=FACILITATION_COLOUR, marker="o", markersize=5,
        linewidth=2.4, zorder=4,
    )
    ax.axhline(0, linestyle="--", linewidth=0.8, color="gray", zorder=0)

    if results.get("group_boundary_usable", False):
        _plot_sigmoid_curve(
            ax, results["group_sigmoid_fit"],
            results["x_min"], results["x_max"],
            label="_nolegend_", linewidth=2.0, zorder=3,
        )

    ax.text(
        0.98, 0.03,
        f"Group mean\n{results['n_supported']} of {results['n_total']} "
        "support a\nsigmoid boundary",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, color=style.INK, linespacing=1.35,
    )
    ax.set_title("Group", loc="left", fontsize=9.5)
    h1._format_distance_axis(ax, unit="m", ranks=tested or None)

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    for row_index in range(n_rows):
        axes[row_index, 0].set_ylabel(
            f"{results['estimator'].capitalize()} facilitation (ms)"
        )

    fig.suptitle(
        "Each participant's facilitation profile and model comparison "
        f"({results['correction'].replace('_', ' ')})",
        x=0.02, ha="left", y=1.0, color=style.INK,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# =====================================================================
# Supplementary H1a sensitivity analyses
# =====================================================================

def run_supplementary(t, min_cell_n=3, plot=True):
    """
    Repeat H1a for all estimator x correction combinations.

    The preregistered analysis remains estimator="median" with
    correction="delay_matched". The remaining combinations are supplementary
    sensitivity analyses.

    Returns
    -------
    dict
        ``runs`` contains the six complete H1a result dictionaries.
        ``summary`` contains one row per analysis combination.
        ``figure`` is a 2 x 3 grid when ``plot=True``:
            rows    = mean, median
            columns = delay-matched, overall tactile, fastest tactile
    """
    estimators = ("mean", "median")
    corrections = tuple(h1.CORRECTIONS)

    runs = {}
    summary_rows = []

    for estimator in estimators:
        for correction in corrections:
            one = run(
                t,
                estimator=estimator,
                correction=correction,
                min_cell_n=min_cell_n,
                plot=False,
                verbose=False,
            )
            runs[(estimator, correction)] = one

            group = one.get("group_comparison") or {}

            summary_rows.append({
                "estimator": estimator,
                "correction": correction,
                "analysis": (
                    "preregistered"
                    if one["is_preregistered"]
                    else "supplementary"
                ),
                "aic_flat": group.get("aic_flat", np.nan),
                "aic_linear": group.get("aic_linear", np.nan),
                "aic_sigmoid": group.get("aic_sigmoid", np.nan),
                "aic_delta_vs_linear": group.get(
                    "aic_delta_vs_linear", np.nan
                ),
                "aicc_delta_vs_linear": group.get(
                    "aicc_delta_vs_linear", np.nan
                ),
                "r2_sigmoid": group.get("r2_sigmoid", np.nan),
                "amplitude_ms": group.get("amplitude_ms", np.nan),
                "x_c_rank": group.get("x_c", np.nan),
                "se_x_c": group.get("se_x_c", np.nan),
                "x_c_m": one.get("group_x_c_m", np.nan),
                "k": group.get("k", np.nan),
                "k_ok": group.get("k_ok", False),
                "x_c_precise": group.get("x_c_precise", False),
                "boundary_usable": one.get("group_boundary_usable", False),
                "preferred": group.get("preferred", None),
                "preferred_aicc": group.get("preferred_aicc", None),
                "n_sigmoid_supported": one.get("n_supported", 0),
                "n_sigmoid_lowest_aic": one.get("n_preferred", 0),
                "n_fit_failed": one.get("n_fit_failed", 0),
                "n_participants": one.get("n_total", 0),
                "n_trials_used": (one.get("trial_accounting") or {}).get(
                    "n_trials_used", np.nan
                ),
                "n_cells_thin": (one.get("trial_accounting") or {}).get(
                    "n_cells_thin", np.nan
                ),
            })

    output = {"runs": runs, "summary": pd.DataFrame(summary_rows)}

    if not plot:
        return output

    fig, axes = h1.correction_grid(
        estimators=estimators,
        corrections=corrections,
        figsize=(13.5, 8.0),
        ylabel="{estimator} facilitation (ms)",
    )

    for row, estimator in enumerate(estimators):
        for col, correction in enumerate(corrections):
            ax = axes[row, col]
            one = runs[(estimator, correction)]

            observed = one["group_average"].sort_values("position_rank")

            # Participants are averaged only after each participant's own
            # corrected seven-distance profile is built.
            ax.plot(
                observed["position_rank"],
                observed["facilitation_ms"],
                marker="o",
                markersize=5,
                color=FACILITATION_COLOUR,
                linewidth=2,
                label=_observed_label(one),
                zorder=10,
            )

            if one.get("group_boundary_usable", False):
                _plot_sigmoid_curve(
                    ax,
                    one["group_sigmoid_fit"],
                    one["x_min"],
                    one["x_max"],
                    linewidth=2,
                )

            h1._format_distance_axis(ax)

            if row == 0:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            annotation = (
                "Preregistered"
                if one["is_preregistered"]
                else "Supplementary"
            )

            group = one.get("group_comparison") or {}
            x_c_m = one.get("group_x_c_m", np.nan)
            delta = group.get("aic_delta_vs_linear", np.nan)

            if one.get("group_boundary_usable", False) and np.isfinite(x_c_m):
                annotation += f"\n$x_c$ = {x_c_m:.2f} m"
            else:
                annotation += "\nno estimable boundary"

            if np.isfinite(delta):
                annotation += f"\n$\\Delta$AIC vs linear = {delta:+.1f}"

            ax.text(
                0.03,
                0.97,
                annotation,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color=style.INK,
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
        )

    fig.suptitle(
        "H1a sensitivity analyses: sigmoid fits across RT summaries "
        "and corrections",
        y=1.04,
    )
    fig.tight_layout()

    output["figure"] = fig

    return output


# =====================================================================
# Fitting the sigmoid to visuo-tactile RT, as in Noel et al.
# =====================================================================
#
# The preregistered analysis fits the sigmoid to facilitation, T - VT at each
# distance. That requires a tactile estimate at every distance, and at four
# trials per cell the tactile term carries more sampling noise than the effect
# it is subtracted from.
#
# Noel et al. fit the sigmoid to the visuo-tactile reaction times themselves.
# Their unimodal tactile trials are collected at two anchor distances only and
# serve to establish that facilitation exists, not to correct each distance.
# The consequence is that their fitted curve never contains a four-trial term.
#
# This section runs that version on the visuo-tactile profile. It is a
# different quantity, not a better estimate of the same one: the facilitation
# fit asks where the multisensory advantage changes, and the RT fit asks where
# the response to the looming stimulus changes. Under the delay-matched logic
# they differ by whatever the tactile baseline contributes, which is exactly
# what ``baseline_contribution`` quantifies. Report both, and say which is
# which.


def run_rt(
    t,
    condition="VT",
    estimator="median",
    by_speed=True,
    min_cell_n=None,
    centre=True,
    plot=True,
    verbose=True,
):
    """Fit flat, linear and sigmoid to reaction times rather than facilitation.

    Parameters
    ----------
    condition : {"VT", "T"}, default="VT"
        Which reaction times to fit. VT is the comparison to Noel et al.; T is
        included so the tactile profile can be put through the identical
        procedure, which is the control for whether any apparent structure is
        specific to the multisensory condition.
    by_speed : bool, default=True
        Fit each speed separately. The hypotheses are stated per speed, and a
        profile present at one speed and absent at the other is diluted by
        averaging them.
    centre : bool, default=True
        Subtract each participant's own mean before fitting, so the comparison
        is of profile shape and not of overall speed. The fitted plateaus then
        read as deviations from that participant's mean rather than as
        reaction times.

    Returns
    -------
    dict
        ``table`` holds one row per participant (and speed) with the same
        model-comparison columns as ``run``. ``profile`` holds the fitted
        profiles. ``n_supported`` counts participants whose sigmoid wins on
        AIC and describes an estimable boundary, on the same criterion.
    """
    if condition not in ("T", "VT"):
        raise ValueError("condition must be 'T' or 'VT'")

    trials = h1.get_trials(t, min_cell_n=min_cell_n, verbose=verbose)
    trials = trials[trials["sensory_condition"] == condition]

    keys = ["subject"] + (["speed"] if by_speed else []) + ["position_rank"]

    profile = (
        trials
        .groupby(keys, as_index=False)["rt_ms"]
        .agg(estimator)
        .sort_values(keys)
        .reset_index(drop=True)
    )

    if centre:
        group_keys = keys[:-1]
        profile["rt_centred_ms"] = profile["rt_ms"] - profile.groupby(
            group_keys
        )["rt_ms"].transform("mean")
        value = "rt_centred_ms"
    else:
        value = "rt_ms"

    ranks = profile["position_rank"].dropna()
    x_min = float(ranks.min()) if len(ranks) else 1.0
    x_max = float(ranks.max()) if len(ranks) else 7.0

    rows = []

    for key, one in profile.groupby(keys[:-1]):
        one = one.sort_values("position_rank").dropna(subset=[value])

        if len(one) < 5:
            continue

        comparison = compare_models_per_participant(
            one["position_rank"].to_numpy(),
            one[value].to_numpy(),
            x_min=x_min,
            x_max=x_max,
            aic_margin=config.AIC_MARGIN,
            # VT RT is predicted to fall as the stimulus nears the body, so
            # the plateau near the body is the low one and the canonical k is
            # negative here, the mirror of the facilitation case.
            k_direction="negative",
        )

        comparison["preferred"] = _preferred_from_aic(comparison)

        # Keep AIC/AICc from the preregistered comparison, but use the same
        # bounded multi-start geometry fit for the descriptive RT boundary.
        curve = _fit_boundary_sigmoid(
            one["position_rank"].to_numpy(float),
            one[value].to_numpy(float),
            x_min=x_min,
            x_max=x_max,
            k_direction="negative",
            k_max=getattr(config, "K_MAX", 10.0),
        )
        comparison["low"] = curve.get("low", np.nan)
        comparison["high"] = curve.get("high", np.nan)
        comparison["x_c"] = curve.get("x_c", np.nan)
        comparison["se_x_c"] = curve.get("se_x_c", np.nan)
        comparison["k"] = curve.get("k", np.nan)
        comparison["se_k"] = curve.get("se_k", np.nan)
        comparison["amplitude_ms"] = curve.get("amplitude_ms", np.nan)
        comparison["k_at_bound"] = curve.get("k_at_bound", False)
        comparison["x_c_in_range"] = bool(
            curve.get("ok", False)
            and np.isfinite(comparison["x_c"])
            and x_min <= comparison["x_c"] <= x_max
        )
        width = _transition_width_10_90(comparison["k"])
        comparison["k_ok"] = bool(
            curve.get("ok", False)
            and np.isfinite(comparison["k"])
            and comparison["k"] < 0
            and not comparison["k_at_bound"]
            and np.isfinite(width)
            and width >= 1.0
            and width <= (x_max - x_min)
        )
        comparison["x_c_precise"] = bool(
            comparison["x_c_in_range"]
            and comparison["k_ok"]
            and np.isfinite(comparison["se_x_c"])
            and comparison["se_x_c"] > 0
            and comparison["se_x_c"] <= 1.5
        )

        record = dict(zip(keys[:-1], key if isinstance(key, tuple) else (key,)))
        record.update(comparison)
        rows.append(record)

    table = pd.DataFrame(rows)

    if table.empty:
        return {
            "condition": condition,
            "table": table,
            "profile": profile,
            "n_total": 0,
            "n_supported": 0,
            "n_preferred": 0,
        }

    fitted = table["sigmoid_ok"].fillna(False).astype(bool)
    aic_winner = table["preferred"] == "sigmoid"
    checks = (
        table["k_ok"].fillna(False).astype(bool)
        & table["x_c_in_range"].fillna(False).astype(bool)
        & table["x_c_precise"].fillna(False).astype(bool)
    )

    table["sigmoid_lower_aic"] = np.where(
        fitted,
        aic_winner,
        np.nan,
    )

    # ``run`` decides estimability with a trial-level bootstrap and a
    # half-maximum fallback, both of which operate on facilitation
    # profiles. Neither is run here, so no ``boundary_estimable`` column
    # exists and none is borrowed. What can be checked on a reaction-time
    # fit is its geometry: the slope runs in the expected direction, the
    # midpoint falls inside the tested range, and its standard error is
    # small enough to place it.
    table["sigmoid_boundary_geometry_ok"] = np.where(
        fitted,
        checks,
        np.nan,
    )

    table["x_c_m"] = table["x_c"].map(rank_to_metres)

    results = {
        "condition": condition,
        "estimator": estimator,
        "by_speed": by_speed,
        "centred": centre,
        "table": table,
        "profile": profile,
        "value_column": value,
        "x_min": x_min,
        "x_max": x_max,
        "n_total": int(len(table)),
        # Kept identical to ``run``: the sigmoid has the lowest AIC. The
        # two are then comparable across quantities.
        "n_supported": int((table["sigmoid_lower_aic"] == 1).sum()),
        "criterion_supported": "sigmoid has the lowest AIC",
        "n_preferred": int(aic_winner.sum()),
        # The counterpart of ``run``'s ``n_estimable``, but a weaker bar:
        # ``run`` adds a trial-level bootstrap of the midpoint, which
        # resamples facilitation profiles and cannot be run on a reaction
        # time. Only the geometry is checked here.
        "n_geometry_ok": int(
            (table["sigmoid_boundary_geometry_ok"] == 1).sum()
        ),
        "criterion_estimable": (
            "slope direction, midpoint in range, midpoint precise "
            "(no bootstrap)"
        ),
        "n_identifiable": int(table["k_ok"].fillna(False).astype(bool).sum()),
    }

    if verbose:
        report_rt(results)

    if plot:
        results["figure"] = plot_rt_fits(results)
        plt.show()

    return results


def report_rt(results):
    """Print the reaction-time model comparison."""
    table = results["table"]

    print(
        f"Sigmoid fitted to {results['condition']} reaction times"
        + (" (centred within participant)" if results.get("centred") else "")
    )
    print()

    columns = [
        column
        for column in [
            "subject", "speed", "aic_flat", "aic_linear", "aic_sigmoid",
            "aic_delta_vs_linear", "r2_sigmoid", "x_c", "x_c_m", "se_x_c",
            "k", "amplitude_ms", "k_ok", "x_c_precise", "preferred",
            "sigmoid_lower_aic",
        ]
        if column in table.columns
    ]
    print(table[columns].round(2).to_string(index=False))
    print()

    print(
        f"  sigmoid wins on AIC and describes an estimable boundary in "
        f"{results['n_supported']} of {results['n_total']} profiles"
    )
    print(
        f"  sigmoid has the lowest AIC in {results['n_preferred']} of "
        f"{results['n_total']}; transition usable in "
        f"{results['n_identifiable']}"
    )

    usable = table[table["sigmoid_lower_aic"] == 1]

    if len(usable) and "speed" in usable.columns:
        print()
        for speed, one in usable.groupby("speed"):
            print(
                f"  {speed}: boundary {one['x_c_m'].mean():.2f} m "
                f"(SD {one['x_c_m'].std():.2f}, n = {len(one)})"
            )


def plot_rt_fits(results, n_columns=4, save_path=None):
    """One panel per participant: the RT profile and its fitted sigmoid."""
    profile = results["profile"]
    table = results["table"]
    value = results["value_column"]

    subjects = sorted(profile["subject"].dropna().unique())
    n_rows = int(np.ceil(len(subjects) / n_columns))

    fig, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(3.7 * n_columns, 3.2 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )
    flat_axes = axes.ravel()

    speeds = (
        sorted(profile["speed"].dropna().unique())
        if results.get("by_speed") and "speed" in profile.columns
        else [None]
    )
    tested = sorted(profile["position_rank"].dropna().unique())

    for ax, name in zip(flat_axes, subjects):
        for speed in speeds:
            line = profile[profile["subject"] == name]
            row = table[table["subject"] == name]

            if speed is not None:
                line = line[line["speed"] == speed]
                row = row[row["speed"] == speed]

            line = line.sort_values("position_rank")

            if line.empty:
                continue

            colour = style.SPEED[speed] if speed else style.INK

            ax.plot(
                line["position_rank"], line[value],
                color=colour, marker="o", markersize=4.5, linewidth=2.0,
                label=(speed or results["condition"]).capitalize(), zorder=4,
            )

            if len(row) and bool(row["sigmoid_lower_aic"].iloc[0] == 1):
                one = row.iloc[0]
                smooth_x = np.linspace(results["x_min"], results["x_max"], 300)
                ax.plot(
                    smooth_x,
                    sigmoid(smooth_x, one["low"], one["high"],
                            one["x_c"], one["k"]),
                    color=colour, linestyle="--", linewidth=1.6, zorder=3,
                )
                ax.axvline(
                    one["x_c"], color=colour, linestyle=":",
                    linewidth=1.0, alpha=0.7,
                )

        if results.get("centred"):
            ax.axhline(0, linestyle="--", linewidth=0.8, color="gray", zorder=0)

        ax.set_title(str(name), loc="left", fontsize=9.5)
        h1._format_distance_axis(ax, unit="m", ranks=tested or None)

    for ax in flat_axes[len(subjects):]:
        ax.set_visible(False)

    label = (
        f"{results['condition']} RT, centred (ms)"
        if results.get("centred")
        else f"{results['condition']} RT (ms)"
    )
    for row_index in range(n_rows):
        axes[row_index, 0].set_ylabel(label)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            dict(zip(labels, handles)).values(),
            dict(zip(labels, handles)).keys(),
            frameon=False, ncol=len(set(labels)), loc="upper right",
            bbox_to_anchor=(1.0, 1.02), fontsize=8,
        )

    fig.suptitle(
        f"Sigmoid fitted to {results['condition']} reaction times, "
        "per participant",
        x=0.02, ha="left", y=1.0, color=style.INK,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def compare_fit_targets(t, min_cell_n=None, verbose=True):
    """Fit the same models to facilitation, to VT RT, and to T RT.

    The three answer different questions and are affected differently by the
    four-trial tactile cell:

        facilitation  T - VT at each distance. Both terms enter, so the
                      tactile sampling noise propagates in full.
        VT RT         the multisensory response alone, 15 trials per cell.
                      This is what Noel et al. fit.
        T RT          the tactile response alone, 4 trials per cell. Included
                      as the control: any structure recovered here at that
                      cell size is a warning that the procedure finds shape
                      in noise.

    A boundary in VT but not in T supports reading the VT fit as a PPS
    effect rather than an artefact of the fitting procedure.

    ``n_lowest_aic`` and ``n_sigmoid_supported`` are comparable across the
    rows. ``n_boundary_ok`` is not: facilitation carries the bootstrap
    localisation of the midpoint, which resamples facilitation profiles,
    while the reaction-time rows carry the geometry checks alone. The
    ``boundary_criterion`` column records which applied to each row.
    """
    facilitation = run(t, plot=False, verbose=False)

    vt = run_rt(t, condition="VT", min_cell_n=min_cell_n,
                plot=False, verbose=False)
    tactile = run_rt(t, condition="T", min_cell_n=min_cell_n,
                     plot=False, verbose=False)

    summary = pd.DataFrame([
        {
            "target": "facilitation (T - VT)",
            "trials_per_cell": "4 (T) + 15 (VT)",
            "n_profiles": facilitation["n_total"],
            "n_lowest_aic": facilitation["n_preferred"],
            "n_sigmoid_supported": facilitation["n_supported"],
            "n_boundary_ok": facilitation.get("n_estimable", np.nan),
            "boundary_criterion": "geometry + bootstrap",
            "median_r2": facilitation["table"]["r2_sigmoid"].median(),
        },
        {
            "target": "VT reaction time",
            "trials_per_cell": "15",
            "n_profiles": vt["n_total"],
            "n_lowest_aic": vt["n_preferred"],
            "n_sigmoid_supported": vt["n_supported"],
            "n_boundary_ok": vt["n_geometry_ok"],
            "boundary_criterion": "geometry only",
            "median_r2": vt["table"]["r2_sigmoid"].median(),
        },
        {
            "target": "T reaction time (control)",
            "trials_per_cell": "4",
            "n_profiles": tactile["n_total"],
            "n_lowest_aic": tactile["n_preferred"],
            "n_sigmoid_supported": tactile["n_supported"],
            "n_boundary_ok": tactile["n_geometry_ok"],
            "boundary_criterion": "geometry only",
            "median_r2": tactile["table"]["r2_sigmoid"].median(),
        },
    ]).round(2)

    if verbose:
        print("Same models, three different quantities to fit")
        print()
        print(summary.to_string(index=False))
        print()
        print(
            "  Facilitation is counted per participant and the RT rows per "
            "participant and speed, so n_profiles differs."
        )
        print(
            "  n_lowest_aic and n_sigmoid_supported are comparable across "
            "the rows: the same models, compared by the same criterion."
        )
        print(
            "  n_boundary_ok is not. The facilitation row adds a "
            "trial-level bootstrap of the midpoint, which resamples "
            "facilitation profiles and cannot be run on a reaction time, "
            "so the RT rows use the geometry checks alone. That is a "
            "weaker bar, and the counts are not the same quantity."
        )

    return {
        "summary": summary,
        "facilitation": facilitation,
        "vt": vt,
        "tactile": tactile,
    }

def fit_boundary_trial_level(
    trials,
    condition="VT",
    k_direction="negative",
    by=("subject", "speed"),
    x_column="position_rank",
    n_anchor=2,
    verbose=True,
):
    """Fit the boundary to individual trials rather than to cell summaries.

    The usual fit collapses each distance to one median and fits seven points.
    That discards most of the data: a cell holding fifteen trials contributes
    the same single number as a cell holding four, and the fit has no way to
    know which of its seven points it should trust. Here every trial is a data
    point, so a distance sampled more often pulls the curve harder, exactly as
    it should.

    The curve is unchanged: plateaus fixed from the data, midpoint and slope
    estimated. What changes is that the residuals are now trial-level, so the
    standard error on the midpoint reflects the trial-to-trial variability the
    cell medians were hiding.

    ``x_column`` may be the distance rank or any per-trial quantity on the same
    axis, including the measured foreperiod. Passing the foreperiod uses the
    exact onset time of each trial rather than the nominal distance it was
    assigned to.
    """
    from scipy.optimize import curve_fit

    data = trials[trials["sensory_condition"] == condition].copy()
    data = data.dropna(subset=[x_column, "rt_ms"])

    if data.empty:
        print(f"No usable {condition} trials.")
        return pd.DataFrame()

    by = list(by)
    rows = []

    for key, one in data.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        label = dict(zip(by, key))

        x = one[x_column].to_numpy(dtype=float)
        y = one["rt_ms"].to_numpy(dtype=float)

        if len(np.unique(x)) < 5:
            continue

        # The plateaus come from the cell means at each end, so they are
        # estimated on all the trials at those distances rather than on two
        # summary points.
        positions = np.sort(np.unique(x))
        near_positions = positions[:n_anchor]
        far_positions = positions[-n_anchor:]
        near_plateau = float(np.mean(y[np.isin(x, near_positions)]))
        far_plateau = float(np.mean(y[np.isin(x, far_positions)]))
        amplitude = far_plateau - near_plateau

        if not np.isfinite(amplitude) or amplitude == 0:
            continue

        xc_lo = float(positions.min()) + 1.0
        xc_hi = float(positions.max()) - 1.0

        def _curve(x_value, x_c, k):
            return near_plateau + amplitude / (1.0 + np.exp(k * (x_value - x_c)))

        best, best_rss = None, np.inf

        for x_c0 in np.linspace(xc_lo, xc_hi, 5):
            for k0 in (0.5, 2.0, 5.0):
                for sign in (1.0, -1.0):
                    try:
                        popt, pcov = curve_fit(
                            _curve, x, y, p0=[float(x_c0), sign * float(k0)],
                            bounds=([xc_lo, -100.0], [xc_hi, 100.0]),
                            maxfev=10000,
                        )
                    except Exception:
                        continue

                    rss = float(np.sum((y - _curve(x, *popt)) ** 2))

                    if np.isfinite(rss) and rss < best_rss:
                        best, best_rss = (popt, pcov), rss

        if best is None:
            continue

        popt, pcov = best
        errors = (np.sqrt(np.diag(pcov)) if pcov is not None
                  and np.all(np.isfinite(pcov)) else [np.nan, np.nan])

        # Same canonicalisation as ``_fit_boundary_sigmoid``: positive means
        # the profile starts high near the body and falls with distance.
        k = float(popt[1]) * float(np.sign(far_plateau - near_plateau))

        direction_ok = (
            (k_direction == "positive" and k > 0)
            or (k_direction == "negative" and k < 0)
            or k_direction == "any"
        )

        rows.append({
            **label,
            "n_trials": len(x),
            "n_positions": len(positions),
            "x_c": float(popt[0]),
            "se_x_c": float(errors[0]),
            "k": k,
            "se_k": float(errors[1]),
            "amplitude_ms": abs(float(amplitude)),
            "direction_ok": direction_ok,
            "at_bound": bool(
                abs(popt[0] - xc_lo) < 1e-6 or abs(popt[0] - xc_hi) < 1e-6
            ),
        })

    table = pd.DataFrame(rows)

    if table.empty:
        return table

    table["usable"] = (
        table["direction_ok"]
        & ~table["at_bound"]
        & table["k"].abs().between(0.5, 4.0)
        & table["se_x_c"].between(0, 1.0)
    )

    if verbose:
        print(f"Trial-level boundary fits, {condition}, x = {x_column}")
        print(f"  {int(table['usable'].sum())} of {len(table)} usable")
        print(f"  median trials per fit: {table['n_trials'].median():.0f}")
        print()

    return table.round(4)


def fit_boundary_hierarchical(
    trials,
    condition="VT",
    x_column="position_rank",
    n_anchor=2,
    n_boot=0,
    verbose=True,
):
    """One group boundary with a per-participant offset, fitted together.

    EXPERIMENTAL. This is a least-squares approximation to a hierarchical fit,
    not a hierarchical model: the offsets are free parameters constrained to
    sum to zero rather than draws from an estimated distribution, so there is
    no shrinkage and no variance component. In testing the shared slope runs
    to the bound and its standard error is not usable, which means the group
    boundary should be read as a rough common estimate and the individual
    offsets not at all. A proper version needs mixed-model machinery, and
    ``fit_boundary_trial_level`` is the better tool until then.

    Fitting each participant alone gives the ones with weak profiles nothing
    to lean on, so their midpoints run to the edge of the range or their
    slopes become walls. Fitting everyone at once lets the group constrain
    them: a shared slope and a shared amplitude are estimated across all
    participants, and each participant contributes only an offset to the
    common midpoint.

    That trade is the point. A participant whose own profile could not support
    a four-parameter curve can still support a single offset, so an estimate
    exists for everyone. The cost is that the individual boundaries are pulled
    toward the group, and a participant who genuinely differs will be
    understated. The spread of the offsets says how much the participants
    actually differ, and if it is near zero the group fit is doing all the
    work.

    Reaction time level differs between people for reasons that have nothing
    to do with peripersonal space, so each participant also gets their own
    intercept. Only the boundary offset is the quantity of interest.
    """
    from scipy.optimize import curve_fit

    data = trials[trials["sensory_condition"] == condition].copy()
    data = data.dropna(subset=[x_column, "rt_ms"])

    if data.empty:
        print(f"No usable {condition} trials.")
        return {}

    people = sorted(data["subject"].dropna().unique())
    index = {person: i for i, person in enumerate(people)}

    x = data[x_column].to_numpy(dtype=float)
    y = data["rt_ms"].to_numpy(dtype=float)
    who = data["subject"].map(index).to_numpy(dtype=int)

    positions = np.sort(np.unique(x))
    xc_lo = float(positions.min()) + 1.0
    xc_hi = float(positions.max()) - 1.0

    # Parameters: [x_c_group, k, amplitude, offset_1..offset_(p-1),
    #              level_1..level_p]
    #
    # Only p-1 offsets are free. A group midpoint plus p free offsets is not
    # identifiable: adding a constant to the group value and subtracting it
    # from every offset leaves the model unchanged, so the two trade off
    # exactly and their standard errors are unbounded. Constraining the
    # offsets to sum to zero fixes the group value as the average boundary and
    # makes each offset a departure from it.
    n_people = len(people)
    n_free = n_people - 1

    def _model(_, *params):
        x_c = params[0]
        k = params[1]
        amplitude = params[2]
        free = np.asarray(params[3:3 + n_free])
        offsets = np.append(free, -free.sum())
        levels = np.asarray(params[3 + n_free:])
        centre = x_c + offsets[who]
        return levels[who] + amplitude / (1.0 + np.exp(k * (x - centre)))

    per_person_level = data.groupby("subject")["rt_ms"].mean()
    start = (
        [float(np.median(positions)), -2.0, float(np.ptp(y) / 2)]
        + [0.0] * n_free
        + [float(per_person_level[p]) for p in people]
    )
    lower = (
        [xc_lo, -100.0, -np.inf]
        + [-3.0] * n_free
        + [-np.inf] * n_people
    )
    upper = (
        [xc_hi, 100.0, np.inf]
        + [3.0] * n_free
        + [np.inf] * n_people
    )

    try:
        popt, pcov = curve_fit(
            _model, np.zeros_like(y), y, p0=start,
            bounds=(lower, upper), maxfev=200000,
        )
    except Exception as error:
        print(f"Hierarchical fit failed: {type(error).__name__}: {error}")
        return {}

    errors = (np.sqrt(np.diag(pcov)) if pcov is not None
              and np.all(np.isfinite(pcov)) else np.full(len(popt), np.nan))

    x_c_group = float(popt[0])
    free = np.asarray(popt[3:3 + n_free], dtype=float)
    offsets = np.append(free, -free.sum())
    offset_errors = np.append(errors[3:3 + n_free], np.nan)

    per_subject = pd.DataFrame({
        "subject": people,
        "offset": offsets,
        "se_offset": offset_errors,
        "x_c": x_c_group + offsets,
        "n_trials": [int((who == i).sum()) for i in range(n_people)],
    }).round(3)

    results = {
        "x_c_group": x_c_group,
        "se_x_c_group": float(errors[0]),
        "k": float(popt[1]),
        "se_k": float(errors[1]),
        "amplitude_ms": float(popt[2]),
        "per_subject": per_subject,
        "n_trials": len(y),
        "n_subjects": n_people,
    }

    if verbose:
        print(f"Hierarchical boundary fit, {condition}, x = {x_column}")
        print(f"  {len(y)} trials from {n_people} participants")
        print(f"  group boundary  = {x_c_group:.2f} "
              f"(SE {results['se_x_c_group']:.2f})")
        print(f"  shared slope    = {results['k']:.2f} "
              f"(SE {results['se_k']:.2f})")
        print(f"  amplitude       = {results['amplitude_ms']:.1f} ms")
        print(f"  offset spread   = {offsets.std(ddof=1):.2f} ranks "
              f"(range {offsets.min():+.2f} to {offsets.max():+.2f})")
        print()
        print(per_subject.to_string(index=False))
        print()

    return results