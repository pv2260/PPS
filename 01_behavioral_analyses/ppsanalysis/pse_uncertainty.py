"""Uncertainty on the collision boundary, and on its speed-driven shift.

The PSE is a ratio, $-(\\beta_0)/\\beta_{cl}$, so its sampling distribution is
not symmetric and its standard error is not well behaved when the slope is
shallow: dividing by a small number with its own error attached sends the
boundary a long way. A participant with a steep curve has a boundary located
to within a centimetre or two; a participant with a shallow one may have a
boundary that the data barely constrain at all, even though the point estimate
looks perfectly definite.

This resamples trials within each speed, refits, and reports percentile
intervals, so the asymmetry and the width are both visible. Draws are paired
by index across the two speeds, so the interval on the difference is computed
from the same resample of the same participant rather than from two
independent ones.

Usage
-----
    from ppsanalysis import pse_uncertainty as pu

    result = pu.estimate(clin.collision_trials, "PT001",
                         ball_radius_cm=BALL_RADIUS_CM)
    pu.report(result)
    pu.plot(result, save_as="pse_uncertainty_patient")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style
from . import figures
from . import config as cfg


SPEEDS = ["slow", "fast"]

# Above this the fit has not located a boundary inside anything the task
# sampled, and the draw is treated as failed rather than allowed to drag the
# interval. Reported as a count, never silently dropped.
MAX_PLAUSIBLE_PSE_CM = 100.0


def _fit_one(clearance, hit):
    """Logistic fit of hit responses on clearance. Returns (intercept, slope).

    Maximum likelihood through statsmodels, falling back to NaN rather than
    raising: a bootstrap draw that happens to contain one response category
    cannot be fitted, and that is information about the participant, not an
    error to stop on.
    """
    import statsmodels.api as sm

    if len(np.unique(hit)) < 2 or len(np.unique(clearance)) < 3:
        return np.nan, np.nan

    design = sm.add_constant(clearance, has_constant="add")

    try:
        fit = sm.Logit(hit, design).fit(disp=False)
    except Exception:
        return np.nan, np.nan

    return float(fit.params[0]), float(fit.params[1])


def _pse(intercept, slope):
    """Clearance at P(hit) = 0.5, guarded against a slope near zero."""
    if not np.isfinite(intercept) or not np.isfinite(slope) or slope == 0:
        return np.nan

    value = -intercept / slope

    return value if abs(value) <= MAX_PLAUSIBLE_PSE_CM else np.nan


def estimate(
    collision,
    subject,
    ball_radius_cm=0.0,
    session=None,
    n_boot=None,
    seed=None,
    near_only=False,
):
    """Boundary, slope and speed shift for one participant, with intervals.

    Parameters
    ----------
    collision : DataFrame
        Collision trials, e.g. ``clin.collision_trials``.
    subject : str
        Which participant.
    ball_radius_cm : float
        Subtracted so the axis is clearance, the gap between the ball's
        surface and the shoulder edge.
    near_only : bool
        Restrict to near-boundary trials. Off by default: the whole clearance
        range is what constrains the slope, and dropping the clear trials
        makes a shallow fit shallower still.

    Returns
    -------
    dict with keys ``observed`` (one row per speed plus the difference),
    ``draws`` (the bootstrap values), and ``settings``.
    """
    n_boot = cfg.N_BOOT if n_boot is None else n_boot
    seed = cfg.RANDOM_SEED if seed is None else seed

    data = collision[collision["subject"] == subject].copy()

    if "usable" in data.columns:
        data = data[data["usable"]]

    if session is not None and "session" in data.columns:
        data = data[data["session"] == session]

    if near_only and "trial_type" in data.columns:
        data = data[data["trial_type"].isin(["near_hit", "near_miss"])]

    data["clearance_cm"] = (
        pd.to_numeric(data["offset_from_edge_cm"], errors="coerce")
        - ball_radius_cm
    )

    if "response_hit" in data.columns:
        data["hit"] = pd.to_numeric(data["response_hit"], errors="coerce")
    else:
        data["hit"] = (
            data["participant_response"].astype(str).str.upper().eq("HIT")
            .astype(float)
        )

    data = data.dropna(subset=["clearance_cm", "hit", "speed"])

    if data.empty:
        raise ValueError(f"no usable collision trials for {subject!r}")

    # Observed fit, per speed.
    observed = {}

    for speed in SPEEDS:
        cell = data[data["speed"] == speed]
        intercept, slope = _fit_one(
            cell["clearance_cm"].to_numpy(float),
            cell["hit"].to_numpy(float),
        )
        observed[speed] = {
            "intercept": intercept,
            "slope": slope,
            "pse_cm": _pse(intercept, slope),
            "n_trials": int(len(cell)),
        }

    rng = np.random.default_rng(seed)

    # Trials are resampled within speed, so each draw keeps the design's trial
    # counts. The two speeds share a draw index, which is what makes the
    # interval on their difference meaningful.
    index_by_speed = {
        speed: data.index[data["speed"] == speed].to_numpy()
        for speed in SPEEDS
    }

    rows = []

    for draw in range(int(n_boot)):
        record = {"draw": draw}

        for speed in SPEEDS:
            available = index_by_speed[speed]

            if len(available) < 10:
                record[f"pse_{speed}"] = np.nan
                record[f"slope_{speed}"] = np.nan
                continue

            taken = rng.choice(available, size=len(available), replace=True)
            cell = data.loc[taken]

            intercept, slope = _fit_one(
                cell["clearance_cm"].to_numpy(float),
                cell["hit"].to_numpy(float),
            )

            record[f"pse_{speed}"] = _pse(intercept, slope)
            record[f"slope_{speed}"] = slope

        record["delta_coll"] = record.get("pse_fast", np.nan) - record.get(
            "pse_slow", np.nan
        )
        rows.append(record)

    draws = pd.DataFrame(rows)

    summary = []

    for name, column, point in [
        ("PSE slow", "pse_slow", observed["slow"]["pse_cm"]),
        ("PSE fast", "pse_fast", observed["fast"]["pse_cm"]),
        ("Delta_coll", "delta_coll",
         observed["fast"]["pse_cm"] - observed["slow"]["pse_cm"]),
        ("slope slow", "slope_slow", observed["slow"]["slope"]),
        ("slope fast", "slope_fast", observed["fast"]["slope"]),
    ]:
        values = draws[column].dropna()

        summary.append({
            "quantity": name,
            "observed": point,
            "boot_median": float(values.median()) if len(values) else np.nan,
            "ci_low": float(values.quantile(0.025)) if len(values) else np.nan,
            "ci_high": float(values.quantile(0.975)) if len(values) else np.nan,
            "width": (
                float(values.quantile(0.975) - values.quantile(0.025))
                if len(values) else np.nan
            ),
            "p_below_zero": (
                float((values < 0).mean()) if len(values) else np.nan
            ),
            "usable_draws": int(len(values)),
            "failed_draws": int(len(draws) - len(values)),
        })

    return {
        "subject": subject,
        "observed": pd.DataFrame(summary).round(3),
        "draws": draws,
        "settings": {
            "n_boot": int(n_boot),
            "seed": seed,
            "ball_radius_cm": ball_radius_cm,
            "near_only": near_only,
            "n_trials": {s: observed[s]["n_trials"] for s in SPEEDS},
        },
    }


def report(result):
    """Print the intervals and what they say about the estimate."""
    settings = result["settings"]

    print(f"{result['subject']}: boundary uncertainty")
    print(
        f"  {settings['n_boot']} draws, "
        f"{settings['n_trials']['slow']} slow and "
        f"{settings['n_trials']['fast']} fast trials"
    )
    print()
    print(result["observed"].to_string(index=False))

    failed = result["observed"]["failed_draws"].max()

    if failed:
        print(
            f"\n  {failed} draws produced no usable boundary. A high count "
            "means the estimate rests on the slope staying steep enough to "
            "divide by."
        )

    delta = result["observed"].loc[
        result["observed"]["quantity"] == "Delta_coll"
    ].iloc[0]

    print()
    print(
        f"  Delta_coll = {delta['observed']:.2f} cm, 95% interval "
        f"[{delta['ci_low']:.2f}, {delta['ci_high']:.2f}], "
        f"{100 * delta['p_below_zero']:.1f}% of draws negative."
    )

    if delta["ci_low"] < 0 < delta["ci_high"]:
        print(
            "  The interval spans zero, so the direction of the shift is not "
            "established for this participant."
        )


def plot(result, save_as=None):
    """Three panels: the two boundaries and their difference.

    Each panel is the bootstrap distribution with the observed value marked.
    A wide, skewed distribution is the signature of a shallow slope: the
    boundary is a ratio, and dividing by a small number with its own error
    attached spreads the estimate asymmetrically.
    """
    draws = result["draws"]
    observed = result["observed"].set_index("quantity")

    panels = [
        ("pse_slow", "PSE slow", "Boundary, slow (cm)", style.SLOW),
        ("pse_fast", "PSE fast", "Boundary, fast (cm)", style.FAST),
        ("delta_coll", "Delta_coll", "Fast minus slow (cm)", style.INK),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), squeeze=False)
    axes = axes[0]

    for ax, (column, name, label, colour) in zip(axes, panels):
        values = draws[column].dropna()

        if values.empty:
            ax.text(
                0.5, 0.5, "no usable draws", transform=ax.transAxes,
                ha="center", va="center", color=style.MUTED, fontsize=9,
            )
            style.clean(ax)
            continue

        ax.hist(
            values, bins=50, color=colour, alpha=0.45,
            edgecolor="white", linewidth=0.3,
        )

        row = observed.loc[name]

        ax.axvline(
            row["observed"], color=style.INK, linewidth=2.0,
            label=f"observed {row['observed']:.2f}",
        )
        ax.axvline(row["ci_low"], color=style.MUTED, linestyle="--",
                   linewidth=1.2)
        ax.axvline(row["ci_high"], color=style.MUTED, linestyle="--",
                   linewidth=1.2)
        ax.axvline(0, color=style.MUTED, linestyle=":", linewidth=1.0)

        ax.set_xlabel(label)
        ax.set_ylabel("Draws" if column == "pse_slow" else "")
        ax.set_title(
            f"{name}: [{row['ci_low']:.1f}, {row['ci_high']:.1f}]",
            loc="left", fontsize=9.5,
        )
        ax.legend(frameon=False, fontsize=8)
        style.clean(ax)

    fig.suptitle(
        f"{result['subject']}: bootstrap uncertainty on the collision "
        "boundary (dashed = 95% interval, dotted = contact)",
        x=0.02, ha="left", color=style.INK,
    )
    style.finish_figure(fig)

    if save_as:
        figures.save(fig, save_as)

    return fig
