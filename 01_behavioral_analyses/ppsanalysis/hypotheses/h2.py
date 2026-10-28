"""
H2. The PPS boundary moves outward when the stimulus approaches faster.

Facilitation should rise as the stimulus gets close to the body, and the point
where it rises is the PPS boundary. We call that point x_c (the inflection of a
sigmoid fitted to the facilitation profile).

If the stimulus is coming at you FAST, you should start reacting to it from
further away. So the boundary should sit further out on fast trials.

We measure that shift as:

    Delta_PPS = x_c(fast) - x_c(slow)

Delta_PPS > 0 means the boundary moved outward for fast approach.

HOW WE DECIDE IF IT IS SUPPORTED
--------------------------------
One Delta_PPS per participant, then bootstrap the group mean.
Supported if the mean is positive AND the bootstrap 95% CI lies above zero.
The Wilcoxon test is reported but does not decide support.

** note: 
A sigmoid will "fit" data that has no boundary in it. It just
puts the inflection outside the range we tested, or makes the slope vert
steep. Neither is a real estimate. So before we draw a fitted curve, we check:

    - x_c must fall STRICTLY INSIDE the distances we actually tested
    - |k| (the slope) must not be larger than K_MAX

If either check fails, we do NOT draw the curve and we do NOT report an x_c.
We just connect the points. It is better to show an honest scatter than a
confident-looking curve that means nothing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style

from .. import config
from .. import figures
from ..stats_utils import (
    sigmoid,
    bootstrap_mean_ci,
    wilcoxon_one_sided,
)
from .h1a import _fit_boundary_sigmoid


# ---------------------------------------------------------------------
# Participant display names
# ---------------------------------------------------------------------
# Figures name the role rather than the identifier: a reader needs to know
# which participant is the patient and which is the control. Which specific
# control it was belongs in the caption, where it is stated once.
#
# The map lives in ``style`` so every module labels the same person the same
# way. The fallback below keeps this module working against a style module
# that predates it.

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



def safe_sigmoid_fit(x, y, k_direction="positive"):
    """Fit a boundary sigmoid, using the same routine as H1a.

    H2 differences the boundaries H1a estimates, so the two must come from the
    same fit. They did not previously: H1a used its own boundary fitter while
    H2 called a four-parameter fit in ``stats_utils``, so a Delta_PPS could be
    computed from midpoints that H1a had never reported. Routing both through
    ``_fit_boundary_sigmoid`` removes that gap.

    The plateaus are fixed from the data and only the midpoint and slope are
    estimated, which keeps the midpoint off the edge of the sampled range.
    """
    return _fit_boundary_sigmoid(x, y, k_direction=k_direction)


def sigmoid_fit_is_trustworthy(fit, x_min, x_max, k_max=None):
    """Return True only if this sigmoid fit is a real estimate.

    A fit that "converged" is not the same as a fit that means something.
    """

    if k_max is None:
        k_max = config.K_MAX

    # Did curve_fit actually succeed?
    if not fit.get("ok", False):
        return False

    x_c = fit.get("x_c", np.nan)
    k = fit.get("k", np.inf)

    # Is the boundary a real number?
    if not np.isfinite(x_c):
        return False
    if not (x_min < x_c < x_max):
        return False

    if abs(k) > k_max:
        return False

    return True
def run(t, plot=True):
    """Run H2."""

    results = {}

    # ------------------------------------------------------------------
    # Step 1. One Delta_PPS per participant.
    # ------------------------------------------------------------------
    # t.delta_pps_young has one row per (subject, session), with x_c fitted
    # separately for slow and fast. Average over sessions.

    delta_per_subject = t.delta_pps_young.groupby(
        "subject",
        as_index=False,
    )["delta_pps"].mean()

    results["delta_per_subject"] = delta_per_subject

    # ------------------------------------------------------------------
    # Step 2. Group test.
    # ------------------------------------------------------------------

    values = delta_per_subject["delta_pps"].dropna().to_numpy()

    boot = bootstrap_mean_ci(
        values,
        n_boot=config.N_BOOT,
        ci=95,
        seed=config.RANDOM_SEED,
    )

    wilcoxon = wilcoxon_one_sided(values, alternative="greater")

    supported = bool(boot["mean"] > 0) and bool(boot["ci_low"] > 0)

    results["values"] = values
    results["boot"] = boot
    results["wilcoxon"] = wilcoxon
    results["supported"] = supported

    # ------------------------------------------------------------------
    # Step 3. Pooled facilitation profile per speed (for the second figure).
    # ------------------------------------------------------------------

    h2_subjects = delta_per_subject["subject"].dropna().unique()

    
    facilitation = t.facilitation_young.loc[
    t.facilitation_young["subject"].isin(h2_subjects)
        ].dropna(
            subset=["position_rank", "facilitation_ms"]
        )

    pooled = facilitation.groupby(
        ["speed", "position_rank"],
        as_index=False,
    )["facilitation_ms"].mean()

    results["pooled_profile"] = pooled
    results["facilitation_raw"] = facilitation

    # Fit a sigmoid to each speed, and check whether we can trust it.
    fits = {}
    trustworthy = {}

    if len(pooled) > 0:
        ranks = np.sort(pooled["position_rank"].unique())
        x_min = float(ranks.min())
        x_max = float(ranks.max())

        for speed in [config.SLOW_LABEL, config.FAST_LABEL]:
            one_speed = pooled[pooled["speed"] == speed].sort_values("position_rank")

            # Facilitation is larger near the body, so the transition runs
            # positive; the same direction H1a uses for this quantity.
            fit = safe_sigmoid_fit(
                one_speed["position_rank"].to_numpy(),
                one_speed["facilitation_ms"].to_numpy(),
                k_direction="positive",
            )

            fits[speed] = fit
            trustworthy[speed] = sigmoid_fit_is_trustworthy(fit, x_min, x_max)

        results["x_min"] = x_min
        results["x_max"] = x_max

    results["fits"] = fits
    results["trustworthy"] = trustworthy

    # ------------------------------------------------------------------
    # Step 4. Report and plot.
    # ------------------------------------------------------------------

    report(results)

    if plot:
        fig = make_figure(results)
        if fig is not None:
            figures.save(fig, "h2_delta_pps")
            plt.show()
            results["figure"] = fig

    return results


def report(results):
    """Print the H2 numbers."""

    boot = results["boot"]
    wilcoxon = results["wilcoxon"]

    print("H2, step 1: Delta_PPS per participant")
    print("  Delta_PPS = x_c(fast) - x_c(slow). Positive = boundary moved outward.")
    print()
    print(results["delta_per_subject"].round(3).to_string(index=False))
    print()

    print("H2, step 2: group test")
    print(f"  number of participants  = {boot['n']}")
    print(f"  mean Delta_PPS          = {boot['mean']:.3f}")
    print(f"  bootstrap 95% CI        = [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]")
    print(f"  Wilcoxon (one-sided)    = W {wilcoxon['W']}, p {wilcoxon['p']:.4f}")
    print()

    if results["supported"]:
        print("  H2 IS SUPPORTED: mean > 0 and the 95% CI lies above zero.")
    else:
        print("  H2 is NOT supported.")
    print()

    # Warn loudly if the sigmoid fits were rejected. This is important: if the
    # fits are untrustworthy then Delta_PPS itself is untrustworthy, and the
    # group test above is testing noise.
    trustworthy = results.get("trustworthy", {})
    rejected = [speed for speed, ok in trustworthy.items() if not ok]

    if rejected:
        print("  WARNING: the pooled sigmoid fit was rejected for:", ", ".join(rejected))
        print("  A rejected fit means the boundary fell outside the tested range, or the")
        print("  slope was implausible. If this happens, Delta_PPS is not measuring a")
        print("  boundary shift, and H2 cannot be interpreted. Check the raw profile.")


def make_figure(results):
    """Build the H2 figure. Two panels."""

    if "pooled_profile" not in results or len(results["pooled_profile"]) == 0:
        print("H2 figure skipped: no facilitation data.")
        return None

    fig, axes = figures.new_figure(n_panels=2, height=3.0)
# --- Panel A: Delta_PPS per participant ----------------------------------
    ax = axes[0]

    delta_per_subject = results["delta_per_subject"]
    values = results["values"]
    boot = results["boot"]

    x_positions = np.arange(len(values))

    ax.scatter(
        x_positions,
        values,
        s=40,
        color=style.CONTROL,
        edgecolor=style.INK,
        linewidth=0.6,
        zorder=3,
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=0.8,
        color="gray",
    )

    if np.isfinite(boot["mean"]):
        lower_error = boot["mean"] - boot["ci_low"]
        upper_error = boot["ci_high"] - boot["mean"]

        ax.errorbar(
            len(values) + 0.5,
            boot["mean"],
            yerr=[[lower_error], [upper_error]],
            fmt="o",
            color=style.INK,
            markersize=6,
            capsize=4,
            zorder=3,
            label="mean (95% CI)",
        )

    ax.set_xticks(
        list(x_positions) + [len(values) + 0.5]
    )

    ax.set_xticklabels(
        _display_names(delta_per_subject["subject"]) + ["Mean"],
        rotation=30,
        ha="right",
    )

    ax.set_ylabel(r"$\Delta_{PPS}$")
    ax.set_title("Speed-induced boundary shift")
    ax.legend(frameon=False)

    style.clean(ax)

# --- Panel B: descriptive group facilitation profiles -----------------
    ax = axes[1]

    pooled = results["pooled_profile"]
    raw = results["facilitation_raw"]
    fits = results["fits"]
    trustworthy = results["trustworthy"]

    x_min = results["x_min"]
    x_max = results["x_max"]

    smooth_x = np.linspace(x_min, x_max, 200)

    speed_colours = {
        config.SLOW_LABEL: style.SLOW,
        config.FAST_LABEL: style.FAST,
    }

    for speed, colour in speed_colours.items():

        # Individual participant-level facilitation values
        raw_this_speed = raw.loc[
            raw["speed"] == speed
        ]

        ax.scatter(
            raw_this_speed["position_rank"]
            + style.jitter(
                len(raw_this_speed),
                0.06,
                seed=1,
            ),
            raw_this_speed["facilitation_ms"],
            s=7,
            color=style.shade(colour, 0.35),
            alpha=0.35,
            edgecolor="none",
            zorder=2,
        )

        # Group mean at each distance
        pooled_this_speed = (
            pooled.loc[
                pooled["speed"] == speed
            ]
            .sort_values("position_rank")
        )

        ax.scatter(
            pooled_this_speed["position_rank"],
            pooled_this_speed["facilitation_ms"],
            s=28,
            color=style.shade(colour, -0.1),
            alpha=1.0,
            edgecolor="none",
            zorder=4,
        )

        fit = fits[speed]

        # Descriptive fit to the GROUP-MEAN profile only
        if trustworthy[speed]:

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
                color=colour,
                linewidth=1.8,
                label=f"{speed} (group-mean fit)",
                zorder=3,
            )

        else:

            ax.plot(
                pooled_this_speed["position_rank"],
                pooled_this_speed["facilitation_ms"],
                linestyle="--",
                color=colour,
                linewidth=1.2,
                label=f"{speed} (no fit)",
                zorder=3,
            )


    ax.axhline(
        0,
        linestyle="--",
        linewidth=0.8,
        color="gray",
    )

    if not all(trustworthy.values()):
        ax.text(
            0.5,
            0.02,
            "sigmoid fit rejected; showing points only",
            transform=ax.transAxes,
            ha="center",
            fontsize=6,
            color=style.MUTED,
        )

    ax.set_xlabel(
        "Distance rank (1 = closest to body)"
    )

    ax.set_ylabel(
        "Facilitation, T $-$ VT (ms)"
    )

    ax.set_title(
        "Group-mean facilitation profiles"
    )

    ax.legend(
        frameon=False
    )

    style.clean(ax)

    figures.label_panels(axes)
    fig.tight_layout()

    return fig


DEFAULT_CORRECTIONS = (
    "delay_matched",
    "expectancy_model",
    "fastest_tactile",
    "fastest_n_tactile",
)



def boundaries_by_speed(
    trials,
    corrections=DEFAULT_CORRECTIONS,
    estimator="median",
    k_direction="positive",
    verbose=True,
    expectancy_degree=1,
):
    from . import h1 as h1

    fit_rows = []
    profiles = []

    for correction in corrections:
        profile = h1.facilitation_profile(
            trials,
            estimator=estimator,
            correction=correction,
            expectancy_degree=expectancy_degree,
        ).copy()

        profile["correction"] = correction
        profiles.append(profile)

        for (subject, speed), one in profile.groupby(["subject", "speed"]):
            one = (
                one
                .dropna(subset=["position_rank", "facilitation_ms"])
                .sort_values("position_rank")
            )

            if len(one) < 5:
                continue

            x = one["position_rank"].to_numpy(dtype=float)
            y = one["facilitation_ms"].to_numpy(dtype=float)

            fit = _fit_boundary_sigmoid(
                x,
                y,
                k_direction=k_direction,
            )

            fit_rows.append({
                "correction": correction,
                "subject": subject,
                "speed": speed,
                "n_points": len(x),
                "x_c": fit["x_c"],
                "se_x_c": fit["se_x_c"],
                "k": fit["k"],
                "amplitude_ms": fit["amplitude_ms"],
                "profile_range_ms": float(np.ptp(y)),
                "fit_ok": fit["ok"],
            })

    speed_fits = pd.DataFrame(fit_rows)

    if speed_fits.empty:
        return speed_fits, pd.DataFrame()

    usable = speed_fits.loc[
        speed_fits["fit_ok"].fillna(False)
    ].copy()

    boundaries = (
        usable
        .pivot(
            index=["correction", "subject"],
            columns="speed",
            values="x_c",
        )
        .reset_index()
    )

    boundaries.columns.name = None

    for speed in (
        config.FAST_LABEL,
        config.SLOW_LABEL,
    ):
        if speed not in boundaries.columns:
            boundaries[speed] = np.nan

    boundaries["delta_pps"] = (
        boundaries[config.FAST_LABEL]
        - boundaries[config.SLOW_LABEL]
    )

    if verbose:
        print("Boundaries per participant and speed")
        print(
            "  fitted with the H1a boundary sigmoid: "
            "plateaus fixed from the data, midpoint and slope estimated"
        )
        print()

        for correction in corrections:
            rows = speed_fits.loc[
                speed_fits["correction"].eq(correction)
            ]

            both = boundaries.loc[
                boundaries["correction"].eq(correction)
                & boundaries["delta_pps"].notna()
            ]

            print(
                f"  {correction:18s} "
                f"{int(rows['fit_ok'].fillna(False).sum()):2d} "
                f"of {len(rows):2d} fits sound, "
                f"{len(both):2d} participant(s) with both speeds"
            )

        print()

    return speed_fits, boundaries


def run_by_speed(
    trials,
    corrections=DEFAULT_CORRECTIONS,
    estimator="median",
    plot=True,
    expectancy_degree=1,
):
    """H2 end to end from the trial table, one correction at a time.

    Fits the boundaries, differences them, and runs the group test for each
    correction separately, so the reader can see whether the answer depends on
    the tactile baseline.
    """
    speed_fits, boundaries = boundaries_by_speed(
        trials,
        corrections=corrections,
        estimator=estimator,
        expectancy_degree=expectancy_degree,
    )

    results = {
        "speed_fits": speed_fits,
        "boundaries": boundaries,
        "by_correction": {},
    }

    for correction in corrections:
        delta = boundaries[boundaries["correction"] == correction]
        values = delta["delta_pps"].dropna().to_numpy(dtype=float)

        if len(values) < 2:
            print(f"{correction}: {len(values)} boundary pair(s), no group test")
            results["by_correction"][correction] = {"n": len(values)}
            continue

        boot = bootstrap_mean_ci(values, n_boot=config.N_BOOT,
                                 seed=config.RANDOM_SEED)
        signed = wilcoxon_one_sided(values, alternative="greater")

        print(f"{correction}")
        print(f"  n                = {len(values)}")
        print(f"  mean Delta_PPS   = {values.mean():+.3f} ranks")
        print(f"  bootstrap 95% CI = "
              f"[{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}]")
        print(f"  Wilcoxon         = "
              f"W {signed['W']:.1f}, p {signed['p']:.4f}")
        print()

        results["by_correction"][correction] = {
            "n": len(values),
            "values": values,
            "mean": float(values.mean()),
            "boot": boot,
            "wilcoxon": signed,
        }

    return results

# ----------------------------------------------------------------------
# Per-participant speed fits
# ----------------------------------------------------------------------


def _draw_trial_points(ax, trials, subject, correction,
                       expectancy_degree=1):
    """The individual corrected trials behind one participant's profile.

    Drawn with jitter and low alpha, beneath everything else: they are
    context for the fit rather than part of it, and at four trials a cell
    an unjittered column of points is unreadable.
    """
    from . import h1 as h1

    # facilitation_profile summarises cells, so the per-trial values are
    # recovered from the visuo-tactile trials and the cell baseline the
    # profile already carries.
    profile = h1.facilitation_profile(
        trials, estimator="median", correction=correction,
        expectancy_degree=expectancy_degree,
    )
    profile = profile[profile["subject"] == subject]

    if "tactile_baseline_ms" not in profile.columns:
        return

    visuotactile = trials[
        (trials["subject"] == subject)
        & (trials["sensory_condition"] == "VT")
        & trials["rt_ms"].notna()
    ]

    if visuotactile.empty:
        return

    baselines = profile.set_index(
        ["speed", "position_rank"]
    )["tactile_baseline_ms"]

    rng = np.random.default_rng(0)

    for speed in [config.SLOW_LABEL, config.FAST_LABEL]:
        one = visuotactile[visuotactile["speed"] == speed]

        for rank, cell in one.groupby("position_rank"):
            baseline = baselines.get((speed, rank))

            if baseline is None or not np.isfinite(baseline):
                continue

            values = baseline - cell["rt_ms"].to_numpy(dtype=float)

            ax.scatter(
                rank + rng.normal(0, 0.06, size=len(values)),
                values,
                s=8, color=_SPEED_COLOUR[speed], alpha=0.18,
                edgecolors="none", zorder=0,
            )


# The same mapping make_figure builds locally, named once so the two
# figures cannot drift apart.
_SPEED_COLOUR = {
    config.SLOW_LABEL: style.SLOW,
    config.FAST_LABEL: style.FAST,
}


def _speed_curve(y, k, x_c, x):
    """The fitted sigmoid, rebuilt as ``_fit_boundary_sigmoid`` builds it.

    The fitter fixes the plateaus from the data and returns only the
    midpoint, slope and an unsigned amplitude, so the plateaus are
    recomputed here from the same two-point means rather than read back
    from the fit. Drawing the curve any other way puts a line beside the
    points instead of through them.
    """
    near, far = float(np.mean(y[:2])), float(np.mean(y[-2:]))
    return far - (far - near) / (1.0 + np.exp(k * (x - x_c)))


def _draw_speed_panel(ax, by_speed, fits, subject, title, trials=None,
                      correction=None, expectancy_degree=1):
    """One panel: both speeds, with their points, curves and midpoints.

    When ``trials`` is given, the per-trial facilitation values behind
    each cell are drawn faintly beneath the profile. A cell of four
    trials summarised by its median tells the reader nothing about how
    much those four disagreed, and that spread is what the fitted
    standard errors do not capture.
    """
    if trials is not None:
        _draw_trial_points(ax, trials, subject, correction,
                           expectancy_degree)

    for speed in [config.SLOW_LABEL, config.FAST_LABEL]:
        one = (
            by_speed[
                (by_speed["subject"] == subject)
                & (by_speed["speed"] == speed)
            ]
            .sort_values("position_rank")
        )

        if one.empty:
            continue

        y = one["facilitation_ms"].to_numpy(dtype=float)
        colour = _SPEED_COLOUR[speed]

        ax.plot(one["position_rank"], y, "o", markersize=4,
                color=colour, zorder=3)

        row = fits[
            (fits["subject"] == subject) & (fits["speed"] == speed)
        ]

        if not len(row):
            continue

        r = row.iloc[0]

        # A fit that failed its checks is drawn faint rather than
        # omitted: the panel should show that a curve was attempted and
        # rejected, not leave a speed with no line and no reason.
        sound = bool(r.get("fit_ok", True)) and not bool(
            r.get("k_at_bound", False)
        )

        x = np.linspace(1, 7, 200)

        ax.plot(
            x, _speed_curve(y, r["k"], r["x_c"], x),
            color=colour,
            linewidth=2.0 if sound else 1.2,
            linestyle="-" if sound else ":",
            alpha=1.0 if sound else 0.55,
            zorder=2,
            label=f"{speed.capitalize()} ($x_c$ = {r['x_c']:.2f})"
                  + ("" if sound else ", at bound"),
        )
        ax.axvline(r["x_c"], color=colour, linestyle="--",
                   linewidth=0.9, alpha=0.6, zorder=1)

    ax.axhline(0, linestyle=":", linewidth=0.8, color=style.MUTED, zorder=0)
    ax.set_title(title, loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    style.clean(ax)


def plot_speed_fits(
    trials,
    speed_fits,
    correction=None,
    estimator="median",
    expectancy_degree=1,
    exclude=None,
    subjects=None,
    show_trials=True,
    columns=3,
    panel_size=(3.4, 2.9),
    save_as=None,
):
    """One panel per participant plus the group, both speeds together.

    The per-participant panels use the fits ``boundaries_by_speed``
    produced. The group panel fits the averaged profile rather than
    averaging the individual midpoints, since those are different
    quantities when the profiles differ in shape, and it is the averaged
    profile the panel draws.
    """
    from . import h1 as h1

    # Filtering the trials as well as the fits, so the group panel
    # averages the same participants the individual panels show. Dropping
    # only the fits would leave an excluded participant contributing to
    # the group profile while having no panel of their own.
    if subjects is not None:
        trials = trials[trials["subject"].isin(list(subjects))]

    if exclude:
        dropped = sorted(set(exclude) & set(trials["subject"]))
        trials = trials[~trials["subject"].isin(list(exclude))]

        if dropped:
            print(f"Excluded: {', '.join(dropped)}")

        missing = sorted(set(exclude) - set(dropped))

        if missing:
            print(f"  not present: {', '.join(missing)}")

    if correction is None:
        correction = (
            speed_fits["correction"].iloc[0]
            if "correction" in speed_fits.columns and len(speed_fits)
            else DEFAULT_CORRECTIONS[0]
        )

    profile = h1.facilitation_profile(
        trials, estimator=estimator, correction=correction,
        expectancy_degree=expectancy_degree,
    )
    by_speed = h1.subject_profile(profile, by=["speed"])

    fits = speed_fits

    if "correction" in fits.columns:
        fits = fits[fits["correction"] == correction]

    fits = fits[fits["subject"].isin(trials["subject"].unique())]

    people = sorted(fits["subject"].dropna().unique())
    panels = people + ["group"]
    rows = int(np.ceil(len(panels) / columns))

    fig, axes = plt.subplots(
        rows, columns,
        figsize=(panel_size[0] * columns, panel_size[1] * rows),
        sharex=True, sharey=True, squeeze=False,
    )
    flat = axes.ravel()

    for ax, subject in zip(flat, people):
        _draw_speed_panel(
            ax, by_speed, fits, subject, _display_name(subject),
            trials=trials if show_trials else None,
            correction=correction, expectancy_degree=expectancy_degree,
        )

    # --- the group panel ---------------------------------------------
    ax = flat[len(people)]
    group_rows = []

    for speed in [config.SLOW_LABEL, config.FAST_LABEL]:
        mean_profile = (
            by_speed[by_speed["speed"] == speed]
            .groupby("position_rank")["facilitation_ms"]
            .mean()
            .sort_index()
        )

        if mean_profile.empty:
            continue

        y = mean_profile.to_numpy(dtype=float)
        x_observed = mean_profile.index.to_numpy(dtype=float)
        colour = _SPEED_COLOUR[speed]

        fit = safe_sigmoid_fit(x_observed, y)

        ax.plot(x_observed, y, "o", markersize=5, color=colour, zorder=3)

        if not fit.get("ok"):
            continue

        x = np.linspace(1, 7, 200)

        ax.plot(x, _speed_curve(y, fit["k"], fit["x_c"], x),
                color=colour, linewidth=2.4, zorder=2,
                label=f"{speed.capitalize()} "
                      f"($x_c$ = {fit['x_c']:.2f})")
        ax.axvline(fit["x_c"], color=colour, linestyle="--",
                   linewidth=1.0, alpha=0.6, zorder=1)

        group_rows.append({
            "speed": speed,
            "x_c": fit["x_c"],
            "se_x_c": fit["se_x_c"],
            "k": fit["k"],
            "k_at_bound": fit.get("k_at_bound", False),
        })

    ax.axhline(0, linestyle=":", linewidth=0.8, color=style.MUTED, zorder=0)
    ax.set_title(f"Group mean (n = {len(people)})", loc="left",
                 fontsize=9, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    style.clean(ax)

    for ax in flat[len(panels):]:
        ax.set_visible(False)

    # Scaled to the cell profiles, not to the individual trials. The
    # trials are context drawn behind the fit, and letting their spread
    # set the limits compresses every profile into the middle of the
    # panel, which is the thing the figure is meant to show.
    profiles_drawn = by_speed[by_speed["subject"].isin(people)]
    values = profiles_drawn["facilitation_ms"].dropna()

    if len(values):
        low, high = float(values.min()), float(values.max())
        pad = 0.12 * (high - low if high > low else 1.0)

        for ax in flat[:len(panels)]:
            ax.set_ylim(low - pad, high + pad)

    for ax in flat[:len(panels)]:
        ax.set_xticks(range(1, 8))
        ax.set_xticklabels([f"D{i}" for i in range(1, 8)], fontsize=7)

    for row in range(rows):
        axes[row, 0].set_ylabel("Facilitation (ms)", fontsize=9)

    for ax in flat[max(len(panels) - columns, 0):len(panels)]:
        ax.set_xlabel("Distance (D1 = closest)", fontsize=9)

    fig.suptitle(
        "Speed-specific boundary fits",
        x=0.02, ha="left", color=style.INK,
    )
    style.finish_figure(fig)

    if save_as:
        figures.save(fig, save_as)

    return fig, pd.DataFrame(group_rows)