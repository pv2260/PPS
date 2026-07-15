"""
H2. The PPS boundary moves outward when the stimulus approaches faster.

WHAT WE ARE TESTING
-------------------
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

A NOTE ON THE SIGMOID FIT
-------------------------
A sigmoid will happily "fit" data that has no boundary in it at all. It just
puts the inflection outside the range you tested, or makes the slope absurdly
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
    safe_sigmoid_fit,
    bootstrap_mean_ci,
    wilcoxon_one_sided,
)


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

    # Is the boundary inside the range we tested? If it is outside, we are
    # extrapolating, and the "boundary" is an artefact of the fit.
    if not (x_min < x_c < x_max):
        return False

    # Is the slope plausible? A huge k means a vertical step, which usually
    # means the fit latched onto noise between two adjacent points.
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

    facilitation = t.facilitation_young.dropna(
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

            fit = safe_sigmoid_fit(
                one_speed["position_rank"].to_numpy(),
                one_speed["facilitation_ms"].to_numpy(),
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

    for i, subject in enumerate(delta_per_subject["subject"]):
        ax.text(x_positions[i], values[i], f" {subject}", fontsize=6, va="center")

    ax.axhline(0, linestyle="--", linewidth=0.8, color="gray")

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

    ax.set_xticks(list(x_positions) + [len(values) + 0.5])
    ax.set_xticklabels(list(delta_per_subject["subject"]) + ["mean"], rotation=30, ha="right")
    ax.set_ylabel(r"$\Delta_{PPS}$")
    ax.set_title("Speed-induced boundary shift")
    ax.legend()

    style.clean(ax)

    # --- Panel B: facilitation profile per speed -----------------------------
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

        # Faint raw points, so the reader can see how noisy the data actually is.
        # Without these, a clean line can look convincing while resting on nothing.
        raw_this_speed = raw[raw["speed"] == speed]

        ax.scatter(
            raw_this_speed["position_rank"] + style.jitter(len(raw_this_speed), 0.06, seed=1),
            raw_this_speed["facilitation_ms"],
            s=8,
            color=style.shade(colour, 0.35),
            alpha=0.5,
            edgecolor="none",
            zorder=2,
        )

        # The pooled mean at each distance.
        pooled_this_speed = pooled[pooled["speed"] == speed].sort_values("position_rank")

        ax.scatter(
            pooled_this_speed["position_rank"],
            pooled_this_speed["facilitation_ms"],
            s=25,
            color=style.shade(colour, -0.1),
            alpha=0.9,
            edgecolor="none",
            zorder=4,
        )

        # Draw the fitted sigmoid ONLY if we decided we can trust it.
        fit = fits[speed]

        if trustworthy[speed]:
            smooth_y = sigmoid(smooth_x, fit["low"], fit["high"], fit["x_c"], fit["k"])

            ax.plot(smooth_x, smooth_y, color=colour, linewidth=1.8,
                    label=f"{speed} (fitted)", zorder=3)

            # A dotted line marking where the boundary is.
            midpoint_y = (fit["low"] + fit["high"]) / 2
            ax.plot([fit["x_c"], fit["x_c"]], [0, midpoint_y],
                    linestyle=":", linewidth=1.0,
                    color=style.shade(colour, -0.1), zorder=1)
        else:
            # Not trustworthy: just connect the points with a dashed line and
            # say so, rather than drawing a curve that implies a boundary.
            ax.plot(
                pooled_this_speed["position_rank"],
                pooled_this_speed["facilitation_ms"],
                linestyle="--",
                color=colour,
                linewidth=1.2,
                label=f"{speed} (no fit)",
                zorder=3,
            )

    ax.axhline(0, linestyle="--", linewidth=0.8, color="gray")

    # If BOTH fits are trustworthy, draw the arrow showing Delta_PPS.
    both_ok = trustworthy.get(config.SLOW_LABEL) and trustworthy.get(config.FAST_LABEL)

    if both_ok:
        x_c_slow = fits[config.SLOW_LABEL]["x_c"]
        x_c_fast = fits[config.FAST_LABEL]["x_c"]

        arrow_y = np.mean([
            (fits[s]["low"] + fits[s]["high"]) / 2
            for s in [config.SLOW_LABEL, config.FAST_LABEL]
        ])

        ax.annotate(
            "",
            xy=(x_c_fast, arrow_y),
            xytext=(x_c_slow, arrow_y),
            arrowprops=dict(arrowstyle="<->", color=style.INK, lw=0.9),
        )
        ax.text(
            (x_c_slow + x_c_fast) / 2,
            arrow_y,
            f"  $\\Delta_{{PPS}}$ = {x_c_fast - x_c_slow:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=style.INK,
        )
    else:
        ax.text(
            0.5, 0.02,
            "sigmoid fit rejected; showing points only",
            transform=ax.transAxes,
            ha="center",
            fontsize=6,
            color="#8A8A8A",
        )

    ax.set_xlabel("Distance rank (1 = closest to body)")
    ax.set_ylabel("Facilitation, T $-$ VT (ms)")
    ax.set_title("Facilitation by distance and speed")
    ax.legend()

    style.clean(ax)

    figures.label_panels(axes)
    fig.tight_layout()

    return fig
