"""
H1. PPS facilitation increases as the stimulus approaches.

WHAT WE ARE TESTING
-------------------
When a looming visual stimulus is close to the body, a touch on the hand should
be detected FASTER than when the stimulus is far away. That speed-up is called
"facilitation".

We measure facilitation at each distance as:

    facilitation = median RT on tactile-only trials
                 - median RT on visuo-tactile trials

Positive facilitation means the visual stimulus made the person faster.

Then we summarise each person with one number, the "near-far index":

    NearFar_PPS = mean facilitation at the 2 NEAREST distances (D1, D2)
                - mean facilitation at the 2 FARTHEST distances (D6, D7)

If PPS exists, facilitation should be bigger near the body, so NearFar_PPS
should be positive.

HOW WE DECIDE IF IT IS SUPPORTED
--------------------------------
We take one NearFar_PPS value per participant and bootstrap the group mean.
H1 is supported if the bootstrap 95% confidence interval lies entirely above
zero. A one-sided Wilcoxon test is reported as well, but it does NOT decide
support: it is only a sensitivity check. This is what the preregistration says.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style

from .. import config
from .. import figures
from ..stats_utils import bootstrap_mean_ci, wilcoxon_one_sided


def run(t, plot=True):
    """Run H1. `t` is the Tables object built in tables.py."""

    results = {}

    # ------------------------------------------------------------------
    # Step 1. Facilitation at each distance, for each participant.
    # ------------------------------------------------------------------
    # facilitation_young already has one row per
    # (subject, session, speed, position). We average over sessions and speeds
    # so that each participant has one facilitation value per distance.

    facilitation = t.facilitation_young

    facilitation_by_distance = facilitation.groupby(
        ["subject", "position_rank"],
        as_index=False,
    )["facilitation_ms"].mean()

    # If Unity logged the real distance in metres, attach it. It is only used
    # for labelling; the fitting still uses position_rank.
    has_metres = "distance_m" in facilitation.columns
    if has_metres and facilitation["distance_m"].notna().any():
        distance_lookup = (
            facilitation[["position_rank", "distance_m"]]
            .dropna()
            .drop_duplicates("position_rank")
        )
        facilitation_by_distance = facilitation_by_distance.merge(
            distance_lookup,
            on="position_rank",
            how="left",
        )

    # Reshape so that rows = distance, columns = participant.
    # This is the table we plot in panel A.
    profile = facilitation_by_distance.pivot_table(
        index="position_rank",
        columns="subject",
        values="facilitation_ms",
    )

    results["facilitation_by_distance"] = facilitation_by_distance
    results["profile"] = profile

    # ------------------------------------------------------------------
    # Step 2. One near-far index per participant.
    # ------------------------------------------------------------------
    # t.nearfar_young has one row per (subject, session). Average over sessions
    # so that each participant contributes exactly one number to the group test.
    # (Averaging is fine here: nearfar_pps is near_mean - far_mean, and the mean
    # of a difference equals the difference of the means.)

    nearfar_per_subject = t.nearfar_young.groupby(
        "subject",
        as_index=False,
    )[["near_mean", "far_mean", "nearfar_pps"]].mean()

    results["nearfar_per_subject"] = nearfar_per_subject

    # ------------------------------------------------------------------
    # Step 3. Group test.
    # ------------------------------------------------------------------

    values = nearfar_per_subject["nearfar_pps"].dropna().to_numpy()

    boot = bootstrap_mean_ci(
        values,
        n_boot=config.N_BOOT,
        ci=95,
        seed=config.RANDOM_SEED,
    )

    wilcoxon = wilcoxon_one_sided(values, alternative="greater")

    # The preregistered criterion: the bootstrap CI must be entirely above zero.
    supported = bool(boot["ci_low"] > 0)

    results["values"] = values
    results["boot"] = boot
    results["wilcoxon"] = wilcoxon
    results["supported"] = supported

    # ------------------------------------------------------------------
    # Step 4. Report and plot.
    # ------------------------------------------------------------------

    report(results)

    if plot:
        fig = make_figure(results)
        figures.save(fig, "h1_facilitation")
        plt.show()
        results["figure"] = fig

    return results


def report(results):
    """Print the H1 numbers in a readable way."""

    profile = results["profile"]
    nearfar = results["nearfar_per_subject"]
    boot = results["boot"]
    wilcoxon = results["wilcoxon"]

    print("H1, step 1: facilitation (ms) at each distance")
    print("  rows = distance rank, 1 = closest to the body")
    print("  positive = the visual stimulus made the person faster")
    print()
    print(profile.round(1).to_string())
    print()

    print("H1, step 2: near-far index per participant")
    print("  near = D1 and D2, far = D6 and D7")
    print()
    print(nearfar.round(2).to_string(index=False))
    print()

    print("H1, step 3: group test on the near-far index")
    print(f"  number of participants  = {boot['n']}")
    print(f"  mean NearFar_PPS        = {boot['mean']:.2f} ms")
    print(f"  bootstrap 95% CI        = [{boot['ci_low']:.2f}, {boot['ci_high']:.2f}] ms")
    print(f"  Wilcoxon (one-sided)    = W {wilcoxon['W']}, p {wilcoxon['p']:.4f}")
    print()
    print("  The bootstrap CI decides support. The Wilcoxon is a sensitivity check only.")
    print()

    if results["supported"]:
        print("  H1 IS SUPPORTED: the 95% CI lies entirely above zero.")
    else:
        print("  H1 is NOT supported: the 95% CI includes zero.")


def make_figure(results):
    """Build the H1 figure. Two panels, double-column width."""

    profile = results["profile"]
    nearfar = results["nearfar_per_subject"]
    values = results["values"]
    boot = results["boot"]

    fig, axes = figures.new_figure(n_panels=2, height=3.0)

    # --- Panel A: facilitation profile, one line per participant -------------
    ax = axes[0]

    for subject in profile.columns:
        ax.plot(
            profile.index,
            profile[subject],
            marker="o",
            linewidth=1.0,
            alpha=0.55,
            label=str(subject),
        )

    # The group mean, drawn thicker so it stands out.
    if profile.shape[1] >= 1:
        group_mean = profile.mean(axis=1)
        ax.plot(
            group_mean.index,
            group_mean.values,
            marker="o",
            color=style.INK,
            linewidth=2.0,
            label="group mean",
        )

    # A line at zero: above it, the visual stimulus helped; below it, it hurt.
    ax.axhline(0, linestyle="--", linewidth=0.8, color="gray")

    ax.set_xlabel("Distance rank (1 = closest to body)")
    ax.set_ylabel("Facilitation, T $-$ VT (ms)")
    ax.set_title("Facilitation by distance")

    # Only show a legend if there are few enough participants for it to be
    # readable. With 20 subjects a legend is just clutter.
    if profile.shape[1] <= 6:
        ax.legend()

    style.clean(ax)

    # --- Panel B: near-far index, one dot per participant --------------------
    ax = axes[1]

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

    # Label each dot with the participant's ID.
    for i, subject in enumerate(nearfar["subject"]):
        ax.text(x_positions[i], values[i], f" {subject}", fontsize=6, va="center")

    ax.axhline(0, linestyle="--", linewidth=0.8, color="gray")

    # The group mean with its bootstrap CI, drawn to the right of the dots.
    if np.isfinite(boot["mean"]):
        mean_x = len(values) + 0.5

        lower_error = boot["mean"] - boot["ci_low"]
        upper_error = boot["ci_high"] - boot["mean"]

        ax.errorbar(
            mean_x,
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
    ax.set_xticklabels(list(nearfar["subject"]) + ["mean"], rotation=30, ha="right")
    ax.set_ylabel("NearFar$_{PPS}$ (ms)")
    ax.set_title("Near-far index")
    ax.legend()

    style.clean(ax)

    figures.label_panels(axes)
    fig.tight_layout()

    return fig
