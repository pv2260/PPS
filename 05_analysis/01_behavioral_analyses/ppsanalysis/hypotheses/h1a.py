"""
H1a. The facilitation profile is sigmoidal, not flat and not a straight line.

WHAT WE ARE TESTING
-------------------
H1 showed that facilitation is stronger near the body. But "stronger near the
body" could look like several different shapes:

    flat     : facilitation is the same everywhere (no PPS at all)
    linear   : facilitation increases steadily as the stimulus gets closer
    sigmoid  : facilitation is low far away, jumps up at some distance, then
               levels off close to the body

The sigmoid shape is the one that implies a BOUNDARY. A straight line does not:
it says the effect just grows gradually, with no special distance. So H1a asks
whether the sigmoid actually fits better than the other two.

HOW WE COMPARE THE THREE SHAPES
-------------------------------
We fit all three to each participant's profile and compare their AIC values.
AIC penalises models for having more parameters, so a sigmoid (5 parameters)
has to fit noticeably better than a straight line (3) to win.

    LOWER AIC = BETTER

We record two verdicts:

    "sigmoid_lower_aic"  (the preregistered criterion)
        Sigmoid AIC is lower than BOTH the flat AIC and the linear AIC.

    "preferred"          (a stricter, descriptive extra)
        Sigmoid AIC is lower by more than AIC_MARGIN (default 2), AND the
        boundary x_c falls inside the tested range, AND the slope k is not
        trivially small.

The prereg criterion is what we report as support. The stricter one is extra
information: if a participant passes the loose test but fails the strict one,
the sigmoid is winning by a hair and you should not lean on it.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style

from .. import config
from .. import figures
from ..stats_utils import sigmoid, fit_sigmoid_aic, compare_models_per_participant


def run(t, plot=True):
    """Run H1a."""

    results = {}

    # ------------------------------------------------------------------
    # Step 1. Work out the range of distances we actually tested.
    # ------------------------------------------------------------------
    # We need this because a sigmoid whose boundary falls OUTSIDE the tested
    # range is not a boundary. It is the fit giving up and putting the step
    # somewhere we never looked.

    valid_ranks = t.pps_trials["position_rank"].dropna()

    if len(valid_ranks) > 0:
        x_min = float(valid_ranks.min())
        x_max = float(valid_ranks.max())
    else:
        x_min = 1.0
        x_max = 7.0

    print(f"Distance ranks present in the data: {x_min:.0f} to {x_max:.0f}")
    print()

    results["x_min"] = x_min
    results["x_max"] = x_max

    # ------------------------------------------------------------------
    # Step 2. One facilitation profile per participant.
    # ------------------------------------------------------------------
    # Average over sessions and over speeds, so each participant has one
    # facilitation value per distance.

    profile_per_subject = (
        t.facilitation_young
        .dropna(subset=["position_rank", "facilitation_ms"])
        .groupby(["subject", "position_rank"], as_index=False)["facilitation_ms"]
        .mean()
    )

    results["profile_per_subject"] = profile_per_subject

    # ------------------------------------------------------------------
    # Step 3. Fit flat / linear / sigmoid to each participant.
    # ------------------------------------------------------------------

    rows = []

    for subject, one_subject in profile_per_subject.groupby("subject"):

        one_subject = one_subject.sort_values("position_rank")

        comparison = compare_models_per_participant(
            one_subject["position_rank"].to_numpy(),
            one_subject["facilitation_ms"].to_numpy(),
            x_min=x_min,
            x_max=x_max,
            aic_margin=config.AIC_MARGIN,
        )

        row = {"subject": subject}
        row.update(comparison)
        rows.append(row)

    # Build the table with explicit column names, so that even an EMPTY result
    # still has the right columns and the code below does not crash.
    columns = ["subject", "aic_flat", "aic_linear", "aic_sigmoid", "x_c", "k", "preferred"]
    table = pd.DataFrame(rows, columns=columns)

    # The preregistered criterion.
    table["sigmoid_lower_aic"] = (
        (table["aic_sigmoid"] < table["aic_flat"])
        & (table["aic_sigmoid"] < table["aic_linear"])
    )

    results["table"] = table

    # ------------------------------------------------------------------
    # Step 4. Nothing to do if there is no data.
    # ------------------------------------------------------------------

    if len(table) == 0:
        print("H1a: no participant has usable facilitation data.")
        print()
        print("Check that:")
        print("  - young_subjects is not empty")
        print("  - subjects are labelled group == 'control' in SUBJECT_META")
        print("  - VT and T trials exist at the SAME position within a subject/session")
        print("  - rt_ms is computable for both T and VT trials")

        results["n_supported"] = 0
        results["n_total"] = 0
        results["group_comparison"] = None
        return results

    # ------------------------------------------------------------------
    # Step 5. Also fit the group-averaged profile, as a descriptive extra.
    # ------------------------------------------------------------------

    group_average = (
        profile_per_subject
        .groupby("position_rank", as_index=False)["facilitation_ms"]
        .mean()
        .sort_values("position_rank")
    )

    # A sigmoid has 5 parameters. Fitting 5 parameters to fewer than 5 points is
    # meaningless, so we do not even try.
    if len(group_average) >= 5:
        group_comparison = compare_models_per_participant(
            group_average["position_rank"].to_numpy(),
            group_average["facilitation_ms"].to_numpy(),
            x_min=x_min,
            x_max=x_max,
            aic_margin=config.AIC_MARGIN,
        )
    else:
        group_comparison = {
            "preferred": "insufficient data",
            "x_c": np.nan,
            "k": np.nan,
            "aic_flat": np.nan,
            "aic_linear": np.nan,
            "aic_sigmoid": np.nan,
        }

    results["group_average"] = group_average
    results["group_comparison"] = group_comparison

    n_total = len(table)
    n_supported = int(table["sigmoid_lower_aic"].sum())
    n_preferred = int((table["preferred"] == "sigmoid").sum())

    results["n_total"] = n_total
    results["n_supported"] = n_supported
    results["n_preferred"] = n_preferred

    report(results)

    if plot:
        fig = make_figure(results)
        figures.save(fig, "h1a_sigmoid_shape")
        plt.show()
        results["figure"] = fig

    return results


def report(results):
    """Print the H1a numbers."""

    table = results["table"]

    print("H1a: model comparison for each participant")
    print("  lower AIC = better fit")
    print()
    print(table.round(2).to_string(index=False))
    print()

    print("H1a: group-averaged profile (descriptive)")
    print()
    print(pd.Series(results["group_comparison"]).round(3).to_string())
    print()

    n_total = results["n_total"]
    n_supported = results["n_supported"]
    n_preferred = results["n_preferred"]

    print("H1a verdict (preregistered criterion):")
    print(f"  sigmoid AIC lower than BOTH flat and linear "
          f"in {n_supported} of {n_total} participants")
    print()
    print("H1a extra (stricter, descriptive):")
    print(f"  sigmoid also wins by a margin of {config.AIC_MARGIN}, with a boundary")
    print(f"  inside the tested range, in {n_preferred} of {n_total} participants")


def make_figure(results):
    """Build the H1a figure. One panel, single-column width."""

    profile_per_subject = results["profile_per_subject"]
    group_average = results["group_average"]

    x_min = results["x_min"]
    x_max = results["x_max"]

    fig, axes = figures.new_figure(n_panels=1, width=figures.ONE_HALF_COLUMN, height=3.2)
    ax = axes[0]

    # One faint line per participant.
    for subject, one_subject in profile_per_subject.groupby("subject"):
        one_subject = one_subject.sort_values("position_rank")

        ax.plot(
            one_subject["position_rank"],
            one_subject["facilitation_ms"],
            marker="o",
            linestyle="-",
            color=style.CONTROL,
            alpha=0.6,
            linewidth=1.0,
            label=str(subject),
        )

    # The group sigmoid on top, but only if we have enough points to fit one.
    if len(group_average) >= 5:
        group_fit = fit_sigmoid_aic(
            group_average["position_rank"].to_numpy(),
            group_average["facilitation_ms"].to_numpy(),
        )

        if group_fit.get("ok"):
            smooth_x = np.linspace(x_min, x_max, 200)
            smooth_y = sigmoid(
                smooth_x,
                group_fit["low"],
                group_fit["high"],
                group_fit["x_c"],
                group_fit["k"],
            )

            ax.plot(
                smooth_x,
                smooth_y,
                color=style.INK,
                linewidth=2.0,
                label=f"group sigmoid ($x_c$ = {group_fit['x_c']:.2f})",
            )

    ax.axhline(0, linestyle="--", linewidth=0.8, color="gray")

    ax.set_xlabel("Distance rank (1 = closest to body)")
    ax.set_ylabel("Facilitation, T $-$ VT (ms)")
    ax.set_title(
        f"Sigmoid preferred in {results['n_preferred']}/{results['n_total']} participants"
    )
    ax.legend(ncol=2)

    style.clean(ax)
    fig.tight_layout()

    return fig
