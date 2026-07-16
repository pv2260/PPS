"""
H7a. Across young controls: does PPS plasticity go with Hit-or-Miss performance?

WHAT WE ARE TESTING
-------------------
Delta_PPS measures how much a person moves their PPS boundary when the stimulus
speeds up. It is a measure of how PLASTIC their body-space representation is.

If that plasticity is a real, general ability, then people who have more of it
should also do better on the Hit-or-Miss task. Three predictions:

    H8a.1  more PPS plasticity  ->  more collision-boundary plasticity
           correlate Delta_PPS with Delta_coll.  Predicted POSITIVE.

    H8a.2  more PPS plasticity  ->  better accuracy near the boundary
           correlate Delta_PPS with near-boundary accuracy.  Predicted POSITIVE.
           (Near-boundary accuracy is the PRIMARY measure. Overall accuracy is
           reported too, but only as a descriptive extra: most trials are easy,
           so overall accuracy is dominated by trials that tell us nothing.)

    H8a.3  more PPS plasticity  ->  LESS carryover
           correlate Delta_PPS with carryover.  Predicted NEGATIVE.
           Carryover means the previous trial's speed is still influencing your
           decision boundary. That is a failure to update, so someone who updates
           well should show less of it. Note this is the only one where we expect
           a NEGATIVE correlation, so we use alternative='less'.

WHY AN EXACT PERMUTATION TEST
-----------------------------
With 5 participants, a normal p-value from a correlation table is meaningless.
Instead we enumerate ALL 5! = 120 possible ways of pairing up the two variables
and count how many give a correlation at least as extreme as the real one.
That is exact, not approximate.

Be honest about what this can detect: at n = 5, a one-sided exact p < 0.05
requires rho >= 0.9. So a null result here does not mean there is no relationship.
It means we could not have detected anything short of a near-perfect one.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style

from .. import config
from .. import figures
from ..stats_utils import exact_spearman_permutation


def build_table(t):
    """One row per young control: Delta_PPS plus every Hit-or-Miss index."""

    # Average over sessions, so each participant contributes one row.
    pps_part = t.delta_pps_young.groupby("subject", as_index=False)["delta_pps"].mean()

    collision_part = t.collision_indices_young.groupby("subject", as_index=False).agg({
        "delta_coll": "mean",
        "accuracy": "mean",
        "near_boundary_accuracy": "mean",
        "carryover_cm": "mean",
    })

    # inner join: a participant needs BOTH tasks to appear here.
    table = pps_part.merge(collision_part, on="subject", how="inner")

    print("H7a data table (one row per young control):")
    print()
    print(table.round(3).to_string(index=False))
    print()
    print(f"Number of participants with both tasks: {table['subject'].nunique()}")
    print()

    return table


def correlate(table, x_column, y_column, direction, label):
    """Run one exact permutation Spearman correlation and print the result.

    direction is 'greater' if we predict a positive correlation, 'less' if we
    predict a negative one.
    """

    result = exact_spearman_permutation(
        table[x_column],
        table[y_column],
        alternative=direction,
    )

    rho = result["spearman_rho"]
    p_value = result["exact_permutation_p"]

    # Support requires BOTH the right sign AND a small p-value. A large |rho| in
    # the WRONG direction is not support, it is evidence against.
    if direction == "greater":
        correct_sign = rho > 0
    else:
        correct_sign = rho < 0

    supported = bool(correct_sign) and bool(p_value < config.ALPHA)

    print(f"{label}")
    print(f"  n                = {result['n']}")
    print(f"  Spearman rho     = {rho:.3f}")
    print(f"  exact p ({direction:7s}) = {p_value:.4f}")
    print(f"  supported        = {supported}")
    print()

    result["supported"] = supported
    result["x_column"] = x_column
    result["y_column"] = y_column

    return result


def scatter(ax, table, x_column, y_column, x_label, y_label, title, result):
    """Draw one labelled scatter plot with the correlation in the title."""

    ax.scatter(
        table[x_column],
        table[y_column],
        s=45,
        color=style.CONTROL,
        edgecolor=style.INK,
        linewidth=0.6,
        zorder=3,
    )

    # Label each dot with the participant's ID. With 5 subjects this is readable
    # and it lets you spot immediately if one person is driving the correlation.
    for _, row in table.dropna(subset=[x_column, y_column]).iterrows():
        ax.text(row[x_column], row[y_column], f"  {row['subject']}", fontsize=6)

    ax.axhline(0, linestyle="--", linewidth=0.8, color="gray")
    ax.axvline(0, linestyle="--", linewidth=0.8, color="gray")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(
        f"{title}\n"
        f"$\\rho$ = {result['spearman_rho']:.2f}, "
        f"p = {result['exact_permutation_p']:.3f}"
    )

    style.clean(ax)


def run(t, plot=True):
    """Run all three parts of H7a."""

    table = build_table(t)

    results = {"table": table}

    # ------------------------------------------------------------------
    # The three correlations.
    # ------------------------------------------------------------------

    results["h7a_1"] = correlate(
        table, "delta_pps", "delta_coll", "greater",
        "H7a.1: Delta_PPS with Delta_coll (predicted POSITIVE)",
    )

    results["h7a_2_near"] = correlate(
        table, "delta_pps", "near_boundary_accuracy", "greater",
        "H7a.2: Delta_PPS with near-boundary accuracy (PRIMARY, predicted POSITIVE)",
    )

    results["h7a_2_overall"] = correlate(
        table, "delta_pps", "accuracy", "greater",
        "H7a.2: Delta_PPS with overall accuracy (descriptive only)",
    )

    results["h7a_3"] = correlate(
        table, "delta_pps", "carryover_cm", "less",
        "H7a.3: Delta_PPS with carryover (predicted NEGATIVE)",
    )

    if plot:
        fig = make_figure(results)
        figures.save(fig, "h7a_correlations")
        plt.show()
        results["figure"] = fig

    return results


def make_figure(results):
    """Build the H7a figure. Four scatter panels."""

    table = results["table"]

    fig, axes = figures.new_figure(
        n_panels=4,
        width=figures.DOUBLE_COLUMN,
        height=2.4,
    )

    scatter(
        axes[0], table, "delta_pps", "delta_coll",
        r"$\Delta_{PPS}$", r"$\Delta_{coll}$ (cm)",
        "H7a.1", results["h7a_1"],
    )

    scatter(
        axes[1], table, "delta_pps", "near_boundary_accuracy",
        r"$\Delta_{PPS}$", "Near-boundary accuracy",
        "H7a.2 (primary)", results["h7a_2_near"],
    )

    scatter(
        axes[2], table, "delta_pps", "accuracy",
        r"$\Delta_{PPS}$", "Overall accuracy",
        "H7a.2 (descriptive)", results["h7a_2_overall"],
    )

    scatter(
        axes[3], table, "delta_pps", "carryover_cm",
        r"$\Delta_{PPS}$", "Carryover (cm)",
        "H7a.3", results["h7a_3"],
    )

    figures.label_panels(axes)
    fig.tight_layout()

    return fig
