"""
H4. Test-retest reliability: do our measures give the same answer twice?

WHAT WE ARE TESTING
-------------------
Every participant does the task twice. If a measure is any good, a participant
who scored high in session 1 should also score high in session 2.

We check four measures:

    x_c         the PPS boundary (pooled over speeds)
    pse_pooled  the collision boundary (pooled over speeds)
    delta_pps   the speed-induced shift in the PPS boundary
    delta_coll  the speed-induced shift in the collision boundary

WHAT WE COMPUTE
---------------
1. Spearman rank correlation between session 1 and session 2.
   With only 5 participants we cannot use a normal p-value, so we enumerate all
   5! = 120 possible orderings and count how many give a correlation at least as
   big as the one we saw. That is an EXACT permutation p-value.

2. SDC, the Smallest Detectable Change:

       SDC = 1.96 * SD(session1 - session2) * sqrt(2)

   This is the NOISE FLOOR. Any within-person change smaller than the SDC could
   just be measurement error. H6b uses the SDC for delta_pps to decide whether a
   dopamine effect is real or is just the task being noisy.

INTERPRETATION
--------------
At n = 5 this is DESCRIPTIVE. We are not testing a hypothesis here, we are
reporting how noisy our own measures are. The SDC is the number that actually
gets used later.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style

from .. import config
from .. import figures
from ..stats_utils import exact_spearman_permutation, compute_sdc


def build_session_pairs(measure_table, value_column):
    """Reshape a per-session measure into one row per subject: s1 and s2.

    Only subjects who have BOTH sessions are kept. A subject with one session
    cannot tell us anything about reliability.
    """

    columns_we_need = ["subject", "session", value_column]
    simple = measure_table[columns_we_need].dropna()

    if len(simple) == 0:
        return pd.DataFrame(columns=["subject", "s1", "s2"])

    # One column per session.
    wide = simple.pivot_table(
        index="subject",
        columns="session",
        values=value_column,
        aggfunc="mean",
    )

    sessions = sorted(wide.columns.tolist())

    # If there is only one session in the data, there is nothing to correlate.
    if len(sessions) < 2:
        return pd.DataFrame(columns=["subject", "s1", "s2"])

    wide = wide.rename(columns={sessions[0]: "s1", sessions[1]: "s2"})

    # dropna() here removes subjects who are missing one of the two sessions.
    return wide[["s1", "s2"]].dropna().reset_index()


def run(t, plot=True):
    """Run H4."""

    results = {}

    # ------------------------------------------------------------------
    # Step 1. Build the collision PSE, pooled over speeds.
    # ------------------------------------------------------------------
    # The collision indices table stores pse_slow and pse_fast separately.
    # The prereg asks for the POOLED boundary, so we average the two.

    collision_with_pooled_pse = t.collision_indices_young.copy()

    collision_with_pooled_pse["pse_pooled"] = (
        collision_with_pooled_pse["pse_slow"]
        + collision_with_pooled_pse["pse_fast"]
    ) / 2.0

    # ------------------------------------------------------------------
    # Step 2. The four measures we test, and where each one lives.
    # ------------------------------------------------------------------

    measures = [
        ("x_c",        t.xc_young,                 "rank"),
        ("pse_pooled", collision_with_pooled_pse,  "cm"),
        ("delta_pps",  t.delta_pps_young,          "rank"),
        ("delta_coll", t.collision_indices_young,  "cm"),
    ]

    results["measures"] = measures

    # ------------------------------------------------------------------
    # Step 3. Correlate session 1 with session 2 for each measure.
    # ------------------------------------------------------------------

    rows = []

    for column, table, unit in measures:

        pairs = build_session_pairs(table, column)

        # Fewer than 2 subjects with both sessions: nothing to correlate.
        if len(pairs) < 2:
            rows.append({
                "measure": column,
                "unit": unit,
                "n_subjects": len(pairs),
                "spearman_rho": np.nan,
                "exact_p": np.nan,
                "sdc": np.nan,
            })
            continue

        spearman = exact_spearman_permutation(
            pairs["s1"],
            pairs["s2"],
            alternative="greater",
        )

        sdc = compute_sdc(pairs["s1"], pairs["s2"])

        rows.append({
            "measure": column,
            "unit": unit,
            "n_subjects": len(pairs),
            "spearman_rho": spearman["spearman_rho"],
            "exact_p": spearman["exact_permutation_p"],
            "sdc": sdc,
        })

    reliability = pd.DataFrame(rows)
    results["reliability"] = reliability

    # ------------------------------------------------------------------
    # Step 4. Pull out the SDC for delta_pps. H6b needs it.
    # ------------------------------------------------------------------

    delta_pps_row = reliability[reliability["measure"] == "delta_pps"]

    if len(delta_pps_row) > 0 and pd.notna(delta_pps_row["sdc"].iloc[0]):
        sdc_delta_pps = float(delta_pps_row["sdc"].iloc[0])
    else:
        sdc_delta_pps = np.nan

    # Named in capitals because the notebook assigns it to t.sdc_delta_pps and
    # hands it to H6b. This is the one number H4 exports.
    results["SDC_DELTA_PPS"] = sdc_delta_pps

    report(results)

    if plot:
        fig = make_figure(results)
        figures.save(fig, "h4_reliability")
        plt.show()
        results["figure"] = fig

    return results


def report(results):
    """Print the H4 numbers."""

    print("H4: test-retest reliability (descriptive, n is small)")
    print()
    print(results["reliability"].round(3).to_string(index=False))
    print()

    sdc = results["SDC_DELTA_PPS"]

    if np.isfinite(sdc):
        print(f"SDC for Delta_PPS = {sdc:.3f}")
        print("  This is the noise floor. H6b will only call a dopamine effect real")
        print("  if it is BIGGER than this.")
    else:
        print("SDC for Delta_PPS: not estimable (need at least 2 subjects with 2 sessions).")
        print("  H6b will therefore have no noise floor to check against.")


def make_figure(results):
    """Build the H4 figure. One panel per measure."""

    measures = results["measures"]

    fig, axes = figures.new_figure(
        n_panels=len(measures),
        width=figures.DOUBLE_COLUMN,
        height=2.2,
    )

    for ax, (column, table, unit) in zip(axes, measures):

        pairs = build_session_pairs(table, column)

        if len(pairs) >= 2:

            ax.scatter(
                pairs["s1"],
                pairs["s2"],
                s=35,
                color=style.CONTROL,
                edgecolor=style.INK,
                linewidth=0.6,
                zorder=3,
            )

            for _, row in pairs.iterrows():
                ax.text(row["s1"], row["s2"], f"  {row['subject']}", fontsize=6)

            # The identity line. A perfectly reliable measure would put every dot
            # exactly on it.
            low = min(pairs["s1"].min(), pairs["s2"].min())
            high = max(pairs["s1"].max(), pairs["s2"].max())

            ax.plot([low, high], [low, high],
                    linestyle="--", color="gray", linewidth=0.8, alpha=0.7, zorder=1)
        else:
            ax.text(
                0.5, 0.5,
                "not enough\nsubjects with\nboth sessions",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=6,
                color="#8A8A8A",
            )

        ax.set_xlabel(f"Session 1 ({unit})")
        ax.set_ylabel(f"Session 2 ({unit})")
        ax.set_title(column)

        style.clean(ax)

    figures.label_panels(axes)
    fig.tight_layout()

    return fig
