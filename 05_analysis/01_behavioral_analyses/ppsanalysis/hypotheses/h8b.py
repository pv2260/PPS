"""
H8b. In the patient-control pair: does reduced PPS plasticity go WITH reduced
     Hit-or-Miss performance?

WHAT WE ARE TESTING
-------------------
H8a asked this across five healthy controls. H8b asks it within our one patient.

We have five deviations, each measured as (patient minus control):

    D_PPS       the patient's PPS plasticity, relative to their control
    D_coll      collision-boundary plasticity
    D_acc       overall accuracy
    D_nearacc   near-boundary accuracy
    D_carryover carryover

THE KEY IDEA: A JOINT SIGN CRITERION
------------------------------------
The naive way to do this is to test D_PPS separately, test D_coll separately,
and then say "both are negative, so they go together". That is WRONG, and it is
worth understanding why.

Testing them separately tells you: "D_PPS is probably negative" and "D_coll is
probably negative". It does NOT tell you that they are negative TOGETHER, in the
same resamples. They could each be negative 96% of the time, but negative at the
SAME TIME only 60% of the time, if the two measures are anti-correlated across
resamples.

What we actually want to claim is a CO-OCCURRENCE, so we have to measure the
co-occurrence directly:

    P( D_PPS < 0  AND  D_coll < 0 )  >=  0.95   ->  concordant under-calibration
    P( D_PPS > 0  AND  D_coll > 0 )  >=  0.95   ->  concordant over-calibration
    opposite-sign joint pattern      >=  0.95   ->  discordant
    nothing reaches 0.95                        ->  inconclusive

WHY THE BOOTSTRAP MUST BE PAIRED
--------------------------------
For the joint criterion to mean anything, resample #37 of the PPS task and
resample #37 of the Hit-or-Miss task must come from the SAME person on the SAME
bootstrap draw. So we merge on the bootstrap index. This is also why H8b reuses
H6's PPS bootstrap rather than running a fresh one: a fresh one would have a
different random pairing and the joint probabilities would be meaningless.

H8b.3 IS THE ODD ONE OUT
------------------------
For H8b.1 and H8b.2, "worse" means a NEGATIVE deviation (less plasticity, lower
accuracy). But for carryover, "worse" means a POSITIVE deviation: more carryover
means you are failing to update. So the concordant pattern for H8b.3 is
(D_PPS < 0 AND D_carryover > 0), i.e. OPPOSITE signs. This is easy to get wrong.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style

from .. import config
from .. import figures
from ..stats_utils import classify_h8b_concordant, classify_h8b_opposite
from ..collision import (
    compute_collision_indices_one_session,
    bootstrap_collision_indices_one_session,
)


def build_bootstrap(t, h6_results):
    """Build the paired bootstrap table that all three H8b tests share."""

    if not t.has_patient:
        print("H8b skipped: there is no patient in this dataset.")
        return {"skipped": True, "boot": pd.DataFrame()}

    patient_session = t.patient_session
    control_session = t.control_session

    # ------------------------------------------------------------------
    # Step 1. Collision trials for each person.
    # ------------------------------------------------------------------

    patient_trials = t.collision_trials[
        (t.collision_trials["subject"] == t.patient_id)
        & (t.collision_trials["session"] == patient_session)
    ].copy()

    control_trials = t.collision_trials[
        (t.collision_trials["subject"] == t.control_id)
        & (t.collision_trials["session"] == control_session)
    ].copy()

    # ------------------------------------------------------------------
    # Step 2. The OBSERVED deviations (patient minus control).
    # ------------------------------------------------------------------

    patient_indices = compute_collision_indices_one_session(patient_trials)
    control_indices = compute_collision_indices_one_session(control_trials)

    observed = {
        "d_pps": h6_results.get("patient_delta_pps", np.nan)
                 - h6_results.get("control_delta_pps", np.nan),
        "d_coll": patient_indices["delta_coll"] - control_indices["delta_coll"],
        "d_acc": patient_indices["accuracy"] - control_indices["accuracy"],
        "d_nearacc": (patient_indices["near_boundary_accuracy"]
                      - control_indices["near_boundary_accuracy"]),
        "d_carryover": patient_indices["carryover_cm"] - control_indices["carryover_cm"],
    }

    observed_table = pd.DataFrame([
        {"measure": "D_PPS",       "patient": h6_results.get("patient_delta_pps", np.nan),
         "control": h6_results.get("control_delta_pps", np.nan), "difference": observed["d_pps"]},
        {"measure": "D_coll",      "patient": patient_indices["delta_coll"],
         "control": control_indices["delta_coll"], "difference": observed["d_coll"]},
        {"measure": "D_acc",       "patient": patient_indices["accuracy"],
         "control": control_indices["accuracy"], "difference": observed["d_acc"]},
        {"measure": "D_nearacc",   "patient": patient_indices["near_boundary_accuracy"],
         "control": control_indices["near_boundary_accuracy"], "difference": observed["d_nearacc"]},
        {"measure": "D_carryover", "patient": patient_indices["carryover_cm"],
         "control": control_indices["carryover_cm"], "difference": observed["d_carryover"]},
    ])

    print("H8b: observed deviations (patient minus control)")
    print()
    print(observed_table.round(3).to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # Step 3. Bootstrap the collision task for each person.
    # ------------------------------------------------------------------

    patient_boot = bootstrap_collision_indices_one_session(
        patient_trials,
        n_boot=config.N_BOOT,
        seed=config.RANDOM_SEED + 100,
    )
    control_boot = bootstrap_collision_indices_one_session(
        control_trials,
        n_boot=config.N_BOOT,
        seed=config.RANDOM_SEED + 101,
    )

    collision_pair = patient_boot.merge(
        control_boot,
        on="boot",
        suffixes=("_patient", "_control"),
    )

    collision_pair["d_coll"] = (
        collision_pair["delta_coll_patient"] - collision_pair["delta_coll_control"])
    collision_pair["d_acc"] = (
        collision_pair["accuracy_patient"] - collision_pair["accuracy_control"])
    collision_pair["d_nearacc"] = (
        collision_pair["near_boundary_accuracy_patient"]
        - collision_pair["near_boundary_accuracy_control"])
    collision_pair["d_carryover"] = (
        collision_pair["carryover_cm_patient"] - collision_pair["carryover_cm_control"])

    # ------------------------------------------------------------------
    # Step 4. Merge with H6's PPS bootstrap ON THE BOOTSTRAP INDEX.
    # ------------------------------------------------------------------
    # This is the step that makes the joint criterion valid. Merging on "boot"
    # yokes the two tasks together, so row 37 is the same resample of the same
    # person in both.

    pps_pair = h6_results.get("h6_pair_boot", pd.DataFrame())

    if len(pps_pair) == 0:
        print("H8b: no PPS bootstrap available (H6 Test A did not pass), so the")
        print("  paired bootstrap cannot be built and H8b cannot run.")
        boot = pd.DataFrame()
    else:
        boot = pps_pair[["boot", "d_pps"]].merge(
            collision_pair[["boot", "d_coll", "d_acc", "d_nearacc", "d_carryover"]],
            on="boot",
            how="inner",
        )
        print(f"H8b paired bootstrap samples: {len(boot)}")
        print()

    return {
        "skipped": False,
        "boot": boot,
        "observed": observed,
        "observed_table": observed_table,
    }


def run_one(bootstrap, y_column, y_label, title, concordant_means_same_sign, plot_ax=None):
    """Run one H8b sub-test.

    concordant_means_same_sign :
        True  for H8b.1 and H8b.2, where "both worse" means both NEGATIVE.
        False for H8b.3, where "both worse" means D_PPS negative but carryover
              POSITIVE (more carryover = worse updating).
    """

    boot = bootstrap["boot"]
    observed = bootstrap["observed"]

    if len(boot) == 0:
        print(f"{title}: cannot run (no paired bootstrap).")
        return {"classification": "not estimable"}

    if concordant_means_same_sign:
        result = classify_h8b_concordant(
            boot["d_pps"],
            boot[y_column],
            threshold=config.BOOT_SIGN_THRESHOLD,
        )
    else:
        result = classify_h8b_opposite(
            boot["d_pps"],
            boot[y_column],
            threshold=config.BOOT_SIGN_THRESHOLD,
        )

    print(f"{title}")
    print()
    print(pd.Series(result).to_string())
    print()

    if plot_ax is not None:
        draw_joint_scatter(
            plot_ax,
            boot,
            y_column,
            observed["d_pps"],
            observed[y_column],
            y_label,
            f"{title}\n{result['classification']}",
        )

    return result


def draw_joint_scatter(ax, boot, y_column, observed_x, observed_y, y_label, title):
    """Scatter every bootstrap resample, with the observed value marked.

    The four quadrants are what matter. A concordant result means the cloud sits
    almost entirely in ONE quadrant. If the cloud straddles the axes, the result
    is inconclusive, and that is immediately visible here in a way it is not in a
    table of numbers.
    """

    ax.scatter(
        boot["d_pps"],
        boot[y_column],
        s=2,
        alpha=0.2,
        color=style.CONTROL,
        edgecolor="none",
        zorder=2,
    )

    ax.scatter(
        observed_x,
        observed_y,
        s=70,
        color=style.PATIENT,
        marker="X",
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
        label="observed",
    )

    ax.axhline(0, linestyle="--", linewidth=0.8, color="gray", zorder=1)
    ax.axvline(0, linestyle="--", linewidth=0.8, color="gray", zorder=1)

    ax.set_xlabel(r"$D_{PPS}$ (patient $-$ control)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()

    style.clean(ax)


def run(t, h6_results, plot=True):
    """Run all three parts of H8b. Needs the H6 results for the paired bootstrap."""

    bootstrap = build_bootstrap(t, h6_results)

    if bootstrap["skipped"] or len(bootstrap["boot"]) == 0:
        return {"skipped": True}

    results = {"skipped": False, "bootstrap": bootstrap}

    if plot:
        fig, axes = figures.new_figure(n_panels=3, width=figures.DOUBLE_COLUMN, height=2.8)
    else:
        axes = [None, None, None]

    results["h8b_1"] = run_one(
        bootstrap, "d_coll", r"$D_{coll}$ (cm)", "H8b.1",
        concordant_means_same_sign=True, plot_ax=axes[0],
    )

    results["h8b_2_near"] = run_one(
        bootstrap, "d_nearacc", r"$D_{nearacc}$", "H8b.2 (primary)",
        concordant_means_same_sign=True, plot_ax=axes[1],
    )

    # H8b.3: "worse" carryover is MORE carryover, so the concordant pattern has
    # OPPOSITE signs. See the module docstring.
    results["h8b_3"] = run_one(
        bootstrap, "d_carryover", r"$D_{carryover}$ (cm)", "H8b.3",
        concordant_means_same_sign=False, plot_ax=axes[2],
    )

    # Also run overall accuracy, but do not plot it: it is descriptive only.
    results["h8b_2_overall"] = run_one(
        bootstrap, "d_acc", r"$D_{acc}$", "H8b.2 (descriptive: overall accuracy)",
        concordant_means_same_sign=True, plot_ax=None,
    )

    if plot:
        figures.label_panels(axes)
        fig.tight_layout()
        figures.save(fig, "h8b_joint_signs")
        plt.show()
        results["figure"] = fig

    # ------------------------------------------------------------------
    # Summary table.
    # ------------------------------------------------------------------

    summary = pd.DataFrame([
        {"hypothesis": "H8b.1",                    "classification": results["h8b_1"]["classification"]},
        {"hypothesis": "H8b.2 (near, primary)",    "classification": results["h8b_2_near"]["classification"]},
        {"hypothesis": "H8b.2 (overall, descr.)",  "classification": results["h8b_2_overall"]["classification"]},
        {"hypothesis": "H8b.3",                    "classification": results["h8b_3"]["classification"]},
    ])

    print("H8b summary")
    print()
    print(summary.to_string(index=False))

    results["summary"] = summary

    return results
