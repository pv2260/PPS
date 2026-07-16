"""
H6. Is the patient's speed-dependent COLLISION boundary shift altered?

This is the Hit-or-Miss twin of H6. Same two-test structure, same gating:

    TEST A. Does the patient show any Delta_coll?
            Permute the slow/fast labels on their collision trials.

    TEST B. Does the patient differ from their matched control?
            Bootstrap  D_coll = Delta_coll(patient) - Delta_coll(control).
            Only interpreted if Test A passed.

WHICH SESSIONS
--------------
The same ones H6 used. H6 writes its choice into t.patient_session, and H7 reads
it. If you run H7 without running H6 first, this module raises an error rather
than silently guessing, because guessing the wrong session would change the
answer.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style

from .. import config
from .. import figures
from ..collision import (
    compute_collision_indices_one_session,
    bootstrap_collision_indices_one_session,
)
from ..permutations import permute_speed_labels_delta_coll


def run(t, plot=True):
    """Run H6. Requires that h6.run(t) has already been called."""

    if not t.has_patient:
        print("H6 skipped: there is no patient in this dataset.")
        return {"skipped": True}

    # H6 chooses which patient session to use. If it has not run, we stop, rather
    # than picking a session ourselves and possibly using the wrong one.
    if t.patient_session is None:
        raise RuntimeError(
            "H6 needs t.patient_session, which H6 sets.\n"
            "Run H6 first:\n"
            "    r_h6 = h6.run(t)\n"
            "    t.patient_session = r_h6['PATIENT_SESSION_USED']\n"
        )

    results = {"skipped": False}

    patient_session = t.patient_session
    control_session = t.control_session

    print(f"Patient session: {patient_session}")
    print(f"Control session: {control_session}")
    print()

    # ------------------------------------------------------------------
    # Step 1. Get each person's collision trials.
    # ------------------------------------------------------------------

    patient_trials = t.collision_trials[
        (t.collision_trials["subject"] == t.patient_id)
        & (t.collision_trials["session"] == patient_session)
    ].copy()

    control_trials = t.collision_trials[
        (t.collision_trials["subject"] == t.control_id)
        & (t.collision_trials["session"] == control_session)
    ].copy()

    results["patient_trials"] = patient_trials
    results["control_trials"] = control_trials

    # ------------------------------------------------------------------
    # Step 2. The observed Delta_coll for each.
    # ------------------------------------------------------------------

    patient_indices = compute_collision_indices_one_session(patient_trials)
    control_indices = compute_collision_indices_one_session(control_trials)

    patient_delta_coll = patient_indices["delta_coll"]
    control_delta_coll = control_indices["delta_coll"]

    d_coll_observed = patient_delta_coll - control_delta_coll

    results["patient_indices"] = patient_indices
    results["control_indices"] = control_indices
    results["d_coll_observed"] = d_coll_observed

    print(f"Patient Delta_coll = {patient_delta_coll:.2f} cm")
    print(f"Control Delta_coll = {control_delta_coll:.2f} cm")
    print(f"Observed D_coll    = {d_coll_observed:.2f} cm")
    print()

    # ------------------------------------------------------------------
    # Step 3. TEST A.
    # ------------------------------------------------------------------

    print("--- Test A: does the patient show any Delta_coll? ---")

    test_a = permute_speed_labels_delta_coll(
        patient_trials,
        n_perm=config.N_BOOT,
        seed=config.RANDOM_SEED,
    )

    test_a_supported = (test_a["observed"] > 0) and (test_a["p"] < config.ALPHA)

    print(f"  observed Delta_coll = {test_a['observed']:.2f} cm")
    print(f"  permutation p       = {test_a['p']:.4f}")
    print(f"  Test A supported    = {test_a_supported}")
    print()

    results["test_a"] = test_a
    results["test_a_supported"] = test_a_supported

    # ------------------------------------------------------------------
    # Step 4. TEST B, gated on Test A.
    # ------------------------------------------------------------------

    if not test_a_supported:
        print("Test A did not pass, so Test B is NOT run.")

        results["h6_classification"] = "Test A failed; Test B not run"
        results["h6_pair_boot"] = pd.DataFrame()

    else:
        print("--- Test B: does the patient differ from the matched control? ---")

        patient_boot = bootstrap_collision_indices_one_session(
            patient_trials,
            n_boot=config.N_BOOT,
            seed=config.RANDOM_SEED + 10,
        )
        control_boot = bootstrap_collision_indices_one_session(
            control_trials,
            n_boot=config.N_BOOT,
            seed=config.RANDOM_SEED + 11,
        )

        paired = patient_boot.merge(
            control_boot,
            on="boot",
            suffixes=("_patient", "_control"),
        )

        paired["d_coll"] = paired["delta_coll_patient"] - paired["delta_coll_control"]

        proportion_negative = float((paired["d_coll"] < 0).mean())
        proportion_positive = float((paired["d_coll"] > 0).mean())

        if proportion_negative >= config.BOOT_SIGN_THRESHOLD:
            classification = "supported: under-calibration"
        elif proportion_positive >= config.BOOT_SIGN_THRESHOLD:
            classification = "supported: over-calibration"
        else:
            classification = "inconclusive"

        ci_low, ci_high = np.percentile(paired["d_coll"].dropna(), [2.5, 97.5])

        print(f"  D_coll 95% CI   = [{ci_low:.2f}, {ci_high:.2f}] cm")
        print(f"  P(D_coll < 0)   = {proportion_negative:.3f}")
        print(f"  P(D_coll > 0)   = {proportion_positive:.3f}")
        print(f"  classification  = {classification}")

        results["h6_pair_boot"] = paired
        results["h6_classification"] = classification
        results["ci_low"] = ci_low
        results["ci_high"] = ci_high

    if plot:
        fig = make_figure(results)
        figures.save(fig, "h6_patient_collision")
        plt.show()
        results["figure"] = fig

    return results


def make_figure(results):
    """Build the H6 figure. Test A null on the left, Test B bootstrap on the right."""

    fig, axes = figures.new_figure(n_panels=2, height=2.8)

    # --- Panel A: Test A null ------------------------------------------------
    ax = axes[0]

    test_a = results["test_a"]

    ax.hist(
        test_a["permuted"],
        bins=40,
        color=style.CONTROL,
        edgecolor="white",
        linewidth=0.3,
    )

    ax.axvline(
        test_a["observed"],
        color=style.PATIENT,
        linewidth=1.8,
        label=f"observed = {test_a['observed']:.2f} cm",
    )
    ax.axvline(0, linestyle="--", linewidth=0.8, color="gray")

    ax.set_xlabel(r"$\Delta_{coll}$ under shuffled speed labels (cm)")
    ax.set_ylabel("Count")
    ax.set_title(f"Test A: p = {test_a['p']:.4f}")
    ax.legend()

    style.clean(ax)

    # --- Panel B: Test B bootstrap -------------------------------------------
    ax = axes[1]

    paired = results["h6_pair_boot"]

    if len(paired) == 0:
        ax.text(
            0.5, 0.5,
            "Test B not run\n(Test A did not pass)",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=8,
            color="#8A8A8A",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.hist(
            paired["d_coll"].dropna(),
            bins=40,
            color=style.PATIENT,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.3,
        )

        ax.axvline(
            results["d_coll_observed"],
            color=style.INK,
            linewidth=1.8,
            label=f"observed = {results['d_coll_observed']:.2f} cm",
        )
        ax.axvline(0, linestyle="--", linewidth=0.8, color="gray")

        ax.set_xlabel(r"$D_{coll}$ = patient $-$ control (cm)")
        ax.set_ylabel("Count")
        ax.set_title(results["h6_classification"])
        ax.legend()

        style.clean(ax)

    figures.label_panels(axes)
    fig.tight_layout()

    return fig
