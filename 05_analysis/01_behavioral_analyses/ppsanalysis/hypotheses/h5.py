"""
H5. Is the patient's speed-dependent PPS recalibration altered?

WHAT WE ARE TESTING
-------------------
H2 showed (or did not show) that healthy people move their PPS boundary outward
when the stimulus comes at them faster. H5 asks two questions about the patient:

    TEST A. Does the patient recalibrate AT ALL?
            Is their Delta_PPS bigger than chance?

    TEST B. Does the patient recalibrate DIFFERENTLY from their matched control?
            Is  D_PPS = Delta_PPS(patient) - Delta_PPS(control)  reliably non-zero?

TEST B IS GATED ON TEST A
-------------------------
We only interpret Test B if Test A passes. The reason: if the patient shows no
recalibration signal at all, then their Delta_PPS is just noise, and comparing
noise to a control tells you nothing. Running Test B anyway would let you claim
"reduced recalibration" when what you actually have is "no measurement".

WHICH SESSIONS
--------------
The prereg says: patient's HIGH-dopamine session, control's session 1.
H7 and H8b both reuse this choice, so we save it into t.patient_session.

HOW TEST A WORKS
----------------
Same permutation logic as H5, but we shuffle SPEED labels instead of distance
labels. If speed does not matter, then relabelling slow trials as fast (and vice
versa) should not change Delta_PPS. Do that 5000 times and you get the null.

HOW TEST B WORKS
----------------
We bootstrap the patient's trials and the control's trials, recompute Delta_PPS
for each on every resample, and take the difference. Then we ask: what fraction
of the resamples give a difference of the same sign?

    D_PPS < 0 in >= 95% of resamples  ->  under-calibration
    D_PPS > 0 in >= 95% of resamples  ->  over-calibration
    otherwise                         ->  inconclusive
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style

from .. import config
from .. import figures
from ..pps import compute_facilitation, sigmoid_boundary_by_subject, bootstrap_pps_delta_one_session
from ..permutations import permute_speed_labels_delta_pps


def pick_high_dopamine_session(patient_trials, preferred_cohort="high"):
    """Find the patient's HIGH-dopamine session.

    Falls back to the first session if no cohort is labelled 'high'. That
    fallback is deliberate: it lets the pipeline run end to end on data where
    dopamine state was not recorded, rather than crashing. But if it fires, the
    session you get is NOT necessarily the high-dopamine one, so we say so.
    """

    sessions = (
        patient_trials[["session", "cohort"]]
        .dropna(subset=["session"])
        .drop_duplicates()
    )

    if len(sessions) == 0:
        return None, False

    matches_cohort = sessions["cohort"].str.lower().str.contains(
        preferred_cohort,
        na=False,
    )

    if matches_cohort.any():
        session = int(sessions.loc[matches_cohort, "session"].iloc[0])
        return session, True

    # No cohort label found. Use the first session, and flag that we guessed.
    return int(sessions["session"].iloc[0]), False


def delta_pps_for_one_session(trials):
    """Compute Delta_PPS for one person's one session. NaN if it cannot be fitted."""

    facilitation = compute_facilitation(trials)
    boundary_table = sigmoid_boundary_by_subject(facilitation, split_speed=True)

    if len(boundary_table) == 0:
        return np.nan

    return float(boundary_table["delta_pps"].iloc[0])


def run(t, plot=True):
    """Run H5."""

    if not t.has_patient:
        print("H5 skipped: there is no patient in this dataset.")
        return {"skipped": True}

    results = {"skipped": False}

    patient_trials_all = t.pps_trials[t.pps_trials["subject"] == t.patient_id].copy()

    # ------------------------------------------------------------------
    # Step 1. Choose the sessions.
    # ------------------------------------------------------------------

    patient_session, cohort_was_labelled = pick_high_dopamine_session(patient_trials_all)
    control_session = t.control_session   # prereg convention: 1

    if not cohort_was_labelled:
        print("WARNING: no session is labelled with a 'high' cohort.")
        print(f"  Falling back to session {patient_session}, which may NOT be the")
        print()

    print(f"Patient session (high dopamine): {patient_session}")
    print(f"Control session:                 {control_session}")
    print()

    # These two names are what the notebook copies into t.patient_session and
    # t.control_session, which H7 and H8b then read.
    results["PATIENT_SESSION_USED"] = patient_session
    results["CONTROL_SESSION_USED"] = control_session

    patient_trials = t.pps_trials[
        (t.pps_trials["subject"] == t.patient_id)
        & (t.pps_trials["session"] == patient_session)
    ].copy()

    control_trials = t.pps_trials[
        (t.pps_trials["subject"] == t.control_id)
        & (t.pps_trials["session"] == control_session)
    ].copy()

    results["patient_trials"] = patient_trials
    results["control_trials"] = control_trials

    # ------------------------------------------------------------------
    # Step 2. The observed Delta_PPS for each person.
    # ------------------------------------------------------------------

    patient_delta_pps = delta_pps_for_one_session(patient_trials)
    control_delta_pps = delta_pps_for_one_session(control_trials)

    d_pps_observed = patient_delta_pps - control_delta_pps

    results["patient_delta_pps"] = patient_delta_pps
    results["control_delta_pps"] = control_delta_pps
    results["d_pps_observed"] = d_pps_observed

    print(f"Patient Delta_PPS = {patient_delta_pps:.3f}")
    print(f"Control Delta_PPS = {control_delta_pps:.3f}")
    print(f"Observed D_PPS    = {d_pps_observed:.3f}   (patient minus control)")
    print()

    # ------------------------------------------------------------------
    # Step 3. TEST A: does the patient recalibrate at all?
    # ------------------------------------------------------------------

    print("--- Test A: does the patient show any Delta_PPS? ---")

    test_a = permute_speed_labels_delta_pps(
        patient_trials,
        n_perm=config.N_BOOT,
        seed=config.RANDOM_SEED,
    )

    test_a_supported = (test_a["observed"] > 0) and (test_a["p"] < config.ALPHA)

    print(f"  observed Delta_PPS = {test_a['observed']:.3f}")
    print(f"  permutation p      = {test_a['p']:.4f}")
    print(f"  Test A supported   = {test_a_supported}")
    print()

    results["test_a"] = test_a
    results["test_a_supported"] = test_a_supported

    # ------------------------------------------------------------------
    # Step 4. TEST B, but only if Test A passed.
    # ------------------------------------------------------------------

    if not test_a_supported:
        print("Test A did not pass, so Test B is NOT run.")
        print("  Without a recalibration signal in the patient, comparing their")
        print("  Delta_PPS to the control's would just be comparing noise.")

        results["h5_classification"] = "Test A failed; Test B not run"
        results["h5_pair_boot"] = pd.DataFrame()

    else:
        print("--- Test B: does the patient differ from the matched control? ---")

        # Bootstrap each person separately. Different seeds so the two are not
        # accidentally resampled in lockstep.
        patient_boot = bootstrap_pps_delta_one_session(
            patient_trials,
            n_boot=config.N_BOOT,
            seed=config.RANDOM_SEED,
        )
        control_boot = bootstrap_pps_delta_one_session(
            control_trials,
            n_boot=config.N_BOOT,
            seed=config.RANDOM_SEED + 1,
        )

        # Merge on the bootstrap index, so resample #37 of the patient is paired
        # with resample #37 of the control. H8b later reuses this same pairing.
        paired = patient_boot.merge(
            control_boot,
            on="boot",
            suffixes=("_patient", "_control"),
        )

        paired["d_pps"] = paired["delta_pps_patient"] - paired["delta_pps_control"]

        proportion_negative = float((paired["d_pps"] < 0).mean())
        proportion_positive = float((paired["d_pps"] > 0).mean())

        if proportion_negative >= config.BOOT_SIGN_THRESHOLD:
            classification = "supported: under-calibration"
        elif proportion_positive >= config.BOOT_SIGN_THRESHOLD:
            classification = "supported: over-calibration"
        else:
            classification = "inconclusive"

        ci_low, ci_high = np.percentile(paired["d_pps"].dropna(), [2.5, 97.5])

        print(f"  D_PPS 95% CI     = [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"  P(D_PPS < 0)     = {proportion_negative:.3f}")
        print(f"  P(D_PPS > 0)     = {proportion_positive:.3f}")
        print(f"  classification   = {classification}")

        results["h5_pair_boot"] = paired
        results["h5_classification"] = classification
        results["proportion_negative"] = proportion_negative
        results["proportion_positive"] = proportion_positive
        results["ci_low"] = ci_low
        results["ci_high"] = ci_high

    if plot:
        fig = make_figure(results)
        figures.save(fig, "h5_patient_recalibration")
        plt.show()
        results["figure"] = fig

    return results


def make_figure(results):
    """Build the H5 figure. Test A null on the left, Test B bootstrap on the right."""

    fig, axes = figures.new_figure(n_panels=2, height=2.8)

    # --- Panel A: Test A null distribution -----------------------------------
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
        label=f"observed = {test_a['observed']:.3f}",
    )
    ax.axvline(0, linestyle="--", linewidth=0.8, color="gray")

    ax.set_xlabel(r"$\Delta_{PPS}$ under shuffled speed labels")
    ax.set_ylabel("Count")
    ax.set_title(f"Test A: p = {test_a['p']:.4f}")
    ax.legend()

    style.clean(ax)

    # --- Panel B: Test B bootstrap -------------------------------------------
    ax = axes[1]

    paired = results["h5_pair_boot"]

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
            paired["d_pps"].dropna(),
            bins=40,
            color=style.PATIENT,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.3,
        )

        ax.axvline(
            results["d_pps_observed"],
            color=style.INK,
            linewidth=1.8,
            label=f"observed = {results['d_pps_observed']:.3f}",
        )
        ax.axvline(0, linestyle="--", linewidth=0.8, color="gray")

        ax.set_xlabel(r"$D_{PPS}$ = patient $-$ control")
        ax.set_ylabel("Count")
        ax.set_title(results["h5_classification"])
        ax.legend()

        style.clean(ax)

    figures.label_panels(axes)
    fig.tight_layout()

    return fig
