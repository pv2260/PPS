"""
H6b. Does dopamine change the patient's PPS recalibration? (EXPLORATORY)

WHAT WE ARE TESTING
-------------------
The patient does the task twice: once ON their dopamine medication (HIGH) and
once OFF it (LOW). If dopamine matters for recalibrating PPS to approach speed,
their Delta_PPS should differ between the two sessions.

    D_dopamine = Delta_PPS(HIGH) - Delta_PPS(LOW)

WHY THIS NEEDS A NOISE FLOOR
----------------------------
Here is the trap. If you bootstrap D_dopamine and it comes out positive in 96%
of resamples, you might want to say "dopamine enhances recalibration". But the
bootstrap only tells you about SAMPLING noise within a session. It says nothing
about whether the measure is stable from one session to the next in the first
place.

If Delta_PPS is so unreliable that the same person gets wildly different values
on two ordinary days, then a HIGH-vs-LOW difference means nothing, no matter how
consistent the bootstrap looks.

That is what the SDC from H4 is for. The SDC is the smallest change that is
BIGGER than ordinary test-retest noise. So we require BOTH:

    the bootstrap is consistent in sign (>= 95% of resamples), AND
    the observed |D_dopamine| is at least as big as the SDC

Only then do we call it a real effect.

THIS IS EXPLORATORY. It is not a confirmatory test in the preregistration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style

from .. import config
from .. import figures
from ..pps import bootstrap_pps_delta_one_session
from .h6 import delta_pps_for_one_session


def pick_session_by_cohort(patient_trials, cohort_substring):
    """Find the patient's session whose cohort label contains e.g. 'high' or 'low'.

    Returns None if there is no such session. We do NOT silently fall back to
    another session here: for H6b, using the wrong session would invert the whole
    result.
    """

    sessions = (
        patient_trials[["session", "cohort"]]
        .dropna(subset=["session"])
        .drop_duplicates()
    )

    if len(sessions) == 0:
        return None

    matches = sessions["cohort"].str.lower().str.contains(cohort_substring, na=False)

    if not matches.any():
        return None

    return int(sessions.loc[matches, "session"].iloc[0])


def run(t, plot=True):
    """Run H6b."""

    if not t.has_patient:
        print("H6b skipped: there is no patient in this dataset.")
        return {"skipped": True}

    results = {"skipped": False}

    patient_trials_all = t.pps_trials[t.pps_trials["subject"] == t.patient_id].copy()

    # ------------------------------------------------------------------
    # Step 1. Find the HIGH and LOW dopamine sessions.
    # ------------------------------------------------------------------

    high_session = pick_session_by_cohort(patient_trials_all, "high")
    low_session = pick_session_by_cohort(patient_trials_all, "low")

    print(f"Patient HIGH-dopamine session: {high_session}")
    print(f"Patient LOW-dopamine session:  {low_session}")
    print()

    results["high_session"] = high_session
    results["low_session"] = low_session

    if high_session is None or low_session is None:
        print("H6b cannot run. We need BOTH a HIGH and a LOW session, labelled in")
        print("the 'cohort' column. Check SUBJECT_META in config.py.")

        results["h6b_classification"] = "not estimable: missing HIGH or LOW session"
        return results

    high_trials = t.pps_trials[
        (t.pps_trials["subject"] == t.patient_id)
        & (t.pps_trials["session"] == high_session)
    ].copy()

    low_trials = t.pps_trials[
        (t.pps_trials["subject"] == t.patient_id)
        & (t.pps_trials["session"] == low_session)
    ].copy()

    # ------------------------------------------------------------------
    # Step 2. The observed Delta_PPS in each session.
    # ------------------------------------------------------------------

    delta_high = delta_pps_for_one_session(high_trials)
    delta_low = delta_pps_for_one_session(low_trials)

    d_dopamine_observed = delta_high - delta_low

    results["delta_high"] = delta_high
    results["delta_low"] = delta_low
    results["d_dopamine_observed"] = d_dopamine_observed

    print(f"Delta_PPS HIGH = {delta_high:.3f}")
    print(f"Delta_PPS LOW  = {delta_low:.3f}")
    print(f"D_dopamine     = {d_dopamine_observed:.3f}   (HIGH minus LOW)")
    print()

    # ------------------------------------------------------------------
    # Step 3. Bootstrap D_dopamine.
    # ------------------------------------------------------------------

    boot_high = bootstrap_pps_delta_one_session(
        high_trials,
        n_boot=config.N_BOOT,
        seed=config.RANDOM_SEED,
    )
    boot_low = bootstrap_pps_delta_one_session(
        low_trials,
        n_boot=config.N_BOOT,
        seed=config.RANDOM_SEED + 2,
    )

    paired = boot_high.merge(boot_low, on="boot", suffixes=("_high", "_low"))
    paired["d_dopamine"] = paired["delta_pps_high"] - paired["delta_pps_low"]

    proportion_positive = float((paired["d_dopamine"] > 0).mean())
    proportion_negative = float((paired["d_dopamine"] < 0).mean())

    ci_low, ci_high = np.percentile(paired["d_dopamine"].dropna(), [2.5, 97.5])

    results["paired_boot"] = paired
    results["proportion_positive"] = proportion_positive
    results["proportion_negative"] = proportion_negative
    results["ci_low"] = ci_low
    results["ci_high"] = ci_high

    # ------------------------------------------------------------------
    # Step 4. Check the observed effect against the noise floor from H4.
    # ------------------------------------------------------------------

    sdc = t.sdc_delta_pps

    if np.isfinite(sdc):
        bigger_than_noise = abs(d_dopamine_observed) >= sdc
    else:
        # No SDC available. We cannot rule out that this is just test-retest
        # noise, so we refuse to claim an effect. Being unable to check the noise
        # floor is NOT the same as passing it.
        bigger_than_noise = False

    results["sdc"] = sdc
    results["bigger_than_noise"] = bigger_than_noise

    # ------------------------------------------------------------------
    # Step 5. Classify. BOTH conditions must hold.
    # ------------------------------------------------------------------

    consistent_positive = proportion_positive >= config.BOOT_SIGN_THRESHOLD
    consistent_negative = proportion_negative >= config.BOOT_SIGN_THRESHOLD

    if consistent_positive and bigger_than_noise:
        classification = "supported: dopamine-enhanced recalibration"
    elif consistent_negative and bigger_than_noise:
        classification = "supported: dopamine-reduced recalibration"
    else:
        classification = "no reliable modulation"

    results["h6b_classification"] = classification

    report(results)

    if plot:
        fig = make_figure(results)
        figures.save(fig, "h6b_dopamine")
        plt.show()
        results["figure"] = fig

    return results


def report(results):
    """Print the H6b numbers."""

    print(f"bootstrap 95% CI of D_dopamine = "
          f"[{results['ci_low']:.3f}, {results['ci_high']:.3f}]")
    print(f"P(D_dopamine > 0) = {results['proportion_positive']:.3f}")
    print(f"P(D_dopamine < 0) = {results['proportion_negative']:.3f}")
    print()

    sdc = results["sdc"]
    observed = abs(results["d_dopamine_observed"])

    if np.isfinite(sdc):
        print(f"noise floor check: |D_dopamine| = {observed:.3f}, SDC = {sdc:.3f}")
        if results["bigger_than_noise"]:
            print("  the effect is BIGGER than test-retest noise")
        else:
            print("  the effect is SMALLER than test-retest noise, so it could just be")
            print("  the task being unreliable rather than dopamine doing anything")
    else:
        print("noise floor check: SDC is not estimable (H4 could not compute it).")
        print("  Without a noise floor we cannot rule out test-retest noise, so we do")
        print("  not claim an effect.")

    print()
    print(f"H6b classification: {results['h6b_classification']}")


def make_figure(results):
    """Build the H6b figure. One panel."""

    paired = results["paired_boot"]
    sdc = results["sdc"]

    fig, axes = figures.new_figure(
        n_panels=1,
        width=figures.ONE_HALF_COLUMN,
        height=2.8,
    )
    ax = axes[0]

    ax.hist(
        paired["d_dopamine"].dropna(),
        bins=40,
        color=style.PATIENT,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.3,
    )

    ax.axvline(
        results["d_dopamine_observed"],
        color=style.INK,
        linewidth=1.8,
        label=f"observed = {results['d_dopamine_observed']:.3f}",
    )
    ax.axvline(0, linestyle="--", linewidth=0.8, color="gray")

    # The noise floor from H4, drawn on both sides of zero. An effect that falls
    # BETWEEN these two lines is smaller than ordinary test-retest variation and
    # should not be believed, however tight the bootstrap looks.
    if np.isfinite(sdc):
        ax.axvline(sdc, linestyle=":", linewidth=1.2, color=style.FAST,
                   label=f"noise floor $\\pm$ {sdc:.3f}")
        ax.axvline(-sdc, linestyle=":", linewidth=1.2, color=style.FAST)

    ax.set_xlabel(r"$D_{dopamine} = \Delta_{PPS}(HIGH) - \Delta_{PPS}(LOW)$")
    ax.set_ylabel("Bootstrap count")
    ax.set_title(results["h6b_classification"])
    ax.legend()

    style.clean(ax)
    fig.tight_layout()

    return fig
