"""
H5. Does the patient show distance-dependent PPS facilitation at all?

WHAT WE ARE TESTING
-------------------
This is the single-case version of H1. We have ONE patient, so we cannot do a
group test. Instead we ask: is this patient's near-far index bigger than we would
expect if distance did not matter at all?

HOW THE PERMUTATION TEST WORKS
------------------------------
The logic is simple, and it is worth being clear about because permutation tests
are often described badly.

  1. We compute the patient's real NearFar_PPS. Call it the OBSERVED value.

  2. We then destroy the link between RT and distance, by SHUFFLING the distance
     labels across the patient's trials. A trial that really happened at D1 might
     now be labelled D6. Everything else about the data is untouched.

  3. We recompute NearFar_PPS on the shuffled data. Because the distance labels
     are now meaningless, any near-far difference we get is pure chance.

  4. We repeat steps 2 and 3 five thousand times. That gives us a whole
     distribution of "near-far differences you get by chance". This is the NULL.

  5. The p-value is: what fraction of the null values are at least as extreme as
     the observed one?

If the observed value sits way out in the tail of the null, then distance really
does matter for this patient.

SUPPORTED IF
------------
In EVERY available patient session: NearFar_PPS > 0 AND permutation p < 0.05.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style

from .. import config
from .. import figures
from ..permutations import permute_distance_labels_nearfar


def run(t, plot=True):
    """Run H5. Returns {"skipped": True} if there is no patient in the data."""

    # ------------------------------------------------------------------
    # Step 0. Do we even have a patient?
    # ------------------------------------------------------------------

    if not t.has_patient:
        print("H5 skipped: there is no patient in this dataset.")
        print("  This is expected while you are only running healthy pilots.")
        return {"skipped": True}

    results = {"skipped": False}

    print(f"Patient:         {t.patient_id}")
    print(f"Matched control: {t.control_id}")
    print()

    # ------------------------------------------------------------------
    # Step 1. Get the patient's PPS trials, session by session.
    # ------------------------------------------------------------------

    patient_trials = t.pps_trials[t.pps_trials["subject"] == t.patient_id].copy()

    sessions = sorted(patient_trials["session"].dropna().unique())

    print(f"Patient PPS sessions available: {sessions}")
    print()

    results["patient_trials"] = patient_trials
    results["sessions"] = sessions

    # ------------------------------------------------------------------
    # Step 2. Run the permutation test in each session.
    # ------------------------------------------------------------------

    rows = []
    permutation_results = {}

    for session in sessions:

        one_session = patient_trials[patient_trials["session"] == session].copy()

        # The seed is offset by the session number so that session 1 and session 2
        # do not get the exact same 5000 shuffles. Same seed = same shuffles =
        # correlated results, which would be wrong.
        permutation = permute_distance_labels_nearfar(
            one_session,
            n_perm=config.N_BOOT,
            seed=config.RANDOM_SEED + int(session),
        )

        permutation_results[session] = permutation

        observed = permutation["observed"]
        p_value = permutation["p"]

        session_supported = (observed > 0) and (p_value < config.ALPHA)

        rows.append({
            "session": session,
            "n_usable_trials": int(one_session["usable"].sum()),
            "observed_nearfar_ms": observed,
            "permutation_p": p_value,
            "n_permutations": permutation["n_perm"],
            "supported": session_supported,
        })

    summary = pd.DataFrame(rows)

    results["summary"] = summary
    results["permutation_results"] = permutation_results

    # ------------------------------------------------------------------
    # Step 3. The verdict. Every session must pass.
    # ------------------------------------------------------------------

    if len(summary) > 0:
        supported_overall = bool(summary["supported"].all())
    else:
        supported_overall = False

    # This name is what the notebook's summary table reads.
    results["h5_supported_overall"] = supported_overall

    report(results)

    if plot:
        fig = make_figure(results)
        figures.save(fig, "h5_patient_nearfar")
        plt.show()
        results["figure"] = fig

    return results


def report(results):
    """Print the H5 numbers."""

    print("H5: near-far index in the patient, tested against a shuffled-distance null")
    print()
    print(results["summary"].round(4).to_string(index=False))
    print()

    if results["h5_supported_overall"]:
        print("  H5 IS SUPPORTED: in every session the patient's facilitation was")
        print("  stronger near the body than chance would produce.")
    else:
        print("  H5 is NOT supported in every session.")


def make_figure(results):
    """Build the H5 figure. One null-distribution histogram per session."""

    sessions = results["sessions"]
    permutation_results = results["permutation_results"]

    n_panels = max(1, len(sessions))

    fig, axes = figures.new_figure(
        n_panels=n_panels,
        width=figures.DOUBLE_COLUMN if n_panels > 1 else figures.ONE_HALF_COLUMN,
        height=2.6,
    )

    for ax, session in zip(axes, sessions):

        permutation = permutation_results[session]

        # The grey histogram is the null: every near-far value we got by shuffling
        # the distance labels. If distance did not matter, the real value would
        # land somewhere in here.
        ax.hist(
            permutation["permuted"],
            bins=40,
            color=style.CONTROL,
            edgecolor="white",
            linewidth=0.3,
        )

        # The patient's real value.
        ax.axvline(
            permutation["observed"],
            color=style.PATIENT,
            linewidth=1.8,
            label=f"observed = {permutation['observed']:.1f} ms",
        )

        ax.axvline(0, linestyle="--", linewidth=0.8, color="gray")

        ax.set_xlabel("NearFar$_{PPS}$ under shuffled\ndistance labels (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"Session {session}: p = {permutation['p']:.4f}")
        ax.legend()

        style.clean(ax)

    if len(sessions) > 1:
        figures.label_panels(axes)

    fig.tight_layout()

    return fig
