"""Do the PPS results depend on which baseline correction was used?

The question a reviewer will ask is whether the expectancy model was
chosen because the preregistered alternatives did not give the answer we
wanted. The only honest response is to run every correction and report
what each returns, including the ones that disagree.

Four corrections are available:

    delay_matched      a separate median per distance and speed, which
                       the preregistration specifies and the trial counts
                       could not support
    expectancy_model   the participant-specific regression used in the
                       reported analyses
    fastest_tactile    the single fastest tactile-only condition,
                       subtracted as a constant
    fastest_n_tactile  the median of the fastest few tactile trials, as
                       a constant

The last two are the family \\textcite{noel2018} used: they collected
tactile-only trials at two delays and subtracted the faster of the two
from every experimental trial.

What to report, whichever way it comes out
------------------------------------------
The informative comparison is not whether each correction reaches
significance but whether the direction of the effect changes. A result
that holds in direction across all four and differs only in precision is
robust to the choice; one that reverses is not, and that is worth
knowing before a reviewer finds it.
"""

import numpy as np
import pandas as pd

from ppsanalysis.hypotheses import h1, h1a, h2
from ppsanalysis import config
from ppsanalysis.stats_utils import bootstrap_mean_ci, wilcoxon_one_sided


CORRECTIONS = [
    "delay_matched",
    "expectancy_model",
    "fastest_tactile",
    "fastest_n_tactile",
]


def _group_test(values):
    """Bootstrap mean, CI and one-sided Wilcoxon, as the hypotheses use."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 3:
        return {"n": len(values), "mean": np.nan, "ci_low": np.nan,
                "ci_high": np.nan, "W": np.nan, "p": np.nan}

    boot = bootstrap_mean_ci(values, n_boot=config.N_BOOT, ci=95,
                             seed=config.RANDOM_SEED)
    test = wilcoxon_one_sided(values, alternative="greater")

    return {"n": len(values), "mean": boot["mean"],
            "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
            "W": test["W"], "p": test["p"]}


def facilitation_under_each(trials, corrections=CORRECTIONS,
                            expectancy_degree=1, verbose=True):
    """H1 under every correction: overall facilitation and near--far.

    Both indices are recomputed from scratch for each correction, so the
    comparison is like for like rather than a rescaling of one result.
    """
    rows = []

    for correction in corrections:
        try:
            profile = h1.facilitation_profile(
                trials, estimator="median", correction=correction,
                expectancy_degree=expectancy_degree,
            )
        except Exception as error:
            rows.append({"correction": correction, "index": "failed",
                         "note": str(error)[:60]})
            continue

        averaged = h1.subject_profile(profile)

        overall, nearfar = [], []

        for subject, group in averaged.groupby("subject"):
            y = group.sort_values("position_rank")["facilitation_ms"]
            overall.append(y.mean())

            near = group.loc[group["position_rank"].isin(h1.NEAR),
                             "facilitation_ms"].mean()
            far = group.loc[group["position_rank"].isin(h1.FAR),
                            "facilitation_ms"].mean()
            nearfar.append(near - far)

        for label, values in [("overall", overall), ("near-far", nearfar)]:
            rows.append({"correction": correction, "index": label,
                         **_group_test(values)})

    out = pd.DataFrame(rows)

    if verbose:
        print("H1 under each baseline correction")
        print()
        print(out.round(3).to_string(index=False))
        print()
        print(
            "  The direction is what matters. A mean that stays positive "
            "under every correction is robust to the choice; one that "
            "changes sign is not."
        )

    return out


def boundaries_under_each(trials, corrections=CORRECTIONS,
                          expectancy_degree=1, verbose=True):
    """H2 under every correction: Delta_PPS per participant and as a group.

    Fits that fail or return a slope in the wrong direction are counted
    rather than dropped silently, since a correction that leaves fewer
    participants estimable is itself a result about that correction.
    """
    per_subject, group = [], []

    for correction in corrections:
        try:
            result = h2.run_by_speed(
                trials, corrections=(correction,),
                expectancy_degree=expectancy_degree,
            )
        except Exception as error:
            group.append({"correction": correction, "n": 0,
                          "note": str(error)[:60]})
            continue

        fits = result["speed_fits"]
        boundaries = result["boundaries"]

        usable = boundaries.dropna(subset=["delta_pps"])

        for _, row in usable.iterrows():
            per_subject.append({
                "correction": correction,
                "subject": row["subject"],
                "delta_pps": row["delta_pps"],
            })

        group.append({
            "correction": correction,
            "fits_attempted": len(fits),
            "fits_sound": int(fits["fit_ok"].sum()) if "fit_ok" in fits
                          else np.nan,
            **_group_test(usable["delta_pps"]),
        })

    per_subject = pd.DataFrame(per_subject)
    group = pd.DataFrame(group)

    if verbose:
        print("\nH2 under each baseline correction")
        print()
        print(group.round(3).to_string(index=False))

        if len(per_subject):
            print("\nDelta_PPS per participant")
            print()
            print(
                per_subject.pivot_table(
                    index="subject", columns="correction",
                    values="delta_pps",
                ).round(3).to_string()
            )
            print()
            print(
                "  A participant absent from a column had no usable "
                "boundary under that correction."
            )

    return group, per_subject


def clinical_under_each(trials, patient="PT001", control="PA002",
                        corrections=CORRECTIONS, expectancy_degree=1,
                        verbose=True):
    """H4 and H5 quantities under every correction, for the pair.

    No permutation is run here: the point is whether the descriptive
    values change sign, which a reviewer can check against the reported
    ones without waiting for 5000 refits.
    """
    rows = []

    for correction in corrections:
        try:
            profile = h1.facilitation_profile(
                trials, estimator="median", correction=correction,
                expectancy_degree=expectancy_degree,
            )
            result = h2.run_by_speed(
                trials, corrections=(correction,),
                expectancy_degree=expectancy_degree,
            )
        except Exception as error:
            rows.append({"correction": correction, "subject": "all",
                         "note": str(error)[:60]})
            continue

        averaged = h1.subject_profile(profile)
        boundaries = result["boundaries"].set_index("subject")

        for subject in [patient, control]:
            group = averaged[averaged["subject"] == subject]

            if group.empty:
                continue

            near = group.loc[group["position_rank"].isin(h1.NEAR),
                             "facilitation_ms"].mean()
            far = group.loc[group["position_rank"].isin(h1.FAR),
                            "facilitation_ms"].mean()

            rows.append({
                "correction": correction,
                "subject": subject,
                "overall_ms": group["facilitation_ms"].mean(),
                "nearfar_ms": near - far,
                "delta_pps": boundaries["delta_pps"].get(subject, np.nan),
            })

    out = pd.DataFrame(rows)

    if verbose and len(out):
        print("\nThe clinical pair under each baseline correction")
        print()
        print(out.round(3).to_string(index=False))

        if {"subject", "delta_pps"} <= set(out.columns):
            wide = out.pivot_table(index="correction", columns="subject",
                                   values="delta_pps")

            if {patient, control} <= set(wide.columns):
                wide["D_PPS"] = wide[patient] - wide[control]
                print("\nD_PPS = patient minus control")
                print()
                print(wide.round(3).to_string())

    return out


def run_all(young_trials, clinical_trials, expectancy_degree=1):
    """Every PPS analysis that depends on the baseline, under every correction."""
    facilitation = facilitation_under_each(
        young_trials, expectancy_degree=expectancy_degree,
    )
    group, per_subject = boundaries_under_each(
        young_trials, expectancy_degree=expectancy_degree,
    )
    clinical = clinical_under_each(
        clinical_trials, expectancy_degree=expectancy_degree,
    )

    print("\n" + "=" * 66)
    print("  What to take from this")
    print("=" * 66)
    print("""
Read the direction, not the p-values. If the sign of every effect is the
same under all four corrections, the choice of baseline changed the
precision of the estimates and not the conclusions, and that is what the
robustness check establishes.

Where a correction leaves fewer participants with an estimable boundary,
report the count. A correction under which half the sample cannot be
fitted is not an alternative that would have given a different answer;
it is one that would have given no answer.
""")

    return {
        "facilitation": facilitation,
        "boundaries_group": group,
        "boundaries_per_subject": per_subject,
        "clinical": clinical,
    }
