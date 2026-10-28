"""Bootstrap facilitation and the near-far index, over trials.

``h1.group_test`` bootstraps the group mean by resampling participants.
That answers whether the effect is reliable across people. It says
nothing about how precisely any one participant's value is estimated,
which with six or seven trials per cell is the more pressing question.

This resamples trials within each participant instead. Each draw
rebuilds the whole profile from resampled trials and recomputes both
quantities, so the interval carries the sampling noise of every cell that
went into them.

Two quantities, and they answer different questions.

    Overall facilitation is the mean across positions: how much the
    looming stimulus sped up tactile responses, without regard to
    distance.

    The near-far index is near minus far: whether that facilitation
    depended on distance. This is what H1 tests. A participant can show
    large facilitation at every distance and a near-far index of zero.

Resampling is stratified by speed, position and sensory condition, so
every draw keeps the same design as the session: the same number of
tactile and visuo-tactile trials at each distance and speed. Resampling
the session as a whole would let a draw come out short of tactile trials
at one position, and the baseline there would then be estimated from
almost nothing.

P004 and P006 are excluded by default. Both had controller faults during
testing, and their reaction times are correspondingly noisy.
"""

import numpy as np
import pandas as pd

from ppsanalysis.hypotheses import h1
from ppsanalysis import config
from ppsanalysis.stats_utils import bootstrap_mean_ci, wilcoxon_one_sided


# Controller faults observed during testing.
EXCLUDE = ["P004", "P006", "T002", "T001"]

STRATA = ["subject", "session", "speed", "position_rank",
          "sensory_condition"]


def _indices(profile):
    """Overall facilitation and the near-far index, per subject and speed."""
    by_position = h1.subject_profile(profile, by=["subject", "speed"])

    rows = []

    for (subject, speed), group in by_position.groupby(["subject", "speed"]):
        near = group.loc[
            group["position_rank"].isin(h1.NEAR), "facilitation_ms"
        ].mean()
        far = group.loc[
            group["position_rank"].isin(h1.FAR), "facilitation_ms"
        ].mean()

        rows.append({
            "subject": subject,
            "speed": speed,
            "overall_ms": group["facilitation_ms"].mean(),
            "near_ms": near,
            "far_ms": far,
            "nearfar_ms": near - far,
        })

    return pd.DataFrame(rows)


def bootstrap_indices(trials, subjects=None, exclude=EXCLUDE,
                      correction="expectancy_model", expectancy_degree=1,
                      n_boot=None, seed=None, verbose=True):
    """Percentile intervals on facilitation and the near-far index.

    Returns the observed values, the intervals, and the proportion of
    draws above zero. That proportion is the useful number for a single
    participant: an interval either includes zero or does not, but the
    proportion says how nearly.

    ``failed`` counts draws where the profile could not be built, usually
    because a resample left a cell empty. A count above a few percent
    means the estimate is fragile in a way the interval alone does not
    convey.
    """
    n_boot = config.N_BOOT if n_boot is None else n_boot
    seed = config.RANDOM_SEED if seed is None else seed

    d = trials[trials["usable"]] if "usable" in trials.columns else trials
    d = d[d["sensory_condition"].isin(["T", "VT"])].copy()

    if subjects is not None:
        d = d[d["subject"].isin(subjects)]

    if exclude:
        dropped = sorted(set(exclude) & set(d["subject"]))
        d = d[~d["subject"].isin(exclude)]

        if verbose and dropped:
            print(f"Excluded: {', '.join(dropped)}")

    keys = [k for k in STRATA if k in d.columns]

    observed = _indices(
        h1.facilitation_profile(
            d, estimator="median", correction=correction,
            expectancy_degree=expectancy_degree,
        )
    )
    strata_index = [np.asarray(v) for v in d.groupby(keys, observed=True).indices.values()]
    rng = np.random.default_rng(seed)
    draws = []

    for _ in range(n_boot):
        take = np.concatenate([
            rng.choice(idx, size=len(idx), replace=True)
            for idx in strata_index
        ])
        resampled = d.iloc[take]

        try:
            profile = h1.facilitation_profile(
                resampled, estimator="median", correction=correction,
                expectancy_degree=expectancy_degree,
            )
        except Exception:
            continue

        if profile.empty:
            continue

        draws.append(_indices(profile))

    if not draws:
        raise RuntimeError("Every bootstrap draw failed.")

    boot = pd.concat(draws, ignore_index=True)

    summary = (
        boot.groupby(["subject", "speed"])[["overall_ms", "nearfar_ms"]]
        .agg([
            ("low", lambda s: s.quantile(0.025)),
            ("high", lambda s: s.quantile(0.975)),
            ("above_zero", lambda s: float((s > 0).mean())),
            ("n_draws", "size"),
        ])
    )

    summary.columns = ["_".join(c) for c in summary.columns]

    out = observed.set_index(["subject", "speed"]).join(summary)
    out["failed"] = n_boot - out["overall_ms_n_draws"]

    if verbose:
        print(f"\n{n_boot} draws, resampled within "
              f"{', '.join(keys[1:])}")
        print()
        print(
            out[[
                "overall_ms", "overall_ms_low", "overall_ms_high",
                "overall_ms_above_zero",
                "nearfar_ms", "nearfar_ms_low", "nearfar_ms_high",
                "nearfar_ms_above_zero", "failed",
            ]].round(3).to_string()
        )
        print()
        print(
            "  above_zero is the proportion of draws above zero, which is "
            "the informative number for one participant: an interval "
            "either excludes zero or does not, but this says how nearly."
        )

    return out, boot


def group_test(observed, value="nearfar_ms", verbose=True):
    """The preregistered group test, on whichever index is passed.

    Collapses speed within each participant first, then bootstraps the
    mean across participants and runs the one-sided Wilcoxon. This is
    ``h1.group_test`` with the index selectable, so the same test can be
    applied to overall facilitation as well as to the near-far index.
    """
    per_subject = (
        observed.reset_index().groupby("subject")[value].mean().dropna()
    )
    values = per_subject.to_numpy()

    boot = bootstrap_mean_ci(values, n_boot=config.N_BOOT, ci=95,
                             seed=config.RANDOM_SEED)
    wilcoxon = wilcoxon_one_sided(values, alternative="greater")

    result = {
        "index": value,
        "n": len(values),
        "mean": float(values.mean()),
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "wilcoxon_W": wilcoxon["W"],
        "wilcoxon_p": wilcoxon["p"],
        "supported": bool(boot["ci_low"] > 0),
    }

    if verbose:
        print(f"\nGroup test on {value}, speeds averaged within participant")
        print()
        print(per_subject.round(2).to_string())
        print()
        print(f"  n                = {result['n']}")
        print(f"  mean             = {result['mean']:.2f} ms")
        print(f"  bootstrap 95% CI = [{result['ci_low']:.2f}, "
              f"{result['ci_high']:.2f}]")
        print(f"  Wilcoxon         = W {result['wilcoxon_W']}, "
              f"p {result['wilcoxon_p']:.4f}")
        print()
        print("  SUPPORTED" if result["supported"] else "  not supported")

    return result


def run(trials, subjects=None, exclude=EXCLUDE,
        correction="fastest_tactile", expectancy_degree=1, n_boot=None):
    """Trial-level intervals, then the group test on both indices."""
    observed, boot = bootstrap_indices(
        trials, subjects=subjects, exclude=exclude,
        correction=correction, expectancy_degree=expectancy_degree,
        n_boot=n_boot,
    )

    overall = group_test(observed, value="overall_ms")
    nearfar = group_test(observed, value="nearfar_ms")

    return {
        "per_subject": observed,
        "draws": boot,
        "group_overall": overall,
        "group_nearfar": nearfar,
    }
