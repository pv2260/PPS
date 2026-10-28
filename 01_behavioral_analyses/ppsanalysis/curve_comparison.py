"""Do the two psychometric functions differ, within a participant?

The group test on \\Collshift{} asks whether the 50% crossing points move
across participants. It condenses each person's two curves into a single
number and never uses the trials directly. This asks the question the
other way round: given all of a participant's trials, does a model in
which speed moves the curve fit better than one in which it does not?

Three nested models, compared by likelihood ratio:

    reduced    P(hit) depends on offset only
    shift      offset plus a speed term, which displaces the curve
    full       offset, speed, and their interaction, which lets the two
               speeds differ in slope as well as position

Comparing reduced against shift tests whether the boundary moves.
Comparing shift against full tests whether the curves differ in
steepness, which \\(\\Delta \\log|k|\\) measures but the reported analysis
never tests.

The two approaches can disagree, and the disagreement is informative
rather than a problem. A participant with a large \\Collshift{} but noisy
responses may fail here, because the summary difference does not account
for how well either curve was determined.
"""

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm


BALL_RADIUS_CM = 8.75


def _prepare(trials, subject=None, ball_radius_cm=0.0):
    """Usable trials for one participant, with the predictors built."""
    d = trials.copy()

    if subject is not None:
        d = d[d["subject"] == subject]

    if "usable" in d.columns:
        d = d[d["usable"].fillna(False).astype(bool)]

    d = d.dropna(subset=["offset_from_edge_cm", "response_hit", "speed"])

    if ball_radius_cm:
        d = d.assign(
            offset_from_edge_cm=d["offset_from_edge_cm"] - ball_radius_cm
        )

    return d


def compare_curves(trials, subject=None, ball_radius_cm=0.0):
    """Likelihood-ratio tests on one participant's two curves.

    Returns the two comparisons and the coefficients of the full model,
    so a significant interaction can be read alongside the slopes it
    concerns rather than as a bare p-value.
    """
    d = _prepare(trials, subject, ball_radius_cm)

    if len(d) < 20 or d["speed"].nunique() < 2:
        return None

    y = d["response_hit"].astype(int).to_numpy()
    offset = d["offset_from_edge_cm"].to_numpy(dtype=float)

    # Contrast-coded, so the offset coefficient is the slope averaged
    # over speeds rather than the slope in whichever condition happened
    # to be coded zero.
    speed = d["speed"].map({"slow": -1.0, "fast": 1.0}).to_numpy(dtype=float)

    designs = {
        "reduced": np.column_stack([offset]),
        "shift": np.column_stack([offset, speed]),
        "full": np.column_stack([offset, speed, offset * speed]),
    }

    fits = {}

    for name, design in designs.items():
        try:
            fits[name] = sm.Logit(y, sm.add_constant(design)).fit(disp=0)
        except Exception:
            return None

    def ratio(smaller, larger, df):
        statistic = 2.0 * (fits[larger].llf - fits[smaller].llf)
        return statistic, float(stats.chi2.sf(statistic, df=df))

    shift_stat, shift_p = ratio("reduced", "shift", 1)
    slope_stat, slope_p = ratio("shift", "full", 1)

    coefficients = fits["full"].params

    return {
        "subject": subject if subject is not None else "pooled",
        "n_trials": int(len(d)),
        "shift_lr": shift_stat,
        "shift_p": shift_p,
        "slope_lr": slope_stat,
        "slope_p": slope_p,
        "b_offset": float(coefficients[1]),
        "b_speed": float(coefficients[2]),
        "b_interaction": float(coefficients[3]),
    }


def compare_all(trials, subjects=None, exclude=None, ball_radius_cm=0.0,
                verbose=True):
    """The same comparison for every participant in the frame."""
    people = (
        sorted(trials["subject"].dropna().unique())
        if subjects is None else list(subjects)
    )

    if exclude:
        people = [s for s in people if s not in set(exclude)]

    rows = []

    for subject in people:
        result = compare_curves(trials, subject, ball_radius_cm)

        if result is not None:
            rows.append(result)

    table = pd.DataFrame(rows)

    if verbose and len(table):
        print(
            table[[
                "subject", "n_trials", "shift_lr", "shift_p",
                "slope_lr", "slope_p",
            ]].round(4).to_string(index=False)
        )
        print()

        moved = int((table["shift_p"] < 0.05).sum())
        steeper = int((table["slope_p"] < 0.05).sum())

        print(f"  boundary moved with speed in {moved} of {len(table)}")
        print(f"  slope differed with speed in {steeper} of {len(table)}")
        print()
        print(
            "  These are per-participant tests on all of that person's\n"
            "  trials, not a group test. A participant can show a large\n"
            "  boundary shift and still fail here if neither curve was\n"
            "  well determined."
        )

    return table
