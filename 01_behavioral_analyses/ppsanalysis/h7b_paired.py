r"""H7b, with the two tasks resampled together.

The joint sign criterion asks whether the patient deviates from his control in
the same direction on both tasks. That is a statement about two deviations
occurring together, so every draw has to hold a PPS deviation and a collision
deviation computed from the same resampled version of the same two people.

The previous implementation ran the two bootstraps separately and merged them
on the draw index. That looks like pairing but is not: draw 37 of one table
and draw 37 of the other describe different resamples, so merging them yields
the joint distribution the two quantities would have if they were independent.
Any real association between them, in either direction, is destroyed, and the
95\% criterion cannot be met whatever the data say.

Here each draw resamples a participant's PPS trials and collision trials once,
and both indices are computed from that same resample. The pairing is
therefore a property of the draw rather than of a merge afterwards.

Usage
-----
    from ppsanalysis import h7b_paired as paired

    result = paired.run(clin, ball_radius_cm=BALL_RADIUS_CM)
    paired.report(result)
    paired.plot(result, save_as="h7b_paired")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style
from . import figures
from . import config as cfg


SPEEDS = ["slow", "fast"]

NEAR = [1, 2]
FAR = [6, 7]

# Sign combinations the preregistration names, and what each would mean.
CRITERIA = {
    "concordant under-calibration": (-1, -1),
    "concordant over-calibration": (+1, +1),
    "discordant": None,
}


def _resample(index, rng):
    """Trial indices for one bootstrap draw, resampled within speed."""
    return np.concatenate([
        rng.choice(block, size=len(block), replace=True)
        for block in index
    ])


def _pps_boundary_shift(trials, correction, expectancy_degree):
    """Delta_PPS for one participant, or NaN where a boundary is not fitted."""
    from .hypotheses import h1, h2

    profile = h1.facilitation_profile(
        trials, estimator="median", correction=correction,
        expectancy_degree=expectancy_degree,
    )

    boundaries = {}

    for speed in SPEEDS:
        one = (
            profile[profile["speed"] == speed]
            .dropna(subset=["position_rank", "facilitation_ms"])
            .sort_values("position_rank")
        )

        if len(one) < 5:
            return np.nan

        fit = h2.safe_sigmoid_fit(
            one["position_rank"].to_numpy(float),
            one["facilitation_ms"].to_numpy(float),
            k_direction="positive",
        )

        if not fit.get("ok", False):
            return np.nan

        boundaries[speed] = fit["x_c"]

    return boundaries["fast"] - boundaries["slow"]


def _pps_nearfar_shift(trials, correction, expectancy_degree):
    """Fast minus slow near-far index, with no curve fitted.

    The boundary estimator returns nothing when the sigmoid does not converge,
    which under resampling happens in most draws: the midpoint is recovered
    from seven cells and a resample that flattens the profile leaves nothing to
    locate. The near-far index is a difference of four cell values, so it is
    defined whenever those cells hold trials, and every draw yields a value.

    It measures a related but not identical quantity: how much more strongly
    facilitation is graded toward the body under fast approach, rather than
    where the transition sits.
    """
    from .hypotheses import h1

    profile = h1.facilitation_profile(
        trials, estimator="median", correction=correction,
        expectancy_degree=expectancy_degree,
    )

    by_speed = {}

    for speed in SPEEDS:
        one = profile[profile["speed"] == speed]

        near = one.loc[one["position_rank"].isin(NEAR), "facilitation_ms"]
        far = one.loc[one["position_rank"].isin(FAR), "facilitation_ms"]

        if near.dropna().empty or far.dropna().empty:
            return np.nan

        by_speed[speed] = float(near.mean() - far.mean())

    return by_speed["fast"] - by_speed["slow"]

def _collision_shift(trials, ball_radius_cm):
    """Delta_coll and near-boundary accuracy for one participant."""
    import statsmodels.api as sm

    clearance = (
        pd.to_numeric(trials["offset_from_edge_cm"], errors="coerce")
        - ball_radius_cm
    )
    hit = pd.to_numeric(trials["response_hit"], errors="coerce")

    boundaries = {}

    for speed in SPEEDS:
        cell = trials["speed"] == speed
        x = clearance[cell].to_numpy(float)
        y = hit[cell].to_numpy(float)

        ok = np.isfinite(x) & np.isfinite(y)

        if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
            return np.nan, np.nan

        try:
            fit = sm.Logit(
                y[ok], sm.add_constant(x[ok], has_constant="add")
            ).fit(disp=False)
        except Exception:
            return np.nan, np.nan

        intercept, slope = float(fit.params[0]), float(fit.params[1])

        if slope == 0:
            return np.nan, np.nan

        boundaries[speed] = -intercept / slope

    near = trials["trial_type"].isin(["near_hit", "near_miss"])
    accuracy = (
        float(trials.loc[near, "correct"].mean())
        if near.any() and "correct" in trials.columns else np.nan
    )

    return boundaries["fast"] - boundaries["slow"], accuracy


PPS_ESTIMATORS = {
    "boundary": _pps_boundary_shift,
    "nearfar": _pps_nearfar_shift,
}


def run(
    clinical,
    ball_radius_cm=0.0,
    correction="expectancy_model",
    expectancy_degree=1,
    pps_index="boundary",
    n_boot=None,
    seed=None,
):
    """The joint criterion, with both tasks resampled in the same draw.

    Returns the observed deviations, the paired draws, and the proportion of
    draws falling in each sign quadrant.
    """
    n_boot = cfg.N_BOOT if n_boot is None else n_boot
    seed = cfg.RANDOM_SEED if seed is None else seed

    if pps_index not in PPS_ESTIMATORS:
        raise ValueError(
            f"pps_index must be one of {sorted(PPS_ESTIMATORS)}"
        )

    pps_estimator = PPS_ESTIMATORS[pps_index]

    patient, control = clinical.patient_id, clinical.control_id

    pps = clinical.pps_trials
    collision = clinical.collision_trials

    if "usable" in collision.columns:
        collision = collision[collision["usable"]]

    people = {}

    for role, subject in [("patient", patient), ("control", control)]:
        person_pps = pps[pps["subject"] == subject]
        person_collision = collision[collision["subject"] == subject]

        if clinical.patient_session is not None and role == "patient":
            if "session" in person_pps.columns:
                person_pps = person_pps[
                    person_pps["session"] == clinical.patient_session
                ]
                person_collision = person_collision[
                    person_collision["session"] == clinical.patient_session
                ]

        people[role] = {
            "pps": person_pps,
            "collision": person_collision,
            # Blocks the resampling respects, so each draw keeps the design's
            # trial counts per speed and condition.
            "pps_blocks": [
                group.index.to_numpy()
                for _, group in person_pps.groupby(
                    ["speed", "sensory_condition"], dropna=False
                )
            ],
            "collision_blocks": [
                group.index.to_numpy()
                for _, group in person_collision.groupby("speed", dropna=False)
            ],
        }

    # Observed values, no resampling.
    observed = {}

    for role in ["patient", "control"]:
        observed[f"pps_{role}"] = pps_estimator(
            people[role]["pps"], correction, expectancy_degree,
        )
        observed[f"coll_{role}"], observed[f"acc_{role}"] = _collision_shift(
            people[role]["collision"], ball_radius_cm,
        )

    observed["d_pps"] = observed["pps_patient"] - observed["pps_control"]
    observed["d_coll"] = observed["coll_patient"] - observed["coll_control"]
    observed["d_acc"] = observed["acc_patient"] - observed["acc_control"]

    rng = np.random.default_rng(seed)
    rows = []

    for draw in range(int(n_boot)):
        record = {"draw": draw}

        # One resample per participant, used for both tasks. This is the
        # whole point: the PPS and collision values in this row describe the
        # same resampled version of the same person.
        for role in ["patient", "control"]:
            person = people[role]

            pps_draw = person["pps"].loc[
                _resample(person["pps_blocks"], rng)
            ]
            collision_draw = person["collision"].loc[
                _resample(person["collision_blocks"], rng)
            ]

            record[f"pps_{role}"] = pps_estimator(
                pps_draw, correction, expectancy_degree,
            )
            record[f"coll_{role}"], record[f"acc_{role}"] = _collision_shift(
                collision_draw, ball_radius_cm,
            )

        record["d_pps"] = record["pps_patient"] - record["pps_control"]
        record["d_coll"] = record["coll_patient"] - record["coll_control"]
        record["d_acc"] = record["acc_patient"] - record["acc_control"]

        rows.append(record)

    draws = pd.DataFrame(rows)

    return {
        "observed": observed,
        "draws": draws,
        "quadrants": _quadrants(draws),
        "settings": {
            "n_boot": int(n_boot),
            "seed": seed,
            "correction": correction,
            "pps_index": pps_index,
            "patient": patient,
            "control": control,
            "threshold": cfg.BOOT_SIGN_THRESHOLD,
        },
    }


def _quadrants(draws, x="d_pps", y="d_coll"):
    """Proportion of paired draws in each sign combination."""
    usable = draws[[x, y]].dropna()

    if usable.empty:
        return pd.DataFrame()

    rows = [
        {
            "quadrant": "both negative",
            "meaning": "concordant under-calibration",
            "proportion": float(
                ((usable[x] < 0) & (usable[y] < 0)).mean()
            ),
        },
        {
            "quadrant": "both positive",
            "meaning": "concordant over-calibration",
            "proportion": float(
                ((usable[x] > 0) & (usable[y] > 0)).mean()
            ),
        },
        {
            "quadrant": "PPS negative, collision positive",
            "meaning": "discordant",
            "proportion": float(
                ((usable[x] < 0) & (usable[y] > 0)).mean()
            ),
        },
        {
            "quadrant": "PPS positive, collision negative",
            "meaning": "discordant",
            "proportion": float(
                ((usable[x] > 0) & (usable[y] < 0)).mean()
            ),
        },
    ]

    out = pd.DataFrame(rows)
    out["usable_draws"] = len(usable)
    out["failed_draws"] = len(draws) - len(usable)

    return out.round(3)


def report(result):
    """Print the observed deviations, the quadrants, and the verdict."""
    observed = result["observed"]
    settings = result["settings"]

    units = "ranks" if settings["pps_index"] == "boundary" else "ms"

    print(
        f"H7b, paired bootstrap: {settings['patient']} against "
        f"{settings['control']}, {settings['n_boot']} draws, "
        f"PPS index = {settings['pps_index']}"
    )
    print()
    print(
        f"  observed  D_PPS  = {observed['d_pps']:+.2f} {units}"
        f"   ({observed['pps_patient']:+.2f} vs "
        f"{observed['pps_control']:+.2f})"
    )
    print(
        f"            D_coll = {observed['d_coll']:+.2f} cm"
        f"      ({observed['coll_patient']:+.2f} vs "
        f"{observed['coll_control']:+.2f})"
    )
    print()

    quadrants = result["quadrants"]

    if quadrants.empty:
        print("  no usable paired draws")
        return

    print(quadrants.to_string(index=False))

    failed = int(quadrants["failed_draws"].iloc[0])

    if failed:
        print(
            f"\n  {failed} of {settings['n_boot']} draws produced no usable "
            "boundary in one task or the other."
        )

    threshold = settings["threshold"]
    winner = quadrants.loc[quadrants["proportion"].idxmax()]

    print()

    if winner["proportion"] >= threshold:
        print(f"  {winner['meaning']}: {winner['proportion']:.3f} of draws.")
    else:
        print(
            f"  Inconclusive: the largest quadrant holds "
            f"{winner['proportion']:.3f} of draws, below the "
            f"{threshold:.2f} criterion."
        )


def plot(result, save_as=None):
    """The paired draws in the D_PPS by D_coll plane, with the observed point.

    A concordant result puts the cloud almost entirely in one quadrant. A cloud
    straddling the axes is inconclusive, and that is visible here in a way it
    is not in a table of proportions.
    """
    draws = result["draws"]
    observed = result["observed"]

    usable = draws[["d_pps", "d_coll"]].dropna()

    fig, ax = plt.subplots(figsize=(5.4, 5.0))

    ax.scatter(
        usable["d_pps"], usable["d_coll"],
        s=3, alpha=0.20, color=style.CONTROL, edgecolors="none", zorder=2,
    )
    ax.scatter(
        observed["d_pps"], observed["d_coll"],
        s=90, marker="X", color=style.INK, edgecolors="white", linewidth=0.8,
        zorder=5, label="observed",
    )

    ax.axhline(0, linestyle="--", linewidth=0.8, color=style.MUTED, zorder=1)
    ax.axvline(0, linestyle="--", linewidth=0.8, color=style.MUTED, zorder=1)

    ax.set_xlabel(r"$D_{\mathrm{PPS}}$ (ranks)")
    ax.set_ylabel(r"$D_{\mathrm{coll}}$ (cm)")
    ax.set_title(
        "Patient minus control, both tasks resampled together",
        loc="left", fontsize=9.5,
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    style.clean(ax)
    style.finish_figure(fig)

    if save_as:
        figures.save(fig, save_as)

    return fig