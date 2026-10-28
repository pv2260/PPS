"""
H7a. Across the young controls: does PPS adaptation go with Hit-or-Miss
     performance?

WHAT WE ARE TESTING
-------------------
H7b asks whether reduced PPS plasticity and reduced collision performance
co-occur within the one patient. H7a asks the group version: across the young
controls, do the two speed-driven measures covary from person to person?

Three correlations, each between per-subject dPPS and one collision measure:

    H7a.1   dPPS vs delta_coll               predicted POSITIVE
    H7a.2   dPPS vs near-boundary accuracy   predicted POSITIVE
    H7a.3   dPPS vs post-switch carryover    predicted NEGATIVE

WHY PERMUTATION AND NOT THE ASYMPTOTIC p
----------------------------------------
With seven subjects the sampling distribution of Spearman's rho has nothing to
do with its large-sample approximation. Instead we enumerate every possible
pairing of the two variables. Each subject's dPPS is held fixed and the
collision values are permuted across subjects, which gives n! rearrangements
(5040 for n = 7). The one-sided p is the proportion of those rearrangements at
least as extreme as the observed rho, in the predicted direction.

Enumeration is exact only while n! stays tractable. Above EXACT_MAX_N the
module samples random pairings instead and says so in its output.

TIES
----
rho is computed as Pearson correlation on midranks rather than the
1 - 6*sum(d^2) shortcut, because the shortcut is wrong whenever two subjects
tie on a measure. Accuracy values in particular tie easily.

SUBJECT SELECTION
-----------------
Only subjects present in BOTH the boundaries table and the collision trials can
enter. Anyone missing from either is dropped and named in the output, so the n
behind each rho is never silent.
"""

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

import style

from .. import config
from .. import figures
from ..collision import compute_collision_indices_one_session


# ---------------------------------------------------------------------
# Participant display names
# ---------------------------------------------------------------------
# Figures name the role rather than the identifier: a reader needs to know
# which participant is the patient and which is the control. Which specific
# control it was belongs in the caption, where it is stated once.
#
# The map lives in ``style`` so every module labels the same person the same
# way. The fallback below keeps this module working against a style module
# that predates it.

def _display_name(subject):
    """Figure label for a participant. Unlisted ids pass through unchanged."""
    if hasattr(style, "display_name"):
        return style.display_name(subject)

    return {
        "PT001": "Patient",
        "PA002": "Age-matched control",
        "PA001": "Adult control",
    }.get(str(subject), str(subject))


def _display_names(subjects):
    """Vectorised form, for axis tick labels."""
    return [_display_name(s) for s in subjects]



# Above this n, n! enumeration stops being sensible and we sample instead.
EXACT_MAX_N = 8


def spearman_rho(x, y):
    """Spearman's rho as Pearson correlation on midranks (tie-safe)."""

    rx = rankdata(x) - np.mean(rankdata(x))
    ry = rankdata(y) - np.mean(rankdata(y))

    denominator = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))

    if denominator == 0:
        return np.nan

    return float(np.dot(rx, ry) / denominator)


def permutation_p(x, y, direction, n_perm=None, seed=None):
    """One-sided permutation p for Spearman's rho.

    direction :
        "positive" -> p = P(rho_perm >= rho_observed)
        "negative" -> p = P(rho_perm <= rho_observed)

    The observed pairing is itself one of the enumerated permutations, so p can
    never come back as exactly zero.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    rho_observed = spearman_rho(x, y)

    if not np.isfinite(rho_observed):
        return {
            "rho": np.nan, "p": np.nan, "n": n,
            "n_perm": 0, "exact": False,
        }

    rx = rankdata(x) - np.mean(rankdata(x))
    ry = rankdata(y) - np.mean(rankdata(y))
    denominator = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))

    exact = n <= EXACT_MAX_N

    if exact:
        orders = np.array(list(itertools.permutations(range(n))))
    else:
        rng = np.random.default_rng(
            config.RANDOM_SEED if seed is None else seed)
        n_draw = config.N_BOOT if n_perm is None else n_perm
        orders = np.array([rng.permutation(n) for _ in range(n_draw)])
        # Keep the observed pairing in the reference set.
        orders = np.vstack([np.arange(n), orders])

    rho_perm = (ry[orders] @ rx) / denominator

    if direction == "positive":
        p = float(np.mean(rho_perm >= rho_observed))
    elif direction == "negative":
        p = float(np.mean(rho_perm <= rho_observed))
    else:
        raise ValueError(f"direction must be positive or negative, got {direction!r}")

    return {
        "rho": rho_observed,
        "p": p,
        "n": n,
        "n_perm": len(orders),
        "exact": exact,
        "rho_perm": rho_perm,
    }


def build_table(boundaries, t, correction="fastest_tactile", session=1):
    """Assemble one row per young control: dPPS beside the collision indices.

    ``boundaries`` is H2's per-subject boundary table, the one carrying
    columns correction, subject, fast, slow, delta_pps. Only the requested
    correction level is used.
    """

    available_corrections = sorted(boundaries["correction"].unique())

    if correction not in available_corrections:
        raise ValueError(
            f"correction {correction!r} not in boundaries table. "
            f"Available: {available_corrections}"
        )

    pps = boundaries[boundaries["correction"] == correction][
        ["subject", "delta_pps"]].copy()

    # Exclude the patient and the matched controls: H7a is the young-control
    # group test.
    clinical_ids = {config.PATIENT_ID}
    matched = config.MATCHED_CONTROL_IDS
    if isinstance(matched, str):
        matched = [matched]
    clinical_ids |= set(matched or [])

    pps = pps[~pps["subject"].isin(clinical_ids)]

    rows = []
    dropped = []

    for subject in sorted(pps["subject"].unique()):
        trials = t.collision_trials[t.collision_trials["subject"] == subject]

        if "session" in trials.columns and session is not None:
            trials = trials[trials["session"] == session]

        if len(trials) == 0:
            dropped.append(subject)
            continue

        indices = compute_collision_indices_one_session(trials.copy())

        rows.append({
            "subject": subject,
            "delta_pps": float(
                pps.loc[pps["subject"] == subject, "delta_pps"].iloc[0]),
            "delta_coll": indices["delta_coll"],
            "accuracy": indices["accuracy"],
            "near_boundary_accuracy": indices["near_boundary_accuracy"],
            "carryover_cm": indices["carryover_cm"],
        })

    table = pd.DataFrame(rows)

    print(f"H7a: correction level {correction!r}, session {session}")
    print()

    if dropped:
        print("H7a: no collision trials, dropped from all three tests:")
        print("  " + ", ".join(dropped))
        print()

    if len(table) == 0:
        print("H7a: no subject has both measures, so nothing can be tested.")
        return table

    print(f"H7a: {len(table)} young controls with both measures")
    print()
    print(table.round(3).to_string(index=False))
    print()

    return table


def run_one(table, y_column, y_label, title, direction, alpha=0.05, plot_ax=None):
    """Run one H7a sub-test.

    direction :
        "positive" for H7a.1 and H7a.2.
        "negative" for H7a.3, where weaker PPS adaptation is predicted to go
        with MORE carryover.
    """

    usable = table[["delta_pps", y_column]].dropna()

    if len(usable) < 3:
        print(f"{title}: cannot run ({len(usable)} subjects with both values).")
        return {"classification": "not estimable", "n": len(usable)}

    result = permutation_p(
        usable["delta_pps"].to_numpy(),
        usable[y_column].to_numpy(),
        direction=direction,
    )

    sign_ok = (result["rho"] > 0) if direction == "positive" else (result["rho"] < 0)

    if sign_ok and result["p"] < alpha:
        classification = "supported"
    elif sign_ok:
        classification = "predicted direction, not significant"
    else:
        classification = "opposite direction"

    result["classification"] = classification
    result["direction"] = direction

    reported = {
        "rho": round(result["rho"], 3),
        "p_one_sided": round(result["p"], 4),
        "n": result["n"],
        "n_perm": result["n_perm"],
        "exact": result["exact"],
        "classification": classification,
    }

    print(f"{title}")
    print()
    print(pd.Series(reported).to_string())
    print()

    if plot_ax is not None:
        draw_scatter(
            plot_ax,
            table.dropna(subset=["delta_pps", y_column]),
            y_column,
            y_label,
            f"{title}\nrho = {result['rho']:.2f}, p = {result['p']:.3f}",
        )

    return result


def draw_scatter(ax, table, y_column, y_label, title):
    """One point per subject, labelled, with the rank trend behind it.

    With seven points the individual subjects matter, so they are named. A
    correlation carried by one outlying subject is visible here and invisible
    in a table of rho values.
    """

    ax.scatter(
        table["delta_pps"],
        table[y_column],
        s=34,
        color=style.CONTROL,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )

    for _, row in table.iterrows():
        ax.annotate(
            _display_name(row["subject"]),
            (row["delta_pps"], row[y_column]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=6,
            color="gray",
            zorder=4,
        )

    ax.axhline(0, linestyle="--", linewidth=0.8, color="gray", zorder=1)
    ax.axvline(0, linestyle="--", linewidth=0.8, color="gray", zorder=1)

    ax.set_xlabel(r"$\Delta_{PPS}$ (cm)")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    style.clean(ax)


def run(boundaries, t, correction="fastest_tactile", session=1, alpha=0.05, plot=True):
    """Run all three parts of H7a.

    boundaries : H2's per-subject boundary table, res["boundaries"].
    t          : the tables object holding collision_trials.
    """

    table = build_table(boundaries, t, correction=correction, session=session)

    if len(table) == 0:
        return {"skipped": True}

    results = {"skipped": False, "table": table}

    if plot:
        fig, axes = figures.new_figure(
            n_panels=3, width=figures.DOUBLE_COLUMN, height=2.8)
    else:
        axes = [None, None, None]

    results["h7a_1"] = run_one(
        table, "delta_coll", r"$\Delta_{coll}$ (cm)", "H7a.1",
        direction="positive", alpha=alpha, plot_ax=axes[0],
    )

    results["h7a_2_near"] = run_one(
        table, "near_boundary_accuracy", "near-boundary accuracy",
        "H7a.2 (primary)",
        direction="positive", alpha=alpha, plot_ax=axes[1],
    )

    # H7a.3: more carryover is worse updating, so the predicted association
    # with dPPS is NEGATIVE.
    results["h7a_3"] = run_one(
        table, "carryover_cm", r"$carryover$ (cm)", "H7a.3",
        direction="negative", alpha=alpha, plot_ax=axes[2],
    )

    # Overall accuracy is descriptive only, so it is tested but not plotted.
    results["h7a_2_overall"] = run_one(
        table, "accuracy", "overall accuracy",
        "H7a.2 (descriptive: overall accuracy)",
        direction="positive", alpha=alpha, plot_ax=None,
    )

    if plot:
        figures.label_panels(axes)
        style.finish_figure(fig)
        figures.save(fig, "h7a_control_correlations")
        plt.show()
        results["figure"] = fig

    summary = pd.DataFrame([
        {"hypothesis": "H7a.1",
         "rho": results["h7a_1"].get("rho", np.nan),
         "p": results["h7a_1"].get("p", np.nan),
         "classification": results["h7a_1"]["classification"]},
        {"hypothesis": "H7a.2 (near, primary)",
         "rho": results["h7a_2_near"].get("rho", np.nan),
         "p": results["h7a_2_near"].get("p", np.nan),
         "classification": results["h7a_2_near"]["classification"]},
        {"hypothesis": "H7a.2 (overall, descr.)",
         "rho": results["h7a_2_overall"].get("rho", np.nan),
         "p": results["h7a_2_overall"].get("p", np.nan),
         "classification": results["h7a_2_overall"]["classification"]},
        {"hypothesis": "H7a.3",
         "rho": results["h7a_3"].get("rho", np.nan),
         "p": results["h7a_3"].get("p", np.nan),
         "classification": results["h7a_3"]["classification"]},
    ])

    print("H7a summary")
    print()
    print(summary.round(3).to_string(index=False))

    results["summary"] = summary

    return results