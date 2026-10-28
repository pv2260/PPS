"""Does PPS adaptation relate to the components of the collision shift?

H7a correlated $\\Delta_{PPS}$ against $\\Delta_{coll}$ and found a
negative association, opposite to the prediction. The decomposition
offers a reason to look further: $\\Delta_{coll}$ tracks the change in
decision criterion almost perfectly and the change in spatial precision
not at all, so correlating $\\Delta_{PPS}$ against it is correlating a
facilitation gradient against a decision threshold.

This plots $\\Delta_{PPS}$ against each component separately:

    Delta_d      change in sensitivity, slow minus fast
    Delta_c      change in criterion, slow minus fast
    Delta_log_k  change in spatial precision, log ratio of slopes

If $\\Delta_{PPS}$ relates to any of the three, it should be the one
describing spatial precision rather than the one describing policy.

At six or seven participants none of these correlations is a test. They
are shown so that the direction and the leverage of individual
participants are visible, and each panel reports the leave-one-out range
alongside the coefficient for that reason.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import style

from ppsanalysis import figures


COMPONENTS = [
    ("delta_coll", r"$\Delta_{coll}$ (cm)", "Boundary shift"),
    ("delta_d", r"$\Delta d'$", "Sensitivity"),
    ("delta_c", r"$\Delta c$", "Criterion"),
    ("delta_log_k", r"$\Delta \log|k|$", "Spatial precision"),
]


def collision_components(trials, subjects=None):
    """Sensitivity, criterion and precision, each as slow minus fast.

    Sensitivity and criterion come from the near trials with the
    loglinear correction. Precision is the log ratio of the offset
    slopes, fitted separately by speed so that a difference between
    conditions is not absorbed by a shared slope.
    """
    from ppsanalysis.collision import safe_logistic_regression

    d = trials[trials["usable"]] if "usable" in trials.columns else trials

    if subjects is not None:
        d = d[d["subject"].isin(subjects)]

    rows = []

    for subject in sorted(d["subject"].unique()):
        record = {"subject": subject}

        for speed in ["slow", "fast"]:
            cell = d[(d["subject"] == subject) & (d["speed"] == speed)]

            fit = cell.dropna(subset=["offset_from_edge_cm", "response_hit"])
            model = (
                safe_logistic_regression(
                    fit[["offset_from_edge_cm"]].to_numpy(),
                    fit["response_hit"].astype(int).to_numpy(),
                )
                if len(fit) > 10 else None
            )
            record[f"k_{speed}"] = (
                abs(float(model.coef_[0][0])) if model is not None else np.nan
            )

            near = cell[cell["trial_type"].isin(["near_hit", "near_miss"])]
            signal = near[near["trial_type"] == "near_hit"]
            noise = near[near["trial_type"] == "near_miss"]

            if not len(signal) or not len(noise):
                record[f"d_{speed}"] = np.nan
                record[f"c_{speed}"] = np.nan
                continue

            # Loglinear correction, applied to everyone rather than only
            # to those at ceiling, so the values stay comparable.
            H = (signal["response_hit"].sum() + 0.5) / (len(signal) + 1)
            F = (noise["response_hit"].sum() + 0.5) / (len(noise) + 1)

            record[f"d_{speed}"] = stats.norm.ppf(H) - stats.norm.ppf(F)
            record[f"c_{speed}"] = -0.5 * (stats.norm.ppf(H)
                                           + stats.norm.ppf(F))

        rows.append(record)

    out = pd.DataFrame(rows).set_index("subject")

    out["delta_d"] = out["d_slow"] - out["d_fast"]
    out["delta_c"] = out["c_slow"] - out["c_fast"]
    out["delta_log_k"] = np.log(out["k_slow"]) - np.log(out["k_fast"])

    # Delta_coll comes from the shared-slope model rather than from the
    # per-speed fits above, since that is the quantity H3 and H7a use.
    # Computing it here keeps every panel on one frame.
    from ppsanalysis.collision import (
        compute_collision_indices_one_session as _indices,
    )

    deltas = {}

    for subject in out.index:
        cell = d[d["subject"] == subject]

        try:
            deltas[subject] = float(_indices(cell.copy())["delta_coll"])
        except Exception:
            deltas[subject] = np.nan

    out["delta_coll"] = pd.Series(deltas)

    return out


def _leave_one_out(x, y):
    """Spearman rho with each participant dropped in turn."""
    values = []

    for i in range(len(x)):
        keep = np.ones(len(x), dtype=bool)
        keep[i] = False
        values.append(stats.spearmanr(x[keep], y[keep]).statistic)

    return np.asarray(values)


def plot_pps_against_components(delta_pps, trials, subjects=None,
                                clinical=(), components_shown=None,
                                show_rho=True, save_as=None):
    """One panel per component, each against Delta_PPS.

    ``clinical`` names participants to draw in their own colour rather
    than as part of the group. They are plotted but excluded from the
    correlation, since a coefficient computed across a young sample and
    two older participants describes neither.

    ``components_shown`` selects which components appear, so the
    precision panel can stand alone in the main text while sensitivity
    and criterion go to an annex. A single panel is drawn larger, since
    the point labels need room that three panels across a page cannot
    give them.

    ``show_rho`` puts the coefficient in the panel title. At five
    participants with a leave-one-out range spanning zero it invites the
    number to be read as a result, so it can be turned off and the
    coefficient reported in the caption where it can be qualified.
    """
    components = collision_components(trials, subjects=subjects)

    table = (
        delta_pps[["subject", "delta_pps"]]
        .drop_duplicates("subject")
        .set_index("subject")
        .join(components, how="inner")
    )

    group = table.drop(index=[c for c in clinical if c in table.index])

    rows = [c for c in COMPONENTS
            if components_shown is None or c[0] in components_shown]

    # One panel gets the room three would have shared. The labels are
    # the constraint here, not the data: at five points in a narrow
    # panel they overlap whatever the axis limits.
    width = 5.6 if len(rows) == 1 else 5.0 * len(rows)

    fig, axes = plt.subplots(
        1, len(rows), figsize=(width, 4.2),
        sharey=True, squeeze=False, gridspec_kw={"wspace": 0.16},
    )
    axes = axes[0]

    for ax, (column, symbol, label) in zip(axes, rows):
        pair = group[["delta_pps", column]].dropna()

        ax.scatter(
            pair[column], pair["delta_pps"],
            s=42, color=style.CONTROL, edgecolor="white", linewidth=0.6,
            zorder=3,
        )

        # Labels are placed on whichever side of the point has more room,
        # decided from the horizontal position within the panel. A fixed
        # offset collides wherever two participants sit close together,
        # which at five points is most of them.
        span = pair[column].max() - pair[column].min() or 1.0

        for name, row in pair.iterrows():
            to_the_right = (row[column] - pair[column].min()) / span < 0.6

            ax.annotate(
                name, (row[column], row["delta_pps"]),
                textcoords="offset points",
                xytext=(6, 4) if to_the_right else (-6, 4),
                ha="left" if to_the_right else "right",
                fontsize=7, color=style.MUTED,
            )

        for name in clinical:
            if name not in table.index:
                continue

            row = table.loc[name]

            if not np.isfinite(row[column]) or not np.isfinite(row["delta_pps"]):
                continue

            colour = (
                style.PATIENT if name.startswith("PT") else style.AGE_MATCHED
            )

            ax.scatter(
                row[column], row["delta_pps"],
                s=60, color=colour, marker="D",
                edgecolor="white", linewidth=0.7, zorder=4,
            )
            to_the_right = (
                (row[column] - table[column].min())
                / (table[column].max() - table[column].min() or 1.0)
            ) < 0.6

            ax.annotate(
                name, (row[column], row["delta_pps"]),
                textcoords="offset points",
                xytext=(8, 4) if to_the_right else (-8, 4),
                ha="left" if to_the_right else "right",
                fontsize=7, color=colour, fontweight="bold",
            )

        ax.axhline(0, linestyle="--", linewidth=0.8, color=style.MUTED,
                   zorder=0)
        ax.axvline(0, linestyle="--", linewidth=0.8, color=style.MUTED,
                   zorder=0)

        ax.set_title(label, loc="left", fontsize=9.5)

        if show_rho and len(pair) >= 4:
            result = stats.spearmanr(pair[column], pair["delta_pps"])
            loo = _leave_one_out(
                pair[column].to_numpy(), pair["delta_pps"].to_numpy()
            )

            # Inside the panel rather than in the title, and with the
            # leave-one-out range beside it: at this n one participant
            # can determine the coefficient, and rho alone does not show
            # whether one has.
            ax.text(
                0.02, 0.02,
                fr"$\rho$ = {result.statistic:+.2f}, n = {len(pair)}"
                "\n"
                f"LOO {loo.min():+.2f} to {loo.max():+.2f}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7, color=style.MUTED, linespacing=1.4,
            )

        ax.set_xlabel(f"{symbol}, slow $-$ fast")
        style.clean(ax)

    axes[0].set_ylabel(r"$\Delta_{PPS}$ (ranks)")

    if clinical:
        handles = [
            axes[0].scatter([], [], s=60, marker="D", color=style.PATIENT,
                            label="Patient"),
            axes[0].scatter([], [], s=60, marker="D",
                            color=style.AGE_MATCHED,
                            label="Matched controls"),
            axes[0].scatter([], [], s=42, color=style.CONTROL,
                            label="Young controls"),
        ]

        # Above the row: every interior corner holds either a point, its
        # label, or the coefficient.
        fig.legend(
            handles=handles, frameon=False, fontsize=8, ncol=3,
            loc="upper left", bbox_to_anchor=(0.08, 1.02),
        )

    fig.suptitle(
        r"PPS adaptation against the components of the collision shift",
        x=0.02, ha="left", y=1.10, color=style.INK,
    )

    fig.text(
        0.5, -0.04,
        "Clinical participants are drawn but excluded from the "
        "correlations, which are computed across the young controls only",
        ha="center", va="top", fontsize=8, color=style.MUTED,
    )

    figures.label_panels(list(axes))
    style.finish_figure(fig, tight=False)

    if save_as:
        figures.save(fig, save_as)

    return fig, table