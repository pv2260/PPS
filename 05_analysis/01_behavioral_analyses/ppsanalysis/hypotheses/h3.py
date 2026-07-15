"""
H3. The collision boundary shifts when the ball moves faster.

WHAT WE ARE TESTING
-------------------
In the Hit-or-Miss task, a ball flies past the participant at some sideways
offset. The participant says whether it would have hit them or missed them.

If the offset is small (the ball passes close to the body), they say HIT.
If the offset is large, they say MISS. Somewhere in between there is an offset
where they are equally likely to say either. That offset is the PSE (Point of
Subjective Equality) and it is the collision boundary.

We fit a logistic regression to each participant:

    logit( P(say HIT) ) = b0 + b_offset * offset + b_speed * speed

where speed is coded slow = -1, fast = +1.

The PSE is where P(HIT) = 0.5, so:

    PSE(slow) = -(b0 - b_speed) / b_offset
    PSE(fast) = -(b0 + b_speed) / b_offset

    Delta_coll = PSE(fast) - PSE(slow)

Delta_coll > 0 means the boundary moved OUTWARD for a fast ball, which is the
same prediction as H2 makes for PPS.

HOW WE DECIDE IF IT IS SUPPORTED
--------------------------------
One Delta_coll per participant, then bootstrap the group mean.
Supported if the mean is positive AND the bootstrap 95% CI lies above zero.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit

import style

from .. import config
from .. import figures
from ..stats_utils import bootstrap_mean_ci, wilcoxon_one_sided
from ..collision import code_speed, safe_logistic_regression


def run(t, plot=True):
    """Run H3."""

    results = {}

    # ------------------------------------------------------------------
    # Step 1. One Delta_coll per participant.
    # ------------------------------------------------------------------

    delta_per_subject = t.collision_indices_young.groupby(
        "subject",
        as_index=False,
    )["delta_coll"].mean()

    results["delta_per_subject"] = delta_per_subject

    # ------------------------------------------------------------------
    # Step 2. Group test.
    # ------------------------------------------------------------------

    values = delta_per_subject["delta_coll"].dropna().to_numpy()

    boot = bootstrap_mean_ci(
        values,
        n_boot=config.N_BOOT,
        ci=95,
        seed=config.RANDOM_SEED,
    )

    wilcoxon = wilcoxon_one_sided(values, alternative="greater")

    supported = bool(boot["mean"] > 0) and bool(boot["ci_low"] > 0)

    results["values"] = values
    results["boot"] = boot
    results["wilcoxon"] = wilcoxon
    results["supported"] = supported

    # ------------------------------------------------------------------
    # Step 3. A group-level logistic fit, for the psychometric figure.
    # ------------------------------------------------------------------
    # This is ILLUSTRATIVE ONLY. It pools every trial from every participant,
    # which ignores between-subject differences. The actual test above is done
    # per participant, which is the right way. This fit exists so the reader can
    # see what the psychometric functions look like.

    trials = t.collision_young[t.collision_young["usable"]].copy()

    trials = trials[trials["speed"].isin([config.SLOW_LABEL, config.FAST_LABEL])]
    trials = trials.dropna(subset=["offset_from_edge_cm", "response_hit"])

    trials["speed_code"] = code_speed(trials["speed"])
    trials = trials.dropna(subset=["speed_code"])

    results["pooled_trials"] = trials

    model = safe_logistic_regression(
        trials[["offset_from_edge_cm", "speed_code"]].to_numpy(),
        trials["response_hit"].astype(int).to_numpy(),
    )

    results["pooled_model"] = model

    report(results)

    if plot:
        fig = make_figure(results)
        if fig is not None:
            figures.save(fig, "h3_delta_coll")
            plt.show()
            results["figure"] = fig

    return results


def report(results):
    """Print the H3 numbers."""

    boot = results["boot"]
    wilcoxon = results["wilcoxon"]

    print("H3, step 1: Delta_coll per participant")
    print("  Delta_coll = PSE(fast) - PSE(slow), in cm.")
    print("  Positive = the collision boundary moved outward for a fast ball.")
    print()
    print(results["delta_per_subject"].round(2).to_string(index=False))
    print()

    print("H3, step 2: group test")
    print(f"  number of participants  = {boot['n']}")
    print(f"  mean Delta_coll         = {boot['mean']:.2f} cm")
    print(f"  bootstrap 95% CI        = [{boot['ci_low']:.2f}, {boot['ci_high']:.2f}] cm")
    print(f"  Wilcoxon (one-sided)    = W {wilcoxon['W']}, p {wilcoxon['p']:.4f}")
    print()

    if results["supported"]:
        print("  H3 IS SUPPORTED: mean > 0 and the 95% CI lies above zero.")
    else:
        print("  H3 is NOT supported.")


def make_figure(results):
    """Build the H3 figure. Two panels."""

    fig, axes = figures.new_figure(n_panels=2, height=3.0)

    # --- Panel A: Delta_coll per participant ---------------------------------
    ax = axes[0]

    delta_per_subject = results["delta_per_subject"]
    values = results["values"]
    boot = results["boot"]

    x_positions = np.arange(len(values))

    ax.scatter(
        x_positions,
        values,
        s=40,
        color=style.CONTROL,
        edgecolor=style.INK,
        linewidth=0.6,
        zorder=3,
    )

    for i, subject in enumerate(delta_per_subject["subject"]):
        ax.text(x_positions[i], values[i], f" {subject}", fontsize=6, va="center")

    ax.axhline(0, linestyle="--", linewidth=0.8, color="gray")

    if np.isfinite(boot["mean"]):
        lower_error = boot["mean"] - boot["ci_low"]
        upper_error = boot["ci_high"] - boot["mean"]

        ax.errorbar(
            len(values) + 0.5,
            boot["mean"],
            yerr=[[lower_error], [upper_error]],
            fmt="o",
            color=style.INK,
            markersize=6,
            capsize=4,
            zorder=3,
            label="mean (95% CI)",
        )

    ax.set_xticks(list(x_positions) + [len(values) + 0.5])
    ax.set_xticklabels(list(delta_per_subject["subject"]) + ["mean"], rotation=30, ha="right")
    ax.set_ylabel(r"$\Delta_{coll}$ (cm)")
    ax.set_title("Speed-induced boundary shift")
    ax.legend()

    style.clean(ax)

    # --- Panel B: psychometric curves ----------------------------------------
    ax = axes[1]

    model = results["pooled_model"]
    trials = results["pooled_trials"]

    if model is None or np.isclose(model.coef_[0][0], 0):
        ax.text(
            0.5, 0.5,
            "logistic fit unavailable\n(too few trials, or a flat offset slope)",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=7,
            color="#8A8A8A",
        )
        ax.set_axis_off()

        figures.label_panels(axes)
        fig.tight_layout()
        return fig

    # Pull the fitted coefficients out of the model.
    b0 = model.intercept_[0]
    b_offset = model.coef_[0][0]
    b_speed = model.coef_[0][1]

    # slow is coded -1, so its intercept is (b0 - b_speed)
    # fast is coded +1, so its intercept is (b0 + b_speed)
    pse_slow = -(b0 - b_speed) / b_offset
    pse_fast = -(b0 + b_speed) / b_offset

    smooth_x = np.linspace(
        trials["offset_from_edge_cm"].min(),
        trials["offset_from_edge_cm"].max(),
        200,
    )

    ax.plot(
        smooth_x,
        expit((b0 - b_speed) + b_offset * smooth_x),
        color=style.SLOW,
        linewidth=1.8,
        label="slow",
        zorder=2,
    )
    ax.plot(
        smooth_x,
        expit((b0 + b_speed) + b_offset * smooth_x),
        color=style.FAST,
        linewidth=1.8,
        label="fast",
        zorder=2,
    )

    # Empirical P(HIT), binned. The dots are sized by how many trials fell in
    # each bin, so a tiny dot warns you not to trust that point.
    bin_edges = np.linspace(
        trials["offset_from_edge_cm"].min(),
        trials["offset_from_edge_cm"].max(),
        9,
    )
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    speed_colours = {
        config.SLOW_LABEL: style.SLOW,
        config.FAST_LABEL: style.FAST,
    }

    for speed, colour in speed_colours.items():

        one_speed = trials[trials["speed"] == speed]

        which_bin = np.digitize(one_speed["offset_from_edge_cm"], bin_edges) - 1
        which_bin = np.clip(which_bin, 0, len(bin_centres) - 1)

        binned = pd.DataFrame({
            "bin": which_bin,
            "hit": one_speed["response_hit"].astype(int).to_numpy(),
        }).groupby("bin")["hit"].agg(["mean", "size"])

        ax.scatter(
            bin_centres[binned.index],
            binned["mean"],
            s=6 + binned["size"] * 0.6,
            color=style.shade(colour, -0.1),
            alpha=0.55,
            edgecolor="none",
            zorder=3,
        )

    ax.axhline(0.5, linestyle="--", linewidth=0.8, color="gray")

    # Mark each PSE, and draw an arrow showing the shift between them.
    for pse, colour in [(pse_slow, style.SLOW), (pse_fast, style.FAST)]:
        ax.plot([pse, pse], [0, 0.5], linestyle=":", linewidth=1.0,
                color=style.shade(colour, -0.1), zorder=1)

    ax.annotate(
        "",
        xy=(pse_fast, 0.5),
        xytext=(pse_slow, 0.5),
        arrowprops=dict(arrowstyle="<->", color=style.INK, lw=0.9),
    )
    ax.text(
        (pse_slow + pse_fast) / 2,
        0.545,
        f"$\\Delta_{{coll}}$ = {pse_fast - pse_slow:.1f} cm",
        ha="center",
        fontsize=7,
        color=style.INK,
    )

    ax.set_xlabel("Offset from shoulder edge (cm)")
    ax.set_ylabel("P(respond HIT)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Collision psychometric")
    ax.legend()

    style.clean(ax)

    figures.label_panels(axes)
    fig.tight_layout()

    return fig
