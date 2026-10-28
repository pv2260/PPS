"""Which tactile baseline, and does the answer depend on it?

Facilitation is a tactile baseline minus the visuo-tactile median. The
corrections differ only in how that baseline is obtained, so every difference
between the curves they produce comes from the baseline alone. This module
computes them side by side and separates three questions that are usually run
together:

    level     Does the correction move the whole curve up or down? Expected,
              and mostly uninteresting: the tactile distribution is
              right-skewed, so a mean baseline sits above a median one and
              facilitation shifts with it.

    shape     Does the correction change how facilitation varies with
              distance? This is the one that matters, because every
              distance-based hypothesis rests on the shape. Two corrections
              differing only by a constant give profiles correlating at 1.0
              and the same conclusion at a different magnitude.

    result    Does the near-far index, and whether a boundary is estimable,
              survive the choice? A result that holds under one correction
              and not another is a result about the correction.

What the corrections are
------------------------
    delay_matched      the tactile median at that exact distance and speed,
                       from its own four trials. Preregistered. Follows the
                       data exactly, including its sampling noise.

    expectancy_model   the same delay-dependence estimated as a function
                       rather than as fourteen independent cell means. Four
                       coefficients from roughly fifty trials. Keeps the
                       expectancy control; changes only the estimator.

    overall_tactile    one value per participant and speed, averaged across
                       distances. Flat by construction, so any
                       delay-dependence is discarded rather than modelled.

    fastest_tactile    the fastest distance's estimate. Also flat, and the
                       most conservative, since visuo-tactile has to beat the
                       participant's best unimodal condition.

The two flat corrections cannot produce a shape of their own: their
facilitation profile is the visuo-tactile profile inverted and shifted. They
are included because that is the point of comparison, not because they are
candidates.

Usage
-----
    from ppsanalysis import baseline_comparison as bc

    result = bc.compare(c.trials, degree=DEGREE)
    bc.report(result)
    bc.plot_profiles(result, save_as="baseline_comparison")

Every table is returned as well as printed, so anything here can go straight
into the thesis.
"""

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import style
from . import figures
from . import config as cfg
from .hypotheses import h1


DEFAULT_CORRECTIONS = (
    "delay_matched",
    "expectancy_model",
    "overall_tactile",
    "fastest_tactile",
    "noel_endpoints",
)

REFERENCE = "delay_matched"

NEAR = [1, 2]
FAR = [6, 7]


# ==================================================================
# The comparison
# ==================================================================


def compare(
    trials,
    corrections=DEFAULT_CORRECTIONS,
    estimator="median",
    degree=1,
    subjects=None,
    exclude=None,
    n_boot=None,
):
    """Build every correction's profile and the three comparisons on them.

    Parameters
    ----------
    trials : DataFrame
        Analysis-ready trials, e.g. ``cohort.trials``.
    corrections : sequence of str
        Names from ``h1.CORRECTIONS``.
    estimator : {"mean", "median"}
        Cell estimator, applied identically under every correction, so the
        visuo-tactile term is held constant and only the baseline varies.
    degree : int
        Polynomial order for the expectancy model, ignored by the others.
    subjects : sequence of str or None
        Restrict to these participants. None keeps everyone.
    exclude : sequence of str or None
        Drop these participants. Applied after ``subjects``.

        The filtering happens before any correction is applied, so every
        correction sees the same sample. Filtering afterwards would let a
        participant contribute to one correction's group estimate and not
        another's, and the comparison would then confound the correction
        with who was in it.
    n_boot : int or None
        Resamples for the near-far group test. None uses ``config.N_BOOT``.

    Returns
    -------
    dict with keys:
        profiles     one row per correction, subject, speed and distance
        levels       the group profile per correction and speed
        shape        pairwise correlations between correction profiles
        nearfar      the near-far index per correction and participant
        group        the group test on that index, per correction
        boundaries   half-maximum boundary and its amplitude, per correction
        settings     what was passed in
    """
    n_boot = cfg.N_BOOT if n_boot is None else n_boot

    unknown = [c for c in corrections if c not in h1.CORRECTIONS]
    if unknown:
        raise ValueError(
            f"unknown correction(s): {unknown}. "
            f"Available: {list(h1.CORRECTIONS)}"
        )

    trials = trials.copy()

    if subjects is not None:
        trials = trials[trials["subject"].isin(list(subjects))]

    if exclude:
        dropped = sorted(set(exclude) & set(trials["subject"]))
        trials = trials[~trials["subject"].isin(list(exclude))]

        missing = sorted(set(exclude) - set(dropped))

        if dropped:
            print(f"Excluded: {', '.join(dropped)}")
        if missing:
            print(f"  not present: {', '.join(missing)}")

    if trials.empty:
        raise ValueError("no trials left after filtering")

    frames = []

    for correction in corrections:
        profile = h1.facilitation_profile(
            trials,
            estimator=estimator,
            correction=correction,
            expectancy_degree=degree,
        )

        if profile.empty:
            continue

        frames.append(profile.assign(correction=correction))

    if not frames:
        raise ValueError("no correction produced a profile")

    profiles = pd.concat(frames, ignore_index=True)

    return {
        "profiles": profiles,
        "levels": _levels(profiles),
        "shape": _shape(profiles),
        "nearfar": _nearfar(profiles),
        "group": _group(profiles, n_boot=n_boot),
        "boundaries": _boundaries(profiles),
        "settings": {
            "corrections": list(corrections),
            "estimator": estimator,
            "degree": degree,
            "n_boot": n_boot,
            "n_subjects": int(profiles["subject"].nunique()),
            "subjects": sorted(profiles["subject"].unique()),
            "excluded": sorted(exclude) if exclude else [],
        },
    }


def _levels(profiles):
    """Group facilitation per correction, speed and distance."""
    subject = (
        profiles
        .groupby(["correction", "subject", "speed", "position_rank"],
                 as_index=False)["facilitation_ms"]
        .mean()
    )

    group = (
        subject
        .groupby(["correction", "speed", "position_rank"], as_index=False)
        ["facilitation_ms"]
        .agg(group_mean="mean", group_sd="std", n="count")
    )
    group["group_sem"] = group["group_sd"] / np.sqrt(group["n"])

    return group.round(1)


def _shape(profiles):
    """Pairwise correlation between the group profiles of each correction.

    A correlation at 1.0 with a non-zero mean difference means the two
    corrections differ by a constant: the distance-dependent conclusion is
    the same and only the reported magnitude changes. A correlation below
    about .95 means the shape itself depends on the choice.
    """
    group = (
        profiles
        .groupby(["correction", "speed", "position_rank"])["facilitation_ms"]
        .mean()
        .unstack("correction")
    )

    names = list(group.columns)
    rows = []

    for speed, one in group.groupby(level="speed"):
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                a = one[first].to_numpy(float)
                b = one[second].to_numpy(float)
                ok = np.isfinite(a) & np.isfinite(b)

                if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
                    correlation = np.nan
                else:
                    correlation = float(np.corrcoef(a[ok], b[ok])[0, 1])

                rows.append({
                    "speed": speed,
                    "comparison": f"{first} vs {second}",
                    "r": correlation,
                    "mean_difference_ms": float(np.mean(a[ok] - b[ok])),
                    "max_difference_ms": float(np.max(np.abs(a[ok] - b[ok]))),
                    "verdict": (
                        "same shape, different level"
                        if np.isfinite(correlation) and correlation >= 0.95
                        else "different shape"
                    ),
                })

    return pd.DataFrame(rows).round(2)


def _nearfar(profiles):
    """Near-minus-far facilitation per correction, participant and speed."""
    rows = []

    for (correction, subject, speed), one in profiles.groupby(
        ["correction", "subject", "speed"]
    ):
        near = one.loc[
            one["position_rank"].isin(NEAR), "facilitation_ms"
        ].mean()
        far = one.loc[
            one["position_rank"].isin(FAR), "facilitation_ms"
        ].mean()

        rows.append({
            "correction": correction,
            "subject": subject,
            "speed": speed,
            "near_ms": near,
            "far_ms": far,
            "nearfar_ms": near - far,
        })

    return pd.DataFrame(rows).round(1)


def _group(profiles, n_boot=None):
    """The near-far group test under each correction, speeds collapsed.

    Uses ``h1.nearfar_table`` and ``h1.group_test`` so the numbers here are
    the same ones H1 reports, not a second implementation of them.
    """
    rows = []

    for correction, one in profiles.groupby("correction"):
        nearfar = h1.nearfar_table(one)

        if nearfar.empty:
            continue

        result = h1.group_test(nearfar)
        result["correction"] = correction
        rows.append(result)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    columns = ["correction"] + [c for c in out.columns if c != "correction"]

    return out[columns].round(3)


def _boundaries(profiles):
    """Half-maximum boundary per correction, with the amplitude behind it.

    A crossing can be interpolated from any profile, including a flat one, so
    the amplitude is carried alongside: a boundary read off a profile whose
    plateaus barely differ is a number without a step under it.
    """
    rows = []

    for correction, one in profiles.groupby("correction"):
        halfmax = h1.half_max_boundary(one)

        if halfmax.empty:
            continue

        halfmax = halfmax.assign(correction=correction)
        rows.append(halfmax)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    columns = ["correction"] + [c for c in out.columns if c != "correction"]

    return out[columns].round(2)


# ==================================================================
# Report
# ==================================================================


def report(result):
    """Print the comparison in the order the questions are asked."""
    settings = result["settings"]

    print("=" * 70)
    print("Baseline corrections compared")
    print("=" * 70)
    print(
        f"{settings['n_subjects']} participants, "
        f"{settings['estimator']} cell estimator, "
        f"expectancy degree {settings['degree']}"
    )

    # Named rather than counted: a comparison between corrections is only
    # meaningful on a fixed sample, so who is in it belongs at the top of
    # the report rather than in the settings dict.
    if settings.get("subjects"):
        print(f"  {', '.join(settings['subjects'])}")

    if settings.get("excluded"):
        print(f"  excluded: {', '.join(settings['excluded'])}")

    print()

    print("1. Level: the group profile under each correction")
    print()
    levels = result["levels"]
    print(
        levels.pivot_table(
            index=["speed", "position_rank"],
            columns="correction",
            values="group_mean",
        ).round(1).to_string()
    )

    print()
    print("   range across distance (how much shape each one has):")
    print(
        levels.groupby(["correction", "speed"])["group_mean"]
        .agg(low="min", high="max", range_ms=lambda v: v.max() - v.min())
        .round(1).to_string()
    )

    print()
    print("2. Shape: do they agree about how facilitation varies with distance")
    print()
    print(result["shape"].to_string(index=False))

    print()
    print("3. Result: the near-far index under each correction")
    print()

    nearfar = result["nearfar"]
    print(
        nearfar.pivot_table(
            index=["subject", "speed"],
            columns="correction",
            values="nearfar_ms",
        ).round(1).to_string()
    )

    group = result["group"]
    if len(group):
        print()
        print("   group test, speeds collapsed:")
        print(group.to_string(index=False))

    boundaries = result["boundaries"]
    if len(boundaries):
        print()
        print("   half-maximum boundary, participants with one:")
        counted = (
            boundaries.groupby("correction")
            .agg(
                n=("subject", "size"),
                with_boundary=("crossover_rank", lambda v: int(v.notna().sum()))
                if "crossover_rank" in boundaries.columns
                else ("subject", "size"),
            )
        )
        print(counted.to_string())


# ==================================================================
# Figure
# ==================================================================


def plot_profiles(result, show_sem=True, save_as=None):
    """One panel per speed: the group profile under every correction.

    Read the vertical spacing and the shapes separately. Curves that run
    parallel differ only in level, and every distance-based conclusion is the
    same under all of them. Curves that cross, or that rise where another
    falls, mean the profile depends on the baseline rather than on the data.
    """
    levels = result["levels"]
    speeds = [s for s in ["slow", "fast"] if s in set(levels["speed"])]
    corrections = result["settings"]["corrections"]

    fig, axes = plt.subplots(
        1, max(len(speeds), 1),
        figsize=(6.0 * max(len(speeds), 1), 4.4),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    palette = _palette(corrections)

    for ax, speed in zip(axes, speeds):
        for correction in corrections:
            line = levels[
                (levels["correction"] == correction)
                & (levels["speed"] == speed)
            ].sort_values("position_rank")

            if line.empty:
                continue

            colour, linestyle, marker = palette[correction]

            if show_sem:
                ax.fill_between(
                    line["position_rank"],
                    line["group_mean"] - line["group_sem"],
                    line["group_mean"] + line["group_sem"],
                    color=colour, alpha=0.10, linewidth=0,
                )

            ax.plot(
                line["position_rank"], line["group_mean"],
                color=colour, linestyle=linestyle, marker=marker,
                markersize=5, linewidth=2.0,
                label=correction.replace("_", " "),
            )

        ax.axhline(0, linestyle=":", linewidth=1.0, color=style.MUTED)

        ranks = sorted(levels["position_rank"].dropna().unique())
        ax.set_xticks(ranks)
        ax.set_xticklabels([f"D{int(r)}" for r in ranks])
        ax.set_xlabel("Distance (D1 = closest)")
        ax.set_title(speed.capitalize(), loc="left")

        ax.grid(axis="y", color=getattr(style, "RULE", "#DDDDDD"),
                linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        style.clean(ax)

    axes[0].set_ylabel("Facilitation, T - VT (ms)")
    axes[0].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Facilitation under each tactile baseline",
        x=0.02, ha="left", color=style.INK,
    )
    style.finish_figure(fig)

    if save_as:
        figures.save(fig, save_as)

    return fig


def _condition_colour(condition):
    """The project colour for a sensory condition."""
    mapping = getattr(style, "CONDITION", None)

    if isinstance(mapping, dict) and str(condition) in mapping:
        return mapping[str(condition)]

    return getattr(style, str(condition), style.CONTROL)


def _palette(corrections):
    """A colour, linestyle and marker per correction.

    The modelled corrections get solid lines and the flat ones dashed, so the
    two kinds are distinguishable before any legend is read.
    """
    look = {
        "delay_matched": (style.SLOW, "--", "s"),
        "expectancy_model": (style.PATIENT, "-", "o"),
        "smooth": (style.CONTROL, "-", "^"),
        "overall_tactile": (style.FAST, ":", None),
        "fastest_tactile": (style.MUTED, "-.", None),
        "fastest_n_tactile": (style.INK, "-.", None),
    }

    return {
        correction: look.get(correction, (style.INK, "-", "o"))
        for correction in corrections
    }


def _order_fraction(trials, within=("subject", "speed")):
    """Position of each trial in its block of the session, scaled 0 to 1.

    Scaled within subject and speed rather than across the session, because
    the two speeds were interleaved: a raw trial number would make one speed
    look uniformly earlier than the other.
    """
    order = next(
        (c for c in ["global_trial", "trial_number", "trial", "trial_index"]
         if c in trials.columns),
        None,
    )

    if order is None:
        position = pd.Series(
            np.arange(len(trials), dtype=float), index=trials.index
        )
    else:
        position = pd.to_numeric(trials[order], errors="coerce")

    keys = [c for c in within if c in trials.columns]

    if not keys:
        span = position.max() - position.min()
        return (position - position.min()) / (span if span else np.nan)

    grouped = position.groupby([trials[c] for c in keys])
    low = grouped.transform("min")
    high = grouped.transform("max")

    return (position - low) / (high - low).replace(0, np.nan)


def _shade(colour, fraction):
    """Pale early, solid late. ``fraction`` runs 0 at the start to 1 at the end."""
    red, green, blue = mcolors.to_rgb(colour)
    lighten = 0.80 * (1.0 - float(np.clip(fraction, 0.0, 1.0)))

    return (
        red + (1.0 - red) * lighten,
        green + (1.0 - green) * lighten,
        blue + (1.0 - blue) * lighten,
    )


def plot_correction_grid(
    result,
    trials,
    corrections=None,
    show_trials=True,
    jitter=0.10,
    save_as=None,
):
    """Four rows, two columns: each correction against each speed.

    Rows are corrections, columns are speeds, so a row is read across for a
    speed effect and a column down for the effect of the baseline choice. The
    axes are shared, so a curve that sits higher really is higher.

    In every panel:

        left axis     the group facilitation profile under that correction,
                      with its standard error. This is the quantity the
                      hypotheses are about.

        right axis    the trials it was built from. Tactile-only in grey,
                      visuo-tactile in blue, each trial shaded by when it
                      arrived in that participant's run of that speed: pale
                      at the start, solid at the end. The heavier lines are
                      the group medians.

    The gradient is there to answer one question the profile cannot: whether
    a distance effect is really a time effect. If the late trials sit
    systematically below the early ones at every distance, the participant
    sped up over the session, and any correction that does not model that is
    carrying drift into the facilitation curve. If early and late are
    interleaved at every distance, the profile is about distance.

    Parameters
    ----------
    result : dict
        Output of ``compare``.
    trials : DataFrame
        The same trials passed to ``compare``, for the overlay.
    corrections : sequence or None
        Which rows to draw, in order. None uses everything in ``result``.
    show_trials : bool
        Draw the individual trials. Turn off for a clean four-panel profile
        comparison.
    """
    levels = result["levels"]

    corrections = (
        result["settings"]["corrections"] if corrections is None
        else list(corrections)
    )
    speeds = [s for s in ["slow", "fast"] if s in set(levels["speed"])]

    ranks = sorted(levels["position_rank"].dropna().unique())
    palette = _palette(corrections)

    data = trials[trials["sensory_condition"].isin(["T", "VT"])].copy()
    data = data[data["rt_ms"].notna()]
    data["order_fraction"] = _order_fraction(data)

    rt_limits = (
        float(np.nanpercentile(data["rt_ms"], 1)),
        float(np.nanpercentile(data["rt_ms"], 99)),
    )

    fig, axes = plt.subplots(
        len(corrections), max(len(speeds), 1),
        figsize=(6.4 * max(len(speeds), 1), 3.0 * len(corrections)),
        sharex=True, sharey=True, squeeze=False,
    )

    rng = np.random.default_rng(7)
    offsets = {"T": -0.16, "VT": 0.16}

    for row, correction in enumerate(corrections):
        for column, speed in enumerate(speeds):
            ax = axes[row, column]

            # --- the trials, on their own axis --------------------
            if show_trials:
                twin = ax.twinx()
                twin.set_ylim(rt_limits)

                at_speed = data[data["speed"] == speed]

                for condition in ["T", "VT"]:
                    colour = _condition_colour(condition)
                    cell = at_speed[
                        at_speed["sensory_condition"] == condition
                    ]

                    if cell.empty:
                        continue

                    x = (
                        cell["position_rank"].to_numpy(float)
                        + offsets[condition]
                        + rng.normal(0, jitter, len(cell))
                    )

                    twin.scatter(
                        x,
                        cell["rt_ms"],
                        s=6,
                        c=[
                            _shade(colour, f)
                            for f in cell["order_fraction"].fillna(0.5)
                        ],
                        alpha=0.45,
                        edgecolors="none",
                        zorder=1,
                    )

                    median = (
                        cell.groupby("position_rank")["rt_ms"]
                        .median().reindex(ranks)
                    )

                    twin.plot(
                        ranks,
                        median.to_numpy(float),
                        color=_shade(colour, 1.0),
                        linewidth=1.4,
                        linestyle="--" if condition == "T" else "-",
                        alpha=0.85,
                        zorder=2,
                    )

                twin.set_ylabel(
                    "RT (ms)" if column == len(speeds) - 1 else "",
                    color=style.MUTED, fontsize=8,
                )
                twin.tick_params(axis="y", labelsize=7, colors=style.MUTED)

                if column != len(speeds) - 1:
                    twin.set_yticklabels([])

                style.clean(twin)

            # --- the facilitation profile -------------------------
            line = levels[
                (levels["correction"] == correction)
                & (levels["speed"] == speed)
            ].sort_values("position_rank")

            colour, linestyle, marker = palette[correction]

            if not line.empty:
                ax.fill_between(
                    line["position_rank"],
                    line["group_mean"] - line["group_sem"],
                    line["group_mean"] + line["group_sem"],
                    color=colour, alpha=0.15, linewidth=0, zorder=3,
                )
                ax.plot(
                    line["position_rank"], line["group_mean"],
                    color=colour, linestyle=linestyle, marker=marker,
                    markersize=5, linewidth=2.4, zorder=4,
                )

            ax.axhline(0, linestyle=":", linewidth=1.0, color=style.MUTED,
                       zorder=2)
            ax.set_zorder(2)
            ax.patch.set_visible(False)

            ax.set_xticks(ranks)
            ax.set_xticklabels([f"D{int(r)}" for r in ranks])

            if row == 0:
                ax.set_title(speed.capitalize(), loc="left")

            if column == 0:
                ax.set_ylabel(
                    f"{correction.replace('_', ' ')}\nT - VT (ms)",
                    fontsize=9,
                )

            if row == len(corrections) - 1:
                ax.set_xlabel("Distance (D1 = closest)")

            style.clean(ax)

    fig.suptitle(
        "Facilitation under each baseline, over the trials behind it "
        "(pale = early in that speed, solid = late)",
        x=0.02, ha="left", color=style.INK,
    )
    style.finish_figure(fig)

    if save_as:
        figures.save(fig, save_as)

    return fig


# Facilitation is one quantity, so it keeps one colour everywhere and the
# corrections are told apart by their marker. Reading four hues as four
# baselines is work the shape of a point can do instead.
FACILITATION_COLOUR = "#0097b2"

CORRECTION_MARKER = {
    "delay_matched": "o",
    "expectancy_model": "s",
    "smooth": "^",
    "overall_tactile": "D",
    "fastest_tactile": "v",
    "fastest_n_tactile": "P",
    "noel_endpoints": "X",
}

GROUP_LABEL = "Group"


def _baseline_column(correction):
    """The column holding the baseline that correction subtracts."""
    return {
        "delay_matched": "T",
        "expectancy_model": "tactile_baseline_ms",
        "smooth": "tactile_baseline_ms",
        "overall_tactile": "tactile_baseline_ms",
        "fastest_tactile": "tactile_baseline_ms",
        "fastest_n_tactile": "tactile_baseline_ms",
        "noel_endpoints": "tactile_baseline_ms",
    }.get(correction, "tactile_baseline_ms")


def plot_correction_by_subject(
    result,
    trials,
    corrections=None,
    subjects=None,
    speed=None,
    show_trials=True,
    jitter=0.10,
    panel_size=(2.5, 2.4),
    save_as=None,
):
    """One row per correction, one column per participant, group last.

    Every panel shows the same four things, so a column is read down for the
    effect of the baseline choice on one participant, and a row across for
    who that choice matters to.

        grey, dashed    the tactile baseline that correction subtracts. Flat
                        for overall_tactile and fastest_tactile, sloped
                        wherever the correction models delay. This is the
                        line that differs between rows; everything else in
                        the panel is identical down the column.

        grey, solid     the tactile-only median at each distance, with the
                        trials behind it. Where the dashed line departs from
                        the solid one, the correction is replacing the cell
                        with something else.

        blue, solid     the visuo-tactile median, with its trials. This never
                        changes between rows.

        #0097b2         facilitation, on the right axis: the gap between the
                        dashed baseline and the visuo-tactile median. The
                        marker names the correction.

    The last column is the group, and its points are the participant medians
    rather than trials, so the same panel logic holds: the points are always
    what the line was computed from.

    Parameters
    ----------
    speed : {"slow", "fast", None}
        None pools the speeds, which is the compact view. Name one to keep
        the speed contexts apart, which is how the hypotheses are stated.
    """
    profiles = result["profiles"]

    corrections = (
        result["settings"]["corrections"] if corrections is None
        else list(corrections)
    )

    # The trials frame is passed in whole, so it still holds anyone
    # compare() excluded. Restricting it to the participants actually in
    # the profiles keeps the figure and the report describing the same
    # sample; otherwise a panel appears for a participant who contributed
    # to none of the numbers beside it.
    trials = trials[trials["subject"].isin(profiles["subject"].unique())]

    data = trials[trials["sensory_condition"].isin(["T", "VT"])].copy()
    data = data[data["rt_ms"].notna()]

    if speed is not None:
        data = data[data["speed"] == speed]
        profiles = profiles[profiles["speed"] == speed]

    people = (
        sorted(data["subject"].dropna().unique()) if subjects is None
        else list(subjects)
    )
    columns = people + [GROUP_LABEL]

    ranks = sorted(profiles["position_rank"].dropna().unique())

    data["order_fraction"] = _order_fraction(data)

    rt_limits = (
        float(np.nanpercentile(data["rt_ms"], 1)),
        float(np.nanpercentile(data["rt_ms"], 99)),
    )

    fig, axes = plt.subplots(
        len(corrections), len(columns),
        figsize=(panel_size[0] * len(columns),
                 panel_size[1] * len(corrections)),
        sharex=True, sharey=True, squeeze=False,
    )

    rng = np.random.default_rng(11)
    offsets = {"T": -0.15, "VT": 0.15}

    facilitation_limits = _facilitation_limits(profiles)

    for row, correction in enumerate(corrections):
        one_correction = profiles[profiles["correction"] == correction]
        baseline_column = _baseline_column(correction)

        for column, name in enumerate(columns):
            ax = axes[row, column]
            ax.set_ylim(rt_limits)

            is_group = name == GROUP_LABEL

            if is_group:
                cells = (
                    one_correction
                    .groupby(["subject", "position_rank"], as_index=False)
                    .mean(numeric_only=True)
                )
                panel = (
                    cells.groupby("position_rank", as_index=False)
                    .mean(numeric_only=True)
                )
                points = cells                      # one point per subject
            else:
                panel = (
                    one_correction[one_correction["subject"] == name]
                    .groupby("position_rank", as_index=False)
                    .mean(numeric_only=True)
                )
                points = None

            panel = panel.sort_values("position_rank")

            # --- the trials, or the subject medians in the group column ---
            if show_trials:
                for condition in ["T", "VT"]:
                    colour = _condition_colour(condition)

                    if is_group:
                        if points is None or condition not in points:
                            continue

                        x = (
                            points["position_rank"].to_numpy(float)
                            + offsets[condition]
                            + rng.normal(0, jitter, len(points))
                        )
                        ax.scatter(
                            x, points[condition],
                            s=9, color=colour, alpha=0.55,
                            edgecolors="none", zorder=2,
                        )
                        continue

                    cell = data[
                        data["subject"].eq(name)
                        & data["sensory_condition"].eq(condition)
                    ]

                    if cell.empty:
                        continue

                    x = (
                        cell["position_rank"].to_numpy(float)
                        + offsets[condition]
                        + rng.normal(0, jitter, len(cell))
                    )
                    ax.scatter(
                        x, cell["rt_ms"],
                        s=5,
                        c=[
                            _shade(colour, f)
                            for f in cell["order_fraction"].fillna(0.5)
                        ],
                        alpha=0.40, edgecolors="none", zorder=1,
                    )

            # --- the two RT medians ---------------------------------
            for condition in ["T", "VT"]:
                if condition not in panel:
                    continue

                ax.plot(
                    panel["position_rank"], panel[condition],
                    color=_shade(_condition_colour(condition), 1.0),
                    linewidth=1.4,
                    marker="o", markersize=3,
                    zorder=4,
                )

            # --- the baseline this correction subtracts ---------------
            if baseline_column in panel:
                ax.plot(
                    panel["position_rank"], panel[baseline_column],
                    color=style.MUTED, linestyle="--", linewidth=1.8,
                    zorder=5,
                )

            # --- facilitation, on its own axis ------------------------
            twin = ax.twinx()
            twin.set_ylim(facilitation_limits)

            twin.axhline(
                0, linestyle=":", linewidth=0.8, color=style.MUTED, zorder=1,
            )
            twin.plot(
                panel["position_rank"], panel["facilitation_ms"],
                color=FACILITATION_COLOUR,
                marker=CORRECTION_MARKER.get(correction, "o"),
                markersize=5, linewidth=2.2, zorder=6,
            )

            twin.tick_params(
                axis="y", labelsize=7, colors=FACILITATION_COLOUR,
            )

            if column != len(columns) - 1:
                twin.set_yticklabels([])
            else:
                twin.set_ylabel(
                    "T - VT (ms)", fontsize=8, color=FACILITATION_COLOUR,
                )

            style.clean(twin)

            ax.set_xticks(ranks)
            ax.set_xticklabels([f"D{int(r)}" for r in ranks], fontsize=7)
            ax.tick_params(axis="y", labelsize=7)

            if row == 0:
                ax.set_title(
                    name, loc="left", fontsize=9,
                    fontweight="bold" if is_group else "normal",
                )

            if column == 0:
                ax.set_ylabel(
                    f"{correction.replace('_', ' ')}\nRT (ms)", fontsize=8,
                )
            else:
                ax.set_yticklabels([])

            if row == len(corrections) - 1:
                ax.set_xlabel("Distance", fontsize=8)

            ax.set_zorder(2)
            ax.patch.set_visible(False)
            style.clean(ax)

    _grid_legend(fig, corrections)

    speed_label = "speeds pooled" if speed is None else f"{speed} only"
    fig.suptitle(
        "Each baseline, the trials behind it, and the facilitation it "
        f"produces ({speed_label})",
        x=0.02, ha="left", color=style.INK,
    )
    style.finish_figure(fig)

    if save_as:
        figures.save(fig, save_as)

    return fig


def _facilitation_limits(profiles, pad=0.08):
    """One facilitation range for every panel, so heights are comparable."""
    values = profiles["facilitation_ms"].dropna()

    if values.empty:
        return (-50.0, 150.0)

    low, high = float(values.min()), float(values.max())
    span = high - low if high > low else 1.0

    return low - pad * span, high + pad * span


def _grid_legend(fig, corrections):
    """One legend for the whole grid, naming what each line is."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=style.MUTED, linestyle="--", linewidth=1.8,
               label="baseline subtracted"),
        Line2D([], [], color=_shade(_condition_colour("T"), 1.0),
               marker="o", markersize=3, label="tactile-only median"),
        Line2D([], [], color=_shade(_condition_colour("VT"), 1.0),
               marker="o", markersize=3, label="visuo-tactile median"),
    ]

    handles += [
        Line2D([], [], color=FACILITATION_COLOUR,
               marker=CORRECTION_MARKER.get(correction, "o"),
               markersize=5, linewidth=2.0,
               label=f"facilitation, {correction.replace('_', ' ')}")
        for correction in corrections
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 4),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )


def plot_group_corrections(
    result,
    corrections=None,
    speed=None,
    show_points=True,
    jitter=0.10,
    panel_size=(3.4, 3.0),
    save_as=None,
):
    """The group panel on its own: one panel per correction, no participants.

    The last column of ``plot_correction_by_subject``, drawn large enough to
    read. Same four elements, same meaning:

        grey, dashed    the baseline that correction subtracts, averaged
                        across participants
        grey, solid     the tactile-only median
        blue, solid     the visuo-tactile median
        #0097b2         facilitation, on the right axis

    The points are the participant values, one per person per distance, since
    that is what each group line was computed from. Their spread is the
    between-participant variability the standard error summarises, shown
    rather than described.

    Parameters
    ----------
    result : dict
        Output of ``compare``.
    corrections : sequence or None
        Which panels to draw, in order. None uses everything in ``result``.
    speed : {"slow", "fast", None}
        None pools the speeds.
    show_points : bool
        Draw the participant values behind each line.
    """
    profiles = result["profiles"]

    corrections = (
        result["settings"]["corrections"] if corrections is None
        else list(corrections)
    )

    if speed is not None:
        profiles = profiles[profiles["speed"] == speed]

    ranks = sorted(profiles["position_rank"].dropna().unique())

    fig, axes = plt.subplots(
        1, len(corrections),
        figsize=(panel_size[0] * len(corrections), panel_size[1]),
        sharex=True, sharey=True, squeeze=False,
    )
    axes = axes[0]

    rng = np.random.default_rng(13)
    offsets = {"T": -0.15, "VT": 0.15}

    rt_values = pd.concat([
        profiles[column] for column in ["T", "VT", "tactile_baseline_ms"]
        if column in profiles.columns
    ]).dropna()

    rt_limits = (
        (float(np.nanpercentile(rt_values, 1)),
         float(np.nanpercentile(rt_values, 99)))
        if len(rt_values) else (200.0, 700.0)
    )

    facilitation_limits = _facilitation_limits(profiles)

    for ax, correction in zip(axes, corrections):
        one = profiles[profiles["correction"] == correction]
        baseline_column = _baseline_column(correction)

        # One value per participant and distance, speeds averaged if pooled.
        per_subject = (
            one.groupby(["subject", "position_rank"], as_index=False)
            .mean(numeric_only=True)
        )
        group = (
            per_subject.groupby("position_rank", as_index=False)
            .mean(numeric_only=True)
            .sort_values("position_rank")
        )

        ax.set_ylim(rt_limits)

        # --- the participant values behind each line ---------------
        if show_points:
            for condition in ["T", "VT"]:
                if condition not in per_subject:
                    continue

                x = (
                    per_subject["position_rank"].to_numpy(float)
                    + offsets[condition]
                    + rng.normal(0, jitter, len(per_subject))
                )

                ax.scatter(
                    x, per_subject[condition],
                    s=14, color=_condition_colour(condition),
                    alpha=0.40, edgecolors="none", zorder=1,
                )

        # --- the two RT medians ------------------------------------
        for condition in ["T", "VT"]:
            if condition not in group:
                continue

            ax.plot(
                group["position_rank"], group[condition],
                color=_shade(_condition_colour(condition), 1.0),
                linewidth=1.8, marker="o", markersize=4, zorder=4,
                label=(
                    "tactile-only median" if condition == "T"
                    else "visuo-tactile median"
                ),
            )

        # --- the baseline this correction subtracts ----------------
        if baseline_column in group:
            ax.plot(
                group["position_rank"], group[baseline_column],
                color=style.MUTED, linestyle="--", linewidth=2.0,
                zorder=5, label="baseline subtracted",
            )

        # --- facilitation ------------------------------------------
        twin = ax.twinx()
        twin.set_ylim(facilitation_limits)

        twin.axhline(0, linestyle=":", linewidth=0.8, color=style.MUTED,
                     zorder=1)

        if show_points:
            x = (
                per_subject["position_rank"].to_numpy(float)
                + rng.normal(0, jitter, len(per_subject))
            )
            twin.scatter(
                x, per_subject["facilitation_ms"],
                s=14, color=FACILITATION_COLOUR, alpha=0.35,
                edgecolors="none", zorder=2,
            )

        twin.plot(
            group["position_rank"], group["facilitation_ms"],
            color=FACILITATION_COLOUR,
            marker=CORRECTION_MARKER.get(correction, "o"),
            markersize=6, linewidth=2.6, zorder=6,
            label="facilitation",
        )

        twin.tick_params(axis="y", labelsize=8, colors=FACILITATION_COLOUR)

        if correction != corrections[-1]:
            twin.set_yticklabels([])
        else:
            twin.set_ylabel("T - VT (ms)", color=FACILITATION_COLOUR)

        style.clean(twin)

        ax.set_xticks(ranks)
        ax.set_xticklabels([f"D{int(r)}" for r in ranks])
        ax.set_xlabel("Distance (D1 = closest)")
        ax.set_title(correction.replace("_", " "), loc="left")

        if correction == corrections[0]:
            ax.set_ylabel("RT (ms)")
        else:
            ax.set_yticklabels([])

        ax.set_zorder(2)
        ax.patch.set_visible(False)
        style.clean(ax)

    # One legend, outside the panels. Inside, it sits on the lines it
    # describes: the two RT medians and the facilitation curve run across
    # the full width, so every interior corner is occupied.
    handles, labels = axes[0].get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()

    fig.legend(
        handles + twin_handles, labels + twin_labels,
        frameon=False, fontsize=8, loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )

    n_subjects = int(profiles["subject"].nunique())
    speed_label = "speeds pooled" if speed is None else f"{speed} only"

    fig.suptitle(
        f"Group facilitation under each baseline "
        f"({n_subjects} participants, {speed_label}; "
        "points are participant values)",
        x=0.02, ha="left", color=style.INK,
    )
    style.finish_figure(fig)

    # tight_layout does not account for a figure legend, so the room has
    # to be reserved after it runs.
    fig.subplots_adjust(right=1.0 - min(0.30 / len(corrections), 0.26))

    if save_as:
        figures.save(fig, save_as)

    return fig


def plot_facilitation(
    result,
    subject=None,
    corrections=None,
    show_sem=True,
    panel_size=(5.2, 4.0),
    save_as=None,
):
    """Facilitation alone: every correction on one axis, one panel per speed.

    No reaction times, no trials. The question here is narrower than in the
    other figures: given the same visuo-tactile medians, how much does the
    curve depend on which baseline was subtracted from them.

    Curves that run parallel differ only in level, so every distance-based
    conclusion is the same under all of them and only the magnitude moves.
    Curves that cross, or that fall where another rises, mean the shape
    itself is a property of the baseline.

    The three flat corrections cannot produce a shape of their own: their
    facilitation profile is the visuo-tactile profile inverted and shifted.
    Where they disagree with delay_matched or expectancy_model about the
    direction of the curve, the disagreement is the delay dependence.

    Parameters
    ----------
    subject : str or None
        None draws the group mean with its standard error. Naming a
        participant draws that person, which is how a single reversal is
        traced to the correction that produced it.
    """
    profiles = result["profiles"]

    corrections = (
        result["settings"]["corrections"] if corrections is None
        else list(corrections)
    )

    if subject is not None:
        profiles = profiles[profiles["subject"] == subject]

        if profiles.empty:
            raise KeyError(f"no profile for {subject!r}")

    speeds = [s for s in ["slow", "fast"] if s in set(profiles["speed"])]
    ranks = sorted(profiles["position_rank"].dropna().unique())

    fig, axes = plt.subplots(
        1, max(len(speeds), 1),
        figsize=(panel_size[0] * max(len(speeds), 1), panel_size[1]),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    for ax, speed in zip(axes, speeds):
        at_speed = profiles[profiles["speed"] == speed]

        for correction in corrections:
            one = at_speed[at_speed["correction"] == correction]

            if one.empty:
                continue

            per_subject = (
                one.groupby(["subject", "position_rank"], as_index=False)
                ["facilitation_ms"].mean()
            )
            line = (
                per_subject.groupby("position_rank", as_index=False)
                ["facilitation_ms"]
                .agg(mean="mean", sd="std", n="count")
                .sort_values("position_rank")
            )
            line["sem"] = line["sd"] / np.sqrt(line["n"])

            if show_sem and subject is None and line["n"].max() > 1:
                ax.fill_between(
                    line["position_rank"],
                    line["mean"] - line["sem"],
                    line["mean"] + line["sem"],
                    color=FACILITATION_COLOUR, alpha=0.08, linewidth=0,
                )

            ax.plot(
                line["position_rank"], line["mean"],
                color=FACILITATION_COLOUR,
                marker=CORRECTION_MARKER.get(correction, "o"),
                markersize=6, linewidth=2.2,
                label=correction.replace("_", " "),
            )

        ax.axhline(0, linestyle=":", linewidth=1.0, color=style.MUTED)

        ax.set_xticks(ranks)
        ax.set_xticklabels([f"D{int(r)}" for r in ranks])
        ax.set_xlabel("Distance (D1 = closest)")
        ax.set_title(speed.capitalize(), loc="left")

        ax.grid(axis="y", color=getattr(style, "RULE", "#DDDDDD"),
                linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        style.clean(ax)

    axes[0].set_ylabel("Facilitation, T - VT (ms)")
    axes[0].legend(frameon=False, fontsize=8)

    who = (
        f"{int(profiles['subject'].nunique())} participants"
        if subject is None else str(subject)
    )
    fig.suptitle(
        f"Facilitation under each baseline ({who})",
        x=0.02, ha="left", color=style.INK,
    )
    style.finish_figure(fig)

    if save_as:
        figures.save(fig, save_as)

    return fig


def nearfar_by_correction(result, subjects=None):
    """Near-far index per participant, one column per correction.

    The compact form of the comparison. A participant whose sign flips
    between columns has a result that depends on the baseline, and the
    corrections it flips between say which property of the baseline is
    responsible: the three flat ones discard the delay dependence, the two
    modelled ones keep it.
    """
    nearfar = result["nearfar"]

    if subjects is not None:
        nearfar = nearfar[nearfar["subject"].isin(list(subjects))]

    table = nearfar.pivot_table(
        index=["subject", "speed"],
        columns="correction",
        values="nearfar_ms",
    ).round(1)

    ordered = [
        c for c in result["settings"]["corrections"] if c in table.columns
    ]

    return table[ordered]