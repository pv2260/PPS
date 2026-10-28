"""
H3. The collision boundary shifts when the ball moves faster.

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

THE FIGURES BELOW THE TEST
--------------------------
H3 reads a boundary off P(HIT) against offset, so the diagnostics describe
that function rather than accuracy summed over offsets. Accuracy pooled
across offsets sits near ceiling by construction, since most trials are far
from the boundary and those are easy.

Every reaction-time figure splits slow from fast. A slow ball is in flight
more than twice as long as a fast one, so a slow trial cannot produce a
short RT even from an identical decision. Pooling the speeds within a trial
category mixes a decision effect with a stimulus-timing one, and the timing
effect is the larger of the two.

TWO WAYS OF SCORING ACCURACY
----------------------------
``plot_collision_indices`` draws either, chosen with ``measure``.

Proportion correct is what the preregistration specifies and what most
readers expect. Its weakness is that it collapses two different things: a
participant who discriminates poorly and one who discriminates well but
answers MISS too readily can score the same.

Signal detection theory separates them. Treating a HIT response as the
response and a would-have-hit trial as signal present:

    d' = z(hit rate) - z(false alarm rate)
    c  = -0.5 * [ z(hit rate) + z(false alarm rate) ]

d' is how sharply the participant separates hits from misses, which is the
same thing the psychometric slope measures. c is where their decision point
sits: positive c means they lean toward answering MISS, so their judged
boundary sits closer to the body, which is the same displacement Delta_coll
measures between speeds. The two frameworks describe one behaviour, and c
is worth reporting wherever a participant's near-hit and near-miss accuracy
diverge, since proportion correct averages that divergence away.

Rates of 0 and 1 have no finite z, and the clear trials reach both. The
loglinear correction is applied throughout: 0.5 is added to each response
count and 1 to each trial count before the rates are taken.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import expit

import style

from .. import config
from .. import figures
from ..stats_utils import bootstrap_mean_ci, wilcoxon_one_sided
from ..collision import (
    code_speed,
    safe_logistic_regression,
    compute_collision_indices_one_session,
)


# ==================================================================
# Constants
# ==================================================================

OFFSET_COLUMN = "offset_from_edge_cm"
RESPONSE_COLUMN = "response_hit"

# Trials nearer the boundary than this are the ones where the judgement
# is genuinely hard, and where a speed effect has to appear.
NEAR_BOUNDARY_CM = 5.0

# Trial categories, from deepest inside the body edge to furthest outside.
TRIAL_CATEGORIES = ["clear_hit", "near_hit", "near_miss", "clear_miss"]

# The two that carry the judgement.
NEAR_CATEGORIES = ["near_hit", "near_miss"]

# Display names. The underlying values stay lowercase with underscores;
# only what the reader sees is capitalised.
CATEGORY_LABEL = {
    "clear_hit": "Clear Hit",
    "near_hit": "Near Hit",
    "near_miss": "Near Miss",
    "clear_miss": "Clear Miss",
}

SPEED_LABEL = {"slow": "Slow", "fast": "Fast"}

STAGES = ["before", "after"]
STAGE_LABEL = {"before": "Before", "after": "After"}

# Fill is the variable's colour at low alpha, the median line is the same
# colour at full strength, and the points sit over both without an edge.
FILL_ALPHA = 0.25
POINT_ALPHA = 0.85

# Bins for the response map, in cm outward from the body edge, labelled
# by the outer edge of each bin.
MAP_BIN_WIDTH = 5.0
MAP_MAX_CM = 30.0

# Signed offset range for the binned figures, in cm from the body edge.
# Negative is inside the edge, where the ball would have hit.
SIGNED_MIN_CM = -25.0
SIGNED_MAX_CM = 40.0

# Upper limit of the RT axis in the inspection figure, per speed. A few
# pre-exclusion trials run to tens of seconds; scaling to the maximum
# compresses every real trial into the leftmost tenth of the axis.
RT_AXIS_CEILING_MS = {"fast": 3000.0, "slow": 5000.0}

# The four collision outcomes, in the project's colours. Read from the
# individual names so the map works whether or not style also defines a
# TRIAL_TYPE dict.
CATEGORY_COLOUR = {
    "clear_hit": getattr(style, "HIT", "#f4792f"),
    "near_hit": getattr(style, "NEAR_HIT", "#fdd3b5"),
    "near_miss": getattr(style, "NEAR_MISS", "#cdc4e0"),
    "clear_miss": getattr(style, "MISS", "#8971b0"),
}

 
# Upper limit of the per-participant RT axis, per speed. None leaves the
# axis to the data.
RT_SUBJECT_CEILING_MS = {"fast": 1500.0, "slow": None}
 
# The indices figure holds one measure per panel, so nothing is encoded
# by colour and everything is drawn in this grey.
INDEX_GREY = getattr(style, "MUTED", "#8A8A8A")


# ==================================================================
# Helpers
# ==================================================================

def _label_category(category):
    return CATEGORY_LABEL.get(category, str(category).replace("_", " ").title())


def _label_speed(speed):
    return SPEED_LABEL.get(speed, str(speed).title())


def _category_colour(category):
    return CATEGORY_COLOUR.get(category, style.CONTROL)


def _display_name(subject, patient=None, controls=None):
    """What a person is called on a figure.

    The patient is "Patient" and a matched control is "Age-matched
    control", because a reader of the figure needs to know the role, not
    the identifier. Which specific control it was belongs in the caption
    or the text, where it can be stated once.

    With more than one control the ids are kept, since "Age-matched
    control" twice would be worse than useless.
    """
    # Without a patient in the figure there is no role to name, so the
    # ids stand. This is what lets the same functions draw the young
    # controls: calling nine of them "Age-matched control (P002)" and so
    # on is both wrong and unreadable.
    if patient is None:
        return str(subject)

    if subject == patient:
        return "Patient"

    if controls and subject in controls:
        if len(controls) == 1:
            return "Age-matched control"

        return f"Age-matched control ({subject})"

    return str(subject)


def _tidy_collision(trials, offset_column=OFFSET_COLUMN,
                    response_column=RESPONSE_COLUMN):
    """The columns these figures need, tidied once. No usable filter.

    The trial category is read from the data rather than reconstructed
    from the offset. Deriving it from the sign of the offset assumes a
    convention, and getting that assumption wrong silently empties two of
    the four categories instead of raising.
    """
    d = trials.copy()

    d = d.dropna(subset=["subject", offset_column, response_column, "speed"])

    d["offset_cm"] = pd.to_numeric(d[offset_column], errors="coerce")
    d["said_hit"] = d[response_column].astype(float)

    # Distance outward from the boundary, so the axis starts at the
    # boundary and increases away from the body whichever side the ball
    # passed on.
    d["distance_cm"] = d["offset_cm"].abs()
    d["near_boundary"] = d["distance_cm"] <= NEAR_BOUNDARY_CM

    if "correct_response" in d.columns:
        d["was_hit"] = (
            d["correct_response"].astype("string").str.upper().eq("HIT")
        )
    else:
        d["was_hit"] = np.nan

    source = next(
        (c for c in ["trial_type", "trial_category", "collision_type"]
         if c in d.columns),
        None,
    )

    if source is None:
        raise KeyError(
            "No trial category column found. Looked for 'trial_type', "
            "'trial_category' and 'collision_type'. The four categories "
            "cannot be reconstructed from the offset without knowing the "
            "sign convention, so they have to come from the data."
        )

    d["trial_category"] = (
        d[source].astype("string").str.strip().str.lower()
    )

    if d["was_hit"].isna().all():
        # Fall back to the category, which does say which side the ball
        # passed on.
        d["was_hit"] = d["trial_category"].isin(["clear_hit", "near_hit"])

    d["correct"] = d["said_hit"].astype(bool) == d["was_hit"].astype(bool)

    if "rt_ms" in d.columns:
        d["rt_ms"] = pd.to_numeric(d["rt_ms"], errors="coerce")

    if "usable" not in d.columns:
        d["usable"] = True

    d["usable"] = d["usable"].fillna(False).astype(bool)

    return d


def _collision_frame(trials, offset_column=OFFSET_COLUMN,
                     response_column=RESPONSE_COLUMN):
    """Tidied collision trials, unusable ones removed.

    This is what every figure except the inspection one wants: the
    trials the analysis will actually use.
    """
    d = _tidy_collision(
        trials, offset_column=offset_column, response_column=response_column,
    )

    return d[d["usable"]]


def collision_trials_at_stage(d, stage="before"):
    """Every trial, or only the ones that survived the exclusion rules.

    ``d`` is the output of ``_tidy_collision``, which keeps everything.
    The inspection figure needs both stages from one frame so the two can
    share an axis range.
    """
    if stage == "before":
        return d

    if stage == "after":
        return d[d["usable"]]

    raise ValueError(f"stage must be 'before' or 'after', got {stage!r}")


def _speeds(d):
    order = [config.SLOW_LABEL, config.FAST_LABEL]
    return [s for s in order if s in set(d["speed"].dropna())]


def _categories(d):
    return [c for c in TRIAL_CATEGORIES if c in set(d["trial_category"])]


def _offset_bins(d, width=2.0):
    """Fixed-width bins of unsigned distance from the boundary.

    Kept for anything outside this module that still calls it. Nothing in
    here does: every figure moved to ``_signed_bins`` because folding the
    two sides of the boundary onto one axis mixes trials that would have
    hit with trials that would have missed, and the design does not
    sample those evenly.
    """
    high = np.ceil(d["distance_cm"].max() / width) * width

    edges = np.arange(0.0, high + width, width)
    centres = 0.5 * (edges[:-1] + edges[1:])

    which = np.clip(
        np.digitize(d["distance_cm"], edges) - 1, 0, len(centres) - 1,
    )

    return centres, which


def _signed_bins(d, width=5.0, minimum=None, maximum=None):
    """Fixed-width bins of signed offset from the body edge.

    Zero is the edge. Negative bins hold balls that would have hit,
    positive ones balls that would have missed, so the two sides never
    average together.

    ``distance_cm`` folds those two cases onto one axis, and the design
    does not sample them evenly: most bands hold one side only, and the
    one band that holds both drops below its neighbours whenever a
    participant is more accurate on one side than the other. That dip is
    the fold, not behaviour. Signed, the asymmetry shows as an asymmetry.

    An edge is placed exactly at zero, so no bin straddles the boundary
    and every bin belongs to one side. Offsets outside the range fall
    into the end bins rather than opening bins holding a handful of
    trials.
    """
    # A range taken from the data keeps the axis to the offsets that were
    # actually shown. Fixed defaults would leave empty bins at both ends
    # whenever a figure is restricted to a subset of the trial types.
    if minimum is None:
        minimum = np.floor(float(d["offset_cm"].min()) / width) * width

    if maximum is None:
        maximum = np.ceil(float(d["offset_cm"].max()) / width) * width

    below = np.arange(0.0, minimum - width, -width)[::-1]
    above = np.arange(0.0, maximum + width, width)

    edges = np.unique(np.concatenate([below, above]))
    centres = 0.5 * (edges[:-1] + edges[1:])

    which = np.clip(
        np.digitize(d["offset_cm"], edges) - 1, 0, len(centres) - 1,
    )

    return centres, which


def _map_bins(d, width=MAP_BIN_WIDTH, minimum=None, maximum=None):
    """Signed bins for the response map, labelled by their outer edge.

    Labelled by the edge further from zero, signed, so a label says which
    side of the body edge its band sits on and how far out it reaches.
    """
    centres, which = _signed_bins(
        d, width=width, minimum=minimum, maximum=maximum,
    )

    edges = np.concatenate([centres - width / 2, [centres[-1] + width / 2]])

    labels = []

    for low, high in zip(edges[:-1], edges[1:]):
        outer = low if abs(low) > abs(high) else high
        labels.append(f"{outer:+.0f}")

    return labels, which


def _edge_position(labels):
    """Where the body edge falls on an axis of signed bin labels.

    Between the last negative bin and the first positive one, which is
    where the ball stops being one that would have hit.
    """
    signs = [1 if label.startswith("+") else -1 for label in labels]

    for index in range(len(signs) - 1):
        if signs[index] < 0 <= signs[index + 1]:
            return index + 0.5

    return -0.5


def _draw_signed_accuracy(ax, d, keys, bin_width, min_trials,
                          by_speed=True, patient=None, controls=None,
                          categories=None, ball_radius_cm=0.0):
    """Proportion correct against signed offset.

    ``by_speed`` draws one line per speed averaged across participants,
    which is the group figure. Otherwise one line per person, which is
    the clinical one.

    The point is the mean across participants of each participant's own
    proportion in that bin, so a participant with more trials does not
    count for more. A bin holding fewer than ``min_trials`` trials for a
    participant is dropped for that participant.

    The quantity is P(respond HIT), not proportion correct. Proportion
    correct changes meaning at zero: left of it a correct answer is HIT,
    right of it a correct answer is MISS, so the line jumps at the
    boundary and the jump is an artefact of the scoring rather than
    anything the participant did. P(HIT) has no such break. It falls
    monotonically from near 1 to near 0, and where it crosses 0.5 is that
    person's judged boundary, so the vertical gap between two people at
    the body edge is the difference between where they place it.
    """
    # Clear hits and clear misses sit at ceiling several bins from the
    # boundary, so including them stretches the axis and flattens the
    # part of it where the judgement is actually made. The default is the
    # two near categories; pass ``categories=TRIAL_CATEGORIES`` for all.
    if categories is None:
        categories = NEAR_CATEGORIES

    d = d[d["trial_category"].isin(categories)].copy()

    if not len(d):
        ax.text(
            0.5, 0.5, "no trials in these categories",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=8, color=style.MUTED,
        )
        return

    centres, which = _signed_bins(d, width=bin_width)

    d["offset_bin"] = which

    drawn_low, drawn_high = np.inf, -np.inf

    for key in keys:
        if by_speed:
            subset = d[d["speed"] == key]
            colour = style.SPEED[key]
            label = _label_speed(key)
        else:
            subset = d[d["subject"] == key]
            colour = _clinical_colour(key, patient, controls)
            label = _display_name(key, patient, controls)

        cell = (
            subset.groupby(["subject", "offset_bin"])["said_hit"]
            .agg(["mean", "size"])
        )

        cell = cell[cell["size"] >= min_trials]

        if not len(cell):
            continue

        binned = cell.groupby("offset_bin")["mean"].mean()
        counts = cell.groupby("offset_bin")["size"].sum()

        x = centres[binned.index]

        drawn_low = min(drawn_low, float(x.min()))
        drawn_high = max(drawn_high, float(x.max()))

        ax.plot(
            x, binned,
            marker="o", markersize=4, linewidth=1.8,
            color=colour, label=label, zorder=3,
        )

        # Marker area tracks the trials behind each point, which is the
        # warning that a wandering stretch of curve is resting on very
        # little.
        ax.scatter(
            x, binned,
            s=8 + counts.to_numpy() * 0.4,
            color=colour, alpha=0.30, edgecolor="none", zorder=2,
        )

    # Where a curve crosses this line is that person's judged boundary.
    ax.axhline(0.5, linestyle="--", linewidth=0.8, color=style.MUTED, zorder=0)

    # Zero is the body edge: the only x value that means anything on its
    # own. A curve crossing 0.5 to the right of it is a person whose
    # judged boundary sits outside their body.
    ax.axvline(0, linestyle="-", linewidth=1.0, color=style.INK, alpha=0.5,
               zorder=1)

    if np.isfinite(drawn_low):
        ax.set_xlim(drawn_low - bin_width, drawn_high + bin_width)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(
        "Clearance from the shoulder edge (cm)"
        if ball_radius_cm else "Offset from the body edge (cm)"
    )
    ax.set_ylabel("P(respond HIT)")

    style.clean(ax)


def _wilson(successes, total, z=1.96):
    """Wilson score interval, which behaves at proportions near 1.

    The normal approximation puts the upper bound above 1 whenever a
    participant is at ceiling, which happens here on the clear trials.
    """
    if not np.isfinite(successes) or not np.isfinite(total) or total == 0:
        return np.nan, np.nan

    p = successes / total
    denominator = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denominator
    spread = (
        z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2))
        / denominator
    )

    return centre - spread, centre + spread


def _sdt(n_signal_hits, n_signal, n_noise_hits, n_noise):
    """d' and criterion from raw counts, with the loglinear correction.

    ``n_signal_hits`` is HIT responses on trials that would have hit;
    ``n_noise_hits`` is HIT responses on trials that would have missed.

    Rates of 0 and 1 have no finite z, and the clear trials reach both,
    so 0.5 is added to each response count and 1 to each trial count
    before the rates are taken. That is the loglinear correction, and it
    is applied to every participant rather than only the ones at
    ceiling, since correcting some and not others would make them
    incomparable.

    Positive criterion means a lean toward answering MISS, so a judged
    boundary sitting closer to the body.
    """
    if not n_signal or not n_noise:
        return np.nan, np.nan, np.nan, np.nan

    hit_rate = (n_signal_hits + 0.5) / (n_signal + 1)
    false_alarm_rate = (n_noise_hits + 0.5) / (n_noise + 1)

    z_hit = stats.norm.ppf(hit_rate)
    z_false_alarm = stats.norm.ppf(false_alarm_rate)

    return (
        float(z_hit - z_false_alarm),
        float(-0.5 * (z_hit + z_false_alarm)),
        float(hit_rate),
        float(false_alarm_rate),
    )


def carryover_by_band(trials, bands=(-10, 0, 8, 16, 30),
                      min_cell=3, min_bands=3, verbose=True):
    """Carryover as a within-band difference in P(respond HIT).

    The preregistered index is ``-b3 / b1`` from the post-switch model.
    At 35 to 60 post-switch trials per participant that is a small
    coefficient divided by a shallow slope, and the fitted values do not
    track the trial-level data: in this sample the participant with the
    largest raw carryover came out negative, while one showing almost
    none came out largest.

    This measures the same thing without the division. Within each band
    of offset, P(HIT) after a slow ball minus P(HIT) after a fast one,
    averaged across bands weighted by trials.

    Binning is what makes the comparison fair. Whether someone says HIT
    depends mostly on the offset, so an imbalance in where the
    previous-slow and previous-fast trials happened to land otherwise
    reads as a carryover effect: one participant here shows a raw
    difference of -0.31 that disappears once offset is held constant,
    because his previous-slow trials landed 4 cm further out.

    Positive means the preceding slow ball left them more willing to call
    a hit, which is the boundary still displaced toward the slower
    context.

    ``min_cell`` drops a band where either condition holds fewer than
    that many trials; ``min_bands`` returns NaN for a participant left
    with too few bands to average. Both matter: at one trial per cell the
    difference can only be -1, 0 or +1, and a value built on one band is
    not an estimate. The counts are returned so a NaN can be traced to
    the reason for it.
    """
    d = trials[trials["usable"]] if "usable" in trials.columns else trials

    d = d[
        (d["transition_status"] == "transition")
        & d["trial_type"].isin(NEAR_CATEGORIES)
    ].copy()

    d["band"] = pd.cut(d["offset_from_edge_cm"], list(bands))

    cell = (
        d.groupby(["subject", "band", "previous_speed"], observed=True)
         ["response_hit"].agg(["mean", "size"])
    )

    rows = []

    for subject, one in cell.groupby("subject"):
        one = one.unstack("previous_speed")

        if not {"slow", "fast"}.issubset(one["mean"].columns):
            rows.append({"subject": subject, "carryover_p": np.nan,
                         "n_bands": 0, "n_trials": 0})
            continue

        n = one["size"].min(axis=1)
        usable = n >= min_cell

        if usable.sum() < min_bands:
            rows.append({
                "subject": subject,
                "carryover_p": np.nan,
                "n_bands": int(usable.sum()),
                "n_trials": int(one["size"].sum(axis=1).sum()),
            })
            continue

        difference = (one["mean"]["slow"] - one["mean"]["fast"])[usable]
        weight = n[usable]

        rows.append({
            "subject": subject,
            "carryover_p": float(np.average(difference, weights=weight)),
            "n_bands": int(usable.sum()),
            "n_trials": int(one["size"].sum(axis=1)[usable].sum()),
        })

    out = pd.DataFrame(rows)

    if verbose and len(out):
        print("Carryover as a within-band difference in P(HIT)")
        print()
        print(out.round(3).to_string(index=False))
        print()
        print(
            "  Positive means the preceding slow ball left them more "
            "willing to call a hit. NaN means too few bands survived the "
            f"minimum of {min_cell} trials per condition; n_bands says how "
            "many did."
        )

    return out


def carryover_permutation(trials, subject, n_perm=5000, seed=0,
                          bands=(-10, 0, 8, 16, 30), min_cell=3):
    """Is one person's carryover bigger than a shuffle of their own trials?

    The previous-speed label is permuted across that participant's own
    post-switch trials, so the offsets, the responses and the band
    structure stay exactly as they were and only the link between the
    preceding speed and the response is broken.

    Two-sided, since a reversed carryover is as much a departure from no
    effect as an ordinary one.

    ``n_perm`` counts only the shuffles that produced a usable statistic.
    A large shortfall means the participant's band structure is thin
    enough that many shuffles leave no band above ``min_cell``, which is
    itself a reason to distrust their estimate.
    """
    d = trials[trials["usable"]] if "usable" in trials.columns else trials

    d = d[
        (d["subject"] == subject)
        & (d["transition_status"] == "transition")
        & d["trial_type"].isin(NEAR_CATEGORIES)
    ].copy()

    d["band"] = pd.cut(d["offset_from_edge_cm"], list(bands))

    def statistic(frame):
        cell = (
            frame.groupby(["band", "previous_speed"], observed=True)
                 ["response_hit"].agg(["mean", "size"])
                 .unstack("previous_speed")
        )

        if not {"slow", "fast"}.issubset(cell["mean"].columns):
            return np.nan

        n = cell["size"].min(axis=1)
        usable = n >= min_cell

        if not usable.any():
            return np.nan

        difference = (cell["mean"]["slow"] - cell["mean"]["fast"])[usable]

        return float(np.average(difference, weights=n[usable]))

    observed = statistic(d)

    if not np.isfinite(observed):
        return {"subject": subject, "observed": np.nan, "p": np.nan,
                "n_perm": 0, "n_trials": len(d)}

    labels = d["previous_speed"].to_numpy()
    rng = np.random.default_rng(seed)

    null = np.array([
        statistic(d.assign(previous_speed=rng.permutation(labels)))
        for _ in range(n_perm)
    ])
    null = null[np.isfinite(null)]

    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1)

    return {"subject": subject, "observed": observed, "p": float(p),
            "n_perm": len(null), "n_trials": len(d)}


def carryover_permutation_all(trials, n_perm=5000, seed=0, verbose=True):
    """``carryover_permutation`` for every participant in the frame."""
    subjects = sorted(trials["subject"].dropna().unique())

    out = pd.DataFrame([
        carryover_permutation(trials, s, n_perm=n_perm, seed=seed)
        for s in subjects
    ])

    if verbose and len(out):
        print("Carryover against a shuffle of each person's own trials")
        print()
        print(out.round(3).to_string(index=False))
        print()
        print(
            "  Two-sided. n_perm counts the shuffles that produced a "
            "usable statistic; a large shortfall means the band structure "
            "is thin enough that many shuffles leave nothing to measure."
        )

    return out


def _box_with_points(ax, groups, positions, colours, width=0.50,
                     point_size=26, seed=0, show_points=True):
    """A box filled in the variable's own colour, with the points over it.

    The fill is the colour at ``FILL_ALPHA``, the median line and the
    whiskers are the same colour at full strength, and the points sit on
    top without an edge so nothing competes with the summary underneath.

    ``show_points`` defaults to True: the points are what let you see how
    many observations a box rests on, which a box alone hides.
    """
    for index, (values, position, colour) in enumerate(
        zip(groups, positions, colours)
    ):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if not len(values):
            continue

        ax.boxplot(
            [values],
            positions=[position],
            widths=width,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(
                facecolor=style.shade(colour, 1 - FILL_ALPHA),
                edgecolor=colour,
                linewidth=1.1,
            ),
            medianprops=dict(color=colour, linewidth=2.0),
            whiskerprops=dict(color=colour, linewidth=1.1),
            capprops=dict(color=colour, linewidth=1.1),
        )

        if show_points:
            ax.scatter(
                position + style.jitter(len(values), width * 0.18,
                                        seed + index),
                values,
                s=point_size,
                color=colour,
                alpha=POINT_ALPHA,
                edgecolor="none",
                zorder=3,
            )


def _draw_category_box(ax, values, position, colour, width, pale=False):
    """One box in a category's colour, optionally in the paler fill.

    The paler fill marks the slow speed where two speeds are drawn as a
    pair, so a pair reads as one category at two speeds rather than as
    two unrelated boxes.
    """
    fill = style.shade(colour, 1 - FILL_ALPHA * (0.5 if pale else 1.0))

    ax.boxplot(
        [values],
        positions=[position],
        widths=width,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=fill, edgecolor=colour, linewidth=1.0),
        medianprops=dict(color=colour, linewidth=1.8),
        whiskerprops=dict(color=colour, linewidth=1.0),
        capprops=dict(color=colour, linewidth=1.0),
    )


def drop_subjects(trials, exclude=None, verbose=True):
    """Remove named subjects from a collision trials frame.
 
    ``exclude`` is a subject id or a list of them. Returns the frame
    unchanged when it is empty or None, so a call site can pass a
    variable that is sometimes empty without branching.
 
    An id that matches no subject is reported rather than ignored: a
    typo in an exclusion list otherwise looks exactly like a successful
    exclusion.
    """
    if not exclude:
        return trials
 
    if isinstance(exclude, str):
        exclude = [exclude]
 
    exclude = list(exclude)
    present = set(trials["subject"].unique())
 
    missing = [s for s in exclude if s not in present]
    dropped = [s for s in exclude if s in present]
 
    if verbose:
        if dropped:
            print(f"Excluded from this analysis: {', '.join(sorted(dropped))}")
 
        if missing:
            print(
                "  Not found in these trials, so nothing was dropped for "
                f"them: {', '.join(sorted(missing))}"
            )
 
    return trials[~trials["subject"].isin(exclude)].copy()
 
# ==================================================================
# The test
# ==================================================================

def run(t_qc, exclude=None, ball_radius_cm=0.0, save_as="h3_delta_coll",
        plot=True):

    results = {}

    trials = t_qc.collision_trials.copy()

    # Offsets are stored to the ball's centre, one radius outside the
    # surface gap the task used. Shifting here, before anything is fitted,
    # puts the PSEs and every figure this function draws on the same axis
    # as the clinical figures. Delta_coll is a difference and so is
    # unaffected; the PSEs move by exactly the radius.
    if ball_radius_cm:
        trials["offset_from_edge_cm"] = (
            trials["offset_from_edge_cm"] - ball_radius_cm
        )

    # Recorded so make_figure can label the axis in the units the data is
    # actually in, rather than assuming one convention.
    results["ball_radius_cm"] = ball_radius_cm
 
    # ------------------------------------------------------------------
    # Step 0. Prepare young-control collision trials
    # ------------------------------------------------------------------
 
    if "usable" in trials.columns:
        trials = trials[
            trials["usable"].fillna(False).astype(bool)
        ].copy()
 
    trials = trials[
        trials["speed"].isin(
            [config.SLOW_LABEL, config.FAST_LABEL]
        )
    ].copy()
 
    trials = trials.dropna(
        subset=[
            "subject",
            "offset_from_edge_cm",
            "response_hit",
            "speed",
        ]
    ).copy()
 
    trials = trials[
        trials["subject"].isin(config.YOUNG_CONTROL_SUBJECTS)
    ].copy()
 
    # Subject-level exclusion, applied after the cohort filter so the
    # printed list names only subjects that were actually in the analysis.
    trials = drop_subjects(trials, exclude)
 
    results["excluded_subjects"] = list(exclude) if exclude else []
    results["subjects_analysed"] = sorted(trials["subject"].unique())
 
    print(
        f"H3 on {len(results['subjects_analysed'])} young controls: "
        f"{', '.join(results['subjects_analysed'])}"
    )
    print()
 
    t_qc.collision_young = trials.copy()
 
    # ------------------------------------------------------------------
    # Step 1. Compute one Delta_coll per participant
    # ------------------------------------------------------------------

    index_rows = []

    for subject, one in trials.groupby("subject"):

        one = one.copy()

        one["speed_code"] = code_speed(one["speed"])

        one = one.dropna(
            subset=[
                "offset_from_edge_cm",
                "speed_code",
                "response_hit",
            ]
        )

        if (
            len(one) < 6
            or one["speed_code"].nunique() < 2
            or one["response_hit"].nunique() < 2
        ):
            index_rows.append({
                "subject": subject,
                "pse_slow": np.nan,
                "pse_fast": np.nan,
                "delta_coll": np.nan,
                "b0": np.nan,
                "b_offset": np.nan,
                "b_speed": np.nan,
                "n_trials": len(one),
                "fit_ok": False,
            })
            continue

        X = one[
            [
                "offset_from_edge_cm",
                "speed_code",
            ]
        ].to_numpy()

        y = one["response_hit"].astype(int).to_numpy()

        model = safe_logistic_regression(X, y)

        if model is None:
            index_rows.append({
                "subject": subject,
                "pse_slow": np.nan,
                "pse_fast": np.nan,
                "delta_coll": np.nan,
                "b0": np.nan,
                "b_offset": np.nan,
                "b_speed": np.nan,
                "n_trials": len(one),
                "fit_ok": False,
            })
            continue

        b0 = float(model.intercept_[0])
        b_offset = float(model.coef_[0][0])
        b_speed = float(model.coef_[0][1])

        if np.isclose(b_offset, 0):
            pse_slow = np.nan
            pse_fast = np.nan
            delta_coll = np.nan
            fit_ok = False
        else:
            pse_slow = -(b0 - b_speed) / b_offset
            pse_fast = -(b0 + b_speed) / b_offset
            delta_coll = pse_fast - pse_slow
            fit_ok = True

        index_rows.append({
            "subject": subject,
            "pse_slow": pse_slow,
            "pse_fast": pse_fast,
            "delta_coll": delta_coll,
            "b0": b0,
            "b_offset": b_offset,
            "b_speed": b_speed,
            "n_trials": len(one),
            "fit_ok": fit_ok,
        })

    collision_indices_young = pd.DataFrame(index_rows)

    t_qc.collision_indices_young = collision_indices_young

    # ------------------------------------------------------------------
    # Step 2. One Delta_coll per participant
    # ------------------------------------------------------------------

    delta_per_subject = (
        collision_indices_young
        .groupby(
            "subject",
            as_index=False,
        )["delta_coll"]
        .mean()
    )

    results["delta_per_subject"] = delta_per_subject
    results["collision_indices_young"] = collision_indices_young

    # ------------------------------------------------------------------
    # Step 3. Group test
    # ------------------------------------------------------------------

    values = (
        delta_per_subject["delta_coll"]
        .dropna()
        .to_numpy()
    )

    boot = bootstrap_mean_ci(
        values,
        n_boot=config.N_BOOT,
        ci=95,
        seed=config.RANDOM_SEED,
    )

    wilcoxon = wilcoxon_one_sided(
        values,
        alternative="greater",
    )

    supported = (
        bool(boot["mean"] > 0)
        and bool(boot["ci_low"] > 0)
    )

    results["values"] = values
    results["boot"] = boot
    results["wilcoxon"] = wilcoxon
    results["supported"] = supported

    # ------------------------------------------------------------------
    # Step 4. Pooled logistic fit for figure only
    # ------------------------------------------------------------------

    pooled = trials.copy()

    pooled["speed_code"] = code_speed(
        pooled["speed"]
    )

    pooled = pooled.dropna(
        subset=[
            "offset_from_edge_cm",
            "speed_code",
            "response_hit",
        ]
    )

    results["pooled_trials"] = pooled

    pooled_model = safe_logistic_regression(
        pooled[
            [
                "offset_from_edge_cm",
                "speed_code",
            ]
        ].to_numpy(),
        pooled["response_hit"]
        .astype(int)
        .to_numpy(),
    )

    results["pooled_model"] = pooled_model

    # ------------------------------------------------------------------
    # Step 5. Report and plot
    # ------------------------------------------------------------------

    report(results)

    if plot:
        fig = make_figure(results)

        if fig is not None:
            figures.save(fig, save_as)
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

    # --- Panel A: Delta_coll per participant -------------------------
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
        ax.text(x_positions[i], values[i], f" {subject}", fontsize=6,
                va="center")

    ax.axhline(0, linestyle="--", linewidth=0.8, color=style.MUTED)

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
            label="Mean (95% CI)",
        )

    ax.set_xticks(list(x_positions) + [len(values) + 0.5])
    ax.set_xticklabels(
        list(delta_per_subject["subject"]) + ["Mean"],
        rotation=30, ha="right",
    )
    ax.set_ylabel(r"$\Delta_{coll}$ (cm)")
    ax.set_title("Speed-induced boundary shift", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)

    style.clean(ax)

    # --- Panel B: psychometric curves --------------------------------
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
            color=style.MUTED,
        )
        ax.set_axis_off()

        figures.label_panels(axes)
        fig.tight_layout()
        return fig

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
        label=_label_speed(config.SLOW_LABEL),
        zorder=2,
    )
    ax.plot(
        smooth_x,
        expit((b0 + b_speed) + b_offset * smooth_x),
        color=style.FAST,
        linewidth=1.8,
        label=_label_speed(config.FAST_LABEL),
        zorder=2,
    )

    # Empirical P(HIT), binned. The dots are sized by how many trials fell
    # in each bin, so a tiny dot warns you not to trust that point.
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

        which_bin = np.digitize(
            one_speed["offset_from_edge_cm"], bin_edges,
        ) - 1
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

    ax.axhline(0.5, linestyle="--", linewidth=0.8, color=style.MUTED)

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

    ax.set_xlabel(
        "Clearance from the shoulder edge (cm)"
        if results.get("ball_radius_cm") else "Offset from shoulder edge (cm)"
    )
    ax.set_ylabel("P(respond HIT)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Collision psychometric", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)

    style.clean(ax)

    figures.label_panels(axes)
    fig.tight_layout()

    return fig


# ==================================================================
# Descriptives
# ==================================================================

def collision_descriptives(trials, verbose=True):
    """Per participant: detection performance and reaction time.

    Accuracy alone cannot distinguish a participant who judges well from
    one who says HIT to everything, so hits and false alarms are reported
    separately. A hit is saying HIT when the ball would have hit; a false
    alarm is saying HIT when it would have missed. A participant with a
    high hit rate and an equally high false-alarm rate is not judging.
    """
    d = _collision_frame(trials)

    rows = []

    for (subject, speed), one in d.groupby(["subject", "speed"]):
        hit_trials = one[one["was_hit"]]
        miss_trials = one[~one["was_hit"]]
        near = one[one["near_boundary"]]

        record = {
            "subject": subject,
            "speed": speed,
            "n_trials": len(one),
            "accuracy": one["correct"].mean(),
            "hit_rate": (
                hit_trials["said_hit"].mean() if len(hit_trials) else np.nan
            ),
            "false_alarm_rate": (
                miss_trials["said_hit"].mean() if len(miss_trials) else np.nan
            ),
            "near_accuracy": near["correct"].mean() if len(near) else np.nan,
            "n_near": len(near),
        }

        if "rt_ms" in one.columns:
            rt = one["rt_ms"].dropna()
            record.update({
                "rt_min": rt.min() if len(rt) else np.nan,
                "rt_median": rt.median() if len(rt) else np.nan,
                "rt_max": rt.max() if len(rt) else np.nan,
                "rt_iqr": (
                    rt.quantile(0.75) - rt.quantile(0.25)
                    if len(rt) else np.nan
                ),
            })

        rows.append(record)

    out = pd.DataFrame(rows).round(3)

    if verbose:
        print("Collision performance per participant and speed")
        print()
        print(out.to_string(index=False))
        print()
        print(
            "  hit_rate is P(say HIT | ball would hit); false_alarm_rate is "
            "P(say HIT | ball would miss). Similar values in both mean the "
            "participant is not discriminating, whatever the accuracy says."
        )

    return out


def collision_rt_outliers(trials, whisker=1.5, verbose=True):
    """Reaction times far from their own cell, by participant and offset bin.

    A trial is flagged when it falls outside the Tukey fences of its own
    participant, speed and offset bin, so a slow responder is not flagged
    for being slow overall, and a trial near the boundary is compared
    against other trials near the boundary rather than against the easy
    ones.
    """
    d = _collision_frame(trials)

    if "rt_ms" not in d.columns:
        raise KeyError("No 'rt_ms' column in the collision trials.")

    # Signed, so a hard trial is compared against other hard trials on
    # the same side of the boundary rather than against its mirror image.
    centres, which = _signed_bins(d, width=5.0)
    d["offset_bin"] = which
    d["bin_centre_cm"] = centres[which]

    grouped = d.groupby(["subject", "speed", "offset_bin"])["rt_ms"]

    q1 = grouped.transform(lambda s: s.quantile(0.25))
    q3 = grouped.transform(lambda s: s.quantile(0.75))
    spread = q3 - q1

    d["lower_fence"] = q1 - whisker * spread
    d["upper_fence"] = q3 + whisker * spread
    d["cell_n"] = grouped.transform("size")
    d["cell_median"] = grouped.transform("median")

    flagged = d[
        (d["rt_ms"] < d["lower_fence"]) | (d["rt_ms"] > d["upper_fence"])
    ].copy()

    columns = [c for c in [
        "subject", "speed", "trial_category", "bin_centre_cm",
        "trial_index", "trial", "rt_ms", "cell_median", "cell_n",
        "lower_fence", "upper_fence",
    ] if c in flagged.columns]

    flagged = flagged[columns].round(1).sort_values("rt_ms", ascending=False)

    if verbose:
        print(
            f"{len(flagged)} of {len(d)} trials outside the {whisker} IQR "
            f"fences of their own participant, speed and offset bin "
            f"({100 * len(flagged) / max(len(d), 1):.1f}%)"
        )
        print()

        if len(flagged):
            print(flagged.head(20).to_string(index=False))
            print()
            print("per participant:")
            print(flagged.groupby("subject").size().to_string())

            if "trial_index" in flagged.columns:
                print()
                print("trial_index values, ready to exclude:")
                print(sorted(flagged["trial_index"].tolist()))

    return flagged


def collision_exclusion_summary(trials, verbose=True):
    """What each rule removed, per participant and speed.

    The inspection figure shows where the excluded trials fell; this
    counts them. ``exclusion_reason`` is read from the trials, so a rule
    added later appears here without changing this function.
    """
    d = _tidy_collision(trials)

    if "exclusion_reason" not in d.columns:
        d["exclusion_reason"] = np.where(d["usable"], "usable", "excluded")

    by_reason = (
        d.groupby(["speed", "exclusion_reason"]).size()
        .unstack("exclusion_reason", fill_value=0)
    )

    by_subject = (
        d[~d["usable"]]
        .groupby(["subject", "exclusion_reason"]).size()
        .unstack("exclusion_reason", fill_value=0)
    )

    kept = d.groupby("subject")["usable"].agg(total="size", kept="sum")
    kept["dropped"] = kept["total"] - kept["kept"]
    kept["percent_kept"] = (100 * kept["kept"] / kept["total"]).round(1)

    if verbose:
        print("Collision exclusions by speed and reason")
        print()
        print(by_reason.to_string())
        print()

        if len(by_subject):
            print("Excluded trials per participant and reason")
            print()
            print(by_subject.to_string())
            print()

        print("Trials kept per participant")
        print()
        print(kept.to_string())

    return {
        "by_reason": by_reason,
        "by_subject": by_subject,
        "kept": kept,
    }


def collision_indices_table(trials, carryover_bands=(-10, 0, 8, 16, 30),
                            min_cell=3, min_bands=3, verbose=True):
    """Per participant: accuracy, sensitivity, bias, and carryover.

    Four things about the same session, answering different questions.
    Overall accuracy says whether the participant did the task.
    Near-boundary accuracy says whether they did it where it was hard,
    which is the only place a speed effect can show. d' and the criterion
    split that performance into how well they discriminated and which way
    they leaned. Carryover says whether the previous speed was still
    pulling on the boundary after a switch.

    The signal detection measures are given twice: over all trials, and
    over the near trials alone. The near-only version is the informative
    one, since the clear trials sit at ceiling and inflate d' without
    saying anything about where the boundary sits.

    The counts are reported alongside because all of these become
    unstable on thin data, and carryover most of all: it is estimated
    from post-switch trials only, which are a fraction of the session.
    """
    d = _collision_frame(trials)

    # The band-difference carryover, computed once for everyone and
    # joined on below. It sits beside the preregistered ``carryover_cm``
    # rather than replacing it, so both are visible and the reader can
    # see that they disagree.
    band = (
        carryover_by_band(
            trials, bands=carryover_bands,
            min_cell=min_cell, min_bands=min_bands, verbose=False,
        ).set_index("subject")
        if "transition_status" in d.columns else pd.DataFrame()
    )

    rows = []

    for subject, one in d.groupby("subject"):
        indices = compute_collision_indices_one_session(one.copy())

        near = one[one["trial_category"].isin(NEAR_CATEGORIES)]

        signal = one[one["was_hit"]]
        noise = one[~one["was_hit"]]

        near_signal = near[near["was_hit"]]
        near_noise = near[~near["was_hit"]]

        d_prime, criterion, hit_rate, false_alarm_rate = _sdt(
            int(signal["said_hit"].sum()), len(signal),
            int(noise["said_hit"].sum()), len(noise),
        )

        near_d_prime, near_criterion, near_hit, near_false_alarm = _sdt(
            int(near_signal["said_hit"].sum()), len(near_signal),
            int(near_noise["said_hit"].sum()), len(near_noise),
        )

        post_switch = (
            int((one["transition_status"] == "transition").sum())
            if "transition_status" in one.columns
            else np.nan
        )

        rows.append({
            "subject": subject,
            "n_trials": len(one),
            "n_correct": int(one["correct"].sum()),
            "accuracy": indices["accuracy"],
            "n_near": len(near),
            "n_near_correct": int(near["correct"].sum()),
            "near_boundary_accuracy": indices["near_boundary_accuracy"],
            "hit_rate": hit_rate,
            "false_alarm_rate": false_alarm_rate,
            "d_prime": d_prime,
            "criterion": criterion,
            "near_hit_rate": near_hit,
            "near_false_alarm_rate": near_false_alarm,
            "near_d_prime": near_d_prime,
            "near_criterion": near_criterion,
            "n_post_switch": post_switch,
            "carryover_cm": indices["carryover_cm"],
            "carryover_ok": bool(indices["carryover_model_ok"]),
            "carryover_p": (
                float(band.loc[subject, "carryover_p"])
                if subject in band.index else np.nan
            ),
            "carryover_p_bands": (
                int(band.loc[subject, "n_bands"])
                if subject in band.index else 0
            ),
        })

    out = pd.DataFrame(rows)

    if verbose:
        print("Collision indices per participant")
        print()
        print(
            out[[
                "subject", "n_trials", "accuracy",
                "n_near", "near_boundary_accuracy",
                "near_d_prime", "near_criterion",
                "n_post_switch", "carryover_cm", "carryover_ok",
                "carryover_p", "carryover_p_bands",
            ]].round(3).to_string(index=False)
        )
        print()
        print(
            "  near_d_prime is sensitivity on the near trials, "
            "near_criterion the bias: positive means a lean toward "
            "answering MISS, so a judged boundary closer to the body."
        )
        print(
            "  carryover_cm is -b3 / b1 from the post-switch model, in the "
            "same centimetres as the offset. Positive means the first "
            "judgement after a speed change is still displaced toward the "
            "previous speed's boundary; near zero means immediate updating."
        )
        print(
            "  carryover_p is the same quantity as a within-band "
            "difference in P(HIT), which does not divide by the offset "
            "slope. Where the two disagree, carryover_p is the one that "
            "matches the trials; carryover_p_bands says how many bands "
            "it rests on."
        )

        # The two measures fail for different reasons and on different
        # participants, so both are named. Reporting only one leaves a
        # NaN in the other column unexplained.
        if (~out["carryover_ok"]).any():
            failed = out.loc[~out["carryover_ok"], "subject"].tolist()
            print()
            print(
                "  carryover_cm not estimable, the offset slope was not "
                f"clearly negative: {', '.join(failed)}"
            )

        if out["carryover_p"].isna().any():
            failed = out.loc[out["carryover_p"].isna(), "subject"].tolist()
            print()
            print(
                f"  carryover_p not estimable, fewer than {min_bands} bands "
                f"held {min_cell} trials in both conditions: "
                f"{', '.join(failed)}"
            )

    return out


# ==================================================================
# Figures
# ==================================================================

def plot_collision_accuracy(
    trials,
    categories=("near_hit", "near_miss"),
    bin_width=4.0,
    min_trials=8,
    show_blocks=True,
    show_speed_legend=True,
    ball_radius_cm=0.0,
    figure_width=None,
    figure_height=3.8,
    save_as=None,
):
    """Accuracy outward from the boundary, and per participant by category.

    Panel A is the mean across participants of each participant's own
    proportion correct in each bin, so everyone counts once regardless of
    how many trials they contributed. A bin holding fewer than
    ``min_trials`` trials for a participant is dropped for that
    participant, and the axis stops at the last bin that survives rather
    than running on through empty space.

    Panel B is one marker per participant and category: the proportion
    correct over all that participant's trials of that category, with a
    95% Wilson interval. When ``show_blocks`` is set and the session has
    more than one block, each block's own proportion is drawn as a small
    point behind the marker, so the summary is visibly a summary of
    something rather than a bare dot.

    Panel B is restricted to ``categories``, by default the two near
    ones. Clear hits and clear misses sit at ceiling with almost no
    spread, so drawing them alongside compresses the axis and hides the
    variation in the categories that carry the judgement. Pass
    ``categories=TRIAL_CATEGORIES`` to draw all four.
    """
    d = _collision_frame(trials)

    # The task positions each ball so its nearest surface sits at the
    # specified gap from the shoulder, putting the centre one radius
    # further out. Subtracting the radius gives the clearance, in which
    # zero is where the ball grazes the body rather than where its centre
    # crosses the edge, and the trial categories then fall either side of
    # zero as the design intends.
    if ball_radius_cm:
        d = d.copy()
        d["offset_from_edge_cm"] = d["offset_from_edge_cm"] - ball_radius_cm

    speeds = _speeds(d)
    people = sorted(d["subject"].unique())

    # Sized from the panels rather than from a fixed page width. Panel B
    # holds one pair of points per participant, so its width has to grow
    # with the sample or the labels overlap; a two-column figure split
    # between two panels leaves neither enough room.
    fig = plt.figure(
        figsize=(figure_width or 4.0 + 0.9 * max(len(people), 4),
                 figure_height),
    )

    # Panel A gets a narrow column, then a spacer, then panel B. The
    # spacer is what the speed legend sits in: inside panel A it covered
    # the curves, and matplotlib will not push a panel aside to make room
    # for a legend, so the room has to be reserved here.
    if show_speed_legend:
        grid = fig.add_gridspec(
            1, 4, width_ratios=[1.0, 0.30, 0.06, 1.9], wspace=0.05,
        )
        ax_offset = fig.add_subplot(grid[0, 0])
        ax_subject = fig.add_subplot(grid[0, 3])
    else:
        grid = fig.add_gridspec(1, 3, wspace=0.30)
        ax_offset = fig.add_subplot(grid[0, 0])
        ax_subject = fig.add_subplot(grid[0, 1:])

    # --- A: accuracy against signed offset -------------------------
    ax = ax_offset

    _draw_signed_accuracy(
        ax, d, speeds, bin_width, min_trials, by_speed=True,
        ball_radius_cm=ball_radius_cm,
    )
    # The bin width and the near-trial restriction belong in the caption,
    # not in a title long enough to collide with the panel beside it.
    # Short, and left-aligned. Panel A is the narrow one, so a title of
    # any length runs past its own axes and into the bold letter sitting
    # at the left edge of panel B.
    ax.set_xlabel(
        "Clearance from the shoulder edge (cm)"
        if ball_radius_cm else "Offset from the body edge (cm)"
    )
    ax.set_title("Boundary shifts with speed", loc="left", fontsize=9)

    if show_speed_legend:
        ax.legend(
            title="Speed", frameon=False, fontsize=7.5, title_fontsize=8,
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
        )
    else:
        legend = ax.get_legend()

        if legend is not None:
            legend.remove()

    # --- B: per participant, one proportion per category -----------
    ax = ax_subject

    drawn = [
        category for category in categories
        if category in set(d["trial_category"])
    ]

    if not drawn:
        raise ValueError(
            f"None of {list(categories)} is present. Available: "
            f"{sorted(d['trial_category'].unique())}"
        )

    near = d[d["trial_category"].isin(drawn)]

    per_cell = (
        near.groupby(["subject", "trial_category"])["correct"]
        .agg(correct="sum", n="size")
        .reset_index()
    )
    per_cell["proportion"] = per_cell["correct"] / per_cell["n"]

    has_blocks = (
        show_blocks
        and "block" in near.columns
        and near["block"].nunique() > 1
    )

    per_block = (
        near.groupby(["subject", "trial_category", "block"])["correct"]
        .mean().reset_index(name="proportion")
        if has_blocks else None
    )

    width = 0.72 / len(drawn)
    offsets = (np.arange(len(drawn)) - (len(drawn) - 1) / 2) * width

    for index, category in enumerate(drawn):
        colour = _category_colour(category)

        if per_block is not None:
            blocks = per_block[per_block["trial_category"] == category]

            for _, row in blocks.iterrows():
                if row["subject"] not in people:
                    continue

                position = people.index(row["subject"]) + offsets[index]

                ax.scatter(
                    position + style.jitter(1, width * 0.16,
                                            int(row["block"])),
                    row["proportion"],
                    s=12, color=colour, alpha=0.55,
                    edgecolor="none", zorder=2,
                )

        rows = per_cell[per_cell["trial_category"] == category]

        for _, row in rows.iterrows():
            if row["subject"] not in people:
                continue

            position = people.index(row["subject"]) + offsets[index]
            low, high = _wilson(row["correct"], row["n"])

            ax.vlines(
                position, low, high,
                color=colour, linewidth=1.6, alpha=0.9, zorder=3,
            )
            ax.scatter(
                position, row["proportion"],
                s=38, color=colour, edgecolor="white", linewidth=0.7,
                zorder=4,
            )

        ax.plot([], [], color=colour, linewidth=3,
                label=_label_category(category))

    ax.axhline(0.5, linestyle="--", linewidth=0.8, color=style.MUTED, zorder=0)

    ax.set_xticks(np.arange(len(people)))
    ax.set_xticklabels(people, rotation=45, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(people) - 0.4)
    ax.set_ylim(0.35, 1.02)
    ax.set_ylabel("Proportion correct")
    ax.set_title("Per participant, near trials", loc="left", fontsize=9)
    ax.legend(fontsize=7.5, ncol=len(drawn), loc="lower left", frameon=False)
    style.clean(ax)

    fig.suptitle(
        "Collision judgements", x=0.02, ha="left", y=1.04, color=style.INK,
    )

    figures.label_panels([ax_offset, ax_subject])
    style.finish_figure(fig, tight=False)

    if save_as:
        figures.save(fig, save_as)

    return fig, per_cell.round(3)


 
def plot_collision_indices(trials, measure="proportion", near_only=True,
                           carryover="p", table=None, save_as=None):
    """Accuracy or sensitivity per participant, with carryover alongside.
 
    ``measure="proportion"`` draws overall accuracy, near-boundary
    accuracy, and carryover. The two accuracy panels share an axis so the
    drop from overall to near-boundary reads as a drop rather than as two
    unrelated numbers. That drop is the point: overall accuracy is held
    up by the clear trials, and the difference between the panels is how
    much of the performance came from the easy half of the design.
 
    ``measure="sdt"`` draws d', criterion, and carryover instead.
    Proportion correct collapses sensitivity and bias into one number, so
    a participant who discriminates poorly and one who discriminates well
    but answers MISS too readily can score the same. These two panels
    pull them apart. ``near_only`` restricts them to the near trials,
    which is the default because the clear trials sit at ceiling and
    inflate d' without saying anything about where the boundary sits.
 
    Everything is drawn in one grey. Each panel holds a single measure,
    so colour would encode nothing and only invite the reader to look for
    a distinction that is not there.
 
    Carryover appears in both versions, since it belongs to neither
    framework and is the only one of the three that is a distance. It has
    a line at zero, the value a participant who updates immediately would
    show. A participant whose model gave no usable estimate appears as an
    open marker on that line, so a missing estimate is visible rather
    than a gap in the row.
    """
    # ``table`` lets a caller pass the frame it already printed, so the
    # figure and the table cannot disagree about a carryover value: with
    # ``None`` the indices are computed a second time here, and any
    # difference in the band parameters between the two calls would show
    # up as a participant marked estimable in one and not the other.
    if table is None:
        table = collision_indices_table(trials, verbose=False)

    if not len(table):
        raise ValueError("No participants with collision trials.")
 
    people = table["subject"].tolist()
    positions = np.arange(len(people))
 
    fig, axes = plt.subplots(
        1, 3,
        figsize=style.figure_size(width="two_column", height="standard"),
        gridspec_kw={"wspace": 0.32},
    )
 
    if measure == "proportion":
        panels = [
            ("accuracy", "n_correct", "n_trials", "Overall accuracy"),
            ("near_boundary_accuracy", "n_near_correct", "n_near",
             "Near-boundary accuracy"),
        ]
 
        for ax, (column, correct_column, total_column, title) in zip(
            axes, panels
        ):
            for position, (_, row) in zip(positions, table.iterrows()):
                low, high = _wilson(row[correct_column], row[total_column])
 
                ax.vlines(
                    position, low, high,
                    color=INDEX_GREY, linewidth=1.8, alpha=0.9, zorder=2,
                )
                ax.scatter(
                    position, row[column],
                    s=44, color=INDEX_GREY, edgecolor="white", linewidth=0.7,
                    zorder=3,
                )
 
            ax.axhline(0.5, linestyle="--", linewidth=0.8, color=style.MUTED,
                       zorder=0)
            ax.set_ylim(0.35, 1.02)
            ax.set_ylabel("Proportion correct")
            ax.set_title(title, fontsize=9.5)
 
    elif measure == "sdt":
        prefix = "near_" if near_only else ""
        scope = "near trials" if near_only else "all trials"
 
        panels = [
            (f"{prefix}d_prime", f"Sensitivity d' ({scope})", None, "d'"),
            (f"{prefix}criterion", f"Criterion c ({scope})", 0.0, "c"),
        ]
 
        for ax, (column, title, reference, label) in zip(axes, panels):
            ax.scatter(
                positions, table[column],
                s=44, color=INDEX_GREY, edgecolor="white", linewidth=0.7,
                zorder=3,
            )
 
            if reference is not None:
                ax.axhline(reference, linestyle="--", linewidth=0.8,
                           color=style.MUTED, zorder=0)
 
            ax.set_ylabel(label)
            ax.set_title(title, fontsize=9.5)
 
        # Say which way the bias runs, since the sign is not self-evident.
        axes[1].text(
            0.02, 0.97, "positive: leans toward MISS",
            transform=axes[1].transAxes, ha="left", va="top",
            fontsize=7, color=style.MUTED,
        )
 
    else:
        raise ValueError(
            f"measure must be 'proportion' or 'sdt', got {measure!r}"
        )
 
    for ax in axes[:2]:
        ax.set_xticks(positions)
        ax.set_xticklabels(people, rotation=45, ha="right", fontsize=8)
        ax.set_xlim(-0.6, len(people) - 0.4)
        style.clean(ax)
 
    # --- C: carryover ----------------------------------------------
    ax = axes[2]

    # ``carryover="cm"`` draws the preregistered index, -b3 / b1 from the
    # post-switch model. ``carryover="p"`` draws the within-band
    # difference in P(HIT), which measures the same thing without
    # dividing by the offset slope. They can disagree, and where they do
    # the band difference is the one that matches the trials, so both are
    # available rather than one being chosen here.
    if carryover == "p":
        column = "carryover_p"
        label = "Carryover, P(HIT) difference"
        usable = table[column].notna()
        missing_label = "not estimable"
    elif carryover == "cm":
        column = "carryover_cm"
        label = "Carryover (cm)"
        usable = table["carryover_ok"].fillna(False).astype(bool)
        missing_label = "no usable estimate"
    else:
        raise ValueError(
            f"carryover must be 'cm' or 'p', got {carryover!r}"
        )

    ax.scatter(
        positions[usable.to_numpy()],
        table.loc[usable, column],
        s=44, color=INDEX_GREY, edgecolor="white", linewidth=0.7, zorder=3,
    )

    # A participant with no usable estimate sits open on the zero line,
    # so the absence is visible rather than an empty slot. Open is not
    # the same as a value of zero, which is why it is drawn unfilled and
    # named in the legend.
    if (~usable).any():
        ax.scatter(
            positions[(~usable).to_numpy()],
            np.zeros(int((~usable).sum())),
            s=44, facecolors="none", edgecolors=INDEX_GREY,
            linewidth=1.0, zorder=3, label=missing_label,
        )

        # Above the panel, not inside it. Sitting in the upper right the
        # legend lands beside whichever participant is plotted furthest
        # right, and reads as though it labelled that one.
        ax.legend(
            frameon=False, fontsize=7.5,
            loc="lower center", bbox_to_anchor=(0.5, 1.06),
        )

    ax.axhline(0, linestyle="--", linewidth=0.8, color=style.MUTED, zorder=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(people, rotation=45, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(people) - 0.4)
    ax.set_ylabel(label)
    ax.set_title("Post-switch carryover", fontsize=9.5)
    style.clean(ax)
 
    fig.suptitle(
        "Collision indices", x=0.02, ha="left", y=1.02, color=style.INK,
    )
 
    figures.label_panels(list(axes))
    style.finish_figure(fig, tight=False)
 
    if save_as:
        figures.save(fig, save_as)
 
    return fig, table





def plot_collision_rt(trials, show_legend=True, save_as=None):
    """Reaction time by speed, then by category within each speed.
 
    Returns two figures. In the first, panel A asks whether the speed
    manipulation changed latency at all, and panels B and C ask whether
    the near categories cost time, once within Slow and once within Fast.
    A trial close to the boundary is harder and should cost time, which
    is what a genuine judgement predicts; that difference is a fraction
    of the flight-time difference between speeds, so the two speeds get
    their own panels rather than sharing one.
 
    The second figure is the per-participant breakdown, one row per
    speed, so an unusual pattern is attributable to a person rather than
    to the group. The rows do not share a y axis: Fast is capped by
    ``RT_SUBJECT_CEILING_MS`` because a range wide enough for the slow
    row leaves every fast box crushed into the bottom third of it.
 
    Trial-level outliers are not drawn here. They are listed by
    ``collision_rt_outliers``, which returns them with their trial index
    so they can be excluded explicitly rather than hidden by the axis.
    """
    d = _collision_frame(trials)
 
    if "rt_ms" not in d.columns:
        raise KeyError("No 'rt_ms' column in the collision trials.")
 
    d = d.dropna(subset=["rt_ms"])
 
    speeds = _speeds(d)
    categories = _categories(d)
    people = sorted(d["subject"].unique())
 
    # ---- Figure 1: by speed, then by category within each speed ---
    fig_summary, axes = plt.subplots(
        1, 1 + len(speeds),
        figsize=style.figure_size(width="two_column", height="standard"),
        gridspec_kw={
            "width_ratios": [0.7] + [1.0] * len(speeds),
            "wspace": 0.32,
        },
    )
 
    axes = np.atleast_1d(axes)
 
    # --- A: by speed -----------------------------------------------
    ax = axes[0]
 
    medians = (
        d.groupby(["subject", "speed"])["rt_ms"].median()
        .unstack("speed").reindex(columns=speeds)
    )
 
    _box_with_points(
        ax,
        [medians[s].dropna().to_numpy() for s in speeds],
        np.arange(len(speeds)),
        [style.SPEED[s] for s in speeds],
        point_size=30,
    )
 
    ax.set_xticks(np.arange(len(speeds)))
    ax.set_xticklabels([_label_speed(s) for s in speeds])
    ax.set_ylabel("Median reaction time (ms)")
    ax.set_title("By speed", fontsize=9.5)
    style.clean(ax)
 
    # --- B and C: by category, one panel per speed -----------------
    by_cell = (
        d.groupby(["subject", "trial_category", "speed"])["rt_ms"]
        .median().reset_index()
    )
 
    for ax, speed in zip(axes[1:], speeds):
        for position, category in enumerate(categories):
            values = by_cell.loc[
                (by_cell["trial_category"] == category)
                & (by_cell["speed"] == speed),
                "rt_ms",
            ].dropna().to_numpy(dtype=float)
 
            if not values.size:
                continue
 
            colour = _category_colour(category)
 
            _draw_category_box(ax, values, position, colour, width=0.52)
 
            ax.scatter(
                position + style.jitter(len(values), 0.10, position),
                values,
                s=22, color=colour, alpha=POINT_ALPHA,
                edgecolor="none", zorder=3,
            )
 
        ax.set_xticks(np.arange(len(categories)))
        ax.set_xticklabels(
            [_label_category(c) for c in categories],
            rotation=20, ha="right",
        )
        ax.set_xlim(-0.6, len(categories) - 0.4)
        ax.set_ylabel("Median reaction time (ms)")
        ax.set_title(
            f"By trial category, {_label_speed(speed)}", fontsize=9.5,
        )
        style.clean(ax)
 
    fig_summary.suptitle(
        "Collision reaction time", x=0.02, ha="left", y=1.02, color=style.INK,
    )
 
    figures.label_panels(list(axes))
    style.finish_figure(fig_summary, tight=False)
 
    # ---- Figure 2: per participant, one row per speed -------------
    fig_subject, subject_axes = plt.subplots(
        len(speeds), 1,
        figsize=(figures.DOUBLE_COLUMN, 3.0 * len(speeds)),
        sharex=True, sharey=False,
        squeeze=False,
    )
    subject_axes = subject_axes[:, 0]
 
    width = 0.78 / max(len(categories), 1)
    offsets = (
        np.arange(len(categories)) - (len(categories) - 1) / 2
    ) * width
 
    for ax, speed in zip(subject_axes, speeds):
        one_speed = d[d["speed"] == speed]
 
        for index, category in enumerate(categories):
            colour = _category_colour(category)
 
            for person in people:
                values = one_speed.loc[
                    (one_speed["subject"] == person)
                    & (one_speed["trial_category"] == category),
                    "rt_ms",
                ].dropna().to_numpy(dtype=float)
 
                if not values.size:
                    continue
 
                _draw_category_box(
                    ax, values,
                    people.index(person) + offsets[index],
                    colour, width=width * 0.80,
                )
 
        ceiling = RT_SUBJECT_CEILING_MS.get(speed)
 
        if ceiling is not None:
            above = int((one_speed["rt_ms"] > ceiling).sum())
 
            ax.set_ylim(top=ceiling)
 
            if above:
                ax.set_title(
                    f"{_label_speed(speed)}, {above} trials above axis",
                    loc="left", fontsize=9.5,
                )
            else:
                ax.set_title(_label_speed(speed), loc="left", fontsize=9.5)
        else:
            ax.set_title(_label_speed(speed), loc="left", fontsize=9.5)
 
        ax.set_ylabel("Reaction time (ms)")
        style.clean(ax)
 
    if show_legend:
        for category in categories:
            subject_axes[0].plot(
                [], [], color=_category_colour(category), linewidth=3,
                label=_label_category(category),
            )

        # Above the top panel rather than inside it: at nine participants
        # and four categories there is no interior space it does not
        # cover.
        subject_axes[0].legend(
            fontsize=8, ncol=len(categories), frameon=False,
            loc="lower center", bbox_to_anchor=(0.5, 1.10),
        )
 
    subject_axes[-1].set_xticks(np.arange(len(people)))
    subject_axes[-1].set_xticklabels(people, rotation=45, ha="right")
    subject_axes[-1].set_xlim(-0.6, len(people) - 0.4)
 
    fig_subject.suptitle(
        "Collision reaction time per participant",
        x=0.02, ha="left", y=1.01, color=style.INK,
    )
 
    style.finish_figure(fig_subject, tight=False)
 
    if save_as:
        figures.save(fig_summary, save_as)
        figures.save(fig_subject, f"{save_as}_by_subject")
 
    return fig_summary, fig_subject, medians.round(0)
 
 


def plot_collision_response_map(trials, categories=None, bin_width=2.5,
                                ball_radius_cm=0.0, by_speed=True,
                                save_as=None):
    """P(HIT) by offset and participant, one row each, split by speed.

    Offset runs left to right from the body edge outward, in 5 cm bins
    labelled by their outer edge. Orange is a HIT judgement and purple a
    MISS, matching the categorical colours. A participant who tracked the
    trajectory shows orange on the left and purple on the right with a
    clean transition near the edge; a flat or patchy row means no
    boundary can be fitted for them whatever their accuracy says.
    """
    d = _collision_frame(trials)

    # The task positions each ball so its nearest surface sits at the
    # specified gap from the shoulder, putting the centre one radius
    # further out. Subtracting the radius gives the clearance, in which
    # zero is where the ball grazes the body rather than where its centre
    # crosses the edge. The same option exists on the accuracy figure, so
    # the two can be drawn in the same units; leaving one uncorrected
    # would put the same trials at different offsets in figures a reader
    # sees side by side.
    if ball_radius_cm:
        d = d.copy()
        d["offset_from_edge_cm"] = d["offset_from_edge_cm"] - ball_radius_cm

    if categories is None:
        categories = NEAR_CATEGORIES

    d = d[d["trial_category"].isin(categories)]

    speeds = _speeds(d) if by_speed else [None]

    labels, which = _map_bins(d, width=bin_width)
    d["offset_bin"] = which

    # Bands nobody has are dropped rather than drawn empty: an outer band
    # that only one participant reached opens a column of grey across
    # every other row, which reads as missing data where there was simply
    # nothing to sample.
    present = sorted(d["offset_bin"].unique())
    keep = {b: i for i, b in enumerate(present)}
    labels = [labels[b] for b in present]
    d["offset_bin"] = d["offset_bin"].map(keep)

    grid_all = (
        d.groupby(["subject", "offset_bin"])["said_hit"].mean()
        .unstack("offset_bin").reindex(columns=range(len(labels)))
    )

    sharpness = (
        grid_all.iloc[:, :2].mean(axis=1)
        - grid_all.iloc[:, -2:].mean(axis=1)
    ).abs().sort_values(ascending=False)

    colours = _hit_miss_colormap()

    fig, axes = plt.subplots(
        1, len(speeds),
        figsize=(figures.DOUBLE_COLUMN, 0.30 * len(sharpness) + 2.1),
        squeeze=False,
    )
    axes = axes[0]

    for index, (ax, speed) in enumerate(zip(axes, speeds)):
        grid = (
            d if speed is None else d[d["speed"] == speed]
            .groupby(["subject", "offset_bin"])["said_hit"].mean()
            .unstack("offset_bin")
            .reindex(index=sharpness.index, columns=range(len(labels)))
        )

        image = ax.imshow(
            grid.to_numpy(float),
            aspect="auto", cmap=colours, vmin=0, vmax=1,
            interpolation="nearest",
        )

        # The body edge now sits inside the axis, where the sign changes.
        ax.axvline(_edge_position(labels), color=style.INK,
                   linewidth=1.2, alpha=0.7)

        ax.set_yticks(np.arange(len(grid)))
        ax.set_yticklabels(grid.index if index == 0 else [])
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
        ax.set_xlabel(
            "Clearance from the shoulder edge (cm)"
            if ball_radius_cm else "Offset from the body edge (cm)"
        )
        ax.set_title(_label_speed(speed), fontsize=9.5)

        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(length=0)

    # The colourbar gets its own axes on the right rather than being cut
    # out of the last panel: passing ``ax=list(axes)`` shrinks the panels
    # unpredictably and leaves the bar sitting over the rightmost one.
    fig.subplots_adjust(right=0.88)
    bar_axes = fig.add_axes([0.90, 0.18, 0.015, 0.62])
    bar = fig.colorbar(image, cax=bar_axes)
    bar.set_label("P(respond HIT)")

    # Grey is a missing cell, not a value on the scale, so it is named
    # beside the bar rather than left for the reader to infer.
    bar_axes.text(
        0.5, -0.08, "grey:\nno trials",
        transform=bar_axes.transAxes, ha="center", va="top",
        fontsize=7, color=style.MUTED, linespacing=1.3,
    )

    fig.suptitle(
        "Collision responses by clearance, near trials"
        if ball_radius_cm else "Collision responses by offset, near trials",
        x=0.02, ha="left", color=style.INK,
    )

    # A colorbar takes axes space that tight_layout cannot account for.
    style.finish_figure(fig, tight=False)

    if save_as:
        figures.save(fig, save_as)

    return fig, sharpness.round(2)


def _hit_miss_colormap():
    """Purple at P(HIT) = 0 through to orange at P(HIT) = 1.

    The midpoint of this scale is near-white, which is also what an empty
    cell looks like if it is left to the axes background. A band nobody
    was tested at would then be indistinguishable from one where the
    participant was split evenly, which is the opposite of the truth. So
    missing cells are given their own grey, set apart from anything the
    scale itself produces.
    """
    from matplotlib.colors import LinearSegmentedColormap

    trial_type = getattr(style, "TRIAL_TYPE", {})

    miss = getattr(style, "MISS", trial_type.get("clear_miss", "#8971b0"))
    hit = getattr(style, "HIT", trial_type.get("clear_hit", "#f4792f"))

    colours = LinearSegmentedColormap.from_list(
        "miss_to_hit", [miss, "#f2f0ee", hit],
    )
    colours.set_bad("#dcdcdc")

    return colours


# ==================================================================
# Inspection
# ==================================================================

def collision_inspection(trials, rt_ceiling=None, save_as=None):
    """Look at the collision trials before deciding anything about them.

    Four rows: Before and After exclusions, each split into Slow and
    Fast. Scales are shared column by column within a speed, so the
    before and after rows can be read against each other. The speeds stay
    apart because a slow ball is in flight more than twice as long, so a
    pooled RT panel would show flight time and little else.

        A. Composition. What proportion of that cell's trials each
           category contributed, labelled with the count. The count is
           where the exclusions become visible: the before and after rows
           differ by exactly the trials the rules removed.
        B. RT density for the two near categories on one axis, since the
           question is whether they differ rather than what each looks
           like alone. The dashed lines are the medians.
        C. RT by category, every trial drawn over the box. The line is
           the median and the filled circle the mean; their separation is
           the skew.
        D. RT per participant and category, so an unusual pattern is
           attributable to a person rather than to the group.

    ``rt_ceiling`` is a dict of speed to upper axis limit in ms,
    defaulting to ``RT_AXIS_CEILING_MS``. A handful of pre-exclusion
    trials run to tens of seconds, and scaling either axis to its own
    maximum would compress every real trial into the leftmost tenth.
    Trials above the ceiling are counted in the panel label rather than
    drawn, so a removed trial is never silently invisible.
    """
    d_all = _tidy_collision(trials)

    if "rt_ms" not in d_all.columns:
        raise KeyError("No 'rt_ms' column in the collision trials.")

    d_all = d_all.dropna(subset=["rt_ms"])

    speeds = _speeds(d_all)
    categories = _categories(d_all)
    people = sorted(d_all["subject"].unique())

    ceiling = dict(RT_AXIS_CEILING_MS)

    if rt_ceiling:
        ceiling.update(rt_ceiling)

    rt_floor = float(d_all["rt_ms"].min()) - 40

    rng = np.random.default_rng(11)

    rows = [(stage, speed) for stage in STAGES for speed in speeds]

    fig = plt.figure(figsize=(19, 4.2 * len(rows)))
    grid = fig.add_gridspec(
        len(rows), 4,
        width_ratios=[0.8, 1.1, 1.0, 2.1],
        wspace=0.30, hspace=0.42,
    )

    axes = {
        key: [fig.add_subplot(grid[index, column]) for column in range(4)]
        for index, key in enumerate(rows)
    }

    for stage, speed in rows:
        ax_composition, ax_density, ax_category, ax_subject = axes[
            (stage, speed)
        ]

        cell = collision_trials_at_stage(d_all, stage=stage)
        cell = cell[cell["speed"] == speed]

        top = ceiling.get(speed, 5000.0)
        limits = (rt_floor, top + 40)
        n_off_scale = int((cell["rt_ms"] > top).sum())

        # --- A. Composition ----------------------------------------
        ax = ax_composition

        counts = (
            cell["trial_category"].value_counts()
            .reindex(categories, fill_value=0)
        )
        total = int(counts.sum())
        percent = 100 * counts / total if total else counts * np.nan

        bars = ax.bar(
            np.arange(len(categories)),
            percent.to_numpy(dtype=float),
            width=0.62,
            color=[_category_colour(c) for c in categories],
            linewidth=0,
        )

        for bar, category in zip(bars, categories):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.0f}%\n{int(counts[category])}",
                ha="center", va="bottom", fontsize=7.5, color=style.INK,
                linespacing=1.25,
            )

        ax.set_xticks(np.arange(len(categories)))
        ax.set_xticklabels(
            [_label_category(c) for c in categories],
            rotation=30, ha="right", fontsize=8,
        )
        ax.set_ylabel(
            f"{STAGE_LABEL[stage]}, {_label_speed(speed)}\n\n% of trials"
        )
        ax.set_ylim(0, 62)
        style.clean(ax)

        # --- B. RT density, the two near categories ----------------
        ax = ax_density

        for category in NEAR_CATEGORIES:
            if category not in categories:
                continue

            values = (
                cell.loc[cell["trial_category"] == category, "rt_ms"]
                .dropna().to_numpy(dtype=float)
            )

            if values.size < 3 or np.std(values) == 0:
                continue

            colour = _category_colour(category)
            kde = stats.gaussian_kde(values)
            x = np.linspace(limits[0], limits[1], 400)

            ax.fill_between(x, 0, kde(x), color=colour, alpha=0.18,
                            linewidth=0)
            ax.plot(
                x, kde(x), color=colour, linewidth=1.4, alpha=0.9,
                label=f"{_label_category(category)} (n = {values.size:,})",
            )
            ax.axvline(
                float(np.median(values)), color=colour,
                linestyle="--", linewidth=1.0, alpha=0.7,
            )

        ax.set_xlim(limits)
        ax.set_ylabel("Density")
        ax.set_xlabel(
            "Reaction time (ms)" if not n_off_scale
            else f"Reaction time (ms), {n_off_scale} trials above axis"
        )
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        style.clean(ax)

        # --- C. RT by category, trials drawn -----------------------
        ax = ax_category

        for position, category in enumerate(categories):
            values = (
                cell.loc[cell["trial_category"] == category, "rt_ms"]
                .dropna().to_numpy(dtype=float)
            )

            if not values.size:
                continue

            colour = _category_colour(category)

            _draw_category_box(ax, values, position, colour, width=0.52)

            ax.scatter(
                position + rng.normal(0, 0.055, values.size),
                values,
                s=4, color=colour, alpha=0.35, edgecolor="none", zorder=3,
            )

            ax.scatter(
                position, values.mean(),
                marker="o", s=30, facecolors=colour, edgecolors="white",
                linewidths=0.6, zorder=6,
            )

        ax.set_xticks(np.arange(len(categories)))
        ax.set_xticklabels(
            [_label_category(c) for c in categories],
            rotation=30, ha="right", fontsize=8,
        )
        ax.set_ylim(limits)
        ax.set_ylabel("RT (ms)")
        style.clean(ax)

        # --- D. RT per participant and category --------------------
        ax = ax_subject

        width = 0.78 / max(len(categories), 1)
        offsets = (
            np.arange(len(categories)) - (len(categories) - 1) / 2
        ) * width

        for index, category in enumerate(categories):
            colour = _category_colour(category)

            for person in people:
                values = cell.loc[
                    (cell["subject"] == person)
                    & (cell["trial_category"] == category),
                    "rt_ms",
                ].dropna().to_numpy(dtype=float)

                if not values.size:
                    continue

                _draw_category_box(
                    ax, values,
                    people.index(person) + offsets[index],
                    colour, width=width * 0.80,
                )

            ax.plot([], [], color=colour, linewidth=3,
                    label=_label_category(category))

        ax.set_xticks(np.arange(len(people)))
        ax.set_xticklabels(people, rotation=45, ha="right", fontsize=8)
        ax.set_xlim(-0.6, len(people) - 0.4)
        ax.set_ylim(limits)
        ax.set_ylabel("RT (ms)")
        style.clean(ax)

        for ax in axes[(stage, speed)]:
            ax.grid(axis="y", color=style.RULE, linewidth=0.6, alpha=0.7)
            ax.set_axisbelow(True)

    titles = [
        "A. Trial composition",
        "B. RT density, near categories",
        "C. RT by category",
        "D. RT per participant",
    ]
    for ax, title in zip(axes[rows[0]], titles):
        ax.set_title(title, loc="left", pad=8, fontsize=9.5)

    axes[rows[0]][3].legend(
        fontsize=8, ncol=len(categories), frameon=False,
        loc="lower center", bbox_to_anchor=(0.5, 1.14),
    )

    fig.suptitle(
        "Collision reaction times before and after exclusions",
        x=0.012, ha="left", y=1.005, color=style.INK,
    )

    style.finish_figure(fig, tight=False)

    if save_as:
        figures.save(fig, save_as)

    return fig


def collision_diagnostics(trials, measure="proportion", carryover="cm",
                          show_speed_legend=True, show_legend=True,
                          show_indices=False, ball_radius_cm=0.0,
                          bin_width=4.0, map_bin_width=2.0,
                          figure_width=None,
                          save_as=None, verbose=True):
    """Descriptives and the figures, in reading order.

    ``measure`` is passed to ``plot_collision_indices``: "proportion" for
    accuracy, "sdt" for d' and criterion. ``carryover`` selects which
    carryover measure panel C draws, "cm" for the preregistered index or
    "p" for the within-band difference.

    ``show_indices`` is off by default because the indices table is
    normally printed on its own beforehand, and printing it again here
    invites the reader to compare two copies of the same numbers. It is
    returned either way, under ``"indices"``.

    The indices are computed once and handed to the figure, so the table
    and panel C cannot disagree about which participants had an
    estimable carryover.

    ``save_as`` is a stem, not a filename: each figure is written as
    ``<stem>_accuracy``, ``<stem>_indices`` and so on, in both SVG and
    PDF. Passing one name for five figures would have them overwrite
    each other.
    """
    performance = collision_descriptives(trials, verbose=verbose)

    indices = collision_indices_table(
        trials, verbose=verbose and show_indices,
    )

    print()
    outliers = collision_rt_outliers(trials, verbose=verbose)

    def name(suffix):
        return f"{save_as}_{suffix}" if save_as else None

    # The two figures bin the same offsets for different purposes, so
    # they take separate widths: the accuracy panel averages across
    # participants and tolerates narrow bins, while the response map
    # shows one person per row and needs enough trials per cell for a
    # colour to mean something.
    accuracy_figure, accuracy_cells = plot_collision_accuracy(
        trials, show_speed_legend=show_speed_legend,
        bin_width=bin_width, ball_radius_cm=ball_radius_cm,
        figure_width=figure_width, save_as=name("accuracy"),
    )
    indices_figure, _ = plot_collision_indices(
        trials, measure=measure, carryover=carryover, table=indices,
        save_as=name("indices"),
    )
    rt_figure, rt_subject_figure, rt_medians = plot_collision_rt(
        trials, show_legend=show_legend, save_as=name("rt"),
    )
    map_figure, sharpness = plot_collision_response_map(
        trials, bin_width=map_bin_width, ball_radius_cm=ball_radius_cm,
        save_as=name("response_map"),
    )

    if verbose:
        print()
        print("Transition sharpness, |P(HIT) near minus far|:")
        print(sharpness.to_string())
        print()
        print(
            "  Near zero means P(HIT) barely changes across the offset "
            "range, so no boundary can be located for that participant."
        )

    return {
        "performance": performance,
        "indices": indices,
        "rt_outliers": outliers,
        "accuracy_by_category": accuracy_cells,
        "rt_medians": rt_medians,
        "sharpness": sharpness,
        "figures": {
            "accuracy": accuracy_figure,
            "indices": indices_figure,
            "rt": rt_figure,
            "rt_by_subject": rt_subject_figure,
            "response_map": map_figure,
        },
    }


# ==================================================================
# The clinical pair: patient against matched controls
# ==================================================================
#
# Running ``collision_diagnostics`` once per cohort puts the patient and
# the control in separate figures with separately scaled axes, so the
# comparison has to be made by eye across two images. Everything below
# sits in shared panels on shared axes: the patient is one marker and
# each control another, and the distance between them is the thing the
# figures are for.


def _clinical_people(trials, patient=None, controls=None):
    """Resolve the patient and control ids, and check they have trials.

    ``patient`` and ``controls`` default to the configured ids.
    ``controls`` accepts one id or several. An id with no collision
    trials is dropped and named, since a missing person otherwise looks
    the same as a person who was never asked for.
    """
    if patient is None:
        patient = config.PATIENT_ID

    if controls is None:
        controls = config.MATCHED_CONTROL_IDS

    if isinstance(controls, str):
        controls = [controls]

    controls = list(controls or [])

    present = set(trials["subject"].unique())

    missing = [
        s for s in [patient] + controls
        if s is not None and s not in present
    ]

    if missing:
        print(
            "No collision trials, so not drawn: "
            f"{', '.join(sorted(missing))}"
        )

    patient = patient if patient in present else None
    controls = [s for s in controls if s in present]

    if patient is None and not controls:
        raise ValueError(
            "Neither the patient nor any control has collision trials."
        )

    return patient, controls


def _clinical_colour(subject, patient, controls):
    """Patient colour for the patient, age-matched for the controls.

    ``CONTROL`` grey belongs to the young group and is deliberately not
    used for the clinical pair: those controls are age-matched, and the
    figure should not invite them to be read as the young sample. With
    two controls the second is drawn in a darker shade of the same
    colour, so the pair stays visibly one category while remaining
    tellable apart where their lines cross.

    With no patient in the figure there is no clinical contrast to make,
    so the young controls get shades of ``CONTROL`` grey instead. Drawing
    nine young participants in the age-matched pink would say they were
    something they are not.
    """
    if patient is None:
        index = controls.index(subject) if subject in controls else 0
        spread = max(len(controls) - 1, 1)

        # Light to dark across the group, so a legend entry and a curve
        # can be matched without counting from either end.
        return style.shade(style.CONTROL, 0.35 - 0.70 * index / spread)

    if subject == patient:
        return style.PATIENT

    index = controls.index(subject) if subject in controls else 0

    return style.shade(style.AGE_MATCHED, -0.18 * index)


def plot_delta_coll(trials, patient=None, controls=None,
                    show_group=None, ball_radius_cm=0.0, save_as=None):
    """The H3 figure for any set of participants, including a pair.

    ``h3.make_figure`` draws the same two panels but only for the young
    controls, because ``h3.run`` filters to them and computes a group
    bootstrap on the way. This takes a trials frame instead, so the same
    figure can be drawn for the clinical pair, for a subset, or for
    anyone else.

    Panel A is each participant's Delta_coll. Panel B is the psychometric
    fitted to their pooled trials, with the two conditions drawn
    separately and the shift between their midpoints marked.

    ``show_group`` adds the group mean and its bootstrap interval to
    panel A. It defaults to on above three participants and off at or
    below, since a mean of two people is the two people and an interval
    around it invites a reading the design cannot support.

    Pooling in panel B is a description of the set as drawn, not an
    estimate of a population. With a patient and a control it is a curve
    through two people who may differ, and the per-participant fits in
    ``plot_clinical_psychometric`` are the honest version of the same
    data.
    """
    d = _collision_frame(trials)

    # Offsets are stored to the ball's centre, one radius outside the
    # surface gap the task used. Shifting here, before the psychometric
    # fits, keeps this figure on the same axis as the rest.
    if ball_radius_cm:
        d = d.copy()
        d["offset_from_edge_cm"] = d["offset_from_edge_cm"] - ball_radius_cm

    patient, controls = _clinical_people(d, patient=patient,
                                         controls=controls)

    people = ([patient] if patient else []) + controls
    d = d[d["subject"].isin(people)]

    if show_group is None:
        show_group = len(people) > 3

    # --- per participant -------------------------------------------
    rows = []

    for subject in people:
        fit = _fit_psychometric(d[d["subject"] == subject])

        rows.append({
            "subject": subject,
            "delta_coll": fit["delta_coll"] if fit else np.nan,
            "pse_slow": fit["pse_slow"] if fit else np.nan,
            "pse_fast": fit["pse_fast"] if fit else np.nan,
        })

    table = pd.DataFrame(rows)

    fig, axes = plt.subplots(
        1, 2, figsize=(figures.DOUBLE_COLUMN, 3.2),
        gridspec_kw={"wspace": 0.30},
    )

    # --- A: Delta_coll per participant -----------------------------
    ax = axes[0]

    positions = np.arange(len(people))

    for position, (_, row) in zip(positions, table.iterrows()):
        ax.scatter(
            position, row["delta_coll"],
            s=48, color=_clinical_colour(row["subject"], patient, controls),
            edgecolor="white", linewidth=0.7, zorder=3,
        )

    labels = [_display_name(s, patient, controls) for s in people]

    if show_group:
        values = table["delta_coll"].dropna().to_numpy()
        boot = bootstrap_mean_ci(
            values, n_boot=config.N_BOOT, ci=95, seed=config.RANDOM_SEED,
        )

        if np.isfinite(boot["mean"]):
            ax.errorbar(
                len(people) + 0.5, boot["mean"],
                yerr=[[boot["mean"] - boot["ci_low"]],
                      [boot["ci_high"] - boot["mean"]]],
                fmt="o", color=style.INK, markersize=6, capsize=4,
                zorder=3,
            )

            positions = np.append(positions, len(people) + 0.5)
            labels = labels + ["Mean (95% CI)"]

    ax.axhline(0, linestyle="--", linewidth=0.8, color=style.MUTED, zorder=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_xlim(-0.6, positions.max() + 0.6)
    ax.set_ylabel(r"$\Delta_{coll}$ (cm)")
    ax.set_title("Speed-induced boundary shift", loc="left", fontsize=9)
    style.clean(ax)

    # --- B: pooled psychometric ------------------------------------
    ax = axes[1]

    fit = _fit_psychometric(d)

    if fit is None:
        ax.text(
            0.5, 0.5, "no usable fit",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=8, color=style.MUTED,
        )
        ax.set_axis_off()
    else:
        low = float(d["offset_from_edge_cm"].min())
        high = float(d["offset_from_edge_cm"].max())

        smooth_x = np.linspace(low, high, 200)
        edges = np.linspace(low, high, 9)
        centres = 0.5 * (edges[:-1] + edges[1:])

        for speed, code, pse in [
            (config.SLOW_LABEL, -1, fit["pse_slow"]),
            (config.FAST_LABEL, +1, fit["pse_fast"]),
        ]:
            one_speed = d[d["speed"] == speed]

            ax.plot(
                smooth_x,
                expit(fit["b0"] + fit["b_speed"] * code
                      + fit["b_offset"] * smooth_x),
                color=style.SPEED[speed], linewidth=1.8,
                label=_label_speed(speed), zorder=2,
            )

            which = np.clip(
                np.digitize(one_speed["offset_from_edge_cm"], edges) - 1,
                0, len(centres) - 1,
            )

            binned = pd.DataFrame({
                "bin": which,
                "hit": one_speed["response_hit"].astype(int).to_numpy(),
            }).groupby("bin")["hit"].agg(["mean", "size"])

            ax.scatter(
                centres[binned.index], binned["mean"],
                s=6 + binned["size"] * 0.6,
                color=style.SPEED[speed], alpha=0.55,
                edgecolor="none", zorder=3,
            )

            ax.plot(
                [pse, pse], [0, 0.5],
                linestyle=":", linewidth=1.0,
                color=style.SPEED[speed], zorder=1,
            )

        ax.annotate(
            "", xy=(fit["pse_fast"], 0.5), xytext=(fit["pse_slow"], 0.5),
            arrowprops=dict(arrowstyle="<->", color=style.INK, lw=0.9),
        )
        ax.text(
            0.5 * (fit["pse_slow"] + fit["pse_fast"]), 0.55,
            f"$\\Delta_{{coll}}$ = "
            f"{fit['pse_fast'] - fit['pse_slow']:.1f} cm",
            ha="center", fontsize=7, color=style.INK,
        )

        ax.axhline(0.5, linestyle="--", linewidth=0.8, color=style.MUTED,
                   zorder=0)
        ax.axvline(0, linestyle="-", linewidth=1.0, color=style.INK,
                   alpha=0.5, zorder=1)

        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(
            "Clearance from the shoulder edge (cm)"
            if ball_radius_cm else "Offset from shoulder edge (cm)"
        )
        ax.set_ylabel("P(respond HIT)")
        ax.set_title(
            f"Pooled across {len(people)} participants", loc="left",
            fontsize=9,
        )
        ax.legend(frameon=False, fontsize=8)
        style.clean(ax)

    fig.suptitle(
        "Collision boundary", x=0.02, ha="left", y=1.03, color=style.INK,
    )

    figures.label_panels(list(axes))
    style.finish_figure(fig, tight=False)

    if save_as:
        figures.save(fig, save_as)

    return fig, table


def collision_clinical_comparison(
    trials,
    patient=None,
    controls=None,
    measure="proportion",
    near_only=True,
    panels=("accuracy", "boundary", "near", "carryover"),
    panel_size=(3.6, 3.6),
    save_as=None,
):
    """Patient and matched controls together, in one set of panels.

    ``controls`` is one control id or a list of them, defaulting to every
    configured matched control. Passing a single id is the usual case
    once a pairing has been decided; passing both is useful while that
    decision is still open, since it shows how much of any patient and
    control difference is just the spread between two healthy people of
    the same age.

    ``measure`` switches panel A between proportion correct and the
    signal detection pair, exactly as in ``plot_collision_indices``.
    Proportion correct collapses sensitivity and bias, which matters here
    because a patient whose near-hit and near-miss accuracy diverge can
    score an ordinary overall number.

    Returns the figure and the per-person table behind it.
    """
    d = _collision_frame(trials)

    patient, controls = _clinical_people(d, patient=patient,
                                         controls=controls)

    people = ([patient] if patient else []) + controls
    positions = np.arange(len(people))

    d = d[d["subject"].isin(people)]

    speeds = _speeds(d)
    colours = {s: _clinical_colour(s, patient, controls) for s in people}

    table = collision_indices_table(d, verbose=False)
    table = table.set_index("subject").reindex(people).reset_index()

    # Per-person PSEs, for the boundary panel.
    pse = {}
    for subject in people:
        indices = compute_collision_indices_one_session(
            d[d["subject"] == subject].copy()
        )
        pse[subject] = (indices["pse_slow"], indices["pse_fast"])

    # Sized from the panels rather than from a fixed page width. Four
    # panels sharing a two-column figure leaves each about 1.7 inches
    # across, which is narrower than the titles and rotated tick labels
    # they carry, so both collide with the neighbouring panel.
    fig, axes = plt.subplots(
        1, 4,
        figsize=(panel_size[0] * 4, panel_size[1]),
        gridspec_kw={"wspace": 0.45},
    )

    # --- A: accuracy, or sensitivity and bias ----------------------
    ax = axes[0]

    if measure == "proportion":
        columns = [
            ("accuracy", "n_correct", "n_trials", "Overall"),
            ("near_boundary_accuracy", "n_near_correct", "n_near", "Near"),
        ]

        width = 0.34
        offsets = (np.arange(len(columns)) - (len(columns) - 1) / 2) * width

        for index, (column, correct_column, total_column, label) in enumerate(
            columns
        ):
            for position, (_, row) in zip(positions, table.iterrows()):
                colour = colours[row["subject"]]
                low, high = _wilson(row[correct_column], row[total_column])

                ax.vlines(
                    position + offsets[index], low, high,
                    color=colour, linewidth=1.8, alpha=0.9, zorder=2,
                )
                ax.scatter(
                    position + offsets[index], row[column],
                    s=44, color=colour,
                    marker="o" if index == 0 else "s",
                    edgecolor="white", linewidth=0.7, zorder=3,
                )

            ax.scatter(
                [], [], s=44, color=style.MUTED,
                marker="o" if index == 0 else "s", label=label,
            )

        ax.axhline(0.5, linestyle="--", linewidth=0.8, color=style.MUTED,
                   zorder=0)
        ax.set_ylim(0.35, 1.02)
        ax.set_ylabel("Proportion correct")
        ax.set_title("Accuracy", loc="left", fontsize=9.5)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    elif measure == "sdt":
        prefix = "near_" if near_only else ""

        twin = ax.twinx()

        for position, (_, row) in zip(positions, table.iterrows()):
            colour = colours[row["subject"]]

            ax.scatter(
                position - 0.14, row[f"{prefix}d_prime"],
                s=44, color=colour, marker="o",
                edgecolor="white", linewidth=0.7, zorder=3,
            )
            twin.scatter(
                position + 0.14, row[f"{prefix}criterion"],
                s=44, color=colour, marker="s",
                edgecolor="white", linewidth=0.7, zorder=3,
            )

        twin.axhline(0, linestyle="--", linewidth=0.8, color=style.MUTED,
                     zorder=0)

        ax.scatter([], [], s=44, color=style.MUTED, marker="o", label="d'")
        ax.scatter([], [], s=44, color=style.MUTED, marker="s", label="c")

        ax.set_ylabel("d'")
        twin.set_ylabel("c (positive: leans toward MISS)", fontsize=7.5)
        ax.set_title(
            "Sensitivity and bias"
            + (", near trials" if near_only else ", all trials"),
            loc="left", fontsize=9.5,
        )
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
        style.clean(twin)

    else:
        raise ValueError(
            f"measure must be 'proportion' or 'sdt', got {measure!r}"
        )

    # --- B: the two PSEs, joined --------------------------------------
    ax = axes[1]

    for position, subject in zip(positions, people):
        pse_slow, pse_fast = pse[subject]

        if not (np.isfinite(pse_slow) and np.isfinite(pse_fast)):
            continue

        colour = colours[subject]

        ax.plot(
            [0, 1], [pse_slow, pse_fast],
            color=colour, linewidth=1.6, alpha=0.9, zorder=2,
        )
        ax.scatter(
            [0, 1], [pse_slow, pse_fast],
            s=42, color=colour, edgecolor="white", linewidth=0.7, zorder=3,
        )
        ax.annotate(
            f" {_display_name(subject, patient, controls)}",
            (1, pse_fast), fontsize=7, va="center", color=colour,
        )

    ax.axhline(0, linestyle="--", linewidth=0.8, color=style.MUTED, zorder=0)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([_label_speed(s) for s in ["slow", "fast"]])
    ax.set_xlim(-0.35, 1.55)
    ax.set_ylabel("Collision boundary (cm)")
    ax.set_title("Boundary at each speed", loc="left", fontsize=9.5)
    style.clean(ax)

    # --- C: near-category accuracy at each speed ----------------------
    ax = axes[2]

    near = d[d["trial_category"].isin(NEAR_CATEGORIES)]

    by_speed = (
        near.groupby(["subject", "trial_category", "speed"])["correct"]
        .agg(correct="sum", n="size").reset_index()
    )
    by_speed["proportion"] = by_speed["correct"] / by_speed["n"]

    speed_position = {speed: index for index, speed in enumerate(speeds)}
    marker = {"near_hit": "o", "near_miss": "s"}
    category_offset = {"near_hit": -0.12, "near_miss": 0.12}

    for _, row in by_speed.iterrows():
        colour = colours[row["subject"]]
        x = speed_position[row["speed"]] + category_offset[
            row["trial_category"]
        ]

        ax.scatter(
            x, row["proportion"],
            s=40, color=colour, marker=marker[row["trial_category"]],
            edgecolor="white", linewidth=0.7, zorder=3,
        )

    # Join each person's two speeds, so a flat pair reads as flat.
    for subject in people:
        for category in NEAR_CATEGORIES:
            line = by_speed[
                (by_speed["subject"] == subject)
                & (by_speed["trial_category"] == category)
            ]

            if len(line) < 2:
                continue

            line = line.set_index("speed").reindex(speeds)

            ax.plot(
                [speed_position[s] + category_offset[category] for s in speeds],
                line["proportion"],
                color=colours[subject], linewidth=1.2, alpha=0.7, zorder=2,
            )

    for category in NEAR_CATEGORIES:
        ax.scatter(
            [], [], s=40, color=style.MUTED, marker=marker[category],
            label=_label_category(category),
        )

    ax.axhline(0.5, linestyle="--", linewidth=0.8, color=style.MUTED, zorder=0)

    ax.set_xticks(list(speed_position.values()))
    ax.set_xticklabels([_label_speed(s) for s in speeds])
    ax.set_xlim(-0.5, len(speeds) - 0.5)
    ax.set_ylim(0.35, 1.02)
    ax.set_ylabel("Proportion correct")
    ax.set_title("Near-category accuracy", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    style.clean(ax)

    # --- D: carryover -------------------------------------------------
    ax = axes[3]

    for position, (_, row) in zip(positions, table.iterrows()):
        colour = colours[row["subject"]]

        if row["carryover_ok"] and np.isfinite(row["carryover_cm"]):
            ax.scatter(
                position, row["carryover_cm"],
                s=44, color=colour, edgecolor="white", linewidth=0.7,
                zorder=3,
            )
        else:
            ax.scatter(
                position, 0,
                s=44, facecolors="none", edgecolors=colour, linewidth=1.0,
                zorder=3,
            )

    ax.axhline(0, linestyle="--", linewidth=0.8, color=style.MUTED, zorder=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [_display_name(s, patient, controls) for s in people],
        rotation=20, ha="right", fontsize=8,
    )
    ax.set_xlim(-0.6, len(people) - 0.4)
    ax.set_ylabel("Carryover (cm)")
    ax.set_title("Post-switch carryover", loc="left", fontsize=9.5)
    style.clean(ax)

    # Panels A and D are indexed by person; B and C are not.
    for ax in [axes[0]]:
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [_display_name(s, patient, controls) for s in people],
            rotation=20, ha="right", fontsize=8,
        )
        ax.set_xlim(-0.6, len(people) - 0.4)
        style.clean(ax)

    label = (
        f"{patient} against {', '.join(controls)}"
        if patient and controls
        else ", ".join(people)
    )

    fig.suptitle(
        f"Collision task: {label}",
        x=0.02, ha="left", y=1.03, color=style.INK,
    )

    figures.label_panels(list(axes))
    style.finish_figure(fig, tight=False)

    if save_as:
        figures.save(fig, save_as)

    return fig, table

def _fit_psychometric(one):
    """Fit the offset-plus-speed logistic to one person's trials.

    Returns the coefficients and the two PSEs, or None when the fit
    fails or the offset slope is flat. A flat slope means P(HIT) does not
    change across the offset range, so there is no boundary to locate and
    a drawn curve would be a straight line pretending otherwise.
    """
    one = one.copy()
    one["speed_code"] = code_speed(one["speed"])

    one = one.dropna(
        subset=["offset_from_edge_cm", "speed_code", "response_hit"]
    )

    if (
        len(one) < 6
        or one["speed_code"].nunique() < 2
        or one["response_hit"].nunique() < 2
    ):
        return None

    model = safe_logistic_regression(
        one[["offset_from_edge_cm", "speed_code"]].to_numpy(),
        one["response_hit"].astype(int).to_numpy(),
    )

    if model is None:
        return None

    b0 = float(model.intercept_[0])
    b_offset = float(model.coef_[0][0])
    b_speed = float(model.coef_[0][1])

    if np.isclose(b_offset, 0):
        return None

    return {
        "b0": b0,
        "b_offset": b_offset,
        "b_speed": b_speed,
        "pse_slow": -(b0 - b_speed) / b_offset,
        "pse_fast": -(b0 + b_speed) / b_offset,
    }


def plot_clinical_psychometric(
    trials,
    patient=None,
    controls=None,
    n_bins=7,
    max_columns=3,
    panel_size=(3.4, 3.0),
    ball_radius_cm=0.0,
    save_as=None,
):
    """One psychometric panel per person, both speeds drawn together.

    The dashed curve is Slow and the solid one Fast, in that person's
    colour. The vertical drop lines mark each PSE, and the arrow between
    them is Delta_coll: rightward means the judged boundary moved outward
    for a fast ball, leftward means it contracted toward the body.

    Axes are shared, so the panels can be read against one another rather
    than each on its own scale.

    The figure grows with the number of people rather than dividing a
    fixed width among them: a curve and its two dotted boundary drops
    need room, and at nine panels across a journal column each one is
    too narrow to show where the curves cross. ``max_columns`` sets where
    the grid wraps and ``panel_size`` the inches given to each panel, so
    a figure that still looks cramped can be widened without editing the
    function.
    """
    d = _collision_frame(trials)

    # Offsets are stored to the ball's centre, one radius outside the
    # surface gap the task used. Subtracting the radius puts this figure
    # on the same axis as the response map, where zero is the point of
    # contact; leaving them different means two figures label the same
    # axis identically and mean different things.
    if ball_radius_cm:
        d = d.copy()
        d["offset_from_edge_cm"] = d["offset_from_edge_cm"] - ball_radius_cm

    patient, controls = _clinical_people(d, patient=patient,
                                         controls=controls)

    people = ([patient] if patient else []) + controls

    d = d[d["subject"].isin(people)]

    offset_low = float(d["offset_from_edge_cm"].min())
    offset_high = float(d["offset_from_edge_cm"].max())

    smooth_x = np.linspace(offset_low, offset_high, 200)
    bin_edges = np.linspace(offset_low, offset_high, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # A fixed width divided among nine panels leaves each too narrow to
    # read, so the figure grows with the number of people instead. Above
    # four it wraps, since a single row of nine is unreadable at any
    # width a page allows.
    columns = min(len(people), max_columns)
    rows = int(np.ceil(len(people) / columns))

    fig, axes = plt.subplots(
        rows, columns,
        figsize=(panel_size[0] * columns, panel_size[1] * rows),
        sharex=True, sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, subject in zip(axes, people):
        one = d[d["subject"] == subject]

        fit = _fit_psychometric(one)

        # --- the responses themselves ------------------------------
        # Each panel holds one person, so the person is already named in
        # the title and colour is free to carry the speed instead. That
        # is the project's scheme everywhere else, and it makes the two
        # curves separable without relying on the dash pattern, which
        # disappears at small sizes.
        for speed in _speeds(one):
            one_speed = one[one["speed"] == speed]

            which = np.clip(
                np.digitize(one_speed["offset_from_edge_cm"], bin_edges) - 1,
                0, len(bin_centres) - 1,
            )

            binned = pd.DataFrame({
                "bin": which,
                "hit": one_speed["response_hit"].astype(int).to_numpy(),
            }).groupby("bin")["hit"].agg(["mean", "size"])

            ax.scatter(
                bin_centres[binned.index],
                binned["mean"],
                s=8 + binned["size"] * 0.7,
                color=style.SPEED[speed],
                alpha=0.55,
                edgecolor="none",
                zorder=3,
            )

        # --- the fitted curves -------------------------------------
        if fit is None:
            ax.text(
                0.5, 0.5, "no usable fit",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color=style.MUTED,
            )
        else:
            for speed, code, pse in [
                (config.SLOW_LABEL, -1, fit["pse_slow"]),
                (config.FAST_LABEL, +1, fit["pse_fast"]),
            ]:
                curve = expit(
                    fit["b0"] + fit["b_speed"] * code
                    + fit["b_offset"] * smooth_x
                )

                ax.plot(
                    smooth_x, curve,
                    color=style.SPEED[speed], linewidth=1.8,
                    label=_label_speed(speed),
                    zorder=2,
                )

                ax.plot(
                    [pse, pse], [0, 0.5],
                    linestyle=":", linewidth=1.0,
                    color=style.SPEED[speed], zorder=1,
                )

            ax.annotate(
                "",
                xy=(fit["pse_fast"], 0.5),
                xytext=(fit["pse_slow"], 0.5),
                arrowprops=dict(arrowstyle="<->", color=style.INK, lw=0.9),
            )

            ax.text(
                0.5 * (fit["pse_slow"] + fit["pse_fast"]),
                0.55,
                f"$\\Delta_{{coll}}$ = "
                f"{fit['pse_fast'] - fit['pse_slow']:.1f} cm",
                ha="center", fontsize=7, color=style.INK,
            )

        ax.axhline(0.5, linestyle="--", linewidth=0.8, color=style.MUTED,
                   zorder=0)
        ax.axvline(0, linestyle="--", linewidth=0.8, color=style.MUTED,
                   zorder=0)

        ax.set_ylim(-0.02, 1.02)
        ax.set_title(
            _display_name(subject, patient, controls),
            loc="left", fontsize=9,
        )
        style.clean(ax)

    for ax in axes[len(people):]:
        ax.set_visible(False)

    # One legend and one axis label for the whole grid. Repeated on every
    # panel they take more room than the curves and say nothing new.
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")

    for ax in axes[:len(people)]:
        ax.set_xlabel("")

    fig.supxlabel(
        "Clearance from the shoulder edge (cm)"
        if ball_radius_cm else "Offset from shoulder edge (cm)",
        fontsize=9,
    )
    fig.supylabel("P(respond HIT)", fontsize=9)

    fig.suptitle(
        "Collision psychometric functions",
        x=0.02, ha="left", y=1.02, color=style.INK,
    )

    fig.text(
        0.5, -0.03,
        "Dotted drops mark each boundary. "
        "Point area is the trials in that bin",
        ha="center", va="top", fontsize=8, color=style.MUTED,
    )

    figures.label_panels(list(axes[:len(people)]))
    style.finish_figure(fig, tight=False)

    if save_as:
        figures.save(fig, save_as)

    return fig

def plot_clinical_psychometric_by_speed(
    trials,
    patient=None,
    controls=None,
    n_bins=7,
    panel_size=(4.2, 3.8),
    ball_radius_cm=0.0,
    save_as=None,
):
    """One panel per speed, one fitted curve per person.

    Axes are shared, so the Slow and Fast panels can be read against each
    other. Each person keeps their colour from the other clinical
    figures: the patient in the patient colour, the matched controls in
    the age-matched one.

    The points behind each curve are that person's binned responses at
    that speed, with area tracking the trials in the bin. They are the
    check on the fit: the model gives both speeds one shared slope, so a
    person whose points are visibly steeper at one speed is not described
    by the curve drawn through them.
    """
    d = _collision_frame(trials)

    # Offsets are stored to the ball's centre, one radius outside the
    # surface gap the task used. Shifting here keeps this figure on the
    # same axis as the response map and the per-person psychometric
    # panels, so the same trial sits at the same offset in all three.
    if ball_radius_cm:
        d = d.copy()
        d["offset_from_edge_cm"] = d["offset_from_edge_cm"] - ball_radius_cm

    patient, controls = _clinical_people(d, patient=patient,
                                         controls=controls)

    people = ([patient] if patient else []) + controls

    d = d[d["subject"].isin(people)]

    speeds = _speeds(d)

    offset_low = float(d["offset_from_edge_cm"].min())
    offset_high = float(d["offset_from_edge_cm"].max())

    smooth_x = np.linspace(offset_low, offset_high, 200)
    bin_edges = np.linspace(offset_low, offset_high, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Fit once per person; both panels draw from the same model, so a
    # curve in the Slow panel and its partner in the Fast panel are two
    # views of one fit rather than two independent ones.
    fits = {
        subject: _fit_psychometric(d[d["subject"] == subject])
        for subject in people
    }

    # Sized from the panels rather than from a fixed page width: every
    # person contributes a curve to both panels, and at six or more the
    # curves separate over a few centimetres of the axis that a narrow
    # panel cannot resolve.
    fig, axes = plt.subplots(
        1, len(speeds),
        figsize=(panel_size[0] * len(speeds), panel_size[1]),
        sharex=True, sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    speed_code = {config.SLOW_LABEL: -1, config.FAST_LABEL: +1}

    for ax, speed in zip(axes, speeds):
        for subject in people:
            colour = _clinical_colour(subject, patient, controls)

            one_speed = d[
                (d["subject"] == subject) & (d["speed"] == speed)
            ]

            if len(one_speed):
                which = np.clip(
                    np.digitize(
                        one_speed["offset_from_edge_cm"], bin_edges,
                    ) - 1,
                    0, len(bin_centres) - 1,
                )

                binned = pd.DataFrame({
                    "bin": which,
                    "hit": one_speed["response_hit"].astype(int).to_numpy(),
                }).groupby("bin")["hit"].agg(["mean", "size"])

                ax.scatter(
                    bin_centres[binned.index],
                    binned["mean"],
                    s=8 + binned["size"] * 0.7,
                    color=colour, alpha=0.45,
                    edgecolor="none", zorder=2,
                )

            fit = fits[subject]

            if fit is None:
                continue

            code = speed_code.get(speed, 0)

            ax.plot(
                smooth_x,
                expit(
                    fit["b0"] + fit["b_speed"] * code
                    + fit["b_offset"] * smooth_x
                ),
                color=colour, linewidth=1.8,
                label=_display_name(subject, patient, controls),
                zorder=3,
            )

            pse = fit["pse_slow"] if code < 0 else fit["pse_fast"]

            ax.plot(
                [pse, pse], [0, 0.5],
                linestyle=":", linewidth=1.0, color=colour, zorder=1,
            )

        ax.axhline(0.5, linestyle="--", linewidth=0.8, color=style.MUTED,
                   zorder=0)
        ax.axvline(0, linestyle="--", linewidth=0.8, color=style.MUTED,
                   zorder=0)

        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("")
        ax.set_title(_label_speed(speed), loc="left", fontsize=9.5)
        style.clean(ax)

    axes[0].set_ylabel("P(respond HIT)")

    # One legend, outside the panels. With six or nine curves it covers
    # the middle of whichever panel it sits in, which is exactly where
    # the curves separate.
    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles, labels, frameon=False, fontsize=8,
            loc="upper left", bbox_to_anchor=(0.99, 0.92),
        )
        # tight_layout does not account for a figure legend, so the room
        # it needs is taken from the axes here.
        # The reserved strip scales with the number of curves: at two
        # people a fixed 22% leaves the panels needlessly narrow.
        fig.subplots_adjust(right=1.0 - min(0.10 + 0.03 * len(labels), 0.24))

    fig.supxlabel(
        "Clearance from the shoulder edge (cm)"
        if ball_radius_cm else "Offset from shoulder edge (cm)",
        fontsize=9,
    )

    fig.suptitle(
        "Collision psychometric functions by speed",
        x=0.02, ha="left", y=1.03, color=style.INK,
    )

    fig.text(
        0.5, -0.05,
        "Dotted drops mark each person's boundary. "
        "Point area is the trials in that bin",
        ha="center", va="top", fontsize=8, color=style.MUTED,
    )

    figures.label_panels(list(axes))
    style.finish_figure(fig, tight=False)

    if save_as:
        figures.save(fig, save_as)

    return fig


def plot_clinical_accuracy(
    trials,
    patient=None,
    controls=None,
    bin_width=4.0,
    min_trials=5,
    ball_radius_cm=0.0,
    save_as=None,
):
    """Accuracy outward from the boundary, and near-category accuracy.

    Panels A and B are proportion correct against distance from the
    boundary, one line per person, Fast then Slow. Splitting by speed
    rather than pooling is what lets the panels answer the question the
    task was built for: whether the extra time a slow ball affords is
    used. A person who uses it sits higher in B than in A; a person who
    does not sits at the same height in both.

    Panel C is near-hit and near-miss accuracy per person, in the
    category colours, with 95% Wilson intervals. The two categories are
    separated because a person can be accurate on one and poor on the
    other, and averaging them into a single near-boundary number hides
    exactly that. A gap between the two markers is a response bias: it
    means the person is answering one way more readily than the evidence
    warrants.

    A bin holding fewer than ``min_trials`` trials for a person is
    dropped for that person rather than drawn on two or three responses.
    """
    d = _collision_frame(trials)

    # Offsets are stored to the ball's centre, one radius outside the
    # surface gap the task used. Shifting keeps every collision figure on
    # the same axis, so a trial sits at the same offset wherever it is
    # drawn.
    if ball_radius_cm:
        d = d.copy()
        d["offset_from_edge_cm"] = d["offset_from_edge_cm"] - ball_radius_cm

    patient, controls = _clinical_people(d, patient=patient,
                                         controls=controls)

    people = ([patient] if patient else []) + controls
    d = d[d["subject"].isin(people)]

    speeds = _speeds(d)
    fig, axes = plt.subplots(
        1, 3,
        figsize=(figures.DOUBLE_COLUMN, 3.2),
        gridspec_kw={"wspace": 0.34},
    )

    # --- A and B: accuracy by signed offset, one panel per speed ---
    # Fast first: it is the harder condition, so the slow panel reads as
    # what the extra time bought.
    ordered_speeds = [s for s in [config.FAST_LABEL, config.SLOW_LABEL]
                      if s in speeds]

    for ax, speed in zip(axes[:2], ordered_speeds):
        _draw_signed_accuracy(
            ax, d[d["speed"] == speed], people, bin_width, min_trials,
            by_speed=False, patient=patient, controls=controls,
            ball_radius_cm=ball_radius_cm,
        )

        ax.set_title(_label_speed(speed), fontsize=9.5)

    # One range across both, so the panels can be read against each other.
    if len(axes[:2]) > 1:
        low = min(ax.get_xlim()[0] for ax in axes[:2])
        high = max(ax.get_xlim()[1] for ax in axes[:2])

        for ax in axes[:2]:
            ax.set_xlim(low, high)

    # The two panels hold the same people and the same axis, so one
    # legend and one x label serve both. Repeating them takes space from
    # the curves and invites the reader to check whether the second copy
    # says something different.
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")

    axes[1].set_ylabel("")
    axes[1].set_xlabel("")
    axes[0].set_xlabel("")

    # --- C: near categories per person -----------------------------
    ax = axes[2]

    # All four types, not just the near ones. The clear trials are the
    # check that a low near score is a hard judgement rather than a
    # participant who was not doing the task: someone at ceiling on clear
    # hits and clear misses understood it and found the near ones hard.
    per_cell = (
        d.groupby(["subject", "trial_category"])["correct"]
        .agg(correct="sum", n="size").reset_index()
    )
    per_cell["proportion"] = per_cell["correct"] / per_cell["n"]

    drawn = [c for c in TRIAL_CATEGORIES
             if c in set(per_cell["trial_category"])]

    width = 0.80 / max(len(drawn), 1)
    offsets = (np.arange(len(drawn)) - (len(drawn) - 1) / 2) * width

    for index, category in enumerate(drawn):
        colour = _category_colour(category)
        rows = per_cell[per_cell["trial_category"] == category]

        for _, row in rows.iterrows():
            position = people.index(row["subject"]) + offsets[index]
            low, high = _wilson(row["correct"], row["n"])

            ax.vlines(
                position, low, high,
                color=colour, linewidth=2.0, alpha=0.9, zorder=2,
            )
            ax.scatter(
                position, row["proportion"],
                s=46, color=colour, edgecolor="white", linewidth=0.7,
                zorder=3,
            )

        ax.plot([], [], color=colour, linewidth=3,
                label=_label_category(category))

    ax.axhline(0.5, linestyle="--", linewidth=0.8, color=style.MUTED, zorder=0)

    ax.set_xticks(np.arange(len(people)))
    ax.set_xticklabels(
        [_display_name(s, patient, controls) for s in people],
        rotation=20, ha="right", fontsize=8,
    )
    ax.set_xlim(-0.6, len(people) - 0.4)
    ax.set_ylim(0.30, 1.02)
    ax.set_ylabel("Proportion correct")
    ax.set_title("Correct by trial type, 95% Wilson", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    style.clean(ax)

    fig.suptitle(
        "Collision judgements near the body edge",
        x=0.02, ha="left", y=1.03, color=style.INK,
    )

    # Centred under panels A and B, which share the axis. Panel C has a
    # different x axis and is deliberately left out of it.
    left = axes[0].get_position().x0
    right = axes[1].get_position().x1

    fig.text(
        0.5 * (left + right), 0.005,
        "Clearance from the shoulder edge (cm)"
        if ball_radius_cm else "Offset from the body edge (cm)",
        ha="center", va="top", fontsize=9, color=style.INK,
    )

    fig.text(
        0.5, -0.07,
        "Near trials only. Where a curve crosses 0.5 is that person's "
        "judged boundary. Point area is the trials in that bin",
        ha="center", va="top", fontsize=8, color=style.MUTED,
    )

    figures.label_panels(list(axes))
    style.finish_figure(fig, tight=False)

    if save_as:
        figures.save(fig, save_as)

    return fig


def plot_clinical_rt(
    trials,
    patient=None,
    controls=None,
    ceiling=None,
    save_as=None,
):
    """Reaction time by trial category, one panel per speed.

    Within each panel every category carries one box per person, in that
    person's colour, so the comparison runs across the pair at a fixed
    category and speed.

    The speeds keep their own panels and their own axes. A slow ball is
    in flight more than twice as long, so a shared axis would show the
    flight-time difference and leave the differences between people
    compressed into a fraction of the range.

    ``ceiling`` is a dict of speed to upper limit in ms, defaulting to
    ``RT_SUBJECT_CEILING_MS``. Trials above it are counted in the panel
    title rather than drawn.
    """
    d = _collision_frame(trials)

    if "rt_ms" not in d.columns:
        raise KeyError("No 'rt_ms' column in the collision trials.")

    d = d.dropna(subset=["rt_ms"])

    patient, controls = _clinical_people(d, patient=patient,
                                         controls=controls)

    people = ([patient] if patient else []) + controls
    d = d[d["subject"].isin(people)]

    speeds = _speeds(d)
    categories = _categories(d)

    limits = dict(RT_SUBJECT_CEILING_MS)

    if ceiling:
        limits.update(ceiling)

    fig, axes = plt.subplots(
        1, len(speeds),
        figsize=(figures.DOUBLE_COLUMN, 3.2),
        sharex=True, sharey=False,
        squeeze=False,
    )
    axes = axes[0]

    width = 0.72 / max(len(people), 1)
    offsets = (np.arange(len(people)) - (len(people) - 1) / 2) * width

    for ax, speed in zip(axes, speeds):
        one_speed = d[d["speed"] == speed]

        for index, subject in enumerate(people):
            colour = _clinical_colour(subject, patient, controls)

            for position, category in enumerate(categories):
                values = one_speed.loc[
                    (one_speed["subject"] == subject)
                    & (one_speed["trial_category"] == category),
                    "rt_ms",
                ].dropna().to_numpy(dtype=float)

                if not values.size:
                    continue

                _draw_category_box(
                    ax, values, position + offsets[index], colour,
                    width=width * 0.82,
                )

            ax.plot(
                [], [], color=colour, linewidth=3,
                label=_display_name(subject, patient, controls),
            )

        top = limits.get(speed)

        title = _label_speed(speed)

        if top is not None:
            above = int((one_speed["rt_ms"] > top).sum())
            ax.set_ylim(top=top)

            if above:
                title = f"{title}, {above} trials above axis"

        ax.set_xticks(np.arange(len(categories)))
        ax.set_xticklabels(
            [_label_category(c) for c in categories],
            rotation=20, ha="right", fontsize=8,
        )
        ax.set_xlim(-0.6, len(categories) - 0.4)
        ax.set_ylabel("Reaction time (ms)")
        ax.set_title(title, fontsize=9.5)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        style.clean(ax)

    fig.suptitle(
        "Collision reaction time", x=0.02, ha="left", y=1.03, color=style.INK,
    )

    figures.label_panels(list(axes))
    style.finish_figure(fig, tight=False)

    if save_as:
        figures.save(fig, save_as)

    return fig


# The task positions each ball so that its nearest surface sits at the
# specified gap from the shoulder edge, which puts the centre one radius
# further out (TrialGenerator: centre = ballRadius + edgeGap). The offset
# column measures the centre, so subtracting the radius recovers the gap
# the task worked in, and zero then means the ball grazes the body rather
# than its centre crossing the edge.
BALL_RADIUS_CM = 8.75


def plot_clinical_response_map(
    trials,
    patient=None,
    controls=None,
    categories=None,
    bin_width=2.0,
    ball_radius_cm=BALL_RADIUS_CM,
    figure_width=None,
    by_speed=True,
    mark_categories=True,
    show_counts=False,
    show_note=False,
    ax=None,
    save_as=None,
):
    """P(HIT) as a heat map: one row per person and speed.

    One column per band of offset from the body edge. Orange is a HIT
    judgement, purple a MISS, and the pale middle is where the person was
    split. Reading a row left to right is that person's transition at
    that speed; reading down a column is how the four differ at a fixed
    offset.

    Two vertical references, and they are not the same place. The solid
    line is the body edge: a trajectory left of it would have struck the
    body. The dashed line is where near-hit trials end and near-miss
    trials begin, which sits a few centimetres outside the edge because
    the two categories are defined by trajectory rather than by outcome.
    Marking only the body edge invites the columns just outside it to be
    read as errors of the same kind on both sides of the map, when a HIT
    response left of the edge is correct and right of it is a false
    alarm.

    ``bin_width`` defaults to 2 cm rather than 4 because the near-hit
    range spans roughly 14 cm: at 4 cm bins the whole of it collapses
    into two columns, and a difference between people in how they treat
    trajectories that would have struck them is not resolvable.

    Rows run patient first, then each control, with a person's two speeds
    adjacent, so the shift between speeds is read vertically.

    Only the near trials are drawn by default. Clear hits and clear
    misses are answered at ceiling several bands out and would fill both
    ends of the map with solid colour.
    """
    d = _collision_frame(trials)

    patient, controls = _clinical_people(d, patient=patient,
                                         controls=controls)

    people = ([patient] if patient else []) + controls
    d = d[d["subject"].isin(people)]

    if categories is None:
        categories = NEAR_CATEGORIES

    d = d[d["trial_category"].isin(categories)]

    # Shift to the surface gap before binning, so the axis, the zero
    # line and the category division all refer to the same quantity the
    # task used. Without this the solid line sits one ball radius inside
    # the point where contact begins, and the near-hit band straddles it
    # for no reason a reader could infer.
    if ball_radius_cm:
        d = d.copy()
        d["offset_from_edge_cm"] = (
            d["offset_from_edge_cm"] - ball_radius_cm
        )

    speeds = _speeds(d)
    labels, which = _map_bins(d, width=bin_width)
    d["offset_bin"] = which

    # Speeds merged into one row per person when the question is about
    # trial category rather than about speed: splitting halves the trials
    # in every cell, and a difference present at both speeds is carried
    # by neither alone.
    rows = (
        [(subject, speed) for subject in people for speed in speeds]
        if by_speed else [(subject, None) for subject in people]
    )

    grid = np.full((len(rows), len(labels)), np.nan)
    counts = np.zeros((len(rows), len(labels)))

    for index, (subject, speed) in enumerate(rows):
        cell = d[d["subject"] == subject]

        if speed is not None:
            cell = cell[cell["speed"] == speed]

        binned = cell.groupby("offset_bin")["said_hit"].agg(["mean", "size"])

        for bin_index, row in binned.iterrows():
            grid[index, bin_index] = row["mean"]
            counts[index, bin_index] = row["size"]

    # Drop bands nobody has, so the map does not open with empty columns.
    filled = counts.sum(axis=0) > 0
    grid = grid[:, filled]
    counts = counts[:, filled]
    labels = [label for label, keep in zip(labels, filled) if keep]

    # ``ax`` lets the map become one panel of a larger figure. When it is
    # given, the caller owns the figure: no title, no colourbar and no
    # saving happen here, since those belong to the whole figure rather
    # than to this panel.
    owns_figure = ax is None

    if owns_figure:
        fig, ax = plt.subplots(
            figsize=(
                figure_width or figures.DOUBLE_COLUMN,
                0.45 * len(rows) + 2.0,
            ),
        )
    else:
        fig = ax.get_figure()

    image = ax.imshow(
        grid, aspect="auto", cmap=_hit_miss_colormap(),
        vmin=0, vmax=1, interpolation="nearest",
    )

    # The trial count in each cell, so a striking colour resting on three
    # responses is not read as a striking result. Off by default once the
    # speeds are merged and the bins are wide, since the counts are then
    # even enough that printing them costs more legibility than it buys.
    if show_counts:
        for row_index in range(grid.shape[0]):
            for column_index in range(grid.shape[1]):
                n = int(counts[row_index, column_index])

                if not n:
                    continue

                ax.text(
                    column_index, row_index, str(n),
                    ha="center", va="center", fontsize=6,
                    color=style.INK, alpha=0.55,
                )

    ax.axvline(_edge_position(labels), color=style.INK, linewidth=1.4,
               alpha=0.8)

    # Where the two trial categories meet, taken from the data rather
    # than assumed: near-hit and near-miss are defined by trajectory, so
    # the division between them sits outside the body edge and a reader
    # cannot infer it from the edge line alone. Without it the columns to
    # the right of the edge read as the same kind of trial as those to
    # the left, when a HIT response is correct on one side and a false
    # alarm on the other.
    if mark_categories and {"near_hit", "near_miss"} <= set(d["trial_type"]):
        boundary = 0.5 * (
            d.loc[d["trial_type"] == "near_hit", "offset_from_edge_cm"].max()
            + d.loc[d["trial_type"] == "near_miss",
                    "offset_from_edge_cm"].min()
        )

        edges = np.asarray([float(label) for label in labels])
        position = float(np.interp(
            boundary, edges, np.arange(len(labels)),
            left=-0.5, right=len(labels) - 0.5,
        ))

        ax.axvline(position, color=style.INK, linewidth=1.0,
                   linestyle="--", alpha=0.55)

        ax.text(
            position, -0.85, "near hit  |  near miss",
            ha="center", va="bottom", fontsize=7, color=style.MUTED,
        )

    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(
        [
            _display_name(subject, patient, controls)
            + (f", {_label_speed(speed)}" if speed is not None else "")
            for subject, speed in rows
        ],
        fontsize=8,
    )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)

    # The axis label names the adjustment rather than leaving it in the
    # caption: the numbers on this axis are not the offsets stored in the
    # data, and a reader comparing this figure against a table of raw
    # offsets would otherwise find a constant discrepancy with nothing to
    # explain it.
    ax.set_xlabel(
        "Clearance from the shoulder edge (cm)"
        if ball_radius_cm else "Offset from the body edge (cm)"
    )

    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)

    if owns_figure:
        bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        bar.set_label("P(respond HIT)")

        fig.suptitle(
            "Where each person switched from HIT to MISS",
            x=0.02, ha="left", color=style.INK,
        )

    # The footnote is off by default: the colourbar names the quantity,
    # the axis label names the adjustment, and at this width the note
    # runs past the figure. It belongs in the caption, where it can be
    # set in the document's own measure.
    if show_note:
        note = "Orange: called it a hit. Purple: called it a miss. "

        note += (
            f"Offsets are the gap between the ball surface and the "
            f"shoulder edge, one ball radius ({ball_radius_cm:.2f} cm) "
            f"inside the stored trajectory offset"
            if ball_radius_cm else
            "The line is the body edge"
        )

        fig.text(
            0.5, -0.02, note,
            ha="center", va="top", fontsize=8, color=style.MUTED,
        )

    if owns_figure:
        style.finish_figure(fig, tight=False)

        if save_as:
            figures.save(fig, save_as)

    return fig, image