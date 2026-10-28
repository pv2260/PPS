"""Where in the ball's flight the response was made, and where the ball was.

Two ways of asking the same question, kept in one module because they
share the flight computation and would otherwise drift apart:

    plot_flight_timing     when the answer came, as a fraction of the
                           flight, by trial category
    plot_response_heatmap  where the ball was when it came, laid out
                           along the path it travelled
    plot_response_fan      the same, drawn as the four trajectory bands
                           of the task figure

The reaction time alone says how long someone took. What it does not say
is how much of the approach they had seen, and that differs by a factor
of more than two between the two speeds: a slow ball takes 3.33 s to
arrive and a fast one 1.43 s, so the same 800 ms response commits after a
quarter of the slow flight and more than half the fast one.

Three ways of putting the same response on the ball's timeline:

    fraction   how far through the flight, from 0 at motion onset to 1 at
               arrival
    remaining  how far the ball still had to travel, in metres
    elapsed    the raw reaction time, kept for reference

The fraction is the one to compare across speeds. The elapsed time is the
one to compare within a speed, since it is what the participant
controlled.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import style

from ppsanalysis import figures


# Flight duration from the ball's own start distance and speed, rather
# than assumed: the trajectories fan outward, so path length varies
# slightly and a fixed constant would be wrong for the widest ones.
START_DISTANCE_M = 5.0

CATEGORY_ORDER = ["clear_hit", "near_hit", "near_miss", "clear_miss"]

CATEGORY_LABEL = {
    "clear_hit": "Clear hit",
    "near_hit": "Near hit",
    "near_miss": "Near miss",
    "clear_miss": "Clear miss",
}


def flight_position(trials, start_distance_m=START_DISTANCE_M):
    """Add flight fraction and remaining distance to each trial.

    A response later than the flight duration is kept rather than
    discarded, with a fraction above 1: those are the trials answered
    after the ball had already arrived, and how many there are is part of
    what the figure is for.
    """
    d = trials.copy()

    # Checked by name rather than with .get, which returns a scalar
    # default when the column is absent and then fails on .isna().
    if "speed_mps" in d.columns:
        speed = pd.to_numeric(d["speed_mps"], errors="coerce")
    else:
        speed = pd.Series(np.nan, index=d.index)

    if speed.isna().all() and "speed" in d.columns:
        # Fall back to the labelled speed where the numeric column is
        # absent, using the two velocities the task used.
        speed = d["speed"].map({"slow": 1.5, "fast": 3.5})

    rt_s = pd.to_numeric(d["rt_ms"], errors="coerce") / 1000.0
    flight_s = start_distance_m / speed

    d["flight_duration_s"] = flight_s
    d["flight_fraction"] = rt_s / flight_s
    d["remaining_m"] = start_distance_m - (speed * rt_s)
    d["answered_after_arrival"] = d["flight_fraction"] > 1.0

    return d


def summarise(trials, by=("subject", "speed", "trial_type")):
    """Median flight fraction and remaining distance per cell."""
    d = flight_position(trials)
    d = d[d["rt_ms"].notna()]

    return (
        d.groupby(list(by), observed=True)
        .agg(
            n=("rt_ms", "size"),
            rt_ms=("rt_ms", "median"),
            flight_fraction=("flight_fraction", "median"),
            remaining_m=("remaining_m", "median"),
            after_arrival=("answered_after_arrival", "mean"),
        )
        .reset_index()
    )


def plot_flight_timing(
    trials,
    subjects=None,
    exclude=None,
    measure="flight_fraction",
    columns=3,
    panel_size=(3.4, 2.9),
    save_as=None,
    verbose=True,
):
    """One panel per participant: where in the flight each answer came.

    Trial categories on the x axis, the chosen measure on the y, with the
    two speeds drawn separately. Individual trials sit behind the cell
    medians so the spread within a category is visible.
    """
    d = trials.copy()

    if subjects is not None:
        d = d[d["subject"].isin(list(subjects))]

    if exclude:
        dropped = sorted(set(exclude) & set(d["subject"]))
        d = d[~d["subject"].isin(list(exclude))]

        if verbose and dropped:
            print(f"Excluded: {', '.join(dropped)}")

    if "usable" in d.columns:
        d = d[d["usable"].fillna(False).astype(bool)]

    d = flight_position(d)
    d = d[d["rt_ms"].notna()]

    people = sorted(d["subject"].dropna().unique())
    categories = [c for c in CATEGORY_ORDER if c in set(d["trial_type"])]

    rows = int(np.ceil(len(people) / columns))

    fig, axes = plt.subplots(
        rows, columns,
        figsize=(panel_size[0] * columns, panel_size[1] * rows),
        sharex=True, sharey=True, squeeze=False,
    )
    flat = axes.ravel()

    rng = np.random.default_rng(0)
    speeds = [s for s in ["slow", "fast"] if s in set(d["speed"])]
    offsets = dict(zip(speeds, np.linspace(-0.16, 0.16, len(speeds))))

    for ax, subject in zip(flat, people):
        one = d[d["subject"] == subject]

        for speed in speeds:
            colour = style.SPEED[speed] if hasattr(style, "SPEED") else (
                style.SLOW if speed == "slow" else style.FAST
            )
            medians = []

            for position, category in enumerate(categories):
                cell = one[
                    (one["speed"] == speed)
                    & (one["trial_type"] == category)
                ]

                if cell.empty:
                    continue

                y = cell[measure].to_numpy(dtype=float)
                centre = position + offsets[speed]

                ax.scatter(
                    centre + rng.normal(0, 0.035, size=len(y)), y,
                    s=8, color=colour, alpha=0.20, edgecolors="none",
                    zorder=1,
                )

                medians.append((centre, float(np.median(y))))

            if medians:
                ax.plot(
                    [x for x, _ in medians], [v for _, v in medians],
                    "o-", markersize=5, linewidth=2.0, color=colour,
                    zorder=3, label=speed.capitalize(),
                )

        # The ball arrives at 1.0 on the fraction scale and at 0 on the
        # remaining-distance scale. Either way the line marks the moment
        # after which an answer is too late to have been a judgement
        # about an approaching object.
        if measure == "flight_fraction":
            ax.axhline(1.0, linestyle="--", linewidth=0.9,
                       color=style.MUTED, zorder=0)
        elif measure == "remaining_m":
            ax.axhline(0.0, linestyle="--", linewidth=0.9,
                       color=style.MUTED, zorder=0)

        ax.set_title(subject, loc="left", fontsize=9)
        style.clean(ax)

    for ax in flat[len(people):]:
        ax.set_visible(False)

    labels = {
        "flight_fraction": "Fraction of flight elapsed",
        "remaining_m": "Distance still to travel (m)",
        "rt_ms": "Reaction time (ms)",
    }

    for ax in flat[:len(people)]:
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(
            [CATEGORY_LABEL.get(c, c) for c in categories],
            fontsize=7, rotation=30, ha="right",
        )

    for row in range(rows):
        axes[row, 0].set_ylabel(labels.get(measure, measure), fontsize=9)

    flat[0].legend(frameon=False, fontsize=7.5, loc="upper right")

    fig.suptitle(
        "When in the ball's flight the answer came",
        x=0.02, ha="left", color=style.INK,
    )
    style.finish_figure(fig)

    table = summarise(d)

    if verbose:
        print(table.round(3).to_string(index=False))

    if save_as:
        figures.save(fig, save_as)

    return fig, table


CATEGORY_ORDER = ["clear_hit", "near_hit", "near_miss", "clear_miss"]

CATEGORY_LABEL = {
    "clear_hit": "Hit",
    "near_hit": "Near hit",
    "near_miss": "Near miss",
    "clear_miss": "Miss",
}

# The greys of the task figure, darkest for the trajectories that strike
# the body. Keeping them matched means the two figures can sit on facing
# pages without the reader relearning the code.
CATEGORY_GREY = {
    "clear_hit": "#8a8a8a",
    "near_hit": "#a8a8a8",
    "near_miss": "#c9c9c9",
    "clear_miss": "#e4e4e4",
}


def response_position(trials, start_distance_m=START_DISTANCE_M):
    """Where the ball was when each answer came.

    A thin wrapper on ``flight_timing.flight_position`` so the two
    figures compute the flight from one place. Duplicating the speed
    lookup and the arithmetic would let the timing figure and the
    geometry figure disagree about the same trial after any change to
    either.

    ``distance_at_response_m`` is measured from the body, so it starts at
    the release distance and falls to zero on arrival. A negative value
    means the answer came after the ball had passed, which is kept rather
    than dropped: how often that happened is part of the picture.
    """
    d = flight_position(trials, start_distance_m=start_distance_m)

    d["distance_at_response_m"] = d["remaining_m"]
    d["after_arrival"] = d["answered_after_arrival"]

    return d


def plot_response_heatmap(
    trials,
    subject=None,
    by_speed=True,
    bins=20,
    start_distance_m=START_DISTANCE_M,
    save_as=None,
    verbose=True,
):
    """Response density along the flight path, one row per category.

    The x axis runs from the release point on the left to the body on the
    right, so the panel is read in the direction the ball travels. Each
    row is normalised within itself, since the categories hold different
    numbers of trials and the question is where within a category the
    answers fell, not which category had more.
    """
    d = response_position(trials, start_distance_m)

    if subject is not None:
        d = d[d["subject"] == subject]

    if "usable" in d.columns:
        d = d[d["usable"].fillna(False).astype(bool)]

    d = d[d["distance_at_response_m"].notna()]

    categories = [c for c in CATEGORY_ORDER if c in set(d["trial_type"])]
    speeds = (
        [s for s in ["slow", "fast"] if s in set(d["speed"])]
        if by_speed else [None]
    )

    # A perceptually even ramp from the page colour to the ink, so
    # density reads as density. A diverging map would imply a midpoint
    # that a proportion does not have.
    cmap = LinearSegmentedColormap.from_list(
        "density", ["#ffffff", "#bcd9e0", "#0097b2", "#00343f"],
    )

    fig, axes = plt.subplots(
        len(speeds), 1,
        figsize=(7.4, 1.1 * len(categories) * len(speeds) + 1.4),
        sharex=True, squeeze=False,
    )
    flat = axes.ravel()

    summary = []

    for ax, speed in zip(flat, speeds):
        cell = d if speed is None else d[d["speed"] == speed]

        grid = np.full((len(categories), bins), np.nan)

        for row, category in enumerate(categories):
            values = cell.loc[
                cell["trial_type"] == category, "distance_at_response_m"
            ].to_numpy(dtype=float)

            if not len(values):
                continue

            counts, _ = np.histogram(
                np.clip(values, 0.0, start_distance_m),
                bins=np.linspace(0.0, start_distance_m, bins + 1),
            )

            # Reversed so the left of the panel is the release point and
            # the right is the body, matching the direction of travel.
            grid[row] = counts[::-1] / max(counts.sum(), 1)

            summary.append({
                "speed": speed or "pooled",
                "trial_type": category,
                "n": int(len(values)),
                "median_distance_m": float(np.median(values)),
                "after_arrival": float(np.mean(values < 0)),
            })

        image = ax.imshow(
            grid, aspect="auto", cmap=cmap, origin="upper",
            extent=[start_distance_m, 0.0, len(categories) - 0.5, -0.5],
            vmin=0.0,
        )

        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(
            [CATEGORY_LABEL.get(c, c) for c in categories], fontsize=8,
        )

        # The body edge, and the point of contact one ball radius before
        # it. A response to the right of the body line came after the
        # ball had arrived.
        ax.axvline(0.0, color=style.INK, linewidth=1.4)
        ax.text(0.0, -0.72, "body", ha="center", va="bottom", fontsize=7.5,
                color=style.INK)
        ax.text(start_distance_m, -0.72, "release", ha="center",
                va="bottom", fontsize=7.5, color=style.MUTED)

        if speed is not None:
            ax.set_title(speed.capitalize(), loc="left", fontsize=9.5)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(length=0)

    flat[-1].set_xlabel("Distance of the ball from the body at response (m)")

    bar = fig.colorbar(image, ax=list(flat), fraction=0.025, pad=0.02)
    bar.set_label("Proportion of that category's responses")

    fig.suptitle(
        "Where the ball was when the answer came"
        + (f": {subject}" if subject else ""),
        x=0.02, ha="left", color=style.INK,
    )

    table = pd.DataFrame(summary)

    if verbose and len(table):
        print(table.round(3).to_string(index=False))

    if save_as:
        figures.save(fig, save_as)

    return fig, table


def plot_response_fan(
    trials,
    subject=None,
    start_distance_m=START_DISTANCE_M,
    slow_colour="#8a8a8a",
    fast_colour="#d1495b",
    jitter=0.055,
    show_rt=True,
    style_points="box",
    ax=None,
    save_as=None,
    verbose=True,
):
    """Every response as a point on the trajectory it belongs to.

    The four categories fan outward as in the task figure, and each
    response is a dot placed at the distance the ball had reached when
    the answer came. The two speeds are separated by colour rather than
    by panel, since the comparison the figure is for is how much earlier
    or later one commits when the ball is moving faster, and that is only
    legible with the two clouds on the same axis.

    A dot below the body line is an answer that came after the ball had
    arrived.
    """
    d = response_position(trials, start_distance_m)

    if subject is not None:
        d = d[d["subject"] == subject]

    if "usable" in d.columns:
        d = d[d["usable"].fillna(False).astype(bool)]

    d = d[d["distance_at_response_m"].notna()]

    categories = [c for c in CATEGORY_ORDER if c in set(d["trial_type"])]
    speeds = [s for s in ["slow", "fast"] if s in set(d["speed"])]

    colours = {"slow": slow_colour, "fast": fast_colour}

    # ``ax`` lets one fan become a panel of a larger figure. When it is
    # supplied the caller owns the figure, so nothing is saved or laid
    # out here: those belong to the whole figure rather than to one
    # panel of it.
    owns_figure = ax is None

    if owns_figure:
        fig, ax = plt.subplots(figsize=(5.6, 6.6))
    else:
        fig = ax.get_figure()

    rng = np.random.default_rng(0)
    summary = []

    # The bands widen from the release point toward the body, as the
    # trajectories do. Their lateral extent is nominal: the depth of each
    # dot is what the figure carries, and the task figure already shows
    # the true geometry.
    for index, category in enumerate(categories):
        centre = index
        colour = CATEGORY_GREY[category]

        ax.fill_betweenx(
            [0.0, start_distance_m],
            [centre - 0.42, centre - 0.13],
            [centre + 0.42, centre + 0.13],
            color=colour, alpha=0.30, linewidth=0, zorder=0,
        )

        for speed in speeds:
            cell = d[
                (d["trial_type"] == category) & (d["speed"] == speed)
            ]
            values = cell["distance_at_response_m"].to_numpy(dtype=float)

            if not len(values):
                continue

            # Offset within the band so the two distributions do not sit
            # on top of one another, and jittered so repeated values
            # separate.
            side = -0.14 if speed == "slow" else 0.14

            if style_points == "box":
                # The box carries the quartiles and the points the
                # individual trials. Drawing both means a category with
                # six trials cannot be mistaken for one with thirty,
                # which a box alone would hide.
                ax.boxplot(
                    values, positions=[centre + side], widths=0.20,
                    patch_artist=True, showfliers=False, manage_ticks=False,
                    medianprops=dict(color=style.INK, linewidth=1.6),
                    whiskerprops=dict(color=colours[speed], linewidth=1.0),
                    capprops=dict(color=colours[speed], linewidth=1.0),
                    boxprops=dict(facecolor=colours[speed],
                                  edgecolor=colours[speed],
                                  linewidth=1.0, alpha=0.30),
                    zorder=2,
                )

                ax.scatter(
                    centre + side + rng.normal(0, jitter * 0.55,
                                               size=len(values)),
                    values,
                    s=7, color=colours[speed], alpha=0.35,
                    edgecolors="none", zorder=3,
                )
            else:
                ax.scatter(
                    centre + side + rng.normal(0, jitter, size=len(values)),
                    values,
                    s=13, color=colours[speed], alpha=0.45,
                    edgecolors="none", zorder=2,
                )

            median = float(np.median(values))
            median_rt = float(cell["rt_ms"].median())

            if style_points != "box":
                ax.plot(
                    [centre + side - 0.12, centre + side + 0.12],
                    [median] * 2,
                    color=colours[speed], linewidth=2.4,
                    solid_capstyle="butt", zorder=4,
                )

            # The reaction time beside the bar, since the axis carries
            # distance and the two are not interchangeable: the same
            # latency puts the ball in a different place at each speed,
            # which is the point of drawing them together.
            if show_rt:
                ax.text(
                    centre + side + (0.15 if speed == "fast" else -0.15),
                    median,
                    f"{median_rt:.0f}",
                    ha="left" if speed == "fast" else "right",
                    va="center", fontsize=6.5,
                    color=colours[speed], zorder=5,
                )

            summary.append({
                "trial_type": category,
                "speed": speed,
                "n": int(len(values)),
                "median_distance_m": median,
                "median_rt_ms": median_rt,
                "flight_fraction": float(cell["flight_fraction"].median()),
                "after_arrival": float(cell["after_arrival"].mean()),
            })

    ax.axhline(0.0, color=style.INK, linewidth=1.6, zorder=3)
    ax.text(-0.55, 0.10, "body", fontsize=8, color=style.INK, va="bottom")
    ax.text(-0.55, start_distance_m - 0.10, "release", fontsize=8,
            color=style.MUTED, va="top")

    for speed in speeds:
        ax.scatter([], [], s=26, color=colours[speed], alpha=0.7,
                   edgecolors="none", label=speed.capitalize())

    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(
        [CATEGORY_LABEL.get(c, c) for c in categories], fontsize=9,
    )
    ax.set_xlim(-0.7, len(categories) - 0.3)
    ax.set_ylim(min(-0.35, float(d["distance_at_response_m"].min()) - 0.2),
                start_distance_m + 0.25)
    ax.set_ylabel("Distance of the ball from the body at response (m)")
    ax.invert_yaxis()

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.set_title(
        "Where the ball was when the answer came"
        + (f": {subject}" if subject else ""),
        loc="left", fontsize=9.5,
    )

    table = pd.DataFrame(summary)

    if verbose and len(table):
        print(table.round(3).to_string(index=False))

    if owns_figure:
        style.finish_figure(fig)

        if save_as:
            figures.save(fig, save_as)

    return fig, table


def plot_response_fans(
    panels,
    start_distance_m=START_DISTANCE_M,
    slow_colour="#8a8a8a",
    fast_colour="#d1495b",
    show_rt=True,
    style_points="box",
    panel_size=(4.6, 6.2),
    save_as=None,
    verbose=True,
):
    """Several fans side by side on shared axes.

    ``panels`` is a sequence of ``(title, trials, subject)`` triples, or
    ``(title, trials)`` to pool everyone in that frame. The axes are
    shared, since the comparison the figure is for is where one group
    commits relative to another, and separate scales would defeat it.

    Example
    -------
        panels = [
            ("Young controls", young_collision, None),
            ("Patient", clinical, "PT001"),
            ("Age-matched control", clinical, "PA002"),
        ]
    """
    entries = [
        (panel[0], panel[1], panel[2] if len(panel) > 2 else None)
        for panel in panels
    ]

    fig, axes = plt.subplots(
        1, len(entries),
        figsize=(panel_size[0] * len(entries), panel_size[1]),
        sharey=True, squeeze=False,
    )
    flat = axes.ravel()

    tables = []

    for ax, (title, trials, subject) in zip(flat, entries):
        _, table = plot_response_fan(
            trials, subject=subject,
            start_distance_m=start_distance_m,
            slow_colour=slow_colour, fast_colour=fast_colour,
            show_rt=show_rt, style_points=style_points,
            ax=ax, verbose=False,
        )

        ax.set_title(title, loc="left", fontsize=9.5)
        tables.append(table.assign(panel=title))

    # One y label and one legend for the row, since every panel shows the
    # same quantity in the same two colours.
    for ax in flat[1:]:
        ax.set_ylabel("")
        legend = ax.get_legend()

        if legend is not None:
            legend.remove()

    style.finish_figure(fig)

    combined = pd.concat(tables, ignore_index=True)

    if verbose and len(combined):
        print(
            combined.pivot_table(
                index=["trial_type", "speed"], columns="panel",
                values="median_distance_m",
            ).round(2).to_string()
        )

    if save_as:
        figures.save(fig, save_as)

    return fig, combined