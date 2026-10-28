"""style.py

Centralized visual style for all PPS analysis figures and tables.

This module defines:
- project colors
- Matplotlib rcParams
- consistent publication-oriented figure sizes
- subplot spacing
- shared axis-cleaning helpers
- color utilities
- table styling
- bar-with-points plotting helper
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------

# Speed colors
FAST = "#fa525b"
SLOW = "#838689"

# PPS condition colors
VT = "#19d3c5"
T = "#838689"
V = "#e5e5e5"

# Collision outcome colors
HIT = "#f4792f"
NEAR_HIT = "#fdd3b5"
NEAR_MISS = "#cdc4e0"
MISS = "#8971b0"

# PPS near / far colors
NEAR = "#0497b2"
FAR = "#acedf8"

# Group colors
PATIENT = "#c857a0"
CONTROL = "#a0a4a6"
AGE_MATCHED = "#e5bbd8"


# Neutral foreground
INK = "#222222"

FACILITATION = "#0097b2"

#"#c857a0"

# Convenience maps so analysis code never hard-codes colors
SPEED = {
    "slow": SLOW,
    "fast": FAST,
}

GROUP = {
    "patient": PATIENT,
    "control": CONTROL,
    "matched": AGE_MATCHED,
}


PPS_DISTANCE = {
    "near": NEAR,
    "far": FAR,
}

# Light rules and muted text
RULE = "#E6E6E6"
MUTED = "#8A8A8A"

_RULE = RULE
_MUTED = MUTED

# Display names. Figures name the role, not the identifier: a reader needs
# to know which participant is the patient, and which specific control it
# was belongs in the caption where it can be stated once.
DISPLAY_NAME = {
    "PT001": "Patient",
    "PA002": "Age-matched control",
    "PA001": "Adult control",
}


def display_name(subject):
    """Figure label for a participant. Unlisted ids pass through unchanged."""
    return DISPLAY_NAME.get(str(subject), str(subject))


def display_names(subjects):
    """Vectorised form, for axis tick labels."""
    return [display_name(s) for s in subjects]


# ---------------------------------------------------------------------
# Figure geometry
# ---------------------------------------------------------------------

# Approximate publication widths in inches
ONE_COLUMN = 3.5
ONE_AND_HALF_COLUMN = 5.3
TWO_COLUMN = 7.2

# Standard single-row heights
HEIGHT_COMPACT = 2.8
HEIGHT_STANDARD = 3.8
HEIGHT_TALL = 4.8

# Height allocated to each row in vertically stacked figures
PANEL_HEIGHT = 2.8

# Width allocated to each column in dense grids
GRID_PANEL_WIDTH = 1.65

# Height allocated to each row in dense grids
GRID_PANEL_HEIGHT = 2.25

# Default spacing between subplot panels
SUBPLOT_WSPACE = 0.28
SUBPLOT_HSPACE = 0.38


def figure_size(
    width="two_column",
    height="standard",
    *,
    nrows=1,
    ncols=1,
    grid=False,
):
    """
    Return a consistent Matplotlib figure size.

    Parameters
    ----------
    width : {"one_column", "one_and_half", "two_column"} or float
        Base figure width in publication-column units or inches.

    height : {"compact", "standard", "tall"} or float
        Base height for a single-row figure.

    nrows : int, default=1
        Number of subplot rows.

    ncols : int, default=1
        Number of subplot columns.

    grid : bool, default=False
        If True, size the figure by per-panel dimensions. This is useful for
        dense grids such as 2 rows × 7 distances.

    Returns
    -------
    tuple[float, float]
        Width and height in inches.
    """
    widths = {
        "one_column": ONE_COLUMN,
        "one_and_half": ONE_AND_HALF_COLUMN,
        "two_column": TWO_COLUMN,
    }

    heights = {
        "compact": HEIGHT_COMPACT,
        "standard": HEIGHT_STANDARD,
        "tall": HEIGHT_TALL,
    }

    if grid:
        resolved_width = max(
            TWO_COLUMN,
            GRID_PANEL_WIDTH * max(int(ncols), 1),
        )
        resolved_height = (
            GRID_PANEL_HEIGHT * max(int(nrows), 1)
        )
        return resolved_width, resolved_height

    resolved_width = (
        widths[width]
        if isinstance(width, str)
        else float(width)
    )

    resolved_height = (
        heights[height]
        if isinstance(height, str)
        else float(height)
    )

    if nrows > 1:
        resolved_height = PANEL_HEIGHT * int(nrows)

    return resolved_width, resolved_height


# ---------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------

def apply():
    """Apply the project-wide Matplotlib style."""
    mpl.rcParams.update({
        # Canvas
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 120,
        "savefig.dpi": 300,

        # Fonts
        "font.family": "sans-serif",
        "font.size": 10,

        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#B0B0B0",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.axisbelow": True,

        # Titles
        "axes.titlesize": 11,
        "axes.titleweight": "regular",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 10,

        # Axis labels
        "axes.labelsize": 10,
        "axes.labelcolor": "#333333",
        "axes.labelpad": 6,

        # Tick labels
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "xtick.labelcolor": "#444444",
        "ytick.labelcolor": "#444444",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # Legends
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.title_fontsize": 9,

        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "lines.solid_capstyle": "round",

        # Error bars
        "errorbar.capsize": 3,

        # Layout
        "figure.constrained_layout.use": False,
    })


def clean(ax):
    """Apply the final minimal axis cleanup."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)
    return ax


def finish_figure(
    fig,
    *,
    tight=True,
    pad=0.8,
    w_pad=0.6,
    h_pad=0.8,
):
    """
    Apply consistent spacing to a completed figure.

    Use this instead of calling ``fig.tight_layout()`` with different values
    throughout the analysis code.
    """
    if tight:
        fig.tight_layout(
            pad=pad,
            w_pad=w_pad,
            h_pad=h_pad,
        )
    else:
        fig.subplots_adjust(
            wspace=SUBPLOT_WSPACE,
            hspace=SUBPLOT_HSPACE,
        )
    return fig


def panel_labels(
    axes,
    labels=None,
    *,
    x=-0.14,
    y=1.06,
    fontsize=14,
):
    """
    Add bold panel labels such as A, B, C to one or more axes.
    """
    axes = np.atleast_1d(axes).ravel()

    if labels is None:
        labels = [
            chr(ord("A") + index)
            for index in range(len(axes))
        ]

    for ax, label in zip(axes, labels):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
            color=INK,
            ha="left",
            va="top",
        )

    return axes


# ---------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------

def shade(color, amount):
    """
    Lighten or darken a color.

    Parameters
    ----------
    color : Matplotlib color
        Input color.

    amount : float in [-1, 1]
        Positive values lighten the color.
        Negative values darken the color.
    """
    if not -1 <= amount <= 1:
        raise ValueError("amount must be between -1 and 1")

    r, g, b = mcolors.to_rgb(color)

    if amount >= 0:
        r = r + (1 - r) * amount
        g = g + (1 - g) * amount
        b = b + (1 - b) * amount
    else:
        factor = 1 + amount
        r *= factor
        g *= factor
        b *= factor

    return r, g, b


def jitter(n, width=0.09, seed=0):
    """Return reproducible horizontal or vertical jitter."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, size=n)


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------

def table(
    df,
    caption=None,
    bar=None,
    fmt=None,
    bar_color=PATIENT,
    precision=2,
):
    """
    Return a consistently styled pandas Styler.

    The first column is left-aligned and treated as the row label.
    Numeric columns are right-aligned. Optional inline bars can be added to
    one or more numeric columns.
    """
    label_col = df.columns[0]

    styler = (
        df.style
        .hide(axis="index")
        .set_table_styles([
            {
                "selector": "",
                "props": [
                    ("border-collapse", "collapse"),
                    ("font-family", "sans-serif"),
                    ("font-size", "10.5pt"),
                ],
            },
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("text-align", "left"),
                    ("font-size", "9pt"),
                    ("color", MUTED),
                    ("letter-spacing", "0.07em"),
                    ("text-transform", "uppercase"),
                    ("padding", "0 0 8px 2px"),
                ],
            },
            {
                "selector": "th.col_heading",
                "props": [
                    ("background-color", "transparent"),
                    ("color", INK),
                    ("font-weight", "600"),
                    ("text-align", "right"),
                    ("padding", "4px 14px"),
                    ("border", "none"),
                    ("border-bottom", f"1.4px solid {INK}"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "right"),
                    ("padding", "5px 14px"),
                    ("border", "none"),
                    ("border-bottom", f"1px solid {RULE}"),
                    ("color", "#333333"),
                ],
            },
            {
                "selector": "tbody tr:last-child td",
                "props": [
                    ("border-bottom", "1.2px solid #CFCFCF"),
                ],
            },
            {
                "selector": "tbody tr:hover td",
                "props": [
                    ("background-color", "#FAFAFA"),
                ],
            },
        ])
        .set_properties(
            subset=[label_col],
            **{
                "text-align": "left",
                "color": INK,
                "font-weight": "500",
            },
        )
        .format(
            precision=precision,
            na_rep="-",
        )
    )

    if caption:
        styler = styler.set_caption(caption)

    if fmt:
        styler = styler.format(fmt)

    if bar:
        columns = [bar] if isinstance(bar, str) else list(bar)
        styler = styler.bar(
            subset=columns,
            color=mcolors.to_hex(
                shade(bar_color, 0.55)
            ),
            vmin=0,
            align="left",
        )

    return styler


# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------
def style_boxplot(
    bp,
    colors,
    *,
    box_alpha=0.22,
    linewidth=1.5,
):
    """
    Apply category-specific styling to a matplotlib boxplot.

    Parameters
    ----------
    bp : dict
        Object returned by ax.boxplot(..., patch_artist=True).

    colors : sequence
        One color per box/category.

    box_alpha : float
        Alpha of box fill.

    linewidth : float
        Width of box, whisker, cap, and median lines.
    """

    if isinstance(colors, str):
        colors = [colors]

    for i, color in enumerate(colors):

        # Box: transparent fill + full-color border
        box = bp["boxes"][i]

        box.set_facecolor(
            mcolors.to_rgba(
                color,
                box_alpha,
            )
        )

        box.set_edgecolor(color)
        box.set_linewidth(linewidth)

        # Whiskers: two per box
        for whisker in bp["whiskers"][
            2 * i : 2 * i + 2
        ]:
            whisker.set_color(color)
            whisker.set_linewidth(linewidth)

        # Caps: two per box
        for cap in bp["caps"][
            2 * i : 2 * i + 2
        ]:
            cap.set_color(color)
            cap.set_linewidth(linewidth)

        # Median: SAME full category color
        bp["medians"][i].set_color(color)
        bp["medians"][i].set_linewidth(
            linewidth + 0.3
        )

        # Mean, if present
        if (
            "means" in bp
            and len(bp["means"]) > i
        ):
            mean = bp["means"][i]

            mean.set_markerfacecolor(color)
            mean.set_markeredgecolor("none")
            mean.set_markeredgewidth(0)

def boxplot_points(
    ax,
    data,
    *,
    positions=None,
    colors=None,
    labels=None,
    widths=0.62,
    box_alpha=0.25,
    point_alpha=0.60,
    point_size=28,
    jitter_width=0.07,
    median_linewidth=1.6,
    line_width=1.25,
    seed=0,
    showfliers=False,
    showmeans=False,
):
    """
    Draw condition-colored boxplots with superimposed raw observations.

    Styling
    -------
    - box border, whiskers and caps use the full condition color
    - box fill uses the same color at lower alpha
    - points use the same color at a higher alpha than the box fill
    - points have no marker border

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.

    data : sequence of array-like
        One vector of observations per condition.

    positions : sequence, optional
        X positions of the conditions. Defaults to 0..n-1.

    colors : color or sequence of colors, optional
        One condition color per box. Defaults to CONTROL.

    labels : sequence of str, optional
        X tick labels.
    """

    groups = [np.asarray(values, dtype=float) for values in data]
    groups = [values[np.isfinite(values)] for values in groups]
    n = len(groups)

    if positions is None:
        positions = np.arange(n, dtype=float)
    else:
        positions = np.asarray(positions, dtype=float)

    if colors is None:
        colors = [CONTROL] * n
    elif isinstance(colors, str):
        colors = [colors] * n
    else:
        colors = list(colors)

    if len(colors) != n:
        raise ValueError("colors must be a single color or one color per group")

    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=showfliers,
        showmeans=showmeans,
        manage_ticks=False,
        zorder=2,
    )

    for i, color in enumerate(colors):
        # Box: opaque condition-colored border, light condition-colored fill.
        # Set face and edge separately.  Using an RGBA face ensures the
        # fill is translucent while the border remains fully opaque.
        bp["boxes"][i].set_facecolor(mcolors.to_rgba(color, box_alpha))
        bp["boxes"][i].set_edgecolor(color)
        bp["boxes"][i].set_linewidth(line_width)
        bp["boxes"][i].set_fill(True)

        # Two whiskers and two caps belong to each box.
        for artist in bp["whiskers"][2 * i : 2 * i + 2]:
            artist.set(
                color=color,
                linewidth=line_width,
            )

        for artist in bp["caps"][2 * i : 2 * i + 2]:
            artist.set(
                color=color,
                linewidth=line_width,
            )

        # Median uses exactly the same full condition color as the box border.
        bp["medians"][i].set(
            color=color,
            linewidth=median_linewidth,
        )

        # If means are requested, make them use the same full condition color.
        if showmeans and "means" in bp and i < len(bp["means"]):
            mean_artist = bp["means"][i]
            mean_artist.set_marker("o")
            mean_artist.set_markerfacecolor(color)
            mean_artist.set_markeredgecolor("none")
            mean_artist.set_markeredgewidth(0)
            mean_artist.set_color(color)

        # Raw observations: same hue, more visible than the box fill, no edge.
        values = groups[i]
        if len(values):
            x = positions[i] + jitter(
                len(values),
                width=jitter_width,
                seed=seed + i,
            )

            ax.scatter(
                x,
                values,
                s=point_size,
                marker="o",
                facecolor=color,
                alpha=point_alpha,
                edgecolor="none",
                linewidth=0,
                zorder=3,
            )

    if labels is not None:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

    clean(ax)
    return bp


def bar_points(
    ax,
    labels,
    values,
    points=None,
    colors=None,
    orient="h",
    bar_alpha=0.9,
    point_size=34,
    point_alpha=0.85,
    bar_height=0.72,
    jitter_width=0.09,
    seed=0,
):
    """
    Draw reference-style bars with optional jittered observations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.

    labels : sequence
        Category labels.

    values : sequence
        One summary value per category.

    points : sequence of arrays, optional
        One array of individual observations per category.

    colors : sequence or color, optional
        One color per category or a single shared color.

    orient : {"h", "v"}
        Horizontal or vertical orientation.
    """
    n = len(labels)
    values = np.asarray(values, dtype=float)

    if colors is None:
        colors = [CONTROL] * n
    elif isinstance(colors, str):
        colors = [colors] * n

    positions = np.arange(n)

    if orient == "h":
        ax.barh(
            positions,
            values,
            height=bar_height,
            color=colors,
            alpha=bar_alpha,
            edgecolor="none",
            zorder=1,
        )

        if points is not None:
            for index, observations in enumerate(points):
                observations = np.asarray(
                    observations,
                    dtype=float,
                )
                observations = observations[
                    np.isfinite(observations)
                ]

                if not len(observations):
                    continue

                inside = observations <= values[index]
                offsets = jitter(
                    len(observations),
                    jitter_width,
                    seed + index,
                )

                ax.scatter(
                    observations[inside],
                    positions[index] + offsets[inside],
                    s=point_size,
                    color=shade(colors[index], -0.18),
                    alpha=point_alpha,
                    edgecolor="none",
                    zorder=3,
                )

                ax.scatter(
                    observations[~inside],
                    positions[index] + offsets[~inside],
                    s=point_size,
                    color=shade(colors[index], 0.45),
                    alpha=point_alpha,
                    edgecolor="none",
                    zorder=3,
                )

        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()

    elif orient == "v":
        ax.bar(
            positions,
            values,
            width=bar_height,
            color=colors,
            alpha=bar_alpha,
            edgecolor="none",
            zorder=1,
        )

        if points is not None:
            for index, observations in enumerate(points):
                observations = np.asarray(
                    observations,
                    dtype=float,
                )
                observations = observations[
                    np.isfinite(observations)
                ]

                if not len(observations):
                    continue

                inside = observations <= values[index]
                offsets = jitter(
                    len(observations),
                    jitter_width,
                    seed + index,
                )

                ax.scatter(
                    positions[index] + offsets[inside],
                    observations[inside],
                    s=point_size,
                    color=shade(colors[index], -0.18),
                    alpha=point_alpha,
                    edgecolor="none",
                    zorder=3,
                )

                ax.scatter(
                    positions[index] + offsets[~inside],
                    observations[~inside],
                    s=point_size,
                    color=shade(colors[index], 0.45),
                    alpha=point_alpha,
                    edgecolor="none",
                    zorder=3,
                )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

    else:
        raise ValueError("orient must be 'h' or 'v'")

    clean(ax)
    return ax


# Apply the project style when this module is imported.
apply()