"""style.py  -  one place for colors, a minimal matplotlib look, and
consistent tables + charts (clean bar-with-points aesthetic)."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# speed colors
FAST = "#DF4850"
SLOW = "#595959"
# group colors
PATIENT = "#08D8C8"
CONTROL = "#C9C9C9"      # light grey
INK     = "#222222"      # neutral foreground

# convenience maps so cells never hard-code colors
SPEED = {"slow": SLOW, "fast": FAST}
GROUP = {"patient": PATIENT, "control": CONTROL}

# light rule / muted text used by tables and gridlines
_RULE   = "#E6E6E6"
_MUTED  = "#8A8A8A"


def apply():
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#B0B0B0", "axes.linewidth": 0.8,
        "axes.grid": False, "axes.axisbelow": True,
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "regular",
        "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 10, "axes.labelcolor": "#333333",
        "xtick.color": "#666666", "ytick.color": "#666666",
        "xtick.labelcolor": "#444444", "ytick.labelcolor": "#444444",
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 1.8, "lines.solid_capstyle": "round",
        "figure.dpi": 120,
    })


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)
    return ax


# ---------------------------------------------------------------- color utils
def shade(color, amount):
    """Lighten (amount > 0) or darken (amount < 0) a color. amount in [-1, 1]."""
    r, g, b = mcolors.to_rgb(color)
    if amount >= 0:
        r, g, b = (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)
    else:
        f = 1 + amount
        r, g, b = (r * f, g * f, b * f)
    return (r, g, b)


def jitter(n, width=0.09, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, size=n)


# ---------------------------------------------------------------- tables
def table(df, caption=None, bar=None, fmt=None, bar_color=PATIENT, precision=2):
    """Return a consistently styled Styler: hidden index, thin rules, muted
    caption, right-aligned numbers, optional inline bar on `bar` column(s)."""
    label_col = df.columns[0]
    sty = (
        df.style.hide(axis="index")
        .set_table_styles([
            {"selector": "", "props": [
                ("border-collapse", "collapse"), ("font-family", "sans-serif"),
                ("font-size", "10.5pt")]},
            {"selector": "caption", "props": [
                ("caption-side", "top"), ("text-align", "left"),
                ("font-size", "9pt"), ("color", _MUTED), ("letter-spacing", "0.07em"),
                ("text-transform", "uppercase"), ("padding", "0 0 8px 2px")]},
            {"selector": "th.col_heading", "props": [
                ("background-color", "transparent"), ("color", INK),
                ("font-weight", "600"), ("text-align", "right"),
                ("padding", "4px 14px"), ("border", "none"),
                ("border-bottom", f"1.4px solid {INK}")]},
            {"selector": "td", "props": [
                ("text-align", "right"), ("padding", "5px 14px"),
                ("border", "none"), ("border-bottom", f"1px solid {_RULE}"),
                ("color", "#333333")]},
            {"selector": "tbody tr:last-child td", "props": [
                ("border-bottom", f"1.2px solid #CFCFCF")]},
            {"selector": "tbody tr:hover td", "props": [("background-color", "#FAFAFA")]},
        ])
        .set_properties(subset=[label_col], **{
            "text-align": "left", "color": INK, "font-weight": "500"})
        .format(precision=precision, na_rep="-")
    )
    if caption:
        sty = sty.set_caption(caption)
    if fmt:
        sty = sty.format(fmt)
    if bar:
        cols = [bar] if isinstance(bar, str) else list(bar)
        sty = sty.bar(subset=cols, color=mcolors.to_hex(shade(bar_color, 0.55)),
                      vmin=0, align="left")
    return sty


# ---------------------------------------------------------------- charts
def bar_points(ax, labels, values, points=None, colors=None, orient="h",
               bar_alpha=0.9, point_size=34, point_alpha=0.85,
               bar_height=0.72, jitter_width=0.09, seed=0):
    """Reference-style bars with jittered individual points.

    labels  : category names
    values  : one summary value per category (bar length)
    points  : optional list, one array of individual observations per category
    colors  : one color per category, or a single color
    Points inside the bar are drawn darker, points beyond it lighter.
    """
    n = len(labels)
    values = np.asarray(values, dtype=float)
    if colors is None:
        colors = [CONTROL] * n
    elif isinstance(colors, str):
        colors = [colors] * n
    pos = np.arange(n)

    if orient == "h":
        ax.barh(pos, values, height=bar_height, color=colors,
                alpha=bar_alpha, edgecolor="none", zorder=1)
        if points is not None:
            for i, obs in enumerate(points):
                obs = np.asarray(obs, dtype=float)
                obs = obs[np.isfinite(obs)]
                if not len(obs):
                    continue
                inside = obs <= values[i]
                jit = jitter(len(obs), jitter_width, seed + i)
                ax.scatter(obs[inside], pos[i] + jit[inside], s=point_size,
                           color=shade(colors[i], -0.18), alpha=point_alpha,
                           edgecolor="none", zorder=3)
                ax.scatter(obs[~inside], pos[i] + jit[~inside], s=point_size,
                           color=shade(colors[i], 0.45), alpha=point_alpha,
                           edgecolor="none", zorder=3)
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    else:
        ax.bar(pos, values, width=bar_height, color=colors,
               alpha=bar_alpha, edgecolor="none", zorder=1)
        if points is not None:
            for i, obs in enumerate(points):
                obs = np.asarray(obs, dtype=float)
                obs = obs[np.isfinite(obs)]
                if not len(obs):
                    continue
                inside = obs <= values[i]
                jit = jitter(len(obs), jitter_width, seed + i)
                ax.scatter(pos[i] + jit[inside], obs[inside], s=point_size,
                           color=shade(colors[i], -0.18), alpha=point_alpha,
                           edgecolor="none", zorder=3)
                ax.scatter(pos[i] + jit[~inside], obs[~inside], s=point_size,
                           color=shade(colors[i], 0.45), alpha=point_alpha,
                           edgecolor="none", zorder=3)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
    clean(ax)
    return ax