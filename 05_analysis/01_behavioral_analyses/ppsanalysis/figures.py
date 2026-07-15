"""Publication-grade figure settings and helpers.

Everything here exists so that a figure made in this notebook can go straight
into a thesis or a paper without being redrawn.

Three things make a figure publication-grade:

  1. It is a VECTOR file (SVG), so it stays sharp at any size in print.
  2. Its TEXT IS STILL TEXT, not outlines. This means you can open the SVG in
     Illustrator or Inkscape and fix a typo in an axis label without re-running
     the analysis. This is what `svg.fonttype = "none"` does. Most people miss
     this and end up with un-editable figures.
  3. Its SIZE IS THE FINAL SIZE. If you draw a figure at 12 inches wide and then
     shrink it to fit an 85 mm journal column, all the fonts shrink with it and
     end up unreadable. So we draw at the real column width from the start, and
     we never scale the figure afterwards.

Journal column widths (these are the standard ones):
    single column = 85 mm  = 3.35 inches
    1.5 column    = 114 mm = 4.49 inches
    double column = 174 mm = 6.85 inches
"""

import os

import matplotlib.pyplot as plt


# Standard figure widths, in inches.
SINGLE_COLUMN = 3.35
ONE_HALF_COLUMN = 4.49
DOUBLE_COLUMN = 6.85

# Where figures get saved. Set this from the notebook.
FIGURE_DIR = "figures"


def apply():
    """Set the matplotlib defaults for every figure in the notebook.

    Call this once, at the top of the notebook, right after importing style.
    """

    # --- Vector output, with editable text -----------------------------------
    # "none" means: keep letters as letters in the SVG file.
    # The alternative ("path") turns every letter into a shape, which looks the
    # same but cannot be edited or searched.
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42   # TrueType, also editable
    plt.rcParams["ps.fonttype"] = 42

    # --- Fonts ---------------------------------------------------------------
    # A journal will usually want 7 to 9 pt text in the final figure. Because we
    # draw at final size (see the module docstring), we can just set that here.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    plt.rcParams["font.size"] = 8          # default for everything
    plt.rcParams["axes.titlesize"] = 9
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 7

    # Bold panel titles, as requested.
    plt.rcParams["axes.titleweight"] = "bold"

    # --- Lines and markers ---------------------------------------------------
    plt.rcParams["lines.linewidth"] = 1.4
    plt.rcParams["lines.markersize"] = 4
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8

    # --- Frame ---------------------------------------------------------------
    # Remove the top and right box lines. This is the single easiest way to make
    # a plot look like a figure instead of a spreadsheet chart.
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    # --- Legend --------------------------------------------------------------
    plt.rcParams["legend.frameon"] = False

    # --- Saving --------------------------------------------------------------
    plt.rcParams["savefig.dpi"] = 300           # only matters for PNG fallback
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["figure.facecolor"] = "white"

    # Show figures inline as SVG, not as blurry PNG.
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic("config", 'InlineBackend.figure_formats = ["svg"]')
    except Exception:
        # Not running in a notebook. That is fine, just skip this.
        pass


def new_figure(n_panels=1, width=None, height=3.0):
    """Make a figure sized for a journal page.

    n_panels : how many panels side by side
    width    : figure width in inches. If you leave it out, we pick a sensible
               one: single column for 1 panel, double column for 2 or more.
    height   : figure height in inches.

    Returns (fig, axes). `axes` is always a list, even when there is 1 panel, so
    you can always write axes[0], axes[1], ... without special-casing.
    """

    if width is None:
        if n_panels == 1:
            width = SINGLE_COLUMN
        else:
            width = DOUBLE_COLUMN

    fig, axes = plt.subplots(1, n_panels, figsize=(width, height))

    # When n_panels == 1, matplotlib gives back a single Axes, not a list.
    # Wrap it so the calling code can always index into it.
    if n_panels == 1:
        axes = [axes]

    return fig, list(axes)


def label_panels(axes, labels=None):
    """Put a bold A, B, C ... in the top-left corner of each panel.

    Journals nearly always want this for multi-panel figures.
    """

    if labels is None:
        labels = ["A", "B", "C", "D", "E", "F"]

    for ax, letter in zip(axes, labels):
        ax.text(
            -0.15, 1.08, letter,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="right",
        )


def save(fig, name, directory=None):
    """Save a figure as SVG (for editing) and PDF (for LaTeX).

    We do NOT call plt.tight_layout() for you, because sometimes you want to
    control the spacing yourself. Call it before this if you need it.

    name : filename without an extension, e.g. "h1_facilitation"
    """

    if directory is None:
        directory = FIGURE_DIR

    os.makedirs(directory, exist_ok=True)

    svg_path = os.path.join(directory, name + ".svg")
    pdf_path = os.path.join(directory, name + ".pdf")

    fig.savefig(svg_path, format="svg")
    fig.savefig(pdf_path, format="pdf")

    print(f"  saved: {svg_path}")
    print(f"  saved: {pdf_path}")
