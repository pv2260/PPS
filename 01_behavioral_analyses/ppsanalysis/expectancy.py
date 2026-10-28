"""The expectancy model, and whether there is a profile to fit.

One place decides how tactile-only reaction time depends on delay. The
baseline every corrected value is built from comes from here, and so do the
diagnostics that ask whether that baseline has any structure in it.

The delay variable
------------------
``DELAY_COLUMN`` names the regressor, once, for the whole project.

    tactile_foreperiod_ms   elapsed time from trial onset to the vibration,
                            derived in tables._add_pps_timing. This is what
                            temporal expectancy is a function of, and the
                            default.
    position_rank           a proxy. Within one speed the foreperiod is an
                            affine function of rank, so at degree 1 with the
                            speed interaction the two give the same fitted
                            baseline and differ only in the units and sign of
                            the reported coefficient. At degree 2 they
                            diverge, because the quadratic term is shared
                            across speeds while the rank-to-foreperiod
                            scaling is not.

Set it once, in the notebook:

    from ppsanalysis import expectancy as ex
    ex.DELAY_COLUMN = "tactile_foreperiod_ms"

``h1.expectancy_baseline`` and ``qc.expectancy_model`` are both thin aliases
for the function here, so a hypothesis result and the QC figure describing it
cannot be fitted against different variables.

What is here
------------
    expectancy_baseline   the model. Fits RT_T on delay, speed and their
                          interaction per participant, and reads the fitted
                          value at each distance and speed to give the
                          tactile baseline.

    vt_shape              Does the visuo-tactile profile have a shape?
                          Compare a line, a log, a quadratic and a sigmoid per
                          participant, and record whether the sigmoid's
                          midpoint is estimable.

    tactile_shape         Does the tactile-only baseline vary with delay? If
                          it is flat, the delay-matched correction is
                          subtracting a constant with four-trial noise
                          attached. Flat, linear, quadratic and categorical,
                          compared by AIC, fitted against DELAY_COLUMN so the
                          ramp reported here is the ramp the baseline uses.

    cell_precision        Is any of it resolvable? The per-cell standard error
                          against the range the profile spans. A ratio near 1
                          means the profile's whole excursion is the size of
                          its own noise.

Usage
-----
    from ppsanalysis import expectancy as ex

    c = coh["young"]

    fitted, coefficients = ex.expectancy_baseline(c.trials)

    vt  = ex.vt_shape(c.pps_young)
    tac = ex.tactile_shape(c.pps_young)
    pre = ex.cell_precision(c.marked, n_boot=config.N_BOOT)

    ex.report(vt=vt, tactile=tac, precision=pre)

Every diagnostic returns a tidy frame, one row per participant (or per
participant x speed x condition for precision). ``report`` prints them;
``plot_vt_shape`` and ``plot_tactile_shape`` draw them.
"""


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

import style
from . import figures
from . import config as cfg
from . import qc
from .stats_utils import compare_models_per_participant

MUTED = getattr(style, "_MUTED", "gray")

MODELS = ["flat", "linear", "quadratic", "categorical"]

# The regressor the expectancy model is fitted against, for the whole
# project. See the module docstring for what the two candidates mean and
# where they stop agreeing.
DELAY_COLUMN = "tactile_foreperiod_ms"

# A within-trial delay is seconds, not minutes. Both Unity onset columns are
# absolute session clocks, and fitting against one of those silently reduces
# the model to an intercept plus a speed term.
MAX_PLAUSIBLE_DELAY_MS = 60_000


def _resolve_delay(trials, delay_column=None):
    """The delay column to fit against, checked before it is used."""
    column = DELAY_COLUMN if delay_column is None else delay_column

    if column not in trials.columns:
        candidates = sorted(
            c for c in trials.columns if "ms" in c or "rank" in c
        )
        raise KeyError(
            f"delay column {column!r} is not in the trial table. "
            f"Present: {candidates}. tactile_foreperiod_ms is derived in "
            "tables._add_pps_timing, so a table built before that will not "
            "have it."
        )

    values = pd.to_numeric(trials[column], errors="coerce").dropna()

    if not values.empty and values.max() > MAX_PLAUSIBLE_DELAY_MS:
        raise ValueError(
            f"{column!r} reaches {values.max():.0f}, which is a session "
            "timestamp rather than a within-trial delay. Use "
            "tactile_foreperiod_ms (vibrotactile_onset_ms minus "
            "trial_start_ms) or position_rank."
        )

    return column


# ==================================================================
# The expectancy model
# ==================================================================


def expectancy_baseline(
    trials,
    degree=1,
    interaction=True,
    delay_column=None,
):
    """Tactile-only RT as a function of delay, fitted per participant.

    The delay-matched correction rests on a claim about temporal expectancy:
    that tactile RT varies with the delay between trial onset and the tactile
    event. That is a claim about a *function of delay*, and a function of
    delay does not need one free parameter per distance. Estimating it as
    fourteen unrelated cell means, each from four trials, spends far more
    parameters than the claim requires and carries the sampling noise of
    every one of them into the corrected score.

    This estimates the same function directly. For each participant,

        RT_T = b0 + b1 * delay + b2 * speed + b3 * (delay x speed)

    is fitted by least squares on all of that participant's tactile-only
    trials at once, and the fitted value read off at each distance and speed
    to give the matched baseline.

    The expectancy-control principle is unchanged: the baseline still varies
    with delay, the two speed contexts are still kept apart, and the
    correction is still applied at every distance. Only the estimator
    changes, from fourteen medians of four observations to four coefficients
    from roughly fifty.

    Parameters
    ----------
    trials : DataFrame
        Usable trials. Only the tactile-only rows are used.
    degree : int, default 1
        Polynomial order in delay. Reports disagree about the form of this
        function, describing it as monotonic (Kandula et al., 2017),
        logarithmic (Hobeika et al., 2020) and bell-shaped (Bertoni et al.,
        2022), so ``degree=2`` is the check for curvature rather than an
        assumption of it. Note that the two candidate delay columns are not
        interchangeable at degree 2.
    interaction : bool, default True
        Include the delay-by-speed term, so the expectancy function may
        differ between speed contexts rather than being assumed to.
    delay_column : str or None
        None uses the module-level ``DELAY_COLUMN``, which is where the
        project's choice is recorded. Naming a column here is for comparing
        the two, not for routine use.

    Returns
    -------
    fitted : DataFrame
        One row per subject, session, speed and position: the predicted
        baseline, its standard error, the delay it was evaluated at, and the
        column that delay came from.
    coefficients : DataFrame
        One row per participant: the fitted terms with their standard errors,
        the residual SD and R-squared. ``delay`` is the ramp in milliseconds
        per unit of ``delay_column``, and ``delay_x_fast`` is how much
        steeper it is at the fast speed. Under a foreperiod a negative
        ``delay`` means RT falls as the wait lengthens; under a rank the sign
        is reversed, because rank grows outward while foreperiod shrinks.
    """
    tactile = trials[trials["sensory_condition"] == "T"]

    if tactile.empty:
        return pd.DataFrame(), pd.DataFrame()

    delay_column = _resolve_delay(tactile, delay_column)

    keys = [c for c in ["subject", "session"] if c in tactile.columns]

    fitted_rows, coefficient_rows = [], []

    for key, one in tactile.groupby(keys, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)

        x = pd.to_numeric(one[delay_column], errors="coerce").to_numpy(float)
        y = one["rt_ms"].to_numpy(dtype=float)
        fast = (one["speed"] == "fast").to_numpy(dtype=float)

        ok = np.isfinite(x) & np.isfinite(y)
        x, y, fast = x[ok], y[ok], fast[ok]

        if len(x) < 6 or len(np.unique(x)) < 3:
            continue

        # Delay is centred so the intercept reads as RT at a delay that was
        # actually sampled rather than extrapolated to zero, and so the speed
        # term is the difference between contexts at that delay.
        centre = float(np.mean(x))
        x_c = x - centre

        columns, names = [np.ones_like(x_c)], ["intercept"]

        for power in range(1, int(degree) + 1):
            columns.append(x_c ** power)
            names.append("delay" if power == 1 else f"delay_{power}")

        if len(np.unique(fast)) > 1:
            columns.append(fast)
            names.append("speed_fast")

            if interaction:
                columns.append(x_c * fast)
                names.append("delay_x_fast")

        design = np.column_stack(columns)

        try:
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        residual = y - design @ beta
        residual_df = len(y) - design.shape[1]

        if residual_df <= 0:
            continue

        residual_sd = float(np.sqrt(np.sum(residual ** 2) / residual_df))

        try:
            covariance = np.linalg.inv(design.T @ design) * residual_sd ** 2
            errors = np.sqrt(np.diag(covariance))
        except np.linalg.LinAlgError:
            covariance, errors = None, np.full(len(beta), np.nan)

        total = float(np.sum((y - y.mean()) ** 2))

        record = dict(zip(keys, key_values))
        record["n_trials"] = int(len(y))

        for name, value, error in zip(names, beta, errors):
            record[name] = float(value)
            record[f"se_{name}"] = float(error)

        record["residual_sd_ms"] = residual_sd
        record["r2"] = (
            1.0 - float(np.sum(residual ** 2)) / total if total > 0 else np.nan
        )
        record["delay_column"] = delay_column
        record["delay_centre"] = centre
        record["degree"] = int(degree)
        coefficient_rows.append(record)

        # The baseline is read off at each cell's own delay, taken from the
        # trials rather than from the nominal design value, so a cell whose
        # delays ran slightly long is corrected against where it actually sat.
        for speed in one["speed"].dropna().unique():
            is_fast = 1.0 if speed == "fast" else 0.0
            at_speed = one[one["speed"] == speed]

            for rank in sorted(at_speed["position_rank"].dropna().unique()):
                cell = at_speed[at_speed["position_rank"] == rank]

                delay_value = float(
                    pd.to_numeric(cell[delay_column], errors="coerce").mean()
                )

                if not np.isfinite(delay_value):
                    continue

                point = [1.0]

                for power in range(1, int(degree) + 1):
                    point.append((delay_value - centre) ** power)

                if "speed_fast" in names:
                    point.append(is_fast)

                    if interaction:
                        point.append((delay_value - centre) * is_fast)

                point = np.asarray(point, dtype=float)

                fitted_rows.append({
                    **dict(zip(keys, key_values)),
                    "speed": speed,
                    "position_rank": rank,
                    "delay": delay_value,
                    "delay_column": delay_column,
                    "tactile_baseline_ms": float(point @ beta),
                    "baseline_se_ms": (
                        float(np.sqrt(point @ covariance @ point))
                        if covariance is not None else np.nan
                    ),
                    "n_trials_fit": int(len(y)),
                    "n_trials_cell": int(cell["rt_ms"].notna().sum()),
                })

    return pd.DataFrame(fitted_rows), pd.DataFrame(coefficient_rows)


# ==================================================================
# Visuo-tactile profile shape
# ==================================================================


def vt_shape(trials, condition="VT", k_direction="negative"):
    """Compare candidate shapes for each participant's VT profile.

    Fits a line, a log, a quadratic and a sigmoid to the median RT at each
    distance, and reports RMSE and AIC for each, the sigmoid's parameters,
    and whether its midpoint is estimable.

    ``estimable`` requires all three of: the slope runs in the predicted
    direction, the midpoint falls inside the sampled distance range, and its
    standard error is small enough to localise it. A curve can win on AIC and
    still fail all three.

    The profile is indexed by ``position_rank`` rather than by delay, because
    a boundary is a distance and this is where it is located. That is
    independent of ``DELAY_COLUMN``, which governs the baseline model only.

    Parameters
    ----------
    trials : DataFrame
        Usable trials, e.g. ``cohort.pps_young``.
    condition : str, default "VT"
        Sensory condition to profile.
    k_direction : {"negative", "positive"}
        Expected sign of the sigmoid slope.
    """
    data = trials.loc[trials["sensory_condition"] == condition]
    rows = []

    for keys, one in data.groupby(_group_keys(data), dropna=False):
        record = dict(zip(_group_keys(data), _as_tuple(keys)))

        profile = (
            one.groupby("position_rank")["rt_ms"].median().sort_index()
        )
        x = profile.index.to_numpy(float)
        y = profile.to_numpy(float)

        if len(x) < 4:
            continue

        c = compare_models_per_participant(
            x, y, x_min=x.min(), x_max=x.max(), k_direction=k_direction,
        )

        record["n_points"] = len(x)

        for model in ["linear", "log", "quadratic", "sigmoid"]:
            record[f"rmse_{model}"] = _rmse(c[f"rss_{model}"], len(x))
            record[f"aic_{model}"] = c[f"aic_{model}"]

        record.update({
            "k": c["k"],
            "x_c": c["x_c"],
            "se_x_c": c["se_x_c"],
            "r2_sigmoid": c["r2_sigmoid"],

            "k_ok": c["k_ok"],
            "x_c_in_range": c["x_c_in_range"],
            "x_c_precise": c["x_c_precise"],

            "x_turn_quadratic": c["x_turn_quadratic"],
            "quadratic_nonmonotonic": c["quadratic_nonmonotonic"],

            "shape": c["shape"],
            "shape_aicc": c["shape_aicc"],
            "preferred": c["preferred"],

            "delta_aic_log": c["aic_delta_log_vs_linear"],
            "delta_aic_quadratic": c["aic_delta_quadratic_vs_linear"],
            "delta_aic_sigmoid": c["aic_delta_vs_linear"],
        })

        rows.append(record)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["estimable"] = (
        out["k_ok"] & out["x_c_in_range"] & out["x_c_precise"]
    )

    return out.sort_values(_group_keys(data)).reset_index(drop=True)


# ==================================================================
# Tactile-only baseline shape
# ==================================================================


def tactile_shape(trials, condition="T", delay_column=None):
    """Does tactile-only RT vary with delay, and in what form?

    Four models per participant, all controlling for speed:

        flat          rt ~ speed
        linear        rt ~ speed + delay
        quadratic     rt ~ speed + delay + delay^2
        categorical   rt ~ speed + C(position_rank)

    Fitted on trials, not on cell medians, so the comparison uses every
    observation rather than seven summaries of four. ``preferred_model`` is
    the lowest AIC; the ``delta_*_vs_flat`` columns are negative when that
    model beats flat.

    The regressor is ``DELAY_COLUMN``, the same variable
    ``expectancy_baseline`` fits against, so the ramp reported here is the
    ramp the correction removes rather than a second estimate of it in other
    units. The categorical model stays indexed by distance, because it exists
    to ask whether the cells differ from each other beyond any smooth
    function of delay.

    A flat winner is the substantive result: it means the delay-matched
    correction has no delay-dependence to remove, and is subtracting a
    constant estimated from four trials per cell.

    ``linear_beta`` is in milliseconds per unit of the delay column, so per
    millisecond of foreperiod under the default. ``linear_beta_per_s`` gives
    the same slope per second, which is the readable form.
    """
    data = trials.loc[trials["sensory_condition"] == condition].copy()

    if data.empty:
        return data

    delay_column = _resolve_delay(data, delay_column)

    data["delay_c"] = (
        pd.to_numeric(data[delay_column], errors="coerce")
        - pd.to_numeric(data[delay_column], errors="coerce").mean()
    )

    keys = _group_keys(data)
    rows = []

    for group, one in data.groupby(keys, dropna=False):
        record = dict(zip(keys, _as_tuple(group)))
        record["n"] = len(one)
        record["delay_column"] = delay_column

        if one["speed"].nunique() < 2 or one["position_rank"].nunique() < 3:
            continue

        fits = {
            "flat": smf.ols("rt_ms ~ speed", data=one).fit(),
            "linear": smf.ols("rt_ms ~ speed + delay_c", data=one).fit(),
            "quadratic": smf.ols(
                "rt_ms ~ speed + delay_c + I(delay_c**2)", data=one
            ).fit(),
            "categorical": smf.ols(
                "rt_ms ~ speed + C(position_rank)", data=one
            ).fit(),
        }

        aic = {name: fit.aic for name, fit in fits.items()}

        for name in MODELS:
            record[f"aic_{name}"] = aic[name]
        for name in MODELS[1:]:
            record[f"delta_{name}_vs_flat"] = aic[name] - aic["flat"]

        record["preferred_model"] = min(aic, key=aic.get)

        beta = fits["linear"].params.get("delay_c", np.nan)
        record["linear_beta"] = beta
        record["linear_beta_per_s"] = (
            beta * 1000.0 if delay_column.endswith("_ms") else np.nan
        )
        record["linear_p"] = fits["linear"].pvalues.get("delay_c", np.nan)

        term = "I(delay_c ** 2)"
        record["quadratic_beta"] = fits["quadratic"].params.get(term, np.nan)
        record["quadratic_p"] = fits["quadratic"].pvalues.get(term, np.nan)

        f, p, df = fits["categorical"].compare_f_test(fits["flat"])
        record["F_categorical_vs_flat"] = f
        record["p_categorical_vs_flat"] = p
        record["df_diff"] = df

        rows.append(record)

    out = pd.DataFrame(rows)
    return out.sort_values(keys).reset_index(drop=True) if len(out) else out


# ==================================================================
# Cell precision: the noise floor
# ==================================================================


def cell_precision(marked, n_boot=None, statistic="median"):
    """Per-cell standard error against the range the profile spans.

    ``ratio`` is the profile's range divided by the mean standard error of
    the cells composing it. Below about 2 there is no distance structure to
    resolve: the whole excursion is the size of the noise on a single point.

    Takes the marked table (``cohort.marked``), not the filtered one.
    """
    n_boot = cfg.N_BOOT if n_boot is None else n_boot

    cells = qc.cell_precision(marked, statistic=statistic, n_boot=n_boot)
    cells["design"] = cells["subject"].map(
        getattr(cfg, "DESIGN_OF_SUBJECT", {})
    )

    keys = [k for k in ["design", "subject", "speed", "sensory_condition"]
            if k in cells.columns]

    out = (
        cells.groupby(keys)
        .agg(
            n_per_cell=("n_trials", "mean"),
            mean_se=("se_ms", "mean"),
            range_ms=("estimate_ms", lambda v: v.max() - v.min()),
        )
        .round(1)
        .reset_index()
    )
    out["ratio"] = (out["range_ms"] / out["mean_se"]).round(2)

    return out


# ==================================================================
# Report
# ==================================================================


def report(vt=None, tactile=None, precision=None, coefficients=None,
           by="design"):
    """Print the tables and the counts that matter in each.

    ``coefficients`` is the second return value of ``expectancy_baseline``.
    Passing it prints the fitted ramp alongside the model comparison, so the
    delay term and the AIC verdict on it are read together.
    """

    if coefficients is not None and len(coefficients):
        print("=" * 66)
        print("Expectancy model, per participant")
        print("=" * 66)
        print(f"fitted against {coefficients['delay_column'].unique()}")
        print()
        print(_show(coefficients, [
            "design", "subject", "n_trials",
            "delay", "se_delay", "speed_fast", "delay_x_fast",
            "residual_sd_ms", "r2",
        ], decimals=4))

    if vt is not None and len(vt):
        print("\n" + "=" * 66)
        print("VT profile shape, per participant")
        print("=" * 66)
        print(_show(vt, [
            "design", "subject", "n_points",
            "rmse_linear", "rmse_log", "rmse_quadratic", "rmse_sigmoid",
            "k", "x_c", "se_x_c",
            "k_ok", "x_c_in_range", "x_c_precise",
            "shape", "preferred",
        ]))

        if by in vt.columns:
            print("\nshape classification:")
            print(pd.crosstab(vt[by], vt["shape"]).to_string())

            print("\nestimable boundaries:")
            print(
                vt.groupby(by)
                .agg(
                    n=("estimable", "size"),
                    estimable=("estimable", "sum"),
                    median_rmse_linear=("rmse_linear", "median"),
                    median_rmse_sigmoid=("rmse_sigmoid", "median"),
                )
                .round(2)
                .to_string()
            )

        print("\nAIC vs linear (negative = the alternative wins):")
        print(_show(vt, [
            "design", "subject",
            "delta_aic_log", "delta_aic_quadratic", "delta_aic_sigmoid",
            "shape",
        ]))

    if tactile is not None and len(tactile):
        print("\n" + "=" * 66)
        print("Tactile-only baseline shape, per participant")
        print("=" * 66)

        if "delay_column" in tactile.columns:
            print(f"fitted against {tactile['delay_column'].unique()}")
            print()

        print(_show(tactile, [
            "design", "subject", "n",
            "delta_linear_vs_flat", "delta_quadratic_vs_flat",
            "delta_categorical_vs_flat",
            "preferred_model",
            "linear_beta_per_s", "linear_p",
            "quadratic_beta", "quadratic_p",
            "p_categorical_vs_flat",
        ]))

        print("\npreferred model:")
        print(tactile["preferred_model"].value_counts().to_string())

    if precision is not None and len(precision):
        print("\n" + "=" * 66)
        print("Cell precision: profile range against its own noise")
        print("=" * 66)
        print(precision.to_string(index=False))

        group = [k for k in [by, "sensory_condition"]
                 if k in precision.columns]
        if group:
            print("\nby " + " and ".join(group) + ":")
            print(
                precision.groupby(group)[
                    ["n_per_cell", "mean_se", "range_ms", "ratio"]
                ]
                .mean()
                .round(2)
                .to_string()
            )


# ==================================================================
# Helpers
# ==================================================================


def _group_keys(data):
    return [k for k in ["design", "subject"] if k in data.columns]


def _as_tuple(keys):
    return keys if isinstance(keys, tuple) else (keys,)


def _rmse(rss, n):
    return float(np.sqrt(rss / n)) if np.isfinite(rss) else np.nan


def _show(frame, columns, decimals=2):
    columns = [c for c in columns if c in frame.columns]
    return frame[columns].round(decimals).to_string(index=False)


# ==================================================================
# Figures
# ==================================================================
#
# Both figures take the trials and the table the matching function
# produced, so the curve drawn is the curve that was scored and the
# label under each panel is the classification the AIC comparison
# returned, not an impression of the shape.
#
# Colours come from ``style`` and sizes from ``style.figure_size``, so
# these panels sit at the same scale as every other figure and can be
# saved straight to SVG with ``figures.save``.


# A fitted model is not a speed, a group or a sensory condition, so the
# curves get their own map rather than borrowing one that means something
# else. The data in each panel is coloured by ``style.CONDITION``, which
# is what T and VT mean everywhere in the project.
MODEL_COLOUR = {
    "flat": style.MUTED,
    "linear": style.SLOW,
    "quadratic": style.CONTROL,
    "sigmoid": style.PATIENT,
}


def _condition_colour(condition):
    """The project colour for a sensory condition."""
    return getattr(style, "CONDITION", {}).get(
        str(condition), style.CONTROL,
    )


TRIAL_ALPHA = 0.18


def _fit_curves(x, y):
    """Line and sigmoid through the same points, for drawing only."""
    from scipy.optimize import curve_fit

    out = {"linear": np.polyfit(x, y, 1)}

    ranks = np.sort(np.unique(x))
    near = float(np.mean(y[x <= ranks[1]]))
    far = float(np.mean(y[x >= ranks[-2]]))

    def sigmoid(xx, x_c, k):
        return far + (near - far) / (1.0 + np.exp(k * (xx - x_c)))

    try:
        popt, _ = curve_fit(
            sigmoid, x, y,
            p0=[float(np.mean(x)), 1.0],
            bounds=([x.min(), -8.0], [x.max(), 8.0]),
            maxfev=20000,
        )
        out["sigmoid"] = (popt, near, far)
    except Exception:
        out["sigmoid"] = None

    return out


def plot_vt_shape(
    trials,
    shapes=None,
    condition="VT",
    n_columns=3,
    order_by="design",
    save_as=None,
):
    """One panel per participant: the VT profile with the candidate curves.

    The points are the median RT at each distance, over the trials they came
    from. The line and the sigmoid are fitted to those same points, so the
    panel shows what the AIC comparison was choosing between rather than
    asserting the answer. A participant whose midpoint is not estimable is
    marked, because a curve can win on AIC and still not locate a boundary.

    Parameters
    ----------
    trials : DataFrame
        Usable trials, e.g. ``cohort.pps_young``.
    shapes : DataFrame or None
        Output of ``vt_shape``, which supplies the classification labels.
    order_by : str or None
        Column used to order and group the panels, normally ``design``.
    save_as : str or None
        Filename stem. Saved as SVG and PDF through ``figures.save``.
    """
    data = trials.loc[trials["sensory_condition"] == condition]

    order = _panel_order(data, shapes, order_by)

    fig, axes = _participant_grid(len(order), n_columns)

    smooth = np.linspace(1, 7, 200)

    for ax, (subject, label) in zip(axes, order):
        one = data.loc[data["subject"] == subject]

        profile = one.groupby("position_rank")["rt_ms"].median().sort_index()
        x = profile.index.to_numpy(float)
        y = profile.to_numpy(float)

        _draw_trials(ax, one)

        ax.plot(
            x, y, "o",
            color=style.INK,
            markerfacecolor=_condition_colour(condition),
            markeredgewidth=0.7,
            markersize=5,
            zorder=4,
        )

        if len(x) >= 4:
            fits = _fit_curves(x, y)

            ax.plot(
                smooth,
                np.polyval(fits["linear"], smooth),
                color=MODEL_COLOUR["linear"],
                linewidth=1.2,
                label="linear",
                zorder=2,
            )

            if fits["sigmoid"] is not None:
                (x_c, k), near, far = fits["sigmoid"]

                ax.plot(
                    smooth,
                    far + (near - far) / (1.0 + np.exp(k * (smooth - x_c))),
                    color=MODEL_COLOUR["sigmoid"],
                    linewidth=1.6,
                    label="sigmoid",
                    zorder=3,
                )
                ax.axvline(
                    x_c,
                    color=MODEL_COLOUR["sigmoid"],
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.8,
                    zorder=1,
                )

        _panel_finish(ax, subject, label)

    _grid_finish(
        fig, axes, len(order),
        ylabel=f"{condition} reaction time (ms)",
        title=f"{condition} profile: which shape describes it",
    )

    if save_as:
        figures.save(fig, save_as)

    return fig


def plot_tactile_shape(
    trials,
    shapes=None,
    condition="T",
    n_columns=3,
    order_by="design",
    save_as=None,
):
    """One panel per participant: the tactile baseline against a flat line.

    The horizontal line is the participant's overall tactile median, which is
    what the flat model predicts. The dashed line is the fitted linear model,
    with its slope printed. Where the two coincide there is no
    delay-dependence for the delay-matched correction to remove, and it is
    subtracting a constant estimated from a handful of trials per cell.

    Error bars are the standard error of each cell median, so the scatter of
    the points can be read against the precision behind them.

    Drawn against distance, because that is the axis the correction is
    applied on. The slope printed is refitted here on the cell medians for
    drawing, so it is in milliseconds per distance step; the fitted ramp in
    delay units is in ``tactile_shape`` and in ``expectancy_baseline``.
    """
    data = trials.loc[trials["sensory_condition"] == condition]

    order = _panel_order(
        data, shapes, order_by, label_column="preferred_model",
    )

    fig, axes = _participant_grid(len(order), n_columns)

    smooth = np.linspace(1, 7, 200)

    for ax, (subject, label) in zip(axes, order):
        one = data.loc[data["subject"] == subject]

        cells = (
            one.groupby("position_rank")["rt_ms"]
            .agg(median="median", n="size", sd="std")
            .sort_index()
        )
        cells["se"] = cells["sd"] / np.sqrt(cells["n"])

        x = cells.index.to_numpy(float)
        y = cells["median"].to_numpy(float)

        _draw_trials(ax, one)

        ax.errorbar(
            x, y,
            yerr=cells["se"].to_numpy(float),
            fmt="o",
            color=style.INK,
            markerfacecolor=_condition_colour(condition),
            markeredgewidth=0.7,
            markersize=5,
            linewidth=1.0,
            zorder=4,
        )

        ax.axhline(
            float(one["rt_ms"].median()),
            color=MODEL_COLOUR["flat"],
            linewidth=1.4,
            label="flat (overall median)",
            zorder=2,
        )

        if len(x) >= 3:
            slope, intercept = np.polyfit(x, y, 1)

            ax.plot(
                smooth,
                intercept + slope * smooth,
                color=MODEL_COLOUR["sigmoid"],
                linestyle="--",
                linewidth=1.2,
                label="linear fit",
                zorder=3,
            )
            ax.text(
                0.04, 0.05,
                f"{slope:+.1f} ms/step",
                transform=ax.transAxes,
                fontsize=7,
                color=style.MUTED,
            )

        _panel_finish(ax, subject, label)

    _grid_finish(
        fig, axes, len(order),
        ylabel=f"{condition} reaction time (ms)",
        title="Tactile-only baseline: does it vary with distance",
    )

    if save_as:
        figures.save(fig, save_as)

    return fig


# ------------------------------------------------------------------
# Panel plumbing
# ------------------------------------------------------------------

def _participant_grid(n_panels, n_columns):
    """A participant grid at the project's per-panel dimensions."""
    n_columns = min(n_columns, max(n_panels, 1))
    n_rows = int(np.ceil(n_panels / n_columns))

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=style.figure_size(
            grid=True, nrows=n_rows, ncols=n_columns,
        ),
        sharex=True,
        squeeze=False,
    )

    return fig, axes.ravel()


def _draw_trials(ax, one, seed=3):
    """Every trial behind the medians, so the spread stays visible."""
    condition = (
        one["sensory_condition"].iloc[0] if len(one) else None
    )

    ax.scatter(
        one["position_rank"] + style.jitter(len(one), 0.07, seed),
        one["rt_ms"],
        s=7,
        color=_condition_colour(condition),
        alpha=TRIAL_ALPHA,
        edgecolor="none",
        zorder=1,
    )


def _panel_finish(ax, subject, label):
    ax.set_title(f"{subject}\n{label}" if label else str(subject))
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels([f"D{r}" for r in range(1, 8)])
    style.clean(ax)


def _grid_finish(fig, axes, n_panels, ylabel, title):
    for ax in axes[n_panels:]:
        ax.remove()

    axes[0].legend(loc="best")

    fig.supxlabel("Distance (D1 = closest)")
    fig.supylabel(ylabel)
    fig.suptitle(title, x=0.02, ha="left", color=style.INK)

    style.finish_figure(fig)


def _panel_order(data, shapes, order_by, label_column="shape"):
    """Panels ordered by design, each carrying its classification label."""
    subjects = sorted(data["subject"].dropna().unique())

    labels = {}

    if shapes is not None and label_column in shapes.columns:
        labels = dict(zip(shapes["subject"], shapes[label_column]))

    if order_by and order_by in data.columns:
        design = data.groupby("subject")[order_by].first().to_dict()

        subjects = sorted(
            subjects, key=lambda s: (str(design.get(s, "")), s)
        )

        return [
            (s, f"{design.get(s, '')} | {labels.get(s, '')}".strip(" |"))
            for s in subjects
        ]

    return [(s, str(labels.get(s, ""))) for s in subjects]