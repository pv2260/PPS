"""Model fitting, bootstrap, permutation and classification helpers.

Task-agnostic: nothing here knows about PPS or Hit/Miss specifically.

Notes on the model comparison
-----------------------------
``curve_fit``'s covariance matrix is kept rather than discarded, so x_c comes
back with a standard error. The preregistration gates H2 and H5 on x_c being
estimable with usable precision, and that number has to be computed somewhere.

AICc is reported alongside AIC. With five parameters on seven points the ratio
n/K is 1.4, far below the range in which plain AIC is valid, and the correction
term for the sigmoid is 60 against 8 for the line and 3 for flat. AIC alone is
anti-conservative towards the sigmoid at this profile length.

The boundary and slope checks that decide ``preferred`` are also returned as
their own columns. ``preferred`` retains the original flat/linear/sigmoid
comparison; ``shape`` additionally compares logarithmic and quadratic trends
for descriptive diagnosis of non-sigmoidal profiles.
"""
from itertools import permutations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from .config import N_BOOT, RANDOM_SEED, PATIENT_ID, MATCHED_CONTROL_ID, CLINICAL_SESSION, CLINICAL_COHORT


def sigmoid(x, low, high, x_c, k):
    """Four-parameter logistic.

        y = low + (high - low) / (1 + exp(k * (x - x_c)))

    With x as distance rank and rank 1 closest to the body, k > 0 gives the
    profile PPS predicts: the high plateau near the body, the low plateau far
    away, and the transition centred on x_c. The half-width of that
    transition is roughly 2 / k in rank units.

    The parameterisation is redundant: swapping low and high while negating k
    describes the identical curve, so the raw sign of k does not identify the
    direction of the effect. ``compare_models_per_participant`` canonicalises
    it against the signed amplitude.
    """
    return low + (high - low) / (1 + np.exp(k * (x - x_c)))

def _fit_sigmoid(x, y, x_c_bounds=None, x_c_margin=1.0):
    """Least-squares sigmoid fit with a small multi-start search.

    By default the sigmoid midpoint is constrained to the interior of the
    sampled x-range. With ranks 1..7 and ``x_c_margin=1.0``, this gives
    2 <= x_c <= 6. This prevents the optimiser from reporting an apparent
    boundary at the extreme edge of the sampled space.

    ``x_c_bounds=(lower, upper)`` can be supplied explicitly when a caller
    needs a different admissible boundary range.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    failed = {
        "low": np.nan, "high": np.nan, "x_c": np.nan, "k": np.nan,
        "se_low": np.nan, "se_high": np.nan, "se_x_c": np.nan, "se_k": np.nan,
        "rss": np.nan, "n": len(x), "ok": False,
    }

    if len(x) < 5 or len(np.unique(x)) < 4:
        return failed

    x_lo = float(np.nanmin(x))
    x_hi = float(np.nanmax(x))

    if x_c_bounds is None:
        xc_lo = x_lo + float(x_c_margin)
        xc_hi = x_hi - float(x_c_margin)
    else:
        xc_lo, xc_hi = map(float, x_c_bounds)

    if not np.isfinite(xc_lo) or not np.isfinite(xc_hi) or xc_lo >= xc_hi:
        return failed

    bounds = (
        [-np.inf, -np.inf, xc_lo, -100.0],
        [ np.inf,  np.inf, xc_hi,  100.0],
    )

    # Small multi-start set spanning the admissible x_c interval.
    x_mid = float(np.clip(np.nanmedian(x), xc_lo, xc_hi))
    x_c_starts = [
        xc_lo + 0.25 * (xc_hi - xc_lo),
        x_mid,
        xc_lo + 0.75 * (xc_hi - xc_lo),
    ]
    x_c_starts = sorted(set(float(np.clip(v, xc_lo, xc_hi)) for v in x_c_starts))
    k_starts = [0.5, 2.0, 5.0]

    best = None
    best_rss = np.inf

    for x_c0 in x_c_starts:
        for k0 in k_starts:
            p0 = [
                float(np.nanmin(y)),
                float(np.nanmax(y)),
                float(x_c0),
                float(k0),
            ]

            try:
                popt, pcov = curve_fit(
                    sigmoid,
                    x,
                    y,
                    p0=p0,
                    bounds=bounds,
                    maxfev=5000,
                    absolute_sigma=False,
                )
            except Exception:
                continue

            residuals = y - sigmoid(x, *popt)
            rss = float(np.sum(residuals ** 2))

            if rss < best_rss:
                best_rss = rss
                best = (popt, pcov, residuals)

    if best is None:
        return failed

    popt, pcov, residuals = best

    with np.errstate(invalid="ignore"):
        errors = np.sqrt(np.diag(pcov))

    return {
        "low": float(popt[0]),
        "high": float(popt[1]),
        "x_c": float(popt[2]),
        "k": float(popt[3]),
        "se_low": float(errors[0]),
        "se_high": float(errors[1]),
        "se_x_c": float(errors[2]),
        "se_k": float(errors[3]),
        "rss": float(np.sum(residuals ** 2)),
        "n": len(x),
        "ok": True,
    }

def safe_sigmoid_fit(x, y, x_c_bounds=None, x_c_margin=1.0):
    """Fit a sigmoid and return its parameters, or NaNs if fitting fails.

    By default x_c is restricted to the interior of the sampled range
    (one rank in from each edge).
    """
    return _fit_sigmoid(
        x, y,
        x_c_bounds=x_c_bounds,
        x_c_margin=x_c_margin,
    )

def fit_log_aic(x, y, full=False):
    """Logarithmic Gaussian fit: y = a + b*log(x)."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]

    n = len(x)

    if n < 3:
        empty = {
            "aic": np.nan,
            "aicc": np.nan,
            "rss": np.nan,
            "n": n,
            "intercept": np.nan,
            "slope": np.nan,
        }
        return empty if full else np.nan

    lx = np.log(x)

    slope, intercept = np.polyfit(lx, y, 1)

    y_hat = intercept + slope * lx

    rss = float(np.sum((y - y_hat) ** 2))

    # intercept + slope + residual variance
    n_params = 3

    result = {
        "aic": _gaussian_aic(rss, n, n_params),
        "aicc": _gaussian_aicc(rss, n, n_params),
        "rss": rss,
        "n": n,
        "intercept": float(intercept),
        "slope": float(slope),
    }

    return result if full else result["aic"]


def fit_quadratic_aic(x, y, full=False):
    """Quadratic Gaussian fit: y = a*x^2 + b*x + c."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    n = len(x)

    if n < 4:
        empty = {
            "aic": np.nan,
            "aicc": np.nan,
            "rss": np.nan,
            "n": n,
            "a": np.nan,
            "b": np.nan,
            "c": np.nan,
        }
        return empty if full else np.nan

    a, b, c = np.polyfit(x, y, 2)

    y_hat = a * x**2 + b * x + c

    rss = float(np.sum((y - y_hat) ** 2))

    # a + b + c + residual variance
    n_params = 4

    result = {
        "aic": _gaussian_aic(rss, n, n_params),
        "aicc": _gaussian_aicc(rss, n, n_params),
        "rss": rss,
        "n": n,
        "a": float(a),
        "b": float(b),
        "c": float(c),
    }

    return result if full else result["aic"]

def _gaussian_aic(rss, n, n_params):
    """AIC for a Gaussian model fitted by least squares.

    ``n_params`` counts the residual variance as a parameter.
    """
    sigma2 = rss / n if rss > 0 else 1e-12
    log_lik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    return 2 * n_params - 2 * log_lik


def _gaussian_aicc(rss, n, n_params):
    """Second-order AIC, which is what a seven-point profile requires.

    AIC assumes n is large relative to the number of parameters. The sigmoid
    has five on seven points, a ratio of 1.4, so the correction term
    2K(K+1)/(n-K-1) is not a refinement here but the difference between a
    usable criterion and an anti-conservative one: it adds 3 to the flat
    model, 8 to the line, and 60 to the sigmoid.

    Returns infinity where the correction is undefined, that is where
    n <= n_params + 1, since a model with no residual degrees of freedom
    cannot be compared on these grounds at all.
    """
    aic = _gaussian_aic(rss, n, n_params)
    denominator = n - n_params - 1

    if denominator <= 0:
        return np.inf

    return aic + 2 * n_params * (n_params + 1) / denominator


def _r_squared(y, rss):
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 0:
        return np.nan
    return 1.0 - rss / total


def fit_flat_aic(y, full=False):
    """Flat (intercept-only) Gaussian fit."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)

    if n < 2:
        empty = {"aic": np.nan, "aicc": np.nan, "rss": np.nan, "n": n}
        return empty if full else np.nan

    rss = float(np.sum((y - y.mean()) ** 2))
    n_params = 2  # mean + sigma

    result = {
        "aic": _gaussian_aic(rss, n, n_params),
        "aicc": _gaussian_aicc(rss, n, n_params),
        "rss": rss,
        "n": n,
    }
    return result if full else result["aic"]


def fit_linear_aic(x, y, full=False):
    """Linear Gaussian fit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)

    if n < 3:
        empty = {"aic": np.nan, "aicc": np.nan, "rss": np.nan, "n": n}
        return empty if full else np.nan

    slope, intercept = np.polyfit(x, y, 1)
    rss = float(np.sum((y - (slope * x + intercept)) ** 2))
    n_params = 3  # slope + intercept + sigma

    result = {
        "aic": _gaussian_aic(rss, n, n_params),
        "aicc": _gaussian_aicc(rss, n, n_params),
        "rss": rss,
        "n": n,
        "slope": float(slope),
    }
    return result if full else result["aic"]


def fit_sigmoid_aic(x, y, x_c_bounds=None, x_c_margin=1.0):
    """Sigmoid Gaussian fit with AIC, AICc, parameters, standard errors, R-squared.

    By default x_c is constrained one rank inside the sampled range.
    """
    fit = _fit_sigmoid(
        x, y,
        x_c_bounds=x_c_bounds,
        x_c_margin=x_c_margin,
    )
    n = fit["n"]

    failed = {
        "aic": np.nan, "aicc": np.nan, "x_c": np.nan, "k": np.nan,
        "se_x_c": np.nan, "se_k": np.nan,
        "rss": np.nan, "r2": np.nan, "n": n, "ok": False,
    }

    if not fit["ok"]:
        return failed

    y_used = np.asarray(y, dtype=float)
    y_used = y_used[np.isfinite(y_used)]

    n_params = 5  # low + high + x_c + k + sigma

    return {
        "aic": float(_gaussian_aic(fit["rss"], n, n_params)),
        "aicc": float(_gaussian_aicc(fit["rss"], n, n_params)),
        "low": fit["low"],
        "high": fit["high"],
        "x_c": fit["x_c"],
        "k": fit["k"],
        "se_low": fit["se_low"],
        "se_high": fit["se_high"],
        "se_x_c": fit["se_x_c"],
        "se_k": fit["se_k"],
        "rss": fit["rss"],
        "r2": _r_squared(y_used, fit["rss"]),
        "n": n,
        "ok": True,
    }

def compare_models_per_participant(
    x, y, x_min, x_max, aic_margin=0.0,
    min_k=0.5, max_k=4.0, k_direction="any", max_se_x_c=1.0,
    x_c_margin=1.0,
):
    """Compare flat, linear, logarithmic, quadratic and sigmoid profiles.

    ``preferred`` and ``preferred_aicc`` preserve the original preregistered
    flat/linear/sigmoid comparison.

    ``shape`` is an additional diagnostic classification across:
        flat
        linear
        logarithmic
        quadratic_monotonic
        quadratic_nonmonotonic
        sigmoid

    The sigmoid is allowed to win the diagnostic shape comparison only when
    x_c lies in the sampled range and k passes the requested slope criterion.
    ``x_c_precise`` remains reported separately and is not required for
    ``shape`` so that model shape and boundary precision are not conflated.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x_used, y_used = x[ok], y[ok]

    flat_fit = fit_flat_aic(y_used, full=True)
    linear_fit = fit_linear_aic(x_used, y_used, full=True)
    log_fit = fit_log_aic(x_used, y_used, full=True)
    quadratic_fit = fit_quadratic_aic(x_used, y_used, full=True)
    xc_min = float(x_min) + float(x_c_margin)
    xc_max = float(x_max) - float(x_c_margin)

    sigmoid_fit = fit_sigmoid_aic(
        x_used,
        y_used,
        x_c_bounds=(xc_min, xc_max),
        x_c_margin=x_c_margin,
    )

    # Canonicalise sigmoid slope sign.
    low = sigmoid_fit.get("low", np.nan)
    high = sigmoid_fit.get("high", np.nan)
    amplitude = high - low

    if np.isfinite(amplitude) and amplitude != 0:
        k = sigmoid_fit.get("k", np.nan) * np.sign(amplitude)
    else:
        k = np.nan

    if k_direction == "positive":
        k_ok = bool(
            np.isfinite(k)
            and k > min_k
            and (not np.isfinite(max_k) or k <= max_k)
        )
    elif k_direction == "negative":
        k_ok = bool(
            np.isfinite(k)
            and k < -min_k
            and (not np.isfinite(max_k) or k >= -max_k)
        )
    else:
        k_ok = bool(
            np.isfinite(k)
            and abs(k) > min_k
            and (not np.isfinite(max_k) or abs(k) <= max_k)
        )

    x_c = sigmoid_fit.get("x_c", np.nan)
    se_x_c = sigmoid_fit.get("se_x_c", np.nan)

    x_c_in_range = bool(
        np.isfinite(x_c)
        and xc_min <= x_c <= xc_max
    )

    x_c_precise = bool(
        np.isfinite(se_x_c)
        and 0 < se_x_c <= max_se_x_c
    )

    # Quadratic diagnostics
    qa = quadratic_fit.get("a", np.nan)
    qb = quadratic_fit.get("b", np.nan)

    if np.isfinite(qa) and qa != 0 and np.isfinite(qb):
        x_turn = float(-qb / (2.0 * qa))
    else:
        x_turn = np.nan

    quadratic_nonmonotonic = bool(
        np.isfinite(x_turn)
        and x_min < x_turn < x_max
    )

    quadratic_label = (
        "quadratic_nonmonotonic"
        if quadratic_nonmonotonic
        else "quadratic_monotonic"
    )

    def _shape_winner(criterion="aic"):
        """Descriptive winner across all candidate shapes."""
        candidates = {
            "flat": flat_fit.get(criterion, np.nan),
            "linear": linear_fit.get(criterion, np.nan),
            "logarithmic": log_fit.get(criterion, np.nan),
            quadratic_label: quadratic_fit.get(criterion, np.nan),
        }

        # Only allow an interpretable sigmoid into the descriptive competition.
        if (
            sigmoid_fit.get("ok", False)
            and x_c_in_range
            and k_ok
        ):
            candidates["sigmoid"] = sigmoid_fit.get(criterion, np.nan)

        finite = {
            name: value
            for name, value in candidates.items()
            if np.isfinite(value)
        }

        if not finite:
            return "unclassified"

        return min(finite, key=finite.get)

    def _winner(flat_value, linear_value, sigmoid_value):
        """Original preregistered flat / linear / sigmoid decision."""
        usable = (
            np.isfinite(sigmoid_value)
            and np.isfinite(flat_value)
            and np.isfinite(linear_value)
            and sigmoid_fit.get("ok", False)
        )

        if (
            usable
            and sigmoid_value < flat_value - aic_margin
            and sigmoid_value < linear_value - aic_margin
            and x_c_in_range
            and k_ok
        ):
            return "sigmoid"

        if (
            np.isfinite(flat_value)
            and np.isfinite(linear_value)
            and linear_value < flat_value - aic_margin
        ):
            return "linear"

        return "flat"

    return {
        # AIC
        "aic_flat": flat_fit["aic"],
        "aic_linear": linear_fit["aic"],
        "aic_log": log_fit["aic"],
        "aic_quadratic": quadratic_fit["aic"],
        "aic_sigmoid": sigmoid_fit["aic"],

        "aic_delta_vs_flat": sigmoid_fit["aic"] - flat_fit["aic"],
        "aic_delta_vs_linear": sigmoid_fit["aic"] - linear_fit["aic"],
        "aic_delta_log_vs_linear": log_fit["aic"] - linear_fit["aic"],
        "aic_delta_quadratic_vs_linear": (
            quadratic_fit["aic"] - linear_fit["aic"]
        ),

        # AICc
        "aicc_flat": flat_fit["aicc"],
        "aicc_linear": linear_fit["aicc"],
        "aicc_log": log_fit["aicc"],
        "aicc_quadratic": quadratic_fit["aicc"],
        "aicc_sigmoid": sigmoid_fit["aicc"],

        "aicc_delta_vs_flat": sigmoid_fit["aicc"] - flat_fit["aicc"],
        "aicc_delta_vs_linear": sigmoid_fit["aicc"] - linear_fit["aicc"],
        "aicc_delta_log_vs_linear": log_fit["aicc"] - linear_fit["aicc"],
        "aicc_delta_quadratic_vs_linear": (
            quadratic_fit["aicc"] - linear_fit["aicc"]
        ),

        # RSS
        "rss_flat": flat_fit["rss"],
        "rss_linear": linear_fit["rss"],
        "rss_log": log_fit["rss"],
        "rss_quadratic": quadratic_fit["rss"],
        "rss_sigmoid": sigmoid_fit["rss"],

        # Sigmoid diagnostics
        "r2_sigmoid": sigmoid_fit.get("r2", np.nan),
        "x_c": x_c,
        "se_x_c": se_x_c,
        "k": k,
        "se_k": sigmoid_fit.get("se_k", np.nan),
        "amplitude_ms": abs(amplitude) if np.isfinite(amplitude) else np.nan,
        "sigmoid_ok": bool(sigmoid_fit.get("ok", False)),
        "x_c_in_range": x_c_in_range,
        "x_c_precise": x_c_precise,
        "k_ok": k_ok,

        # Quadratic diagnostics
        "quadratic_a": qa,
        "quadratic_b": qb,
        "quadratic_c": quadratic_fit.get("c", np.nan),
        "x_turn_quadratic": x_turn,
        "quadratic_nonmonotonic": quadratic_nonmonotonic,

        # Original preregistered result
        "preferred": _winner(
            flat_fit["aic"],
            linear_fit["aic"],
            sigmoid_fit["aic"],
        ),
        "preferred_aicc": _winner(
            flat_fit["aicc"],
            linear_fit["aicc"],
            sigmoid_fit["aicc"],
        ),

        # Additional descriptive shape result
        "shape": _shape_winner("aic"),
        "shape_aicc": _shape_winner("aicc"),
    }

def percentile_ci(values, low=2.5, high=97.5):
    values = pd.Series(values).dropna().astype(float)
    if len(values) == 0:
        return np.nan, np.nan
    return np.percentile(values, low), np.percentile(values, high)


def one_sample_summary(values, mu=0):
    values = pd.Series(values).dropna().astype(float)
    n = len(values)
    if n < 2:
        return pd.Series({
            "n": n,
            "mean": values.mean() if n else np.nan,
            "sd": np.nan,
            "t": np.nan,
            "p": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        })
    t, p = stats.ttest_1samp(values, popmean=mu)
    mean = values.mean()
    sd = values.std(ddof=1)
    sem = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    return pd.Series({
        "n": n,
        "mean": mean,
        "sd": sd,
        "t": t,
        "p": p,
        "ci_low": mean - tcrit * sem,
        "ci_high": mean + tcrit * sem,
    })


def subject_filter(df, subjects_to_keep=None, group=None, exclude_subjects=None):
    out = df.copy()

    if subjects_to_keep:
        out = out[out["subject"].isin(subjects_to_keep)]
    elif group is not None:
        out = out[out["group"].eq(group)]

    if exclude_subjects:
        out = out[~out["subject"].isin(exclude_subjects)]

    return out


def get_first_value(df, subject, col):
    vals = df.loc[df["subject"] == subject, col].dropna()
    if len(vals) == 0:
        return np.nan
    return vals.iloc[0]


def choose_clinical_pair(pps_df, collision_df):
    patients = sorted(set(pps_df.loc[pps_df["group"].eq("patient"), "subject"]))
    controls = sorted(set(pps_df.loc[pps_df["group"].eq("control"), "subject"]))

    patient = PATIENT_ID if PATIENT_ID is not None else (patients[0] if patients else None)
    control = MATCHED_CONTROL_ID if MATCHED_CONTROL_ID is not None else (controls[0] if controls else None)

    if patient is None or control is None:
        # No patient + matched control in this dataset (for example a single
        # healthy pilot session). Return Nones so Aim 2 / Aim 3b can skip
        # cleanly instead of raising. Set PATIENT_ID / MATCHED_CONTROL_ID to
        # force a specific pair.
        return None, None, None, None

    patient_sessions = (
        pps_df.loc[pps_df["subject"].eq(patient), ["subject", "session", "cohort"]]
        .drop_duplicates()
        .sort_values(["session", "cohort"])
    )

    if len(patient_sessions) == 0:
        raise ValueError("No PPS trials found for selected patient.")

    session = CLINICAL_SESSION if CLINICAL_SESSION is not None else patient_sessions.iloc[0]["session"]
    cohort = CLINICAL_COHORT if CLINICAL_COHORT is not None else patient_sessions.iloc[0]["cohort"]

    return patient, control, session, cohort


def bootstrap_mean_ci(values, n_boot=N_BOOT, ci=95, seed=RANDOM_SEED):
    """Bootstrap percentile CI of the mean across participants.

    Resamples participant-level values with replacement n_boot times,
    computes the mean each time, and returns the percentile CI.
    """
    rng_local = np.random.default_rng(seed)
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = len(values)
    if n < 2:
        return {"n": n, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng_local.choice(values, size=n, replace=True).mean()
    low = np.percentile(means, (100 - ci) / 2)
    high = np.percentile(means, 100 - (100 - ci) / 2)
    return {"n": n, "mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def wilcoxon_one_sided(values, alternative="greater"):
    """One-sided Wilcoxon signed-rank test against zero.

    alternative='greater' tests H1: median > 0.
    alternative='less' tests H1: median < 0.
    """
    values = pd.Series(values).dropna().astype(float)
    n = len(values)
    if n < 2:
        return {"n": n, "W": np.nan, "p": np.nan}
    try:
        result = stats.wilcoxon(values, alternative=alternative, zero_method="wilcox")
        return {"n": n, "W": float(result.statistic), "p": float(result.pvalue)}
    except Exception:
        return {"n": n, "W": np.nan, "p": np.nan}


def compute_sdc(s1_values, s2_values):
    """Smallest detectable change from paired session-1 and session-2 values.

    SDC = 1.96 * SD(s1 - s2) * sqrt(2).
    """
    pair = pd.DataFrame({"s1": s1_values, "s2": s2_values}).dropna()
    if len(pair) < 2:
        return np.nan
    diffs = pair["s1"] - pair["s2"]
    return float(1.96 * diffs.std(ddof=1) * np.sqrt(2))


def exact_spearman_permutation(x, y, alternative="two-sided"):
    """
    Exact permutation test for Spearman correlation.

    For N=5, this enumerates all 5! = 120 permutations.
    """
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)

    valid = x.notna() & y.notna()
    x = x.loc[valid].to_numpy()
    y = y.loc[valid].to_numpy()

    n = len(x)
    if n < 3:
        return {
            "n": n,
            "spearman_rho": np.nan,
            "exact_permutation_p": np.nan,
            "n_permutations": np.nan,
        }

    observed_rho, _ = stats.spearmanr(x, y)

    perm_rhos = []
    for y_perm in permutations(y):
        rho, _ = stats.spearmanr(x, np.asarray(y_perm))
        perm_rhos.append(rho)
    perm_rhos = np.asarray(perm_rhos)

    if alternative == "greater":
        p = (np.sum(perm_rhos >= observed_rho) + 1) / (len(perm_rhos) + 1)
    elif alternative == "less":
        p = (np.sum(perm_rhos <= observed_rho) + 1) / (len(perm_rhos) + 1)
    else:
        p = (np.sum(np.abs(perm_rhos) >= abs(observed_rho)) + 1) / (len(perm_rhos) + 1)

    return {
        "n": n,
        "spearman_rho": observed_rho,
        "exact_permutation_p": p,
        "n_permutations": len(perm_rhos),
    }


def joint_sign_proportion(boot_d1, boot_d2, sign1, sign2):
    """Proportion of paired bootstrap resamples in which D1 has sign1 and D2 has sign2.

    sign1, sign2 are +1 (positive) or -1 (negative).
    The two arrays must be paired (same bootstrap iteration order).
    """
    pair = pd.DataFrame({"d1": boot_d1, "d2": boot_d2}).dropna()
    if len(pair) == 0:
        return np.nan
    if sign1 > 0:
        cond1 = pair["d1"] > 0
    else:
        cond1 = pair["d1"] < 0
    if sign2 > 0:
        cond2 = pair["d2"] > 0
    else:
        cond2 = pair["d2"] < 0
    return float((cond1 & cond2).mean())


def classify_h7b_concordant(boot_d1, boot_d2, threshold=0.95):
    """Classify a pair of patient-minus-control bootstrap deviations.

    Concordant under: both negative in >= threshold of resamples.
    Concordant over: both positive in >= threshold of resamples.
    Discordant: opposite signs in >= threshold of resamples.
    Inconclusive: no joint sign pattern reaches threshold.
    """
    p_uu = joint_sign_proportion(boot_d1, boot_d2, -1, -1)
    p_oo = joint_sign_proportion(boot_d1, boot_d2, +1, +1)
    p_uo = joint_sign_proportion(boot_d1, boot_d2, -1, +1)
    p_ou = joint_sign_proportion(boot_d1, boot_d2, +1, -1)

    if p_uu >= threshold:
        cls = "supported: concordant under-calibration"
    elif p_oo >= threshold:
        cls = "supported: concordant over-calibration"
    elif (p_uo >= threshold) or (p_ou >= threshold):
        cls = "discordant"
    else:
        cls = "inconclusive"

    return {
        "p_both_negative": p_uu,
        "p_both_positive": p_oo,
        "p_d1neg_d2pos": p_uo,
        "p_d1pos_d2neg": p_ou,
        "classification": cls,
    }


def classify_h7b_opposite(boot_d_pps, boot_d_carry, threshold=0.95):
    """H7b.3 classification: predicted pattern is opposite-direction.

    Concordant under-calibration with increased carryover: D_PPS < 0 AND D_carryover > 0.
    Concordant over-calibration with reduced carryover: D_PPS > 0 AND D_carryover < 0.
    """
    p_pps_neg_carry_pos = joint_sign_proportion(boot_d_pps, boot_d_carry, -1, +1)
    p_pps_pos_carry_neg = joint_sign_proportion(boot_d_pps, boot_d_carry, +1, -1)
    p_both_neg = joint_sign_proportion(boot_d_pps, boot_d_carry, -1, -1)
    p_both_pos = joint_sign_proportion(boot_d_pps, boot_d_carry, +1, +1)

    if p_pps_neg_carry_pos >= threshold:
        cls = "supported: reduced PPS plasticity with increased carryover"
    elif p_pps_pos_carry_neg >= threshold:
        cls = "supported: exaggerated PPS plasticity with reduced carryover"
    elif (p_both_neg >= threshold) or (p_both_pos >= threshold):
        cls = "discordant"
    else:
        cls = "inconclusive"

    return {
        "p_pps_neg_carry_pos": p_pps_neg_carry_pos,
        "p_pps_pos_carry_neg": p_pps_pos_carry_neg,
        "p_both_negative": p_both_neg,
        "p_both_positive": p_both_pos,
        "classification": cls,
    }

def fit_sigmoid_reduced_aic(x, y, n_anchor=2, x_c_bounds=None, x_c_margin=1.0):
    """Sigmoid with the two plateaus fixed, so only the boundary is fitted.

    The four-parameter sigmoid asks seven points to determine a low plateau, a
    high plateau, a midpoint and a slope. That is three degrees of freedom, and
    in practice the plateaus and the midpoint trade against each other: the
    optimiser can move a plateau outward and the midpoint with it, reaching the
    edge of the admissible range without the fit getting much worse. This is
    what produces midpoints pinned at the bounds and standard errors in the
    thousands.

    Here the plateaus are read off the data instead of fitted: the near plateau
    is the mean of the ``n_anchor`` nearest positions, the far plateau the mean
    of the ``n_anchor`` farthest. Only the midpoint and the slope are then
    estimated, which halves the free parameters and leaves five degrees of
    freedom on seven points.

    The cost is an assumption: that the profile really is flat at both ends,
    and that the outermost points estimate those plateaus. Where a profile is
    still rising at the far edge, the far anchor is not a plateau and the fit
    will be wrong in a way the full model would have signalled by pushing x_c
    to the bound. ``amplitude`` and ``anchor_gap`` are returned so that case
    can be spotted: a small amplitude relative to the profile's own range means
    the anchors have not captured the transition.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    failed = {
        "aic": np.nan, "aicc": np.nan, "low": np.nan, "high": np.nan,
        "x_c": np.nan, "k": np.nan, "se_x_c": np.nan, "se_k": np.nan,
        "amplitude": np.nan, "anchor_gap": np.nan,
        "rss": np.nan, "r2": np.nan, "n": len(x), "ok": False,
    }

    if len(x) < 5 or len(np.unique(x)) < 4:
        return failed

    order = np.argsort(x)
    x, y = x[order], y[order]

    n_anchor = int(max(1, min(n_anchor, len(x) // 2 - 1)))
    low = float(np.mean(y[:n_anchor]))
    high = float(np.mean(y[-n_anchor:]))
    amplitude = high - low

    if not np.isfinite(amplitude) or amplitude == 0:
        return failed

    x_lo, x_hi = float(x.min()), float(x.max())

    if x_c_bounds is None:
        xc_lo = x_lo + float(x_c_margin)
        xc_hi = x_hi - float(x_c_margin)
    else:
        xc_lo, xc_hi = map(float, x_c_bounds)

    if not np.isfinite(xc_lo) or not np.isfinite(xc_hi) or xc_lo >= xc_hi:
        return failed

    def _curve(x_value, x_c, k):
        return low + (high - low) / (1.0 + np.exp(-k * (x_value - x_c)))

    bounds = ([xc_lo, -100.0], [xc_hi, 100.0])
    best, best_rss = None, np.inf

    for x_c0 in np.linspace(xc_lo, xc_hi, 5):
        for k0 in (0.5, 2.0, 5.0):
            try:
                popt, pcov = curve_fit(
                    _curve, x, y, p0=[float(x_c0), float(k0)],
                    bounds=bounds, maxfev=5000,
                )
            except Exception:
                continue

            rss = float(np.sum((y - _curve(x, *popt)) ** 2))

            if np.isfinite(rss) and rss < best_rss:
                best, best_rss = (popt, pcov), rss

    if best is None:
        return failed

    popt, pcov = best
    se = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan]
    n = len(x)
    n_params = 3  # x_c + k + sigma; the plateaus are fixed, not estimated

    return {
        "aic": float(_gaussian_aic(best_rss, n, n_params)),
        "aicc": float(_gaussian_aicc(best_rss, n, n_params)),
        "low": low,
        "high": high,
        "x_c": float(popt[0]),
        "k": float(popt[1]),
        "se_x_c": float(se[0]),
        "se_k": float(se[1]),
        "amplitude": float(amplitude),
        "anchor_gap": float(abs(amplitude) / (np.ptp(y) or np.nan)),
        "rss": best_rss,
        "r2": _r_squared(y, best_rss),
        "n": n,
        "ok": True,
    }