"""Model fitting, bootstrap, permutation and classification helpers.

Task-agnostic: nothing here knows about PPS or Hit/Miss specifically.
"""
from itertools import permutations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from .config import N_BOOT, RANDOM_SEED, PATIENT_ID, MATCHED_CONTROL_ID, CLINICAL_SESSION, CLINICAL_COHORT


def sigmoid(x, low, high, x_c, k):
    """Simple sigmoid. k controls steepness."""
    return low + (high - low) / (1 + np.exp(k * (x - x_c)))


def safe_sigmoid_fit(x, y):
    """Fit a sigmoid and return x_c. Returns NaN if fitting fails."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    if len(x) < 4 or len(np.unique(x)) < 4:
        return {"x_c": np.nan, "ok": False}

    try:
        p0 = [np.nanmin(y), np.nanmax(y), np.nanmedian(x), 5.0]
        bounds = (
            [-np.inf, -np.inf, np.nanmin(x) - 0.5, -100],
            [ np.inf,  np.inf, np.nanmax(x) + 0.5,  100],
        )
        popt, _ = curve_fit(sigmoid, x, y, p0=p0, bounds=bounds, maxfev=20000)
        return {
            "low": popt[0],
            "high": popt[1],
            "x_c": popt[2],
            "k": popt[3],
            "ok": True,
        }
    except Exception:
        return {"x_c": np.nan, "ok": False}


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


def fit_flat_aic(y):
    """Flat (intercept-only) Gaussian fit. Returns AIC."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 2:
        return np.nan
    rss = float(np.sum((y - y.mean()) ** 2))
    sigma2 = rss / n if rss > 0 else 1e-12
    log_lik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    return 2 * 2 - 2 * log_lik  # 2 params: mean + sigma


def fit_linear_aic(x, y):
    """Linear Gaussian fit. Returns AIC."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3:
        return np.nan
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    rss = float(np.sum((y - yhat) ** 2))
    sigma2 = rss / n if rss > 0 else 1e-12
    log_lik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    return 2 * 3 - 2 * log_lik  # 3 params: slope + intercept + sigma


def fit_sigmoid_aic(x, y):
    """Sigmoid Gaussian fit. Returns AIC and parameters."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return {"aic": np.nan, "x_c": np.nan, "k": np.nan, "ok": False}
    try:
        p0 = [float(np.nanmin(y)), float(np.nanmax(y)), float(np.nanmedian(x)), 5.0]
        bounds = (
            [-np.inf, -np.inf, float(np.nanmin(x)) - 0.5, -100.0],
            [np.inf, np.inf, float(np.nanmax(x)) + 0.5, 100.0],
        )
        popt, _ = curve_fit(sigmoid, x, y, p0=p0, bounds=bounds, maxfev=20000)
        yhat = sigmoid(x, *popt)
        rss = float(np.sum((y - yhat) ** 2))
        sigma2 = rss / n if rss > 0 else 1e-12
        log_lik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
        aic = 2 * 5 - 2 * log_lik  # 5 params: low + high + x_c + k + sigma
        return {"aic": float(aic), "low": float(popt[0]), "high": float(popt[1]),
                "x_c": float(popt[2]), "k": float(popt[3]), "ok": True}
    except Exception:
        return {"aic": np.nan, "x_c": np.nan, "k": np.nan, "ok": False}


def compare_models_per_participant(x, y, x_min, x_max, aic_margin=2.0):
    """Compare flat / linear / sigmoid for one participant's facilitation profile.

    Sigmoid is preferred only if its AIC is lower than both others by aic_margin,
    AND x_c falls inside [x_min, x_max], AND |k| is non-trivial.
    """
    flat = fit_flat_aic(y)
    lin = fit_linear_aic(x, y)
    sig = fit_sigmoid_aic(x, y)

    sig_aic = sig["aic"]
    if (np.isfinite(sig_aic) and np.isfinite(flat) and np.isfinite(lin)
        and sig.get("ok", False)
        and sig_aic < flat - aic_margin
        and sig_aic < lin - aic_margin
        and (x_min <= sig.get("x_c", np.nan) <= x_max)
        and abs(sig.get("k", 0)) > 0.5):
        preferred = "sigmoid"
    elif np.isfinite(flat) and np.isfinite(lin) and lin < flat - aic_margin:
        preferred = "linear"
    else:
        preferred = "flat"

    return {
        "aic_flat": flat,
        "aic_linear": lin,
        "aic_sigmoid": sig_aic,
        "x_c": sig.get("x_c", np.nan),
        "k": sig.get("k", np.nan),
        "preferred": preferred,
    }


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


def classify_h8b_concordant(boot_d1, boot_d2, threshold=0.95):
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


def classify_h8b_opposite(boot_d_pps, boot_d_carry, threshold=0.95):
    """H8b.3 classification: predicted pattern is opposite-direction.

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
