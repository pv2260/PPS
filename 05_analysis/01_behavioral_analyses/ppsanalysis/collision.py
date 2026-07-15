"""Hit/Miss (Task 2) derived measures: PSE, Delta_coll, carryover, accuracy."""
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from .config import SLOW_LABEL, FAST_LABEL, N_BOOT, RANDOM_SEED


def code_speed(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            str(SLOW_LABEL).lower(): -1,
            str(FAST_LABEL).lower(): 1,
        })
    )


def safe_logistic_regression(X, y, min_trials=6):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[ok]
    y = y[ok]

    if len(y) < min_trials or len(np.unique(y)) < 2:
        return None

    try:
        model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=1000,
        )
        model.fit(X, y)
        return model
    except Exception:
        return None


def compute_accuracy_series(df):
    if "accuracy" in df.columns:
        return pd.to_numeric(df["accuracy"], errors="coerce")

    if {"correct_response", "participant_response"}.issubset(df.columns):
        return (
            df["correct_response"].astype(str).str.upper()
            == df["participant_response"].astype(str).str.upper()
        ).astype(float)

    return pd.Series(np.nan, index=df.index)


def prepare_collision_trials(df):
    out = df.copy()

    sort_cols = [
        c for c in ["subject", "session", "block", "trial"]
        if c in out.columns
    ]
    out = out.sort_values(sort_cols).copy()

    if "previous_speed" not in out.columns:
        group_cols = [
            c for c in ["subject", "session", "block"]
            if c in out.columns
        ]

        if group_cols:
            out["previous_speed"] = out.groupby(group_cols)["speed"].shift(1)
        else:
            out["previous_speed"] = out["speed"].shift(1)

    if "post_switch" not in out.columns:
        out["post_switch"] = (
            out["previous_speed"].notna()
            & out["speed"].notna()
            & out["previous_speed"].ne(out["speed"])
        ).astype(int)

    return out


def empty_collision_indices():
    return {
        "n_trials": 0,
        "accuracy": np.nan,
        "near_boundary_accuracy": np.nan,
        "n_near_boundary_trials": 0,
        "pse_slow": np.nan,
        "pse_fast": np.nan,
        "delta_coll": np.nan,
        "slope": np.nan,
        "pse_bias": np.nan,
        "min_offset_cm": np.nan,
        "max_offset_cm": np.nan,
        "pse_in_range": False,
        "carryover_cm": np.nan,
        "main_model_ok": False,
        "carryover_model_ok": False,
    }


def compute_collision_indices_one_session(df):
    """
    Computes Hit/Miss indices for one subject/session.

    Main model:
        logit(P(HIT)) = b0 + b1*offset + b2*speed

    with:
        slow = -1
        fast = +1

    Therefore:
        PSE slow = -(b0 - b_speed) / b_offset
        PSE fast = -(b0 + b_speed) / b_offset
        Delta_coll = PSE_fast - PSE_slow

    Positive Delta_coll means:
        fast boundary is farther outward than slow boundary.
    """

    df = prepare_collision_trials(df)

    if "usable" in df.columns:
        df = df.loc[df["usable"]].copy()
    else:
        df = df.copy()

    df["speed"] = df["speed"].astype(str).str.strip().str.lower()
    df = df[df["speed"].isin([
        str(SLOW_LABEL).lower(),
        str(FAST_LABEL).lower(),
    ])]

    df = df.dropna(subset=["offset_from_edge_cm", "response_hit"])

    if len(df) == 0:
        return empty_collision_indices()

    df["accuracy_clean"] = compute_accuracy_series(df)

    accuracy = df["accuracy_clean"].mean()

    near = df["trial_type"].astype(str).str.contains("near", na=False)

    near_boundary_accuracy = (
        df.loc[near, "accuracy_clean"].mean()
        if near.any()
        else np.nan
    )

    n_near_boundary_trials = int(near.sum())

    # --------------------------------------------------
    # Main speed-dependent psychometric model
    # --------------------------------------------------

    model_df = df.copy()
    model_df["speed_code"] = code_speed(model_df["speed"])
    model_df = model_df.dropna(
        subset=["offset_from_edge_cm", "speed_code", "response_hit"]
    )

    pse_slow = np.nan
    pse_fast = np.nan
    delta_coll = np.nan
    slope = np.nan
    pse_bias = np.nan
    pse_in_range = False
    main_model_ok = False

    min_offset_cm = model_df["offset_from_edge_cm"].min()
    max_offset_cm = model_df["offset_from_edge_cm"].max()

    X = model_df[["offset_from_edge_cm", "speed_code"]].to_numpy()
    y = model_df["response_hit"].astype(int).to_numpy()

    model = safe_logistic_regression(X, y)

    if model is not None:
        b0 = model.intercept_[0]
        b_offset = model.coef_[0][0]
        b_speed = model.coef_[0][1]

        slope = b_offset

        # Expected direction:
        # larger offset_from_edge_cm = object farther outside shoulder
        # so P(HIT) should decrease.
        # Therefore b_offset should be negative.
        valid_slope = not np.isclose(b_offset, 0) and b_offset < 0

        if valid_slope:
            pse_slow = -(b0 - b_speed) / b_offset
            pse_fast = -(b0 + b_speed) / b_offset

            delta_coll = pse_fast - pse_slow
            pse_bias = (pse_slow + pse_fast) / 2

            pse_in_range = (
                min_offset_cm <= pse_slow <= max_offset_cm
                and min_offset_cm <= pse_fast <= max_offset_cm
            )

            main_model_ok = bool(pse_in_range)

    # --------------------------------------------------
    # Carryover model
    # --------------------------------------------------

    carryover_cm = np.nan
    carryover_model_ok = False

    carry_df = df.copy()
    carry_df["current_speed_code"] = code_speed(carry_df["speed"])
    carry_df["previous_speed_code"] = code_speed(carry_df["previous_speed"])

    carry_df["post_switch_int"] = pd.to_numeric(
        carry_df["post_switch"],
        errors="coerce"
    ).fillna(0).astype(int)

    carry_df["previous_x_postswitch"] = (
        carry_df["previous_speed_code"]
        * carry_df["post_switch_int"]
    )

    carry_df = carry_df.dropna(
        subset=[
            "offset_from_edge_cm",
            "response_hit",
            "current_speed_code",
            "previous_x_postswitch",
        ]
    )

    if len(carry_df) > 0:
        Xc = carry_df[
            [
                "offset_from_edge_cm",
                "current_speed_code",
                "previous_x_postswitch",
            ]
        ].to_numpy()

        yc = carry_df["response_hit"].astype(int).to_numpy()

        carry_model = safe_logistic_regression(Xc, yc)

        if carry_model is not None:
            b_offset_carry = carry_model.coef_[0][0]
            b_carry = carry_model.coef_[0][2]

            valid_carry_slope = (
                not np.isclose(b_offset_carry, 0)
                and b_offset_carry < 0
            )

            if valid_carry_slope:
                carryover_cm = -b_carry / b_offset_carry
                carryover_model_ok = True

    return {
        "n_trials": len(df),
        "accuracy": accuracy,
        "near_boundary_accuracy": near_boundary_accuracy,
        "n_near_boundary_trials": n_near_boundary_trials,
        "pse_slow": pse_slow,
        "pse_fast": pse_fast,
        "delta_coll": delta_coll,
        "slope": slope,
        "pse_bias": pse_bias,
        "min_offset_cm": min_offset_cm,
        "max_offset_cm": max_offset_cm,
        "pse_in_range": pse_in_range,
        "carryover_cm": carryover_cm,
        "main_model_ok": main_model_ok,
        "carryover_model_ok": carryover_model_ok,
    }


def compute_collision_indices_by_subject(df):
    rows = []

    group_cols = ["subject", "session", "group", "cohort"]

    for keys, g in df.groupby(group_cols, dropna=False):
        indices = compute_collision_indices_one_session(g)

        rows.append({
            "subject": keys[0],
            "session": keys[1],
            "group": keys[2],
            "cohort": keys[3],
            **indices,
        })

    return pd.DataFrame(rows)


def bootstrap_collision_indices_one_session(df, n_boot=N_BOOT, seed=RANDOM_SEED):
    rng_local = np.random.default_rng(seed)
    df = prepare_collision_trials(df)
    df = df.loc[df["usable"]].copy()
    df = df.dropna(subset=["offset_from_edge_cm", "speed", "response_hit"])

    if len(df) == 0:
        return pd.DataFrame()

    if "trial_type" in df.columns:
        group_cols = ["speed", "trial_type"]
    else:
        group_cols = ["speed"]

    groups = [g.copy() for _, g in df.groupby(group_cols)]
    rows = []

    for i in range(n_boot):
        parts = []
        for g in groups:
            parts.append(g.sample(n=len(g), replace=True, random_state=int(rng_local.integers(0, 1_000_000_000))))
        boot_df = pd.concat(parts, ignore_index=True)
        idx = compute_collision_indices_one_session(boot_df)
        rows.append({"boot": i, **idx})

    return pd.DataFrame(rows)
