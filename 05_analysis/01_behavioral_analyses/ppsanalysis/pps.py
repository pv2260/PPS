"""PPS (Task 1) derived measures.

facilitation -> near_far_index -> sigmoid_boundary_by_subject -> Delta_PPS
"""
import numpy as np
import pandas as pd

from .config import SLOW_LABEL, FAST_LABEL, N_BOOT, RANDOM_SEED
from .stats_utils import safe_sigmoid_fit


def compute_facilitation(pps_df):
    """Median RT per subject/session/speed/position, then facilitation = T - VT.

    The output keeps `position` (label), `position_rank` (data-driven x-axis), and
    optionally `distance_m` (from setup.json) attached for convenience. distance_m
    is merged back after the pivot to avoid losing rows when it is all NaN.
    """
    df = pps_df.loc[pps_df["usable"]].copy()
    df = df[df["sensory_condition"].isin(["T", "VT"])]

    index_cols = ["subject", "session", "group", "cohort", "speed",
                  "position", "position_rank"]

    med = (
        df.groupby(index_cols + ["sensory_condition"],
                   as_index=False)["rt_ms"].median()
    )

    wide = med.pivot_table(
        index=index_cols, columns="sensory_condition", values="rt_ms"
    ).reset_index()

    wide.columns.name = None
    if "T" not in wide.columns:
        wide["T"] = np.nan
    if "VT" not in wide.columns:
        wide["VT"] = np.nan

    wide["facilitation_ms"] = wide["T"] - wide["VT"]

    # Attach optional distance_m via position lookup (kept out of the pivot index
    # so all-NaN distance_m does not cause rows to drop).
    if "distance_m" in df.columns and df["distance_m"].notna().any():
        dist_map = (
            df[["subject", "session", "position", "distance_m"]]
            .drop_duplicates(subset=["subject", "session", "position"])
        )
        wide = wide.merge(dist_map, on=["subject", "session", "position"], how="left")
    else:
        wide["distance_m"] = np.nan

    return wide


def near_far_index(fac_df, near_n=2, far_n=2, axis_col="position_rank"):
    """Near-far index per subject/session.

    Near = the `near_n` positions with smallest axis value (closest to body if
    axis_col = "position_rank").  Far = the `far_n` positions with largest axis
    value (farthest from body).
    """
    rows = []
    columns = ["subject", "session", "group", "cohort", "near_mean", "far_mean", "nearfar_pps"]

    for keys, g in fac_df.groupby(["subject", "session", "group", "cohort"], dropna=False):
        by_axis = (
            g.dropna(subset=[axis_col, "facilitation_ms"])
             .groupby(axis_col, as_index=False)["facilitation_ms"].mean()
             .sort_values(axis_col)
        )
        if len(by_axis) == 0:
            continue
        near = by_axis.head(near_n)["facilitation_ms"].mean()
        far = by_axis.tail(far_n)["facilitation_ms"].mean()
        rows.append({
            "subject": keys[0],
            "session": keys[1],
            "group": keys[2],
            "cohort": keys[3],
            "near_mean": near,
            "far_mean": far,
            "nearfar_pps": near - far,
        })

    return pd.DataFrame(rows, columns=columns)


def sigmoid_boundary_by_subject(fac_df, split_speed=False, axis_col="position_rank"):
    """Fit a sigmoid to facilitation as a function of `axis_col` per subject/session.

    axis_col defaults to "position_rank" (data-driven; 1 = closest, N = farthest).
    For a per-speed fit, set split_speed=True; the function returns one row per
    subject/session with x_c_slow, x_c_fast, and delta_pps = x_c_fast - x_c_slow.
    """
    rows = []
    group_cols = ["subject", "session", "group", "cohort"]

    for keys, g in fac_df.groupby(group_cols, dropna=False):
        base = {
            "subject": keys[0],
            "session": keys[1],
            "group": keys[2],
            "cohort": keys[3],
        }

        if not split_speed:
            avg = (
                g.dropna(subset=[axis_col, "facilitation_ms"])
                 .groupby(axis_col, as_index=False)["facilitation_ms"].mean()
            )
            fit = safe_sigmoid_fit(avg[axis_col], avg["facilitation_ms"])
            rows.append({**base, "x_c": fit.get("x_c", np.nan), "sigmoid_ok": fit.get("ok", False)})

        else:
            estimates = {}
            for speed in [SLOW_LABEL, FAST_LABEL]:
                gg = g[g["speed"].eq(speed)].dropna(subset=[axis_col, "facilitation_ms"])
                avg = gg.groupby(axis_col, as_index=False)["facilitation_ms"].mean()
                fit = safe_sigmoid_fit(avg[axis_col], avg["facilitation_ms"])
                estimates[f"x_c_{speed}"] = fit.get("x_c", np.nan)
                estimates[f"ok_{speed}"] = fit.get("ok", False)

            rows.append({
                **base,
                **estimates,
                "delta_pps": estimates.get("x_c_fast", np.nan) - estimates.get("x_c_slow", np.nan),
            })

    return pd.DataFrame(rows)


def bootstrap_pps_delta_one_session(df, n_boot=N_BOOT, seed=RANDOM_SEED):
    rng_local = np.random.default_rng(seed)
    df = df.loc[df["usable"]].copy()
    df = df[df["sensory_condition"].isin(["T", "VT"])]
    df = df[df["speed"].isin([SLOW_LABEL, FAST_LABEL])]

    if len(df) == 0:
        return pd.DataFrame()

    group_cols = ["speed", "position", "sensory_condition"]
    groups = [g.copy() for _, g in df.groupby(group_cols)]

    rows = []
    for i in range(n_boot):
        parts = []
        for g in groups:
            parts.append(g.sample(n=len(g), replace=True, random_state=int(rng_local.integers(0, 1_000_000_000))))
        boot_df = pd.concat(parts, ignore_index=True)
        fac = compute_facilitation(boot_df)
        delta_tbl = sigmoid_boundary_by_subject(fac, split_speed=True)
        delta = delta_tbl["delta_pps"].iloc[0] if len(delta_tbl) else np.nan
        rows.append({"boot": i, "delta_pps": delta})

    return pd.DataFrame(rows)
