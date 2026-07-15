"""Label-permutation nulls for the single-case tests (H5, H6 Test A, H7 Test A)."""
import numpy as np
import pandas as pd

from .config import SLOW_LABEL, FAST_LABEL, N_BOOT, RANDOM_SEED
from .pps import compute_facilitation, near_far_index, sigmoid_boundary_by_subject
from .collision import compute_collision_indices_one_session, prepare_collision_trials


def permute_distance_labels_nearfar(pps_df, n_perm=N_BOOT, seed=RANDOM_SEED):
    """H5 permutation test: shuffle position labels (with position_rank) across patient's trials,
    recompute NearFar each time, build null distribution.
    """
    rng_local = np.random.default_rng(seed)
    df = pps_df.loc[pps_df["usable"]].copy()
    df = df[df["sensory_condition"].isin(["T", "VT"])]
    df = df.dropna(subset=["position_rank"])

    if len(df) == 0:
        return {"observed": np.nan, "permuted": np.array([]), "p": np.nan, "n_perm": 0}

    fac = compute_facilitation(df)
    nf = near_far_index(fac)
    observed = float(nf["nearfar_pps"].mean()) if len(nf) else np.nan

    position_arr = df["position"].to_numpy()
    rank_arr = df["position_rank"].to_numpy()
    if "distance_m" in df.columns:
        distance_arr = df["distance_m"].to_numpy()
    else:
        distance_arr = None

    perm_values = np.empty(n_perm)
    perm_values[:] = np.nan
    for i in range(n_perm):
        order = rng_local.permutation(len(df))
        perm_df = df.copy()
        perm_df["position"] = position_arr[order]
        perm_df["position_rank"] = rank_arr[order]
        if distance_arr is not None:
            perm_df["distance_m"] = distance_arr[order]
        boot_fac = compute_facilitation(perm_df)
        boot_nf = near_far_index(boot_fac)
        if len(boot_nf):
            perm_values[i] = boot_nf["nearfar_pps"].mean()

    perm_values = perm_values[np.isfinite(perm_values)]
    if len(perm_values) == 0 or not np.isfinite(observed):
        p = np.nan
    else:
        p = float((np.sum(perm_values >= observed) + 1) / (len(perm_values) + 1))

    return {"observed": observed, "permuted": perm_values, "p": p, "n_perm": int(len(perm_values))}


def permute_speed_labels_delta_pps(pps_df, n_perm=N_BOOT, seed=RANDOM_SEED):
    """H6 Test A: shuffle slow/fast labels across patient's PPS trials,
    recompute Delta_PPS each time, build null distribution.
    """
    rng_local = np.random.default_rng(seed)
    df = pps_df.loc[pps_df["usable"]].copy()
    df = df[df["sensory_condition"].isin(["T", "VT"])]
    df = df[df["speed"].isin([SLOW_LABEL, FAST_LABEL])]
    df = df.dropna(subset=["position_rank"])

    if len(df) == 0:
        return {"observed": np.nan, "permuted": np.array([]), "p": np.nan, "n_perm": 0}

    fac = compute_facilitation(df)
    delta_tbl = sigmoid_boundary_by_subject(fac, split_speed=True)
    observed = float(delta_tbl["delta_pps"].mean()) if len(delta_tbl) else np.nan

    speed_arr = df["speed"].to_numpy()

    perm_values = np.empty(n_perm)
    perm_values[:] = np.nan
    for i in range(n_perm):
        order = rng_local.permutation(len(df))
        perm_df = df.copy()
        perm_df["speed"] = speed_arr[order]
        boot_fac = compute_facilitation(perm_df)
        boot_delta = sigmoid_boundary_by_subject(boot_fac, split_speed=True)
        if len(boot_delta):
            perm_values[i] = boot_delta["delta_pps"].mean()

    perm_values = perm_values[np.isfinite(perm_values)]
    if len(perm_values) == 0 or not np.isfinite(observed):
        p = np.nan
    else:
        p = float((np.sum(perm_values >= observed) + 1) / (len(perm_values) + 1))

    return {"observed": observed, "permuted": perm_values, "p": p, "n_perm": int(len(perm_values))}


def permute_speed_labels_delta_coll(collision_df, n_perm=N_BOOT, seed=RANDOM_SEED):
    """H7 Test A: shuffle slow/fast labels across patient's collision trials,
    recompute Delta_coll each time, build null distribution.
    """
    rng_local = np.random.default_rng(seed)
    df = prepare_collision_trials(collision_df)
    df = df.loc[df["usable"]].copy()
    df = df[df["speed"].isin([SLOW_LABEL, FAST_LABEL])]
    df = df.dropna(subset=["offset_from_edge_cm", "response_hit"])

    if len(df) == 0:
        return {"observed": np.nan, "permuted": np.array([]), "p": np.nan, "n_perm": 0}

    observed = compute_collision_indices_one_session(df)["delta_coll"]
    observed = float(observed) if np.isfinite(observed) else np.nan

    speed_arr = df["speed"].to_numpy()

    perm_values = np.empty(n_perm)
    perm_values[:] = np.nan
    for i in range(n_perm):
        order = rng_local.permutation(len(df))
        perm_df = df.copy()
        perm_df["speed"] = speed_arr[order]
        idx = compute_collision_indices_one_session(perm_df)
        perm_values[i] = idx["delta_coll"]

    perm_values = perm_values[np.isfinite(perm_values)]
    if len(perm_values) == 0 or not np.isfinite(observed):
        p = np.nan
    else:
        p = float((np.sum(perm_values >= observed) + 1) / (len(perm_values) + 1))

    return {"observed": observed, "permuted": perm_values, "p": p, "n_perm": int(len(perm_values))}
