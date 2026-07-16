"""
export_repairs.py

File-scoped repairs for individual raw exports that carry known logging bugs.

The default loader stays untouched. Repairs are opt-in per file, registered in
`config.EXPORT_REPAIRS` by (subject, session, task). A file that is not
registered passes through unchanged, and every repair additionally self-checks
for its own bug signature, so an accidental or stale registration cannot
silently corrupt a clean export.

Loader hook: add one line where you read each raw trials CSV, AFTER you have
parsed subject / session / task from the filename and BEFORE any RT parsing or
usable computation:

    from . import export_repairs
    df = export_repairs.apply(df, subject=subject, session=session, task=task)
"""

from __future__ import annotations

import pandas as pd

from . import config


_TRUE_TOKENS = {"true", "1", "1.0", "yes", "y", "t"}


# ---------------------------------------------------------------------------
# Individual repairs
# ---------------------------------------------------------------------------

def _repair_pam_s1_task1(df: pd.DataFrame) -> pd.DataFrame:
    """sub-pam, session-1, task-1 only. Two export bugs, each self-guarded so
    this is a no-op on a correctly exported file.

    Bug A  Trailing column shift. The exporter omitted the `trial_interrupted`
           column, so the trailing ISO timestamp landed in the
           `trial_interrupted` slot and `timestamp` came out empty on every row.
           Left alone, every trial reads as interrupted and 0 survive.

    Bug B  `reaction_time_ms` holds the loom-arrival schedule (constant within
           each distance level, logged in seconds), not the response latency.
           The real, trial-varying latency, already in ms, is in
           `response_time_ms`. We put the latency where the loader expects the
           RT and keep the schedule under an honest name.
    """
    df = df.copy()

    # --- Bug A: restore timestamp, make trial_interrupted a real boolean ---
    if "trial_interrupted" in df.columns and "timestamp" in df.columns:
        ti = df["trial_interrupted"].astype("string")
        looks_like_timestamp = ti.str.match(r"\d{4}-\d{2}-\d{2}T", na=False)
        if df["timestamp"].isna().all() and looks_like_timestamp.any():
            df["timestamp"] = df["trial_interrupted"]
            df["trial_interrupted"] = False

    if "trial_interrupted" in df.columns:
        col = df["trial_interrupted"]
        if col.dtype == object or str(col.dtype) == "string":
            df["trial_interrupted"] = (
                col.astype("string").str.strip().str.lower().isin(_TRUE_TOKENS)
            )
        else:
            df["trial_interrupted"] = col.fillna(0).astype(bool)

    # --- Bug B: put the true latency where the RT is read from ---
    if {"reaction_time_ms", "response_time_ms", "distance_level"} <= set(df.columns):
        sched = pd.to_numeric(df["reaction_time_ms"], errors="coerce")
        # signature: constant within every distance level => it is a schedule
        is_schedule = sched.groupby(df["distance_level"]).std().fillna(0).max() < 1e-6
        if is_schedule:
            df["loom_arrival_ms_scheduled"] = sched * 1000.0  # s -> ms, honest name
            # write the latency into BOTH candidate RT columns so the fix holds
            # regardless of which one the default loader reads.
            df["reaction_time_ms"] = pd.to_numeric(df["response_time_ms"], errors="coerce")

    return df


# name -> callable
_REPAIRS = {
    "pam_s1_task1": _repair_pam_s1_task1,
}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply(df: pd.DataFrame, subject, session, task) -> pd.DataFrame:
    """Apply a registered repair when (subject, session, task) appears in
    config.EXPORT_REPAIRS. Any other file is returned unchanged."""
    try:
        key = (str(subject), int(session), int(task))
    except (TypeError, ValueError):
        return df

    name = getattr(config, "EXPORT_REPAIRS", {}).get(key)
    if not name:
        return df

    fn = _REPAIRS.get(name)
    if fn is None:
        raise KeyError(
            f"config.EXPORT_REPAIRS lists unknown repair '{name}' for {key}. "
            f"Known repairs: {sorted(_REPAIRS)}"
        )
    return fn(df)