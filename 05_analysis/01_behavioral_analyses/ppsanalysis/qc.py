"""Trial usability marking and data-quality checks.

mark_pps_usable is the single gate deciding which trials enter every hypothesis.
Change it here and it changes everywhere. That is the point of pulling it out.
"""
import numpy as np
import pandas as pd

from .config import RT_MIN_MS, RT_MAX_MS, DROP_POSITIONS
from ._compat import display


def mark_pps_usable(pps, rt_min_ms=100, rt_max_ms=1000, drop_positions=None):
    pps = pps.copy()

    drop_positions = [] if drop_positions is None else list(drop_positions)

    valid_rt = pps["rt_ms"].between(rt_min_ms, rt_max_ms, inclusive="neither")
    dropped_pos = pps["position_rank"].isin(drop_positions)

    pps["usable"] = valid_rt & ~dropped_pos

    pps["exclusion_reason"] = "usable"
    pps.loc[pps["sensory_condition"].eq("V"), "exclusion_reason"] = "visual_only"
    pps.loc[pps["rt_ms"].isna(), "exclusion_reason"] = "missing_rt"
    pps.loc[pps["rt_ms"] <= rt_min_ms, "exclusion_reason"] = "too_fast"
    pps.loc[pps["rt_ms"] >= rt_max_ms, "exclusion_reason"] = "too_slow"
    pps.loc[dropped_pos, "exclusion_reason"] = "dropped_position"

    return pps


def summarize_trials(name, df):
    n = len(df)
    usable = int(df["usable"].sum()) if n and "usable" in df.columns else 0

    return {
        "task": name,
        "rows": n,
        "usable": usable,
        "excluded": n - usable,
        "usable_%": round(100 * usable / n, 1) if n else 0,
    }


def pps_qc(pps):
    if pps.empty:
        print("No PPS trials.")
        return

    print("PPS trial counts by condition:")
    display(
        pps.pivot_table(
            index=["subject", "session"],
            columns="sensory_condition",
            values="trial",
            aggfunc="count",
            fill_value=0,
        )
    )

    print("PPS usable trials by distance:")
    display(
        pps[pps["sensory_condition"].isin(["T", "VT"])]
        .groupby(["subject", "position", "position_rank"], dropna=False)
        .agg(
            total=("trial", "size"),
            usable=("usable", "sum"),
        )
        .reset_index()
        .assign(
            excluded=lambda d: d["total"] - d["usable"],
            usable_percent=lambda d: round(100 * d["usable"] / d["total"], 1),
        )
        .sort_values(["subject", "position_rank"])
    )

    print("PPS exclusion reasons:")
    display(
        pps["exclusion_reason"]
        .value_counts(dropna=False)
        .rename_axis("reason")
        .reset_index(name="n")
    )


def quick_qc(pps, collision=None):
    rows = []

    if pps is not None and len(pps):
        rows.append({
            "dataset": "PPS",
            "rows": len(pps),
            "usable": int(pps["usable"].sum()),
            "excluded": int((~pps["usable"]).sum()),
            "usable_%": round(100 * pps["usable"].mean(), 1),
        })

    if collision is not None and len(collision):
        rows.append({
            "dataset": "Hit/Miss",
            "rows": len(collision),
            "usable": int(collision["usable"].sum()),
            "excluded": int((~collision["usable"]).sum()),
            "usable_%": round(100 * collision["usable"].mean(), 1),
        })

    display(pd.DataFrame(rows))
