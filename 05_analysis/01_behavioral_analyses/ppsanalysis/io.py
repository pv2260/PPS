"""Raw Unity export loading and tidying.

Function bodies are ported VERBATIM from the notebook (extracted with ast, not
retyped, so they cannot have silently drifted). Only the plumbing changed:
constants that were notebook globals now come from config.
"""
import os, re, glob, json

import numpy as np
import pandas as pd

from .config import TRIAL_FILE, POSITION_COL_PATTERN, SUBJECT_META, PILOT_DATA_DIR, OUT_DIR


def read_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def subject_id(x):
    return str(x).strip().replace("-task1", "").replace("-task2", "")


def pick(df, *cols, default=np.nan):
    for c in cols:
        if c in df.columns:
            return df[c]
    return pd.Series(default, index=df.index)


def clean_pps(df, subject, session, meta):
    out = pd.DataFrame({
        "subject": subject,
        "session": session,
        "group": meta["group"],
        "cohort": meta["cohort"],
        "has_dbs": meta["has_dbs"],
        "trial": pd.to_numeric(pick(df, "trial_number", "trial"), errors="coerce"),
        "block": pd.to_numeric(pick(df, "block_number", "block"), errors="coerce"),
        "sensory_condition": pick(df, "trial_type", "sensory_condition").astype(str).str.strip().str.upper(),
        "position": pick(df, "distance_level", "position").astype(str).str.strip(),
        "speed": pick(df, "current_speed", "speed").astype(str).str.strip().str.lower(),
        "width": pick(df, "width").astype(str).str.strip().str.lower(),
        "vibrotactile_onset_ms": pd.to_numeric(pick(df, "vibrotactile_onset_ms"), errors="coerce"),
        "response_time_ms": pd.to_numeric(pick(df, "response_time_ms"), errors="coerce"),
    })

    # D1 = closest, D7 = farthest
    out["position_rank"] = pd.to_numeric(
        out["position"].str.extract(r"D(\d+)")[0],
        errors="coerce"
    )

    if "reaction_time_ms" in df.columns:
        out["rt_ms"] = pd.to_numeric(df["reaction_time_ms"], errors="coerce")
    else:
        out["rt_ms"] = out["response_time_ms"] - out["vibrotactile_onset_ms"]

    out["response_made"] = (
        pick(df, "response_made")
        .astype(str).str.strip().str.lower()
        .isin(["true", "1", "1.0", "yes"])
    )

    # Final PPS usability rule
    out["usable"] = out["rt_ms"].between(100, 3000, inclusive="neither")

    return out


def clean_collision(df, subject, session, meta):
    shoulder_width = meta.get("shoulder_width_cm", 42.0)
    if pd.isna(shoulder_width) or shoulder_width <= 0:
        shoulder_width = 42.0

    out = pd.DataFrame({
        "subject": pick(df, "subject_id", default=subject).astype(str).apply(subject_id),
        "session": pd.to_numeric(pick(df, "session_number", default=session), errors="coerce").fillna(session).astype(int),
        "group": meta["group"],
        "cohort": meta["cohort"],
        "has_dbs": meta["has_dbs"],
        "block": pd.to_numeric(pick(df, "block_number", "block"), errors="coerce"),
        "trial": pd.to_numeric(pick(df, "trial_number", "trial"), errors="coerce"),
        "trial_type": pick(df, "trial_type").astype(str).str.strip().str.lower(),
        "speed": pick(df, "current_speed", "speed").astype(str).str.strip().str.lower(),
        "trajectory_offset_cm": pd.to_numeric(pick(df, "trajectory_offset_cm"), errors="coerce"),
        "participant_response": pick(df, "participant_response").astype(str).str.strip().str.upper(),
        "correct_response": pick(df, "correct_response").astype(str).str.strip().str.upper(),
        "accuracy": pd.to_numeric(pick(df, "accuracy"), errors="coerce"),
        "rt_ms": pd.to_numeric(pick(df, "reaction_time_ms", "rt_ms"), errors="coerce"),
        "trial_interrupted": pd.to_numeric(pick(df, "trial_interrupted"), errors="coerce").fillna(0).astype(int),
    })

    out["shoulder_width_cm"] = shoulder_width
    out["offset_from_midline_cm"] = out["trajectory_offset_cm"]
    out["offset_from_edge_cm"] = out["trajectory_offset_cm"].abs() - shoulder_width / 2
    out["response_hit"] = out["participant_response"].eq("HIT").astype(int)

    out["usable"] = (
        out["speed"].isin(["slow", "fast"])
        & out["offset_from_edge_cm"].notna()
        & out["participant_response"].isin(["HIT", "MISS"])
        & out["trial_interrupted"].eq(0)
    )

    return out


def load_unity_exports(data_dir, verbose=True):
    pps_rows = []
    collision_rows = []
    subject_rows = []

    files = sorted(glob.glob(
        os.path.join(data_dir, "**", "sub-*_session-*_task*_trials*.csv"),
        recursive=True,
    ))

    if verbose:
        print("Trial files found:", len(files))
        for f in files:
            print(" -", os.path.basename(f), "| folder:", os.path.basename(os.path.dirname(f)))

    for path in files:
        name = os.path.basename(path)
        match = TRIAL_FILE.search(name)

        if match is None:
            print("Skipping filename I do not recognize:", name)
            continue

        subject = subject_id(match.group("subject"))
        session = int(match.group("session"))
        task = int(match.group("task"))

        meta = {
            "group": "unknown",
            "cohort": "unknown",
            "has_dbs": False,
            "shoulder_width_cm": 42.0,
            **SUBJECT_META.get(subject, {}),
        }

        df = read_csv(path)

        subject_rows.append({
            "subject": subject,
            "session": session,
            "task": task,
            "file": name,
            "folder": os.path.basename(os.path.dirname(path)),
            "n_raw_rows": len(df),
            **meta,
        })

        if df.empty:
            continue

        if task == 1:
            pps_rows.append(clean_pps(df, subject, session, meta))
        elif task == 2:
            collision_rows.append(clean_collision(df, subject, session, meta))

    pps = pd.concat(pps_rows, ignore_index=True) if pps_rows else pd.DataFrame()
    collision = pd.concat(collision_rows, ignore_index=True) if collision_rows else pd.DataFrame()
    subjects = pd.DataFrame(subject_rows)

    if not pps.empty:
        pps = pps.drop_duplicates(
            ["subject", "session", "trial", "block", "sensory_condition", "position", "speed"]
        )

    if not collision.empty:
        collision = collision.drop_duplicates(
            ["subject", "session", "block", "trial", "trial_type", "speed", "trajectory_offset_cm"]
        )

    return pps, collision, subjects


def make_analysis_csvs(data_dir=PILOT_DATA_DIR, out_dir=OUT_DIR):
    pps_trials, collision_trials, subjects = load_unity_exports(data_dir)

    os.makedirs(out_dir, exist_ok=True)

    for name, df in {
        "pps_trials.csv": pps_trials,
        "collision_trials.csv": collision_trials,
        "subjects_summary.csv": subjects,
    }.items():
        df.to_csv(os.path.join(out_dir, name), index=False)

    print(f"Loaded from:  {data_dir}")
    print(f"Saved CSVs to: {out_dir}")

    return pps_trials, collision_trials, subjects
