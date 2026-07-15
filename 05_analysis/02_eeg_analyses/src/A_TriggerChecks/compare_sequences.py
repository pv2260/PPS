import difflib
import pandas as pd
import os


def compare_sequences(vmrk_df, csv_df, save_path, trigger_map=None, normalize_map=None):
    """
    Positional (chronological-order) diff between the EEG trigger sequence
    (.vmrk) and the task-logged trigger sequence (.csv), using
    difflib.SequenceMatcher.

    Parameters
    ----------
    vmrk_df : output of load_vmrk()
    csv_df  : output of load_task_csv()
    trigger_map : dict[int,str], optional - useåd only to add readable
                  labels to the report.
    normalize_map : dict[int,int], optional - remaps known CSV logging
                  quirks onto their true EEG code before comparing, e.g.
                  {127: 63}  # pps_vib_fired is logged as 127 but the EEG
                             # marker actually sent is 63 (Vibration)
                  Applied to the CSV sequence only.

    Returns
    -------
    mismatches_df : pd.DataFrame, one row per problem (missing / extra /
                    mismatched trigger), ready to write to CSV.
    alignment_df  : pd.DataFrame, the full opcode-level alignment
                    (for debugging / deeper inspection).
    """
    trigger_map = trigger_map or {}
    normalize_map = normalize_map or {}

    # Exclude non-numeric markers (e.g. "New Segment") from the EEG side.
    vmrk_seq_df = vmrk_df.dropna(subset=["trigger_value"]).reset_index(drop=True)
    a = vmrk_seq_df["trigger_value"].astype(int).tolist()

    csv_seq_df = csv_df.dropna(subset=["TriggerValue"]).reset_index(drop=True)
    b_raw = csv_seq_df["TriggerValue"].astype(int).tolist()
    b = [normalize_map.get(v, v) for v in b_raw]

    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    opcodes = sm.get_opcodes()

    def vmrk_row(i):
        r = vmrk_seq_df.iloc[i]
        return r["marker_num"], int(r["trigger_value"]), int(r["position"])

    def csv_row(j):
        r = csv_seq_df.iloc[j]
        raw_val = int(r["TriggerValue"])
        return j, float(r["Time"]), r.get("EventCode"), r.get("TrialId"), r.get("Category"), raw_val

    alignment_rows = []
    mismatch_rows = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                mk_num, mk_val, mk_pos = vmrk_row(i)
                j_idx, t, ev, trial, cat, raw_val = csv_row(j)
                alignment_rows.append({
                    "issue_type": "match", "vmrk_marker_num": mk_num,
                    "vmrk_trigger_value": mk_val, "vmrk_position": mk_pos,
                    "csv_row_index": j_idx, "csv_time_s": t, "csv_event_code": ev,
                    "csv_trial_id": trial, "csv_category": cat,
                    "csv_trigger_value_raw": raw_val, "csv_trigger_value_used": b[j],
                    "label": trigger_map.get(mk_val, f"Unknown({mk_val})"),
                    "note": "",
                })
            continue

        if tag == "replace":
            # pair up element-by-element; leftover on either side becomes
            # extra_in_vmrk / extra_in_csv
            n = max(i2 - i1, j2 - j1)
            for k in range(n):
                i = i1 + k if i1 + k < i2 else None
                j = j1 + k if j1 + k < j2 else None
                if i is not None and j is not None:
                    mk_num, mk_val, mk_pos = vmrk_row(i)
                    j_idx, t, ev, trial, cat, raw_val = csv_row(j)
                    row = {
                        "issue_type": "value_mismatch", "vmrk_marker_num": mk_num,
                        "vmrk_trigger_value": mk_val, "vmrk_position": mk_pos,
                        "csv_row_index": j_idx, "csv_time_s": t, "csv_event_code": ev,
                        "csv_trial_id": trial, "csv_category": cat,
                        "csv_trigger_value_raw": raw_val, "csv_trigger_value_used": b[j],
                        "label": (f"{trigger_map.get(mk_val, f'Unknown({mk_val})')} vs "
                                  f"{trigger_map.get(b[j], f'Unknown({b[j]})')}"),
                        "note": "Same position in both sequences, different trigger code",
                    }
                elif i is not None:
                    mk_num, mk_val, mk_pos = vmrk_row(i)
                    row = {
                        "issue_type": "extra_in_vmrk", "vmrk_marker_num": mk_num,
                        "vmrk_trigger_value": mk_val, "vmrk_position": mk_pos,
                        "csv_row_index": None, "csv_time_s": None, "csv_event_code": None,
                        "csv_trial_id": None, "csv_category": None,
                        "csv_trigger_value_raw": None, "csv_trigger_value_used": None,
                        "label": trigger_map.get(mk_val, f"Unknown({mk_val})"),
                        "note": "Marker present in EEG recording but no matching row in CSV log",
                    }
                else:
                    j_idx, t, ev, trial, cat, raw_val = csv_row(j)
                    row = {
                        "issue_type": "extra_in_csv", "vmrk_marker_num": None,
                        "vmrk_trigger_value": None, "vmrk_position": None,
                        "csv_row_index": j_idx, "csv_time_s": t, "csv_event_code": ev,
                        "csv_trial_id": trial, "csv_category": cat,
                        "csv_trigger_value_raw": raw_val, "csv_trigger_value_used": b[j],
                        "label": trigger_map.get(b[j], f"Unknown({b[j]})"),
                        "note": "Row logged in CSV task file but no matching marker in EEG recording",
                    }
                alignment_rows.append(row)
                mismatch_rows.append(row)

        elif tag == "delete":
            # present in vmrk (a) only -> missing from csv
            for i in range(i1, i2):
                mk_num, mk_val, mk_pos = vmrk_row(i)
                row = {
                    "issue_type": "extra_in_vmrk", "vmrk_marker_num": mk_num,
                    "vmrk_trigger_value": mk_val, "vmrk_position": mk_pos,
                    "csv_row_index": None, "csv_time_s": None, "csv_event_code": None,
                    "csv_trial_id": None, "csv_category": None,
                    "csv_trigger_value_raw": None, "csv_trigger_value_used": None,
                    "label": trigger_map.get(mk_val, f"Unknown({mk_val})"),
                    "note": "MRow in EEG rec but no matching row in CSV log",
                }
                alignment_rows.append(row)
                mismatch_rows.append(row)
                
        elif tag == "insert":
            # present in csv (b) only -> missing from vmrk

            # Find the next vmrk marker after this missing trigger
            if i1 < len(vmrk_seq_df):
                next_marker_num, next_trigger, next_position = vmrk_row(i1)
            else:
                next_marker_num = None
                next_trigger = None
                next_position = None

            for j in range(j1, j2):
                j_idx, t, ev, trial, cat, raw_val = csv_row(j)

                row = {
                    "issue_type": "extra_in_csv",

                    # No corresponding VMRK marker exists
                    "vmrk_marker_num": None,
                    "vmrk_trigger_value": None,
                    "vmrk_position": None,

                    # Next VMRK marker after the missing one
                    "vmrk_next_marker_num": next_marker_num,
                    "vmrk_next_trigger_value": next_trigger,
                    "vmrk_next_position": next_position,

                    "csv_row_index": j_idx,
                    "csv_time_s": t,
                    "csv_event_code": ev,
                    "csv_trial_id": trial,
                    "csv_category": cat,
                    "csv_trigger_value_raw": raw_val,
                    "csv_trigger_value_used": b[j],

                    "label": trigger_map.get(
                        b[j],
                        f"Unknown({b[j]})"
                    ),
                    "note": "Row in CSV log but no matching marker in EEG rec",
                }

                alignment_rows.append(row)
                mismatch_rows.append(row)

    #     elif tag == "insert":
    #         # present in csv (b) only -> missing from vmrk
    #         for j in range(j1, j2):
    #             j_idx, t, ev, trial, cat, raw_val = csv_row(j)
    #             row = {
    #                 "issue_type": "extra_in_csv", "vmrk_marker_num": None,
    #                 "vmrk_trigger_value": None, "vmrk_position": None,
    #                 "csv_row_index": j_idx, "csv_time_s": t, "csv_event_code": ev,
    #                 "csv_trial_id": trial, "csv_category": cat,
    #                 "csv_trigger_value_raw": raw_val, "csv_trigger_value_used": b[j],
    #                 "label": trigger_map.get(b[j], f"Unknown({b[j]})"),
    #                 "note": "Row in CSV log but no matching marker in EEG rec",
    #             }
    #             alignment_rows.append(row)
    #             mismatch_rows.append(row)

    # col_order = ["issue_type", "label", "note",
    #              "vmrk_marker_num", "vmrk_trigger_value", "vmrk_position",
    #              "csv_row_index", "csv_time_s", "csv_event_code", "csv_trial_id",
    #              "csv_category", "csv_trigger_value_raw", "csv_trigger_value_used"]

        

    col_order = [
    "issue_type", "label", "note",

    "vmrk_marker_num",
    "vmrk_trigger_value",
    "vmrk_position",

    "vmrk_next_marker_num",
    "vmrk_next_trigger_value",
    "vmrk_next_position",

    "csv_row_index",
    "csv_time_s",
    "csv_event_code",
    "csv_trial_id",
    "csv_category",
    "csv_trigger_value_raw",
    "csv_trigger_value_used"
    ]

    alignment_df = pd.DataFrame(alignment_rows)[col_order] if alignment_rows else pd.DataFrame(columns=col_order)
    mismatches_df = pd.DataFrame(mismatch_rows)[col_order] if mismatch_rows else pd.DataFrame(columns=col_order)

    print(f'Found {len(mismatches_df)} mismatched markers')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mismatches_df.to_csv(save_path, index=False)

    return mismatches_df, alignment_df