import pandas as pd


def repair_vmrk_D7_triggers(vmrk_csv, mismatch_csv, output_csv, trigger_map):
    """
    Repair missing D7 triggers in a vmrk trigger CSV.

    Parameters
    ----------
    vmrk_csv : str
        Path to original vmrk trigger csv.

    mismatch_csv : str
        Path to mismatch report csv containing:
        - issue_type
        - csv_trigger_value_used
        - vmrk_next_marker_num
        - vmrk_next_position

    output_csv : str
        Path where repaired vmrk csv is saved.

    trigger_map : dict
        Dictionary mapping trigger values to names.
    """

    # Load files
    vmrk = pd.read_csv(vmrk_csv)
    mismatch = pd.read_csv(mismatch_csv)


    # Get trigger values associated with D7
    D7_values = [
        trig for trig, name in trigger_map.items()
        if "D7" in name
    ]

    print("D7 triggers:", D7_values)


    # Select missing D7 trials
    missing_D7 = mismatch[
        (mismatch["issue_type"] == "extra_in_csv") &
        (mismatch["csv_trigger_value_used"].isin(D7_values))
    ].copy()

    print(f"Missing D7 triggers to repair: {len(missing_D7)}")


    insert_rows = []


    for _, miss in missing_D7.iterrows():

        trig_value = int(miss["csv_trigger_value_used"])

        # Check that we know the following VMRK marker
        if pd.isna(miss["vmrk_next_marker_num"]):
            print(
                f"No following VMRK marker found for missing trigger {trig_value}"
            )
            continue

        # Marker number is 1-based, dataframe index is 0-based
        next_marker_idx = int(miss["vmrk_next_marker_num"]) - 1
        next_marker_value = int(miss["vmrk_next_trigger_value"])

        # Only repair if the next marker is a trigger 63
        if next_marker_value != 63:
            continue

        # Position should be identical to following trigger (63)
        pos_trigger63 = miss["vmrk_next_position"]

        # Create repaired marker
        new_row = {
            "marker_num": None,
            "mk_type": "Stimulus",
            "description": f"s{trig_value:02d}",
            "trigger_value": trig_value,
            "position": pos_trigger63,
            "size": 1,
            "channel": 0
        }

        insert_rows.append(
            (next_marker_idx, new_row)
        )


    # Insert from bottom to top to preserve indices
    # for idx, row in sorted(insert_rows, reverse=True):
    for idx, row in sorted(insert_rows, key=lambda x: x[0], reverse=True):

        vmrk = pd.concat(
            [
                vmrk.iloc[:idx],
                pd.DataFrame([row]),
                vmrk.iloc[idx:]
            ],
            ignore_index=True
        )


    # Update marker numbering
    vmrk["marker_num"] = range(1, len(vmrk)+1)


    # Save repaired vmrk
    vmrk.to_csv(output_csv, index=False)


    print(
        f"Saved repaired vmrk file:\n{output_csv}"
    )


    return vmrk