import pandas as pd
import os

def keep_rows_after_starts(input_csv, phase_block_value, save_path):
    """
    Read a vmrk-derived CSV and write a new CSV that keeps only the rows
    occurring after (and including) the FIRST marker whose type/description
    matches "phase_block" AND whose trigger_value equals `phase_block_value`.

    Parameters
    ----------
    input_csv : str
        Path to the source vmrk .csv file.
    phase_block_value : int
        The trigger_value associated with the phase_block marker to search for.
    output_csv : str, optional
        Path for the new filtered csv. If None, defaults to
        "<input_stem>_after_phase_block_<value>.csv" next to the input file.

    Returns
    -------
    str
        Path to the written output csv.
    """
    df = pd.read_csv(input_csv)

    required_cols = {"mk_type", "description", "trigger_value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input csv is missing required columns: {missing}")

    # Coerce trigger_value to numeric for safe comparison (handles blanks/NaN)
    trigger_numeric = pd.to_numeric(df["trigger_value"], errors="coerce")

    matches_value = trigger_numeric == phase_block_value

    candidate_rows = df[matches_value]

    if candidate_rows.empty:
        raise ValueError(
            f"No phase_block marker found with trigger_value == {phase_block_value}"
        )

    first_idx = candidate_rows.index[0]

    # Keep everything from that marker row onward
    filtered_df = df.loc[first_idx:].reset_index(drop=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    filtered_df.to_csv(save_path, index=False)
    print(f"Markers saved to: {save_path}")

    
    return filtered_df