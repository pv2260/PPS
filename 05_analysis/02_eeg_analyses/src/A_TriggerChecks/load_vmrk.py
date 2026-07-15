import re
import os
import pandas as pd

def load_vmrk(vmrk_path, save_path):
    """
    Parse a BrainVision .vmrk marker file.

    Returns
    -------
    pd.DataFrame with columns:
        marker_num   : int, marker index as written in the file (Mk1, Mk2, ...)
        mk_type      : str, e.g. 'Stimulus', 'New Segment'
        description  : str, raw description field, e.g. 's17'
        trigger_value: Int64 (nullable), numeric part of the description
                       (NaN for markers with no numeric code, e.g. 'New Segment')
        position     : int, sample index in the EEG recording
        size         : int
        channel      : int
    Rows are kept in file order (== chronological order in the recording).
    """
    pattern = re.compile(
        r"^Mk(?P<num>\d+)=(?P<type>[^,]*),(?P<desc>[^,]*),(?P<pos>\d+),(?P<size>\d+),(?P<chan>\d+)"
    )
    rows = []
    with open(vmrk_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.match(line.strip())
            if not m:
                continue
            desc = m.group("desc").strip()
            num_match = re.search(r"(\d+)", desc)
            trigger_value = int(num_match.group(1)) if num_match else None
            rows.append(
                {
                    "marker_num": int(m.group("num")),
                    "mk_type": m.group("type").strip(),
                    "description": desc,
                    "trigger_value": trigger_value,
                    "position": int(m.group("pos")),
                    "size": int(m.group("size")),
                    "channel": int(m.group("chan")),
                }
            )
    df = pd.DataFrame(rows).sort_values("marker_num").reset_index(drop=True)
    df["trigger_value"] = df["trigger_value"].astype("Int64")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Markers saved to: {save_path}")


    return df