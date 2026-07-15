def load_trigger_map(xlsx_path, sheet="Task1"):
    """
    Read a trigger codebook sheet and build {trigger_value(int): label(str)}.

    Handles both:
      - "structured" rows that have Value/Distance/TrialType/Width/Speed
        (e.g. VisualOnly / VibrotactileOnly / Both trial codes)
      - "flat" rows used for generic events (Value + a single name column,
        e.g. 61 -> 'Task event', 62 -> 'Click', 63 -> 'Vibration', or the
        130-137 'Task2 other events' codes)

    Section header rows (e.g. 'VisualOnly (distance always None)') are
    skipped automatically because their Value column is not numeric.

    Returns
    -------
    trigger_map : dict[int, str]
    raw_df      : pd.DataFrame, the sheet as read (for inspection/debugging)
    """
    raw_df = pd.read_excel(xlsx_path, sheet_name=sheet)
    cols = list(raw_df.columns)

    trigger_map = {}
    for _, row in raw_df.iterrows():
        value = row[cols[0]]
        if not isinstance(value, (int, float)) or pd.isna(value):
            continue  # section header / spacer row
        value = int(value)

        if len(cols) >= 3 and pd.notna(row.get(cols[2])):
            # structured row: TrialType present -> build a descriptive label
            parts = []
            trial_type = row.get(cols[2])
            distance = row.get(cols[1])
            width = row.get(cols[3]) if len(cols) > 3 else None
            speed = row.get(cols[4]) if len(cols) > 4 else None
            if pd.notna(trial_type):
                parts.append(str(trial_type))
            if pd.notna(distance) and str(distance) != "None":
                parts.append(str(distance))
            if pd.notna(width):
                parts.append(str(width))
            if pd.notna(speed):
                parts.append(str(speed))
            label = "_".join(parts) if parts else f"Value{value}"
        else:
            # flat row: second column holds the plain-text event name
            name = row.get(cols[1])
            label = str(name) if pd.notna(name) else f"Value{value}"

        trigger_map[value] = label

    return trigger_map, raw_df