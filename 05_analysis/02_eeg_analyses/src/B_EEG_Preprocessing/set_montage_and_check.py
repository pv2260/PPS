import mne

def set_montage_and_check(
    raw: mne.io.Raw,
    montage_name: str = "standard_1020",
    drop_missing: bool = True,
    plot: bool = True,
):
    """
    Set a standard montage on a Raw object, report unrecognised channels,
    optionally drop them, and display the resulting sensor map.

    Channels absent from the montage (e.g. BIP bipolar channels used for
    EOG/ECG/EMG) have no scalp position and would otherwise be plotted at
    the centre of the head.  This function makes that explicit and gives you
    the choice of dropping them before further processing.

    Parameters
    ----------
    raw : mne.io.Raw
        Recording to modify.  Modified **in-place** (no copy is made so that
        the caller's variable is updated directly — consistent with how MNE
        handles ``set_montage`` and ``drop_channels``).
    montage_name : str
        Any name accepted by ``mne.channels.make_standard_montage()``.
        Common choices: ``"standard_1020"``, ``"easycap-M1"``,
        ``"biosemi64"``, ``"GSN-HydroCel-128"``.
        Run ``mne.channels.get_builtin_montages()`` to see all options.
    drop_missing : bool
        If True (default), channels not found in the montage are permanently
        removed from ``raw``.  Set to False to keep them (they will have no
        position and will appear at the head centre in plots).
    plot : bool
        If True (default), display the sensor map after setting the montage.

    Returns
    -------
    mne.io.Raw
        The same object passed in, modified in-place.

    Examples
    --------
    >>> raw_filt = set_montage_and_check(raw_filt)
    >>> raw_filt = set_montage_and_check(raw_filt, montage_name="easycap-M1",
    ...                                  drop_missing=False, plot=False)
    """
    montage = mne.channels.make_standard_montage(montage_name)

    recognised = [ch for ch in raw.ch_names if ch in montage.ch_names]
    missing    = [ch for ch in raw.ch_names if ch not in montage.ch_names]

    print(f"[set_montage_and_check] Montage : '{montage_name}'")
    print(f"  Recognised ({len(recognised)}) : {recognised}")
    print(f"  Not in montage ({len(missing)}) : {missing}")

    if missing:
        if drop_missing:
            print(f"  → Dropping {len(missing)} unrecognised channel(s).")
            raw.drop_channels(missing)
        else:
            print("  → Keeping unrecognised channels (no position; "
                  "they will appear at head centre in plots).")

    raw.set_montage(montage, on_missing="ignore", verbose=False)

    if plot:
        raw.plot_sensors(show_names=True, show=True)

    return raw