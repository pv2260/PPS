import mne
from pathlib import Path
from typing import Union


def load_raw(
    vhdr_path: Union[str, Path],
    preload: bool = True,
    verbose: bool = True,
):
    """
    Load a BrainVision recording from a .vhdr file.

    The matching .eeg and .vmk files must be in the same directory.

    Parameters
    ----------
    vhdr_path : str or Path
        Path to the .vhdr header file.
    preload : bool
        If True (default), load all data into RAM immediately.
        Required for in-place filtering.
    verbose : bool
        Print a short summary after loading.

    Returns
    -------
    mne.io.Raw
    """
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=preload,verbose=False)
    if verbose:
        print(f"[load_raw] {Path(vhdr_path).name}")
        print(f"  Channels : {len(raw.ch_names)}  — first 5: {raw.ch_names[:5]}")
        print(f"  Duration : {raw.times[-1]:.1f} s  |  Sfreq: {raw.info['sfreq']} Hz")
    return raw
