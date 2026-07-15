import mne
from typing import Optional
import matplotlib.pyplot as plt


def basic_filtering(
    raw: mne.io.Raw,
    l_freq: float = 0.1,
    h_freq: float = 100.0,
    notch_freq: float = 50.0,
    reference: Optional[str] = None,
    verbose: bool = True,
):
    """
    Apply band-pass filtering, a notch filter, and an optional re-reference.

    Parameters
    ----------
    raw : mne.io.Raw
    l_freq : float
        High-pass cut-off in Hz (default 0.1).  Use 1.0 for ERP analyses.
    h_freq : float
        Low-pass cut-off in Hz (default 40.0).
    notch_freq : float
        Power-line frequency to notch out (default 50 Hz for Europe).
    reference : str or None
        Electrode name(s) or "average" for CAR.  None = keep original reference.
    verbose : bool

    Returns
    -------
    mne.io.Raw  (copy, original is not modified)
    """
    raw_filt = raw.copy()
    raw_filt.filter(l_freq=l_freq, h_freq=h_freq, method="fir", fir_window="hamming", verbose=False)
    raw_filt.notch_filter(freqs=[notch_freq], verbose=False)

    if reference is not None:
        raw_filt.set_eeg_reference(reference, projection=(reference == "average"),
                              verbose=False)

    if verbose:
        ref_str = f"  Reference: {reference}" if reference else ""
        print(f"[preprocess] {l_freq}–{h_freq} Hz band-pass | "
              f"{notch_freq} Hz notch{ref_str}")
        
    # ---------------------------------------------------------
    # Plot before and after filtering
    # ---------------------------------------------------------

    channels_to_plot = [
        "Fz",     # frontal
        "FCz",    # frontal/middle
        "Cz",     # middle
        "Pz",     # middle parietal
        "P3",     # left parietal
        "P4"      # right parietal
    ]

    t_start = 600  # seconds
    duration = 120 # seconds

    # Extract 20 seconds
    raw_segment = raw.copy().crop(
        tmin=t_start,
        tmax=t_start + duration
    )

    # Keep only selected channels
    raw_segment.pick(channels_to_plot)

    # Apply filtering
    filtered_segment = raw_filt.copy().crop(
        tmin=t_start,
        tmax=t_start + duration
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(14, 8),
        sharex=True
    )

    for i, ch in enumerate(channels_to_plot):

        row = i // 2
        col = i % 2

        raw_data = raw_segment.get_data(picks=[ch])[0] * 1e6
        filt_data = filtered_segment.get_data(picks=[ch])[0] * 1e6

        times = raw_segment.times + t_start

        # Plot before filtering
        axes[row, col].plot(
            times,
            raw_data,
            label="Before filtering"
        )

        # Plot after filtering
        axes[row, col].plot(
            times,
            filt_data,
            label="After filtering"
        )

        axes[row, col].set_title(ch)
        axes[row, col].set_ylabel("µV")
        axes[row, col].grid(True)
        axes[row, col].legend()

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


    return raw_filt
