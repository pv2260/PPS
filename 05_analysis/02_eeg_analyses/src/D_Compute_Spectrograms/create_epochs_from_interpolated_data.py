import numpy as np
import mne


def create_epochs_from_interpolated_data(
    interp_data,
    sfreq,
    N_baseline_points,
    baseline_start,
    ch_names=None,
    ch_types=None,
    verbose=True
):
    """
    Convert interpolated EEG data dictionary into an MNE EpochsArray.

    Parameters
    ----------
    interp_data : dict
        Dictionary containing interpolated trials.

        Example:
        {
            "VibrotactileOnly_D1_Narrow_Fast":
                ndarray(n_trials, n_channels, n_times),
            "VisualOnly_Narrow_Slow":
                ndarray(n_trials, n_channels, n_times)
        }

    sfreq : float
        Original EEG sampling frequency.

    N_baseline_points : int
        Number of points used for the interpolated baseline segment.

    baseline_start : float
        Start time of the baseline in seconds.
        Example:
            -0.2

    ch_names : list | None
        Channel names.
        If None, channels are named CH1, CH2...

    ch_types : list | None
        Channel types.
        Default: all EEG.

    verbose : bool
        Print information.

    Returns
    -------
    epochs : mne.EpochsArray
        Interpolated epochs.

    event_id : dict
        Condition name to event code mapping.
    """
    # --------------------------------------------------
    # 1) Check that all interpolated epochs have same length
    # --------------------------------------------------
    condition_names = list(interp_data.keys())
    n_times_all = {
        data.shape[2]
        for data in interp_data.values()
    }
    if len(n_times_all) > 1:
        raise ValueError(
            f"Interpolated conditions have different lengths: {n_times_all}. "
            "All conditions must have identical time points."
        )
    n_times = n_times_all.pop()


    # --------------------------------------------------
    # 2) Create event IDs
    # --------------------------------------------------
    event_id = {
        name: idx + 1
        for idx, name in enumerate(condition_names)
    }

    # --------------------------------------------------
    # 3) Build data and events arrays
    # --------------------------------------------------
    all_data = []
    all_events = []
    sample_counter = 0

    for condition, data in interp_data.items():
        n_trials = data.shape[0]
        all_data.append(data)
        code = event_id[condition]
        for _ in range(n_trials):
            all_events.append(
                [
                    sample_counter,
                    0,
                    code
                ]
            )
            sample_counter += n_times

    data = np.concatenate(
        all_data,
        axis=0
    )

    events = np.array(
        all_events,
        dtype=int
    )

    # --------------------------------------------------
    # 4) Define channels
    # --------------------------------------------------
    n_channels = data.shape[1]

    if ch_names is None:
        ch_names = [
            f"CH{i+1}"
            for i in range(n_channels)
        ]

    if ch_types is None:
        ch_types = [
            "eeg"
        ] * n_channels

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )


    # --------------------------------------------------
    # 5) Compute tmin
    # --------------------------------------------------

    # baseline was interpolated as:
    # np.linspace(baseline_start, 0, N_baseline_points)

    # index of stimulus onset
    baseline_duration = (
        N_baseline_points - 1
    ) / sfreq

    tmin = baseline_start


    # --------------------------------------------------
    # 6) Create EpochsArray
    # --------------------------------------------------
    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        event_id=event_id,
        tmin=tmin,
        verbose=False
    )

    if verbose:
        zero_idx = np.argmin(
            np.abs(epochs.times)
        )

        print("--------------------------------")
        print("Interpolated Epochs created")
        print("--------------------------------")
        print(f"Number of epochs : {len(epochs)}")
        print(f"Number of channels : {n_channels}")
        print(f"Number of points : {n_times}")
        print(f"Sampling frequency : {sfreq} Hz")
        print(f"tmin : {epochs.times[0]:.4f} s")
        print(
            f"t=0 located at sample {zero_idx} "
            f"(expected {N_baseline_points-1})"
        )

        print("\nEvent IDs:")
        for name, code in event_id.items():
            print(f"  {code}: {name}")

    return epochs, event_id