import numpy as np


def compute_tfr_by_event(
    epochs,
    freqs=np.arange(4, 40, 1),
    n_cycles=None,
    baseline=(-0.2, 0),
    baseline_mode="logratio",
    method="morlet",
    verbose=True
):
    """
    Compute Morlet wavelet TFR separately for each event type.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data containing event_id.

    freqs : array-like
        Frequencies for wavelet transform.

    n_cycles : array-like | None
        Number of cycles for each frequency.
        If None:
            n_cycles = freqs / 2

    baseline : tuple
        Baseline correction interval.

    baseline_mode : str
        Baseline correction mode.
        Examples:
            'logratio', 'percent', 'zscore'

    method : str
        TFR method.
        Default: 'morlet'

    verbose : bool
        Print progress.


    Returns
    -------
    tfr_single_dict : dict
        Dictionary containing single-trial TFRs.
        Each value is an EpochsTFR object.

    tfr_average_dict : dict
        Dictionary containing averaged TFRs.
        Each value is an AverageTFR object.
    """


    if n_cycles is None:
        n_cycles = freqs / 2


    tfr_single_dict = {}
    tfr_average_dict = {}


    for event_name, event_code in epochs.event_id.items():

        if verbose:
            print(
                f"Computing TFR: {event_name}"
            )


        # --------------------------------------------------
        # Select trials of this condition
        # --------------------------------------------------

        epochs_group = epochs[event_name]


        if len(epochs_group) == 0:
            if verbose:
                print(
                    f"  No trials found for {event_name}"
                )
            continue


        # --------------------------------------------------
        # Compute single-trial TFR
        # --------------------------------------------------

        power = epochs_group.compute_tfr(
            method=method,
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=False,
            average=False
        )


        # --------------------------------------------------
        # Baseline correction
        # --------------------------------------------------

        power.apply_baseline(
            baseline=baseline,
            mode=baseline_mode
        )


        # Store single-trial power
        tfr_single_dict[event_name] = power


        # --------------------------------------------------
        # Average across trials
        # --------------------------------------------------

        tfr_average_dict[event_name] = power.average()


        if verbose:
            print(
                f"  Trials: {len(epochs_group)}"
            )


    return tfr_single_dict, tfr_average_dict