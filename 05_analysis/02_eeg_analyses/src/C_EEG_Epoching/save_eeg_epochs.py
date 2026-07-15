import os
import json
import numpy as np
import mne


def save_eeg_epochs(
    file_path: str,
    file_name: str,
    epochs: mne.Epochs,
    event_id: dict = None,
    events_detailed: dict = None,
    verbose: bool = True,
):
    """
    Save EEG epochs and associated event information.

    Parameters
    ----------
    file_path : str
        Folder where epochs will be saved.

    file_name : str
        Base name of the recording.

    epochs : mne.Epochs
        Created MNE epochs object.

    event_id : dict | None
        Dictionary mapping event names to event codes.

    events_detailed : dict | None
        Trial-level metadata generated during marker grouping.
        Contains stimulus, vibration, and click positions.

    verbose : bool
        Print saving information.
    """

    # --- 1. Create output folder -------------------------------------------
    if not os.path.exists(file_path):
        os.makedirs(file_path)

        if verbose:
            print(
                f"[save_eeg_epochs] Created folder: {file_path}"
            )


    # --- 2. Define filenames -----------------------------------------------
    epochs_filename = os.path.join(
        file_path,
        f"{file_name}_epochs.fif"
    )

    event_id_filename = os.path.join(
        file_path,
        f"{file_name}_event_id.json"
    )

    metadata_filename = os.path.join(
        file_path,
        f"{file_name}_events_detailed.json"
    )


    # --- 3. Save epochs -----------------------------------------------------
    if verbose:
        print("[save_eeg_epochs] Saving epochs...")

    epochs.save(
        epochs_filename,
        overwrite=True
    )


    # --- 4. Save event_id dictionary ---------------------------------------
    if event_id is not None:

        if verbose:
            print("[save_eeg_epochs] Saving event ID...")

        with open(event_id_filename, "w") as f:
            json.dump(
                event_id,
                f,
                indent=4
            )


    # --- 5. Save events metadata -------------------------------------------
    if events_detailed is not None:
        if verbose:
            print("[save_eeg_epochs] Saving events metadata...")
        # Convert numpy values to JSON-compatible values
        metadata_json = {}
        for key, value in events_detailed.items():
            metadata_json[str(key)] = {}

            for k, v in value.items():

                if isinstance(v, (np.integer,)):
                    v = int(v)

                elif isinstance(v, (np.floating,)):
                    v = float(v)

                elif isinstance(v, np.ndarray):
                    v = v.tolist()

                metadata_json[str(key)][k] = v


        with open(metadata_filename, "w") as f:
            json.dump(
                metadata_json,
                f,
                indent=4
            )