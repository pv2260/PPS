import os
import json
import mne


def load_eeg_epochs(
    file_path: str,
    file_name: str,
    verbose: bool = True
):
    """
    Load saved EEG epochs and event information.

    Parameters
    ----------
    file_path : str
        Folder containing the saved epochs files.

    file_name : str
        Base name of the recording.

    verbose : bool
        Print loading information.

    Returns
    -------
    epochs : mne.Epochs
        Loaded MNE epochs object.

    event_id : dict
        Event name to event code mapping.

    events_detailed : dict
        Trial-level metadata containing stimulus, vibration,
        and click positions.
    """

    # --- Define filenames -----------------------------------------------
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


    # --- Load epochs -----------------------------------------------------
    if verbose:
        print("[load_eeg_epochs] Loading epochs...")
        print(f"  → {epochs_filename}")

    epochs = mne.read_epochs(
        epochs_filename,
        preload=True,
        verbose=False
    )


    # --- Load event ID ---------------------------------------------------
    event_id = None

    if os.path.exists(event_id_filename):

        if verbose:
            print("[load_eeg_epochs] Loading event ID...")
            print(f"  → {event_id_filename}")

        with open(event_id_filename, "r") as f:
            event_id = json.load(f)

        event_id = {
            key: int(value)
            for key, value in event_id.items()
        }

    else:
        if verbose:
            print(
                "[load_eeg_epochs] No event_id file found."
            )


    # --- Load events metadata -------------------------------------------
    events_detailed = None

    if os.path.exists(metadata_filename):

        if verbose:
            print("[load_eeg_epochs] Loading events metadata...")
            print(f"  → {metadata_filename}")

        with open(metadata_filename, "r") as f:
            events_detailed = json.load(f)

        # Convert dictionary keys back to integers
        events_detailed = {
            int(key): value
            for key, value in events_detailed.items()
        }

    else:
        if verbose:
            print(
                "[load_eeg_epochs] No events_detailed file found."
            )


    if verbose:
        print("[load_eeg_epochs] Done.")
        print(f"    Number of epochs: {len(epochs)}")

        if event_id is not None:
            print(f"    Event types: {len(event_id)}")

        if events_detailed is not None:
            print(f"    Metadata entries: {len(events_detailed)}")


    return epochs, event_id, events_detailed