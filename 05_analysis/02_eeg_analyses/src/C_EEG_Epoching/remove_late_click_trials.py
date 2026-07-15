import numpy as np
import pandas as pd


def remove_late_click_trials(
    events,
    events_metadata,
    sfreq,
    max_click_delay=3.0,
    verbose=True
):
    """
    Remove EEG events where vibration end/click occurs too late after stimulus.

    Parameters
    ----------
    events : np.ndarray
        MNE events array [sample, 0, event_id].

    events_metadata : dict
        Metadata dictionary from group_eeg_markers().

    sfreq : float
        EEG sampling frequency.

    max_click_delay : float
        Maximum allowed delay between stimulus and click/vibration end (seconds).

    verbose : bool
        Print removal summary.

    Returns
    -------
    filtered_events : np.ndarray
        Filtered MNE events array.

    filtered_metadata : dict
        Filtered event metadata.

    removed_events_df : pd.DataFrame
        Removed events and their delays.
    """

    keep_indices = []
    removed_events = []

    filtered_metadata = {}

    new_counter = 1

    for event_idx, metadata in events_metadata.items():

        stimulus_position = metadata["stimulus_position"]
        click_position = metadata["click_position"]


        # Only check events with vibration/click information
        if not np.isnan(click_position):

            click_delay = (
                click_position - stimulus_position
            ) / sfreq


            if click_delay > max_click_delay:

                removed_events.append({
                    "event_index": event_idx,
                    "event_type": metadata["event_type"],
                    "stimulus_position": stimulus_position,
                    "click_position": click_position,
                    "click_delay_s": click_delay
                })

                continue


        # Keep event
        keep_indices.append(event_idx)

        filtered_metadata[new_counter] = metadata
        new_counter += 1


    # --------------------------------------------------
    # Keep corresponding rows in MNE events array
    # --------------------------------------------------

    filtered_events = []

    for idx in keep_indices:

        # event index starts at 1, numpy index starts at 0
        filtered_events.append(events[idx - 1])


    filtered_events = np.array(filtered_events, dtype=int)


    removed_events_df = pd.DataFrame(removed_events)


    if verbose:

        print("--------------------------------")
        print("Vibration timing filter summary")
        print("--------------------------------")
        print(f"Events before filtering : {len(events)}")
        print(f"Events removed          : {len(removed_events_df)}")
        print(f"Events remaining        : {len(filtered_events)}")

        if len(removed_events_df):
            print("\nRemoved events:")
            print(removed_events_df)


    return filtered_events, filtered_metadata, removed_events_df