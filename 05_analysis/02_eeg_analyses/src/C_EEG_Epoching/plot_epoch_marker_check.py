import matplotlib.pyplot as plt
import numpy as np


def plot_epoch_marker_check(
    epochs,
    events_metadata,
    raw,
    task="Task1_PPS",
    picks=None,
    scale=1e6,
    alpha_trace=0.3
):
    """
    Overlay all epochs of each event type with original EEG markers.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE epochs object.

    events_metadata : dict
        Metadata generated during marker grouping.

    raw : mne.io.Raw
        Original raw object.

    task : str
        Defines marker structure.

        Available:
            - "controller"
                stimulus (1-24) followed by controller (40/41)

            - "vibration"
                stimulus followed by vibration markers

    picks : list | None
        Channels to plot.
        If None, first EEG channel is plotted.

    scale : float
        Conversion factor (V -> µV).

    alpha_trace : float
        Transparency of individual EEG traces.
    """

    sfreq = raw.info["sfreq"]

    if picks is None:
        picks = [0]


    # --------------------------------------------------
    # Metadata lookup for faster matching
    # --------------------------------------------------

    metadata_lookup = {
        m["stimulus_position"]: m
        for m in events_metadata.values()
    }


    # --------------------------------------------------
    # Task-specific marker configuration
    # --------------------------------------------------

    if task.lower() == "task2_hitormiss":

        marker_config = {
            "secondary_key": "controller_position",
            "label_key": "controller_type",
            "colors": {
                "left": "blue",
                "right": "green",
                "default": "purple"
            },
            "label": "Controller"
        }


    elif task.lower() == "task1_pps":

        marker_config = {
            "secondary_key": "vibration_position",
            "label_key": None,
            "colors": {
                "default": "blue"
            },
            "label": "Vibration ON"
        }


    else:
        raise ValueError(
            f"Unknown task '{task}'. "
            "Available: 'Task1_PPS', 'Task2_HitOrMiss'"
        )



    # --------------------------------------------------
    # Loop over event types
    # --------------------------------------------------

    for event_name in epochs.event_id.keys():

        event_epochs = epochs[event_name]


        plt.figure(figsize=(12, 4))


        stim_label = True
        secondary_label = True


        # --------------------------------------------------
        # Loop through trials
        # --------------------------------------------------

        for trial_idx in range(len(event_epochs)):

            epoch = event_epochs[trial_idx]

            epoch_sample = epoch.events[0, 0]


            marker_info = metadata_lookup.get(
                epoch_sample
            )


            if marker_info is None:
                continue



            # EEG signal

            data = epoch.get_data(
                picks=picks
            )[0] * scale


            times = epoch.times



            # Plot EEG traces

            for ch_idx, signal in enumerate(data):

                plt.plot(
                    times,
                    signal,
                    alpha=alpha_trace,
                    label=(
                        epochs.ch_names[picks[ch_idx]]
                        if trial_idx == 0
                        else None
                    )
                )



            # --------------------------------------------------
            # Stimulus marker
            # --------------------------------------------------

            stim_position = marker_info[
                "stimulus_position"
            ]


            plt.axvline(
                0,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="Stimulus"
                if stim_label else None
            )

            stim_label = False



            # --------------------------------------------------
            # Secondary marker
            # --------------------------------------------------

            secondary_position = marker_info.get(
                marker_config["secondary_key"],
                np.nan
            )


            if not np.isnan(secondary_position):

                marker_time = (
                    secondary_position - stim_position
                ) / sfreq



                # Determine color

                color = marker_config["colors"]["default"]

                label = marker_config["label"]


                if marker_config["label_key"]:

                    marker_type = marker_info.get(
                        marker_config["label_key"],
                        ""
                    ).lower()


                    if "left" in marker_type:
                        color = marker_config["colors"]["left"]
                        label = "Controller left"


                    elif "right" in marker_type:
                        color = marker_config["colors"]["right"]
                        label = "Controller right"



                plt.axvline(
                    marker_time,
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label=(
                        label
                        if secondary_label
                        else None
                    )
                )

                secondary_label = False



        # --------------------------------------------------
        # Figure formatting
        # --------------------------------------------------

        plt.title(
            f"Sanity check - {task}: {event_name} "
            f"(n={len(event_epochs)})"
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (µV)")

        plt.grid(True)

        plt.legend()

        plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# def plot_epoch_marker_check(
#     epochs,
#     events_metadata,
#     raw,
#     picks=None,
#     scale=1e6,
#     alpha_trace=0.3
# ):
#     """
#     Overlay all epochs of each event type with original EEG markers.

#     Parameters
#     ----------
#     epochs : mne.Epochs
#         MNE epochs object.

#     events_metadata : dict
#         Trial metadata generated by group_eeg_markers().

#     raw : mne.io.Raw
#         Original raw object (used for sampling frequency).

#     picks : list | None
#         Channels to plot.
#         If None, first EEG channel is plotted.

#     scale : float
#         Conversion factor from Volts to µV.

#     alpha_trace : float
#         Transparency of individual EEG traces.
#     """

#     sfreq = raw.info["sfreq"]

#     if picks is None:
#         picks = [0]


#     # --------------------------------------------------
#     # Loop through each event type
#     # --------------------------------------------------
#     for event_name in epochs.event_id.keys():

#         event_epochs = epochs[event_name]

#         plt.figure(figsize=(12, 4))


#         # Keep track of marker labels
#         stim_label = True
#         vib_label = True
#         click_label = True


#         # --------------------------------------------------
#         # Loop through all trials
#         # --------------------------------------------------
#         for trial_idx in range(len(event_epochs)):
#             # Extract single epoch object
#             epoch = event_epochs[trial_idx]
#             # Original EEG sample of this epoch
#             epoch_sample = epoch.events[0, 0]
            
#             # Find corresponding metadata
#             marker_info = None

#             for _, metadata in events_metadata.items():

#                 if metadata["stimulus_position"] == epoch_sample:
#                     marker_info = metadata
#                     break


#             if marker_info is None:
#                 continue


#             # EEG data
#             data = epoch.get_data(
#                 picks=picks
#             )[0] * scale

#             times = epoch.times


#             # Plot EEG traces
#             for ch_idx, signal in enumerate(data):

#                 plt.plot(
#                     times,
#                     signal,
#                     alpha=alpha_trace,
#                     label=(
#                         epochs.ch_names[picks[ch_idx]]
#                         if trial_idx == 0
#                         else None
#                     )
#                 )


#             # --------------------------------------------------
#             # Plot markers
#             # --------------------------------------------------

#             stim_position = marker_info["stimulus_position"]


#             # Stimulus
#             plt.axvline(
#                 0,
#                 color="red",
#                 linestyle="--",
#                 linewidth=1.5,
#                 alpha=0.7,
#                 label="Stimulus" if stim_label else None
#             )

#             stim_label = False


#             # Vibration ON
#             vibration_position = marker_info.get(
#                 "vibration_position",
#                 np.nan
#             )

#             if not np.isnan(vibration_position):

#                 vibration_time = (
#                     vibration_position - stim_position
#                 ) / sfreq


#                 plt.axvline(
#                     vibration_time,
#                     color="blue",
#                     linestyle="--",
#                     linewidth=1.5,
#                     alpha=0.7,
#                     label="Vibration ON" if vib_label else None
#                 )

#                 vib_label = False


#             # Vibration OFF
#             click_position = marker_info.get(
#                 "click_position",
#                 np.nan
#             )

#             if not np.isnan(click_position):

#                 click_time = (
#                     click_position - stim_position
#                 ) / sfreq


#                 plt.axvline(
#                     click_time,
#                     color="green",
#                     linestyle="--",
#                     linewidth=1.5,
#                     alpha=0.7,
#                     label="Vibration OFF" if click_label else None
#                 )

#                 click_label = False



#         plt.title(
#             f"Sanity check - all epochs: {event_name} "
#             f"(n={len(event_epochs)})"
#         )

#         plt.xlabel("Time (s)")
#         plt.ylabel("Amplitude (µV)")
#         plt.grid(True)
#         plt.legend()

#         plt.show()