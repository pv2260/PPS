import pandas as pd
import numpy as np


def group_eeg_markers_task2(markers_df, trigger_map, verbose=True):
    """
    Groups EEG markers into controller trials and labels correctness.

    Sequence:
        stimulus (1-24) --> controller (40 or 41)

    Event labels:
        <condition>_Correct
        <condition>_Incorrect

    Correctness rules:
        Left controller:
            Correct for Hit conditions

        Right controller:
            Correct for Miss conditions

    Returns
    -------
    events : np.ndarray
        MNE events array [sample, 0, event_id]

    event_id : dict
        Event name -> integer ID

    events_metadata : dict
        Trial metadata

    removed_markers_df : pd.DataFrame
        Invalid trials
    """


    markers_df = (
        markers_df
        .sort_values("position")
        .reset_index(drop=True)
    )


    events = []
    event_id = {}
    events_metadata = {}

    removed_markers = []

    next_event_id = 1
    event_counter = 0


    # --------------------------------------------------
    # Define correctness rules
    # --------------------------------------------------

    correct_controller = {

        # Left = correct
        "Slow_Near_Hit": "controller_left",
        "Slow_Clear_Hit": "controller_left",
        "Fast_Near_Hit": "controller_left",
        "Fast_Clear_Hit": "controller_left",


        # Right = correct
        "Slow_Clear_Miss": "controller_right",
        "Slow_Near_Miss": "controller_right",
        "Fast_Clear_Miss": "controller_right",
        "Fast_Near_Miss": "controller_right",
    }



    i = 0


    while i < len(markers_df):

        marker = markers_df.iloc[i]

        trigger = marker["trigger_value"]


        # Ignore unknown triggers
        if trigger not in trigger_map:
            i += 1
            continue



        # --------------------------------------------------
        # Stimulus markers
        # --------------------------------------------------

        if 1 <= trigger <= 24:

            condition = trigger_map[trigger]


            if i + 1 < len(markers_df):

                controller_marker = markers_df.iloc[i + 1]

                controller_trigger = (
                    controller_marker["trigger_value"]
                )


                # Must be followed by controller
                if controller_trigger in [40, 41]:


                    controller_type = trigger_map[
                        controller_trigger
                    ]


                    # --------------------------------------
                    # Determine correct / incorrect
                    # --------------------------------------

                    if condition in correct_controller:

                        if (
                            controller_type
                            ==
                            correct_controller[condition]
                        ):
                            accuracy = "Correct"

                        else:
                            accuracy = "Incorrect"


                        event_name = (
                            f"{condition}_{accuracy}"
                        )


                    else:
                        # fallback if condition not specified
                        event_name = condition



                    # --------------------------------------
                    # Create event ID
                    # --------------------------------------

                    if event_name not in event_id:

                        event_id[event_name] = (
                            next_event_id
                        )

                        next_event_id += 1



                    # --------------------------------------
                    # Add event
                    # --------------------------------------

                    events.append(
                        [
                            int(marker["position"]),
                            0,
                            event_id[event_name]
                        ]
                    )


                    event_counter += 1


                    events_metadata[event_counter] = {

                        "event_type": event_name,

                        "condition": condition,

                        "accuracy": (
                            accuracy
                            if condition in correct_controller
                            else None
                        ),

                        "stimulus_trigger": trigger,

                        "stimulus_position":
                            int(marker["position"]),

                        "controller_trigger":
                            controller_trigger,

                        "controller_type":
                            controller_type,

                        "controller_position":
                            int(
                                controller_marker["position"]
                            ),

                        "markers": [
                            marker.to_dict(),
                            controller_marker.to_dict()
                        ]
                    }



                    # Skip stimulus + controller
                    i += 2



                else:

                    # Invalid sequence
                    removed_markers.append(
                        marker.to_dict()
                    )

                    i += 1



            else:

                removed_markers.append(
                    marker.to_dict()
                )

                i += 1



        else:

            i += 1



    events = np.array(
        events,
        dtype=int
    )


    removed_markers_df = pd.DataFrame(
        removed_markers
    )



    if verbose:

        print("--------------------------------")
        print("Marker grouping summary")
        print("--------------------------------")

        print(
            f"Total events created : {len(events)}"
        )

        print(
            f"Unique event types  : {len(event_id)}"
        )

        print(
            f"Removed trials      : {len(removed_markers_df)}"
        )


        print("\nEvent IDs:")

        for name, code in event_id.items():

            print(
                f"  {code}: {name}"
            )


        if len(removed_markers_df):

            print(
                "\nRemoved stimulus markers:"
            )

            print(
                removed_markers_df[
                    [
                        "marker_num",
                        "trigger_value",
                        "position"
                    ]
                ]
            )


    return (
        events,
        event_id,
        events_metadata,
        removed_markers_df
    )