import pandas as pd
import numpy as np

def group_eeg_markers_task1(markers_df, trigger_map, verbose=True):
    """
    Groups EEG markers into experimental events and creates MNE events array.

    Returns
    -------
    events : np.ndarray
        MNE events array [sample, 0, event_id]

    event_id : dict
        Mapping between event names and integer IDs

    events_metadata : dict
        Additional information for each event:
            - event_type
            - stimulus_position
            - vibration_position
            - click_position
            - original markers

    removed_markers_df : pd.DataFrame
        Markers removed due to invalid vibration sequences
    """

    markers_df = markers_df.sort_values("position").reset_index(drop=True)

    events = []
    event_id = {}
    events_metadata = {}
    removed_markers = []

    next_event_id = 1
    event_counter = 0

    i = 0

    while i < len(markers_df):

        marker = markers_df.iloc[i]
        trigger = marker["trigger_value"]

        # Ignore markers not defined in trigger map
        if trigger not in trigger_map:
            i += 1
            continue

        event_type = trigger_map[trigger]

        # Create unique ID for each event type
        if event_type not in event_id:
            event_id[event_type] = next_event_id
            next_event_id += 1


        # --------------------------------------------------
        # VISUAL ONLY EVENT
        # --------------------------------------------------
        if isinstance(event_type, str) and "VisualOnly" in event_type:

            events.append([
                int(marker["position"]),
                0,
                event_id[event_type]
            ])

            event_counter += 1

            events_metadata[event_counter] = {
                "event_type": event_type,
                "stimulus_position": int(marker["position"]),
                "vibration_position": np.nan,
                "click_position": np.nan,
                "markers": [
                    marker.to_dict()
                ]
            }

            i += 1


        # --------------------------------------------------
        # VIBROTACTILE / BOTH EVENT
        # --------------------------------------------------
        elif any(x in event_type for x in ["VibrotactileOnly", "Both"]):

            if i + 2 < len(markers_df):

                m63 = markers_df.iloc[i + 1]
                m62 = markers_df.iloc[i + 2]

                valid_sequence = (
                    m63["trigger_value"] == 63 and
                    m62["trigger_value"] == 62
                )

                if valid_sequence:

                    events.append([
                        int(marker["position"]),
                        0,
                        event_id[event_type]
                    ])

                    event_counter += 1

                    events_metadata[event_counter] = {
                        "event_type": event_type,
                        "stimulus_position": int(marker["position"]),
                        "vibration_position": int(m63["position"]),
                        "click_position": int(m62["position"]),
                        "markers": [
                            marker.to_dict(),
                            m63.to_dict(),
                            m62.to_dict()
                        ]
                    }

                    i += 3

                else:
                    # Invalid vibration sequence:
                    # remove the stimulus marker
                    removed_markers.append(marker.to_dict())
                    i += 1

            else:
                removed_markers.append(marker.to_dict())
                i += 1


        else:
            i += 1


    events = np.array(events, dtype=int)
    removed_markers_df = pd.DataFrame(removed_markers)


    if verbose:

        print("--------------------------------")
        print("Marker grouping summary")
        print("--------------------------------")
        print(f"Total events created : {len(events)}")
        print(f"Unique event types  : {len(event_id)}")
        print(f"Markers removed     : {len(removed_markers_df)}")

        print("\nEvent IDs:")
        for name, code in event_id.items():
            print(f"  {code}: {name}")

        if len(removed_markers_df):
            print("\nRemoved markers:")
            print(
                removed_markers_df[
                    ["marker_num", "trigger_value", "position"]
                ]
            )


    return events, event_id, events_metadata, removed_markers_df

# def group_eeg_markers(markers_df, trigger_map, verbose=True):
#     """
#     Groups EEG markers into experimental events and creates MNE events array.

#     Returns
#     -------
#     events : np.ndarray
#         MNE events array [sample, 0, event_id]

#     event_id : dict
#         Mapping between event names and integer IDs

#     removed_markers_df : pd.DataFrame
#         Markers removed due to invalid vibration sequences
#     """

#     markers_df = markers_df.sort_values("position").reset_index(drop=True)

#     events = []
#     event_id = {}
#     removed_markers = []

#     next_event_id = 1
#     i = 0

#     while i < len(markers_df):

#         marker = markers_df.iloc[i]
#         trigger = marker["trigger_value"]

#         # Ignore markers not defined in trigger map
#         if trigger not in trigger_map:
#             i += 1
#             continue

#         event_type = trigger_map[trigger]

#         # Create unique ID for each event type
#         if event_type not in event_id:
#             event_id[event_type] = next_event_id
#             next_event_id += 1


#         # --------------------------------------------------
#         # VISUAL ONLY EVENT
#         # --------------------------------------------------
#         if isinstance(event_type, str) and "VisualOnly" in event_type:

#             events.append([
#                 int(marker["position"]),
#                 0,
#                 event_id[event_type]
#             ])

#             i += 1


#         # --------------------------------------------------
#         # VIBROTACTILE / BOTH EVENT
#         # --------------------------------------------------
#         elif any(x in event_type for x in ["VibrotactileOnly", "Both"]):

#             if i + 2 < len(markers_df):

#                 m63 = markers_df.iloc[i + 1]
#                 m62 = markers_df.iloc[i + 2]

#                 valid_sequence = (
#                     m63["trigger_value"] == 63 and
#                     m62["trigger_value"] == 62
#                 )

#                 if valid_sequence:

#                     # Epoch locked to stimulus marker
#                     events.append([
#                         int(marker["position"]),
#                         0,
#                         event_id[event_type]
#                     ])

#                     i += 3

#                 else:
#                     # Invalid vibration sequence:
#                     # remove the stimulus marker
#                     removed_markers.append(marker.to_dict())
#                     i += 1

#             else:
#                 i += 1


#         else:
#             i += 1


#     events = np.array(events, dtype=int)
#     removed_markers_df = pd.DataFrame(removed_markers)


#     if verbose:

#         print("--------------------------------")
#         print("Marker grouping summary")
#         print("--------------------------------")
#         print(f"Total events created : {len(events)}")
#         print(f"Unique event types  : {len(event_id)}")
#         print(f"Markers removed     : {len(removed_markers_df)}")

#         print("\nEvent IDs:")
#         for name, code in event_id.items():
#             print(f"  {code}: {name}")

#         if len(removed_markers_df):
#             print("\nRemoved markers:")
#             print(
#                 removed_markers_df[
#                     ["marker_num", "trigger_value", "position"]
#                 ]
#             )


#     return events, event_id, removed_markers_df
