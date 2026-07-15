import numpy as np
from scipy.interpolate import interp1d
import re


def _interpolate_segment(data, times, t_start, t_end, n_points, extrapolate=True):
    """
    Interpolate a single time segment of a trial to a fixed number of points.

    Parameters
    ----------
    data : ndarray (n_channels, n_times)
        Full trial data.
    times : ndarray (n_times,)
        Full trial time vector.
    t_start, t_end : float
        Start/end time (seconds) of the segment to extract.
    n_points : int
        Number of points to interpolate the segment to.
    extrapolate : bool
        Whether to allow extrapolation at segment edges (useful when
        t_start/t_end fall slightly outside the available sample times).

    Returns
    -------
    seg_interp : ndarray (n_channels, n_points)
    new_t : ndarray (n_points,)
    """

    idx = (times >= t_start) & (times <= t_end)

    seg = data[:, idx]
    t_seg = times[idx]

    new_t = np.linspace(t_start, t_end, n_points)

    seg_interp = np.zeros((data.shape[0], n_points))

    for ch in range(data.shape[0]):

        if extrapolate:
            f = interp1d(
                t_seg,
                seg[ch],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate"
            )
        else:
            f = interp1d(
                t_seg,
                seg[ch],
                kind="linear"
            )

        seg_interp[ch] = f(new_t)

    return seg_interp, new_t


def get_interpolation_key(event_name):
    """
    Extract interpolation configuration key from event name.

    Examples
    --------
    VibrotactileOnly_D4_Narrow_Slow
        -> D4_Narrow_Slow

    Both_D7_Narrow_Fast
        -> D7_Narrow_Fast
    """

    match = re.search(
        r"(D\d+_Narrow_(?:Fast|Slow))",
        event_name
    )

    if match:
        return match.group(1)

    elif "VisualOnly" in event_name:
        return event_name

    else:
        return None


def interpolate_epoch_groups(
    epochs,
    events_metadata,
    raw,
    interpolation_config
):
    """
    Time-normalize EEG epochs differently for each event type, preserving
    a time-normalized pre-stimulus baseline segment.

    For vibrotactile/both trials:
        Segment 0 (baseline):
            baseline_start -> stimulus onset (time 0)
            interpolated to N_baseline_points

        Segment 1:
            Stimulus onset -> vibration onset
            interpolated to N1_points

        Segment 2:
            Vibration onset -> N_sec after vibration onset
            interpolated to N2_points


    For visual-only trials:
        Segment 0 (baseline):
            baseline_start -> stimulus onset (time 0)
            interpolated to N_baseline_points

        Segment 1:
            Stimulus onset -> N_sec after onset
            interpolated to N_visual_points


    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data. Should include a pre-stimulus period (tmin < 0)
        if a baseline segment is requested via interpolation_config.

    events_metadata : dict
        Metadata from group_eeg_markers().

    raw : mne.io.Raw
        Raw EEG object (used for sampling frequency).

    interpolation_config : dict
        Configuration for each trial type.

        Example:

        {
            "D7_Narrow_Slow": {
                "N1_points": 3000,
                "N2_points": 3000,
                "N_sec_after_vibration": 2,
                "N_baseline_points": 500,       # optional
                "baseline_start": -0.2          # optional, seconds relative to stim onset
            },

            "VisualOnly_Narrow_Slow": {
                "N_visual_points": 4000,
                "N_sec_after_onset": 2,
                "N_baseline_points": 500,        # optional
                "baseline_start": -0.2           # optional
            }
        }

        Notes
        -----
        - If "N_baseline_points" is omitted (or None) for a given condition,
          no baseline segment is added for that condition (backward-compatible
          behavior identical to the original function).
        - If "baseline_start" is omitted, it defaults to the earliest available
          time in the epoch (epoch.times.min()), i.e. the full pre-stim window.


    Returns
    -------
    interpolated_epochs : dict
        {
            event_name:
                ndarray(n_trials, n_channels, n_points)
        }
        n_points = N_baseline_points (if any) + N1_points + N2_points
                   (vibrotactile/both)
        or
        n_points = N_baseline_points (if any) + N_visual_points
                   (visual only)

    interpolated_times : dict
        Time vector for each condition. Baseline times are negative
        (relative to stimulus onset at t=0), matching the original
        epoch's time convention.
    """

    sfreq = raw.info["sfreq"]

    interpolated_epochs = {}
    interpolated_times = {}


    # --------------------------------------------------
    # Loop through conditions
    # --------------------------------------------------

    for event_name in epochs.event_id.keys():

        # Find corresponding interpolation configuration
        interp_key = get_interpolation_key(event_name)

        if interp_key not in interpolation_config:
            print(
                f"No interpolation parameters for {event_name}, "
                f"searched key: {interp_key}"
            )
            continue

        cfg = interpolation_config[interp_key]
        condition_epochs = epochs[event_name]

        n_baseline_points = cfg.get("N_baseline_points", None)
        include_baseline = n_baseline_points is not None


        all_trials = []
        all_times = None


        # --------------------------------------------------
        # Loop through trials
        # --------------------------------------------------

        for trial_idx in range(len(condition_epochs)):

            epoch = condition_epochs[trial_idx]

            epoch_sample = epoch.events[0, 0]


            # Find metadata
            marker_info = None

            for _, metadata in events_metadata.items():

                if metadata["stimulus_position"] == epoch_sample:
                    marker_info = metadata
                    break


            if marker_info is None:
                continue


            data = epoch.get_data()[0]
            times = epoch.times


            # ==================================================
            # BASELINE SEGMENT (shared across trial types)
            # ==================================================

            baseline_interp = None
            new_t_baseline = None

            if include_baseline:

                baseline_start = cfg.get("baseline_start", times.min())

                # Guard against requesting a baseline window that isn't
                # actually available in this epoch's data
                if baseline_start < times.min():
                    baseline_start = times.min()

                baseline_interp, new_t_baseline = _interpolate_segment(
                    data,
                    times,
                    t_start=baseline_start,
                    t_end=0,
                    n_points=n_baseline_points,
                    extrapolate=True
                )


            # ==================================================
            # VIBROTACTILE / BOTH
            # ==================================================

            if not np.isnan(
                marker_info["vibration_position"]
            ):

                stim_time = 0

                vibration_time = (
                    marker_info["vibration_position"]
                    -
                    marker_info["stimulus_position"]
                ) / sfreq


                # -----------------------------
                # Segment 1:
                # stimulus -> vibration
                # -----------------------------

                seg1_interp, new_t1 = _interpolate_segment(
                    data,
                    times,
                    t_start=stim_time,
                    t_end=vibration_time,
                    n_points=cfg["N1_points"],
                    extrapolate=True
                )


                # -----------------------------
                # Segment 2:
                # vibration -> N seconds after
                # -----------------------------

                end_time = (
                    vibration_time
                    +
                    cfg["N_sec_after_vibration"]
                )

                seg2_interp, new_t2 = _interpolate_segment(
                    data,
                    times,
                    t_start=vibration_time,
                    t_end=end_time,
                    n_points=cfg["N2_points"],
                    extrapolate=True
                )


                # -----------------------------
                # Concatenate baseline + seg1 + seg2
                # -----------------------------

                segments = []
                time_segments = []

                if include_baseline:
                    segments.append(baseline_interp)
                    time_segments.append(new_t_baseline)

                segments.append(seg1_interp)
                time_segments.append(new_t1)

                segments.append(seg2_interp)
                time_segments.append(new_t2)

                trial_interp = np.concatenate(segments, axis=1)
                trial_time = np.concatenate(time_segments)


            # ==================================================
            # VISUAL ONLY
            # ==================================================

            else:

                end_time = cfg["N_sec_after_onset"]

                visual_interp, new_t_visual = _interpolate_segment(
                    data,
                    times,
                    t_start=0,
                    t_end=end_time,
                    n_points=cfg["N_visual_points"],
                    extrapolate=False
                )


                # -----------------------------
                # Concatenate baseline + visual segment
                # -----------------------------

                segments = []
                time_segments = []

                if include_baseline:
                    segments.append(baseline_interp)
                    time_segments.append(new_t_baseline)

                segments.append(visual_interp)
                time_segments.append(new_t_visual)

                trial_interp = np.concatenate(segments, axis=1)
                trial_time = np.concatenate(time_segments)



            all_trials.append(trial_interp)

            all_times = trial_time



        if len(all_trials):

            interpolated_epochs[event_name] = np.stack(
                all_trials,
                axis=0
            )

            interpolated_times[event_name] = all_times



    return interpolated_epochs, interpolated_times


# import numpy as np
# from scipy.interpolate import interp1d


# def interpolate_epoch_groups(
#     epochs,
#     events_metadata,
#     raw,
#     interpolation_config
# ):
#     """
#     Time-normalize EEG epochs differently for each event type.

#     For vibrotactile/both trials:
#         Segment 1:
#             Stimulus onset -> vibration onset
#             interpolated to N1_points

#         Segment 2:
#             Vibration onset -> N_sec after vibration onset
#             interpolated to N2_points


#     For visual-only trials:
#         Stimulus onset -> N_sec after onset
#         interpolated to N_visual_points


#     Parameters
#     ----------
#     epochs : mne.Epochs
#         Epoched EEG data.

#     events_metadata : dict
#         Metadata from group_eeg_markers().

#     raw : mne.io.Raw
#         Raw EEG object (used for sampling frequency).

#     interpolation_config : dict
#         Configuration for each trial type.

#         Example:

#         {
#             "VibrotactileOnly_D7_Narrow_Slow": {
#                 "N1_points": 3000,
#                 "N2_points": 3000,
#                 "N_sec_after_vibration": 2
#             },

#             "VisualOnly_Narrow_Slow": {
#                 "N_visual_points": 4000,
#                 "N_sec_after_onset": 2
#             }
#         }


#     Returns
#     -------
#     interpolated_epochs : dict
#         {
#             event_name:
#                 ndarray(n_trials, n_channels, n_points)
#         }

#     interpolated_times : dict
#         Time vector for each condition.
#     """

#     sfreq = raw.info["sfreq"]

#     interpolated_epochs = {}
#     interpolated_times = {}


#     # --------------------------------------------------
#     # Loop through conditions
#     # --------------------------------------------------

#     for event_name in epochs.event_id.keys():

#         # Find corresponding interpolation configuration
#         interp_key = get_interpolation_key(event_name)

#         if interp_key not in interpolation_config:
#             print(
#                 f"No interpolation parameters for {event_name}, "
#                 f"searched key: {interp_key}"
#             )
#             continue

#         cfg = interpolation_config[interp_key]
#         condition_epochs = epochs[event_name]


#         all_trials = []
#         all_times = None


#         # --------------------------------------------------
#         # Loop through trials
#         # --------------------------------------------------

#         for trial_idx in range(len(condition_epochs)):

#             epoch = condition_epochs[trial_idx]

#             epoch_sample = epoch.events[0, 0]


#             # Find metadata
#             marker_info = None

#             for _, metadata in events_metadata.items():

#                 if metadata["stimulus_position"] == epoch_sample:
#                     marker_info = metadata
#                     break


#             if marker_info is None:
#                 continue


#             data = epoch.get_data()[0]


#             times = epoch.times


#             # ==================================================
#             # VIBROTACTILE / BOTH
#             # ==================================================

#             if not np.isnan(
#                 marker_info["vibration_position"]
#             ):

#                 stim_time = 0

#                 vibration_time = (
#                     marker_info["vibration_position"]
#                     -
#                     marker_info["stimulus_position"]
#                 ) / sfreq


#                 # -----------------------------
#                 # Segment 1:
#                 # stimulus -> vibration
#                 # -----------------------------

#                 idx1 = (
#                     (times >= stim_time) &
#                     (times <= vibration_time)
#                 )

#                 seg1 = data[:, idx1]
#                 t1 = times[idx1]


#                 new_t1 = np.linspace(
#                     0,
#                     vibration_time,
#                     cfg["N1_points"]
#                 )


#                 seg1_interp = np.zeros(
#                     (
#                         data.shape[0],
#                         cfg["N1_points"]
#                     )
#                 )


#                 for ch in range(data.shape[0]):

#                     f = interp1d(
#                         t1,
#                         seg1[ch],
#                         kind="linear",
#                         bounds_error=False,
#                         fill_value="extrapolate"
#                     )

#                     seg1_interp[ch] = f(new_t1)



#                 # -----------------------------
#                 # Segment 2:
#                 # vibration -> N seconds after
#                 # -----------------------------

#                 end_time = (
#                     vibration_time
#                     +
#                     cfg["N_sec_after_vibration"]
#                 )


#                 idx2 = (
#                     (times >= vibration_time) &
#                     (times <= end_time)
#                 )


#                 seg2 = data[:, idx2]
#                 t2 = times[idx2]


#                 new_t2 = np.linspace(
#                     vibration_time,
#                     end_time,
#                     cfg["N2_points"]
#                 )


#                 seg2_interp = np.zeros(
#                     (
#                         data.shape[0],
#                         cfg["N2_points"]
#                     )
#                 )


#                 for ch in range(data.shape[0]):

#                     f = interp1d(
#                         t2,
#                         seg2[ch],
#                         kind="linear",
#                         bounds_error=False,
#                         fill_value="extrapolate"
#                     )

#                     seg2_interp[ch] = f(new_t2)


#                 trial_interp = np.concatenate(
#                     [
#                         seg1_interp,
#                         seg2_interp
#                     ],
#                     axis=1
#                 )


#                 trial_time = np.concatenate(
#                     [
#                         new_t1,
#                         new_t2
#                     ]
#                 )


#             # ==================================================
#             # VISUAL ONLY
#             # ==================================================

#             else:

#                 end_time = cfg["N_sec_after_onset"]


#                 idx = (
#                     (times >= 0) &
#                     (times <= end_time)
#                 )


#                 segment = data[:, idx]
#                 segment_times = times[idx]


#                 new_time = np.linspace(
#                     0,
#                     end_time,
#                     cfg["N_visual_points"]
#                 )


#                 trial_interp = np.zeros(
#                     (
#                         data.shape[0],
#                         cfg["N_visual_points"]
#                     )
#                 )


#                 for ch in range(data.shape[0]):

#                     f = interp1d(
#                         segment_times,
#                         segment[ch],
#                         kind="linear"
#                     )

#                     trial_interp[ch] = f(new_time)


#                 trial_time = new_time



#             all_trials.append(trial_interp)

#             all_times = trial_time



#         if len(all_trials):

#             interpolated_epochs[event_name] = np.stack(
#                 all_trials,
#                 axis=0
#             )

#             interpolated_times[event_name] = all_times



#     return interpolated_epochs, interpolated_times


# import re


# def get_interpolation_key(event_name):
#     """
#     Extract interpolation configuration key from event name.

#     Examples
#     --------
#     VibrotactileOnly_D4_Narrow_Slow
#         -> D4_Narrow_Slow

#     Both_D7_Narrow_Fast
#         -> D7_Narrow_Fast
#     """

#     match = re.search(
#         r"(D\d+_Narrow_(?:Fast|Slow))",
#         event_name
#     )

#     if match:
#         return match.group(1)

#     elif "VisualOnly" in event_name:
#         return event_name

#     else:
#         return None