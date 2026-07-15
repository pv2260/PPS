import os
import re
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot_average_tfr(
    tfr_average_dict,
    event_name,
    interpolation_config,
    n_baseline_pts,
    t_min=None,
    t_max=None,
    f_min=None,
    f_max=None,
    colormin=-3,
    colormax=3,
    scale_db=10,
    cmap="RdBu_r",
    colorbar_label="Power (dB)",
    output_path=None,
    groups=None
):
    """
    Plot averaged TFR spectrograms channel by channel.

    Parameters
    ----------
    tfr_average_dict : dict
        Dictionary containing AverageTFR objects.

    event_name : str
        Trajectory/event name.

    interpolation_config : dict
        Interpolation parameters.

    cmap : str or matplotlib.colors.Colormap, optional
        Colormap used for the spectrogram (default "RdBu_r").

    colorbar_label : str, optional
        Label displayed next to the colorbar.

    output_path : str
        Root folder where figures are saved.

        Saved as:
        output_path/
                event_name/
                    channel.png
    """


    if event_name not in tfr_average_dict:
        raise ValueError(
            f"{event_name} not found"
        )


    power = tfr_average_dict[event_name].copy()


    # -----------------------------
    # Crop
    # -----------------------------
    power.crop(
        tmin=t_min,
        tmax=t_max,
        fmin=f_min,
        fmax=f_max
    )


    ch_names = power.info["ch_names"]
    freqs = power.freqs
    times = power.times


    # number of epochs averaged
    n_epochs = power.nave


    print(
        f"{event_name}: n={n_epochs} epochs"
    )


    data2plot = power.data * scale_db


    norm = TwoSlopeNorm(
        vmin=colormin,
        vcenter=0,
        vmax=colormax
    )


    # -----------------------------
    # Determine vibration onset
    # -----------------------------

    interp_key = get_interpolation_key(
        event_name, groups
    )


    vibration_time = None


    if interp_key in interpolation_config:

        cfg = interpolation_config[interp_key]


        if cfg["type"] == "vibrotactile":

            N1 = cfg["N1_points"]
            N2 = cfg["N2_points"]

            # index where vibration starts
            vib_idx = N1 + n_baseline_pts

            if vib_idx < len(times):

                vibration_time = times[vib_idx]



    # -----------------------------
    # Saving folder
    # -----------------------------

    if output_path is not None:

        clean_event_name = event_name.replace("-tfr.h5", "")
        save_folder = os.path.join(
            output_path,
            clean_event_name
        )

        os.makedirs(
            save_folder,
            exist_ok=True
        )

    else:
        save_folder = None



    # -----------------------------
    # Plot each channel
    # -----------------------------

    for i, ch_name in enumerate(ch_names):

        spectrogram = data2plot[i]


        fig, ax = plt.subplots(
            figsize=(6,4)
        )


        im = ax.imshow(
            spectrogram,
            aspect="auto",
            origin="lower",
            extent=[
                times[0],
                times[-1],
                freqs[0],
                freqs[-1]
            ],
            cmap=cmap,
            norm=norm
        )


        # -----------------------------
        # Colorbar
        # -----------------------------

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label, fontsize=13)
        cbar.ax.tick_params(labelsize=11)

        clean_event_name = event_name.replace("-tfr.h5", "")
        fig.suptitle(
            f"{clean_event_name}\n"
            f"{ch_name} (n={n_epochs})",
            fontsize=18
        )


        # Frequency references
        ax.axhline(
            4,
            color="k",
            linestyle="--",
            linewidth=2
        )

        ax.axhline(
            8,
            color="k",
            linestyle="--",
            linewidth=2
        )

        ax.axhline(
            13,
            color="k",
            linestyle="--",
            linewidth=2
        )


        # Stimulus onset
        ax.axvline(
            0,
            color="k",
            linestyle="--",
            linewidth=2,
            label="Stimulus"
        )


        # Vibration onset
        if vibration_time is not None:

            ax.axvline(
                vibration_time,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Vibration onset"
            )


        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.tick_params(
            axis="both",
            labelsize=15
        )


        plt.tight_layout()



        if save_folder:
            clean_event_name = event_name.replace("-tfr.h5", "")
            filename = (
                f"{ch_name}_{clean_event_name}_TFR.png"
                .replace("/", "_")
            )


            plt.savefig(
                os.path.join(
                    save_folder,
                    filename
                ),
                dpi=300,
                bbox_inches="tight"
            )


        plt.show()



def get_interpolation_key(event_name, groups=None):
    """
    Extract interpolation key from original or grouped event names.

    Parameters
    ----------
    event_name : str
        Condition or grouped condition name.

    groups : dict | None
        Dictionary defining grouped conditions.
        Example:
        {
            "Both_Near_Fast": [
                "Both_D1_Narrow_Fast",
                "Both_D2_Narrow_Fast"
            ]
        }

    Returns
    -------
    str | None
        Key matching interpolation_config.
    """

    # --------------------------------------------------
    # If grouped name, replace by first original condition
    # --------------------------------------------------
    if groups is not None:

        if event_name in groups:

            # Take first condition as representative
            # event_name = groups[event_name][0]
            return event_name


    # --------------------------------------------------
    # Extract D and speed
    # --------------------------------------------------
    match = re.search(
        r"(D\d+_Narrow_(?:Fast|Slow))",
        event_name
    )

    if match:
        return match.group(1)


    # --------------------------------------------------
    # Visual conditions
    # --------------------------------------------------
    if "VisualOnly" in event_name:
        return event_name


    return None


# def get_interpolation_key(event_name):
#     """
#     Extract D and speed key from event name.

#     Example:
#         Both_D7_Narrow_Slow
#         -> D7_Narrow_Slow
#     """

#     match = re.search(
#         r"(D\d+_Narrow_(?:Fast|Slow))",
#         event_name
#     )

#     if match:
#         return match.group(1)

#     elif "VisualOnly" in event_name:
#         return event_name

#     return None