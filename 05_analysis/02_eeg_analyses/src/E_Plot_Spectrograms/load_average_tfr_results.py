import os
import mne


def load_average_tfr_results(
    file_path,
    verbose=True
):
    """
    Load averaged MNE TFR spectrograms saved as .h5 files.

    Parameters
    ----------
    file_path : str
        Folder containing averaged TFR files.

    verbose : bool
        Print loading information.

    Returns
    -------
    tfr_average_dict : dict
        Dictionary:
            {
                event_name: AverageTFR
            }
    """

    tfr_average_dict = {}


    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Folder not found: {file_path}"
        )


    for filename in os.listdir(file_path):

        if filename.endswith("tfr.h5"):

            # Remove suffix to recover event name
            event_name = filename.replace(
                "_tfr.h5",
                ""
            )


            filepath = os.path.join(
                file_path,
                filename
            )


            if verbose:
                print(
                    f"Loading {event_name}"
                )


            # Load TFR
            tfr = mne.time_frequency.read_tfrs(
                filepath
            )


            # Compatibility with older MNE versions
            if isinstance(tfr, list):
                tfr = tfr[0]


            tfr_average_dict[event_name] = tfr


    if verbose:
        print("--------------------------------")
        print(
            f"Loaded {len(tfr_average_dict)} averaged TFRs"
        )


    return tfr_average_dict