import os

def save_tfr_results(
    file_path: str,
    file_name: str,
    tfr_single_dict: dict,
    tfr_average_dict: dict,
    verbose: bool = True
):
    """
    Save single-trial and averaged TFR results.

    Parameters
    ----------
    file_path : str
        Folder where TFR results will be saved.

    file_name : str
        Base filename.

        Example:
            "Pilote001_Task1_PPS"

    tfr_single_dict : dict
        Dictionary of single-trial EpochsTFR objects.
        Keys are event names.

    tfr_average_dict : dict
        Dictionary of averaged AverageTFR objects.
        Keys are event names.

    verbose : bool
        Print saving information.

    Returns
    -------
    save_paths : dict
        Dictionary containing saved file paths.
    """


    # --------------------------------------------------
    # Create folders
    # --------------------------------------------------

    single_path = os.path.join(
        file_path,
        f"{file_name}_tfr_single"
    )

    average_path = os.path.join(
        file_path,
        f"{file_name}_tfr_average"
    )


    os.makedirs(
        single_path,
        exist_ok=True
    )

    os.makedirs(
        average_path,
        exist_ok=True
    )


    save_paths = {
        "single": {},
        "average": {}
    }


    # --------------------------------------------------
    # Save single-trial TFRs
    # --------------------------------------------------

    if verbose:
        print("\nSaving single-trial TFRs...")


    for event_name, tfr in tfr_single_dict.items():

        # Replace characters unsafe for filenames
        # safe_name = event_name.replace("/", "_")

        filename = os.path.join(
            single_path,
            f"{event_name}_tfr.h5"
        )


        tfr.save(
            filename,
            overwrite=True
        )


        save_paths["single"][event_name] = filename


        if verbose:
            print(
                f"  Saved {event_name}"
            )


    # --------------------------------------------------
    # Save averaged TFRs
    # --------------------------------------------------

    if verbose:
        print("\nSaving averaged TFRs...")


    for event_name, tfr in tfr_average_dict.items():

        safe_name = event_name.replace("/", "_")

        filename = os.path.join(
            average_path,
            f"{safe_name}-tfr.h5"
        )


        tfr.save(
            filename,
            overwrite=True
        )


        save_paths["average"][event_name] = filename


        if verbose:
            print(
                f"  Saved {event_name}"
            )

    return save_paths