import os
import json
import mne


def save_eeg_cleaning_results(
    file_path: str,
    file_name: str,
    raw_clean: mne.io.Raw,
    ica: mne.preprocessing.ICA,
    verbose: bool = True,
):
    """
    Save cleaned EEG data and ICA cleaning information.

    Parameters
    ----------
    file_path : str
        Folder where results will be saved.

    file_name : str
        Base name of the recording (without extension).

        Example:
            "sub01_task1"

    raw_clean : mne.io.Raw
        Cleaned EEG recording after ICA correction.

    ica : mne.preprocessing.ICA
        Fitted ICA object.

    excluded : list[int]
        ICA components removed during cleaning.

    verbose : bool
        Print saving information.

    Returns
    -------
    save_paths : dict
        Paths of saved files.
    """

    # --- 1. Create output folder -------------------------------------------
    if not os.path.exists(file_path):
        os.makedirs(file_path)

        if verbose:
            print(
                f"[save_eeg_cleaning_results] Created folder: {file_path}"
            )


    # --- 2. Define filenames -----------------------------------------------
    raw_filename = os.path.join(file_path,f"{file_name}_clean_raw.fif")
    ica_filename = os.path.join(file_path,f"{file_name}_ica.fif")
    excluded_filename = os.path.join(file_path,f"{file_name}_ica_excluded.json")

    # --- 3. Save cleaned EEG ------------------------------------------------
    if verbose:
        print("[save_eeg_cleaning_results] Saving cleaned EEG...")
    raw_clean.save(raw_filename,overwrite=True,verbose=False)

    # --- 4. Save ICA object -------------------------------------------------
    if verbose:print("[save_eeg_cleaning_results] Saving ICA...")
    ica.save(ica_filename,overwrite=True)

    if verbose:
        print("[save_eeg_cleaning_results] Done.")
        print(f"    Raw: {raw_filename}")
        print(f"    ICA: {ica_filename}")
        print(f"    Excluded: {excluded_filename}")