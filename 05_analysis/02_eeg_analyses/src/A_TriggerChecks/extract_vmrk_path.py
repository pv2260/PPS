import os
import glob

def extract_vmrk_path(data_path, subjectID, task):
    """
    Extract the unique .vmrk file path from a BrainVision EEG folder.

    Parameters
    ----------
    data_path : str
        Main data directory.
    subjectID : str
        Subject identifier.
    task : str
        Task folder name (e.g., 'Task1_PPS').

    Returns
    -------
    vmrk_path : str
        Full path to the unique .vmrk file.

    Raises
    ------
    FileNotFoundError
        If the EEG folder or .vmrk file does not exist.
    RuntimeError
        If multiple .vmrk files are found.
    """

    # Build EEG folder path
    eeg_path = os.path.join(data_path, subjectID, task, 'Raw', 'EEG')

    # Check that the folder exists
    if not os.path.isdir(eeg_path):
        raise FileNotFoundError(f"EEG folder does not exist: {eeg_path}")

    # Find .vmrk files
    vmrk_files = glob.glob(os.path.join(eeg_path, '*.vmrk'))

    # Check number of files
    if len(vmrk_files) == 0:
        raise FileNotFoundError(f"No .vmrk file found in: {eeg_path}")

    elif len(vmrk_files) > 1:
        raise RuntimeError(
            f"Multiple .vmrk files found ({len(vmrk_files)}):\n" +
            "\n".join(vmrk_files)
        )

    # Return unique .vmrk path
    return vmrk_files[0]