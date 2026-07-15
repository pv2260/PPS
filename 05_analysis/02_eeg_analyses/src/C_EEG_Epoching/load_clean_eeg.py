import os
import mne

def load_clean_eeg(data_path, subjectID, date, task, verbose=True):
    """
    Load cleaned EEG data from a .fif file and extracts basic metadata associated with the EEG reccording.

    Args:
        - subjectID: subjectID ID (str)
        - date: Date of the session (format: YYYY.MM.DD) (str)
        - task: Experimental task (e.g., 'Sham', 'Stim', 'NoStim') (str)
        - verbose: If True, print loading information (bool)

    Returns:
        - raw_cleaned: MNE Raw object containing the cleaned EEG data (mne.io.Raw)
        - sfreq: Sampling frequency in Hz (float)
        - descriptions: List of event/marker descriptions from mne object annotations (list)
    """
    # Build the full path to the cleaned EEG file
    eeg_path = os.path.join(data_path, subjectID, task, 'Output', 'clean_EEG', f'{subjectID}_{date}_{task}_clean_raw.fif')

    if verbose:
        print("[load_clean_eeg] Loading cleaned EEG file:")
        print(f"  → {eeg_path}")

    # Load the cleaned EEG data
    raw_cleaned = mne.io.read_raw_fif(eeg_path, preload=True, verbose=False)

    # Extract the sampling frequency
    sfreq = raw_cleaned.info['sfreq']
    # Extract annotation descriptions (event markers)
    descriptions = raw_cleaned.annotations.description

    return raw_cleaned, sfreq, descriptions