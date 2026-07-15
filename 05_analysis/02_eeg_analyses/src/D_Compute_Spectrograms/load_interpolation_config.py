import json

def load_interpolation_config(
    file_path,
    verbose=True
):
    """
    Load interpolation configuration from JSON.

    Returns
    -------
    interpolation_config : dict
    """

    with open(file_path, "r") as f:
        interpolation_config = json.load(f)

    if verbose:
        print("[load_interpolation_config] Interpolation config loaded")
        print(f"  → {file_path}")

    return interpolation_config