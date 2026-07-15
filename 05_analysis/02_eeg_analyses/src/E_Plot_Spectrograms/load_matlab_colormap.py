import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os


def load_matlab_colormap(
    colormap_path,
    colormap_name,
    variable_name=None,
    register=True,
    plot_preview=False
):
    """
    Load a MATLAB (.mat) colormap and convert it to a Matplotlib colormap.

    Parameters
    ----------
    colormap_path : str
        Path to the .mat colormap file.

    colormap_name : str
        Name to give to the Matplotlib colormap.

    variable_name : str | None
        Name of the variable inside the .mat file.
        If None, automatically selects the first non-private variable.

    register : bool
        If True, register the colormap globally in Matplotlib.

    plot_preview : bool
        If True, display a preview of the colormap.

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        Matplotlib colormap object.
    """

    if not os.path.exists(colormap_path):
        raise FileNotFoundError(
            f"Colormap file not found: {colormap_path}"
        )

    # Load MATLAB file
    mat = scipy.io.loadmat(colormap_path)

    # Automatically detect variable name
    if variable_name is None:
        variables = [
            key for key in mat.keys()
            if not key.startswith("__")
        ]

        if len(variables) == 0:
            raise ValueError(
                "No valid colormap variable found in .mat file"
            )

        variable_name = variables[0]


    cmap_array = mat[variable_name]


    # Ensure RGB values are float in [0,1]
    if cmap_array.max() > 1:
        cmap_array = cmap_array / 255.0

    
    # Create matplotlib colormap
    cmap = LinearSegmentedColormap.from_list(
        colormap_name,
        cmap_array
    )


    # Register globally
    if register:
        try:
            plt.colormaps.register(cmap)
        except ValueError:
            # Already registered
            pass


    # Preview
    if plot_preview:

        plt.figure(figsize=(6, 1))

        plt.imshow(
            [range(len(cmap_array))],
            cmap=cmap,
            aspect="auto"
        )

        plt.axis("off")
        plt.title(colormap_name)
        plt.show()


    return cmap