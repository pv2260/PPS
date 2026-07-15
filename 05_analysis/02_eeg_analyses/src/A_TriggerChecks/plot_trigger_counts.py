import matplotlib.pyplot as plt

def plot_trigger_counts(vmrk_df, trigger_map, save_path=None, log_scale=False,
                        title="Trigger occurrence count (EEG recording)"):
    """
    Horizontal bar plot of trigger occurrence counts from a .vmrk file.

    Triggers are sorted by occurrence count (most frequent on top).
    Labels are taken from the trigger codebook.

    Parameters
    ----------
    vmrk_df : pandas.DataFrame
        DataFrame containing a 'trigger_value' column.
    trigger_map : dict
        Dictionary mapping trigger codes to names.
    save_path : str, optional
        Path to save the figure.
    log_scale : bool
        Use logarithmic x-axis.
    title : str
        Plot title.

    Returns
    -------
    fig, counts
        Matplotlib figure and sorted trigger counts.
    """

    # Count trigger occurrences
    counts = (
        vmrk_df.dropna(subset=["trigger_value"])
        .groupby("trigger_value")
        .size()
        .sort_values(ascending=False)
    )

    # Extract codes and labels
    codes = counts.index.astype(int).tolist()
    labels = [trigger_map.get(c, f"Unknown({c})") for c in codes]
    values = counts.values

    # Combine labels with codes
    y_labels = labels

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, max(4, len(codes) * 0.4)))

    bars = ax.barh(
        y=range(len(codes)),
        width=values,
        color="#B04C9C"
    )

    # Put highest count at the top
    ax.invert_yaxis()

    # Labels
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel("Occurrences")
    ax.set_ylabel("Trigger")
    ax.set_title(title)

    # Add values at the end of bars
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_width() + max(values)*0.01,
            bar.get_y() + bar.get_height()/2,
            str(int(v)),
            va="center",
            fontsize=8
        )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, counts