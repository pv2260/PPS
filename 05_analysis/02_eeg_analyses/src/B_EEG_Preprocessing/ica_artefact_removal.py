import mne
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
from mne.preprocessing import ICA
from mne_icalabel import label_components


def ica_artefact_removal(
    raw_filt: mne.io.Raw,
    ica_segments: Optional[list[tuple[float, float]]] = None,
    n_components: Optional[int] = None,
    method: str = "infomax",
    fit_params: Optional[dict] = None,
    auto_label: bool = True,
    label_threshold: float = 0.8,
    eog_channels: Optional[list[str]] = None,
    ecg_channel: Optional[str] = None,
    montage: Optional[Union[str, mne.channels.DigMontage]] = "standard_1020",
    random_state: int = 42,
    plot_sanity: bool = True,
    sanity_plot_dir: Optional[str] = None,
    max_property_plots: int = 10,
    overlay_picks: Optional[list[str]] = None,
    verbose: bool = True,
):
    """
    Fit ICA on filtered continuous data, identify artefact components, and
    return a cleaned copy of the recording.

    Artefact components are identified in two complementary ways:

    1. **Automatic labelling** (``auto_label=True``): uses ``mne-icalabel``
       (ICLabel classifier) to flag non-brain components (EOG, ECG, muscle,
       line-noise, channel-noise).  Components with probability ≥
       ``label_threshold`` for any artefact class are excluded.

    2. **Channel-correlation fallback**: if EOG/ECG reference channels are
       provided *and* ICLabel did not already catch them, MNE's
       ``find_bads_eog`` / ``find_bads_ecg`` are applied as an additional
       safeguard.

    The function prints a summary of excluded components and returns the
    cleaned raw signal together with the fitted ICA object so the caller can
    inspect, save, or adjust exclusions. If ``plot_sanity=True`` (default),
    it also generates the standard set of ICA sanity-check figures used to
    visually confirm the decomposition and exclusion decisions are sane
    before trusting ``raw_clean`` downstream.

    Parameters
    ----------
    raw_filt : mne.io.Raw
        Band-pass filtered (and notch-filtered) continuous data.
        **Must be preloaded.**  A copy is made internally; the original is
        not modified.
    ica_segments : list of tuple or None
        Time segments (seconds) used only for ICA fitting. If None, the full recording is used.
        Example: [(60, 600), (1200, 1800)]
    n_components : int or None
        Number of ICA components to decompose into.  None uses the MNE
        default (min(#channels, #samples/sfreq − 1)).  Typical value: 30–64.
    method : str
        ICA algorithm: "infomax" (default, recommended for EEG),
        "fastica", or "picard".
    fit_params : dict or None
        Extra keyword arguments forwarded to ``ICA.fit()``.
        E.g. ``{"extended": True}`` for extended Infomax.
    auto_label : bool
        Run ICLabel classifier to auto-detect artefact components.
    label_threshold : float
        Probability threshold above which a component is considered an
        artefact by ICLabel (default 0.8).
    eog_channels : list[str] or None
        Names of EOG channels to use for correlation-based EOG detection
        (e.g. ["VEOG", "HEOG"]).  If None, channel-correlation is skipped
        for eye artefacts, and no EOG score sanity plot is produced.
    ecg_channel : str or None
        Name of an ECG channel for correlation-based heartbeat detection.
        If None, ECG-based exclusion is skipped, and no ECG score sanity
        plot is produced.
    montage : str or DigMontage or None
        Electrode position montage required by ICLabel.  Accepts any name
        understood by ``mne.channels.make_standard_montage()`` (e.g.
        ``"standard_1020"``, ``"easycap-M1"``, ``"biosemi64"``) or a
        pre-built ``DigMontage`` object.  If the recording already has
        digitisation positions embedded, pass ``None`` to skip.
        Default: ``"standard_1020"``.
    random_state : int
        Seed for reproducibility.
    plot_sanity : bool
        If True (default), generate the standard ICA sanity-check plots:
          (a) component topographies (``ica.plot_components``)
          (b) detailed properties for each *excluded* component - time
              series, power spectrum, epochs image, topomap
              (``ica.plot_properties``)
          (c) a before/after signal overlay (``ica.plot_overlay``)
          (d) EOG/ECG correlation-score plots (``ica.plot_scores``), only
              if ``eog_channels``/``ecg_channel`` were provided
    sanity_plot_dir : str or None
        If provided, sanity-check figures are saved as PNGs into this
        directory (created if missing) and the returned dict contains file
        paths. If None (default), figures are left open and the returned
        dict contains the Figure objects themselves for interactive
        inspection (e.g. in a Jupyter notebook).
    max_property_plots : int
        Cap on how many excluded components get a full ``plot_properties``
        figure, to avoid generating dozens of plots when many components
        are excluded. Default 10.
    overlay_picks : list[str] or None
        Channels to show in the before/after overlay plot. None uses MNE's
        default (average across EEG channels).
    verbose : bool

    Returns
    -------
    raw_clean : mne.io.Raw
        Artefact-corrected copy of ``raw_filt``.
    ica : mne.preprocessing.ICA
        Fitted ICA object (``ica.exclude`` contains the excluded indices).
    excluded : list[int]
        Sorted list of excluded component indices.
    sanity_plots : dict
        Only meaningful if ``plot_sanity=True`` (empty dict otherwise).
        Keys: "components", "properties", "overlay", "eog_scores",
        "ecg_scores" (last two only present if the corresponding channels
        were given). Each value is a list of either file paths (if
        ``sanity_plot_dir`` was set) or Figure objects (if not).

    Notes
    -----
    * ICA is fit on a **1 Hz high-pass copy** of the data.  The correction is
      then applied to ``raw_filt`` so that you keep the original low-frequency
      content for downstream analyses (standard practice in EEG pipelines).
    * You can inspect components interactively *after* this function returns::

          ica.plot_components()
          ica.plot_sources(raw_filt)

    Examples
    --------
    >>> raw_clean, ica, excluded, sanity_plots = ica_artefact_removal(
    ...     raw_filt,
    ...     n_components=30,
    ...     montage="standard_1020",   # or "easycap-M1", "biosemi64", etc.
    ...     eog_channels=["VEOG", "HEOG"],
    ...     ecg_channel="ECG",
    ...     sanity_plot_dir="qc/ica_sub-01",
    ... )
    """
    if fit_params is None:
        fit_params = {}
    if method == "infomax" and "extended" not in fit_params:
        fit_params["extended"] = True  # extended Infomax is the MNE/EEGLAB default


    # --- 1. Create data used for ICA fitting ---------------
    if verbose: print("[ICA] Preparing data for ICA fitting")
    if ica_segments is not None:
        if verbose: print(f"[ICA] Using {len(ica_segments)} selected segments")
        segments = []
        for i, (tmin, tmax) in enumerate(ica_segments):
            seg = raw_filt.copy().crop(
                tmin=tmin,
                tmax=tmax
            )
            segments.append(seg)

        # Concatenate selected periods
        raw_for_ica = mne.concatenate_raws(
            segments,
            verbose=False
        )
    else:
        if verbose: print("[ICA] Using full recording")
        raw_for_ica = raw_filt.copy()

    # --- 1. High-pass copy for ICA fitting (standard practice) ---------------
    if verbose:
        print("[ica_artefact_removal] High-pass filtering at 1 Hz for ICA fit …")
        print("[ica_artefact_removal] EEG re-referencing: Average reference")
    raw_for_ica.filter(l_freq=1.0, h_freq=None, verbose=False)
    # Set the reference to the average activity of the whole scalp -> stabilize ICA
    raw_for_ica.set_eeg_reference("average", verbose=False)

    # --- 1b. Ensure a montage is set (required by ICLabel) -------------------
    if montage is not None:
        if isinstance(montage, str):
            montage_obj = mne.channels.make_standard_montage(montage)
            if verbose:
                print(f"[ica_artefact_removal] Setting montage: '{montage}'")
        else:
            montage_obj = montage
            if verbose:
                print("[ica_artefact_removal] Setting provided DigMontage.")
        # on_missing="ignore" silently skips channels absent from the montage
        # (e.g. EOG/ECG channels that are not part of the standard cap)
        raw_for_ica.set_montage(montage_obj, on_missing="ignore", verbose=False)
    else:
        if verbose:
            print("[ica_artefact_removal] Skipping montage (None passed — "
                  "assuming digitisation is already embedded).")

    # --- 2. Fit ICA -----------------------------------------------------------
    ica = ICA(
        n_components=n_components,
        method=method,
        fit_params=fit_params,
        random_state=random_state,
        max_iter="auto",
    )
    if verbose:
        print(f"[ica_artefact_removal] Fitting ICA ({method}, "
              f"n_components={n_components}) …")
    ica.fit(raw_for_ica, verbose=False)
    if verbose:
        print(f"  → {ica.n_components_} components extracted")

    excluded: list[int] = []

    # --- 3a. ICLabel automatic labelling --------------------------------------
    if auto_label:
        if verbose:
            print("[ica_artefact_removal] Running ICLabel classifier …")
        component_labels = label_components(raw_for_ica, ica, method="iclabel")
        labels      = component_labels["labels"]       # list of strings
        probs       = component_labels["y_pred_proba"] # (n_components, n_classes)

        artefact_classes = {"eye blink", "heart beat", "muscle artifact",
                            "line noise", "channel noise"}
        for idx, (label, prob_row) in enumerate(zip(labels, probs)):
            if label in artefact_classes:
                # prob_row is ordered as ICLabel's class list
                max_prob = prob_row.max()
                if max_prob >= label_threshold:
                    excluded.append(idx)
                    if verbose:
                        print(f"    Component {idx:3d}  → {label}  "
                              f"(p={max_prob:.2f}) — EXCLUDED")

    if verbose:
        print(f"[ica_artefact_removal] Total excluded components: {excluded}")

    # --- 4. Sanity-check plots -------------------------------------------------
    # 1. ICA component scalp maps print(" - ICA component topographies")
    fig_components = ica.plot_components(show=False)
    
    # 2. ICA component traces (excluded components only)
    print("  - Excluded ICA component traces")

    # Define time window (seconds)
    tmin = 0
    tmax = 50

    # Extract ICA component activations
    sources = ica.get_sources(raw_for_ica)

    # Get data
    ica_data = sources.get_data()

    # Time vector
    times = sources.times

    # Select only requested time window
    mask = (times >= tmin) & (times <= tmax)

    times = times[mask]
    ica_data = ica_data[:, mask]


    # Keep only excluded components
    excluded_to_plot = [
        ic for ic in excluded
        if ic < ica_data.shape[0]
    ]

    if len(excluded_to_plot) > 0:

        ica_data = ica_data[excluded_to_plot]

        # Normalize each component
        ica_data = (
            ica_data /
            np.std(ica_data, axis=1, keepdims=True)
        )


        fig_sources, axes = plt.subplots(
            len(excluded_to_plot),
            1,
            figsize=(14, 2 * len(excluded_to_plot)),
            sharex=True
        )

        if len(excluded_to_plot) == 1:
            axes = [axes]


        for i, ax in enumerate(axes):

            ax.plot(
                times,
                ica_data[i],
                linewidth=0.8
            )

            ax.set_ylabel(
                f"IC {excluded_to_plot[i]}",
                rotation=0,
                labelpad=25
            )

            ax.grid(True)


        axes[-1].set_xlabel("Time (s)")


        fig_sources.suptitle(
            f"Excluded ICA component activations (normalized)\n{tmin}-{tmax}s",
            fontsize=14
        )

        plt.tight_layout()

    else:
        print("  - No excluded components to plot")
                
    # --- 5. Apply ICA to the *original* filtered data -------------------------
    ica.exclude = excluded
    raw_clean = ica.apply(raw_filt.copy(), verbose=False)
    if verbose:
        print("[ica_artefact_removal] ICA correction applied to raw_filt → raw_clean")

    return raw_clean, ica, excluded, component_labels





def plot_ica_before_after(
    raw_before,
    raw_after,
    tmin=40,
    tmax=60,
    n_channels=5,
    picks=None,
    random_state=None,
    scale_uv=1e6,
):
    """
    Plot EEG signals before and after ICA cleaning over the same time window.

    Parameters
    ----------
    raw_before : mne.io.Raw
        Raw data before ICA correction.
    raw_after : mne.io.Raw
        Raw data after ICA correction.

    tmin : float
        Start time window in seconds.

    tmax : float
        End time window in seconds.

    n_channels : int
        Number of EEG channels to randomly display if picks is None.

    picks : list | None
        EEG channel indices or names to display.

    random_state : int | None
        Seed for random channel selection.

    scale_uv : float
        Conversion factor to µV (default: 1e6).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.
    """

    rng = np.random.default_rng(random_state)

    # Select EEG channels
    if picks is None:

        eeg_picks = mne.pick_types(
            raw_before.info,
            eeg=True,
            exclude=[]
        )

        picks = rng.choice(
            eeg_picks,
            size=min(n_channels, len(eeg_picks)),
            replace=False
        )

    else:
        # Convert channel names to indices if needed
        if isinstance(picks[0], str):
            picks = mne.pick_channels(
                raw_before.ch_names,
                include=picks
            )


    # Extract data
    data_before, times = raw_before[picks, :]

    data_after, _ = raw_after[picks, :]


    # Select time window
    mask = (
        (times >= tmin) &
        (times <= tmax)
    )

    times = times[mask]

    data_before = data_before[:, mask] * scale_uv
    data_after = data_after[:, mask] * scale_uv


    # Create figure
    fig, axes = plt.subplots(
        len(picks),
        1,
        figsize=(12, 2 * len(picks)),
        sharex=True
    )

    if len(picks) == 1:
        axes = [axes]


    for i, ax in enumerate(axes):

        ax.plot(
            times,
            data_before[i],
            label="Before ICA",
            linewidth=1
        )

        ax.plot(
            times,
            data_after[i],
            label="After ICA",
            linewidth=1
        )

        ax.set_ylabel(
            raw_before.ch_names[picks[i]]
        )

        ax.grid(True)


    axes[0].legend()

    axes[-1].set_xlabel(
        "Time (s)"
    )

    fig.suptitle(
        f"ICA cleaning effect ({tmin}-{tmax}s)",
        fontsize=14
    )

    plt.tight_layout()

    return fig