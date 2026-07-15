import mne
import numpy as np
import matplotlib.pyplot as plt


def detect_bad_channels(
    raw: mne.io.Raw,
    flat_thresh_uv: float = 0.5,
    noisy_z_thresh: float = 3.0,
    corr_thresh: float = 0.4,
    corr_window_sec: float = 1.0,
    n_neighbors: int = 4,
    plot: bool = True,
    mark_as_bad: bool = True,
    verbose: bool = True,
):
    """
    Automatically detect bad EEG channels using three complementary criteria
    and optionally mark them in ``raw.info["bads"]`` so that downstream steps
    (ICA, referencing, interpolation) handle them correctly.

    Three criteria are applied independently and their results are unioned:

    1. **Flat channels** — channels whose peak-to-peak amplitude across the
       entire recording falls below ``flat_thresh_uv`` µV.  These are
       dead or disconnected electrodes.

    2. **Noisy channels** — channels whose log-variance is more than
       ``noisy_z_thresh`` standard deviations above the median log-variance
       across all EEG channels.  This catches electrodes with excessive
       high-frequency noise or drift.

    3. **Poorly correlated channels** — channels whose mean absolute
       correlation with their ``n_neighbors`` nearest neighbours (by 3-D
       scalp distance) averaged across 1-second windows falls below
       ``corr_thresh``.  A good EEG electrode is spatially coherent with its
       neighbours; an outlier is likely bridged, noisy, or misplaced.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG recording.  Must have a montage set (run
        ``set_montage_and_check`` first) so that spatial distances can be
        computed for criterion 3.  Modified in-place if ``mark_as_bad=True``.
    flat_thresh_uv : float
        Peak-to-peak amplitude threshold in µV below which a channel is
        considered flat (default 0.5 µV).
    noisy_z_thresh : float
        Z-score threshold on log-variance above which a channel is flagged as
        noisy (default 3.0).  Higher = more permissive (fewer flags).
    corr_thresh : float
        Mean neighbour-correlation threshold below which a channel is flagged
        (default 0.4).  Channels with no montage positions are skipped.
    corr_window_sec : float
        Window length in seconds used for the sliding correlation computation
        (default 1.0 s).
    n_neighbors : int
        Number of nearest spatial neighbours used for the correlation check
        (default 4).
    plot : bool
        If True (default), show a summary figure with per-channel variance,
        neighbour correlation, and a scalp map highlighting bad channels.
    mark_as_bad : bool
        If True (default), append detected channels to ``raw.info["bads"]``.
        Existing entries in ``raw.info["bads"]`` are preserved.
    verbose : bool

    Returns
    -------
    raw : mne.io.Raw
        Same object, with ``raw.info["bads"]`` updated if ``mark_as_bad=True``.
    report : dict
        Keys: ``"flat"``, ``"noisy"``, ``"uncorrelated"``, ``"all_bad"``
        (union, sorted).  Useful for logging or downstream decisions.

    Notes
    -----
    * Always call this **after** ``set_montage_and_check`` and **before**
      ``ica_artefact_removal``.  ICA excludes channels in ``raw.info["bads"]``
      automatically.
    * After ICA you should **interpolate** the bad channels rather than
      leaving them missing::

          raw_clean.interpolate_bads(reset_bads=True)

    Examples
    --------
    >>> raw_filt, report = detect_bad_channels(raw_filt)
    >>> print(report["all_bad"])

    >>> # More permissive — only catch extreme outliers
    >>> raw_filt, report = detect_bad_channels(
    ...     raw_filt, noisy_z_thresh=4.0, corr_thresh=0.3)
    """
    # select channels whose type is EEG
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        raise ValueError("No EEG channels found. Check channel types.")

    eeg_names = [raw.ch_names[i] for i in eeg_picks]
    # extracts the EEG signal data from the Raw object only for the selected EEG channels
    data_uv   = raw.get_data(picks=eeg_picks) * 1e6   # (n_ch, n_times)

    # ── Criterion 1: Flat channels ────────────────────────────────────────────
    ptp        = np.ptp(data_uv, axis=1)               # peak-to-peak per channel
    flat_mask  = ptp < flat_thresh_uv
    flat_chs   = [eeg_names[i] for i in np.where(flat_mask)[0]]

    # ── Criterion 2: Noisy channels (log-variance z-score) ───────────────────
    log_var    = np.log(np.var(data_uv, axis=1) + 1e-12)
    z_scores   = (log_var - np.median(log_var)) / (np.std(log_var) + 1e-12)
    noisy_mask = z_scores > noisy_z_thresh
    noisy_chs  = [eeg_names[i] for i in np.where(noisy_mask)[0]]

    # ── Criterion 3: Poorly correlated with spatial neighbours ───────────────
    # Requires montage positions; skip channels with no 3-D location
    ch_pos_dict = raw.info.get_montage().get_positions()["ch_pos"]         if raw.info.get_montage() is not None else {}

    pos_array  = []   # (n_ch_with_pos, 3)
    pos_names  = []
    for name in eeg_names:
        if name in ch_pos_dict:
            pos_array.append(ch_pos_dict[name])
            pos_names.append(name)

    uncorr_chs = []
    if len(pos_names) >= n_neighbors + 1:
        pos_array  = np.array(pos_array)               # (n, 3)
        n_wins     = int(raw.times[-1] / corr_window_sec)
        win_samp   = int(corr_window_sec * raw.info["sfreq"])

        # Build index map: pos_names → row in data_uv
        name_to_idx = {name: eeg_names.index(name) for name in pos_names}

        # Pairwise Euclidean distances
        from scipy.spatial.distance import cdist
        dists = cdist(pos_array, pos_array)
        np.fill_diagonal(dists, np.inf)

        mean_corr = np.zeros(len(pos_names))
        for i, ch_name in enumerate(pos_names):
            neighbor_idx = np.argsort(dists[i])[:n_neighbors]
            neighbor_rows = [name_to_idx[pos_names[j]] for j in neighbor_idx]
            ch_row        = name_to_idx[ch_name]

            win_corrs = []
            for w in range(n_wins):
                sl = slice(w * win_samp, (w + 1) * win_samp)
                ch_seg = data_uv[ch_row, sl]
                nb_seg = data_uv[neighbor_rows, sl]   # (n_neighbors, win_samp)
                corrs  = [
                    np.abs(np.corrcoef(ch_seg, nb_seg[k])[0, 1])
                    for k in range(len(neighbor_rows))
                ]
                win_corrs.append(np.mean(corrs))
            mean_corr[i] = np.mean(win_corrs)

        low_corr_mask = mean_corr < corr_thresh
        uncorr_chs    = [pos_names[i] for i in np.where(low_corr_mask)[0]]
    else:
        mean_corr  = np.full(len(pos_names), np.nan)
        low_corr_mask = np.zeros(len(pos_names), dtype=bool)
        if verbose:
            print("[detect_bad_channels] Skipping correlation check "
                  "(too few channels with positions).")

    # ── Union ─────────────────────────────────────────────────────────────────
    all_bad = sorted(set(flat_chs) | set(noisy_chs) | set(uncorr_chs))

    report = dict(flat=flat_chs, noisy=noisy_chs,
                  uncorrelated=uncorr_chs, all_bad=all_bad)

    if verbose:
        print(f"[detect_bad_channels] Flat        : {flat_chs}")
        print(f"[detect_bad_channels] Noisy       : {noisy_chs}")
        print(f"[detect_bad_channels] Uncorrelated: {uncorr_chs}")
        print(f"[detect_bad_channels] → Total bad : {all_bad}")

    if mark_as_bad:
        existing = raw.info["bads"]
        new_bads = sorted(set(existing) | set(all_bad))
        raw.info["bads"] = new_bads
        if verbose:
            print(f"[detect_bad_channels] raw.info['bads'] updated: {new_bads}")

    # ── Diagnostic plot ───────────────────────────────────────────────────────
    if plot:
        n_panels = 2 + (1 if len(pos_names) > 0 else 0)
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(6 * n_panels, 5))
        fig.suptitle("Bad channel detection summary", fontsize=13,
                     fontweight="bold")

        # Panel 1 — peak-to-peak amplitude
        ax = axes[0]
        colors_ptp = ["#E53935" if flat_mask[i] else "#37474F"
                      for i in range(len(eeg_names))]
        ax.bar(range(len(eeg_names)), ptp, color=colors_ptp, width=0.8)
        ax.axhline(flat_thresh_uv, color="#E53935", lw=1.2, ls="--",
                   label=f"Flat threshold ({flat_thresh_uv} µV)")
        ax.set_xticks(range(len(eeg_names)))
        ax.set_xticklabels(eeg_names, rotation=90, fontsize=6)
        ax.set_ylabel("Peak-to-peak (µV)")
        ax.set_title("Flat channel check")
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

        # Panel 2 — log-variance z-scores
        ax = axes[1]
        colors_z = ["#E53935" if noisy_mask[i] else "#37474F"
                    for i in range(len(eeg_names))]
        ax.bar(range(len(eeg_names)), z_scores, color=colors_z, width=0.8)
        ax.axhline(noisy_z_thresh, color="#E53935", lw=1.2, ls="--",
                   label=f"Z threshold ({noisy_z_thresh})")
        ax.set_xticks(range(len(eeg_names)))
        ax.set_xticklabels(eeg_names, rotation=90, fontsize=6)
        ax.set_ylabel("Z-score of log-variance")
        ax.set_title("Noisy channel check")
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

        # Panel 3 — neighbour correlation (if positions available)
        if n_panels == 3:
            ax = axes[2]
            colors_c = ["#E53935" if low_corr_mask[i] else "#37474F"
                        for i in range(len(pos_names))]
            ax.bar(range(len(pos_names)), mean_corr, color=colors_c, width=0.8)
            ax.axhline(corr_thresh, color="#E53935", lw=1.2, ls="--",
                       label=f"Corr threshold ({corr_thresh})")
            ax.set_xticks(range(len(pos_names)))
            ax.set_xticklabels(pos_names, rotation=90, fontsize=6)
            ax.set_ylabel("Mean neighbour correlation")
            ax.set_title("Spatial correlation check")
            ax.legend(fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        plt.show()

        # Scalp map with bad channels highlighted
        if raw.info.get_montage() is not None:
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            fig2.suptitle(
                f"Bad channels on scalp map  (red = bad: {all_bad})",
                fontsize=11, fontweight="bold",
            )
            ch_colors = {
                ch: "red" if ch in all_bad else "steelblue"
                for ch in raw.ch_names
            }
            mne.viz.plot_sensors(
                raw.info, kind="topomap", ch_type="eeg",
                show_names=True, axes=ax2, show=False,
            )
            plt.show()

    return raw, report
