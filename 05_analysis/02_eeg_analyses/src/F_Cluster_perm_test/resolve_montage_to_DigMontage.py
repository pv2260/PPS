import mne

def resolve_montage_to_DigMontage(source):
    """
    Turn MONTAGE_SOURCE into a proper mne.channels.DigMontage, whatever
    form it was given in (name string, DigMontage, or an object -- Raw,
    Epochs, Info -- that already carries real digitized positions).
    """
    if source is None:
        return None
    if isinstance(source, mne.channels.DigMontage):
        return source
    if isinstance(source, str):
        return mne.channels.make_standard_montage(source)
    # otherwise assume it's a Raw / Epochs / Info (or anything with .info)
    info = source.info if hasattr(source, "info") else source
    montage = info.get_montage()
    if montage is None:
        raise ValueError(
            "MONTAGE_SOURCE object has no digitization/montage attached "
            "(info.get_montage() returned None). Pass a Raw/Epochs that "
            "still has its original dig points, a montage name string, "
            "or a DigMontage object instead."
        )
    return montage
