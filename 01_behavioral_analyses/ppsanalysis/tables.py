"""Trial tables, one per cohort.

``build`` prepares the raw exports once: timing context for the PPS trials,
speed-sequence context for the collision trials. ``cohorts`` then runs QC per
cohort and returns a ``Cohort`` for each.

    data = pa.tables.build(pps_trials, collision_trials, subjects)
    coh = pa.tables.cohorts(data, {
        "young":   config.YOUNG_CONTROL_SUBJECTS,
        "matched": config.MATCHED_CONTROL_IDS,
        "patient": [config.PATIENT_ID],
    })

    c = coh["young"]

    qc.exclusion_table(c.marked)        # every row, with exclusion reasons
    h1.get_trials(c)                    # usable T and VT trials
    h2.run_by_speed(c.trials)

One object per cohort, three views of it:

    c.marked      every row, ``usable`` and ``exclusion_reason`` set   -> qc
    c.pps_young   usable T and VT trials                              -> h1, h1a
    c.trials      the analysis-ready trial table                      -> h1, h2

``pps_young`` keeps its name because the hypothesis modules look for that
attribute. It is the cohort's trials, not only the young controls'.
"""
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import config as C
from .stats_utils import choose_clinical_pair


CONDITIONS = ["T", "VT"]


# ==================================================================
# Raw tables
# ==================================================================


@dataclass
class Data:
    """The prepared exports, before any cohort is selected."""

    pps_trials: pd.DataFrame
    collision_trials: pd.DataFrame
    subjects: pd.DataFrame

    patient_id: Optional[str] = None
    control_id: Optional[str] = None

    @property
    def has_patient(self):
        return self.patient_id is not None and self.control_id is not None

    def __repr__(self):
        return (
            f"<Data: {len(self.pps_trials)} PPS trials, "
            f"{len(self.collision_trials)} collision trials, "
            f"{self.pps_trials['subject'].nunique()} subject(s)>"
        )


def build(pps_trials, collision_trials, subjects) -> Data:
    """Prepare the raw exports. No rows are removed and no cohort is selected."""

    pps_trials = _add_pps_timing(pps_trials)
    collision_trials = _add_collision_speed_context(collision_trials)

    patient_id, control_id, _, _ = _clinical_pair(pps_trials, collision_trials)

    return Data(
        pps_trials=pps_trials,
        collision_trials=collision_trials,
        subjects=subjects,
        patient_id=patient_id,
        control_id=control_id,
    )


def _clinical_pair(pps_trials, collision_trials):
    """Resolve the patient and matched control, tolerating their absence."""
    try:
        return choose_clinical_pair(pps_trials, collision_trials)
    except ValueError:
        return None, None, None, None


# ==================================================================
# Cohorts
# ==================================================================


@dataclass
class Cohort:
    """One cohort's trials, in the shapes the modules expect."""

    name: str
    marked: pd.DataFrame
    subjects: pd.DataFrame = None
    collision: pd.DataFrame = None

    _trials: pd.DataFrame = field(default=None, repr=False)

    @property
    def pps_young(self):
        """Usable T and VT trials. The name the hypothesis modules look for."""
        return self.marked.loc[
            self.marked["usable"].fillna(False).astype(bool)
            & self.marked["sensory_condition"].isin(CONDITIONS)
        ].copy()

    @property
    def trials(self):
        """Analysis-ready trials, built once on first access."""
        if self._trials is None:
            from .hypotheses import h1

            self._trials = h1.get_trials(self, verbose=False)
        return self._trials

    @property
    def collision_trials(self):
        """Alias for ``collision``. The hypothesis modules look for this name."""
        return self.collision

    @property
    def subject_ids(self):
        return sorted(self.marked["subject"].dropna().unique())


def cohorts(
    data: Data,
    groups: dict,
    rt_min=None,
    rt_max=None,
    min_cell_n=None,
    design_map=None,
    verbose=True,
) -> dict:
    """Run QC once per cohort and return a ``Cohort`` for each.

    Parameters
    ----------
    groups : dict
        Name to list of subject ids, e.g. ``{"young": [...], "patient": [...]}``.
        Subjects absent from the exports are reported, not an error.
    rt_min, rt_max : float or None
        Absolute RT window. Defaults to ``config.RT_MIN_MS`` / ``RT_MAX_MS``.
    min_cell_n : int or None
        Off by default. Setting it excludes cells below that many trials,
        which is a real exclusion and belongs in a sensitivity analysis.
    design_map : dict or None
        Subject to design label, added as a ``design`` column.
    """
    from . import qc

    rt_min = C.RT_MIN_MS if rt_min is None else rt_min
    rt_max = C.RT_MAX_MS if rt_max is None else rt_max

    present = set(data.pps_trials["subject"].dropna().unique())
    out = {}

    for name, wanted in groups.items():
        wanted = [s for s in (wanted or []) if s is not None]
        missing = [s for s in wanted if s not in present]
        found = [s for s in wanted if s in present]

        if not found:
            if verbose:
                print(f"{name:8s} no trials found; skipped")
            continue

        _, marked = qc.prepare_analysis_tables(
            data,
            subjects=found,
            rt_min=rt_min,
            rt_max=rt_max,
            min_cell_n=min_cell_n,
        )

        marked = marked.copy()
        marked["cohort_label"] = name
        if design_map is not None:
            marked["design"] = marked["subject"].map(design_map)

        collision = data.collision_trials.loc[
            data.collision_trials["subject"].isin(found)
        ].copy()

        out[name] = Cohort(
            name=name,
            marked=marked,
            subjects=data.subjects,
            collision=collision,
        )

        if verbose:
            print(out[name])
            if missing:
                print(f"{'':8s} not in the exports: {missing}")

    return out


def describe(data: Data, coh: dict = None) -> None:
    """One screen. Read this before trusting anything downstream."""
    print(f"PPS trials       : {len(data.pps_trials):5d}")
    print(f"Collision trials : {len(data.collision_trials):5d}")
    print(f"RT window        : {C.RT_MIN_MS}-{C.RT_MAX_MS} ms")
    print(f"Dropped positions: {C.DROP_POSITIONS or 'none'}")

    if data.has_patient:
        print(
            f"Clinical pair    : patient={data.patient_id}, "
            f"control={data.control_id}"
        )
    else:
        print("Clinical pair    : none. Aim 2 (H5-H7) and Aim 3b will skip.")

    if coh:
        print()
        for c in coh.values():
            usable = int(c.marked["usable"].fillna(False).sum())
            rt = c.marked.loc[c.marked["usable"].fillna(False), "rt_ms"]
            print(
                f"  {c.name:8s} {usable:5d} usable, "
                f"{len(c.subject_ids)} subject(s), "
                f"RT {rt.min():.0f}-{rt.max():.0f} ms"
            )
            print(f"  {'':8s} {c.subject_ids}")


# ==================================================================
# Context added to the raw tables
# ==================================================================


def _add_pps_timing(pps_trials: pd.DataFrame) -> pd.DataFrame:
    """Derive the within-trial tactile foreperiod.

    ``vibrotactile_onset_ms`` may be an absolute or session-relative
    timestamp, so temporal expectancy uses elapsed time from trial start:

        tactile_foreperiod_ms = vibrotactile_onset_ms - trial_start_ms

    No rows are removed.
    """
    pps = pps_trials.copy()

    required = ["trial_start_ms", "vibrotactile_onset_ms"]
    missing = [c for c in required if c not in pps.columns]

    if missing:
        raise KeyError(
            f"Cannot derive the tactile foreperiod. Missing: {missing}. "
            "Keep these columns when the canonical PPS trial table is built."
        )

    for column in required:
        pps[column] = pd.to_numeric(pps[column], errors="coerce")

    pps["tactile_foreperiod_ms"] = (
        pps["vibrotactile_onset_ms"] - pps["trial_start_ms"]
    )

    return pps


def _add_collision_speed_context(collision: pd.DataFrame) -> pd.DataFrame:
    """Add previous-speed, speed-sequence and transition-status variables.

    Exported values are preserved where present; anything missing is
    reconstructed from trial order. ``speed_sequence`` takes values such as
    ``slow-fast``; the first trial of a session is ``start-slow`` or
    ``start-fast``.
    """
    collision = collision.copy()

    if collision.empty:
        return collision

    missing = {"subject", "speed"}.difference(collision.columns)
    if missing:
        raise KeyError(
            f"Cannot build collision speed context. Missing: {sorted(missing)}"
        )

    original_index = collision.index

    sort_columns = [
        c for c in ["subject", "session", "block", "trial"]
        if c in collision.columns
    ]
    ordered = (
        collision.sort_values(sort_columns) if sort_columns else collision
    ).copy()

    # Lagged quantities must never cross a participant or a session.
    group_columns = ["subject"]
    if "session" in ordered.columns:
        group_columns.append("session")

    ordered["speed"] = _clean(ordered["speed"])

    if "previous_speed" not in ordered.columns:
        ordered["previous_speed"] = (
            ordered.groupby(group_columns, sort=False)["speed"].shift(1)
        )
    else:
        ordered["previous_speed"] = _clean(ordered["previous_speed"])

    has_previous = ordered["previous_speed"].isin(["slow", "fast"])
    has_current = ordered["speed"].isin(["slow", "fast"])

    if "speed_sequence" not in ordered.columns:
        sequence = pd.Series(pd.NA, index=ordered.index, dtype="string")

        both = has_previous & has_current
        sequence.loc[both] = (
            ordered.loc[both, "previous_speed"] + "-" + ordered.loc[both, "speed"]
        )

        start = ~has_previous & has_current
        sequence.loc[start] = "start-" + ordered.loc[start, "speed"]

        ordered["speed_sequence"] = sequence
    else:
        ordered["speed_sequence"] = _clean(ordered["speed_sequence"])

    if "transition_status" not in ordered.columns:
        status = pd.Series("transition", index=ordered.index, dtype="string")
        status.loc[~has_previous] = "start"

        same = ordered["previous_speed"].eq(ordered["speed"]).fillna(False)
        status.loc[has_previous & same] = "repetition"

        ordered["transition_status"] = status
    else:
        ordered["transition_status"] = _clean(ordered["transition_status"])

    # Return the row order build() received.
    return ordered.reindex(original_index)


def _clean(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


# ==================================================================
# Clinical adapter
# ==================================================================
#
# The Aim 2 and Aim 3 modules (h4 to h7b) were written against the old
# ``Tables`` object, which carried the raw trial tables, the patient and
# control ids, and the chosen sessions all in one place. ``Cohort``
# splits those across several objects, so this rebuilds the one shape
# those modules expect.
#
# The trials it hands over are the QC'd ones, not the raw export, so the
# single-case tests run on the same trials as everything else. That was
# ambiguous in the old structure and is worth pinning down: a permutation
# test on unfiltered trials would be testing a different dataset from the
# one the group analyses used.


@dataclass
class Clinical:
    """What the single-case modules read.

    ``patient_session`` starts as None. H5 chooses it and writes it back,
    and H6 and H7b then read it rather than choosing again, so all three
    are guaranteed to describe the same session.
    """

    pps_trials: pd.DataFrame
    collision_trials: pd.DataFrame

    patient_id: Optional[str] = None
    control_id: Optional[str] = None

    patient_session: Optional[int] = None
    control_session: int = 1

    # h7a reads these two; the notebook fills them from h2 and the
    # collision indices.
    delta_pps_young: pd.DataFrame = None
    collision_indices_young: pd.DataFrame = None

    @property
    def has_patient(self):
        return (
            self.patient_id is not None
            and self.control_id is not None
        )

    def __repr__(self):
        return (
            f"<Clinical: patient={self.patient_id}, "
            f"control={self.control_id}, "
            f"session={self.patient_session}, "
            f"{len(self.pps_trials)} PPS trials>"
        )


def clinical(data, cohorts, control_id=None, control_session=1):
    """Assemble the object h4 to h7b expect, from the QC'd cohorts.

    Parameters
    ----------
    data : Data
        The output of ``build``, used only for the resolved patient id.
    cohorts : dict of str -> Cohort
        The output of ``cohorts``. Every cohort's usable trials are
        pooled, so the patient, the matched controls and the young
        controls are all reachable by subject id.
    control_id : str or None
        Which matched control to pair with the patient. Defaults to the
        one ``choose_clinical_pair`` resolved, or the first matched
        control if that is unavailable.
    control_session : int, default 1
        The preregistered convention.
    """
    pps = pd.concat(
        [cohort.pps_young for cohort in cohorts.values()],
        ignore_index=True,
    )

    collision = pd.concat(
        [
            cohort.collision for cohort in cohorts.values()
            if cohort.collision is not None
        ],
        ignore_index=True,
    ) if any(c.collision is not None for c in cohorts.values()) else (
        data.collision_trials
    )

    # One control, not a list. The single-case tests pair the patient with
    # one matched control, and handing them a list silently compares a
    # column against two values instead of one.
    if control_id is None:
        control_id = data.control_id

    if isinstance(control_id, (list, tuple, set)):
        control_id = sorted(control_id)[0] if control_id else None

    if control_id is None and "matched" in cohorts:
        available = cohorts["matched"].subject_ids
        control_id = available[0] if available else None

    if control_id is not None and not isinstance(control_id, str):
        raise TypeError(
            f"control_id must be a single subject id, got {control_id!r}"
        )

    return Clinical(
        pps_trials=pps,
        collision_trials=collision,
        patient_id=data.patient_id,
        control_id=control_id,
        control_session=control_session,
    )