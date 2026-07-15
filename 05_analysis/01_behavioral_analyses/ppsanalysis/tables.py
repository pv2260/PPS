"""Shared derived tables.

Every hypothesis draws from the same small set of derived tables. Building them
once, here, is what stops H2 and H6 silently disagreeing because one of them was
run after a config cell was edited.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import config as C
from .stats_utils import subject_filter, choose_clinical_pair
from .pps import compute_facilitation, near_far_index, sigmoid_boundary_by_subject
from .collision import compute_collision_indices_by_subject


@dataclass
class Tables:
    """Everything the hypothesis modules need, built once."""

    pps_trials: pd.DataFrame
    collision_trials: pd.DataFrame
    subjects: pd.DataFrame

    # Young-control arm
    pps_young: pd.DataFrame = field(default=None)
    collision_young: pd.DataFrame = field(default=None)
    facilitation_young: pd.DataFrame = field(default=None)
    nearfar_young: pd.DataFrame = field(default=None)
    xc_young: pd.DataFrame = field(default=None)
    delta_pps_young: pd.DataFrame = field(default=None)
    collision_indices_young: pd.DataFrame = field(default=None)

    # Clinical arm
    patient_id: Optional[str] = None
    control_id: Optional[str] = None
    has_patient: bool = False

    # Carried between hypotheses
    sdc_delta_pps: float = np.nan       # set by h4, consumed by h6b
    patient_session: Optional[int] = None   # set by h6, consumed by h7 and h8b
    control_session: int = 1


def build(pps_trials, collision_trials, subjects) -> Tables:
    """Build every shared derived table from the two tidy trial tables."""
    exclude = [s for s in [C.MATCHED_CONTROL_ID] if s is not None]

    pps_young = subject_filter(
        pps_trials,
        subjects_to_keep=C.YOUNG_CONTROL_SUBJECTS,
        group="control" if not C.YOUNG_CONTROL_SUBJECTS else None,
        exclude_subjects=exclude,
    )
    collision_young = subject_filter(
        collision_trials,
        subjects_to_keep=C.YOUNG_CONTROL_SUBJECTS,
        group="control" if not C.YOUNG_CONTROL_SUBJECTS else None,
        exclude_subjects=exclude,
    )

    facilitation_young = compute_facilitation(pps_young)

    patient_id, control_id, _, _ = choose_clinical_pair(pps_trials, collision_trials)

    return Tables(
        pps_trials=pps_trials,
        collision_trials=collision_trials,
        subjects=subjects,
        pps_young=pps_young,
        collision_young=collision_young,
        facilitation_young=facilitation_young,
        nearfar_young=near_far_index(facilitation_young),
        xc_young=sigmoid_boundary_by_subject(facilitation_young, split_speed=False),
        delta_pps_young=sigmoid_boundary_by_subject(facilitation_young, split_speed=True),
        collision_indices_young=compute_collision_indices_by_subject(collision_young),
        patient_id=patient_id,
        control_id=control_id,
        has_patient=(patient_id is not None) and (control_id is not None),
    )


def describe(t: Tables) -> None:
    """One-screen summary of what was built. Read this before trusting anything."""
    print(f"PPS trials       : {len(t.pps_trials):5d}  "
          f"({int(t.pps_trials['usable'].sum())} usable)")
    print(f"Collision trials : {len(t.collision_trials):5d}  "
          f"({int(t.collision_trials['usable'].sum())} usable)")
    print(f"Young controls   : {sorted(t.pps_young['subject'].unique())}")
    print(f"RT window        : {C.RT_MIN_MS}-{C.RT_MAX_MS} ms")
    print(f"Dropped positions: {C.DROP_POSITIONS or 'none'}")
    if t.has_patient:
        print(f"Clinical pair    : patient={t.patient_id}, control={t.control_id}")
    else:
        print("Clinical pair    : none. Aim 2 (H5-H7) and Aim 3b (H8b) will skip.")
