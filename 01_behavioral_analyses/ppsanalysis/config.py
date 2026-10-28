"""Every setting the analysis uses.

The notebook reads these values and does not define its own. To change how the
analysis is run, edit this file and restart the kernel. Some modules copy a
value when they are first imported, so a change made from the notebook while it
is running does not always reach them.
"""
import re

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PILOT_DATA_DIR = "/Users/pamelavandenenden/Desktop/PPS/05_analysis/data/behavior"
OUT_DIR = "."

RESULTS_DIR = (
    "/Users/pamelavandenenden/Desktop/PPS/05_analysis/"
    "01_behavioral_analyses/results"
)

# PNGs written by save() in the notebook.
FIGURE_DIR = RESULTS_DIR + "/young_controls/data_exclusions"

# SVG and PDF figures written by the package (the save_as= argument).
PACKAGE_FIGURE_DIR = "figures"

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
TRIAL_FILE = re.compile(
    r"sub-(?P<subject>.+?)_session-(?P<session>\d+)_task(?P<task>[12])_trials.*\.csv$"
)

POSITION_COL_PATTERN = re.compile(r"^position_(D\d+)_ms$")

# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------
# Every young control tested.
ALL_SUBJECTS = [
    "P002", "P003", "P004", "P005", "P006",
    "P007", "P008", "P009", "P010", "P011",
    "T001", "T002",
]

# Excluded from every analysis, with the reason.
EXCLUDED_SUBJECTS = {
    "P004": "controller fault observed during testing",
    "P006": "controller fault",
    "P007": "incorrect shoulder width entered",
}

# The young controls that are analysed.
YOUNG_CONTROL_SUBJECTS = [
    s for s in ALL_SUBJECTS if s not in EXCLUDED_SUBJECTS
]

# Young controls run on the classic (Serino-like) trial composition.
CLASSIC = ["T001", "T002"]

PATIENT_ID = "PT001"

# Both matched controls are loaded; MATCHED_CONTROL is the one the patient is
# compared with in sections 6 and 7.
MATCHED_CONTROL_IDS = ["PA001", "PA002"]
MATCHED_CONTROL = "PA002"

# Older name, still read by stats_utils.choose_clinical_pair.
MATCHED_CONTROL_ID = MATCHED_CONTROL_IDS

CLINICAL_SESSION = None
CLINICAL_COHORT = None

# Participants left out of one group analysis only, on top of
# EXCLUDED_SUBJECTS. An empty list means everyone analysed enters it.
GROUP_DROPS = {
    "qc_figures":          CLASSIC,   # 2.2, 2.5, 2.6
    "h1_group_test":       [],        # H1.1 to H1.3: classic design included
    "baseline_comparison": CLASSIC,   # H1.4
    "h1_trial_bootstrap":  CLASSIC,   # H1.5, bf.run
    "h1_nearfar_boxes":    CLASSIC,   # H1.5 figure
    "h1a":                 CLASSIC,
    # P005 is in the H2 test (and so in H7a) but not in the H2 figure.
    "h2":                  CLASSIC,
    "h2_figure":           CLASSIC + ["P005"],
    "h3":                  [],        # T001, T002, P010 have no collision trials
}

# ---------------------------------------------------------------------------
# Subject metadata (setup.json is not read: it contains inaccurate information)
# ---------------------------------------------------------------------------
SUBJECT_META = {
    "P002": {"group": "control", "cohort": "control", "has_dbs": False, "shoulder_width_cm": 44.0},
    "P003": {"group": "control", "cohort": "control", "has_dbs": False, "shoulder_width_cm": 38.0},
    "P004": {"group": "control", "cohort": "control", "has_dbs": False, "shoulder_width_cm": 44.0},
    "P005": {"group": "control", "cohort": "control", "has_dbs": False, "shoulder_width_cm": 46.0},
    "P006": {"group": "control", "cohort": "control", "has_dbs": False, "shoulder_width_cm": 48.0},
    # P007: 44 cm, but entered as 24 cm on task 2 by mistake.
    "P007": {"group": "control", "cohort": "control", "has_dbs": False, "shoulder_width_cm": 44.0},
    "P008": {"group": "control", "cohort": "control", "has_dbs": False, "shoulder_width_cm": 36.0},
    "P009": {"group": "control", "cohort": "control", "has_dbs": False, "shoulder_width_cm": 38.0},
    "P010": {"group": "control", "cohort": "control", "has_dbs": False, "shoulder_width_cm": 42.0},
    "P011": {"group": "control", "cohort": "control", "has_dbs": False, "shoulder_width_cm": 43.0},
    "T001": {"group": "control", "cohort": "control", "has_dbs": False, "shoulder_width_cm": 48.0},
    "T002": {"group": "control", "cohort": "control", "has_dbs": False, "shoulder_width_cm": 39.0},
    "PA001": {"group": "control", "cohort": "matched", "has_dbs": False, "shoulder_width_cm": 46.0},
    "PA002": {"group": "control", "cohort": "matched", "has_dbs": False, "shoulder_width_cm": 42.0},
    "PT001": {"group": "patient", "cohort": "patient", "has_dbs": True, "shoulder_width_cm": 49.0},
}

# ---------------------------------------------------------------------------
# Design variants
# ---------------------------------------------------------------------------
# Three trial compositions were run. They differ in two respects that the
# literature treats separately, and which are confounded across some pairs:
#
#   cue reliability   the proportion of trials carrying the moving visual
#                     stimulus that also carry a tactile target. This is the
#                     quantity Kandula et al. (2017) manipulated; catch
#                     trials still show the light, so unimodal tactile
#                     trials fall outside it. They found the sigmoidal
#                     pattern unrecoverable at 75% and recoverable at 50%
#                     and 25%.
#   tactile sampling  the number of unimodal tactile trials, and therefore
#                     the trials per delay-matched baseline cell, which sets
#                     the precision of the correction (Holmes et al.).
#
# Comparing SERINO_LIKE with INTERMEDIATE isolates cue reliability, since
# the tactile sampling is comparable (5 and 4 per cell). Comparing
# INTERMEDIATE with REBALANCED varies both, so a difference between them
# cannot be attributed to either alone.

DESIGNS = {
    "serino_like": {
        "label": "Serino-like",
        "subjects": ["T001", "T002"],
        "n_vt": 238, "n_t": 70, "n_v": 34,
        "cue_reliability": 0.875,
        "t_per_cell": 5,
        "note": "standard implementation, minimal catch trials",
    },
    "intermediate": {
        "label": "Intermediate",
        "subjects": ["P002", "P004", "P003", "P005", "P006", "P007", "P008"],
        "n_vt": 210, "n_t": 56, "n_v": 114,
        "cue_reliability": 0.648,
        "t_per_cell": 4,
        "note": "catch trials raised to 30%; tactile sampling unchanged",
    },
    "rebalanced": {
        "label": "Rebalanced",
        "subjects": ["P009", "P010", "P011"],
        "n_vt": 140, "n_t": 126, "n_v": 114,
        "cue_reliability": 0.551,
        "t_per_cell": 9,
        "note": "trials moved from VT to T; cue reliability falls further",
    },
}

DESIGN_OF_SUBJECT = {
    subject: name
    for name, spec in DESIGNS.items()
    for subject in spec["subjects"]
}


def design_table():
    """One row per design, for the methods table and for sanity checks."""
    import pandas as pd

    return pd.DataFrame([
        {
            "design": spec["label"],
            "n_subjects": len(spec["subjects"]),
            "VT": spec["n_vt"], "T": spec["n_t"], "V": spec["n_v"],
            "total": spec["n_vt"] + spec["n_t"] + spec["n_v"],
            "cue_reliability": spec["cue_reliability"],
            "t_per_cell": spec["t_per_cell"],
            "subjects": ", ".join(spec["subjects"]),
        }
        for spec in DESIGNS.values()
    ])


# ---------------------------------------------------------------------------
# Condition labels
# ---------------------------------------------------------------------------
SLOW_LABEL = "slow"
FAST_LABEL = "fast"

# ---------------------------------------------------------------------------
# Trial exclusions
# ---------------------------------------------------------------------------
# PPS trials: a trial is kept only if RT_MIN_MS < rt_ms < RT_MAX_MS.
# Not applied to collision trials.
RT_MIN_MS = 100
RT_MAX_MS = 900

# Positions to exclude entirely. Set to (7,) to drop D7, () to keep everything.
DROP_POSITIONS = ()

# Collision trials, young controls only: an RT more than this many
# interquartile ranges outside that participant's trials of the same speed
# and trial type is excluded.
COLLISION_RT_IQR = 3.0

# ---------------------------------------------------------------------------
# Tactile baseline
# ---------------------------------------------------------------------------
# Which tactile baseline facilitation is computed against. See section 0.4 of
# the notebook for what each option means.
CORRECTION = "expectancy_model"

# Polynomial order of the expectancy model in delay.
DEGREE = 1

# The delay variable the expectancy model is fitted against.
DELAY_COLUMN = "tactile_foreperiod_ms"

# ---------------------------------------------------------------------------
# Collision task
# ---------------------------------------------------------------------------
# clearance_cm = offset_from_edge_cm - BALL_RADIUS_CM
BALL_RADIUS_CM = 8.75

# ---------------------------------------------------------------------------
# H1a
# ---------------------------------------------------------------------------
# A cell with fewer usable trials than this gives no estimate.
H1A_MIN_CELL_N = 3

# Resamples used to judge whether x_c is precise. At 1 the precision check
# cannot fail on width; set to N_BOOT before reporting.
H1A_N_BOOT = 1

# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
RANDOM_SEED = 123
N_BOOT = 5000

# ---------------------------------------------------------------------------
# Model-fit limits
# ---------------------------------------------------------------------------
# A sigmoid whose inflection falls outside the sampled range, or whose slope is
# absurd, is not an estimate. It is a fitting artefact that happens to converge.
AIC_MARGIN = 0.0
K_MAX = 6.0

# Support thresholds
ALPHA = 0.05
BOOT_SIGN_THRESHOLD = 0.95