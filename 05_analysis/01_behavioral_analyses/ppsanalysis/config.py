"""All settings in one place.

Everything the analysis depends on that is not data lives here. The notebook
should not define analysis constants; it should import them, and override them
here if it needs to. That is the difference between a notebook you can trust and
one where a stale cell silently changes a result three cells later.
"""
import re

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PILOT_DATA_DIR = "/Users/pamelavandenenden/Desktop/PPS/05_analysis/pilots/"
OUT_DIR = "."

# ---------------------------------------------------------------------------
# Filename / column parsing
# ---------------------------------------------------------------------------
TRIAL_FILE = re.compile(
    r"sub-(?P<subject>.+?)_session-(?P<session>\d+)_task(?P<task>[12])_trials.*\.csv$"
)
POSITION_COL_PATTERN = re.compile(r"^position_(D\d+)_ms$")

# ---------------------------------------------------------------------------
# Subject metadata (setup.json is not read: it contains inaccurate information)
# ---------------------------------------------------------------------------
SUBJECT_META = {
    "theo":  {"group": "control", "cohort": "pilot", "has_dbs": False, "shoulder_width_cm": 42.0},
    "franc": {"group": "control", "cohort": "pilot", "has_dbs": False, "shoulder_width_cm": 42.0},
}

YOUNG_CONTROL_SUBJECTS = ["theo", "franc"]

PATIENT_ID = None
MATCHED_CONTROL_ID = None
CLINICAL_SESSION = None
CLINICAL_COHORT = None

# ---------------------------------------------------------------------------
# Condition labels
# ---------------------------------------------------------------------------
SLOW_LABEL = "slow"
FAST_LABEL = "fast"

# ---------------------------------------------------------------------------
# Trial usability
# ---------------------------------------------------------------------------
# RT_MAX_MS was 3000 in the original notebook. That is longer than an entire
# fast trial, so it admitted responses made near the END of the trial, which are
# inattention rather than detection. Petrizzo et al. (2024) excluded RTs above
# 1000 ms on exactly this task and reported mean RTs of 250-320 ms.
RT_MIN_MS = 100
RT_MAX_MS = 1000

# Positions to exclude entirely. Set to (7,) to drop D7, () to keep everything.
# NOTE: dropping a distance level is a preregistration deviation. Record WHY.
DROP_POSITIONS = ()

# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
RANDOM_SEED = 123
N_BOOT = 5000

# ---------------------------------------------------------------------------
# Model-fit sanity gates
# ---------------------------------------------------------------------------
# A sigmoid whose inflection falls outside the sampled range, or whose slope is
# absurd, is not an estimate. It is a fitting artefact that happens to converge.
AIC_MARGIN = 2.0
K_MAX = 10.0

# Support thresholds
ALPHA = 0.05
BOOT_SIGN_THRESHOLD = 0.95
