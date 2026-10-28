"""PPS behavioural analysis.

    import ppsanalysis as pa

    pps, coll, subs = pa.io.make_analysis_csvs()
    pps = pa.qc.mark_pps_usable(pps)
    t = pa.tables.build(pps, coll, subs)
    pa.tables.describe(t)

    r1 = pa.hypotheses.h1.run(t)
    r4 = pa.hypotheses.h4.run(t)
    t.sdc_delta_pps = r4["SDC_DELTA_PPS"]      # H4 feeds H6b's noise floor
    r6 = pa.hypotheses.h6.run(t)
    t.patient_session = r6.get("PATIENT_SESSION_USED")   # H6 feeds H7 and H8b

The couplings above are the ones the notebook made implicitly through globals.
They are now explicit, which means you cannot run H6b without H4 by accident.
"""
from . import config, io, qc, stats_utils, pps, collision, permutations, tables, hypotheses

__all__ = ["config", "io", "qc", "stats_utils", "pps", "collision",
           "permutations", "tables", "hypotheses"]
