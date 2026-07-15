"""One module per hypothesis.

Every module exposes `run(t, plot=True)` taking a `ppsanalysis.tables.Tables` and
returning a dict of results. h8b additionally takes the H6 result, because its
paired bootstrap must reuse H6's PPS resamples on the same bootstrap index.

    Aim 1 (young controls)   h1, h1a, h2, h3, h4
    Aim 2 (patient)          h5, h6, h6b, h7
    Aim 3 (PPS <-> Hit/Miss) h8a (controls), h8b (patient-control pair)
"""
from . import h1, h1a, h2, h3, h4, h5, h6, h6b, h7, h8a, h8b

__all__ = ["h1", "h1a", "h2", "h3", "h4", "h5", "h6", "h6b", "h7", "h8a", "h8b"]
