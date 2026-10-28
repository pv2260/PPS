"""One module per hypothesis.

Every module exposes `run(t, plot=True)` taking a `ppsanalysis.tables.Tables` and
returning a dict of results. 
"""
from . import h1, h1a, h2, h3, h4, h5, h6, h7a, h7b

__all__ = ["h1", "h1a", "h2", "h3", "h4", "h5", "h6", "h7a", "h7b"]
