"""Code that runs INSIDE an environment under test.

Hard rule (README, rule 13): modules here may import only the standard
library, NumPy, SciPy and the library under test. Nothing from the orchestrator
(duckdb, pandas-as-analysis, yaml) may be imported here, and results leave
the environment only as .npy/.npz and JSON files.
"""
