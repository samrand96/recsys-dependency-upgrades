"""silentrank: harness for dependency-upgrade ranking regression experiments.

This package is the *orchestrator*: it builds environments, launches runs,
evaluates results and keeps the ledger. Code that runs inside an environment
under test lives in ``silentrank.worker`` and must stay free of orchestrator
dependencies (README, rule 13).
"""

__version__ = "0.1.0"
