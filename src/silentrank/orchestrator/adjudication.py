"""Record mechanism and cause-class decisions for real cases.

Append-only: a revised decision is a new row with a later date. The latest row per
(case_id, mechanism) is the current decision. Synthetic controls are never adjudicated here.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from silentrank import paths
from silentrank.orchestrator import ledger

CAUSE_CLASSES = ("upstream_defect", "downstream_incompatibility", "intended_semantic_change",
                 "numerical_variation", "unresolved")


def record(case_id: str, mechanism: str, cause_class: str, evidence: list[str], adjudicator: str) -> dict:
    if cause_class not in CAUSE_CLASSES:
        raise ValueError(f"cause class must be one of {', '.join(CAUSE_CLASSES)}")
    if not mechanism.strip() or not evidence:
        raise ValueError("a mechanism and at least one evidence item are required")
    root = paths.repo_root()
    for item in evidence:
        if not item.startswith(("http://", "https://")) and not (root / item).exists():
            raise ValueError(f"evidence path does not exist: {item}")
    row = [case_id, mechanism.strip(), cause_class, " | ".join(evidence), adjudicator, dt.date.today()]
    with ledger.writer() as con:
        con.execute("INSERT INTO adjudications VALUES (?, ?, ?, ?, ?, ?)", row)
    return dict(zip(("case_id", "mechanism", "cause_class", "evidence", "adjudicator", "decided_at"), map(str, row)))
