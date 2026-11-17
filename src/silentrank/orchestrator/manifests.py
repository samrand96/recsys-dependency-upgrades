"""Case manifests in ``protocol/cases/*.yaml``.

Two shapes are accepted:

* transition manifests (``systematic_single``, ``systematic_snapshot``): a mapping with
  ``collection`` and a ``transitions`` list;
* the issue-guided lead list (``issue_guided.yaml``): a list of leads, or a mapping with a
  ``leads`` list. Leads are not transitions until someone sets up a reproduction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from silentrank import paths

TRANSITION_REQUIRED = ("id", "family", "split", "change", "status")


class ManifestError(ValueError):
    pass


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _transition_entries(doc: dict, source: Path) -> list[dict]:
    collection = doc.get("collection")
    if not collection:
        raise ManifestError(f"{source.name}: missing 'collection'")
    entries = []
    for raw in doc.get("transitions") or []:
        missing = [key for key in TRANSITION_REQUIRED if key not in raw]
        if missing:
            raise ManifestError(f"{source.name}: transition {raw.get('id', '?')} lacks {missing}")
        entries.append(
            {
                "kind": "transition",
                "id": str(raw["id"]),
                "collection": collection,
                "family": raw["family"],
                "split": raw["split"],
                "status": raw["status"],
                "change": raw["change"],
                "held_fixed": raw.get("held_fixed") or {},
                "note": raw.get("note"),
                "selection_reason": doc.get("selection_rule"),
                "manifest_file": source.name,
            }
        )
    return entries


def _lead_entries(doc: Any, source: Path) -> list[dict]:
    if isinstance(doc, dict):
        raw_leads = doc.get("leads") or doc.get("cases") or doc.get("entries") or []
    else:
        raw_leads = doc or []
    entries = []
    for raw in raw_leads:
        if not isinstance(raw, dict) or "id" not in raw:
            raise ManifestError(f"{source.name}: every lead needs an 'id'")
        entry = dict(raw)
        entry.update(
            kind="lead",
            id=str(raw["id"]),
            collection="issue_guided",
            status=raw.get("status", "lead"),
            manifest_file=source.name,
        )
        entries.append(entry)
    return entries


def load_all(directory: Path | None = None) -> list[dict]:
    """Every transition and lead in the manifests, in file order. Ids must be unique."""
    directory = directory or paths.manifests_dir()
    entries: list[dict] = []
    if not directory.is_dir():
        return entries
    for source in sorted(directory.glob("*.yaml")):
        doc = _load_yaml(source)
        if doc is None:
            continue
        if isinstance(doc, dict) and "transitions" in doc:
            entries.extend(_transition_entries(doc, source))
        else:
            entries.extend(_lead_entries(doc, source))
    seen: set[str] = set()
    for entry in entries:
        if entry["id"] in seen:
            raise ManifestError(f"duplicate id {entry['id']!r} across manifests")
        seen.add(entry["id"])
    return entries


def sync_transitions(con, entries: list[dict] | None = None) -> int:
    """Copy manifest transitions into the ledger without touching screening results."""
    entries = load_all() if entries is None else entries
    count = 0
    for entry in entries:
        if entry["kind"] != "transition":
            continue
        con.execute(
            """INSERT INTO transitions
                   (transition_id, family, collection, split, direct_change_json, held_fixed_json,
                    selection_reason, status, manifest_file)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (transition_id) DO UPDATE SET
                   family = excluded.family, collection = excluded.collection, split = excluded.split,
                   direct_change_json = excluded.direct_change_json,
                   held_fixed_json = excluded.held_fixed_json,
                   selection_reason = excluded.selection_reason, manifest_file = excluded.manifest_file""",
            [
                entry["id"], entry["family"], entry["collection"], entry["split"],
                json.dumps(entry["change"], sort_keys=True), json.dumps(entry["held_fixed"], sort_keys=True),
                entry["selection_reason"], entry["status"], entry["manifest_file"],
            ],
        )
        count += 1
    return count
