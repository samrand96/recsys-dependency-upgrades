"""Generate every paper table and in-text number from the ledgers.

Run from the project root, in the orchestrator environment:

    uv run --locked python paper/scripts/make_tables.py

Inputs (read-only): results/ledger.duckdb and results/compare/*.json for the
primary host; results/hosts/<job>/results/{ledger.duckdb,compare/} for the
Hugging Face Jobs hosts.
Outputs: paper/generated/*.tex. Nothing here is hand-edited; rerun to refresh.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "generated"
PRIMARY = ROOT / "results"
HOSTS = {  # job folder -> host label used in the paper (B ran as two Jobs)
    "second-cpu-1": "B",
    "epyc-log1p-1": "B",
    "intel-cpu-1": "C",
    "gpu-t4-1": "GPU",
}

OUTCOMES = ("identical", "changed", "regression", "contract", "uncertain", "failure", "not_applicable")

# Mechanism ids M1-M8 as in the paper. Matching is on the adjudication text.
MECHANISMS = [
    ("M1", r"tie", "NumPy 2.0 unstable selection reorders exact ties"),
    ("M2", r"Cooperlake", "OpenBLAS core selection (ALS)"),
    ("M3", r"EASE", "OpenBLAS 0.3.31 bundled with NumPy 2.4.6 (EASE inverse)"),
    ("M4", r"log1p", "NumPy 2.0 \\texttt{log1p} last-bit change (BM25 weights)"),
    ("M5", r"PyTorch", "PyTorch 2.10 CUDA training (BPR)"),
    ("M6", r"np\.NaN", "Removed \\texttt{np.NaN} alias (Recommenders SAR)"),
    ("M7", r"in1d", "Missing \\texttt{np.in1d} (implicit 0.7.2, NumPy 2.4.4)"),
    ("M8", r"pandas<3", "Recommenders requires \\texttt{pandas<3}"),
]


def _case_outcome(report: dict, applicable) -> tuple[str, dict]:
    """Collapse one case's compare report (all modes) into one outcome category."""
    if applicable is False or report.get("status") == "not_applicable":
        return "not_applicable", {}
    modes = report.get("modes", {})
    if report.get("status") != "ok" or any(m.get("status") != "ok" for m in modes.values()):
        return "failure", {}
    verdicts = [v["verdict"] for m in modes.values() for v in m.get("verdicts", [])]
    changed_frac, losses, stages = 0.0, [], set()
    for m in modes.values():
        ab = m.get("AB", {})
        if ab.get("contract_failure"):
            return "contract", {}
        for pair in ab.get("runs", []):
            if pair.get("changed_stages"):
                stages.add(pair["changed_stages"][0])
        beh = ab.get("behavior") or {}
        changed_frac = max(changed_frac, beh.get("users_changed_frac") or 0.0)
        q = ((ab.get("quality") or {}).get("ndcg10") or {}).get("primary")
        if q:
            losses.append(q["estimate"])
    detail = {"changed_frac": changed_frac, "max_loss": max(losses, key=abs, default=0.0), "stages": sorted(stages)}
    if "harmful" in verdicts:
        return "regression", detail
    if "inconclusive" in verdicts:
        return "uncertain", detail
    return ("changed" if stages else "identical"), detail


def _load(results: Path) -> dict[str, dict]:
    """case_id -> {collection, split, dataset, pipeline, outcome, detail}."""
    con = duckdb.connect(str(results / "ledger.duckdb"), read_only=True)
    try:
        rows = con.execute(
            """SELECT c.case_id, c.pipeline, c.dataset, c.config_json, c.applicable, t.split
               FROM cases c LEFT JOIN transitions t USING (transition_id)"""
        ).fetchall()
    finally:
        con.close()
    cases = {}
    for case_id, pipeline, dataset, config, applicable, split in rows:
        path = results / "compare" / f"{case_id}.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        outcome, detail = _case_outcome(report, applicable)
        cases[case_id] = {
            "collection": json.loads(config)["collection"],
            "split": split,
            "dataset": dataset,
            "pipeline": pipeline,
            "outcome": outcome,
            "detail": detail,
        }
    return cases


def _adjudications() -> dict[str, str]:
    """Mechanism id -> sorted case ids, latest adjudication per case and mechanism family."""
    con = duckdb.connect(str(PRIMARY / "ledger.duckdb"), read_only=True)
    try:
        rows = con.execute("SELECT case_id, mechanism FROM adjudications ORDER BY decided_at").fetchall()
    finally:
        con.close()
    found: dict[str, set] = defaultdict(set)
    for case_id, text in rows:
        for mid, pattern, _ in MECHANISMS:
            if re.search(pattern, text):
                found[mid].add(case_id)
                break
    return {mid: sorted(found[mid]) for mid, _, _ in MECHANISMS}


def _fmt(x: float, digits: int = 2) -> str:
    """LaTeX math for a small number: 0, or m x 10^e."""
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.{digits - 1}e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def _write(name: str, text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text, encoding="utf-8")


def main() -> None:
    primary = _load(PRIMARY)
    hosts: dict[str, dict] = defaultdict(dict)
    for job, label in HOSTS.items():
        for cid, v in _load(ROOT / "results" / "hosts" / job / "results").items():
            assert cid not in hosts[label], f"{cid} run twice on host {label}"
            hosts[label][cid] = v
    macros: dict[str, str] = {}

    # ---- Eligibility: environments and run attempts on the primary host
    con = duckdb.connect(str(PRIMARY / "ledger.duckdb"), read_only=True)
    try:
        envs = dict(con.execute("SELECT status, count(*) FROM envs GROUP BY status").fetchall())
        runs = dict(con.execute("SELECT status, count(*) FROM runs GROUP BY status").fetchall())
        n_transitions = con.execute("SELECT count(*) FROM transitions").fetchone()[0]
    finally:
        con.close()
    macros.update(
        nEnvs=str(sum(envs.values())), nEnvsOk=str(envs.get("ok", 0)),
        nEnvsResolveFailed=str(envs.get("resolve_failed", 0)), nTransitions=str(n_transitions),
        nRunsOk=f"{runs.get('ok', 0):,}".replace(",", "{,}"), nRunsCrash=str(runs.get("crash", 0)),
        nRunsTimeout=str(runs.get("timeout", 0)),
    )

    # ---- RQ1: outcomes by collection and split (primary host, MovieLens 1M primary split)
    main_cases = {k: v for k, v in primary.items() if v["dataset"] == "ml1m"}
    table = defaultdict(Counter)
    for v in main_cases.values():
        table[(v["collection"], v["split"])][v["outcome"]] += 1
    labels = {"systematic_single": "Single dependency", "systematic_snapshot": "Snapshot"}
    lines = []
    total = Counter()
    for coll in ("systematic_single", "systematic_snapshot"):
        for split in ("dev", "heldout"):
            c = table[(coll, split)]
            n = sum(c.values())
            total.update(c)
            lines.append(
                f"{labels[coll]} & {'development' if split == 'dev' else 'held-out'} & {n} & "
                + " & ".join(str(c[o]) for o in OUTCOMES) + r" \\"
            )
    n_all = sum(total.values())
    lines.append(r"\midrule")
    lines.append(f"Total & & {n_all} & " + " & ".join(str(total[o]) for o in OUTCOMES) + r" \\")
    _write("tab_rq1.tex", "\n".join(lines) + "\n")
    macros.update(
        nCases=str(n_all),
        nIdentical=str(total["identical"]),
        nChanged=str(total["changed"]),
        nTransitionsChanged=str(len({cid.split("-", 1)[0] for cid, case in main_cases.items()
                                     if case["outcome"] == "changed"})),
        nRegression=str(total["regression"]),
        nContract=str(total["contract"]),
        nUncertain=str(total["uncertain"]),
        nFailure=str(total["failure"]),
        nNotApplicable=str(total["not_applicable"]),
        nHeldoutChanged=str(sum(table[(c, "heldout")]["changed"] for c in labels)),
        nDevChanged=str(sum(table[(c, "dev")]["changed"] for c in labels)),
    )

    # ---- RQ2: every changed case and mode, with activity-group intervals
    changed_rows, group_bound = [], 0.0
    for cid in sorted(k for k, v in main_cases.items() if v["outcome"] == "changed"):
        report = json.loads((PRIMARY / "compare" / f"{cid}.json").read_text(encoding="utf-8"))
        for mode, m in sorted(report["modes"].items()):
            ab = m["AB"]
            stages = sorted({p["changed_stages"][0] for p in ab.get("runs", []) if p.get("changed_stages")})
            if not stages:
                continue
            q = ab["quality"]["ndcg10"]
            p = q["primary"]
            groups = q.get("activity_groups", {})
            bound = max((max(abs(g["primary"]["low"]), abs(g["primary"]["high"])) for g in groups.values()), default=0.0)
            group_bound = max(group_bound, bound)
            beh = ab["behavior"]
            split = main_cases[cid]["split"]
            changed_rows.append(
                f"\\mbox{{\\texttt{{{cid}}}}} & {'dev' if split == 'dev' else 'held-out'} & {'retrain' if mode == 'retrain' else 'infer'} & "
                f"{', '.join(stages)} & {100 * beh['users_changed_frac']:.2f} & "
                f"${_fmt(p['estimate'])}\\ [{_fmt(p['low'])}, {_fmt(p['high'])}]$ & ${_fmt(bound)}$ \\\\"
            )
    _write("tab_changed.tex", "\n".join(changed_rows) + "\n")
    macros["groupBound"] = f"${_fmt(group_bound)}$"

    # ---- RQ2: mechanisms
    mech = _adjudications()
    # Scale runs replay a named primary case and carry no separate ledger adjudication.
    scale_replays = {cid.removesuffix("-ml20m"): cid for cid, case in primary.items()
                     if case["dataset"] == "ml20m" and case["outcome"] == "changed"}
    assert scale_replays.keys() <= {cid for mid in ("M1", "M2", "M3", "M4") for cid in mech[mid]}
    rows = []
    for mid, _, title in MECHANISMS:
        ids = mech[mid]
        ids_main = [cid for cid in ids if cid in main_cases]
        ids_ext = [cid for cid in ids if cid in primary and cid not in main_cases]
        ids_ext.extend(scale_replays[cid] for cid in ids_main if cid in scale_replays)
        ids_ext.extend(cid for cid in ids if cid in hosts["GPU"])
        details = [main_cases[cid]["detail"] for cid in ids_main if main_cases[cid]["detail"]]
        details.extend(hosts["GPU"][cid]["detail"] for cid in ids_ext if cid in hosts["GPU"])
        frac = max((d["changed_frac"] for d in details), default=0.0)
        loss = max((abs(d["max_loss"]) for d in details), default=0.0)
        stages = sorted({s for d in details for s in d["stages"]})
        effect = f"{len(ids)} SAR cells excluded" if mid == "M8" else "visible failure" if mid in ("M6", "M7") else (
            f"{100 * frac:.1f}\\% lists; $|L|\\le{_fmt(loss)}$" if frac else f"no top-10 change; $|L|\\le{_fmt(loss)}$")
        rows.append(f"{mid} & {title} & {len(ids_main)} & {len(ids_ext)} & {', '.join(stages) or '--'} & {effect} \\\\")
        macros["nCases" + mid.replace("M", "Mech").translate(str.maketrans({"1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight"}))] = str(len(ids))
    _write("tab_mechanisms.tex", "\n".join(rows) + "\n")

    # ---- Hardware: the mechanism cases across hosts
    probe = [
        ("M1", "N1-P1-float32"), ("M1", "Q2-P1-float32"),
        ("M2", "N2-P2-als"), ("M2", "Q3-P2-als"),
        ("M3", "N5-P6-ease"), ("M3", "N6-P6-ease"),
        ("M4", "N1-P2-bm25"), ("M4", "Q2-P2-bm25"),
        ("ctl", "N4-P6-ease"), ("ctl", "S5-P1-float32"),
    ]
    mark = {"changed": "change", "identical": "none", "failure": "fail", "not_applicable": "n/a"}
    hw = []
    for mid, cid in probe:
        cells = []
        for label, cs in (("A", primary), ("B", hosts["B"]), ("C", hosts["C"])):
            v = cs.get(cid)
            cells.append(mark.get(v["outcome"], v["outcome"]) if v else "--")
        hw.append(f"{mid} & \\texttt{{{cid}}} & " + " & ".join(cells) + r" \\")
    _write("tab_hardware.tex", "\n".join(hw) + "\n")
    for label, cs in hosts.items():
        n = len(cs)
        bad = sum(v["outcome"] in ("regression", "contract", "uncertain") for v in cs.values())
        macros[f"nHost{label.replace('GPU', 'Gpu')}"] = str(n)
        macros[f"nHost{label.replace('GPU', 'Gpu')}Bad"] = str(bad)

    # ---- MovieLens 20M
    ml20 = {k: v for k, v in primary.items() if v["dataset"] == "ml20m"}
    macros["nMLtwentyCases"] = str(len(ml20))
    macros["nMLtwentyChanged"] = str(sum(v["outcome"] == "changed" for v in ml20.values()))
    macros["nMLtwentyBad"] = str(sum(v["outcome"] in ("regression", "contract", "uncertain") for v in ml20.values()))
    m1 = ml20.get("N1-P1-float32-ml20m", {}).get("detail", {})
    macros["lossMOneMLtwenty"] = f"{m1.get('max_loss', 0):.5f}"
    m2 = primary.get("N2-P2-als", {}).get("detail", {})
    macros["fracMTwo"] = f"{100 * m2.get('changed_frac', 0):.1f}"

    # RQ3 (synthetic controls, analysis version 2) is generated by make_extras.py.

    _write("numbers.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
    print(json.dumps({"rq1_total": dict(total), "cases": n_all, "macros": macros}, indent=1))


if __name__ == "__main__":
    main()
