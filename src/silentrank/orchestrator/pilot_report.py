"""Summarize the Phase 1 ledger and mode-specific comparison artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from silentrank import paths
from silentrank.orchestrator import ledger


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _interval(interval: dict | None) -> str:
    if interval is None:
        return "—"
    return "{estimate:.6g} [{low:.6g}, {high:.6g}]".format(**interval)


def summarize_pilot() -> dict:
    """Write a case-level Markdown snapshot linked to the local evidence."""
    directory = paths.results_dir()
    with ledger.reader() as con:
        environments = dict(con.execute("SELECT status, count(*) FROM envs GROUP BY status ORDER BY status").fetchall())
        transitions = con.execute("SELECT collection, status, count(*) FROM transitions GROUP BY ALL ORDER BY ALL").fetchall()
        cases = con.execute("SELECT case_id, pipeline, config_json, applicable, dataset FROM cases ORDER BY dataset, case_id").fetchall()
        suite_rows = con.execute("SELECT env_id, package, status FROM testsuite ORDER BY env_id, package").fetchall()
        attempts = dict(con.execute("SELECT status, count(*) FROM runs GROUP BY status ORDER BY status").fetchall())
        timings = con.execute(
            """SELECT mode, count(*), median(wall_s), min(wall_s), max(wall_s), median(cpu_s), max(peak_rss_mb)
               FROM runs WHERE status = 'ok' GROUP BY mode ORDER BY mode"""
        ).fetchall()
        diagnoses = con.execute("SELECT case_id, mechanism, cause_class, evidence FROM adjudications ORDER BY case_id").fetchall()

    suites: dict[str, list[str]] = {}
    for env, package, status in suite_rows:
        suites.setdefault(env, []).append(f"{package} {status}")
    counts = Counter()
    lines = ["# Phase 1 pilot ledger summary", "", "## Eligibility and execution", ""]
    lines.append(f"- Environments by build and import status — {environments}.")
    for collection, status, count in transitions:
        lines.append(f"- {collection} transitions, {status} — {count}.")
    lines.append(f"- Run attempts by status — {attempts}. This includes validation tuning.")
    lines.append(f"- Recorded upstream suite outcomes — {dict(sorted(Counter(row[2] for row in suite_rows).items()))}.")
    lines += ["", "## Case comparisons", "",
              "NDCG@10 loss is old minus new. Intervals are 95%. Verdicts are listed for δ 0.002, 0.005 and 0.010.", "",
              "| Case | Collection | Dataset split | Mode | Outcome | Top-10 overlap | NDCG@10 loss [95% interval] | Verdicts | First changed stage |",
              "|---|---|---|---|---|---:|---:|---|---|"]
    env_pairs = []
    for case_id, pipeline, raw_config, applicable, dataset in cases:
        config = json.loads(raw_config)
        env_pairs.append((case_id, config["old"], config["new"]))
        comparison = directory / "compare" / f"{case_id}.json"
        if comparison.is_file():
            report = _read(comparison)
            modes = report["modes"]
            link = f"[{case_id}](compare/{comparison.name})"
        else:
            modes, link = {}, case_id
        expected = ("prep",) if pipeline == "prep_pandas" else ("inference", "retrain")
        for mode in expected:
            result = modes.get(mode, {})
            ab = result.get("AB", {})
            stages = sorted({pair["changed_stages"][0] for pair in ab.get("runs", []) if pair.get("changed_stages")})
            quality = ((ab.get("quality") or {}).get("ndcg10") or {}).get("primary")
            overlap = (ab.get("behavior") or {}).get("overlap10_mean")
            verdicts = [item["verdict"] for item in result.get("verdicts", [])]
            if applicable is False:
                outcome = "not_applicable"
            elif result.get("status") != "ok":
                outcome = result.get("status", "pending")
            elif ab.get("contract_failure"):
                outcome = "contract_failure"
            elif "harmful" in verdicts:
                outcome = "harmful at one or more δ"
            elif "inconclusive" in verdicts:
                outcome = "inconclusive"
            elif "improved" in verdicts:
                outcome = "improved at one or more δ"
            elif stages:
                outcome = "behavioral change"
            else:
                outcome = "identical checkpoints"
            if dataset == "ml1m" and mode == "inference" and result.get("AA", {}).get("status") == result.get("status") == "ok":
                counts["accepted_inference_controls"] += 1
            counts[f"{mode}:{outcome}" if dataset == "ml1m" else f"{'ml20m' if dataset == 'ml20m' else 'sensitivity'}:{mode}:{outcome}"] += 1
            lines.append("| " + " | ".join((link, config["collection"], dataset, mode, outcome,
                         f"{overlap:.6f}" if overlap is not None else "—", _interval(quality),
                         ", ".join(verdicts) or "—", ", ".join(stages) or "—")) + " |")

    lines += ["", "## Upstream suites by case", "",
              "A suite list contains recorded outcomes for that environment. See the ledger for exact commands and logs.", "",
              "| Case | Old | New |", "|---|---|---|"]
    for case, old, new in env_pairs:
        lines.append(f"| {case} | {', '.join(suites.get(old, [])) or 'pending'} | {', '.join(suites.get(new, [])) or 'pending'} |")

    lines += ["", "## Observed costs", "",
              "Wall time includes container launch and artifact evaluation. Some jobs shared the host with upstream screening.", "",
              "| Mode | Runs | Median wall s | Min–max wall s | Median worker CPU s | Max worker RSS MiB |",
              "|---|---:|---:|---:|---:|---:|"]
    for mode, n, median, low, high, cpu, rss in timings:
        if None not in (median, low, high, cpu, rss):
            lines.append(f"| {mode} | {n} | {median:.2f} | {low:.2f}–{high:.2f} | {cpu:.2f} | {rss:.1f} |")
    lines += ["", "## Confirmed replays", ""]
    for case, mechanism, cause, evidence in diagnoses:
        path = Path(evidence)
        target = path.relative_to(directory).as_posix() if path.is_relative_to(directory) else path.as_posix()
        lines.append(f"- [{case}]({target}) — {mechanism}, `{cause}`.")
    target = directory / "pilot_summary.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"artifact": str(target), "registered_cases": len(cases), "outcomes": dict(sorted(counts.items()))}
