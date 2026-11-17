"""Run the four checks declared in protocol/cloud_reruns.md on the downloaded reruns.

    uv run --locked python paper/scripts/verify_cloud_reruns.py [--jobs gpu-t4-2 ...] [--skip-evaluation]

1. Integrity: every arrays/<case>.tar matches its recorded SHA-256, and so does every file in it.
2. Reproduction: stage hashes of every run equal those of its counterpart in the first Job(s).
3. Re-evaluation: the pinned local evaluator, run on the downloaded arrays, reproduces the
   per-user metrics the Job wrote.
4. Outcomes: every case has the same comparison outcome as in the first Job(s).

Writes results/hosts/rerun_verification.json. Extracted arrays go to a scratch folder next
to each tar and are removed after the case is checked. Nothing in the ledgers is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_tables import ROOT, _load  # noqa: E402

from silentrank.orchestrator import evaluation  # noqa: E402

HOSTS = ROOT / "results" / "hosts"
RERUNS = {"second-cpu-2": ["second-cpu-1", "epyc-log1p-1"], "intel-cpu-2": ["intel-cpu-1"], "gpu-t4-2": ["gpu-t4-1"]}


def _stage_hashes(job: str) -> tuple[dict, dict]:
    con = duckdb.connect(str(HOSTS / job / "results" / "ledger.duckdb"), read_only=True)
    try:
        rows = con.execute("""SELECT r.run_id, r.case_id, r.env_id, r.side, r.mode, r.seed, r.repeat, s.stage, s.canonical_sha256
                              FROM runs r JOIN stage_hashes s USING (run_id) WHERE r.status = 'ok'""").fetchall()
        info = {"runs": dict(con.execute("SELECT status, count(*) FROM runs GROUP BY status").fetchall()),
                "cpu_models": sorted({r[0] for r in con.execute("SELECT cpu_model FROM hosts").fetchall()}),
                "gpus": sorted({str(r[0]) for r in con.execute("SELECT gpu FROM hosts").fetchall()})}
    finally:
        con.close()
    out = {}
    for run_id, case, env, side, mode, seed, repeat, stage, digest in rows:
        inputs = json.loads((HOSTS / job / "results" / "runs" / run_id / "meta.json").read_text(encoding="utf-8"))["inputs"]
        key = (case, env, side, mode, seed, inputs.get("split"), inputs.get("neighbors"), inputs.get("factors"),
               inputs.get("lamb"), repeat)
        out.setdefault(key, {})[stage] = digest
    return out, info


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset() -> Path:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    return ROOT / "data" / manifest["prepared"]["filename"]


def check_job(job: str, firsts: list[str], evaluate: bool) -> dict:
    new, info = _stage_hashes(job)
    old, first_info = {}, []
    for first in firsts:
        hashes, i = _stage_hashes(first)
        old.update(hashes)
        first_info.append(i)
    same_cpu = all(i["cpu_models"] == info["cpu_models"] for i in first_info)
    common = sorted(set(new) & set(old))
    stage_total, stage_equal, differing = 0, 0, []
    for key in common:
        for stage in sorted(set(new[key]) | set(old[key])):
            stage_total += 1
            if new[key].get(stage) == old[key].get(stage):
                stage_equal += 1
            else:
                differing.append([key[0], key[2], key[3], key[4], stage])

    cases_new = _load(HOSTS / job / "results")
    cases_old = {}
    for first in firsts:
        cases_old.update(_load(HOSTS / first / "results"))
    outcome_diff = {c: [v["outcome"], cases_old.get(c, {}).get("outcome")] for c, v in cases_new.items()
                    if v["outcome"] != cases_old.get(c, {}).get("outcome")}

    arrays = HOSTS / job / "arrays"
    dataset = _dataset()
    tars = files = bad_files = evaluated = metric_mismatch = 0
    bad_tars, total_bytes, device_files = [], 0, 0
    for manifest_path in sorted(arrays.glob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tar_path = arrays / f"{manifest['case']}.tar"
        tars += 1
        total_bytes += tar_path.stat().st_size
        if _sha256(tar_path) != manifest["tar_sha256"] or tar_path.stat().st_size != manifest["tar_bytes"]:
            bad_tars.append(manifest["case"])
            continue
        scratch = arrays / f"_extract_{manifest['case']}"
        if scratch.exists():
            shutil.rmtree(scratch)
        with tarfile.open(tar_path) as archive:
            archive.extractall(scratch, filter="data")
        for run_id, listing in manifest["files"].items():
            for relative, digest in listing.items():
                files += 1
                bad_files += _sha256(scratch / run_id / relative) != digest
            device_files += (scratch / run_id / "device.json").is_file()
            meta = json.loads((scratch / run_id / "meta.json").read_text(encoding="utf-8"))
            job_metrics = HOSTS / job / "results" / "metrics" / f"{run_id}.parquet"
            if not evaluate or meta.get("status") != "ok" or not job_metrics.is_file():
                continue
            local = scratch / f"{run_id}.parquet"
            evaluation.evaluate(scratch / run_id, dataset, meta["inputs"]["split"], local)
            evaluated += 1
            metric_mismatch += not pq.read_table(local).equals(pq.read_table(job_metrics))
        shutil.rmtree(scratch)
        print(f"{job} {manifest['case']}: ok", flush=True)

    return {
        "job": job, "first_jobs": firsts, "cpu_models": info["cpu_models"], "gpus": info["gpus"], "same_cpu_model": same_cpu,
        "runs": info["runs"], "runs_with_counterpart": len(common), "runs_without_counterpart": len(set(new) ^ set(old)),
        "stage_hashes_compared": stage_total, "stage_hashes_equal": stage_equal, "stage_differences": differing,
        "cases": len(cases_new), "outcomes": dict(Counter(v["outcome"] for v in cases_new.values())),
        "outcome_differences": outcome_diff,
        "archives": tars, "archive_bytes": total_bytes, "archives_failing_checksum": bad_tars,
        "files_checked": files, "files_failing_checksum": bad_files, "runs_with_device_record": device_files,
        "runs_reevaluated": evaluated, "runs_with_metric_mismatch": metric_mismatch,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", nargs="*", default=list(RERUNS))
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    target = HOSTS / "rerun_verification.json"
    report = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
    for name in args.jobs:
        report[name] = check_job(name, RERUNS[name], not args.skip_evaluation)
        target.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report[name].items() if k not in ("stage_differences",)}, indent=1))
