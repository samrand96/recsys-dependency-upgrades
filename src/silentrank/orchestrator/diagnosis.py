"""Replay and adjudicate isolated native top-K changes in the pilot."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from silentrank import paths
from silentrank.orchestrator import envs, ledger, pilot, runner
from silentrank.worker.stages import Stages

INPUTS = {"scores.npy": "--scores-sha256", "seen_mask.npy": "--seen-mask-sha256", "item_ids.npy": "--item-ids-sha256"}
RANKS = {10: "topk_native10.npy", 20: "topk_native.npy"}
NUMPY_NOTES = "https://numpy.org/doc/2.0/release/2.0.0-notes.html#minor-changes-in-behavior-of-sorting-functions"


def _original_runs(case_id: str, comparison: dict, config: dict) -> tuple[dict, dict]:
    pair, = comparison["runs"]
    comparison_id = comparison["comparison_id"]
    run_ids = (pair["old"], pair["new"])
    with ledger.reader() as con:
        row = con.execute("SELECT run_a, run_b FROM comparisons WHERE comparison_id = ? AND case_id = ? AND kind = 'AB'",
                          [comparison_id, case_id]).fetchone()
        if row != run_ids:
            raise ValueError("inference comparison disagrees with the ledger")
        cursor = con.execute(
            "SELECT run_id, side, env_id, mode, status, host_id, artifact_dir FROM runs WHERE case_id = ? AND run_id IN (?, ?)",
            [case_id, *run_ids],
        )
        keys = [column[0] for column in cursor.description]
        runs = {row[0]: dict(zip(keys, row)) for row in cursor.fetchall()}
    if runs.keys() != set(run_ids):
        raise ValueError("original runs are missing from the ledger")

    _, current_source = pilot._source_version("ref_itemknn")
    for side, run_id in zip(("old", "new"), run_ids):
        run = runs[run_id]
        if (run["side"], run["mode"], run["status"], run["env_id"]) != (side, "inference", "ok", config[side]):
            raise ValueError(f"{run_id} is not a successful original {side} inference run")
        directory = Path(run["artifact_dir"])
        inputs = pilot._read_json(directory / "meta.json")["inputs"]
        host = pilot._read_json(directory / "host.json")
        if not run["host_id"] or envs.host_id_for(host) != run["host_id"]:
            raise ValueError(f"{run_id} is missing matching original host evidence")
        if inputs["worker_sha256"] != current_source or inputs["lock_sha256"] != envs.lock_sha256(run["env_id"]):
            raise ValueError(f"current worker or lock differs from the original {side} inference run")
        run.update(directory=directory, inputs=inputs, host=host, stages=dict(pilot._stage_values(run_id)))
    old, new = (runs[run_id] for run_id in run_ids)
    if old["host_id"] != new["host_id"] or old["inputs"]["image"] != new["inputs"]["image"]:
        raise ValueError("S7 diagnosis requires original runs on one host and runner image")
    stage_names = sorted(old["stages"].keys() | new["stages"].keys())
    changed = [stage for stage in stage_names if old["stages"].get(stage) != new["stages"].get(stage)]
    required = {f"S{index}" for index in range(9)} | {"S7c"}
    if (any(required - run["stages"].keys() for run in (old, new)) or changed != pair["changed_stages"]
            or not changed or changed[0] != "S7" or "S7c" in changed):
        raise ValueError("the original ledger does not show S7 as the first changed stage with unchanged canonical ranks")
    return old, new


def _original_arrays(run: dict) -> tuple[dict, dict]:
    directory = run["directory"]
    names = (*INPUTS, "user_ids.npy", *RANKS.values())
    arrays = {name: np.load(directory / name, mmap_mode="r", allow_pickle=False) for name in names}
    stages = Stages(directory)
    stages.record("S4", {"user_ids": arrays["user_ids.npy"], "item_ids": arrays["item_ids.npy"]}, "")
    stages.record("S5", {"scores": arrays["scores.npy"]}, "")
    stages.record("S6", {"seen_mask": arrays["seen_mask.npy"]}, "")
    stages.record("S7", {f"items_{k}": arrays[name] for k, name in RANKS.items()}, "")
    if any(run["stages"].get(stage) != entry["canonical_sha256"] for stage, entry in stages.rows.items()):
        raise ValueError(f"original artifacts no longer match recorded S4–S7 hashes for {run['run_id']}")

    scores = arrays["scores.npy"]
    catalog = arrays["item_ids.npy"]
    rank_scores = {}
    for name in RANKS.values():
        ranks = arrays[name]
        rows, columns = np.nonzero(ranks != -1)
        values = np.full(ranks.shape, -np.inf, dtype=scores.dtype)
        values[rows, columns] = scores[rows, np.searchsorted(catalog, ranks[rows, columns])]
        rank_scores[name] = values
    hashes = {name: pilot._digest(directory / name) for name in (*INPUTS, *RANKS.values())}
    return hashes, rank_scores


def _output_text(value: str | bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""


def _replay(case_id: str, comparison_id: str, directory: Path, phase: str, environment: dict,
            source: Path, input_hashes: dict, expected_ranks: dict) -> dict:
    output = directory / phase
    output.mkdir()
    replay_id = uuid4().hex
    env_id = environment["env_id"]
    started_at, start = pilot._now(), time.perf_counter()
    status, detail, host_id, stdout, stderr = "crash", None, None, "", ""
    py = f"/venvs/{env_id}/bin/python"
    command = [py, "-m", "silentrank.worker.replay_rank", "--input-dir", "/replay-input", "--output-dir", "/replay-output"]
    for name, option in INPUTS.items():
        command.extend([option, input_hashes[name]])
    wrapped = ["sh", "-c", '"$1" /repo/src/silentrank/worker/probe.py > /replay-output/host.json && exec "$@"',
               "silentrank", *command]
    try:
        result = runner.run(wrapped, timeout=180, image=environment["inputs"]["image"],
                            mounts=((source, "/replay-input", True), (output, "/replay-output", False)),
                            venvs_read_only=True)
        stdout, stderr = result.stdout, result.stderr
        if (output / "host.json").is_file():
            host = pilot._read_json(output / "host.json")
            host_id = envs.host_id_for(host)
        if result.returncode:
            status = "timeout" if result.returncode in (124, 137) else "crash"
            detail = f"runner exited {result.returncode}"
        elif host_id != environment["host_id"] or host["numpy"]["version"] != environment["host"]["numpy"]["version"]:
            status, detail = "mismatch", "replay host or NumPy version differs from the original inference run"
        else:
            metadata = pilot._read_json(output / "replay_meta.json")
            rankings = {name: pilot._digest(output / name) for name in RANKS.values()}
            if metadata["input_sha256"] != input_hashes:
                status, detail = "mismatch", "replay input hashes differ from the original upstream artifacts"
            elif metadata["output_sha256"] != rankings or rankings != expected_ranks:
                status, detail = "mismatch", "replayed native lists differ from the original at K=10 or K=20"
            else:
                status = "ok"
    except subprocess.TimeoutExpired as exc:
        status, detail = "timeout", str(exc)
        stdout, stderr = _output_text(exc.stdout), _output_text(exc.stderr)
    except (runner.DockerUnavailable, OSError, ValueError, KeyError) as exc:
        detail = f"{type(exc).__name__}: {exc}"
    (output / "stdout.log").write_text(stdout, encoding="utf-8")
    (output / "stderr.log").write_text(stderr, encoding="utf-8")
    record = dict(replay_id=replay_id, case_id=case_id, comparison_id=comparison_id, phase=phase, env_id=env_id,
                  host_id=host_id, status=status, started_at=started_at, wall_s=time.perf_counter() - start,
                  artifact_dir=str(output), detail=detail)
    pilot._write_json(output / "attempt.json", record)
    with ledger.writer() as con:
        con.execute(f"INSERT INTO diagnostic_replays ({', '.join(record)}) VALUES ({', '.join('?' for _ in record)})",
                    list(record.values()))
    return record


def diagnose_case(case_id: str) -> dict:
    """Compare old, upgraded, rolled-back and reapplied native top-K on frozen inputs."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
        raise ValueError("invalid pilot case ID")
    with ledger.reader() as con:
        row = con.execute(
            "SELECT c.pipeline, c.config_json, t.resolved_diff_json FROM cases c "
            "JOIN transitions t USING (transition_id) WHERE c.case_id = ?", [case_id],
        ).fetchone()
    if row is None or row[0] != "ref_itemknn":
        raise ValueError("S7 diagnosis currently supports only reference item-kNN cases")
    config, recorded_diff = json.loads(row[1]), json.loads(row[2] or "null")
    comparison_file = paths.results_dir() / "compare" / f"{case_id}.json"
    if not comparison_file.is_file():
        raise ValueError(f"compare {case_id} before diagnosis")
    report = pilot._read_json(comparison_file)
    ab = report["modes"].get("inference", {}).get("AB", {})
    if (report["case"] != case_id or ab.get("status") != "ok" or ab.get("contract_failure")
            or not ab.get("behavior", {}).get("canonical_equal") or len(ab.get("runs", [])) != 1
            or ab["runs"][0].get("changed_stages", [])[:1] != ["S7"]):
        raise ValueError("diagnosis requires a successful inference comparison first differing at S7 with equal canonical ranks")

    old, new = _original_runs(case_id, ab, config)
    resolved_diff = envs._resolved_diff(old["env_id"], new["env_id"])
    if resolved_diff != recorded_diff:
        raise ValueError("replay lockfile diff disagrees with the screened transition")
    old_hashes, old_scores = _original_arrays(old)
    new_hashes, new_scores = _original_arrays(new)
    input_hashes = {name: old_hashes[name] for name in INPUTS}
    if any(new_hashes[name] != digest for name, digest in input_hashes.items()):
        raise ValueError("S7 diagnosis requires identical upstream score, seen-mask and item-map bytes")
    ties_only = all(np.array_equal(old_scores[name], new_scores[name]) for name in RANKS.values())

    comparison_id = ab["comparison_id"]
    replay_entry = Path(__file__).resolve().parents[1] / "worker" / "replay_rank.py"
    replay_source = pilot._digest(replay_entry)
    commit, worker_source = pilot._source_version("ref_itemknn")
    directory = paths.results_dir() / "diagnosis" / case_id / uuid4().hex
    directory.mkdir(parents=True)
    phases = (("old", old, old_hashes), ("new", new, new_hashes),
              ("rollback", old, old_hashes), ("reapply", new, new_hashes))
    replays = [
        _replay(case_id, comparison_id, directory, phase, run, old["directory"], input_hashes,
                {name: hashes[name] for name in RANKS.values()})
        for phase, run, hashes in phases
    ]
    result = dict(case=case_id, comparison_id=comparison_id, status="unresolved", stage="S7",
                  changed_packages=list(resolved_diff), ties_only=ties_only, replays=replays,
                  original_runs={run["side"]: run["run_id"] for run in (old, new)}, input_sha256=input_hashes,
                  code_commit=commit, worker_sha256=worker_source, replay_script_sha256=replay_source,
                  artifact=str(directory / "diagnosis.json"))
    if pilot._digest(replay_entry) != replay_source or pilot._source_version("ref_itemknn")[1] != worker_source:
        result["reason"] = "worker or replay source changed during the replay batch"
    elif all(row["status"] == "ok" for row in replays):
        numpy_only = set(resolved_diff) == {"numpy"}
        documented = (numpy_only and ties_only and old["host"]["numpy"]["version"].startswith("1.26.")
                      and new["host"]["numpy"]["version"].startswith("2.0."))
        mechanism = "Native NumPy top-K tie selection" if numpy_only and ties_only else "Environment-dependent native top-K"
        cause_class = "intended_semantic_change" if documented else "unresolved"
        result.update(status="reproduced", mechanism=mechanism, cause_class=cause_class,
                      attribution="numpy" if numpy_only else "resolved_environment")
        if documented:
            result["source"] = NUMPY_NOTES
    pilot._write_json(Path(result["artifact"]), result)
    if result["status"] == "reproduced":
        with ledger.writer() as con:
            con.execute("INSERT INTO adjudications VALUES (?, ?, ?, ?, ?, ?)",
                        [case_id, result["mechanism"], result["cause_class"], result["artifact"],
                         "pilot S7 replay", pilot._now().date()])
    return result
