"""Isolate reference item-kNN neighbor selection after an S3 retraining divergence."""

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
from silentrank.orchestrator.diagnosis import NUMPY_NOTES, _output_text
from silentrank.worker.stages import Stages


def _arrays(path: Path, expected: str, stage: str) -> dict:
    with np.load(path, allow_pickle=False) as bundle:
        arrays = dict(bundle.items())
    stages = Stages(path.parent)
    stages.record(stage, arrays, path.name)
    if stages.rows[stage]["canonical_sha256"] != expected:
        raise ValueError(f"{path} disagrees with its recorded {stage} hash")
    return arrays


def _original(case_id: str, seed: int) -> tuple[dict, dict, list[dict]]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
        raise ValueError("invalid pilot case ID")
    report = pilot._read_json(paths.results_dir() / "compare" / f"{case_id}.json")
    comparison = report["modes"].get("retrain", {}).get("AB", {})
    if report["case"] != case_id or comparison.get("status") != "ok":
        raise ValueError("neighbor diagnosis requires a saved successful retraining comparison")
    pair = next((row for row in comparison["runs"] if row["seed"] == seed), None)
    if pair is None or pair["changed_stages"][:1] != ["S3"]:
        raise ValueError(f"retraining seed {seed} has no first divergence at S3")

    with ledger.reader() as con:
        row = con.execute(
            "SELECT c.pipeline, c.config_json, t.resolved_diff_json FROM cases c JOIN transitions t USING (transition_id) "
            "WHERE c.case_id = ?", [case_id],
        ).fetchone()
        index_row = con.execute(
            "SELECT run_a, run_b FROM comparisons WHERE comparison_id = ? AND case_id = ? AND kind = 'AB'",
            [comparison["comparison_id"], case_id],
        ).fetchone()
    first = comparison["runs"][0]
    if row is None or row[0] != "ref_itemknn" or index_row != (first["old"], first["new"]):
        raise ValueError("retraining comparison is not a recorded reference item-kNN A/B pair")
    config = json.loads(row[1])
    _, current_source = pilot._source_version("ref_itemknn")
    run_rows = {run["run_id"]: run for run in pilot._run_rows(case_id)}
    originals = []
    for side in ("old", "new"):
        run = run_rows.get(pair[side])
        if run is None or (run["side"], run["mode"], run["status"], run["seed"]) != (side, "retrain", "ok", seed):
            raise ValueError(f"missing successful original {side} retraining run for seed {seed}")
        directory = Path(run["artifact_dir"])
        inputs = pilot._read_json(directory / "meta.json")["inputs"]
        host = pilot._read_json(directory / "host.json")
        if (inputs["worker_sha256"] != current_source or inputs["env_id"] != config[side]
                or inputs["lock_sha256"] != envs.lock_sha256(config[side]) or inputs["split"] != "test"
                or not run["host_id"] or envs.host_id_for(host) != run["host_id"]):
            raise ValueError(f"original {side} run has mismatched worker, environment or host evidence")
        stages = dict(pilot._stage_values(run["run_id"]))
        if not {"S0", "S1", "S2", "S3"} <= stages.keys():
            raise ValueError(f"original {side} run is missing stage evidence")
        stage2 = directory / "stages" / "S2.npz"
        _arrays(stage2, stages["S2"], "S2")
        model = _arrays(directory / "model.npz", stages["S3"], "S3")
        originals.append({**run, "env_id": config[side], "inputs": inputs, "host": host,
                          "stage2": stage2, "stages": stages, "model": model})

    old, new = originals
    common = ("image", "neighbors", "dataset_sha256")
    if (old["host_id"] != new["host_id"] or any(old["inputs"][key] != new["inputs"][key] for key in common)
            or any(old["stages"][stage] != new["stages"][stage] for stage in ("S0", "S1", "S2"))
            or old["stages"]["S3"] == new["stages"]["S3"]
            or pilot._digest(old["stage2"]) != pilot._digest(new["stage2"])):
        raise ValueError("neighbor diagnosis requires matching environment controls and exact S2 input bytes")
    diff = envs._resolved_diff(old["env_id"], new["env_id"])
    if not diff or diff != json.loads(row[2] or "null"):
        raise ValueError("lockfile diff disagrees with the screened transition")
    return comparison, diff, originals


def _attempt(case: str, comparison: str, directory: Path, phase: str, run: dict, source: Path,
             input_hash: str, *, capture: bool = False) -> dict:
    output = directory / phase
    output.mkdir()
    started_at, start = pilot._now(), time.perf_counter()
    metadata, host_id, detail, status, stdout, stderr = {}, None, None, "crash", "", ""
    python = f"/venvs/{run['env_id']}/bin/python"
    command = [python, "-m", "silentrank.worker.replay_neighbors", "capture" if capture else "prune",
               "--input", "/neighbor-input.npz", "--input-sha256", input_hash, "--output-dir", "/neighbor-output"]
    if not capture:
        command += ["--neighbors", str(run["inputs"]["neighbors"]),
                    "--training-input-sha256", run["inputs"]["dataset_sha256"]]
    wrapped = ["sh", "-c", '"$1" /repo/src/silentrank/worker/probe.py > /neighbor-output/host.json && exec "$@"',
               "silentrank", *command]
    try:
        result = runner.run(wrapped, timeout=180, image=run["inputs"]["image"],
                            mounts=((source, "/neighbor-input.npz", True), (output, "/neighbor-output", False)),
                            venvs_read_only=True)
        stdout, stderr = result.stdout, result.stderr
        host_path = output / "host.json"
        host = pilot._read_json(host_path) if host_path.is_file() else {}
        host_id = envs.host_id_for(host) if host else None
        if result.returncode:
            status = "timeout" if result.returncode in (124, 137) else "crash"
            detail = f"runner exited {result.returncode}"
        elif host_id != run["host_id"] or host["numpy"]["version"] != run["host"]["numpy"]["version"]:
            status, detail = "mismatch", "replay host or NumPy version differs from original evidence"
        else:
            metadata = pilot._read_json(output / "replay_meta.json")
            artifact = output / ("weights.npz" if capture else "model.npz")
            if metadata["input_sha256"] != input_hash or metadata["output_sha256"] != pilot._digest(artifact):
                status, detail = "mismatch", "replay input or output bytes disagree with the replay manifest"
            elif not capture and metadata["hashes"]["S3"] != run["stages"]["S3"]:
                status, detail = "mismatch", "replayed model differs from the original S3 hash"
            else:
                key = "encounter_order" if capture else "S3"
                _arrays(artifact, metadata["hashes"][key], key)
                status = "ok"
    except subprocess.TimeoutExpired as exc:
        status, detail = "timeout", str(exc)
        stdout, stderr = _output_text(exc.stdout), _output_text(exc.stderr)
    except (runner.DockerUnavailable, OSError, ValueError, KeyError) as exc:
        detail = f"{type(exc).__name__}: {exc}"
    (output / "stdout.log").write_text(stdout, encoding="utf-8")
    (output / "stderr.log").write_text(stderr, encoding="utf-8")
    record = dict(replay_id=uuid4().hex, case_id=case, comparison_id=comparison, phase=f"S3:{phase}",
                  env_id=run["env_id"], host_id=host_id, status=status, started_at=started_at,
                  wall_s=time.perf_counter() - start, artifact_dir=str(output), detail=detail)
    pilot._write_json(output / "attempt.json", record)
    with ledger.writer() as con:
        con.execute(f"INSERT INTO diagnostic_replays ({', '.join(record)}) VALUES ({', '.join('?' for _ in record)})",
                    list(record.values()))
    return {**record, "metadata": metadata}


def _row(model: dict, index: int) -> tuple[np.ndarray, np.ndarray]:
    start, stop = model["indptr"][index:index + 2]
    return model["indices"][start:stop], model["data"][start:stop]


def _changed_rows(weights: dict, old: dict, new: dict) -> list[dict]:
    differences = []
    for row in range(int(weights["shape"][0])):
        old_ids, old_weights = _row(old, row)
        new_ids, new_weights = _row(new, row)
        if np.array_equal(old_ids, new_ids) and np.array_equal(old_weights, new_weights):
            continue
        removed = old_weights[~np.isin(old_ids, new_ids)]
        added = new_weights[~np.isin(new_ids, old_ids)]
        shared_old = old_weights[np.isin(old_ids, new_ids)]
        shared_new = new_weights[np.isin(new_ids, old_ids)]
        tied = bool(removed.size and np.array_equal(np.sort(removed), np.sort(added))
                    and np.array_equal(shared_old, shared_new))
        _, values = _row(weights, row)
        threshold = float(np.partition(values, values.size - len(old_ids))[values.size - len(old_ids)]) if len(old_ids) else None
        boundary = bool(tied and threshold is not None and np.all(removed == threshold) and np.all(added == threshold))
        differences.append({"target_index": row, "removed_count": int(removed.size), "added_count": int(added.size),
                            "exact_ties_only": tied, "all_swaps_at_pruning_boundary": boundary})
    return differences


def diagnose_neighbors(case_id: str, *, seed: int = 17) -> dict:
    """Reconstruct pre-pruning weights twice, then replay selection on one frozen artifact."""
    comparison, diff, (old, new) = _original(case_id, seed)
    comparison_id = comparison["comparison_id"]
    directory = paths.results_dir() / "diagnosis" / case_id / uuid4().hex
    directory.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "worker" / "replay_neighbors.py"
    script_hash = pilot._digest(source)
    commit, worker_hash = pilot._source_version("ref_itemknn")
    input_hash = pilot._digest(old["stage2"])
    attempts = [_attempt(case_id, comparison_id, directory, f"capture_{side}", run, old["stage2"], input_hash,
                         capture=True) for side, run in (("old", old), ("new", new))]
    result = dict(case=case_id, seed=seed, comparison_id=comparison_id, status="unresolved", stage="S3",
                  changed_packages=list(diff), attempts=attempts, original_runs={"old": old["run_id"], "new": new["run_id"]},
                  s2_sha256=input_hash, code_commit=commit, worker_sha256=worker_hash,
                  replay_script_sha256=script_hash, artifact=str(directory / "diagnosis.json"))
    if any(attempt["status"] != "ok" for attempt in attempts):
        result["reason"] = "failed to reconstruct pre-pruning weights in both original environments"
    else:
        old_meta, new_meta = (attempt["metadata"] for attempt in attempts)
        result["canonical_weights_equal"] = old_meta["hashes"]["canonical"] == new_meta["hashes"]["canonical"]
        result["encounter_order_equal"] = old_meta["hashes"]["encounter_order"] == new_meta["hashes"]["encounter_order"]
        if not result["canonical_weights_equal"] or not result["encounter_order_equal"]:
            result["reason"] = "original environments differ before pruning"
        else:
            frozen = directory / "capture_old" / "weights.npz"
            result["frozen_weights_sha256"] = old_meta["output_sha256"]
            phases = (("old", old), ("new", new), ("rollback", old), ("reapply", new))
            attempts.extend(_attempt(case_id, comparison_id, directory, phase, run, frozen, old_meta["output_sha256"])
                            for phase, run in phases)
            if any(attempt["status"] != "ok" for attempt in attempts):
                result["reason"] = "at least one frozen-weight pruning replay did not reproduce its original model"
            elif pilot._digest(source) != script_hash or pilot._source_version("ref_itemknn")[1] != worker_hash:
                result["reason"] = "worker or replay source changed during diagnosis"
            else:
                weights = _arrays(frozen, old_meta["hashes"]["encounter_order"], "encounter_order")
                changed = _changed_rows(weights, old["model"], new["model"])
                ties_only = bool(changed) and all(row["all_swaps_at_pruning_boundary"] for row in changed)
                numpy_only = set(diff) == {"numpy"}
                documented = (numpy_only and ties_only and old["host"]["numpy"]["version"].startswith("1.26.")
                              and new["host"]["numpy"]["version"].startswith("2.0."))
                result.update(status="reproduced", changed_target_count=len(changed), changed_targets=changed,
                              all_changes_at_exact_boundary_ties=ties_only,
                              attribution="numpy" if numpy_only else "resolved_environment",
                              cause_class="intended_semantic_change" if documented else "unresolved",
                              mechanism="Native NumPy top-K tie selection" if documented else "Environment-dependent neighbor pruning")
                if documented:
                    result["source"] = NUMPY_NOTES
    pilot._write_json(Path(result["artifact"]), result)
    if result["status"] == "reproduced":
        with ledger.writer() as con:
            con.execute("INSERT INTO adjudications VALUES (?, ?, ?, ?, ?, ?)",
                        [case_id, result["mechanism"], result["cause_class"], result["artifact"],
                         "pilot S3 replay", pilot._now().date()])
    return result
