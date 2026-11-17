"""Run Phase 1 cases in screened Linux environments and record every attempt."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from silentrank import paths
from silentrank.evaluator import identity
from silentrank.orchestrator import envs, evaluation, ledger, runner
from silentrank.stats.paired import PILOT_SEEDS

NEIGHBOR_GRID = (50, 100, 200)
CASE_ID = re.compile(r"(N\d+|S\d+|D\d+|B\d+|Q\d+|T\d+)-(P0|P1|P2|P3|P4|P6|P7)(?:-(float32|float64|als|cosine|bm25|svd|sar|ease|knn|bpr|gpu))?(-sensitivity|-ml20m)?")
IMPLICIT_VARIANTS = ("als", "cosine", "bm25")
ALS_GRID = (32, 64)
# P3/P4 grids, declared in protocol/pipelines_p3_p4.md before their ranking runs.
SVD_GRID = (50, 100)
SAR_GRID = ("jaccard", "cosine", "lift")
# P6 grids, declared in protocol/pipelines_p6_cornac.md before their ranking runs.
EASE_GRID = (100.0, 500.0, 1000.0)
CORNAC_KNN_GRID = (20, 50, 100)
VARIANT_GROUP = {"P2": "implicit", "P3": "surprise", "P4": "sar", "P6": "cornac", "P7": "torch"}
# P7 grid, declared in protocol/pipelines_p7_torch.md before its ranking runs.
TORCH_DIM_GRID = (32, 64)
RUN_TIMEOUT = 900


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _token(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=paths.repo_root(), encoding="utf-8").strip()


def _pilot_state(case_id: str) -> tuple[dict, dict]:
    matched = CASE_ID.fullmatch(case_id)
    if matched is None:
        raise ValueError("expected a case ID such as N1-P1-float32 or D1-P0")
    transition, pipeline, variant, sensitivity = matched.groups()
    if pipeline == "P0" and (variant or sensitivity or transition[0] not in "DNBQ"):
        raise ValueError("P0 cases use pandas, NumPy or snapshot transitions and have no suffix")
    if pipeline == "P1" and (variant not in {"float32", "float64"} or transition.startswith("D")):
        raise ValueError("P1 cases use NumPy, SciPy or snapshot transitions and require a dtype suffix")
    if pipeline == "P2" and (variant not in IMPLICIT_VARIANTS or transition.startswith("D")):
        raise ValueError("P2 cases use NumPy, SciPy or snapshot transitions and require an ALS, cosine or BM25 suffix")
    if pipeline == "P3" and (variant != "svd" or transition[0] not in "NS" or sensitivity):
        raise ValueError("P3 cases use NumPy or SciPy transitions and the svd suffix")
    if (pipeline == "P7") != transition.startswith("T") or (pipeline == "P7" and (variant not in {"bpr", "gpu"} or sensitivity)):
        raise ValueError("P7 cases use PyTorch transitions (T*) with the bpr or gpu suffix, and T* cases use P7 only")
    if pipeline == "P6" and (variant not in {"ease", "knn"} or transition[0] not in "NS" or sensitivity):
        raise ValueError("P6 cases use NumPy or SciPy transitions and the ease or knn suffix")
    if pipeline == "P4" and (variant != "sar" or transition[0] in "BQ" or sensitivity):
        raise ValueError("P4 cases use NumPy, SciPy or pandas transitions and the sar suffix")
    with ledger.reader() as con:
        pair = con.execute(
            "SELECT family, env_old, env_new, collection, split FROM transitions WHERE transition_id = ?",
            [transition],
        ).fetchone()
        if pair is None or pair[1] is None or pair[2] is None:
            raise ValueError(f"screen the environment pair for {transition} first")
        variant_group = VARIANT_GROUP.get(pipeline) if pair[3] != "systematic_snapshot" else None
        if pipeline == "P7" and variant == "gpu":
            variant_group = "torch_gpu"
        if pair[4] == "heldout":
            _git("rev-parse", "--verify", "protocol-v1^{commit}")  # held-out cases need the frozen v1 protocol
        ids = envs.pilot_pairs(variant_group)[transition] if variant_group else pair[1:3]
        screened = con.execute("SELECT env_id, status, image_digest, lock_sha256 FROM envs WHERE env_id IN (?, ?)", ids).fetchall()
    environments = {row[0]: row[1:] for row in screened}
    if any(env_id not in environments or environments[env_id][0] != "ok" for env_id in ids):
        raise ValueError(f"{transition} has an environment that did not pass build and import screening")
    if environments[ids[0]][1] != environments[ids[1]][1]:
        raise ValueError(f"{transition} environments must use the same runner image")
    model = {"P0": "prep_pandas", "P1": "ref_itemknn", "P2": f"implicit_{variant}", "P3": "surprise_svd",
             "P4": "sar", "P6": f"cornac_{variant}", "P7": "torch_bpr"}[pipeline]
    config = {
        "case": case_id, "transition": transition, "collection": pair[3],
        "pipeline": model, "dtype": {"P2": "float32", "P3": "float64", "P4": "float64", "P6": "float64", "P7": "float32"}.get(pipeline, variant),
        "family": pair[0], "old": ids[0], "new": ids[1],
        "neighbors_grid": (list(CORNAC_KNN_GRID) if model == "cornac_knn" else list(NEIGHBOR_GRID)
                           if model not in {"prep_pandas", "implicit_als", "surprise_svd", "sar", "cornac_ease", "torch_bpr"} else []),
        "factors_grid": (list(ALS_GRID) if model == "implicit_als" else list(SVD_GRID) if model == "surprise_svd"
                         else list(TORCH_DIM_GRID) if model == "torch_bpr" else []),
        "k": 20, "threads": 1,
    }
    if model == "sar":
        config["similarity_grid"] = list(SAR_GRID)
    if model == "cornac_ease":
        config["lamb_grid"] = list(EASE_GRID)
    if model == "implicit_als":
        config.update(iterations=15, regularization=0.01, alpha=20.0)
    if sensitivity == "-sensitivity":
        config["dataset"] = "ml1m_per_user_70_10_20"
    elif sensitivity == "-ml20m":
        if pipeline not in {"P1", "P2"}:
            raise ValueError("MovieLens 20M cases use P1 or P2 only (protocol/dataset_ml20m.md)")
        config["dataset"] = "ml20m"
    return config, environments


def _record_case(config: dict) -> None:
    with ledger.writer() as con:
        con.execute(
            """INSERT INTO cases (case_id, transition_id, pipeline, dataset, config_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (case_id) DO NOTHING""",
            [config["case"], config["transition"], config["pipeline"], config.get("dataset", "ml1m"),
             json.dumps(config, sort_keys=True), _now()],
        )


def _record_applicability(case_id: str, applicable: bool, evidence: dict, reason: str | None = None) -> None:
    with ledger.writer() as con:
        con.execute(
            "UPDATE cases SET applicable = ?, applicability_evidence = ?, exclusion_reason = ? WHERE case_id = ?",
            [applicable, json.dumps(evidence, sort_keys=True), reason, case_id],
        )


def _run_rows(case_id: str) -> list[dict]:
    with ledger.reader() as con:
        cursor = con.execute(
            "SELECT run_id, side, mode, seed, repeat, status, artifact_dir, host_id, code_commit, evaluator_version "
            "FROM runs WHERE case_id = ? ORDER BY started_at", [case_id],
        )
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]


def _source_version(pipeline: str) -> tuple[str, str]:
    commit = _git("rev-parse", "HEAD")
    _git("rev-parse", "--verify", "protocol-v0^{commit}")
    source = hashlib.sha256()
    adapters = ({"implicit_models.py", "ref_itemknn.py"} if pipeline.startswith("implicit_")
                else {"cornac_models.py"} if pipeline.startswith("cornac_") else {f"{pipeline}.py"})
    for path in sorted(Path(__file__).resolve().parents[1].joinpath("worker").rglob("*.py")):
        if path.name in {"replay_rank.py", "replay_neighbors.py"}:
            continue
        if path.parent.name == "pipelines" and path.name not in adapters | {"__init__.py"}:
            continue
        source.update(path.relative_to(paths.repo_root()).as_posix().encode("utf-8") + b"\0")
        source.update(path.read_text(encoding="utf-8").encode("utf-8") + b"\0")
    return commit, source.hexdigest()


def _initial_factors(dataset: Path, dataset_sha: str, seed: int, factors: int) -> Path:
    directory = paths.results_dir() / "inputs" / "implicit" / dataset_sha
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"seed{seed}-factors{factors}.npz"
    with np.load(dataset, allow_pickle=False) as data:
        users, items = len(data["user_ids"]), len(data["item_ids"])
    rng = np.random.Generator(np.random.PCG64(seed))
    arrays = {
        "user_factors": rng.random((users, factors), dtype=np.float32) * np.float32(0.01),
        "item_factors": rng.random((items, factors), dtype=np.float32) * np.float32(0.01),
        "seed": np.asarray(seed, dtype=np.int64),
    }
    if path.exists():
        with np.load(path, allow_pickle=False) as previous:
            if not all(np.array_equal(previous[key], value) for key, value in arrays.items()):
                raise ValueError(f"initial factor inputs changed at {path}")
    else:
        with tempfile.TemporaryDirectory(dir=directory) as staging:
            candidate = Path(staging) / path.name
            np.savez(candidate, **arrays)
            candidate.replace(path)
    return path


def _torch_inputs(dataset: Path, dataset_sha: str, seed: int, dim: int) -> Path:
    """Frozen P7 inputs: initial embeddings, per-epoch order and negatives (pipelines_p7_torch.md)."""
    from silentrank.worker.pipelines.torch_bpr import EPOCHS

    directory = paths.results_dir() / "inputs" / "torch_bpr" / dataset_sha
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"seed{seed}-dim{dim}.npz"
    if path.exists():
        return path
    with np.load(dataset, allow_pickle=False) as data:
        users, items, train = len(data["user_ids"]), len(data["item_ids"]), data["train"]
    positives = train[train[:, 2] >= 4][:, :2].astype(np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    user_init = (rng.standard_normal((users, dim)) * 0.1).astype(np.float32)
    item_init = (rng.standard_normal((items, dim)) * 0.1).astype(np.float32)
    known = set((positives[:, 0] * items + positives[:, 1]).tolist())
    orders = np.empty((EPOCHS, len(positives)), dtype=np.int64)
    negatives = np.empty((EPOCHS, len(positives)), dtype=np.int64)
    for epoch in range(EPOCHS):
        orders[epoch] = rng.permutation(len(positives))
        draw = rng.integers(0, items, size=len(positives), dtype=np.int64)
        clash = np.fromiter(((u * items + j) in known for u, j in zip(positives[:, 0].tolist(), draw.tolist())), bool, len(draw))
        while clash.any():
            draw[clash] = rng.integers(0, items, size=int(clash.sum()), dtype=np.int64)
            rows = np.flatnonzero(clash)
            clash[rows] = np.fromiter(((positives[r, 0] * items + draw[r]) in known for r in rows.tolist()), bool, len(rows))
        negatives[epoch] = draw
    with tempfile.TemporaryDirectory(dir=directory) as staging:
        candidate = Path(staging) / path.name
        np.savez(candidate, user_init=user_init, item_init=item_init, order=orders, negatives=negatives,
                 seed=np.asarray(seed, dtype=np.int64), dim=np.asarray(dim, dtype=np.int64))
        candidate.replace(path)
    return path


def _launch(config: dict, environments: dict, *, side: str, mode: str, seed: int, split: str = "test",
            neighbors: int | None = None, factors: int | None = None, similarity: str | None = None,
            lamb: float | None = None, model_dir: Path | None = None) -> dict:
    root = paths.repo_root()
    sensitivity = config.get("dataset") == "ml1m_per_user_70_10_20"
    ml20m = config.get("dataset") == "ml20m"
    data_dir = root / "data" / ("sensitivity" if sensitivity else "ml20m" if ml20m else "")
    data_manifest = _read_json(data_dir / "manifest.json")
    approved_manifest = json.loads(_git("show", "dataset-ml20m-v1:data/ml20m/manifest.json" if ml20m
                                        else "protocol-v0:data/manifest.json"))
    if sensitivity:
        recorded_split = data_manifest["selected_split"]
        if (data_manifest["source"] != approved_manifest["source"]
                or recorded_split["method"] != "per_user_chronological"
                or recorded_split["target_percentages"] != [70, 10, 20]):
            raise ValueError("sensitivity data differs from the approved per-user split")
    elif data_manifest != approved_manifest:
        raise ValueError("dataset manifest differs from approved protocol-v0")
    dataset = data_dir / data_manifest["prepared"]["filename"]
    code_commit, worker_sha = _source_version(config["pipeline"])
    evaluator_version = identity()
    env_id = config["new" if side == "new" else "old"]
    _, image, lock_sha = environments[env_id]
    if not image or not lock_sha or envs.lock_sha256(env_id) != lock_sha:
        raise ValueError(f"{env_id} lock or runner image no longer matches its screening record")
    source_model = model_dir / "model.npz" if model_dir is not None else None
    als_retrain = config["pipeline"] == "implicit_als" and mode == "retrain"
    initial_model = _initial_factors(dataset, data_manifest["prepared"]["sha256"], seed, factors) if als_retrain else None
    if config["pipeline"] == "torch_bpr" and mode == "retrain":
        initial_model = _torch_inputs(dataset, data_manifest["prepared"]["sha256"], seed, factors)
    inputs = {
        "case": config["case"], "env_id": env_id, "side": side, "mode": mode, "seed": seed,
        "split": split, "neighbors": neighbors, "source_model": _digest(source_model) if source_model else None,
        "factors": factors, "initial_model": _digest(initial_model) if initial_model else None,
        "dataset_sha256": data_manifest["prepared"]["sha256"], "worker_sha256": worker_sha,
        "evaluator_version": evaluator_version, "image": image, "lock_sha256": lock_sha,
    }
    if similarity is not None:  # only P4; keeps earlier pipelines' input records unchanged
        inputs["similarity"] = similarity
    if lamb is not None:  # only P6 EASE
        inputs["lamb"] = lamb
    previous = [row for row in _run_rows(config["case"]) if
                (row["side"], row["mode"], row["seed"]) == (side, mode, seed)]
    for row in reversed(previous):
        if row["status"] in {"ok", "contract_fail"} and _read_json(Path(row["artifact_dir"]) / "meta.json")["inputs"] == inputs:
            return row
    repeat = max((row["repeat"] for row in previous), default=-1) + 1
    run_root = paths.results_dir() / "runs"
    while True:
        run_id = _token({**inputs, "repeat": repeat})
        output = run_root / run_id
        try:
            output.mkdir(parents=True)
            break
        except FileExistsError:
            repeat += 1

    py = f"/venvs/{env_id}/bin/python"
    command = [py, "-m", "silentrank.worker", "--pipeline", config["pipeline"], "--mode", mode,
               "--data", f"/pilot-data/{dataset.name}", "--output-dir", "/pilot-output", "--split", split,
               "--seed", str(seed), "--k", str(config["k"]), "--data-sha256", inputs["dataset_sha256"]]
    if config["family"] != "snapshot":
        command.extend(["--trace", config["family"]])
    if mode == "prep":
        command.extend(["--raw", "/pilot-data/ml-1m.zip", "--manifest", "/pilot-data/manifest.json"])
    elif config["pipeline"] == "implicit_als":
        command.extend(["--dtype", config["dtype"], "--factors", str(factors), "--iterations", str(config["iterations"]),
                        "--regularization", str(config["regularization"]), "--alpha", str(config["alpha"])])
    elif config["pipeline"] == "torch_bpr":
        command.extend(["--dtype", config["dtype"], "--factors", str(factors)])
        if config["case"].endswith("-gpu"):
            command.extend(["--device", "cuda"])
    elif config["pipeline"] == "surprise_svd":
        command.extend(["--dtype", config["dtype"], "--factors", str(factors)])
    elif config["pipeline"] == "sar":
        command.extend(["--dtype", config["dtype"], "--similarity", similarity])
    elif config["pipeline"] == "cornac_ease":
        command.extend(["--dtype", config["dtype"], "--lamb", str(lamb)])
    else:
        command.extend(["--dtype", config["dtype"], "--neighbors", str(neighbors)])
    if source_model is not None:
        command.extend(["--model-in", "/pilot-model/model.npz", "--model-sha256", inputs["source_model"]])
    mounts = [(data_dir, "/pilot-data", True), (output, "/pilot-output", False)]
    if model_dir is not None:
        mounts.append((model_dir, "/pilot-model", True))
    if initial_model is not None:
        mounts.append((initial_model, "/pilot-initial.npz", True))
        command.extend(["--initial-model", "/pilot-initial.npz", "--initial-model-sha256", inputs["initial_model"]])
    wrapped = ["sh", "-c", '"$1" /repo/src/silentrank/worker/probe.py > /pilot-output/host.json && exec "$@"',
               "silentrank", *command]
    started_at = _now()
    start = time.perf_counter()
    status, host_id, error = "crash", None, None
    stages, worker, stdout, stderr = {}, {}, "", ""
    try:
        result = runner.run(wrapped, timeout=RUN_TIMEOUT, image=image, mounts=tuple(mounts), venvs_read_only=True)
        stdout, stderr = result.stdout, result.stderr
        if (output / "host.json").is_file():
            host_id = envs.host_id_for(_read_json(output / "host.json"))
        if result.returncode == 124:
            status, error = "timeout", f"runner reached its deadline (status {result.returncode})"
        elif result.returncode != 0:
            error = f"runner exited with status {result.returncode}"
        else:
            worker = _read_json(output / "worker_meta.json")
            stages = _read_json(output / "stage_hashes.json")
            if host_id is None:
                raise ValueError("worker completed without recording its host")
            if worker.get("model_input_sha256") != inputs["source_model"]:
                raise ValueError("worker loaded a different inference model")
            if worker.get("initial_model_sha256") != inputs["initial_model"]:
                raise ValueError("worker loaded different initial factors")
            if worker.get("input_sha256") != (data_manifest["source"]["sha256"] if mode == "prep" else inputs["dataset_sha256"]):
                raise ValueError("worker used different dataset input")
            if _source_version(config["pipeline"])[1] != worker_sha or envs.lock_sha256(env_id) != lock_sha:
                raise ValueError("worker source or environment lock changed during the run")
            status = "ok"
            if mode != "prep":
                evaluated = evaluation.evaluate(output, dataset, split, paths.results_dir() / "metrics" / f"{run_id}.parquet")
                stages.update(evaluated["stages"])
                if evaluated["violations"]:
                    status = "contract_fail"
    except subprocess.TimeoutExpired as exc:
        status, error = "timeout", f"worker exceeded {RUN_TIMEOUT}s"
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
    except evaluation.ArtifactContractFailure as exc:
        status, error = "contract_fail", str(exc)
        (output / "contracts.json").write_text(
            json.dumps([{"user": None, "codes": [error]}], indent=2) + "\n", encoding="utf-8"
        )
    except (ValueError, OSError, runner.DockerUnavailable) as exc:
        status, error = "crash", str(exc)

    (output / "stdout.log").write_text(stdout, encoding="utf-8")
    (output / "stderr.log").write_text(stderr, encoding="utf-8")
    warnings_file = output / "warnings.log"
    warnings = [line for line in stderr.splitlines() if re.search(r"\b\w*Warning: ", line)]
    warnings_file.write_text("\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8")
    fields = {
        "run_id": run_id, "case_id": config["case"], "env_id": env_id, "side": side, "mode": mode,
        "seed": seed, "repeat": repeat, "threads": 1, "status": status, "host_id": host_id,
        "started_at": started_at, "wall_s": time.perf_counter() - start, "cpu_s": worker.get("cpu_s"),
        "peak_rss_mb": worker.get("peak_rss_mb"), "warnings_n": len(warnings), "artifact_dir": str(output),
        "code_commit": code_commit, "evaluator_version": evaluator_version,
    }
    timing_context = config.get("timing_context")
    _write_json(output / "meta.json", {**fields, "inputs": inputs, "error": error, "timing_context": timing_context})
    with ledger.writer() as con:
        con.execute("BEGIN TRANSACTION")
        con.execute(f"INSERT INTO runs ({', '.join(fields)}) VALUES ({', '.join('?' for _ in fields)})", list(fields.values()))
        for name, stage in stages.items():
            con.execute("INSERT INTO stage_hashes VALUES (?, ?, ?, ?, ?)",
                        [run_id, name, stage["canonical_sha256"], json.dumps(stage["shape"]), stage["dtype"]])
        notes = " | ".join(note for note in (error, timing_context) if note) or None
        con.execute("INSERT INTO costs VALUES (?, ?, ?, ?, ?)", [run_id, f"{mode}:{split}", fields["wall_s"], fields["cpu_s"], notes])
        con.execute("COMMIT")
    return fields


def _validation_score(run_id: str) -> float:
    values = pq.read_table(paths.results_dir() / "metrics" / f"{run_id}.parquet", columns=["ndcg10"])
    return float(values["ndcg10"].to_numpy().mean())


def _stage_values(run_id: str) -> list[tuple]:
    with ledger.reader() as con:
        return con.execute(
            "SELECT stage, canonical_sha256 FROM stage_hashes WHERE run_id = ? ORDER BY stage", [run_id]
        ).fetchall()


def _require_clean_control(old: dict, duplicate: dict) -> None:
    if old["status"] != "ok" or duplicate["status"] != "ok":
        raise ValueError("A/A control did not complete with valid rankings")
    if old["host_id"] != duplicate["host_id"] or _stage_values(old["run_id"]) != _stage_values(duplicate["run_id"]):
        raise ValueError("A/A control differs by host or stage hash; stop before A/B")
    first, second = Path(old["artifact_dir"]), Path(duplicate["artifact_dir"])
    for path in sorted(first.glob("*.npy")):
        if _digest(path) != _digest(second / path.name):
            raise ValueError(f"A/A control has nonidentical {path.name}; stop before A/B")


def run_case(case_id: str, *, inference_only: bool = False, timing_context: str | None = None) -> dict:
    """Tune on baseline validation and run A/A before A/B for a registered case."""
    config, environments = _pilot_state(case_id)
    _record_case(config)
    config["timing_context"] = timing_context
    if config["pipeline"] == "prep_pandas":
        old = _launch(config, environments, side="old", mode="prep", seed=0)
        if old["status"] != "ok":
            return {"case": case_id, "status": old["status"], "runs": [old["run_id"]]}
        evidence = _read_json(Path(old["artifact_dir"]) / "trace.json")
        _record_applicability(case_id, any(evidence.values()), evidence,
                              "no relevant upgraded-package calls observed" if not any(evidence.values()) else None)
        if not any(evidence.values()):
            return {"case": case_id, "status": "not_applicable", "runs": [old["run_id"]]}
        new = _launch(config, environments, side="new", mode="prep", seed=0)
        return {"case": case_id, "status": new["status"], "runs": [old["run_id"], new["run_id"]]}

    parameter = next(name for name in ("factors", "similarity", "lamb", "neighbors") if config.get(f"{name}_grid"))
    values = config[f"{parameter}_grid"]
    grid = [_launch(config, environments, side="old", mode="retrain", seed=PILOT_SEEDS[0], split="validation",
                    **{parameter: value}) for value in values]
    if any(row["status"] != "ok" for row in grid):
        return {"case": case_id, "status": "validation_failed", "runs": [row["run_id"] for row in grid]}
    best = max(range(len(grid)), key=lambda index: _validation_score(grid[index]["run_id"]))
    model_dir = Path(grid[best]["artifact_dir"])
    evidence = _read_json(model_dir / "trace.json")
    applicable = any(evidence.values())
    _record_applicability(case_id, applicable, evidence, None if applicable else "no relevant upgraded-package calls observed")
    if not applicable:
        return {"case": case_id, "status": "not_applicable", "runs": [row["run_id"] for row in grid]}
    selected = {parameter: values[best]}
    fixed = {"mode": "inference", "seed": 0, "model_dir": model_dir, **selected}
    old = _launch(config, environments, side="old", **fixed)
    duplicate = _launch(config, environments, side="old_repeat", **fixed)
    _require_clean_control(old, duplicate)
    new = _launch(config, environments, side="new", **fixed)
    runs = [old, duplicate, new]
    if not inference_only and new["status"] in {"ok", "contract_fail"}:
        for seed in PILOT_SEEDS:
            for side in ("old", "new"):
                runs.append(_launch(config, environments, side=side, mode="retrain", seed=seed, **selected))
    failures = [row["run_id"] for row in runs if row["status"] not in {"ok", "contract_fail"}]
    return {"case": case_id, "status": "run_failed" if failures else "ok", **selected,
            "validation_model_run": grid[best]["run_id"], "runs": [row["run_id"] for row in runs], "failures": failures}
