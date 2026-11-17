"""Build and describe environments under test (``envs/<env_id>/``) inside the runner container.

Every outcome is written to the ledger, including failures (README, rule 2):
``resolve_failed`` (no lock could be produced), ``build_failed`` (sync failed),
``import_failed`` (NumPy would not import), ``ok``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import tarfile
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4

from silentrank import paths
from silentrank.orchestrator import env_tests, ledger, manifests, runner

BUILD_TIMEOUT_S = 1800
PROBE_TIMEOUT_S = 300
PACKAGES = ("numpy", "scipy", "pandas", "implicit", "scikit-surprise", "recommenders", "cornac", "torch")
# Library variants of the systematic single-dependency pairs: suffix, distribution pin,
# and whether pandas transitions apply (only pipelines that call pandas get them).
VARIANTS = {
    "implicit": ("-implicit0.7.3", "implicit==0.7.3", False),
    "surprise": ("-surprise1.1.5", "scikit-surprise==1.1.5", False),
    "sar": ("-recommenders1.2.1", "recommenders==1.2.1", True),
    "cornac": ("-cornac3.1.0", "cornac==3.1.0", False),
}
# Upload cutoff for variant environments created after the protocol-v0 freeze, so their
# locks are reproducible. Declared in protocol/pipelines_p3_p4.md.
VARIANT_EXCLUDE_NEWER = "2026-11-13T00:00:00Z"
# PyTorch transitions hold these pins fixed (protocol/pipelines_p7_torch.md); CPU-only wheels.
TORCH_BASE = "py312-np2.2.6-sp1.16.3-pd2.2.3"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


# GPU condition (protocol/pipelines_p7_gpu.md): CUDA line fixed within each transition.
TORCH_GPU_LINE = {"T1": "cu126", "T2": "cu126", "T3": "cu126", "T4": "cu126", "T5": "cu126", "T6": "cu132"}


def define_torch_env(env_id: str) -> Path:
    """Write envs/<env_id>/pyproject.toml for a CPU (or ``-cuXXX``) PyTorch environment if missing."""
    prefix = f"{TORCH_BASE}-torch"
    if not env_id.startswith(prefix):
        raise ValueError(f"{env_id} is not a declared PyTorch environment")
    version = env_id.removeprefix(prefix)
    line = "cpu"
    if "-cu" in version:
        version, line = version.split("-", 1)
    index_name, index_url = (("pytorch-cpu", TORCH_CPU_INDEX) if line == "cpu"
                             else (f"pytorch-{line}", f"https://download.pytorch.org/whl/{line}"))
    directory = paths.repo_root() / "envs" / env_id
    project = directory / "pyproject.toml"
    if project.is_file():
        return project
    directory.mkdir(parents=True)
    lines = [
        "[project]",
        f'name = "env-{env_id.replace(".", "-")}"',
        'version = "0"',
        'requires-python = "==3.12.*"',
        "dependencies = [",
        '    "numpy==2.2.6",', '    "scipy==1.16.3",', '    "pandas==2.2.3",', '    "threadpoolctl==3.8.0",',
        f'    "torch=={version}",',
        "]",
        "",
        "[tool.uv]",
        "package = false",
        f'exclude-newer = "{VARIANT_EXCLUDE_NEWER}"',
        'constraint-dependencies = ["python-dateutil==4.0.0", "pytz==2026.6", "tzdata==2026.6"]',
        "",
        "[tool.uv.sources]",
        f'torch = {{ index = "{index_name}" }}',
        "",
        "[[tool.uv.index]]",
        f'name = "{index_name}"',
        f'url = "{index_url}"',
        "explicit = true",
    ]
    project.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return project


def env_dir(env_id: str) -> Path:
    if Path(env_id).name != env_id or env_id in ("", ".", ".."):
        raise FileNotFoundError(f"invalid environment id: {env_id!r}")
    path = paths.repo_root() / "envs" / env_id
    if not (path / "pyproject.toml").is_file():
        raise FileNotFoundError(f"no environment definition at envs/{env_id}/pyproject.toml")
    return path


def lock_sha256(env_id: str) -> str | None:
    lock = env_dir(env_id) / "uv.lock"
    return hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def _record_env(env_id: str, **fields) -> None:
    columns = ["env_id", *fields]
    updates = ", ".join(f"{name} = excluded.{name}" for name in fields)
    with ledger.writer() as con:
        con.execute(
            f"INSERT INTO envs ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)}) "
            f"ON CONFLICT (env_id) DO UPDATE SET {updates}",
            [env_id, *fields.values()],
        )


def _tail(text: str, limit: int = 1500) -> str:
    return text.strip()[-limit:]


def _log_dir(env_id: str, operation: str) -> Path:
    stamp = _now().strftime("%Y%m%dT%H%M%S")
    path = paths.results_dir() / "screening" / env_id / f"{stamp}-{operation}-{uuid4().hex[:8]}"
    path.mkdir(parents=True)
    return path


def _logged_run(
    command: list[str],
    log_file: Path,
    *,
    image: str,
    timeout: int,
    env: dict[str, str] | None = None,
    mounts: tuple[tuple[Path, str, bool], ...] = (),
    venvs_read_only: bool = False,
) -> subprocess.CompletedProcess:
    try:
        result = runner.run(
            command, env=env, timeout=timeout, image=image, mounts=mounts, venvs_read_only=venvs_read_only,
        )
    except subprocess.TimeoutExpired as exc:
        result = subprocess.CompletedProcess(
            command, 124, (exc.stdout or b"").decode(errors="replace"), str(exc),
        )
    log_file.write_text(
        json.dumps({"image": image, "command": command, "env": env, "exit_code": result.returncode})
        + "\n\nstdout\n" + result.stdout + "\n\nstderr\n" + result.stderr,
        encoding="utf-8",
    )
    return result


def _failed_env(env_id: str, status: str, log_file: Path, detail: str, **fields) -> dict:
    notes = f"{detail}\nLog: {log_file}"
    _record_env(env_id, status=status, notes=notes, **fields)
    return {"env_id": env_id, "status": status, "detail": detail, "log_path": str(log_file)}


def build(env_id: str) -> dict:
    """Lock (if no lockfile yet) and install the environment into /venvs/<env_id>."""
    env_dir(env_id)
    log_dir = _log_dir(env_id, "build")
    image = runner.image_for(env_id)
    try:
        image_id = runner.ensure_image(image)
    except (runner.DockerUnavailable, subprocess.TimeoutExpired) as exc:
        log_file = log_dir / "image.log"
        log_file.write_text(f"{image}\n{exc}\n", encoding="utf-8")
        return _failed_env(env_id, "build_failed", log_file, str(exc), built_at=_now())
    project = f"/repo/envs/{env_id}"
    uv_env = {"UV_PROJECT_ENVIRONMENT": f"/venvs/{env_id}"}
    fields = {"python": "3.12", "image_digest": image_id, "built_at": _now()}

    if lock_sha256(env_id) is None:
        log_file = log_dir / "resolve.log"
        locked = _logged_run(
            ["uv", "lock", "--project", project], log_file,
            image=image_id, env=uv_env, timeout=BUILD_TIMEOUT_S,
        )
        if locked.returncode != 0:
            return _failed_env(env_id, "resolve_failed", log_file, _tail(locked.stderr), **fields)

    fields["lock_sha256"] = lock_sha256(env_id)
    log_file = log_dir / "sync.log"
    synced = _logged_run(
        ["uv", "sync", "--frozen", "--project", project], log_file,
        image=image_id, env=uv_env, timeout=BUILD_TIMEOUT_S,
    )
    if synced.returncode != 0:
        return _failed_env(env_id, "build_failed", log_file, _tail(synced.stderr), **fields)

    log_file = log_dir / "freeze.log"
    freeze = _logged_run(
        ["uv", "pip", "freeze", "--python", f"/venvs/{env_id}/bin/python"], log_file,
        image=image_id, timeout=PROBE_TIMEOUT_S,
    )
    if freeze.returncode != 0:
        return _failed_env(env_id, "build_failed", log_file, _tail(freeze.stderr), **fields)
    packages = sorted(line for line in freeze.stdout.splitlines() if line.strip())
    _record_env(env_id, status="ok", freeze_json=json.dumps(packages), notes=None, **fields)
    return {
        "env_id": env_id, "status": "ok", "image": image_id, "lock_sha256": fields["lock_sha256"],
        "packages": packages, "log_dir": str(log_dir),
    }


def host_id_for(report: dict) -> str:
    """Stable id of the physical host as the container sees it: CPU model, flags, core count, kernel."""
    cpu = report.get("cpu", {})
    key = "|".join([str(cpu.get("model")), " ".join(cpu.get("flags", [])), str(cpu.get("logical_cpus")), str(report.get("kernel"))])
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _lock_packages(env_id: str) -> list[dict]:
    lock = env_dir(env_id) / "uv.lock"
    if not lock.exists():
        return []
    with lock.open("rb") as handle:
        return [entry for entry in tomllib.load(handle)["package"] if "virtual" not in entry["source"]]


def _libraries(env_id: str) -> list[str]:
    locked = {entry["name"] for entry in _lock_packages(env_id)}
    # Probe a recommender library only where it is the library under test. Recommenders, for
    # example, pulls in cornac and scikit-surprise as unused dependencies; importing them would
    # mark a SAR environment as failed for a package SAR never loads.
    under_test = {"implicit": "-implicit", "scikit-surprise": "-surprise", "recommenders": "-recommenders",
                  "cornac": "-cornac", "torch": "-torch"}
    return [package for package in PACKAGES if package in locked
            and (package not in under_test or under_test[package] in env_id or env_id.startswith("py312-snapshot-"))] or ["numpy"]


def probe(env_id: str, *, image: str | None = None) -> dict:
    """Run the probe inside the built environment; record its runtime info and the host."""
    env_dir(env_id)
    log_file = _log_dir(env_id, "probe") / "probe.log"
    result = _logged_run(
        [f"/venvs/{env_id}/bin/python", "/repo/src/silentrank/worker/probe.py", "--packages", *_libraries(env_id)],
        log_file, image=image or runner.image_for(env_id), timeout=PROBE_TIMEOUT_S,
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        return _failed_env(env_id, "import_failed", log_file, _tail(result.stderr or result.stdout))

    if result.returncode != 0:
        errors = {name: info["import_error"] for name, info in report["packages"].items() if "import_error" in info}
        return _failed_env(
            env_id, "import_failed", log_file, json.dumps(errors),
            runtime_json=json.dumps(report, sort_keys=True),
        )

    host_id = host_id_for(report)
    cpu = report["cpu"]
    with ledger.writer() as con:
        con.execute(
            """INSERT INTO hosts (host_id, cpu_model, cpu_flags, cores, ram_gb, os, kernel, first_seen, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (host_id) DO NOTHING""",
            [host_id, cpu.get("model"), " ".join(cpu.get("flags", [])), cpu.get("logical_cpus"),
             report.get("mem_total_gb"), report.get("platform"), report.get("kernel"), _now(),
             json.dumps({"cpu": cpu, "machine": report.get("machine")}, sort_keys=True)],
        )
    _record_env(env_id, status="ok", runtime_json=json.dumps(report, sort_keys=True), notes=None)
    return {"env_id": env_id, "status": "ok", "host_id": host_id, "report": report}


def _upstream_suite(env_id: str, image: str, package: str, info: dict) -> dict:
    log_dir = _log_dir(env_id, f"test-{package}")
    log_file = log_dir / "pytest.log"
    outcome = {"package": package, "version": info["version"], "log_path": str(log_file)}
    test_path = info["path"]
    config_args = []
    if not info["tests_bundled"]:
        entry = next(
            entry for entry in _lock_packages(env_id)
            if entry["name"] == package and entry["version"] == info["version"]
        )
        sdist = entry.get("sdist")
        try:
            tests = env_tests.release_tests(sdist, log_dir) if sdist else None
        except (OSError, ValueError, tarfile.TarError) as exc:
            reason = f"could not retrieve released tests: {exc}"
            log_file.write_text(reason + "\n", encoding="utf-8")
            return {**outcome, "status": "not_runnable", "reason": reason}
        if tests is None:
            reason = "the installed artifact and locked source distribution do not ship tests"
            log_file.write_text(reason + "\n", encoding="utf-8")
            return {**outcome, "status": "unavailable", "reason": reason}
        test_path = f"/screen/{tests.relative_to(log_dir).as_posix()}"
        config_file = tests.parent / "pyproject.toml"
        if config_file.is_file():
            config_args = ["-c", f"/screen/{config_file.relative_to(log_dir).as_posix()}"]

    requirements = (*env_tests.TEST_TOOLS, *env_tests.PACKAGE_TOOLS.get(package, ()))
    tools = [part for requirement in requirements for part in ("--with", requirement)]
    marker = env_tests.DEFAULT_MARKERS.get(package)
    marker_args = ["-m", marker] if marker else []
    python = f"/venvs/{env_id}/bin/python"
    command = [
        "uv", "run", "--no-project", "--python", python, *tools,
        "python", "-m", "pytest", "-q", "--timeout=30", "--junitxml=/screen/pytest.xml",
        "-o", "cache_dir=/tmp/.pytest_cache",
        *marker_args, *config_args, test_path,
    ]
    result = _logged_run(
        command, log_file, image=image, timeout=env_tests.TEST_TIMEOUT_S,
        env={"PYTHONPATH": f"/venvs/{env_id}/lib/python3.12/site-packages"},
        mounts=((log_dir, "/screen", False),), venvs_read_only=True,
    )
    result_report = env_tests.report(result.returncode, log_dir / "pytest.xml")
    with log_file.open("a", encoding="utf-8") as log:
        log.write("\n\nresult\n" + json.dumps({**outcome, **result_report}, sort_keys=True) + "\n")
    return {**outcome, **result_report}


def _record_suite(
    env_id: str,
    result: dict,
    *,
    ran_at: dt.datetime | None = None,
    only_missing: bool = False,
) -> None:
    conflict = "DO NOTHING" if only_missing else """
        DO UPDATE SET status = excluded.status, n_pass = excluded.n_pass, n_fail = excluded.n_fail,
          log_path = excluded.log_path, ran_at = excluded.ran_at
    """
    with ledger.writer() as con:
        con.execute(
            f"""INSERT INTO testsuite (env_id, package, status, n_pass, n_fail, log_path, ran_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (env_id, package) {conflict}""",
            [
                env_id, result["package"], result["status"], result.get("n_pass"), result.get("n_fail"),
                result["log_path"], ran_at or _now(),
            ],
        )


def _prepare_screen(env_id: str) -> tuple[dict, dict]:
    result = build(env_id)
    if result["status"] != "ok":
        return result, {}
    imported = probe(env_id, image=result["image"])
    if imported["status"] != "ok":
        return imported, {}
    result.update(host_id=imported["host_id"], testsuite=[])
    return result, imported["report"]["packages"]


def prepare_cases(case_ids: list[str]) -> dict:
    """Build and probe only the environments the given cases need, then bind their transitions.

    Used on a second host (protocol/second_host.md). Upstream test suites are not rerun there:
    their status was recorded on the primary host and is reported from it.
    """
    from silentrank.orchestrator.pilot import CASE_ID, VARIANT_GROUP

    base_pairs: dict[str, tuple[str, str]] = {}
    needed: dict[str, None] = {}
    for case_id in case_ids:
        matched = CASE_ID.fullmatch(case_id)
        if matched is None:
            raise ValueError(f"not a case id: {case_id}")
        transition, pipeline = matched.group(1), matched.group(2)
        gpu = case_id.endswith("-gpu")
        base_group = ("torch_gpu" if gpu else "torch") if transition.startswith("T") else "snapshot" if transition[0] in "BQ" else "base"
        base_pairs[transition] = pilot_pairs(base_group)[transition]
        needed.update(dict.fromkeys(base_pairs[transition]))
        group = VARIANT_GROUP.get(pipeline)
        if group and group != "torch" and base_group != "snapshot":
            needed.update(dict.fromkeys(pilot_pairs(group)[transition]))
    for env_id in needed:
        if "-torch" in env_id:
            define_torch_env(env_id)
        elif env_id.startswith("py312-snapshot-"):
            pass  # snapshot definitions are committed
        else:
            group = next((name for name, (suffix, _, _) in VARIANTS.items() if env_id.endswith(suffix)), "base")
            define_variant_env(env_id, group)
    statuses = {env_id: _prepare_screen(env_id)[0]["status"] for env_id in needed}
    with ledger.writer() as con:
        manifests.sync_transitions(con)
        for transition, (old, new) in base_pairs.items():
            ready = statuses[old] == statuses[new] == "ok"
            diff = _resolved_diff(old, new) if lock_sha256(old) and lock_sha256(new) else None
            con.execute(
                "UPDATE transitions SET env_old = ?, env_new = ?, resolved_diff_json = ?, status = ? WHERE transition_id = ?",
                [old, new, json.dumps(diff, sort_keys=True) if diff is not None else None,
                 "screened" if ready else "env_failed", transition],
            )
    return {"environments": statuses, "transitions": base_pairs}


def _screen_tests(entries: list[tuple[dict, dict]], *, jobs: int = 4) -> None:
    # Test jobs only read frozen environments. Write their ledger rows from one thread.
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        pending = {
            pool.submit(_upstream_suite, result["env_id"], result["image"], package, info): result
            for result, packages in entries for package, info in packages.items()
        }
        for future in as_completed(pending):
            result = pending[future]
            suite = future.result()
            _record_suite(result["env_id"], suite)
            result["testsuite"].append(suite)


def screen(env_id: str, *, packages: list[str] | None = None) -> dict:
    """Lock, sync, probe and record released upstream test suites for an environment."""
    result, discovered = _prepare_screen(env_id)
    if packages and result["status"] == "ok":
        missing = set(packages) - discovered.keys()
        if missing:
            raise ValueError(f"{env_id} has no screened package(s): {', '.join(sorted(missing))}")
        discovered = {name: discovered[name] for name in packages}
    _screen_tests([(result, discovered)])
    return result


def recover_suites(*, run_missing: bool = False) -> dict:
    """Fill missing test-suite rows from complete local logs after an interrupted batch."""
    with ledger.reader() as con:
        expected = {
            (env_id, package): (image, info)
            for env_id, image, runtime in con.execute(
                "SELECT env_id, image_digest, runtime_json FROM envs WHERE status = 'ok'"
            ).fetchall()
            for package, info in json.loads(runtime or "{}").get("packages", {}).items()
        }
        recorded = set(con.execute("SELECT env_id, package FROM testsuite").fetchall())

    recovered = []
    remaining = []
    screening_dir = paths.results_dir() / "screening"
    for env_id, package in sorted(expected.keys() - recorded):
        attempts = sorted(
            (screening_dir / env_id).glob(f"*-test-{package}-*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        missing = {"env_id": env_id, "package": package, "reason": "no attempt found"}
        for attempt in attempts:
            log_file = attempt / "pytest.log"
            missing.update(reason="no complete result for the screened image", attempt=str(attempt))
            if not log_file.is_file():
                continue
            contents = log_file.read_text(encoding="utf-8")
            first_line, _, _ = contents.partition("\n")
            try:
                head = json.loads(first_line)
            except json.JSONDecodeError:
                continue
            if (
                not isinstance(head, dict)
                or head.get("image") != expected[(env_id, package)][0]
                or f"/venvs/{env_id}/bin/python" not in head.get("command", [])
            ):
                continue
            _, marker, footer = contents.rpartition("\nresult\n")
            if not marker:
                continue
            try:
                outcome = json.loads(footer)
            except json.JSONDecodeError:
                continue
            if not isinstance(outcome, dict) or outcome.get("status") not in {
                "passed", "failed", "not_runnable", "unavailable",
            }:
                continue
            result = {**outcome, "package": package, "log_path": str(log_file)}
            ran_at = dt.datetime.fromtimestamp(log_file.stat().st_mtime, dt.timezone.utc).replace(tzinfo=None)
            _record_suite(env_id, result, ran_at=ran_at, only_missing=True)
            recovered.append({"env_id": env_id, **result})
            break
        else:
            remaining.append(missing)

    tested = []
    if run_missing:
        # Handle interrupted attempts first, then the recommender's own test suite.
        remaining.sort(key=lambda row: ("attempt" not in row, row["package"] != "implicit"))
        entries = []
        for row in remaining:
            env_id, package = row["env_id"], row["package"]
            image, info = expected[(env_id, package)]
            entries.append(({"env_id": env_id, "image": image, "testsuite": []}, {package: info}))
        _screen_tests(entries, jobs=2)
        tested = [
            {"env_id": entry["env_id"], **suite}
            for entry, _ in entries
            for suite in entry["testsuite"]
        ]
        remaining = []
    return {"status": "ok", "recovered": recovered, "remaining": remaining, "tested": tested}


def define_variant_env(env_id: str, group: str) -> Path:
    """Write envs/<env_id>/pyproject.toml for a library variant if it does not exist yet."""
    suffix, requirement, _ = VARIANTS.get(group, ("", None, True))  # "base" has no added library
    if not env_id.endswith(suffix) or (group == "base" and not env_id.rsplit("-pd", 1)[-1].replace(".", "").isdigit()):
        raise ValueError(f"{env_id} is not a definable {group} environment")
    parts = {part[:2]: part[2:] for part in env_id.removesuffix(suffix).split("-")[1:]}
    versions = {name: parts[key] for name, key in (("numpy", "np"), ("scipy", "sp"), ("pandas", "pd"))}
    directory = paths.repo_root() / "envs" / env_id
    project = directory / "pyproject.toml"
    if project.is_file():
        return project
    directory.mkdir(parents=True)
    name = "env-" + env_id.replace(".", "-")
    dependencies = [f"numpy=={versions['numpy']}", f"scipy=={versions['scipy']}", f"pandas=={versions['pandas']}",
                    "threadpoolctl==3.8.0", *([requirement] if requirement else [])]
    lines = [
        "[project]",
        f'name = "{name}"',
        'version = "0"',
        'requires-python = "==3.12.*"',
        "dependencies = [",
        *(f'    "{item}",' for item in dependencies),
        "]",
        "",
        "[tool.uv]",
        "package = false",
        f'exclude-newer = "{VARIANT_EXCLUDE_NEWER}"',
        'constraint-dependencies = ["python-dateutil==4.0.0", "pytz==2026.6", "tzdata==2026.6"]',
    ]
    project.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return project


def define_snapshot_env(as_of: str, requirements: list[str]) -> Path:
    """Write envs/py312-snapshot-<date>/pyproject.toml (unpinned, date-limited) if missing."""
    env_id = f"py312-snapshot-{as_of}"
    directory = paths.repo_root() / "envs" / env_id
    project = directory / "pyproject.toml"
    if project.is_file():
        return project
    directory.mkdir(parents=True)
    lines = [
        "[project]",
        f'name = "env-{env_id}"',
        'version = "0"',
        'requires-python = "==3.12.*"',
        "dependencies = [",
        *(f'    "{item}",' for item in requirements),
        "]",
        "",
        "[tool.uv]",
        "package = false",
        f'exclude-newer = "{as_of}T00:00:00Z"',
    ]
    project.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return project


def pilot_pairs(group: str, split: str | None = None) -> dict[str, tuple[str, str]]:
    """Manifest transition IDs and environment pairs for one screening group.

    ``split`` limits the result to "dev" or "heldout" transitions. PyTorch transitions
    are screened in Phase 3 on rented compute and are never returned here.
    """
    pairs = {}
    for entry in manifests.load_all():
        if entry["kind"] != "transition" or (split and entry["split"] != split):
            continue
        if entry["family"] == "torch":
            if group in ("torch", "torch_gpu"):
                change = entry["change"]
                suffix = f"-{TORCH_GPU_LINE[entry['id']]}" if group == "torch_gpu" else ""
                pairs[entry["id"]] = tuple(f"{TORCH_BASE}-torch{change[side]}{suffix}" for side in ("old", "new"))
            continue
        if group in ("torch", "torch_gpu"):
            continue
        collection = entry["collection"]
        if collection == "systematic_snapshot" and group in {"snapshot", "all"}:
            change = entry["change"]
            pairs[entry["id"]] = tuple(f"py312-snapshot-{change[f'{side}_as_of']}" for side in ("old", "new"))
        if collection != "systematic_single" or group not in {"base", "all", *VARIANTS}:
            continue
        change = entry["change"]
        if group in VARIANTS and change["package"] == "pandas" and not VARIANTS[group][2]:
            continue
        suffix = VARIANTS[group][0] if group in VARIANTS else ""
        sides = []
        for side in ("old", "new"):
            versions = {"pandas": "2.2.3", **entry["held_fixed"], change["package"]: change[side]}
            sides.append(f"py312-np{versions['numpy']}-sp{versions['scipy']}-pd{versions['pandas']}{suffix}")
        pairs[entry["id"]] = (sides[0], sides[1])
    return pairs


def _resolved_diff(old_env: str, new_env: str) -> dict:
    def packages(env_id: str) -> dict:
        records = {}
        for entry in _lock_packages(env_id):
            records.setdefault(entry["name"], []).append(entry)
        return records

    old = packages(old_env)
    new = packages(new_env)
    return {
        package: {"old": old.get(package, []), "new": new.get(package, [])}
        for package in sorted(old.keys() | new.keys())
        if old.get(package) != new.get(package)
    }


def screen_pilot(group: str = "base", split: str | None = None) -> dict:
    """Screen each unique environment in one group, then record transition bindings."""
    if group not in {"base", "snapshot", "all", "torch", "torch_gpu", *VARIANTS}:
        raise ValueError(f"unknown pilot environment group: {group}")
    pairs = pilot_pairs(group, split)
    if group in ("torch", "torch_gpu"):
        for env_id in dict.fromkeys(env_id for pair in pairs.values() for env_id in pair):
            define_torch_env(env_id)
    if group in {"snapshot", "all"}:
        for entry in manifests.load_all():
            change = entry.get("change") or {}
            if entry["id"] in pairs and change.get("kind") == "snapshot":
                for side in ("old", "new"):
                    define_snapshot_env(change[f"{side}_as_of"], change["requirements"])
    if group in {"base", *VARIANTS}:
        for env_id in dict.fromkeys(env_id for pair in pairs.values() for env_id in pair):
            define_variant_env(env_id, group)
    variants = pilot_pairs("implicit", split) if group == "all" else {}
    env_ids = dict.fromkeys(env_id for pair in (*pairs.values(), *variants.values()) for env_id in pair)
    prepared = [_prepare_screen(env_id) for env_id in env_ids]
    statuses = {result["env_id"]: result["status"] for result, _ in prepared}
    if group not in VARIANTS:  # base, snapshot, all and torch bind their transitions
        with ledger.writer() as con:
            manifests.sync_transitions(con)
            for transition, (old, new) in pairs.items():
                ready = statuses[old] == statuses[new] == "ok"
                diff = _resolved_diff(old, new) if lock_sha256(old) and lock_sha256(new) else None
                con.execute(
                    """UPDATE transitions SET env_old = ?, env_new = ?, resolved_diff_json = ?, status = ?
                       WHERE transition_id = ?""",
                    [old, new, json.dumps(diff, sort_keys=True) if diff is not None else None,
                     "screened" if ready else "env_failed", transition],
                )
    # Resume-friendly: an upstream suite already recorded for an environment is not run again.
    with ledger.reader() as con:
        recorded = set(con.execute("SELECT env_id, package FROM testsuite").fetchall())
    _screen_tests([
        (result, {package: info for package, info in packages.items() if (result["env_id"], package) not in recorded})
        for result, packages in prepared
    ])
    return {
        "group": group, "status": "ok" if all(value == "ok" for value in statuses.values()) else "env_failed",
        "pairs": pairs, "implicit_pairs": variants, "environments": [result for result, _ in prepared],
    }
