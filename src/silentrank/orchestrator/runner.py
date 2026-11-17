"""Run commands inside the pinned runner container (never on the Windows host)."""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

from silentrank import paths

IMAGE = "silentrank-runner:py312"
SOURCE_IMAGE = "silentrank-runner:py312-source"
VENV_VOLUME = "silentrank-venvs"
CACHE_VOLUME = "silentrank-uv-cache"


class DockerUnavailable(RuntimeError):
    pass


def local_mode() -> bool:
    """True inside a Hugging Face Job (or any single container), where there is no Docker.

    The Job image must be the same pinned base as docker/Dockerfile.runner, with g++ installed,
    and the same determinism environment variables. The runner image id recorded in the ledger
    is then ``SILENTRANK_RUNNER_IMAGE``, so local-mode runs never pair with Docker-mode runs.
    """
    return os.environ.get("SILENTRANK_RUNNER") == "local"


def _link(destination: Path, source: Path) -> None:
    """Make an absolute container path point at a host directory or file (local mode only)."""
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        if destination.resolve() == source.resolve():
            return
        raise RuntimeError(f"local runner cannot map {destination}: it already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)


def _run_local(command: list[str], env: dict[str, str] | None, timeout: float | None,
               mounts: tuple[tuple[Path, str, bool], ...]) -> subprocess.CompletedProcess:
    root = paths.repo_root()
    for destination, source in ((Path("/repo/src"), root / "src"), (Path("/repo/envs"), root / "envs")):
        if destination.resolve() != source.resolve():
            _link(destination, source)
    for source, destination, _read_only in mounts:
        _link(Path(destination), source.resolve())
    full_env = {**os.environ, "PYTHONPATH": "/repo/src", **(env or {})}
    if timeout is not None:
        command = ["timeout", "--kill-after=5s", f"{timeout}s", *command]
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout + 20 if timeout else None, env=full_env)


def _docker(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
        )
    except FileNotFoundError as exc:
        raise DockerUnavailable("the `docker` command was not found") from exc


def image_for(env_id: str) -> str:
    # implicit builds from source on old snapshots and needs libgomp; Recommenders pulls lightgbm (libgomp).
    needs_source = env_id.startswith("py312-snapshot-") or "-implicit" in env_id or "-recommenders" in env_id or "-cornac" in env_id
    return SOURCE_IMAGE if needs_source else IMAGE


@lru_cache
def ensure_image(image: str = IMAGE) -> str:
    """Build the runner image if needed and return its image id (a sha256 digest)."""
    if local_mode():
        return os.environ["SILENTRANK_RUNNER_IMAGE"]
    root = paths.repo_root()
    target = "source-runner" if image == SOURCE_IMAGE else "runner"
    build = _docker([
        "build", "-q", "--provenance=false", "--target", target, "-t", image,
        "-f", str(root / "docker" / "Dockerfile.runner"), str(root / "docker"),
    ], timeout=1800)
    if build.returncode != 0:
        raise DockerUnavailable(f"could not build {image}: {build.stderr.strip()[-600:]}")
    inspect = _docker(["image", "inspect", image, "--format", "{{.Id}}"])
    if inspect.returncode or not inspect.stdout.strip():
        raise DockerUnavailable(f"could not inspect {image}: {inspect.stderr.strip()[-600:]}")
    return inspect.stdout.strip()


def run(
    command: list[str],
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    *,
    image: str = IMAGE,
    mounts: tuple[tuple[Path, str, bool], ...] = (),
    venvs_read_only: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command in a fresh container with optional extra mounts."""
    if local_mode():
        return _run_local(command, env, timeout, mounts)
    root = paths.repo_root()
    args = [
        "run", "--rm",
        "-v", f"{root / 'src'}:/repo/src:ro",
        "-v", f"{root / 'envs'}:/repo/envs",
        "-v", f"{VENV_VOLUME}:/venvs{':ro' if venvs_read_only else ''}",
        "-v", f"{CACHE_VOLUME}:/root/.cache/uv",
        "-e", "PYTHONPATH=/repo/src",
    ]
    for key, value in (env or {}).items():
        args += ["-e", f"{key}={value}"]
    for source, destination, read_only in mounts:
        mode = ":ro" if read_only else ""
        args += ["-v", f"{source.resolve()}:{destination}{mode}"]
    # An in-container deadline also kills the worker when Docker's client disconnects.
    if timeout is not None:
        command = ["timeout", "--kill-after=5s", f"{timeout}s", *command]
    return _docker([*args, image, *command], timeout=timeout + 20 if timeout else None)
