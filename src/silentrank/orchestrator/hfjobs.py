"""Hugging Face Jobs: bundle a commit, launch the harness in local mode, fetch results.

Nothing here spends money by itself. ``launch`` is only called after the researcher's explicit
go for that launch (README, rule 18), and it needs the researcher's own ``hf auth login``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from silentrank import paths

BASE_IMAGE = "ghcr.io/astral-sh/uv@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58"
BASE_DIGEST = BASE_IMAGE.rsplit("@", 1)[1]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=paths.repo_root(), check=True, capture_output=True,
                          text=True).stdout.strip()


def make_bundle(cases: list[str], directory: Path) -> dict:
    """Write repo.bundle (HEAD plus protocol tags), cases.txt and hf_job.sh into ``directory``."""
    if _git("status", "--porcelain", "--untracked-files=no", "--", "src", "envs", "protocol", "docker", "data"):
        raise ValueError("commit src/, envs/, protocol/, docker/ and data/ first: the Job runs the committed state")
    if directory.exists():
        raise FileExistsError(f"{directory} already exists; use a new bundle folder")
    directory.mkdir(parents=True)
    commit = _git("rev-parse", "HEAD")
    _git("bundle", "create", str(directory / "repo.bundle"), "HEAD", "--tags")
    (directory / "cases.txt").write_text("\n".join(cases) + "\n", encoding="utf-8", newline="\n")
    shutil.copyfile(paths.repo_root() / "docker" / "hf_job.sh", directory / "hf_job.sh")
    manifest = {"commit": commit, "cases": cases, "base_image": BASE_IMAGE}
    (directory / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def job_spec(bundle: Path, *, name: str, bucket: str, flavor: str, timeout: str, keep_arrays: bool = False) -> dict:
    """Everything ``launch`` would send, so it can be shown to the researcher before approval.

    ``keep_arrays`` also exports every run's raw arrays to the bucket (protocol/cloud_reruns.md).
    """
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValueError("job name: lowercase letters, digits and dashes only")
    env = {"SILENTRANK_BASE_DIGEST": BASE_DIGEST, "SILENTRANK_JOB_NAME": name}
    if keep_arrays:
        env["SILENTRANK_KEEP_ARRAYS"] = "1"
    return {
        "image": BASE_IMAGE,
        "command": ["bash", "/bundle/hf_job.sh"],
        "flavor": flavor,
        "timeout": timeout,
        "env": env,
        "volumes": [f"{bundle.resolve()}:/bundle:ro", f"hf://buckets/{bucket}:/out"],
    }


def launch(spec: dict) -> str:
    """Start the Job with the researcher's local Hugging Face login. Returns the job id."""
    from huggingface_hub import Volume, run_job

    bundle_source = spec["volumes"][0].rsplit(":/bundle", 1)[0]
    bucket = spec["volumes"][1].removeprefix("hf://buckets/").rsplit(":/out", 1)[0]
    job = run_job(
        image=spec["image"], command=spec["command"], flavor=spec["flavor"], timeout=spec["timeout"],
        env=spec["env"],
        volumes=[Volume(type="local", source=bundle_source, mount_path="/bundle", read_only=True),
                 Volume(type="bucket", source=bucket, mount_path="/out")],
    )
    return job.id
