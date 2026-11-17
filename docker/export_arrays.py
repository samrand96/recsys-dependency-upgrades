"""Export one case's raw run artifacts from a Hugging Face Job (protocol/cloud_reruns.md).

Usage, inside the Job after a case has run:

    python docker/export_arrays.py --case N2-P2-als --out /out/<job>/arrays

Writes <out>/<case>.tar with every run directory of the case (model, scores, masks, native
and canonical lists, stage arrays, logs) and <out>/<case>.json with the SHA-256 of the tar
and of every file in it. stages/S2.npz is left out: it is the interaction matrix rebuilt
from the frozen data in every run, its hash stays in stage_hashes.json, and it can be
regenerated from data/. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

SKIP = {"stages/S2.npz"}


class _HashingWriter:
    """Sequential writer that hashes what it writes, for stores that cannot seek."""

    def __init__(self, path: Path) -> None:
        self._file = path.open("wb")
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        self.size += len(data)
        return self._file.write(data)

    def close(self) -> None:
        self._file.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export(case: str, runs_root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for directory in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        meta = directory / "meta.json"
        if meta.is_file() and json.loads(meta.read_text(encoding="utf-8")).get("case_id") == case:
            runs.append(directory)
    writer = _HashingWriter(out / f"{case}.tar")
    files: dict[str, dict[str, str]] = {}
    with tarfile.open(fileobj=writer, mode="w|") as archive:
        for directory in runs:
            listing = {}
            for path in sorted(p for p in directory.rglob("*") if p.is_file()):
                relative = path.relative_to(directory).as_posix()
                if relative in SKIP:
                    continue
                listing[relative] = _sha256(path)
                archive.add(path, arcname=f"{directory.name}/{relative}", recursive=False)
            files[directory.name] = listing
    writer.close()
    manifest = {"case": case, "runs": len(runs), "tar_sha256": writer.digest.hexdigest(), "tar_bytes": writer.size,
                "skipped": sorted(SKIP), "files": files}
    (out / f"{case}.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runs", type=Path, default=Path("results/runs"))
    args = parser.parse_args()
    result = export(args.case, args.runs, args.out)
    print(json.dumps({k: result[k] for k in ("case", "runs", "tar_sha256", "tar_bytes")}))
