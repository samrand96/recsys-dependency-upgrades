"""Run released upstream tests against frozen packages with separate test tooling."""

from __future__ import annotations

import hashlib
import http.client
import io
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

TEST_TIMEOUT_S = 300
TEST_TOOLS = ("pytest==8.4.2", "hypothesis==6.135.20", "pytest-timeout==2.4.0")
PACKAGE_TOOLS = {
    "numpy": ("meson==1.9.1", "ninja==1.13.0"),
    "scipy": ("pooch==1.8.2", "mpmath==1.3.0"),
}
DEFAULT_MARKERS = {
    "numpy": "not slow",
    "scipy": "not slow",
    "pandas": "not slow and not network and not db",
}


def release_tests(sdist: dict, directory: Path) -> Path | None:
    """Extract only tests and pytest configuration from the lockfile's exact sdist."""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(sdist["url"], timeout=30) as response:
                source = response.read()
            break
        except (OSError, http.client.HTTPException) as exc:
            if attempt == 2:
                raise OSError(f"source distribution download failed: {exc}") from exc
    digest = hashlib.sha256(source).hexdigest()
    if sdist["hash"] != f"sha256:{digest}":
        raise ValueError("source archive checksum differs from the environment lock")

    with tarfile.open(fileobj=io.BytesIO(source), mode="r:*") as archive:
        root = None
        members = []
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if len(parts) < 2 or not member.isfile():
                continue
            if parts[1] == "tests":
                members.append(member)
                root = parts[0]
            elif len(parts) == 2 and parts[1] in {"pyproject.toml", "pytest.ini", "setup.cfg", "conftest.py"}:
                members.append(member)
        if root is None:
            return None
        archive.extractall(directory, members=members, filter="data")
    return directory / root / "tests"


def report(exit_code: int, report_path: Path) -> dict:
    if exit_code not in (0, 1):
        reason = "suite exceeded time budget" if exit_code == 124 else f"pytest exit {exit_code}"
        return {"status": "not_runnable", "reason": reason}
    if not report_path.is_file():
        return {"status": "not_runnable", "reason": "no pytest report"}
    try:
        suites = ET.parse(report_path).getroot().iter("testsuite")
        totals = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
        for suite in suites:
            for key in totals:
                totals[key] += int(suite.attrib[key])
    except (ET.ParseError, KeyError, ValueError):
        return {"status": "not_runnable", "reason": "could not parse the pytest report"}

    n_fail = totals["failures"] + totals["errors"]
    n_pass = totals["tests"] - n_fail - totals["skipped"]
    counts = {"n_pass": n_pass, "n_fail": n_fail, "n_skip": totals["skipped"]}
    if exit_code == 0 and n_pass > 0 and n_fail == 0:
        return {"status": "passed", **counts}
    if exit_code == 1 and n_fail > 0:
        return {"status": "failed", **counts}
    return {"status": "not_runnable", "reason": f"pytest exit {exit_code}", **counts}
