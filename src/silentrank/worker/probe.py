"""Describe the environment and host this interpreter runs on, as one JSON object on stdout.

Run with the Python of an environment under test:  python probe.py
Standard library only at import time; NumPy and threadpoolctl are optional.
"""

from __future__ import annotations

import contextlib
import argparse
import importlib
import importlib.metadata
import io
import json
import os
import platform
import sys
from pathlib import Path


def _cpuinfo() -> dict:
    info = {"model": None, "flags": [], "logical_cpus": os.cpu_count()}
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip()
                if key == "model name" and info["model"] is None:
                    info["model"] = value
                elif key == "flags" and not info["flags"]:
                    info["flags"] = sorted(value.split())
    except OSError:
        info["model"] = platform.processor() or None
    return info


def _mem_total_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 2)
    except OSError:
        pass
    return None


def _numpy() -> dict:
    try:
        import numpy as np
    except Exception as exc:  # the import failure itself is the finding
        return {"import_error": f"{type(exc).__name__}: {exc}"}
    out: dict = {"version": np.__version__}
    for module in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            features = __import__(module, fromlist=["__cpu_features__"]).__cpu_features__
            out["simd_enabled"] = sorted(name for name, on in features.items() if on)
            break
        except Exception:
            continue
    try:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            np.show_runtime()
        out["show_runtime"] = buffer.getvalue()
    except Exception as exc:
        out["show_runtime_error"] = f"{type(exc).__name__}: {exc}"
    try:
        out["default_int"] = str(np.array([1]).dtype)
    except Exception:
        pass
    return out


def _threadpools() -> list | str:
    try:
        from threadpoolctl import threadpool_info

        return threadpool_info()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"


def _package(distribution: str) -> dict:
    module_name = {"scikit-surprise": "surprise"}.get(distribution, distribution.replace("-", "_"))
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            module = importlib.import_module(module_name)
        location = Path(module.__file__).resolve()
        return {
            "version": importlib.metadata.version(distribution),
            "module": module_name,
            "path": str(location.parent),
            "tests_bundled": any(location.parent.rglob("test_*.py")) or any(location.parent.rglob("*_test.py")),
            "import_stdout": captured.getvalue(),
        }
    except Exception as exc:
        return {"module": module_name, "import_error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", nargs="+", default=["numpy"])
    args = parser.parse_args()
    report = {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu": _cpuinfo(),
        "mem_total_gb": _mem_total_gb(),
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "numpy": _numpy(),
        "packages": {package: _package(package) for package in args.packages},
        "threadpools": _threadpools(),
    }
    json.dump(report, sys.stdout, sort_keys=True)
    failed = "import_error" in report["numpy"] or any(
        "import_error" in info for info in report["packages"].values()
    )
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
