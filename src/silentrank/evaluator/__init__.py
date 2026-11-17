"""Frozen ranking evaluator for the orchestrator environment."""

from hashlib import sha256
from pathlib import Path

EVALUATOR_VERSION = "1"


def identity() -> str:
    """Identify the evaluator release and its exact Python source contents."""
    digest = sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode("utf-8") + b"\0")
        digest.update(source.read_text(encoding="utf-8").encode("utf-8") + b"\0")
    return f"{EVALUATOR_VERSION}:{digest.hexdigest()}"
