"""Protocol v0 decisions at each predeclared practical tolerance."""

from math import isfinite
from typing import Literal

VERDICT_TOLERANCES = (0.002, 0.005, 0.010)
Verdict = Literal["contract_failure", "harmful", "equivalent", "improved", "inconclusive"]


def classify_verdict(
    low: float | None,
    high: float | None,
    delta: float,
    *,
    contract_failure: bool = False,
) -> Verdict:
    """Classify a two-sided interval for old-minus-new loss at one delta."""
    if not isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")
    if contract_failure:
        return "contract_failure"
    if low is None and high is None:
        return "inconclusive"
    if low is None or high is None or not isfinite(low) or not isfinite(high) or low > high:
        raise ValueError("an interval must have finite ordered bounds")
    if low > delta:
        return "harmful"
    if high < -delta:
        return "improved"
    if low >= -delta and high <= delta:
        return "equivalent"
    return "inconclusive"
