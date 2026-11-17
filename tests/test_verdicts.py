import pytest

from silentrank.stats.verdict import VERDICT_TOLERANCES, classify_verdict


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (0.006, 0.010, "harmful"),
        (-0.010, -0.006, "improved"),
        (-0.005, 0.005, "equivalent"),
        (0.005, 0.005, "equivalent"),
        (0.005, 0.010, "inconclusive"),
        (-0.010, -0.005, "inconclusive"),
        (-0.010, 0.010, "inconclusive"),
        (None, None, "inconclusive"),
    ],
)
def test_interval_verdicts(low, high, expected):
    assert classify_verdict(low, high, 0.005) == expected


def test_contract_failure_takes_precedence():
    assert classify_verdict(None, None, 0.005, contract_failure=True) == "contract_failure"
    assert classify_verdict(-0.01, -0.009, 0.005, contract_failure=True) == "contract_failure"


def test_report_at_each_protocol_tolerance():
    verdicts = {delta: classify_verdict(0.003, 0.004, delta) for delta in VERDICT_TOLERANCES}
    assert verdicts == {0.002: "harmful", 0.005: "equivalent", 0.010: "equivalent"}


@pytest.mark.parametrize("low, high", [(0.0, None), (1.0, 0.0), (0.0, float("inf"))])
def test_malformed_intervals_are_rejected(low, high):
    with pytest.raises(ValueError, match="finite ordered bounds"):
        classify_verdict(low, high, 0.005)


def test_negative_tolerance_is_rejected():
    with pytest.raises(ValueError, match="delta"):
        classify_verdict(0.0, 0.0, -0.005)
