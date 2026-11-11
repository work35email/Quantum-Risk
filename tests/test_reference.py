"""Regression lock for the illustrative model. These are not external benchmark targets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.classical.exact import estimate
from src.data_loader import REFERENCE_DIR
from src.scenarios import build_portfolio

REFERENCE = REFERENCE_DIR / "reference_results.csv"
THRESHOLD = 25_000_000.0


def test_internal_reference_file_matches_exact_enumeration(bundle):
    if not REFERENCE.exists():
        pytest.fail(f"Missing {REFERENCE}. Generate it from exact enumeration before claiming the demo is locked.")
    frame = pd.read_csv(REFERENCE)
    assert set(frame["source"]) == {"internal_exact_enumeration"}
    assert frame["notes"].str.contains("Not an ADB benchmark").all()
    for scenario_id, group in frame.groupby("scenario_id"):
        result = estimate(build_portfolio(bundle, scenario_id), THRESHOLD)
        for _, row in group.iterrows():
            assert row["threshold_usd"] == THRESHOLD
            actual = result.metrics[row["metric"]]
            assert actual == pytest.approx(row["value"], abs=row["tolerance"])
            if row["metric"] in {"expected_loss", "expected_stop_loss"}:
                assert row["units"] == "USD"
            else:
                assert row["units"] == "probability"
