"""Tensor-train feasibility experiment."""

from __future__ import annotations

import pytest

from src.quantum_inspired import tensor_network
from tests.conftest import THRESHOLD


def test_full_bond_contraction_matches_the_annual_table(base_model):
    quimb = pytest.importorskip("quimb")
    assert quimb is not None
    report = tensor_network.analyse(base_model.annual.probability)
    assert report["quimb_matches_numpy"]
    assert "cotengra" in report["optimizer"]
    assert report["valid_for_metrics"]
    assert report["full"]["max_abs_error"] < 1e-8
    assert report["full"]["l1_error"] < 1e-8
    assert report["full"]["n_negative"] == 0
    assert report["sequential_query_max_abs_error"] < 1e-8
    assert report["full"]["parameter_count"] > 0
    assert any(row["l1_error"] > 1e-4 for row in report["truncation"])


def test_valid_tensor_network_metrics_match_enumeration(base_model, exact_base):
    pytest.importorskip("quimb")
    result = tensor_network.estimate(base_model, THRESHOLD, seed=1)
    assert result.valid_estimate
    assert result.method == "tensor_network"
    for key, truth in exact_base.metrics.items():
        if "loss" in key:
            assert result.metrics[key] == pytest.approx(truth, abs=1.0)
        else:
            assert result.metrics[key] == pytest.approx(truth, abs=1e-8)
    text = " ".join(result.notes).lower()
    assert "not a quantum" in text or "not a faster" in text
    assert "truncat" in text
