"""Shared fixtures for the demo model."""

from __future__ import annotations

import pytest

from src.data_loader import load_synthetic
from src.scenarios import build_portfolio

THRESHOLD = 25_000_000.0


@pytest.fixture(scope="session")
def bundle():
    return load_synthetic()


@pytest.fixture(scope="session")
def base_model(bundle):
    return build_portfolio(bundle, "base")


@pytest.fixture(scope="session")
def exact_base(base_model):
    from src.classical.exact import estimate

    return estimate(base_model, THRESHOLD)
