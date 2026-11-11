"""Saved run metadata can reproduce a result."""

from __future__ import annotations

import json

from src.experiment_tracker import load_history, run_experiment
from src.scenarios import build_portfolio
from tests.conftest import THRESHOLD


def test_saved_seed_and_config_reproduce_monte_carlo(bundle, tmp_path):
    model = build_portfolio(bundle, "independent")
    database = tmp_path / "experiments.sqlite"
    first = run_experiment(model, "monte_carlo", THRESHOLD, seed=21, n_samples=1500, db_path=database)
    second = run_experiment(
        model, "monte_carlo", THRESHOLD, seed=21, n_samples=1500, db_path=database, persist=False
    )
    assert first.sample_losses is not None
    assert second.sample_losses is not None
    assert list(first.sample_losses) == list(second.sample_losses)
    history = load_history(database)
    assert len(history) == 1
    assert int(history.loc[0, "seed"]) == 21
    assert history.loc[0, "scenario_id"] == "independent"
    config = json.loads(history.loc[0, "config_json"])
    assert config["requested_seed"] == 21
    assert config["scenario"]["dependence_mode"] == "independent"
    assert config["scenario"]["shock_share"] == 0.0
    assert "disclaimer" in config
    assert abs(history.loc[0, "expected_loss"] - first.metrics["expected_loss"]) < 1e-6
    assert history.loc[0, "abs_error_expected_loss"] is not None
