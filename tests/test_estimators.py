"""Estimator correctness, uncertainty and reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from src.classical import exact, importance_sampling, monte_carlo, sobol, stratified
from src.hazard_model import sample_term
from src.risk_metrics import summarise_pmf
from src.scenarios import build_portfolio
from tests.conftest import THRESHOLD


def _within_sampling_error(estimate: float, truth: float, standard_error: float, absolute_floor: float) -> None:
    gap = abs(estimate - truth)
    limit = max(5.0 * standard_error, absolute_floor)
    assert gap <= limit, f"gap {gap} exceeds {limit} (estimate {estimate}, truth {truth}, se {standard_error})"


def test_exact_enumeration_is_reproducible(base_model, exact_base):
    again = exact.estimate(base_model, THRESHOLD)
    assert again.metrics == exact_base.metrics
    assert abs(exact_base.details["probability_sum"] - 1.0) < 1e-10
    assert exact_base.metrics["expected_loss"] > 0.0
    assert 0.0 < exact_base.metrics["joint_attachment_probability"] < 1.0


def test_exceedance_uses_a_strict_threshold():
    metrics, _, _, _, _ = summarise_pmf(
        loss_a=np.array([25_000_000.0]),
        loss_b=np.array([0.0]),
        probability=np.array([1.0]),
        threshold=25_000_000.0,
        max_a=25_000_000,
        max_b=0,
        countries=("Kyrgyz Republic", "Tajikistan"),
    )
    assert metrics["exceedance_probability"] == 0.0
    assert metrics["expected_stop_loss"] == 0.0
    assert metrics["both_exhausted_probability"] == 1.0


def test_dependence_changes_joint_metrics_not_expected_loss(bundle, base_model):
    from src.classical.exact import estimate

    independent = estimate(build_portfolio(bundle, "independent"), THRESHOLD)
    base = estimate(base_model, THRESHOLD)
    shock = estimate(build_portfolio(bundle, "common_shock"), THRESHOLD)
    assert independent.metrics["joint_attachment_probability"] < base.metrics["joint_attachment_probability"]
    assert base.metrics["joint_attachment_probability"] < shock.metrics["joint_attachment_probability"]
    assert independent.metrics["both_exhausted_probability"] <= shock.metrics["both_exhausted_probability"]
    assert independent.metrics["expected_loss"] == pytest.approx(shock.metrics["expected_loss"], abs=0.05)


def test_rain_scenarios_increase_expected_loss(bundle):
    from src.classical.exact import estimate

    base = estimate(build_portfolio(bundle, "base"), THRESHOLD)
    stress = estimate(build_portfolio(bundle, "rain_stress"), THRESHOLD)
    cover = estimate(build_portfolio(bundle, "rain_cover"), THRESHOLD)
    assert stress.metrics["expected_loss"] > base.metrics["expected_loss"]
    assert cover.metrics["expected_loss"] > base.metrics["expected_loss"]


def test_monte_carlo_matches_exact_and_repeats(base_model, exact_base):
    first = monte_carlo.estimate(base_model, THRESHOLD, seed=7, n_samples=12000)
    second = monte_carlo.estimate(base_model, THRESHOLD, seed=7, n_samples=12000)
    assert np.array_equal(first.sample_losses, second.sample_losses)
    for key, floor in (
        ("expected_loss", 250_000.0),
        ("joint_attachment_probability", 0.01),
        ("exceedance_probability", 0.01),
        ("expected_stop_loss", 250_000.0),
        ("both_exhausted_probability", 0.01),
    ):
        _within_sampling_error(first.metrics[key], exact_base.metrics[key], first.uncertainty[key], floor)


def test_sampler_matches_annual_marginals(base_model):
    rng = np.random.default_rng(11)
    eq_a, eq_b, rain_a, rain_b = sample_term(base_model.annual, 1, 20000, rng)
    observed = float(np.mean(eq_a[:, 0] > 0))
    truth = base_model.annual.occurrence_eq[0]
    standard_error = (truth * (1.0 - truth) / 20000) ** 0.5
    _within_sampling_error(observed, truth, standard_error, 0.005)


def test_sobol_and_stratified_match_exact(base_model, exact_base):
    sobol_result = sobol.estimate(base_model, THRESHOLD, seed=9, n_samples=4096)
    stratified_result = stratified.estimate(base_model, THRESHOLD, seed=9, n_samples=4096)
    assert stratified_result.details["allocation_sum"] == 4096
    assert stratified_result.details["probability_mass"] == pytest.approx(1.0, abs=1e-10)
    for result, money_floor, prob_floor in (
        (sobol_result, 400_000.0, 0.02),
        (stratified_result, 300_000.0, 0.015),
    ):
        for key in exact_base.metrics:
            floor = money_floor if "loss" in key else prob_floor
            _within_sampling_error(result.metrics[key], exact_base.metrics[key], result.uncertainty[key], floor)


def test_importance_sampling_is_weighted_and_unbiased(base_model, exact_base):
    result = importance_sampling.estimate(base_model, THRESHOLD, seed=13, n_samples=12000)
    assert result.details["mean_weight"] != 1.0 or result.details["ess"] < 12000
    _within_sampling_error(result.details["mean_weight"], 1.0, result.details["se_weight"], 0.02)
    for key, floor in (
        ("expected_loss", 400_000.0),
        ("joint_attachment_probability", 0.02),
        ("exceedance_probability", 0.02),
        ("expected_stop_loss", 400_000.0),
        ("both_exhausted_probability", 0.02),
    ):
        _within_sampling_error(result.metrics[key], exact_base.metrics[key], result.uncertainty[key], floor)
    assert "likelihood" in " ".join(result.notes).lower()


def test_loss_chart_has_a_trace(exact_base):
    from src.visualisation import country_loss_figure, loss_distribution_figure

    figure = loss_distribution_figure(exact_base, portfolio_cap=160_000_000)
    country = country_loss_figure(exact_base)
    assert figure.data
    assert country.data

