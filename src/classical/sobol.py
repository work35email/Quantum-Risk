"""Randomised scrambled Sobol comparison for the same term functional."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from src.hazard_model import UNIFORMS_PER_YEAR, PortfolioModel, severities_from_uniforms
from src.payout_engine import portfolio_losses
from src.risk_metrics import RunResult, summarise_samples
from src.scenarios import DISCLAIMER


def estimate(model: PortfolioModel, threshold: float, seed: int, n_samples: int) -> RunResult:
    """Scrambled Sobol estimator with an error estimate from independent scrambles.

    The budget is split across independent Owen-scrambled Sobol sequences.
    Each sequence estimates the metrics with equal weights. The reported
    estimate is the sample-size-weighted average of the scramble estimates.
    The standard error is the standard deviation of those scramble estimates
    divided by sqrt(number of scrambles). It is a randomised-QMC error
    estimate, not an IID Monte Carlo standard error.
    """
    if n_samples < 2:
        raise ValueError("Sobol sampling needs at least 2 points.")
    n_scrambles = 8 if n_samples >= 256 else max(1, n_samples // 32)
    n_scrambles = max(1, min(n_scrambles, n_samples))
    points = n_samples // n_scrambles
    if points < 1:
        points = 1
        n_scrambles = n_samples
    effective = points * n_scrambles
    dimension = model.term_years * UNIFORMS_PER_YEAR
    max_a, max_b = model.maximum_payouts()

    scramble_metrics = []
    losses = []
    weights = []
    for scramble in range(n_scrambles):
        engine = qmc.Sobol(d=dimension, scramble=True, seed=int(seed) + scramble)
        uniforms = np.clip(engine.random(points), 1e-12, 1.0 - 1e-12)
        uniforms = uniforms.reshape(points, model.term_years, UNIFORMS_PER_YEAR)
        eq_a, eq_b, rain_a, rain_b = severities_from_uniforms(model.annual, uniforms)
        loss_a, loss_b = portfolio_losses(eq_a, eq_b, rain_a, rain_b, model.contracts[0], model.contracts[1])
        metrics, _uncertainty, country_mean, _country_se = summarise_samples(
            loss_a, loss_b, threshold, max_a, max_b, model.countries
        )
        scramble_metrics.append((metrics, country_mean))
        portfolio = (loss_a + loss_b).astype(float)
        losses.append(portfolio)
        weights.append(np.full(points, 1.0 / effective))

    metrics = {}
    uncertainty = {}
    for key in scramble_metrics[0][0]:
        values = np.array([item[0][key] for item in scramble_metrics], dtype=float)
        metrics[key] = float(values.mean())
        uncertainty[key] = float(values.std(ddof=1) / np.sqrt(n_scrambles)) if n_scrambles > 1 else 0.0
    country_mean = {}
    country_se = {}
    for country in model.countries:
        values = np.array([item[1][country] for item in scramble_metrics], dtype=float)
        country_mean[country] = float(values.mean())
        country_se[country] = float(values.std(ddof=1) / np.sqrt(n_scrambles)) if n_scrambles > 1 else 0.0

    return RunResult(
        method="sobol",
        method_label="Scrambled Sobol",
        scenario_id=model.scenario_id,
        scenario_name=model.scenario_name,
        seed=int(seed),
        n_samples=int(effective),
        n_evaluations=int(effective),
        evaluation_unit="scrambled Sobol term scenarios",
        threshold=float(threshold),
        currency=model.currency,
        model_version=model.model_version,
        valid_estimate=True,
        metrics=metrics,
        uncertainty=uncertainty,
        country_expected_loss=country_mean,
        country_expected_loss_se=country_se,
        notes=[
            "SciPy Owen-scrambled Sobol sequences, equal weights within each scramble.",
            (
                f"Budget {n_samples} produced {n_scrambles} scrambles of {points} points "
                f"({effective} term scenarios)."
            ),
            "Uncertainty is the standard error across independent scrambles, not an IID standard error.",
            "A stochastic estimate is not a guaranteed payout.",
            DISCLAIMER,
        ],
        details={
            "scramble": "scipy.stats.qmc.Sobol scramble=True",
            "dimension": dimension,
            "n_scrambles": n_scrambles,
            "points_per_scramble": points,
            "effective_samples": effective,
            "requested_samples": int(n_samples),
        },
        sample_losses=np.concatenate(losses),
        sample_weights=np.concatenate(weights),
        countries=model.countries,
    )
