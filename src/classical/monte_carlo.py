"""Independent Monte Carlo for the term loss distribution."""

from __future__ import annotations

import numpy as np

from src.hazard_model import PortfolioModel, sample_term
from src.payout_engine import portfolio_losses
from src.risk_metrics import RunResult, summarise_samples
from src.scenarios import DISCLAIMER


def estimate(model: PortfolioModel, threshold: float, seed: int, n_samples: int) -> RunResult:
    """Estimate the five risk metrics from ``n_samples`` independent terms."""
    if n_samples < 2:
        raise ValueError("Monte Carlo needs at least 2 samples to report a standard error.")
    rng = np.random.default_rng(seed)
    eq_a, eq_b, rain_a, rain_b = sample_term(model.annual, model.term_years, n_samples, rng)
    loss_a, loss_b = portfolio_losses(eq_a, eq_b, rain_a, rain_b, model.contracts[0], model.contracts[1])
    max_a, max_b = model.maximum_payouts()
    metrics, uncertainty, country_mean, country_se = summarise_samples(
        loss_a, loss_b, threshold, max_a, max_b, model.countries
    )
    portfolio = (loss_a + loss_b).astype(float)
    return RunResult(
        method="monte_carlo",
        method_label="Monte Carlo",
        scenario_id=model.scenario_id,
        scenario_name=model.scenario_name,
        seed=int(seed),
        n_samples=int(n_samples),
        n_evaluations=int(n_samples),
        evaluation_unit="independent term scenarios",
        threshold=float(threshold),
        currency=model.currency,
        model_version=model.model_version,
        valid_estimate=True,
        metrics=metrics,
        uncertainty=uncertainty,
        country_expected_loss=country_mean,
        country_expected_loss_se=country_se,
        notes=[
            "Independent Monte Carlo. Standard errors are the usual sample standard errors.",
            "A stochastic estimate is not a guaranteed payout.",
            DISCLAIMER,
        ],
        details={"generator": "numpy.random.Generator", "annual_draws": int(n_samples * model.term_years)},
        sample_losses=portfolio,
        sample_weights=np.full(n_samples, 1.0 / n_samples),
        countries=model.countries,
    )
