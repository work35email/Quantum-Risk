"""Importance sampling with an independent tilted proposal and likelihood weights.

The proposal raises occurrence probabilities toward an interior value, tilts
severity toward higher steps, and removes the common shock. Every annual atom
in the target support keeps positive proposal probability. Reported metrics use
``mean(likelihood_ratio * f)``. Unweighted proposal averages are not reported
as results.
"""

from __future__ import annotations

import numpy as np

from src.hazard_model import (
    OCCURRENCE_ANCHOR,
    OCCURRENCE_MIX,
    SEVERITY_GEOMETRIC,
    PortfolioModel,
    likelihood_ratios,
    proposal_distribution,
    sample_term,
)
from src.payout_engine import portfolio_losses
from src.risk_metrics import RunResult, summarise_samples
from src.scenarios import DISCLAIMER


def estimate(model: PortfolioModel, threshold: float, seed: int, n_samples: int) -> RunResult:
    """Likelihood-weighted importance-sampling estimates."""
    if n_samples < 2:
        raise ValueError("Importance sampling needs at least 2 draws.")
    proposal = proposal_distribution(model.annual)
    rng = np.random.default_rng(seed)
    eq_a, eq_b, rain_a, rain_b = sample_term(proposal, model.term_years, n_samples, rng)
    weights = likelihood_ratios(model.annual, proposal, eq_a, eq_b, rain_a, rain_b)
    loss_a, loss_b = portfolio_losses(eq_a, eq_b, rain_a, rain_b, model.contracts[0], model.contracts[1])
    max_a, max_b = model.maximum_payouts()
    metrics, uncertainty, country_mean, country_se = summarise_samples(
        loss_a, loss_b, threshold, max_a, max_b, model.countries, weights=weights
    )
    weight_mean, weight_se = _mean_se(weights)
    ess = float((weights.sum() ** 2) / np.sum(weights ** 2))
    notes = [
        "Samples come from an independent tilted proposal. Metrics use likelihood ratios p/q.",
        "Unweighted averages under the proposal are not used as estimates.",
        f"Mean likelihood ratio {weight_mean:.4f} (standard error {weight_se:.4f}); effective sample size {ess:.1f}.",
        "A stochastic estimate is not a guaranteed payout.",
        DISCLAIMER,
    ]
    if ess < 50:
        notes.insert(2, "Effective sample size is low, so the importance-sampling variance is high.")
    display_weights = weights / weights.sum() if weights.sum() > 0 else weights
    return RunResult(
        method="importance_sampling",
        method_label="Importance sampling",
        scenario_id=model.scenario_id,
        scenario_name=model.scenario_name,
        seed=int(seed),
        n_samples=int(n_samples),
        n_evaluations=int(n_samples),
        evaluation_unit="proposal term scenarios, reweighted by p/q",
        threshold=float(threshold),
        currency=model.currency,
        model_version=model.model_version,
        valid_estimate=True,
        metrics=metrics,
        uncertainty=uncertainty,
        country_expected_loss=country_mean,
        country_expected_loss_se=country_se,
        notes=notes,
        details={
            "mean_weight": weight_mean,
            "se_weight": weight_se,
            "ess": ess,
            "occurrence_mix": OCCURRENCE_MIX,
            "occurrence_anchor": OCCURRENCE_ANCHOR,
            "severity_geometric": SEVERITY_GEOMETRIC,
            "proposal_dependence": "independent",
            "proposal_occurrence_eq": list(proposal.occurrence_eq),
            "proposal_occurrence_rain": list(proposal.occurrence_rain),
            "annual_draws": int(n_samples * model.term_years),
        },
        sample_losses=(loss_a + loss_b).astype(float),
        sample_weights=display_weights,
        countries=model.countries,
    )


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    return mean, se
