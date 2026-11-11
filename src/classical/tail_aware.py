"""Tail-aware importance sampling for rare joint disasters.

The proposal keeps a strong common shock and tilts severity toward the top
step, so years in which several countries are hit hard are drawn more often
than under the baseline. Metrics use the likelihood ratio p/q. The proposal
is not an independent tilt: that estimator is ``importance_sampling``.
"""

from __future__ import annotations

import numpy as np

from src.hazard_model import PortfolioModel, build_annual_distribution, likelihood_ratios, sample_term, tilt_severity
from src.payout_engine import portfolio_losses
from src.risk_metrics import RunResult, summarise_samples
from src.scenarios import DISCLAIMER

TAIL_OCCURRENCE_MIX = 0.62
TAIL_OCCURRENCE_ANCHOR = 0.68
TAIL_SEVERITY_GEOMETRIC = 2.6
TAIL_SHOCK_SHARE = 0.70


def tail_proposal(annual):
    """Common-shock proposal aimed at joint severe years, with full support."""

    def _tilt(probability: float) -> float:
        mixed = (1.0 - TAIL_OCCURRENCE_MIX) * float(probability) + TAIL_OCCURRENCE_MIX * TAIL_OCCURRENCE_ANCHOR
        return float(np.clip(mixed, 1e-4, 1.0 - 1e-4))

    proposal = build_annual_distribution(
        countries=annual.countries,
        occurrence_eq=(_tilt(annual.occurrence_eq[0]), _tilt(annual.occurrence_eq[1])),
        occurrence_rain=(_tilt(annual.occurrence_rain[0]), _tilt(annual.occurrence_rain[1])),
        severity_eq=(
            tilt_severity(annual.severity_eq[0], TAIL_SEVERITY_GEOMETRIC),
            tilt_severity(annual.severity_eq[1], TAIL_SEVERITY_GEOMETRIC),
        ),
        severity_rain=(
            tilt_severity(annual.severity_rain[0], TAIL_SEVERITY_GEOMETRIC),
            tilt_severity(annual.severity_rain[1], TAIL_SEVERITY_GEOMETRIC),
        ),
        shock_share_eq=TAIL_SHOCK_SHARE,
        shock_share_rain=TAIL_SHOCK_SHARE,
    )
    missing = (annual.probability > 0.0) & (proposal.probability <= 0.0)
    if np.any(missing):
        raise RuntimeError("Tail-aware proposal dropped part of the target support.")
    return proposal


def estimate(model: PortfolioModel, threshold: float, seed: int, n_samples: int) -> RunResult:
    """Likelihood-weighted estimates from the tail-aware proposal."""
    if n_samples < 2:
        raise ValueError("Tail-aware sampling needs at least 2 draws.")
    proposal = tail_proposal(model.annual)
    rng = np.random.default_rng(seed)
    eq_a, eq_b, rain_a, rain_b = sample_term(proposal, model.term_years, n_samples, rng)
    weights = likelihood_ratios(model.annual, proposal, eq_a, eq_b, rain_a, rain_b)
    loss_a, loss_b = portfolio_losses(eq_a, eq_b, rain_a, rain_b, model.contracts[0], model.contracts[1])
    max_a, max_b = model.maximum_payouts()
    metrics, uncertainty, country_mean, country_se = summarise_samples(
        loss_a, loss_b, threshold, max_a, max_b, model.countries, weights=weights
    )
    weight_mean = float(weights.mean())
    weight_se = float(weights.std(ddof=1) / np.sqrt(len(weights)))
    ess = float((weights.sum() ** 2) / np.sum(weights**2))
    notes = [
        "The proposal raises the common-shock share and tilts severity upward so joint disasters are sampled more often.",
        "Metrics use likelihood ratios p/q. Unweighted proposal averages are not reported as results.",
        f"Mean likelihood ratio {weight_mean:.4f} (standard error {weight_se:.4f}); effective sample size {ess:.1f}.",
        "A stochastic estimate is not a guaranteed payout.",
        DISCLAIMER,
    ]
    if ess < 50:
        notes.insert(2, "Effective sample size is low, so the tail-aware variance is high.")
    display_weights = weights / weights.sum() if weights.sum() > 0 else weights
    return RunResult(
        method="tail_aware",
        method_label="Tail-aware proposal",
        scenario_id=model.scenario_id,
        scenario_name=model.scenario_name,
        seed=int(seed),
        n_samples=int(n_samples),
        n_evaluations=int(n_samples),
        evaluation_unit="tail-aware proposal terms, reweighted by p/q",
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
            "occurrence_mix": TAIL_OCCURRENCE_MIX,
            "occurrence_anchor": TAIL_OCCURRENCE_ANCHOR,
            "severity_geometric": TAIL_SEVERITY_GEOMETRIC,
            "proposal_shock_share": TAIL_SHOCK_SHARE,
            "proposal_dependence": "common_shock",
        },
        sample_losses=loss_a + loss_b,
        sample_weights=display_weights,
        countries=model.countries,
    )
