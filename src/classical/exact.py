"""Exact dynamic-programming enumeration of the small discrete term model."""

from __future__ import annotations

import numpy as np

from src.hazard_model import PortfolioModel
from src.payout_engine import ContractTerms
from src.risk_metrics import METRIC_KEYS, RunResult, summarise_pmf
from src.scenarios import DISCLAIMER


def enumerate_country_distribution(
    probability: np.ndarray,
    contract_a: ContractTerms,
    contract_b: ContractTerms,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Propagate the annual joint distribution across the contract term.

    The state is the earthquake amount paid so far in each country and the
    highest precipitation step reached so far. Years are independent, so the
    update is exact.
    """
    if contract_a.term_years != contract_b.term_years:
        raise ValueError("Contracts must share a term.")
    coords = np.argwhere(probability > 0.0)
    atom_p = probability[probability > 0.0]
    eq_a = coords[:, 0]
    eq_b = coords[:, 1]
    rain_a = coords[:, 2]
    rain_b = coords[:, 3]
    step_eq_a = contract_a.eq_step_amounts()
    step_eq_b = contract_b.eq_step_amounts()
    step_rain_a = contract_a.precip_step_amounts()
    step_rain_b = contract_b.precip_step_amounts()

    states: dict[tuple[int, int, int, int], float] = {(0, 0, 0, 0): 1.0}
    transitions = 0
    for _year in range(contract_a.term_years):
        updated: dict[tuple[int, int, int, int], float] = {}
        for (paid_eq_a, paid_eq_b, paid_rain_a, paid_rain_b), mass in states.items():
            next_eq_a = np.minimum(paid_eq_a + step_eq_a[eq_a], contract_a.eq_cap)
            next_eq_b = np.minimum(paid_eq_b + step_eq_b[eq_b], contract_b.eq_cap)
            next_rain_a = np.maximum(paid_rain_a, step_rain_a[rain_a])
            next_rain_b = np.maximum(paid_rain_b, step_rain_b[rain_b])
            contributions = mass * atom_p
            for index, contribution in enumerate(contributions):
                key = (
                    int(next_eq_a[index]),
                    int(next_eq_b[index]),
                    int(next_rain_a[index]),
                    int(next_rain_b[index]),
                )
                updated[key] = updated.get(key, 0.0) + float(contribution)
            transitions += len(atom_p)
        states = updated

    loss_a = np.empty(len(states), dtype=np.int64)
    loss_b = np.empty(len(states), dtype=np.int64)
    probs = np.empty(len(states), dtype=float)
    for index, ((paid_eq_a, paid_eq_b, paid_rain_a, paid_rain_b), mass) in enumerate(states.items()):
        loss_a[index] = min(contract_a.contract_cap, paid_eq_a + paid_rain_a)
        loss_b[index] = min(contract_b.contract_cap, paid_eq_b + paid_rain_b)
        probs[index] = mass
    return loss_a, loss_b, probs, transitions


def estimate(model: PortfolioModel, threshold: float) -> RunResult:
    """Exact metrics for the scenario model. The seed is not used."""
    loss_a, loss_b, probs, transitions = enumerate_country_distribution(
        model.annual.probability, model.contracts[0], model.contracts[1]
    )
    max_a, max_b = model.maximum_payouts()
    metrics, uncertainty, country_mean, country_se, pmf = summarise_pmf(
        loss_a, loss_b, probs, threshold, max_a, max_b, model.countries
    )
    return RunResult(
        method="exact",
        method_label="Exact enumeration",
        scenario_id=model.scenario_id,
        scenario_name=model.scenario_name,
        seed=None,
        n_samples=None,
        n_evaluations=int(model.annual.probability.size),
        evaluation_unit="annual joint atoms in the probability tensor",
        threshold=float(threshold),
        currency=model.currency,
        model_version=model.model_version,
        valid_estimate=True,
        metrics=metrics,
        uncertainty=uncertainty,
        country_expected_loss=country_mean,
        country_expected_loss_se=country_se,
        notes=[
            "Exact probabilities for the discrete term model. Floating-point rounding is the only numerical error.",
            "This is the reference for the other estimators. It is not an official benchmark target.",
            DISCLAIMER,
        ],
        details={
            "n_terminal_states": int(len(probs)),
            "n_atoms": int(model.annual.n_atoms),
            "state_transitions": int(transitions),
            "probability_sum": float(probs.sum()),
            "abs_error": {key: 0.0 for key in METRIC_KEYS},
        },
        loss_pmf=pmf,
        countries=model.countries,
    )
