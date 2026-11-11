"""Stratified term sampling on annual event strata.

Each year falls in one of four strata: no event, one country affected without
an extreme severity, both countries affected without an extreme severity, or
any extreme severity (earthquake step 3 or precipitation step 2). A term path
is the sequence of annual strata. Paths are sampled with known probabilities.
The estimator is ``sum p_g * mean(f | g)`` after rare paths are merged so each
reported group has at least two draws when the budget allows it.

Samples are not pooled with equal weights. Rare strata are intentionally
oversampled, and the group probabilities put that design back on the original
measure.
"""

from __future__ import annotations

import numpy as np

from src.hazard_model import PortfolioModel
from src.payout_engine import portfolio_losses
from src.risk_metrics import RunResult, stratified_metrics
from src.scenarios import DISCLAIMER

N_STRATA = 4


def atom_stratum(eq_a: int, eq_b: int, rain_a: int, rain_b: int) -> int:
    """Partition one annual atom into a stratum index."""
    hit_a = eq_a > 0 or rain_a > 0
    hit_b = eq_b > 0 or rain_b > 0
    extreme = eq_a == 3 or eq_b == 3 or rain_a == 2 or rain_b == 2
    hits = int(hit_a) + int(hit_b)
    if hits == 0:
        return 0
    if extreme:
        return 3
    if hits == 2:
        return 2
    return 1


def estimate(model: PortfolioModel, threshold: float, seed: int, n_samples: int) -> RunResult:
    """Stratified estimator with known stratum probabilities."""
    if n_samples < 2:
        raise ValueError("Stratified sampling needs at least 2 samples.")
    prepared = _prepare(model)
    groups, min_per = _merge_paths(prepared["path_probability"], n_samples)
    allocation = _allocate(np.array([_mass(prepared["path_probability"], group) for group in groups]), n_samples, min_per)
    rng = np.random.default_rng(seed)
    max_a, max_b = model.maximum_payouts()

    collected = []
    plot_losses = []
    plot_weights = []
    for group, n_group in zip(groups, allocation):
        probability = _mass(prepared["path_probability"], group)
        eq_a, eq_b, rain_a, rain_b = _sample_group(rng, model, prepared, group, int(n_group))
        loss_a, loss_b = portfolio_losses(eq_a, eq_b, rain_a, rain_b, model.contracts[0], model.contracts[1])
        collected.append(
            {
                "probability": probability,
                "loss_a": loss_a.astype(float),
                "loss_b": loss_b.astype(float),
            }
        )
        portfolio = (loss_a + loss_b).astype(float)
        plot_losses.append(portfolio)
        plot_weights.append(np.full(len(portfolio), probability / len(portfolio)))

    metrics, uncertainty, country_mean, country_se = stratified_metrics(
        collected, threshold, max_a, max_b, model.countries
    )
    sample_losses = np.concatenate(plot_losses)
    sample_weights = np.concatenate(plot_weights)
    check = float(np.sum(sample_weights * sample_losses))
    if abs(check - metrics["expected_loss"]) > 1e-5 * max(1.0, abs(metrics["expected_loss"])):
        raise RuntimeError("Stratified display weights do not reproduce the expected-loss estimator.")

    return RunResult(
        method="stratified",
        method_label="Stratified sampling",
        scenario_id=model.scenario_id,
        scenario_name=model.scenario_name,
        seed=int(seed),
        n_samples=int(n_samples),
        n_evaluations=int(n_samples),
        evaluation_unit="stratified term scenarios",
        threshold=float(threshold),
        currency=model.currency,
        model_version=model.model_version,
        valid_estimate=True,
        metrics=metrics,
        uncertainty=uncertainty,
        country_expected_loss=country_mean,
        country_expected_loss_se=country_se,
        notes=[
            "Strata are annual event patterns with probabilities computed from the joint distribution.",
            "The estimate is the probability-weighted average of within-stratum means, not an equal-weight pool.",
            "A stochastic estimate is not a guaranteed payout.",
            DISCLAIMER,
        ],
        details={
            "n_groups": len(groups),
            "min_samples_per_group": int(min_per),
            "allocation_sum": int(np.sum(allocation)),
            "probability_mass": float(sum(group["probability"] for group in collected)),
            "stratum_probabilities": prepared["stratum_probability"].tolist(),
            "annual_draws": int(n_samples * model.term_years),
        },
        sample_losses=sample_losses,
        sample_weights=sample_weights,
        countries=model.countries,
    )


def _prepare(model: PortfolioModel) -> dict:
    coords = np.argwhere(model.annual.probability > 0.0)
    probs = model.annual.probability[model.annual.probability > 0.0]
    strata = np.array(
        [atom_stratum(int(row[0]), int(row[1]), int(row[2]), int(row[3])) for row in coords],
        dtype=np.int64,
    )
    stratum_probability = np.zeros(N_STRATA, dtype=float)
    atoms_by_stratum = []
    conditional = []
    for stratum in range(N_STRATA):
        mask = strata == stratum
        stratum_probability[stratum] = float(probs[mask].sum())
        atoms_by_stratum.append(coords[mask])
        mass = stratum_probability[stratum]
        conditional.append(probs[mask] / mass if mass > 0 else np.array([]))
    n_paths = N_STRATA ** model.term_years
    path_probability = np.empty(n_paths, dtype=float)
    path_strata = np.empty((n_paths, model.term_years), dtype=np.int64)
    for index in range(n_paths):
        digits = []
        remainder = index
        probability = 1.0
        for _year in range(model.term_years):
            stratum = remainder % N_STRATA
            digits.append(stratum)
            probability *= stratum_probability[stratum]
            remainder //= N_STRATA
        path_strata[index] = digits
        path_probability[index] = probability
    return {
        "atoms_by_stratum": atoms_by_stratum,
        "conditional": conditional,
        "stratum_probability": stratum_probability,
        "path_probability": path_probability,
        "path_strata": path_strata,
    }


def _merge_paths(path_probability: np.ndarray, n_samples: int) -> tuple[list[list[int]], int]:
    active = [index for index, probability in enumerate(path_probability) if probability > 0.0]
    groups = [[index] for index in active]
    min_per = 2 if n_samples >= 2 else 1

    def mass(group: list[int]) -> float:
        return float(sum(path_probability[index] for index in group))

    while len(groups) * min_per > n_samples and len(groups) > 1:
        groups.sort(key=mass)
        groups[1] = groups[0] + groups[1]
        del groups[0]
    if len(groups) * min_per > n_samples:
        min_per = 1
    return groups, min_per


def _mass(path_probability: np.ndarray, group: list[int]) -> float:
    return float(sum(path_probability[index] for index in group))


def _allocate(probabilities: np.ndarray, n_samples: int, min_per: int) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    count = len(probabilities)
    base = np.full(count, min_per, dtype=int)
    if int(base.sum()) > n_samples:
        raise ValueError("Sample budget cannot cover the stratified groups.")
    remaining = n_samples - int(base.sum())
    raw = remaining * probabilities
    extra = np.floor(raw).astype(int)
    shortfall = remaining - int(extra.sum())
    order = np.argsort(-(raw - extra))
    for index in range(shortfall):
        extra[order[index % count]] += 1
    allocation = base + extra
    while int(allocation.sum()) > n_samples:
        donor = int(np.argmax(allocation))
        if allocation[donor] <= min_per:
            break
        allocation[donor] -= 1
    while int(allocation.sum()) < n_samples:
        allocation[int(np.argmax(probabilities))] += 1
    if int(allocation.sum()) != n_samples:
        raise RuntimeError("Stratified allocation does not match the sample budget.")
    return allocation


def _sample_group(
    rng: np.random.Generator,
    model: PortfolioModel,
    prepared: dict,
    group: list[int],
    n_group: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path_probability = np.array([prepared["path_probability"][index] for index in group], dtype=float)
    path_probability = path_probability / path_probability.sum()
    chosen = rng.choice(len(group), size=n_group, p=path_probability)
    years = model.term_years
    eq_a = np.empty((n_group, years), dtype=np.int64)
    eq_b = np.empty((n_group, years), dtype=np.int64)
    rain_a = np.empty((n_group, years), dtype=np.int64)
    rain_b = np.empty((n_group, years), dtype=np.int64)
    for year in range(years):
        strata = np.array([prepared["path_strata"][group[pick]][year] for pick in chosen], dtype=np.int64)
        for stratum in np.unique(strata):
            mask = strata == stratum
            count = int(mask.sum())
            atoms = prepared["atoms_by_stratum"][int(stratum)]
            conditional = prepared["conditional"][int(stratum)]
            picks = rng.choice(len(conditional), size=count, p=conditional)
            selected = atoms[picks]
            eq_a[mask, year] = selected[:, 0]
            eq_b[mask, year] = selected[:, 1]
            rain_a[mask, year] = selected[:, 2]
            rain_b[mask, year] = selected[:, 3]
    return eq_a, eq_b, rain_a, rain_b
