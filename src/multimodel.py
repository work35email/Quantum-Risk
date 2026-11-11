"""Configurable multi-country, multi-hazard portfolio.

The shipped Kyrgyz Republic and Tajikistan fixture keeps using the original
two-country engine, so its reference numbers do not move. Every other country
or hazard set is built here. Dependence within a hazard is the same
Marshall-Olkin common shock, extended from two countries to N. Hazards stay
independent of each other, and years stay independent.

Earthquake payments still accumulate and cap. Precipitation still pays once,
at the highest step. Any further hazard uses the aggregation named in
``peril_terms``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from src import __version__
from src.classical.stratified import _allocate, _mass, _merge_paths
from src.classical.tail_aware import (
    TAIL_OCCURRENCE_ANCHOR,
    TAIL_OCCURRENCE_MIX,
    TAIL_SEVERITY_GEOMETRIC,
    TAIL_SHOCK_SHARE,
)
from src.data_loader import DataBundle
from src.hazard_model import OCCURRENCE_ANCHOR, OCCURRENCE_MIX, SEVERITY_GEOMETRIC, _categorical, tilt_severity
from src.payout_engine import money_fraction
from src.risk_metrics import METRIC_KEYS, RunResult
from src.scenarios import DISCLAIMER, _contract_from_row, _data_labels, get_scenario, validate_bundle

N_STRATA = 4


@dataclass
class HazardLaw:
    """One hazard's occurrence marginals, severity conditionals and shock."""

    name: str
    occurrence: np.ndarray
    severity: tuple[np.ndarray, ...]
    shock_share: float
    alpha: float
    beta: np.ndarray


@dataclass
class Axis:
    """One country-hazard payout coordinate in the annual tensor."""

    hazard: str
    country: str
    aggregation: str
    cap: int
    steps: np.ndarray


@dataclass
class GeneralPortfolio:
    """Scenario model for an arbitrary country and hazard list."""

    scenario_id: str
    scenario_name: str
    description: str
    assumption_label: str
    countries: tuple[str, ...]
    hazard_names: tuple[str, ...]
    term_years: int
    currency: str
    hazards: tuple[HazardLaw, ...]
    axes: tuple[Axis, ...]
    country_caps: dict[str, int]
    probability: np.ndarray
    parameters: dict
    warnings: list[str]
    data_labels: list[str]
    model_version: str

    def maximum_payouts(self) -> tuple[int, ...]:
        maxima = []
        for country in self.countries:
            total = 0
            for axis in self.axes:
                if axis.country != country:
                    continue
                if axis.aggregation == "sum_capped":
                    total += int(axis.cap)
                else:
                    total += int(axis.steps.max())
            maxima.append(int(min(self.country_caps[country], total)))
        return tuple(maxima)

    @property
    def n_countries(self) -> int:
        return len(self.countries)

    @property
    def uniforms_per_year(self) -> int:
        return len(self.hazards) * (1 + 2 * self.n_countries)


def build_general_portfolio(
    bundle: DataBundle,
    scenario_id: str,
    shock_share_override: float | None = None,
    precip_scale_override: float | None = None,
) -> GeneralPortfolio:
    """Validate inputs and build a portfolio for any accepted country set."""
    validation = validate_bundle(bundle)
    if not validation.ok:
        raise ValueError("Invalid input data:\n- " + "\n- ".join(validation.errors))
    scenario = get_scenario(bundle, scenario_id)
    if shock_share_override is not None and not 0.0 <= float(shock_share_override) <= 1.0:
        raise ValueError("Shock-share override must be between 0 and 1.")
    if precip_scale_override is not None and not 0.0 < float(precip_scale_override) <= 5.0:
        raise ValueError("Precipitation-scale override must be in (0, 5].")

    countries = tuple(dict.fromkeys(str(value) for value in bundle.exposure["country"]))
    hazard_names = tuple(dict.fromkeys(str(value).strip().lower() for value in bundle.hazards["hazard"]))
    shock_share = scenario.shock_share if shock_share_override is None else float(shock_share_override)
    dependence_mode = scenario.dependence_mode if shock_share_override is None else (
        "independent" if shock_share == 0.0 else "common_shock"
    )
    if dependence_mode == "independent":
        shock_share = 0.0
    eq_scale = scenario.eq_probability_scale
    precip_scale = scenario.precip_probability_scale if precip_scale_override is None else float(precip_scale_override)

    warnings = list(validation.warnings)
    laws = []
    tensors = []
    for hazard in hazard_names:
        scale = 1.0
        if hazard == "earthquake":
            scale = eq_scale
        elif hazard == "precipitation":
            scale = precip_scale
        occurrence, note = _scaled_occurrence(bundle, countries, hazard, scale)
        if note:
            warnings.append(note)
        severity = tuple(_severity_vector(bundle, country, hazard) for country in countries)
        tensor, alpha, beta = hazard_tensor(occurrence, severity, shock_share)
        laws.append(
            HazardLaw(
                name=hazard,
                occurrence=occurrence,
                severity=severity,
                shock_share=float(shock_share),
                alpha=alpha,
                beta=beta,
            )
        )
        tensors.append(tensor)

    axes, country_caps = _axes_from_contracts(bundle, scenario, countries, hazard_names, warnings)
    probability = combine_hazard_tensors(tensors)
    term_years = _term_years(bundle)
    currency = str(bundle.contracts["currency"].iloc[0]).strip().upper()
    parameters = _parameters(
        scenario,
        countries,
        laws,
        country_caps,
        dependence_mode,
        shock_share,
        eq_scale,
        precip_scale,
        shock_share_override,
        precip_scale_override,
        term_years,
        currency,
    )
    return GeneralPortfolio(
        scenario_id=scenario.scenario_id,
        scenario_name=scenario.scenario_name,
        description=scenario.description,
        assumption_label=scenario.assumption_label,
        countries=countries,
        hazard_names=hazard_names,
        term_years=term_years,
        currency=currency,
        hazards=tuple(laws),
        axes=tuple(axes),
        country_caps=country_caps,
        probability=probability,
        parameters=parameters,
        warnings=warnings,
        data_labels=_labels(bundle, scenario),
        model_version=__version__,
    )


def hazard_tensor(
    occurrence: np.ndarray,
    severity: tuple[np.ndarray, ...],
    shock_share: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Annual severity tensor for one hazard and any number of countries."""
    occurrence = np.asarray(occurrence, dtype=float)
    alpha = float(shock_share) * float(np.min(occurrence))
    if alpha >= 1.0 - 1e-15:
        beta = np.zeros_like(occurrence)
    else:
        beta = (occurrence - alpha) / (1.0 - alpha)
    count = len(occurrence)
    tensor = np.zeros(tuple(len(vector) + 1 for vector in severity), dtype=float)
    rest = 0.0
    all_ones = (1,) * count
    for bits in itertools.product((0, 1), repeat=count):
        if bits == all_ones:
            continue
        probability = (1.0 - alpha) * float(np.prod([value if bit else (1.0 - value) for bit, value in zip(bits, beta)]))
        rest += probability
        vectors = [_conditional_vector(bit, vector) for bit, vector in zip(bits, severity)]
        tensor += probability * _outer(vectors)
    both = 1.0 - rest
    if both < -1e-8:
        raise ValueError("Common-shock probabilities are negative. Check the occurrence rates and shock share.")
    both = max(both, 0.0)
    tensor += both * _outer([_conditional_vector(1, vector) for vector in severity])
    total = float(tensor.sum())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Hazard tensor sums to {total}, not 1.")
    return tensor, float(alpha), np.asarray(beta, dtype=float)


def combine_hazard_tensors(tensors: list[np.ndarray]) -> np.ndarray:
    """Independent product of per-hazard tensors, hazards in listed order."""
    combined = np.ones((), dtype=float)
    for tensor in tensors:
        combined = combined.reshape(combined.shape + (1,) * tensor.ndim) * tensor.reshape((1,) * combined.ndim + tensor.shape)
    combined = np.asarray(combined, dtype=float)
    total = float(combined.sum())
    if abs(total - 1.0) > 1e-7:
        raise ValueError(f"Annual distribution sums to {total}, not 1.")
    return combined


def estimate_general(
    model: GeneralPortfolio,
    method: str,
    threshold: float,
    seed: int,
    n_samples: int,
) -> RunResult:
    """Run one estimator on a general portfolio."""
    if method == "exact":
        return _exact(model, threshold)
    if method == "tensor_network":
        return _tensor(model, threshold, seed)
    if n_samples < 2:
        raise ValueError("Sample budget must be at least 2.")
    if method == "monte_carlo":
        return _monte_carlo(model, threshold, seed, n_samples)
    if method == "sobol":
        return _sobol(model, threshold, seed, n_samples)
    if method == "stratified":
        return _stratified(model, threshold, seed, n_samples)
    if method == "importance_sampling":
        return _reweighted(model, threshold, seed, n_samples, tail=False)
    if method == "tail_aware":
        return _reweighted(model, threshold, seed, n_samples, tail=True)
    raise ValueError(f"Unknown method '{method}'.")


def _exact(model: GeneralPortfolio, threshold: float, probability: np.ndarray | None = None, renormalize: bool = False) -> RunResult:
    table = model.probability if probability is None else np.asarray(probability, dtype=float)
    if table.size > 500:
        return _unavailable(
            model,
            "exact",
            "Exact enumeration",
            threshold,
            (
                f"Exact enumeration supports annual tables up to 500 entries. This model has {int(table.size)}. "
                "Use Monte Carlo, Sobol, stratified sampling or a reweighted proposal."
            ),
        )
    losses, probs, transitions = _enumerate(model, table)
    if renormalize:
        probs = probs / probs.sum()
    maxima = model.maximum_payouts()
    metrics, uncertainty, country_mean, country_se, pmf = _summarise_pmf(losses, probs, threshold, maxima, model.countries)
    return _pack(
        model,
        method="exact",
        label="Exact enumeration",
        threshold=threshold,
        seed=None,
        n_samples=None,
        evaluations=int(transitions),
        unit="annual atoms propagated through the term",
        metrics=metrics,
        uncertainty=uncertainty,
        country_mean=country_mean,
        country_se=country_se,
        notes=[
            "Exact dynamic programming on the discrete term model.",
            "A model estimate is not a guaranteed payout.",
            DISCLAIMER,
        ],
        details={"state_transitions": int(transitions)},
        loss_pmf=pmf,
    )


def _tensor(model: GeneralPortfolio, threshold: float, seed: int) -> RunResult:
    if model.probability.size > 500:
        return _unavailable(
            model,
            "tensor_network",
            "Tensor network (experimental)",
            threshold,
            (
                f"The tensor-train term calculation supports annual tables up to 500 entries. "
                f"This model has {int(model.probability.size)}. Use a sampling method."
            ),
            seed=seed,
        )
    from src.quantum_inspired import tensor_network

    try:
        report = tensor_network.analyse(model.probability)
    except ImportError as exc:
        return _invalid(model, threshold, seed, [f"Quimb is not installed ({exc}).", DISCLAIMER], {"import_error": str(exc)})
    details = {
        "library": report["library"],
        "optimizer": report.get("optimizer"),
        "cotengra": report.get("cotengra"),
        "bond_dimensions": report["full"]["bond_dimensions"],
        "parameter_count": report["full"]["parameter_count"],
        "dense_size": report["dense_size"],
        "l1_error": report["full"]["l1_error"],
        "max_abs_error": report["full"]["max_abs_error"],
        "truncation": report["truncation"],
        "sequential_query_max_abs_error": report["sequential_query_max_abs_error"],
        "valid_for_metrics": report["valid_for_metrics"],
    }
    notes = list(report["limitations"]) + [DISCLAIMER]
    if not report["valid_for_metrics"]:
        notes.insert(0, "The tensor train did not reproduce the annual table, so no risk metrics are reported.")
        return _invalid(model, threshold, seed, notes, details)
    exact = _exact(model, threshold, probability=report["reconstructed"], renormalize=True)
    notes.insert(
        0,
        "Experimental tensor-train check. Cotengra chooses the contraction order. "
        "Full-bond agreement with the table means the term metrics match enumeration. "
        "This is not a quantum computation and not an accuracy or speed advantage.",
    )
    exact.method = "tensor_network"
    exact.method_label = "Tensor network (experimental)"
    exact.seed = int(seed)
    exact.n_evaluations = int(model.probability.size)
    exact.evaluation_unit = "elements of the annual probability tensor"
    exact.notes = notes
    exact.details.update(details)
    return exact


def _monte_carlo(model: GeneralPortfolio, threshold: float, seed: int, n_samples: int) -> RunResult:
    rng = np.random.default_rng(seed)
    uniforms = rng.random((n_samples, model.term_years, model.uniforms_per_year))
    losses = _losses_from_uniforms(model, uniforms, model.hazards)
    return _sample_result(
        model, "monte_carlo", "Monte Carlo", threshold, seed, n_samples, losses, None,
        "independent Monte Carlo terms",
        ["Independent Monte Carlo terms drawn from the annual model.", "A stochastic estimate is not a guaranteed payout.", DISCLAIMER],
        {},
    )


def _sobol(model: GeneralPortfolio, threshold: float, seed: int, n_samples: int) -> RunResult:
    from scipy.stats import qmc

    dimension = model.term_years * model.uniforms_per_year
    engine = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    points = engine.random(int(n_samples))
    uniforms = points.reshape(n_samples, model.term_years, model.uniforms_per_year)
    losses = _losses_from_uniforms(model, uniforms, model.hazards)
    return _sample_result(
        model, "sobol", "Scrambled Sobol", threshold, seed, n_samples, losses, None,
        "scrambled Sobol term scenarios",
        ["Scrambled Sobol points mapped through the same annual uniforms as Monte Carlo.", "A stochastic estimate is not a guaranteed payout.", DISCLAIMER],
        {"dimension": dimension},
    )


def _reweighted(model: GeneralPortfolio, threshold: float, seed: int, n_samples: int, tail: bool) -> RunResult:
    proposal_hazards, proposal_probability = _proposal(model, tail=tail)
    missing = (model.probability > 0.0) & (proposal_probability <= 0.0)
    if np.any(missing):
        raise RuntimeError("Proposal is missing support of the target model.")
    rng = np.random.default_rng(seed)
    uniforms = rng.random((n_samples, model.term_years, model.uniforms_per_year))
    levels = _levels_from_uniforms(proposal_hazards, model.n_countries, uniforms)
    weights = _likelihood(model.probability, proposal_probability, levels)
    losses = _apply_levels(model, levels)
    if tail:
        method, label = "tail_aware", "Tail-aware proposal"
        unit = "tail-aware proposal terms, reweighted by p/q"
        lead = "The proposal raises the common-shock share and tilts severity so joint disasters are sampled more often."
        details = {"proposal_shock_share": TAIL_SHOCK_SHARE, "proposal_dependence": "common_shock"}
    else:
        method, label = "importance_sampling", "Importance sampling"
        unit = "proposal term scenarios, reweighted by p/q"
        lead = "Samples come from an independent tilted proposal. Metrics use likelihood ratios p/q."
        details = {"proposal_dependence": "independent", "occurrence_mix": OCCURRENCE_MIX}
    weight_mean = float(weights.mean())
    weight_se = float(weights.std(ddof=1) / np.sqrt(len(weights)))
    ess = float((weights.sum() ** 2) / np.sum(weights**2))
    details.update({"mean_weight": weight_mean, "se_weight": weight_se, "ess": ess})
    notes = [
        lead,
        "Unweighted averages under the proposal are not used as estimates.",
        f"Mean likelihood ratio {weight_mean:.4f} (standard error {weight_se:.4f}); effective sample size {ess:.1f}.",
        "A stochastic estimate is not a guaranteed payout.",
        DISCLAIMER,
    ]
    return _sample_result(model, method, label, threshold, seed, n_samples, losses, weights, unit, notes, details)


def _stratified(model: GeneralPortfolio, threshold: float, seed: int, n_samples: int) -> RunResult:
    prepared = _prepare_strata(model)
    groups, min_per = _merge_paths(prepared["path_probability"], n_samples)
    allocation = _allocate(np.array([_mass(prepared["path_probability"], group) for group in groups]), n_samples, min_per)
    rng = np.random.default_rng(seed)
    collected = []
    plot_losses = []
    plot_weights = []
    for group, n_group in zip(groups, allocation):
        probability = _mass(prepared["path_probability"], group)
        levels = _sample_stratum_group(rng, model, prepared, group, int(n_group))
        losses = _apply_levels(model, levels)
        collected.append({"probability": probability, "losses": losses})
        portfolio = losses.sum(axis=1)
        plot_losses.append(portfolio)
        plot_weights.append(np.full(len(portfolio), probability / max(len(portfolio), 1)))
    maxima = model.maximum_payouts()
    metrics, uncertainty, country_mean, country_se = _stratified_metrics(collected, threshold, maxima, model.countries)
    return _pack(
        model,
        method="stratified",
        label="Stratified sampling",
        threshold=threshold,
        seed=seed,
        n_samples=n_samples,
        evaluations=n_samples,
        unit="stratified term scenarios",
        metrics=metrics,
        uncertainty=uncertainty,
        country_mean=country_mean,
        country_se=country_se,
        notes=[
            "Strata separate quiet years, single-country years, joint years and extreme severities.",
            "The estimate is the probability-weighted average of within-stratum means.",
            "A stochastic estimate is not a guaranteed payout.",
            DISCLAIMER,
        ],
        details={"n_groups": len(groups), "min_samples_per_group": int(min_per)},
        sample_losses=np.concatenate(plot_losses),
        sample_weights=np.concatenate(plot_weights),
    )


def _enumerate(model: GeneralPortfolio, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    coords = np.argwhere(probability > 0.0)
    atom_p = probability[probability > 0.0].astype(float)
    states: dict[tuple[int, ...], float] = {tuple(0 for _ in model.axes): 1.0}
    transitions = 0
    for _year in range(model.term_years):
        updated: dict[tuple[int, ...], float] = {}
        for state, mass in states.items():
            nxt = []
            for axis_index, axis in enumerate(model.axes):
                amounts = axis.steps[coords[:, axis_index]]
                if axis.aggregation == "sum_capped":
                    nxt.append(np.minimum(state[axis_index] + amounts, axis.cap).astype(np.int64))
                else:
                    nxt.append(np.maximum(state[axis_index], amounts).astype(np.int64))
            contributions = mass * atom_p
            for index, contribution in enumerate(contributions):
                key = tuple(int(component[index]) for component in nxt)
                updated[key] = updated.get(key, 0.0) + float(contribution)
            transitions += len(atom_p)
        states = updated
    losses = np.empty((len(states), model.n_countries), dtype=np.int64)
    probs = np.empty(len(states), dtype=float)
    for index, (state, mass) in enumerate(states.items()):
        for country_index, country in enumerate(model.countries):
            total = 0
            for axis_index, axis in enumerate(model.axes):
                if axis.country == country:
                    total += state[axis_index]
            losses[index, country_index] = min(model.country_caps[country], total)
        probs[index] = mass
    return losses, probs, transitions


def _losses_from_uniforms(model: GeneralPortfolio, uniforms: np.ndarray, hazards: tuple[HazardLaw, ...]) -> np.ndarray:
    return _apply_levels(model, _levels_from_uniforms(hazards, model.n_countries, uniforms))


def _levels_from_uniforms(hazards: tuple[HazardLaw, ...], n_countries: int, uniforms: np.ndarray) -> list[np.ndarray]:
    cursor = 0
    levels = []
    for hazard in hazards:
        shared = uniforms[:, :, cursor]
        cursor += 1
        idiosyncratic = uniforms[:, :, cursor : cursor + n_countries]
        cursor += n_countries
        severity_u = uniforms[:, :, cursor : cursor + n_countries]
        cursor += n_countries
        occurs = (shared < hazard.alpha)[:, :, None] | (idiosyncratic < hazard.beta.reshape(1, 1, -1))
        for country_index, severity in enumerate(hazard.severity):
            drawn = _categorical(severity_u[:, :, country_index], severity) + 1
            levels.append(np.where(occurs[:, :, country_index], drawn, 0).astype(np.int64))
    return levels


def _apply_levels(model: GeneralPortfolio, levels: list[np.ndarray]) -> np.ndarray:
    n_samples = levels[0].shape[0]
    paid = {(axis.country, axis.hazard): np.zeros(n_samples, dtype=float) for axis in model.axes}
    for axis, severity in zip(model.axes, levels):
        amounts = axis.steps[severity]
        if axis.aggregation == "sum_capped":
            paid[(axis.country, axis.hazard)] = np.minimum(amounts.sum(axis=1), axis.cap)
        else:
            paid[(axis.country, axis.hazard)] = amounts.max(axis=1)
    losses = np.zeros((n_samples, model.n_countries), dtype=float)
    for country_index, country in enumerate(model.countries):
        total = np.zeros(n_samples, dtype=float)
        for hazard in model.hazard_names:
            total += paid[(country, hazard)]
        losses[:, country_index] = np.minimum(total, model.country_caps[country])
    return losses


def _likelihood(target: np.ndarray, proposal: np.ndarray, levels: list[np.ndarray]) -> np.ndarray:
    ratios = np.ones(levels[0].shape[0], dtype=float)
    for year in range(levels[0].shape[1]):
        index = tuple(level[:, year] for level in levels)
        numerator = target[index]
        denominator = proposal[index]
        if np.any((denominator <= 0.0) & (numerator > 0.0)):
            raise RuntimeError("Proposal is missing support of the target model.")
        year_ratio = np.zeros(len(ratios), dtype=float)
        positive = denominator > 0.0
        year_ratio[positive] = numerator[positive] / denominator[positive]
        ratios *= year_ratio
    return ratios


def _proposal(model: GeneralPortfolio, tail: bool) -> tuple[tuple[HazardLaw, ...], np.ndarray]:
    laws = []
    tensors = []
    for hazard in model.hazards:
        if tail:
            occurrence = np.array([_mix(value, TAIL_OCCURRENCE_MIX, TAIL_OCCURRENCE_ANCHOR) for value in hazard.occurrence])
            severity = tuple(tilt_severity(vector, TAIL_SEVERITY_GEOMETRIC) for vector in hazard.severity)
            share = TAIL_SHOCK_SHARE
        else:
            occurrence = np.array([_mix(value, OCCURRENCE_MIX, OCCURRENCE_ANCHOR) for value in hazard.occurrence])
            severity = tuple(tilt_severity(vector, SEVERITY_GEOMETRIC) for vector in hazard.severity)
            share = 0.0
        tensor, alpha, beta = hazard_tensor(occurrence, severity, share)
        laws.append(
            HazardLaw(hazard.name, occurrence, severity, share, alpha, beta)
        )
        tensors.append(tensor)
    return tuple(laws), combine_hazard_tensors(tensors)


def _prepare_strata(model: GeneralPortfolio) -> dict:
    coords = np.argwhere(model.probability > 0.0)
    probs = model.probability[model.probability > 0.0]
    strata = np.array([_atom_stratum(tuple(int(value) for value in row), model) for row in coords], dtype=np.int64)
    stratum_probability = np.zeros(N_STRATA, dtype=float)
    atoms_by_stratum = []
    conditional = []
    for stratum in range(N_STRATA):
        mask = strata == stratum
        stratum_probability[stratum] = float(probs[mask].sum()) if mask.any() else 0.0
        atoms_by_stratum.append(coords[mask])
        mass = stratum_probability[stratum]
        conditional.append(probs[mask] / mass if mass > 0 else np.array([]))
    n_paths = N_STRATA ** model.term_years
    path_probability = np.empty(n_paths, dtype=float)
    path_strata = np.empty((n_paths, model.term_years), dtype=np.int64)
    for index in range(n_paths):
        remainder = index
        probability = 1.0
        digits = []
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


def _atom_stratum(levels: tuple[int, ...], model: GeneralPortfolio) -> int:
    hits = 0
    extreme = False
    for country in model.countries:
        hit = False
        for axis_index, axis in enumerate(model.axes):
            if axis.country != country:
                continue
            level = levels[axis_index]
            if level > 0:
                hit = True
            if level == len(axis.steps) - 1 and level > 0:
                extreme = True
        hits += int(hit)
    if hits == 0:
        return 0
    if extreme:
        return 3
    if hits >= 2:
        return 2
    return 1


def _sample_stratum_group(rng, model: GeneralPortfolio, prepared: dict, group: list[int], n_group: int) -> list[np.ndarray]:
    path_probability = np.array([prepared["path_probability"][index] for index in group], dtype=float)
    path_probability = path_probability / path_probability.sum()
    chosen = rng.choice(len(group), size=n_group, p=path_probability)
    levels = [np.empty((n_group, model.term_years), dtype=np.int64) for _ in model.axes]
    for year in range(model.term_years):
        strata = np.array([prepared["path_strata"][group[pick]][year] for pick in chosen], dtype=np.int64)
        for stratum in np.unique(strata):
            mask = strata == stratum
            count = int(mask.sum())
            conditional = prepared["conditional"][int(stratum)]
            atoms = prepared["atoms_by_stratum"][int(stratum)]
            if count and (len(conditional) == 0 or float(conditional.sum()) <= 0.0):
                raise RuntimeError("A sampled stratum has no annual atoms.")
            picks = rng.choice(len(conditional), size=count, p=conditional)
            selected = atoms[picks]
            for axis_index in range(len(model.axes)):
                levels[axis_index][mask, year] = selected[:, axis_index]
    return levels


def _summarise_pmf(losses, probability, threshold, maxima, countries):
    probability = np.asarray(probability, dtype=float)
    total = float(probability.sum())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Distribution probabilities sum to {total}, not 1.")
    portfolio = losses.sum(axis=1)
    maxima_array = np.asarray(maxima)
    metrics = {
        "expected_loss": float(np.sum(probability * portfolio)),
        "joint_attachment_probability": float(np.sum(probability[np.all(losses > 0, axis=1)])),
        "exceedance_probability": float(np.sum(probability[portfolio > threshold])),
        "expected_stop_loss": float(np.sum(probability * np.maximum(portfolio - threshold, 0.0))),
        "both_exhausted_probability": float(np.sum(probability[np.all(losses == maxima_array, axis=1)])),
    }
    uncertainty = {key: 0.0 for key in METRIC_KEYS}
    country_mean = {country: float(np.sum(probability * losses[:, index])) for index, country in enumerate(countries)}
    country_se = {country: 0.0 for country in countries}
    order = np.argsort(portfolio)
    pmf = _collapse(portfolio[order], probability[order])
    return metrics, uncertainty, country_mean, country_se, pmf


def _summarise_samples(losses, threshold, maxima, countries, weights):
    portfolio = losses.sum(axis=1)
    multiplier = np.ones(len(portfolio)) if weights is None else np.asarray(weights, dtype=float)
    maxima_array = np.asarray(maxima)
    quantities = {
        "expected_loss": portfolio,
        "joint_attachment_probability": np.all(losses > 0, axis=1).astype(float),
        "exceedance_probability": (portfolio > threshold).astype(float),
        "expected_stop_loss": np.maximum(portfolio - threshold, 0.0),
        "both_exhausted_probability": np.all(losses == maxima_array, axis=1).astype(float),
    }
    metrics = {}
    uncertainty = {}
    for key, values in quantities.items():
        metrics[key], uncertainty[key] = _mean_se(multiplier * values)
    country_mean = {}
    country_se = {}
    for index, country in enumerate(countries):
        country_mean[country], country_se[country] = _mean_se(multiplier * losses[:, index])
    return metrics, uncertainty, country_mean, country_se


def _stratified_metrics(groups, threshold, maxima, countries):
    def _estimate(extractor):
        point = 0.0
        variance = 0.0
        for group in groups:
            values = np.asarray(extractor(group), dtype=float)
            point += group["probability"] * float(values.mean())
            if len(values) >= 2:
                variance += (group["probability"] ** 2) * float(values.var(ddof=1)) / len(values)
        return point, float(np.sqrt(variance))

    maxima_array = np.asarray(maxima)

    def portfolio(group):
        return group["losses"].sum(axis=1)

    metrics = {}
    uncertainty = {}
    metrics["expected_loss"], uncertainty["expected_loss"] = _estimate(portfolio)
    metrics["joint_attachment_probability"], uncertainty["joint_attachment_probability"] = _estimate(
        lambda group: np.all(group["losses"] > 0, axis=1).astype(float)
    )
    metrics["exceedance_probability"], uncertainty["exceedance_probability"] = _estimate(
        lambda group: (portfolio(group) > threshold).astype(float)
    )
    metrics["expected_stop_loss"], uncertainty["expected_stop_loss"] = _estimate(
        lambda group: np.maximum(portfolio(group) - threshold, 0.0)
    )
    metrics["both_exhausted_probability"], uncertainty["both_exhausted_probability"] = _estimate(
        lambda group: np.all(group["losses"] == maxima_array, axis=1).astype(float)
    )
    country_mean = {}
    country_se = {}
    for index, country in enumerate(countries):
        country_mean[country], country_se[country] = _estimate(lambda group, index=index: group["losses"][:, index])
    return metrics, uncertainty, country_mean, country_se


def _sample_result(model, method, label, threshold, seed, n_samples, losses, weights, unit, notes, details):
    maxima = model.maximum_payouts()
    metrics, uncertainty, country_mean, country_se = _summarise_samples(losses, threshold, maxima, model.countries, weights)
    display = None
    if weights is not None:
        total = float(np.sum(weights))
        display = weights / total if total > 0 else weights
    return _pack(
        model, method, label, threshold, seed, n_samples, n_samples, unit,
        metrics, uncertainty, country_mean, country_se, notes, details,
        sample_losses=losses.sum(axis=1),
        sample_weights=display if display is not None else np.full(len(losses), 1.0 / len(losses)),
    )


def _pack(
    model, method, label, threshold, seed, n_samples, evaluations, unit,
    metrics, uncertainty, country_mean, country_se, notes, details,
    loss_pmf=None, sample_losses=None, sample_weights=None,
) -> RunResult:
    return RunResult(
        method=method,
        method_label=label,
        scenario_id=model.scenario_id,
        scenario_name=model.scenario_name,
        seed=None if seed is None else int(seed),
        n_samples=None if n_samples is None else int(n_samples),
        n_evaluations=int(evaluations),
        evaluation_unit=unit,
        threshold=float(threshold),
        currency=model.currency,
        model_version=model.model_version,
        valid_estimate=True,
        metrics=metrics,
        uncertainty=uncertainty,
        country_expected_loss=country_mean,
        country_expected_loss_se=country_se,
        notes=notes,
        details=details,
        loss_pmf=loss_pmf,
        sample_losses=sample_losses,
        sample_weights=sample_weights,
        countries=model.countries,
    )


def _unavailable(model, method: str, label: str, threshold: float, message: str, seed: int | None = None) -> RunResult:
    result = _invalid(model, threshold, seed, [message, "A model estimate is not a guaranteed payout.", DISCLAIMER], {"reason": message})
    result.method = method
    result.method_label = label
    return result


def _invalid(model, threshold, seed, notes, details) -> RunResult:
    return RunResult(
        method="tensor_network",
        method_label="Tensor network (experimental)",
        scenario_id=model.scenario_id,
        scenario_name=model.scenario_name,
        seed=None if seed is None else int(seed),
        n_samples=None,
        n_evaluations=0,
        evaluation_unit="not evaluated",
        threshold=float(threshold),
        currency=model.currency,
        model_version=model.model_version,
        valid_estimate=False,
        metrics={key: float("nan") for key in METRIC_KEYS},
        uncertainty={key: float("nan") for key in METRIC_KEYS},
        country_expected_loss={country: float("nan") for country in model.countries},
        country_expected_loss_se={country: float("nan") for country in model.countries},
        notes=notes,
        details=details,
        countries=model.countries,
    )


def _conditional_vector(bit: int, severity: np.ndarray) -> np.ndarray:
    vector = np.zeros(len(severity) + 1, dtype=float)
    if bit:
        vector[1:] = severity
    else:
        vector[0] = 1.0
    return vector


def _outer(vectors: list[np.ndarray]) -> np.ndarray:
    out = np.ones((), dtype=float)
    for vector in vectors:
        out = np.asarray(out[..., None] * vector, dtype=float)
    return out


def _mix(probability: float, weight: float, anchor: float) -> float:
    mixed = (1.0 - weight) * float(probability) + weight * anchor
    return float(np.clip(mixed, 1e-4, 1.0 - 1e-4))


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, 0.0
    return mean, float(array.std(ddof=1) / np.sqrt(len(array)))


def _collapse(losses: np.ndarray, probability: np.ndarray):
    if len(losses) == 0:
        return np.array([]), np.array([])
    unique_losses = []
    unique_probs = []
    current = float(losses[0])
    mass = 0.0
    for loss, prob in zip(losses, probability):
        if float(loss) == current:
            mass += float(prob)
        else:
            unique_losses.append(current)
            unique_probs.append(mass)
            current = float(loss)
            mass = float(prob)
    unique_losses.append(current)
    unique_probs.append(mass)
    return np.asarray(unique_losses, dtype=float), np.asarray(unique_probs, dtype=float)


def _scaled_occurrence(bundle: DataBundle, countries: tuple[str, ...], hazard: str, scale: float):
    frame = bundle.hazards.copy()
    frame["hazard"] = frame["hazard"].astype(str).str.strip().str.lower()
    values = []
    capped = False
    for country in countries:
        rows = frame[(frame["country"] == country) & (frame["hazard"] == hazard)]
        scaled = float(rows["annual_occurrence_probability"].iloc[0]) * scale
        if scaled > 1.0:
            scaled = 1.0
            capped = True
        values.append(scaled)
    note = f"{hazard} occurrence was capped at 1 after applying scale {scale}." if capped else None
    return np.asarray(values, dtype=float), note


def _severity_vector(bundle: DataBundle, country: str, hazard: str) -> np.ndarray:
    frame = bundle.hazards.copy()
    frame["hazard"] = frame["hazard"].astype(str).str.strip().str.lower()
    rows = frame[(frame["country"] == country) & (frame["hazard"] == hazard)].sort_values("severity_given_event")
    return rows["conditional_probability"].astype(float).to_numpy()


def _axes_from_contracts(bundle, scenario, countries, hazard_names, warnings):
    axes = []
    country_caps = {}
    for hazard in hazard_names:
        for country in countries:
            if hazard in ("earthquake", "precipitation"):
                contract = _contract_from_row(bundle, country, scenario)
                country_caps[country] = int(contract.contract_cap)
                if hazard == "earthquake":
                    axes.append(Axis(hazard, country, "sum_capped", int(contract.eq_cap), contract.eq_step_amounts()))
                else:
                    steps = contract.precip_step_amounts()
                    axes.append(Axis(hazard, country, "max_once", int(steps.max()), steps))
            else:
                axis, cap = _axis_from_peril(bundle, country, hazard)
                axes.append(axis)
                country_caps.setdefault(country, cap)
    for country, cap in country_caps.items():
        maximum = 0
        for axis in axes:
            if axis.country == country:
                maximum += int(axis.cap if axis.aggregation == "sum_capped" else axis.steps.max())
        if cap + 1e-6 < maximum:
            warnings.append(f"{country}: the contract cap can bind before every peril limit is reached.")
    return axes, country_caps


def _axis_from_peril(bundle: DataBundle, country: str, hazard: str) -> tuple[Axis, int]:
    frame = bundle.peril_terms.copy()
    frame["hazard"] = frame["hazard"].astype(str).str.strip().str.lower()
    rows = frame[(frame["country"].astype(str) == country) & (frame["hazard"] == hazard)].sort_values("severity")
    if rows.empty:
        raise ValueError(f"{country} {hazard} has no peril terms.")
    principal = int(round(float(rows["principal"].iloc[0])))
    cap = int(round(float(rows["cap"].iloc[0])))
    country_cap = int(round(float(rows["country_cap"].iloc[0])))
    aggregation = str(rows["aggregation"].iloc[0]).strip().lower()
    width = int(rows["severity"].max()) + 1
    steps = np.zeros(width, dtype=np.int64)
    for _, row in rows.iterrows():
        steps[int(row["severity"])] = money_fraction(float(row["payout_fraction"]), principal)
    return Axis(hazard, country, aggregation, cap, steps), country_cap


def _term_years(bundle: DataBundle) -> int:
    return int(bundle.contracts["term_years"].iloc[0])


def _labels(bundle: DataBundle, scenario) -> list[str]:
    labels = list(_data_labels(bundle, scenario))
    frame = getattr(bundle, "peril_terms", None)
    if frame is not None and "data_label" in frame.columns:
        for value in frame["data_label"].dropna().unique():
            text = str(value)
            if text not in labels:
                labels.append(text)
    return labels


def _parameters(
    scenario, countries, laws, country_caps, dependence_mode, shock_share,
    eq_scale, precip_scale, shock_share_override, precip_scale_override,
    term_years: int, currency: str,
) -> dict:
    occurrence = {}
    alphas = {}
    for law in laws:
        alphas[law.name] = law.alpha
        for index, country in enumerate(countries):
            occurrence.setdefault(country, {})[law.name] = float(law.occurrence[index])
    overrides = {}
    if shock_share_override is not None:
        overrides["shock_share"] = float(shock_share_override)
    if precip_scale_override is not None:
        overrides["precip_probability_scale"] = float(precip_scale_override)
    return {
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.scenario_name,
        "description": scenario.description,
        "assumption_label": scenario.assumption_label,
        "dependence_mode": dependence_mode,
        "shock_share": shock_share,
        "alpha_earthquake": alphas.get("earthquake"),
        "alpha_precipitation": alphas.get("precipitation"),
        "alphas": alphas,
        "eq_probability_scale": eq_scale,
        "precip_probability_scale": precip_scale,
        "occurrence_probability": occurrence,
        "term_years": int(term_years),
        "currency": currency,
        "country_caps": country_caps,
        "countries": list(countries),
        "hazards": [law.name for law in laws],
        "user_overrides": overrides,
        "threshold_note": "Portfolio exceedance is the strict probability P(loss > T).",
        "dependence_construction": (
            "Marshall-Olkin common shock within each hazard, for every modelled country. "
            "Hazards are independent of each other. Marginals match the scaled occurrence probabilities."
        ),
    }

