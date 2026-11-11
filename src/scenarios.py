"""Five labelled demo scenarios and the portfolio builder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import __version__
from src.data_loader import DataBundle
from src.hazard_model import AnnualDistribution, PortfolioModel, build_annual_distribution
from src.payout_engine import ContractTerms
from src.validation import COUNTRIES, HAZARDS, SCENARIO_IDS, validate_bundle

DISCLAIMER = (
    "Synthetic demonstration only. Figures are model estimates under the stated "
    "assumptions, not guaranteed payouts, prices, or catastrophe forecasts. "
    "No quantum advantage is claimed. Contract rules are illustrative and are "
    "not an official ADB payout schedule."
)

MAX_OVERRIDE_SCALE = 5.0


@dataclass(frozen=True)
class ScenarioSpec:
    """One predefined scenario after the CSV row has been parsed."""

    scenario_id: str
    scenario_name: str
    dependence_mode: str
    shock_share: float
    precip_probability_scale: float
    eq_probability_scale: float
    precip_severity_1_fraction: float | None
    precip_severity_2_fraction: float | None
    description: str
    assumption_label: str


def list_scenarios(bundle: DataBundle) -> list[ScenarioSpec]:
    specs = [_parse_scenario_row(row) for _, row in bundle.scenarios.iterrows()]
    order = {scenario_id: index for index, scenario_id in enumerate(SCENARIO_IDS)}
    return sorted(
        specs,
        key=lambda spec: (0, order[spec.scenario_id]) if spec.scenario_id in order else (1, spec.scenario_id),
    )


def is_canonical_demo(bundle: DataBundle) -> bool:
    """True for the shipped two-country earthquake and precipitation fixture."""
    countries = tuple(str(value) for value in bundle.exposure["country"])
    if countries != COUNTRIES:
        return False
    hazards: list[str] = []
    for value in bundle.hazards["hazard"].astype(str).str.strip().str.lower():
        if value not in hazards:
            hazards.append(value)
    if tuple(hazards) != HAZARDS:
        return False
    peril_terms = getattr(bundle, "peril_terms", None)
    return peril_terms is None or len(peril_terms) == 0


def get_scenario(bundle: DataBundle, scenario_id: str) -> ScenarioSpec:
    key = scenario_id.strip().lower()
    for spec in list_scenarios(bundle):
        if spec.scenario_id == key:
            return spec
    raise KeyError(f"Unknown scenario '{scenario_id}'.")


def build_portfolio(
    bundle: DataBundle,
    scenario_id: str,
    shock_share_override: float | None = None,
    precip_scale_override: float | None = None,
) -> PortfolioModel:
    """Validate inputs, apply a scenario, and return a runnable portfolio model."""
    validation = validate_bundle(bundle)
    if not validation.ok:
        raise ValueError("Invalid input data:\n- " + "\n- ".join(validation.errors))

    scenario = get_scenario(bundle, scenario_id)
    if shock_share_override is not None and not 0.0 <= float(shock_share_override) <= 1.0:
        raise ValueError("Shock-share override must be between 0 and 1.")
    if precip_scale_override is not None and not 0.0 < float(precip_scale_override) <= MAX_OVERRIDE_SCALE:
        raise ValueError(f"Precipitation-scale override must be in (0, {MAX_OVERRIDE_SCALE}].")
    if not is_canonical_demo(bundle):
        from src.multimodel import build_general_portfolio

        return build_general_portfolio(
            bundle,
            scenario_id,
            shock_share_override=shock_share_override,
            precip_scale_override=precip_scale_override,
        )

    hazards = _hazard_table(bundle.hazards)
    shock_share = scenario.shock_share if shock_share_override is None else float(shock_share_override)
    dependence_mode = "independent" if shock_share == 0.0 else "common_shock"
    if shock_share_override is None:
        dependence_mode = scenario.dependence_mode
        shock_share = 0.0 if dependence_mode == "independent" else scenario.shock_share

    eq_scale = scenario.eq_probability_scale
    precip_scale = (
        scenario.precip_probability_scale if precip_scale_override is None else float(precip_scale_override)
    )
    warnings = list(validation.warnings)
    occurrence_eq, occurrence_notes = _scale_occurrence(hazards, "earthquake", eq_scale)
    occurrence_rain, rain_notes = _scale_occurrence(hazards, "precipitation", precip_scale)
    warnings.extend(note for note in (occurrence_notes, rain_notes) if note)

    contracts = tuple(_contract_from_row(bundle, country, scenario) for country in COUNTRIES)
    annual = build_annual_distribution(
        countries=COUNTRIES,
        occurrence_eq=occurrence_eq,
        occurrence_rain=occurrence_rain,
        severity_eq=_severity_pair(hazards, "earthquake"),
        severity_rain=_severity_pair(hazards, "precipitation"),
        shock_share_eq=shock_share,
        shock_share_rain=shock_share,
    )
    parameters = _parameter_snapshot(
        scenario=scenario,
        annual=annual,
        contracts=contracts,
        dependence_mode=dependence_mode,
        shock_share=shock_share,
        eq_scale=eq_scale,
        precip_scale=precip_scale,
        shock_share_override=shock_share_override,
        precip_scale_override=precip_scale_override,
    )
    labels = _data_labels(bundle, scenario)
    return PortfolioModel(
        scenario_id=scenario.scenario_id,
        scenario_name=scenario.scenario_name,
        description=scenario.description,
        assumption_label=scenario.assumption_label,
        countries=COUNTRIES,
        term_years=contracts[0].term_years,
        currency=contracts[0].currency,
        contracts=contracts,
        annual=annual,
        parameters=parameters,
        warnings=warnings,
        data_labels=labels,
        model_version=__version__,
    )


def parameter_rows(model: PortfolioModel) -> pd.DataFrame:
    """Flatten resolved scenario parameters for display and export."""
    rows = []
    parameters = model.parameters
    scalar_keys = [
        ("dependence_mode", "Dependence mode"),
        ("shock_share", "Shared-shock share of the smaller marginal"),
        ("alpha_earthquake", "Earthquake shared-shock probability"),
        ("alpha_precipitation", "Precipitation shared-shock probability"),
        ("eq_probability_scale", "Earthquake probability scale"),
        ("precip_probability_scale", "Precipitation probability scale"),
        ("precip_severity_1_fraction", "Precipitation moderate payout fraction"),
        ("precip_severity_2_fraction", "Precipitation severe payout fraction"),
        ("term_years", "Term (years)"),
        ("threshold_note", "Exceedance rule"),
    ]
    for key, label in scalar_keys:
        if key in parameters:
            rows.append({"parameter": label, "value": _display_value(parameters[key]), "group": "scenario"})
    for country, hazards in parameters["occurrence_probability"].items():
        for hazard, value in hazards.items():
            rows.append(
                {
                    "parameter": f"{country} {hazard} annual occurrence probability",
                    "value": _display_value(value),
                    "group": "hazard",
                }
            )
    overrides = parameters.get("user_overrides", {})
    for key, value in overrides.items():
        rows.append({"parameter": f"User override: {key}", "value": _display_value(value), "group": "override"})
    return pd.DataFrame(rows)


def _display_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _parse_scenario_row(row: pd.Series) -> ScenarioSpec:
    def _optional(column: str) -> float | None:
        if column not in row or pd.isna(row[column]):
            return None
        return float(row[column])

    return ScenarioSpec(
        scenario_id=str(row["scenario_id"]).strip().lower(),
        scenario_name=str(row["scenario_name"]).strip(),
        dependence_mode=str(row["dependence_mode"]).strip().lower(),
        shock_share=float(row["shock_share"]),
        precip_probability_scale=float(row["precip_probability_scale"]),
        eq_probability_scale=float(row["eq_probability_scale"]),
        precip_severity_1_fraction=_optional("precip_severity_1_fraction"),
        precip_severity_2_fraction=_optional("precip_severity_2_fraction"),
        description=str(row["description"]).strip(),
        assumption_label=str(row["assumption_label"]).strip(),
    )


def _hazard_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["hazard"] = out["hazard"].astype(str).str.strip().str.lower()
    out["country"] = out["country"].astype(str)
    return out


def _scale_occurrence(
    hazards: pd.DataFrame,
    hazard: str,
    scale: float,
) -> tuple[tuple[float, float], str | None]:
    values = []
    capped = False
    for country in COUNTRIES:
        rows = hazards[(hazards["country"] == country) & (hazards["hazard"] == hazard)]
        base = float(rows["annual_occurrence_probability"].iloc[0])
        scaled = base * scale
        if scaled > 1.0:
            scaled = 1.0
            capped = True
        values.append(scaled)
    note = None
    if capped:
        note = f"{hazard} occurrence was capped at 1 after applying scale {scale}."
    return (values[0], values[1]), note


def _severity_pair(hazards: pd.DataFrame, hazard: str) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    for country in COUNTRIES:
        rows = hazards[(hazards["country"] == country) & (hazards["hazard"] == hazard)]
        rows = rows.sort_values("severity_given_event")
        vectors.append(rows["conditional_probability"].astype(float).to_numpy())
    return vectors[0], vectors[1]


def _contract_from_row(bundle: DataBundle, country: str, scenario: ScenarioSpec) -> ContractTerms:
    row = bundle.contracts[bundle.contracts["country"].astype(str) == country].iloc[0]
    precip = [
        float(row["precip_severity_1_fraction"]),
        float(row["precip_severity_2_fraction"]),
    ]
    if scenario.precip_severity_1_fraction is not None:
        precip[0] = scenario.precip_severity_1_fraction
    if scenario.precip_severity_2_fraction is not None:
        precip[1] = scenario.precip_severity_2_fraction
    if precip[0] > precip[1]:
        raise ValueError(f"{country}: precipitation fractions decreased after the scenario override.")
    return ContractTerms(
        country=country,
        term_years=int(row["term_years"]),
        eq_principal=int(round(float(row["eq_principal"]))),
        precip_principal=int(round(float(row["precip_principal"]))),
        currency=str(row["currency"]).strip().upper(),
        eq_fractions=(
            float(row["eq_severity_1_fraction"]),
            float(row["eq_severity_2_fraction"]),
            float(row["eq_severity_3_fraction"]),
        ),
        precip_fractions=(precip[0], precip[1]),
        eq_cap=int(round(float(row["eq_cap"]))),
        contract_cap=int(round(float(row["contract_cap"]))),
    )


def _parameter_snapshot(
    scenario: ScenarioSpec,
    annual: AnnualDistribution,
    contracts: tuple[ContractTerms, ContractTerms],
    dependence_mode: str,
    shock_share: float,
    eq_scale: float,
    precip_scale: float,
    shock_share_override: float | None,
    precip_scale_override: float | None,
) -> dict:
    occurrence = {}
    for index, country in enumerate(COUNTRIES):
        occurrence[country] = {
            "earthquake": annual.occurrence_eq[index],
            "precipitation": annual.occurrence_rain[index],
        }
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
        "alpha_earthquake": annual.alpha_eq,
        "alpha_precipitation": annual.alpha_rain,
        "beta_earthquake": list(annual.beta_eq),
        "beta_precipitation": list(annual.beta_rain),
        "eq_probability_scale": eq_scale,
        "precip_probability_scale": precip_scale,
        "occurrence_probability": occurrence,
        "severity_eq": [annual.severity_eq[0].tolist(), annual.severity_eq[1].tolist()],
        "severity_rain": [annual.severity_rain[0].tolist(), annual.severity_rain[1].tolist()],
        "precip_severity_1_fraction": contracts[0].precip_fractions[0],
        "precip_severity_2_fraction": contracts[0].precip_fractions[1],
        "eq_fractions": list(contracts[0].eq_fractions),
        "term_years": contracts[0].term_years,
        "currency": contracts[0].currency,
        "principals": {
            country: {
                "earthquake": contract.eq_principal,
                "precipitation": contract.precip_principal,
                "eq_cap": contract.eq_cap,
                "contract_cap": contract.contract_cap,
            }
            for country, contract in zip(COUNTRIES, contracts)
        },
        "user_overrides": overrides,
        "threshold_note": "Portfolio exceedance is the strict probability P(loss > T).",
        "dependence_construction": (
            "Marshall-Olkin common shock within each hazard. "
            "Marginals match the scaled occurrence probabilities. "
            "Severities are conditional on occurrence and independent across countries given occurrence."
        ),
    }


def _data_labels(bundle: DataBundle, scenario: ScenarioSpec) -> list[str]:
    labels = [scenario.assumption_label]
    for frame in (bundle.hazards, bundle.dependence, bundle.exposure, bundle.contracts):
        if "data_label" in frame.columns:
            labels.extend(str(value) for value in frame["data_label"].dropna().unique())
    seen = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    return seen
