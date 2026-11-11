"""Input validation messages and cross-table consistency."""

from __future__ import annotations

from src.data_loader import DataBundle, replace_tables
from src.validation import validate_bundle


def _broken(bundle, **tables) -> DataBundle:
    return replace_tables(bundle, tables)


def test_synthetic_bundle_is_valid(bundle):
    result = validate_bundle(bundle)
    assert result.ok, result.errors


def test_probability_above_one_names_the_column(bundle):
    hazards = bundle.hazards.copy()
    hazards.loc[0, "annual_occurrence_probability"] = 1.2
    result = validate_bundle(_broken(bundle, hazards=hazards))
    assert not result.ok
    assert any("annual_occurrence_probability" in message and "between 0 and 1" in message for message in result.errors)


def test_conditional_probabilities_must_sum_to_one(bundle):
    hazards = bundle.hazards.copy()
    hazards.loc[0, "conditional_probability"] = 0.2
    result = validate_bundle(_broken(bundle, hazards=hazards))
    assert not result.ok
    assert any("conditional probabilities sum" in message for message in result.errors)


def test_missing_column_is_named(bundle):
    hazards = bundle.hazards.drop(columns=["hazard"])
    result = validate_bundle(_broken(bundle, hazards=hazards))
    assert not result.ok
    assert any("missing required columns" in message and "hazard" in message for message in result.errors)


def test_exposure_must_match_principals(bundle):
    exposure = bundle.exposure.copy()
    exposure.loc[0, "insured_exposure"] = 1
    result = validate_bundle(_broken(bundle, exposure=exposure))
    assert not result.ok
    assert any("insured exposure" in message for message in result.errors)


def test_decreasing_payout_fractions_are_rejected(bundle):
    contracts = bundle.contracts.copy()
    contracts.loc[0, "eq_severity_1_fraction"] = 0.8
    result = validate_bundle(_broken(bundle, contracts=contracts))
    assert not result.ok
    assert any("non-decreasing" in message for message in result.errors)


def test_unknown_country_is_rejected(bundle):
    exposure = bundle.exposure.copy()
    exposure.loc[0, "country"] = "Exampleland"
    result = validate_bundle(_broken(bundle, exposure=exposure))
    assert not result.ok
    assert any("Kyrgyz Republic" in message for message in result.errors)


def test_cap_above_principal_is_rejected(bundle):
    contracts = bundle.contracts.copy()
    contracts.loc[0, "eq_cap"] = 90_000_000
    result = validate_bundle(_broken(bundle, contracts=contracts))
    assert not result.ok
    assert any("eq_cap" in message for message in result.errors)


def test_base_scenario_must_agree_with_the_dependence_file(bundle):
    scenarios = bundle.scenarios.copy()
    base = scenarios["scenario_id"] == "base"
    scenarios.loc[base, "shock_share"] = 0.5
    result = validate_bundle(_broken(bundle, scenarios=scenarios))
    assert not result.ok
    assert any("shock_share must match dependence.csv" in message for message in result.errors)
