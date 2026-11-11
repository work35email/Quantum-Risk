"""Scenario definitions, marginals and dependence."""

from __future__ import annotations

import numpy as np
import pytest

from src.scenarios import build_portfolio, list_scenarios
from src.validation import SCENARIO_NAMES


def test_five_named_scenarios(bundle):
    specs = list_scenarios(bundle)
    assert [spec.scenario_name for spec in specs] == [
        "Base",
        "Independent",
        "Common shock",
        "Rain stress",
        "Rain cover",
    ]
    assert [spec.scenario_id for spec in specs] == list(SCENARIO_NAMES)


def test_independent_and_common_shock_preserve_baseline_marginals(bundle):
    base = build_portfolio(bundle, "base")
    independent = build_portfolio(bundle, "independent")
    shock = build_portfolio(bundle, "common_shock")
    assert independent.annual.alpha_eq == 0.0
    assert independent.annual.alpha_rain == 0.0
    assert shock.annual.alpha_eq > base.annual.alpha_eq > 0.0
    assert shock.annual.alpha_rain > base.annual.alpha_rain > 0.0
    assert independent.annual.occurrence_eq == base.annual.occurrence_eq == shock.annual.occurrence_eq
    assert independent.annual.occurrence_rain == base.annual.occurrence_rain == shock.annual.occurrence_rain
    for model in (base, independent, shock):
        marginal_a, marginal_b = model.annual.marginal_eq()
        assert np.allclose(marginal_a[0], 1.0 - model.annual.occurrence_eq[0])
        assert np.allclose(marginal_b[0], 1.0 - model.annual.occurrence_eq[1])
        assert np.allclose(marginal_a[1:], model.annual.occurrence_eq[0] * model.annual.severity_eq[0])
        assert np.allclose(marginal_b[1:], model.annual.occurrence_eq[1] * model.annual.severity_eq[1])


def test_common_shock_raises_joint_earthquake_probability(bundle):
    independent = build_portfolio(bundle, "independent")
    shock = build_portfolio(bundle, "common_shock")

    def both(model) -> float:
        joint = model.annual.probability.sum(axis=(2, 3))
        return float(joint[1:, 1:].sum())

    assert both(shock) > both(independent)


def test_rain_stress_scales_precipitation_and_rain_cover_changes_the_payout(bundle):
    base = build_portfolio(bundle, "base")
    stress = build_portfolio(bundle, "rain_stress")
    cover = build_portfolio(bundle, "rain_cover")
    assert stress.annual.occurrence_rain[0] == pytest.approx(base.annual.occurrence_rain[0] * 1.75)
    assert stress.annual.occurrence_rain[1] == pytest.approx(base.annual.occurrence_rain[1] * 1.75)
    assert cover.contracts[0].precip_fractions == (1.0, 1.0)
    assert base.contracts[0].precip_fractions == (0.5, 1.0)
    assert "demo assumption" in cover.description.lower() or cover.assumption_label == "demo_assumption"


def test_shock_share_override_is_recorded(bundle):
    model = build_portfolio(bundle, "common_shock", shock_share_override=0.0)
    assert model.annual.alpha_eq == 0.0
    assert model.parameters["dependence_mode"] == "independent"
    assert model.parameters["user_overrides"]["shock_share"] == 0.0

