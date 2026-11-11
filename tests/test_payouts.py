"""Payout boundaries, caps and portfolio aggregation."""

from __future__ import annotations

import numpy as np

from src.payout_engine import ContractTerms, country_losses, money_fraction, portfolio_losses


def _contract(**overrides) -> ContractTerms:
    values = dict(
        country="Kyrgyz Republic",
        term_years=3,
        eq_principal=70_000_000,
        precip_principal=10_000_000,
        currency="USD",
        eq_fractions=(0.20, 0.50, 1.00),
        precip_fractions=(0.50, 1.00),
        eq_cap=70_000_000,
        contract_cap=80_000_000,
    )
    values.update(overrides)
    return ContractTerms(**values)


def test_step_amounts_are_exact_currency_units():
    contract = _contract()
    assert money_fraction(0.20, 70_000_000) == 14_000_000
    assert list(contract.eq_step_amounts()) == [0, 14_000_000, 35_000_000, 70_000_000]
    assert list(contract.precip_step_amounts()) == [0, 5_000_000, 10_000_000]
    assert contract.maximum_payout() == 80_000_000


def test_zero_severity_pays_nothing():
    loss = country_losses(np.zeros((1, 3), dtype=int), np.zeros((1, 3), dtype=int), _contract())
    assert int(loss[0]) == 0


def test_earthquake_steps_and_cap():
    contract = _contract()
    one_extreme = country_losses(np.array([[3, 0, 0]]), np.zeros((1, 3), dtype=int), contract)
    three_extreme = country_losses(np.array([[3, 3, 3]]), np.zeros((1, 3), dtype=int), contract)
    three_moderate = country_losses(np.array([[1, 1, 1]]), np.zeros((1, 3), dtype=int), contract)
    three_severe = country_losses(np.array([[2, 2, 2]]), np.zeros((1, 3), dtype=int), contract)
    assert int(one_extreme[0]) == 70_000_000
    assert int(three_extreme[0]) == 70_000_000
    assert int(three_moderate[0]) == 42_000_000
    assert int(three_severe[0]) == 70_000_000


def test_precipitation_pays_the_highest_step_once_per_term():
    contract = _contract()
    repeated = country_losses(np.zeros((1, 3), dtype=int), np.array([[2, 2, 2]]), contract)
    mixed = country_losses(np.zeros((1, 3), dtype=int), np.array([[1, 2, 1]]), contract)
    moderate = country_losses(np.zeros((1, 3), dtype=int), np.array([[1, 0, 0]]), contract)
    assert int(repeated[0]) == 10_000_000
    assert int(mixed[0]) == 10_000_000
    assert int(moderate[0]) == 5_000_000


def test_full_cover_scenario_pays_the_principal_on_a_moderate_rain_event():
    contract = _contract(precip_fractions=(1.0, 1.0))
    loss = country_losses(np.zeros((1, 3), dtype=int), np.array([[1, 0, 0]]), contract)
    assert int(loss[0]) == 10_000_000


def test_country_losses_sum_to_the_portfolio():
    contract_a = _contract()
    contract_b = _contract(country="Tajikistan")
    loss_a, loss_b = portfolio_losses(
        np.array([[3, 0, 0]]),
        np.array([[1, 0, 0]]),
        np.array([[2, 0, 0]]),
        np.zeros((1, 3), dtype=int),
        contract_a,
        contract_b,
    )
    assert int(loss_a[0]) == 80_000_000
    assert int(loss_b[0]) == 14_000_000
    assert int(loss_a[0] + loss_b[0]) == 94_000_000


def test_contract_cap_binds_before_the_sum_of_peril_limits():
    contract = _contract(contract_cap=60_000_000, eq_cap=70_000_000)
    loss = country_losses(np.array([[3, 0, 0]]), np.array([[2, 0, 0]]), contract)
    assert int(loss[0]) == 60_000_000
    assert contract.maximum_payout() == 60_000_000
