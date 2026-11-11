"""Contract payouts for the illustrative two-peril term structure.

Earthquake: each modelled year pays the step for that year's severity.
Those payments accumulate and are capped at the earthquake cap.

Precipitation: the contract pays once per term, at the highest precipitation
step reached in any year.

Country loss is the sum of the two peril payments, capped at the contract cap.
Portfolio loss is the sum of the two country losses. These rules are
illustrative synthetic terms, not an official benchmark schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def money_fraction(fraction: float, principal: int) -> int:
    """Convert a payout fraction of a principal into whole currency units."""
    return int(round(float(fraction) * int(principal)))


@dataclass(frozen=True)
class ContractTerms:
    """Numeric contract terms for one country."""

    country: str
    term_years: int
    eq_principal: int
    precip_principal: int
    currency: str
    eq_fractions: tuple[float, float, float]
    precip_fractions: tuple[float, float]
    eq_cap: int
    contract_cap: int

    def eq_step_amounts(self) -> np.ndarray:
        """Payout by earthquake severity. Index 0 is no event."""
        amounts = [0]
        amounts.extend(money_fraction(fraction, self.eq_principal) for fraction in self.eq_fractions)
        return np.asarray(amounts, dtype=np.int64)

    def precip_step_amounts(self) -> np.ndarray:
        """Payout by precipitation severity. Index 0 is no event."""
        amounts = [0]
        amounts.extend(money_fraction(fraction, self.precip_principal) for fraction in self.precip_fractions)
        return np.asarray(amounts, dtype=np.int64)

    def maximum_payout(self) -> int:
        """Largest country payment the contract rules can produce."""
        precip_max = int(self.precip_step_amounts().max())
        return int(min(self.contract_cap, int(self.eq_cap) + precip_max))


def country_losses(eq_severity: np.ndarray, rain_severity: np.ndarray, contract: ContractTerms) -> np.ndarray:
    """Aggregate a country's term losses.

    ``eq_severity`` and ``rain_severity`` have shape ``(samples, years)``.
    """
    eq_steps = contract.eq_step_amounts()[np.asarray(eq_severity, dtype=np.int64)]
    rain_steps = contract.precip_step_amounts()[np.asarray(rain_severity, dtype=np.int64)]
    eq_paid = np.minimum(eq_steps.sum(axis=1), contract.eq_cap)
    rain_paid = rain_steps.max(axis=1)
    return np.minimum(eq_paid + rain_paid, contract.contract_cap).astype(np.int64)


def portfolio_losses(
    eq_a: np.ndarray,
    eq_b: np.ndarray,
    rain_a: np.ndarray,
    rain_b: np.ndarray,
    contract_a: ContractTerms,
    contract_b: ContractTerms,
) -> tuple[np.ndarray, np.ndarray]:
    """Return country losses ``(loss_a, loss_b)`` for each sample."""
    if contract_a.term_years != contract_b.term_years:
        raise ValueError("Country contracts must share a term.")
    if eq_a.shape != eq_b.shape or eq_a.shape != rain_a.shape or eq_a.shape != rain_b.shape:
        raise ValueError("Severity arrays must share a shape.")
    if eq_a.shape[1] != contract_a.term_years:
        raise ValueError("Severity array length does not match the contract term.")
    loss_a = country_losses(eq_a, rain_a, contract_a)
    loss_b = country_losses(eq_b, rain_b, contract_b)
    return loss_a, loss_b
