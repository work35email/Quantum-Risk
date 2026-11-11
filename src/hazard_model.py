"""Annual joint hazard distribution and term sampling.

Country axis 0 is the Kyrgyz Republic and country axis 1 is Tajikistan.
Earthquake severity is in ``{0, 1, 2, 3}`` and precipitation severity is in
``{0, 1, 2}``. Severity 0 means the hazard did not occur that year.
Earthquake and precipitation are independent of each other. Within a hazard,
the two countries follow the common-shock model in ``dependence_model``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.dependence_model import idiosyncratic_probabilities, occurrence_joint, shock_alpha
from src.payout_engine import ContractTerms

EQ_DIM = 4
RAIN_DIM = 3
UNIFORMS_PER_YEAR = 10
OCCURRENCE_MIX = 0.45
OCCURRENCE_ANCHOR = 0.55
SEVERITY_GEOMETRIC = 1.75


@dataclass
class AnnualDistribution:
    """Discrete joint distribution of one modelled year."""

    countries: tuple[str, str]
    probability: np.ndarray
    alpha_eq: float
    alpha_rain: float
    beta_eq: tuple[float, float]
    beta_rain: tuple[float, float]
    occurrence_eq: tuple[float, float]
    occurrence_rain: tuple[float, float]
    severity_eq: tuple[np.ndarray, np.ndarray]
    severity_rain: tuple[np.ndarray, np.ndarray]
    shock_share_eq: float
    shock_share_rain: float

    def __post_init__(self) -> None:
        self.probability = np.asarray(self.probability, dtype=float)
        total = float(self.probability.sum())
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Annual distribution sums to {total}, not 1.")
        if self.probability.shape != (EQ_DIM, EQ_DIM, RAIN_DIM, RAIN_DIM):
            raise ValueError("Annual distribution has the wrong shape.")

    @property
    def n_atoms(self) -> int:
        return int(np.count_nonzero(self.probability > 0.0))

    def marginal_eq(self) -> tuple[np.ndarray, np.ndarray]:
        rain = self.probability.sum(axis=(2, 3))
        return rain.sum(axis=1), rain.sum(axis=0)

    def marginal_rain(self) -> tuple[np.ndarray, np.ndarray]:
        eq = self.probability.sum(axis=(0, 1))
        return eq.sum(axis=1), eq.sum(axis=0)


@dataclass
class PortfolioModel:
    """Scenario-ready portfolio: annual hazards plus both contracts."""

    scenario_id: str
    scenario_name: str
    description: str
    assumption_label: str
    countries: tuple[str, str]
    term_years: int
    currency: str
    contracts: tuple[ContractTerms, ContractTerms]
    annual: AnnualDistribution
    parameters: dict
    warnings: list[str]
    data_labels: list[str]
    model_version: str

    def maximum_payouts(self) -> tuple[int, int]:
        return self.contracts[0].maximum_payout(), self.contracts[1].maximum_payout()


def severity_joint(occurrence: np.ndarray, severity_a: np.ndarray, severity_b: np.ndarray) -> np.ndarray:
    """Expand an occurrence joint into a severity joint, with 0 meaning no event."""
    n_a = len(severity_a)
    n_b = len(severity_b)
    out = np.zeros((n_a + 1, n_b + 1), dtype=float)
    out[0, 0] = occurrence[0, 0]
    out[1:, 0] = occurrence[1, 0] * severity_a
    out[0, 1:] = occurrence[0, 1] * severity_b
    out[1:, 1:] = occurrence[1, 1] * np.outer(severity_a, severity_b)
    return out


def build_annual_distribution(
    countries: tuple[str, str],
    occurrence_eq: tuple[float, float],
    occurrence_rain: tuple[float, float],
    severity_eq: tuple[np.ndarray, np.ndarray],
    severity_rain: tuple[np.ndarray, np.ndarray],
    shock_share_eq: float,
    shock_share_rain: float,
) -> AnnualDistribution:
    """Build the annual probability tensor from marginals and shock shares."""
    alpha_eq = shock_alpha(occurrence_eq[0], occurrence_eq[1], shock_share_eq)
    alpha_rain = shock_alpha(occurrence_rain[0], occurrence_rain[1], shock_share_rain)
    beta_eq = idiosyncratic_probabilities(occurrence_eq[0], occurrence_eq[1], alpha_eq)
    beta_rain = idiosyncratic_probabilities(occurrence_rain[0], occurrence_rain[1], alpha_rain)
    eq_joint = severity_joint(
        occurrence_joint(occurrence_eq[0], occurrence_eq[1], alpha_eq),
        np.asarray(severity_eq[0], dtype=float),
        np.asarray(severity_eq[1], dtype=float),
    )
    rain_joint = severity_joint(
        occurrence_joint(occurrence_rain[0], occurrence_rain[1], alpha_rain),
        np.asarray(severity_rain[0], dtype=float),
        np.asarray(severity_rain[1], dtype=float),
    )
    probability = eq_joint[:, :, None, None] * rain_joint[None, None, :, :]
    return AnnualDistribution(
        countries=countries,
        probability=probability,
        alpha_eq=alpha_eq,
        alpha_rain=alpha_rain,
        beta_eq=beta_eq,
        beta_rain=beta_rain,
        occurrence_eq=(float(occurrence_eq[0]), float(occurrence_eq[1])),
        occurrence_rain=(float(occurrence_rain[0]), float(occurrence_rain[1])),
        severity_eq=(np.asarray(severity_eq[0], dtype=float), np.asarray(severity_eq[1], dtype=float)),
        severity_rain=(np.asarray(severity_rain[0], dtype=float), np.asarray(severity_rain[1], dtype=float)),
        shock_share_eq=float(shock_share_eq),
        shock_share_rain=float(shock_share_rain),
    )


def tilt_severity(severity: np.ndarray, geometric: float = SEVERITY_GEOMETRIC) -> np.ndarray:
    """Shift a conditional severity distribution toward higher steps without adding support."""
    weights = np.asarray(severity, dtype=float) * (geometric ** np.arange(1, len(severity) + 1))
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Severity distribution has no mass to tilt.")
    return weights / total


def proposal_distribution(annual: AnnualDistribution) -> AnnualDistribution:
    """Independent proposal with full support on the original annual atoms.

    Occurrence probabilities are pulled toward an interior value so every
    original occurrence pattern keeps positive proposal probability. Severity
    conditionals keep their original zeros. Likelihood ratios ``p/q`` are
    therefore defined on the support of the model.
    """
    def _tilt_occurrence(probability: float) -> float:
        mixed = (1.0 - OCCURRENCE_MIX) * float(probability) + OCCURRENCE_MIX * OCCURRENCE_ANCHOR
        return float(np.clip(mixed, 1e-6, 1.0 - 1e-6))

    return build_annual_distribution(
        countries=annual.countries,
        occurrence_eq=(_tilt_occurrence(annual.occurrence_eq[0]), _tilt_occurrence(annual.occurrence_eq[1])),
        occurrence_rain=(_tilt_occurrence(annual.occurrence_rain[0]), _tilt_occurrence(annual.occurrence_rain[1])),
        severity_eq=(tilt_severity(annual.severity_eq[0]), tilt_severity(annual.severity_eq[1])),
        severity_rain=(tilt_severity(annual.severity_rain[0]), tilt_severity(annual.severity_rain[1])),
        shock_share_eq=0.0,
        shock_share_rain=0.0,
    )


def _categorical(uniforms: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(np.asarray(probabilities, dtype=float))
    cdf[-1] = 1.0
    clipped = np.clip(uniforms, 0.0, np.nextafter(1.0, 0.0))
    return np.searchsorted(cdf, clipped, side="right").astype(np.int64)


def _hazard_pair(
    shared: np.ndarray,
    idio_a: np.ndarray,
    idio_b: np.ndarray,
    severity_a_u: np.ndarray,
    severity_b_u: np.ndarray,
    alpha: float,
    beta: tuple[float, float],
    severity: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    occurs_a = (shared < alpha) | (idio_a < beta[0])
    occurs_b = (shared < alpha) | (idio_b < beta[1])
    level_a = _categorical(severity_a_u, severity[0]) + 1
    level_b = _categorical(severity_b_u, severity[1]) + 1
    return np.where(occurs_a, level_a, 0), np.where(occurs_b, level_b, 0)


def severities_from_uniforms(
    annual: AnnualDistribution,
    uniforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map uniforms of shape ``(samples, years, 10)`` to severity arrays."""
    uniforms = np.asarray(uniforms, dtype=float)
    if uniforms.ndim != 3 or uniforms.shape[2] != UNIFORMS_PER_YEAR:
        raise ValueError("Expected uniforms with shape (samples, years, 10).")
    samples, years, _ = uniforms.shape
    eq_a = np.empty((samples, years), dtype=np.int64)
    eq_b = np.empty((samples, years), dtype=np.int64)
    rain_a = np.empty((samples, years), dtype=np.int64)
    rain_b = np.empty((samples, years), dtype=np.int64)
    for year in range(years):
        draw = uniforms[:, year, :]
        eq_a[:, year], eq_b[:, year] = _hazard_pair(
            draw[:, 0], draw[:, 1], draw[:, 2], draw[:, 3], draw[:, 4],
            annual.alpha_eq, annual.beta_eq, annual.severity_eq,
        )
        rain_a[:, year], rain_b[:, year] = _hazard_pair(
            draw[:, 5], draw[:, 6], draw[:, 7], draw[:, 8], draw[:, 9],
            annual.alpha_rain, annual.beta_rain, annual.severity_rain,
        )
    return eq_a, eq_b, rain_a, rain_b


def sample_term(
    annual: AnnualDistribution,
    term_years: int,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``n_samples`` independent terms from the annual model."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    uniforms = rng.random((n_samples, term_years, UNIFORMS_PER_YEAR))
    return severities_from_uniforms(annual, uniforms)


def likelihood_ratios(
    target: AnnualDistribution,
    proposal: AnnualDistribution,
    eq_a: np.ndarray,
    eq_b: np.ndarray,
    rain_a: np.ndarray,
    rain_b: np.ndarray,
) -> np.ndarray:
    """Product of annual likelihood ratios ``p/q`` over the term."""
    ratios = np.ones(eq_a.shape[0], dtype=float)
    for year in range(eq_a.shape[1]):
        numerator = target.probability[eq_a[:, year], eq_b[:, year], rain_a[:, year], rain_b[:, year]]
        denominator = proposal.probability[eq_a[:, year], eq_b[:, year], rain_a[:, year], rain_b[:, year]]
        if np.any((denominator <= 0.0) & (numerator > 0.0)):
            raise RuntimeError("Importance-sampling proposal is missing support of the target model.")
        year_ratio = np.zeros(eq_a.shape[0], dtype=float)
        positive = denominator > 0.0
        year_ratio[positive] = numerator[positive] / denominator[positive]
        ratios *= year_ratio
    return ratios
