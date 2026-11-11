"""Marshall-Olkin common-shock dependence for two countries.

The construction preserves each country's occurrence probability.

Independent Bernoulli draws:

- ``S`` with probability ``alpha`` (the shared shock)
- ``I_c`` with probability ``beta_c`` (an idiosyncratic event in country ``c``)

Country ``c`` is affected when ``S`` or ``I_c`` occurs. ``beta_c`` is chosen
so the marginal probability equals the configured occurrence probability.
Earthquake and precipitation each have their own shock. Given occurrence,
severity is drawn from the country's conditional distribution and does not
depend on whether the trigger was shared or idiosyncratic.
"""

from __future__ import annotations

import numpy as np


def shock_alpha(occurrence_a: float, occurrence_b: float, shock_share: float) -> float:
    """Shared-shock probability ``share * min(occurrence)``."""
    if shock_share < -1e-15 or shock_share > 1.0 + 1e-15:
        raise ValueError("shock_share must be between 0 and 1.")
    smaller = min(float(occurrence_a), float(occurrence_b))
    alpha = float(np.clip(shock_share, 0.0, 1.0)) * smaller
    if alpha > smaller + 1e-9:
        raise ValueError("Shared-shock probability exceeds a country occurrence probability.")
    return float(min(alpha, smaller))


def idiosyncratic_probabilities(occurrence_a: float, occurrence_b: float, alpha: float) -> tuple[float, float]:
    """Idiosyncratic probabilities that reproduce the requested marginals."""
    if alpha < -1e-12 or alpha > min(occurrence_a, occurrence_b) + 1e-9:
        raise ValueError("alpha is outside the range allowed by the marginals.")
    if alpha >= 1.0 - 1e-15:
        return 0.0, 0.0
    beta_a = (float(occurrence_a) - alpha) / (1.0 - alpha)
    beta_b = (float(occurrence_b) - alpha) / (1.0 - alpha)
    beta_a = float(np.clip(beta_a, 0.0, 1.0))
    beta_b = float(np.clip(beta_b, 0.0, 1.0))
    return beta_a, beta_b


def occurrence_joint(occurrence_a: float, occurrence_b: float, alpha: float) -> np.ndarray:
    """Return a 2 by 2 joint probability matrix indexed ``[country_a, country_b]``."""
    beta_a, beta_b = idiosyncratic_probabilities(occurrence_a, occurrence_b, alpha)
    not_shared = 1.0 - alpha
    neither = not_shared * (1.0 - beta_a) * (1.0 - beta_b)
    only_a = not_shared * beta_a * (1.0 - beta_b)
    only_b = not_shared * (1.0 - beta_a) * beta_b
    both = 1.0 - neither - only_a - only_b
    joint = np.array([[neither, only_b], [only_a, both]], dtype=float)
    joint = np.clip(joint, 0.0, None)
    total = joint.sum()
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Occurrence joint sums to {total}, not 1.")
    return joint
