"""Portfolio risk metrics and the shared run result."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

METRIC_KEYS = (
    "expected_loss",
    "joint_attachment_probability",
    "exceedance_probability",
    "expected_stop_loss",
    "both_exhausted_probability",
)

METRIC_LABELS = {
    "expected_loss": "Expected principal loss E[L]",
    "joint_attachment_probability": "Joint attachment probability P(L1 > 0 and L2 > 0)",
    "exceedance_probability": "Portfolio exceedance probability P(L > T)",
    "expected_stop_loss": "Expected stop-loss E[max(L - T, 0)]",
    "both_exhausted_probability": "Both bonds exhausted",
}

MONEY_METRICS = {"expected_loss", "expected_stop_loss"}


@dataclass
class RunResult:
    """One estimator run, including uncertainty and plotting payloads."""

    method: str
    method_label: str
    scenario_id: str
    scenario_name: str
    seed: int | None
    n_samples: int | None
    n_evaluations: int
    evaluation_unit: str
    threshold: float
    currency: str
    model_version: str
    valid_estimate: bool
    metrics: dict
    uncertainty: dict
    country_expected_loss: dict
    country_expected_loss_se: dict
    runtime_seconds: float = 0.0
    peak_memory_bytes: int = 0
    notes: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    loss_pmf: tuple[np.ndarray, np.ndarray] | None = None
    sample_losses: np.ndarray | None = None
    sample_weights: np.ndarray | None = None
    countries: tuple[str, str] = ("Kyrgyz Republic", "Tajikistan")

    def error_against(self, reference: "RunResult") -> dict[str, float]:
        errors = {}
        for key in METRIC_KEYS:
            if key not in self.metrics or key not in reference.metrics:
                continue
            errors[key] = abs(float(self.metrics[key]) - float(reference.metrics[key]))
        return errors


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, 0.0
    se = float(array.std(ddof=1) / np.sqrt(len(array)))
    return mean, se


def summarise_pmf(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    max_a: int,
    max_b: int,
    countries: tuple[str, str],
) -> tuple[dict, dict, dict, dict, tuple[np.ndarray, np.ndarray]]:
    """Exact metrics from a discrete distribution over country losses."""
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    probability = np.asarray(probability, dtype=float)
    total_p = float(probability.sum())
    if abs(total_p - 1.0) > 1e-6:
        raise ValueError(f"Distribution probabilities sum to {total_p}, not 1.")
    portfolio = loss_a + loss_b
    metrics = {
        "expected_loss": float(np.sum(probability * portfolio)),
        "joint_attachment_probability": float(np.sum(probability[(loss_a > 0) & (loss_b > 0)])),
        "exceedance_probability": float(np.sum(probability[portfolio > threshold])),
        "expected_stop_loss": float(np.sum(probability * np.maximum(portfolio - threshold, 0.0))),
        "both_exhausted_probability": float(np.sum(probability[(loss_a == max_a) & (loss_b == max_b)])),
    }
    uncertainty = {key: 0.0 for key in METRIC_KEYS}
    country_mean = {
        countries[0]: float(np.sum(probability * loss_a)),
        countries[1]: float(np.sum(probability * loss_b)),
    }
    country_se = {countries[0]: 0.0, countries[1]: 0.0}
    order = np.argsort(portfolio)
    pmf = _collapse_pmf(portfolio[order], probability[order])
    return metrics, uncertainty, country_mean, country_se, pmf


def summarise_samples(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    threshold: float,
    max_a: int,
    max_b: int,
    countries: tuple[str, str],
    weights: np.ndarray | None = None,
) -> tuple[dict, dict, dict, dict]:
    """Sample metrics.

    When ``weights`` is provided they are likelihood ratios. The estimator is
    the unbiased average of ``weight * f``, not a self-normalised weighted mean
    and not the unweighted sample mean.
    """
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    portfolio = loss_a + loss_b
    if weights is None:
        multiplier = np.ones(len(portfolio), dtype=float)
    else:
        multiplier = np.asarray(weights, dtype=float)
        if len(multiplier) != len(portfolio):
            raise ValueError("Weight vector length does not match the sample.")

    quantities = {
        "expected_loss": portfolio,
        "joint_attachment_probability": ((loss_a > 0) & (loss_b > 0)).astype(float),
        "exceedance_probability": (portfolio > threshold).astype(float),
        "expected_stop_loss": np.maximum(portfolio - threshold, 0.0),
        "both_exhausted_probability": ((loss_a == max_a) & (loss_b == max_b)).astype(float),
    }
    metrics = {}
    uncertainty = {}
    for key, values in quantities.items():
        mean, se = _mean_se(multiplier * values)
        metrics[key] = mean
        uncertainty[key] = se
    country_mean = {}
    country_se = {}
    for name, losses in ((countries[0], loss_a), (countries[1], loss_b)):
        mean, se = _mean_se(multiplier * losses)
        country_mean[name] = mean
        country_se[name] = se
    return metrics, uncertainty, country_mean, country_se


def stratified_metrics(
    groups: list[dict],
    threshold: float,
    max_a: int,
    max_b: int,
    countries: tuple[str, str],
) -> tuple[dict, dict, dict, dict]:
    """Unbiased stratified estimator ``sum p_g * mean(f | g)`` and its standard error."""

    def _estimate(extractor) -> tuple[float, float]:
        point = 0.0
        variance = 0.0
        for group in groups:
            values = extractor(group)
            point += group["probability"] * float(values.mean())
            if len(values) >= 2:
                variance += (group["probability"] ** 2) * float(values.var(ddof=1)) / len(values)
        return point, float(np.sqrt(variance))

    portfolio = lambda group: group["loss_a"] + group["loss_b"]
    metrics = {}
    uncertainty = {}
    metrics["expected_loss"], uncertainty["expected_loss"] = _estimate(portfolio)
    metrics["joint_attachment_probability"], uncertainty["joint_attachment_probability"] = _estimate(
        lambda group: ((group["loss_a"] > 0) & (group["loss_b"] > 0)).astype(float)
    )
    metrics["exceedance_probability"], uncertainty["exceedance_probability"] = _estimate(
        lambda group: ((group["loss_a"] + group["loss_b"]) > threshold).astype(float)
    )
    metrics["expected_stop_loss"], uncertainty["expected_stop_loss"] = _estimate(
        lambda group: np.maximum(group["loss_a"] + group["loss_b"] - threshold, 0.0)
    )
    metrics["both_exhausted_probability"], uncertainty["both_exhausted_probability"] = _estimate(
        lambda group: ((group["loss_a"] == max_a) & (group["loss_b"] == max_b)).astype(float)
    )
    country_mean = {}
    country_se = {}
    country_mean[countries[0]], country_se[countries[0]] = _estimate(lambda group: group["loss_a"])
    country_mean[countries[1]], country_se[countries[1]] = _estimate(lambda group: group["loss_b"])
    return metrics, uncertainty, country_mean, country_se


def _collapse_pmf(losses: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(losses) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
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


def weighted_histogram(
    losses: np.ndarray,
    weights: np.ndarray | None,
    upper: float,
    bins: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram whose bar heights sum to 1 when weights do."""
    losses = np.asarray(losses, dtype=float)
    if weights is None:
        weights = np.full(len(losses), 1.0 / max(len(losses), 1))
    else:
        weights = np.asarray(weights, dtype=float)
        total = float(weights.sum())
        if total > 0:
            weights = weights / total
    edges = np.linspace(0.0, max(upper, 1.0), bins + 1)
    counts, edges = np.histogram(losses, bins=edges, weights=weights)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, counts.astype(float)
