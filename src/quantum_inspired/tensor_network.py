"""Tensor-train / MPS representation of the annual joint distribution.

A tensor train stores a high-dimensional probability table as a chain of small
cores. Quimb holds those cores. Cotengra searches for a cheaper contraction
order. The object is a classical factorisation, not a quantum circuit and not
a run on quantum hardware.

A full-bond decomposition is a representation check. Risk metrics are taken
from the contracted tensor only when the reconstruction is non-negative and
agrees with the original table to numerical tolerance. Those metrics then use
the same term calculation as exact enumeration. Matching the exact result on
this toy model is a validity check. It is not evidence of an accuracy, speed,
or quantum advantage.

Truncated bond dimensions are reported as an approximation experiment. A
truncated network is not used as a risk estimator.
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from src.classical.exact import enumerate_country_distribution
from src.hazard_model import PortfolioModel
from src.risk_metrics import METRIC_KEYS, RunResult, summarise_pmf
from src.scenarios import DISCLAIMER

RECONSTRUCTION_TOLERANCE = 1e-8
TRUNCATION_BONDS = (1, 2, 4)


def estimate(model: PortfolioModel, threshold: float, seed: int | None = None) -> RunResult:
    """Contract a Quimb tensor train and, if it is exact, compute risk metrics.

    ``seed`` is recorded for run metadata. Contraction does not use it.
    """
    started = time.perf_counter()
    try:
        report = analyse(model.annual.probability)
    except ImportError as exc:
        return _invalid(
            model,
            threshold,
            seed,
            ["Quimb is not installed, so the tensor-network experiment did not run.", str(exc), DISCLAIMER],
            {"import_error": str(exc)},
        )
    setup_seconds = time.perf_counter() - started
    notes = list(report["limitations"])
    notes.append(DISCLAIMER)
    details = {
        "library": report["library"],
        "optimizer": report.get("optimizer"),
        "cotengra": report.get("cotengra"),
        "bond_dimensions": report["full"]["bond_dimensions"],
        "parameter_count": report["full"]["parameter_count"],
        "dense_size": report["dense_size"],
        "l1_error": report["full"]["l1_error"],
        "max_abs_error": report["full"]["max_abs_error"],
        "probability_sum": report["full"]["probability_sum"],
        "n_negative": report["full"]["n_negative"],
        "setup_seconds": setup_seconds,
        "truncation": report["truncation"],
        "quimb_matches_numpy": report["quimb_matches_numpy"],
        "sequential_query_max_abs_error": report["sequential_query_max_abs_error"],
        "valid_for_metrics": report["valid_for_metrics"],
    }
    if not report["valid_for_metrics"]:
        notes.insert(
            0,
            "The tensor train did not reproduce the annual distribution closely enough. "
            "No risk metrics are reported for it.",
        )
        result = _invalid(model, threshold, seed, notes, details)
        result.n_evaluations = int(model.annual.probability.size)
        return result

    query_started = time.perf_counter()
    loss_a, loss_b, probs, transitions = enumerate_country_distribution(
        report["reconstructed"], model.contracts[0], model.contracts[1]
    )
    # The reconstructed table can differ from 1 by a few ulps. Rescale only
    # after the validity check has already required a tiny total-variation gap.
    probs = probs / probs.sum()
    max_a, max_b = model.maximum_payouts()
    metrics, uncertainty, country_mean, country_se, pmf = summarise_pmf(
        loss_a, loss_b, probs, threshold, max_a, max_b, model.countries
    )
    details["query_seconds"] = time.perf_counter() - query_started
    details["state_transitions"] = int(transitions)
    details["abs_error_source"] = (
        "Metrics come from the contracted tensor train, then the same term payout calculation as enumeration."
    )
    notes.insert(
        0,
        "Experimental representation check. Full-bond contraction matched the annual table, "
        "so the term metrics are mathematically the exact discrete-model values. "
        "This is not a faster or more accurate estimator than enumeration, and it is not a quantum computation.",
    )
    return RunResult(
        method="tensor_network",
        method_label="Tensor network (experimental)",
        scenario_id=model.scenario_id,
        scenario_name=model.scenario_name,
        seed=None if seed is None else int(seed),
        n_samples=None,
        n_evaluations=int(model.annual.probability.size),
        evaluation_unit="elements of the annual probability tensor",
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
        loss_pmf=pmf,
        countries=model.countries,
    )


def analyse(probability: np.ndarray) -> dict:
    """Decompose ``probability`` with a tensor train and contract it in Quimb."""
    import quimb

    import cotengra

    probability = np.asarray(probability, dtype=float)
    full_cores = tt_svd(probability, max_bond=None)
    reconstructed = contract_with_quimb(full_cores)
    numpy_reconstructed = reconstruct_numpy(full_cores)
    quimb_gap = float(np.max(np.abs(reconstructed - numpy_reconstructed)))
    full = _error_record(probability, reconstructed, full_cores)
    truncation = []
    for max_bond in TRUNCATION_BONDS:
        cores = tt_svd(probability, max_bond=max_bond)
        approx = contract_with_quimb(cores)
        truncation.append(_error_record(probability, approx, cores, max_bond=max_bond))
    sequential_error = _sequential_query_error(full_cores, probability)
    valid = (
        full["max_abs_error"] <= RECONSTRUCTION_TOLERANCE
        and full["l1_error"] <= RECONSTRUCTION_TOLERANCE
        and full["n_negative"] == 0
        and abs(full["probability_sum"] - 1.0) <= RECONSTRUCTION_TOLERANCE
        and sequential_error <= RECONSTRUCTION_TOLERANCE
    )
    return {
        "library": f"quimb {quimb.__version__}",
        "optimizer": "cotengra.HyperOptimizer",
        "cotengra": cotengra.__version__,
        "dense_size": int(probability.size),
        "full": full,
        "truncation": truncation,
        "quimb_matches_numpy": quimb_gap <= RECONSTRUCTION_TOLERANCE,
        "sequential_query_max_abs_error": sequential_error,
        "valid_for_metrics": valid,
        "reconstructed": reconstructed,
        "limitations": [
            "The tensor train (matrix-product state) stores the annual probability table as a chain of smaller cores.",
            "Cotengra searches for the contraction order. Quimb performs the contraction. Neither step uses a quantum device.",
            "Setup cost (decomposition and contraction) is separate from the term-metric query.",
            "Truncated bond dimensions are an approximation study only and are excluded from risk estimates.",
            "On this small model, exact enumeration is already cheap. Agreement with it is a validity check, not an improvement.",
        ],
    }


def tt_svd(probability: np.ndarray, max_bond: int | None) -> list[np.ndarray]:
    """Tensor-train SVD. ``max_bond=None`` keeps the numerical rank."""
    working = np.asarray(probability, dtype=float)
    dims = working.shape
    cores: list[np.ndarray] = []
    left_bond = 1
    for site, dim in enumerate(dims[:-1]):
        working = working.reshape(left_bond * dim, -1)
        u_matrix, singular, v_matrix = np.linalg.svd(working, full_matrices=False)
        rank = len(singular)
        if singular.size:
            rank = int(np.sum(singular > singular[0] * 1e-12))
        rank = max(rank, 1)
        if max_bond is not None:
            rank = min(rank, int(max_bond))
        core = u_matrix[:, :rank].reshape(left_bond, dim, rank)
        cores.append(core)
        working = singular[:rank, None] * v_matrix[:rank, :]
        left_bond = rank
    cores.append(working.reshape(left_bond, dims[-1], 1))
    return cores


def reconstruct_numpy(cores: list[np.ndarray]) -> np.ndarray:
    """Dense reconstruction used as a cross-check on the Quimb contraction."""
    accumulated = cores[0][0]
    for core in cores[1:]:
        accumulated = np.tensordot(accumulated, core, axes=([-1], [0]))
    return np.asarray(accumulated[..., 0], dtype=float)


def contract_with_quimb(cores: list[np.ndarray]) -> np.ndarray:
    """Contract the tensor train with Quimb, using Cotengra to pick the order."""
    import cotengra as ctg
    import quimb.tensor as qtn

    tensors = []
    for index, core in enumerate(cores):
        tensors.append(
            qtn.Tensor(
                np.asarray(core, dtype=float),
                inds=(f"b{index}", f"p{index}", f"b{index + 1}"),
                tags=(f"site-{index}",),
            )
        )
    network = qtn.TensorNetwork(tensors)
    output_inds = [f"p{index}" for index in range(len(cores))] + ["b0", f"b{len(cores)}"]
    optimizer = ctg.HyperOptimizer(methods=["greedy"], max_repeats=4, progbar=False, parallel=False)
    result = network.contract(output_inds=output_inds, optimize=optimizer)
    data = np.asarray(result.data, dtype=float)
    return np.squeeze(data)


def _error_record(
    original: np.ndarray,
    reconstructed: np.ndarray,
    cores: list[np.ndarray],
    max_bond: int | None = None,
) -> dict:
    gap = reconstructed - original
    return {
        "max_bond": "full" if max_bond is None else int(max_bond),
        "bond_dimensions": [int(core.shape[0]) for core in cores] + [int(cores[-1].shape[2])],
        "parameter_count": int(sum(core.size for core in cores)),
        "l1_error": float(np.sum(np.abs(gap))),
        "max_abs_error": float(np.max(np.abs(gap))),
        "probability_sum": float(reconstructed.sum()),
        "n_negative": int(np.sum(reconstructed < -1e-10)),
    }


def _sequential_query_error(cores: list[np.ndarray], probability: np.ndarray) -> float:
    """Compare sequential conditionals from the tensor train with the dense table.

    This is the query a sampler would use. It is exact only when the cores
    reproduce a non-negative distribution, which the full-bond train does.
    """
    right = [np.array([1.0])]
    for core in reversed(cores):
        right.append(np.einsum("axb,b->a", core, right[-1]))
    right = right[::-1]
    max_gap = 0.0
    ranges = [range(dim) for dim in probability.shape]
    for outcome in itertools.product(*ranges):
        queried = _sequential_probability(cores, right, outcome)
        max_gap = max(max_gap, abs(queried - float(probability[outcome])))
    return float(max_gap)


def _sequential_probability(cores: list[np.ndarray], right_env: list[np.ndarray], outcome: tuple[int, ...]) -> float:
    weight = np.array([1.0])
    probability = 1.0
    for site, symbol in enumerate(outcome):
        scores = np.einsum("a,axb,b->x", weight, cores[site], right_env[site + 1])
        total = float(scores.sum())
        if total <= 0.0:
            return 0.0
        probability *= float(scores[symbol] / total)
        weight = np.einsum("a,ab->b", weight, cores[site][:, symbol, :])
    return float(probability)


def _invalid(model: PortfolioModel, threshold: float, seed: int | None, notes: list[str], details: dict) -> RunResult:
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
