"""Run estimators, record metadata, and export tables."""

from __future__ import annotations

import json
import math
import sqlite3
import time
import tracemalloc
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import __version__
from src.classical import exact, importance_sampling, monte_carlo, sobol, stratified, tail_aware
from src.data_loader import ROOT
from src.hazard_model import PortfolioModel
from src.quantum_inspired import tensor_network
from src.risk_metrics import METRIC_KEYS, RunResult
from src.scenarios import DISCLAIMER, build_portfolio

DEFAULT_DB = ROOT / "outputs" / "experiments.sqlite"

METHOD_LABELS = {
    "exact": "Exact enumeration",
    "monte_carlo": "Monte Carlo",
    "sobol": "Scrambled Sobol",
    "stratified": "Stratified sampling",
    "importance_sampling": "Importance sampling",
    "tail_aware": "Tail-aware proposal",
    "tensor_network": "Tensor network (experimental)",
}
METHOD_ORDER = tuple(METHOD_LABELS)


def library_versions() -> dict[str, str]:
    import numpy
    import pandas
    import scipy

    versions = {
        "quantumrisk": __version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import quimb

        versions["quimb"] = quimb.__version__
    except ImportError:
        versions["quimb"] = "not installed"
    try:
        import plotly

        versions["plotly"] = plotly.__version__
    except ImportError:
        versions["plotly"] = "not installed"
    try:
        import cotengra

        versions["cotengra"] = cotengra.__version__
    except ImportError:
        versions["cotengra"] = "not installed"
    try:
        from importlib.metadata import version

        versions["flask"] = version("flask")
    except Exception:
        versions["flask"] = "not installed"
    return versions


def run_experiment(
    model: PortfolioModel,
    method: str,
    threshold: float,
    seed: int,
    n_samples: int,
    db_path: Path | None = None,
    reference: RunResult | None = None,
    persist: bool = True,
) -> RunResult:
    """Time one estimator, attach its error against exact enumeration, and save it."""
    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown method '{method}'.")
    if threshold < 0:
        raise ValueError("Threshold must be non-negative.")
    if n_samples < 2:
        raise ValueError("Sample budget must be at least 2.")

    result, elapsed, peak = _timed(lambda: _dispatch(model, method, threshold, seed, n_samples))
    result.runtime_seconds = elapsed
    result.peak_memory_bytes = peak
    result.details["requested_seed"] = int(seed)
    result.details["seed_affects_estimate"] = method not in {"exact", "tensor_network"}
    if method == "exact" and result.valid_estimate:
        result.details["abs_error"] = {key: 0.0 for key in METRIC_KEYS}
        reference_result = result
    elif method == "exact":
        result.details["abs_error"] = {key: None for key in METRIC_KEYS}
        reference_result = None
    else:
        reference_result = reference if reference is not None else _exact_reference(model, threshold)
        if reference_result is not None and not reference_result.valid_estimate:
            reference_result = None
        if result.valid_estimate and reference_result is not None:
            result.details["abs_error"] = result.error_against(reference_result)
        else:
            result.details["abs_error"] = {key: None for key in METRIC_KEYS}
    if persist:
        _insert_run(result, model, db_path or DEFAULT_DB, int(seed), int(n_samples))
    result.details["reference_expected_loss"] = (
        None if reference_result is None else reference_result.metrics.get("expected_loss")
    )
    return result


def compare_methods(
    model: PortfolioModel,
    threshold: float,
    seed: int,
    n_samples: int,
    db_path: Path | None = None,
    persist: bool = True,
) -> list[RunResult]:
    """Run every estimator on one model with the same threshold, seed and budget."""
    exact_result = run_experiment(
        model, "exact", threshold, seed, n_samples, db_path=db_path, persist=persist
    )
    results = [exact_result]
    for method in METHOD_ORDER:
        if method == "exact":
            continue
        results.append(
            run_experiment(
                model,
                method,
                threshold,
                seed,
                n_samples,
                db_path=db_path,
                reference=exact_result,
                persist=persist,
            )
        )
    return results


def compare_scenarios(
    bundle,
    method: str,
    threshold: float,
    seed: int,
    n_samples: int,
    shock_share_override: float | None = None,
    precip_scale_override: float | None = None,
    db_path: Path | None = None,
    persist: bool = True,
) -> list[RunResult]:
    """Run one estimator on each of the five scenarios."""
    from src.scenarios import list_scenarios

    results = []
    for spec in list_scenarios(bundle):
        model = build_portfolio(
            bundle,
            spec.scenario_id,
            shock_share_override=shock_share_override,
            precip_scale_override=precip_scale_override,
        )
        results.append(
            run_experiment(
                model,
                method,
                threshold,
                seed,
                n_samples,
                db_path=db_path,
                persist=persist,
            )
        )
    return results


def load_history(db_path: Path | None = None) -> pd.DataFrame:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return pd.DataFrame()
    with sqlite3.connect(path) as connection:
        found = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'runs'").fetchone()
        if found is None:
            return pd.DataFrame()
        return pd.read_sql_query("SELECT * FROM runs ORDER BY created_utc DESC", connection)


def _exact_reference(model, threshold: float) -> RunResult:
    from src.multimodel import GeneralPortfolio, estimate_general

    if isinstance(model, GeneralPortfolio):
        return estimate_general(model, "exact", threshold, 0, 2)
    return exact.estimate(model, threshold)


def _dispatch(model: PortfolioModel, method: str, threshold: float, seed: int, n_samples: int) -> RunResult:
    from src.multimodel import GeneralPortfolio, estimate_general

    if isinstance(model, GeneralPortfolio):
        return estimate_general(model, method, threshold, seed, n_samples)
    if method == "exact":
        return exact.estimate(model, threshold)
    if method == "monte_carlo":
        return monte_carlo.estimate(model, threshold, seed, n_samples)
    if method == "sobol":
        return sobol.estimate(model, threshold, seed, n_samples)
    if method == "stratified":
        return stratified.estimate(model, threshold, seed, n_samples)
    if method == "importance_sampling":
        return importance_sampling.estimate(model, threshold, seed, n_samples)
    if method == "tail_aware":
        return tail_aware.estimate(model, threshold, seed, n_samples)
    if method == "tensor_network":
        return tensor_network.estimate(model, threshold, seed)
    raise ValueError(f"Unknown method '{method}'.")


def _timed(function):
    tracemalloc.start()
    started = time.perf_counter()
    try:
        value = function()
    finally:
        elapsed = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return value, elapsed, int(peak)


def _insert_run(result: RunResult, model: PortfolioModel, db_path: Path, seed: int, n_samples: int) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + result.method + "_" + uuid.uuid4().hex[:8]
    result.details["run_id"] = run_id
    errors = result.details.get("abs_error") or {}
    config = {
        "model_version": model.model_version,
        "libraries": library_versions(),
        "scenario": model.parameters,
        "description": model.description,
        "assumption_label": model.assumption_label,
        "data_labels": model.data_labels,
        "warnings": model.warnings,
        "threshold": result.threshold,
        "requested_seed": seed,
        "n_samples": n_samples,
        "method": result.method,
        "disclaimer": DISCLAIMER,
        "details": result.details,
    }
    countries = list(result.countries)
    country_b = countries[1] if len(countries) > 1 else ""
    payload = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_id": result.scenario_id,
        "scenario_name": result.scenario_name,
        "method": result.method,
        "method_label": result.method_label,
        "seed": seed,
        "n_samples": result.n_samples,
        "threshold": result.threshold,
        "currency": result.currency,
        "model_version": result.model_version,
        "valid_estimate": int(result.valid_estimate),
        "expected_loss": _finite(result.metrics.get("expected_loss")),
        "joint_attachment_probability": _finite(result.metrics.get("joint_attachment_probability")),
        "exceedance_probability": _finite(result.metrics.get("exceedance_probability")),
        "expected_stop_loss": _finite(result.metrics.get("expected_stop_loss")),
        "both_exhausted_probability": _finite(result.metrics.get("both_exhausted_probability")),
        "se_expected_loss": _finite(result.uncertainty.get("expected_loss")),
        "se_joint_attachment_probability": _finite(result.uncertainty.get("joint_attachment_probability")),
        "se_exceedance_probability": _finite(result.uncertainty.get("exceedance_probability")),
        "se_expected_stop_loss": _finite(result.uncertainty.get("expected_stop_loss")),
        "se_both_exhausted_probability": _finite(result.uncertainty.get("both_exhausted_probability")),
        "abs_error_expected_loss": _finite(errors.get("expected_loss")),
        "abs_error_joint_attachment_probability": _finite(errors.get("joint_attachment_probability")),
        "abs_error_exceedance_probability": _finite(errors.get("exceedance_probability")),
        "abs_error_expected_stop_loss": _finite(errors.get("expected_stop_loss")),
        "abs_error_both_exhausted_probability": _finite(errors.get("both_exhausted_probability")),
        "country_a": countries[0] if countries else "",
        "country_b": country_b,
        "expected_loss_country_a": _finite(result.country_expected_loss.get(countries[0])) if countries else None,
        "expected_loss_country_b": _finite(result.country_expected_loss.get(country_b)) if country_b else None,
        "runtime_seconds": result.runtime_seconds,
        "peak_memory_bytes": result.peak_memory_bytes,
        "n_evaluations": result.n_evaluations,
        "evaluation_unit": result.evaluation_unit,
        "ess": _finite(result.details.get("ess")),
        "config_json": json.dumps(json_ready(config)),
        "notes": " ".join(result.notes),
    }
    with sqlite3.connect(db_path) as connection:
        _ensure_schema(connection)
        columns = ", ".join(payload)
        placeholders = ", ".join(f":{column}" for column in payload)
        connection.execute(f"INSERT INTO runs ({columns}) VALUES ({placeholders})", payload)


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            created_utc TEXT NOT NULL,
            scenario_id TEXT NOT NULL,
            scenario_name TEXT NOT NULL,
            method TEXT NOT NULL,
            method_label TEXT NOT NULL,
            seed INTEGER,
            n_samples INTEGER,
            threshold REAL NOT NULL,
            currency TEXT NOT NULL,
            model_version TEXT NOT NULL,
            valid_estimate INTEGER NOT NULL,
            expected_loss REAL,
            joint_attachment_probability REAL,
            exceedance_probability REAL,
            expected_stop_loss REAL,
            both_exhausted_probability REAL,
            se_expected_loss REAL,
            se_joint_attachment_probability REAL,
            se_exceedance_probability REAL,
            se_expected_stop_loss REAL,
            se_both_exhausted_probability REAL,
            abs_error_expected_loss REAL,
            abs_error_joint_attachment_probability REAL,
            abs_error_exceedance_probability REAL,
            abs_error_expected_stop_loss REAL,
            abs_error_both_exhausted_probability REAL,
            country_a TEXT,
            country_b TEXT,
            expected_loss_country_a REAL,
            expected_loss_country_b REAL,
            runtime_seconds REAL,
            peak_memory_bytes INTEGER,
            n_evaluations INTEGER,
            evaluation_unit TEXT,
            ess REAL,
            config_json TEXT NOT NULL,
            notes TEXT
        )
        """
    )


def json_ready(value):
    """Convert result metadata into JSON-safe objects."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def _finite(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number
