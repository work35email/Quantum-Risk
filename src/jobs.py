"""SQLite-backed background jobs for local simulation runs."""

from __future__ import annotations

import json
import sqlite3
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.data_loader import bundle_from_snapshot, snapshot_bundle
from src.experiment_tracker import compare_methods, compare_scenarios, json_ready, run_experiment
from src.scenarios import build_portfolio

_worker_lock = threading.Lock()
_worker: threading.Thread | None = None
_stop = threading.Event()


def start_worker(db_path: Path) -> None:
    """Start one daemon thread that drains queued jobs for this database."""
    global _worker
    path = Path(db_path)
    with _worker_lock:
        if _worker is not None and _worker.is_alive():
            return
        _stop.clear()
        _worker = threading.Thread(target=_loop, args=(path,), name="quantumrisk-jobs", daemon=True)
        _worker.start()


def enqueue(db_path: Path, kind: str, payload: dict) -> str:
    """Insert a queued job and return its id. The payload is the full input snapshot."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    with sqlite3.connect(path) as connection:
        _ensure_jobs(connection)
        connection.execute(
            """
            INSERT INTO jobs (job_id, created_utc, status, kind, payload_json, result_json, error, run_ids_json)
            VALUES (?, ?, 'queued', ?, ?, NULL, NULL, NULL)
            """,
            (job_id, _now(), kind, json.dumps(json_ready(payload))),
        )
    return job_id


def execute_pending(db_path: Path, limit: int = 1) -> int:
    """Run up to ``limit`` queued jobs in this process. Returns how many finished."""
    path = Path(db_path)
    if not path.exists():
        return 0
    finished = 0
    for _ in range(limit):
        job = _claim(path)
        if job is None:
            break
        try:
            public = _execute(path, job)
            _finish(path, job["job_id"], "succeeded", public, None)
        except Exception as exc:
            _finish(path, job["job_id"], "failed", None, f"{exc}\n{traceback.format_exc()}")
        finished += 1
    return finished


def get_job(db_path: Path, job_id: str) -> dict | None:
    path = Path(db_path)
    if not path.exists():
        return None
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_jobs(connection)
        row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    job = dict(row)
    if job.get("result_json"):
        job["result"] = json.loads(job["result_json"])
    return job


def list_jobs(db_path: Path, limit: int = 40) -> list[dict]:
    path = Path(db_path)
    if not path.exists():
        return []
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_jobs(connection)
        rows = connection.execute(
            "SELECT job_id, created_utc, status, kind, error FROM jobs ORDER BY created_utc DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def save_distribution(db_path: Path, result) -> Path | None:
    """Write the loss distribution used by the dashboard chart."""
    run_id = result.details.get("run_id")
    if not run_id:
        return None
    folder = Path(db_path).parent / "distributions"
    folder.mkdir(parents=True, exist_ok=True)
    payload: dict = {"run_id": run_id, "method": result.method, "currency": result.currency}
    if result.loss_pmf is not None:
        losses, probability = result.loss_pmf
        payload["kind"] = "pmf"
        payload["losses"] = [float(value) for value in losses]
        payload["probability"] = [float(value) for value in probability]
    elif result.sample_losses is not None:
        limit = 8000
        losses = np.asarray(result.sample_losses, dtype=float)[:limit]
        payload["kind"] = "samples"
        payload["losses"] = [float(value) for value in losses]
        if result.sample_weights is not None:
            weights = np.asarray(result.sample_weights, dtype=float)[: len(losses)]
            payload["weights"] = [float(value) for value in weights]
    path = folder / f"{run_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_distribution(db_path: Path, run_id: str) -> dict | None:
    path = Path(db_path).parent / "distributions" / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _loop(db_path: Path) -> None:
    while not _stop.is_set():
        try:
            ran = execute_pending(db_path, limit=1)
        except Exception:
            ran = 0
        if not ran:
            _stop.wait(0.5)


def _claim(db_path: Path) -> dict | None:
    with sqlite3.connect(db_path, timeout=30) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_jobs(connection)
        row = connection.execute(
            """
            UPDATE jobs
            SET status = 'running'
            WHERE job_id = (
                SELECT job_id FROM jobs WHERE status = 'queued' ORDER BY created_utc LIMIT 1
            )
            RETURNING job_id, created_utc, status, kind, payload_json
            """
        ).fetchone()
    return None if row is None else dict(row)


def _execute(db_path: Path, job: dict) -> dict:
    payload = json.loads(job["payload_json"])
    bundle = bundle_from_snapshot(payload["bundle"])
    threshold = float(payload["threshold"])
    seed = int(payload["seed"])
    n_samples = int(payload["n_samples"])
    shock = payload.get("shock_share_override")
    precip = payload.get("precip_scale_override")
    kind = job["kind"]
    if kind == "run":
        model = build_portfolio(
            bundle,
            payload["scenario_id"],
            shock_share_override=shock,
            precip_scale_override=precip,
        )
        result = run_experiment(model, payload["method"], threshold, seed, n_samples, db_path=db_path)
        save_distribution(db_path, result)
        return {"runs": [_public(result)]}
    if kind == "compare_methods":
        model = build_portfolio(
            bundle,
            payload["scenario_id"],
            shock_share_override=shock,
            precip_scale_override=precip,
        )
        results = compare_methods(model, threshold, seed, n_samples, db_path=db_path)
    elif kind == "compare_scenarios":
        results = compare_scenarios(
            bundle,
            payload["method"],
            threshold,
            seed,
            n_samples,
            shock_share_override=shock,
            precip_scale_override=precip,
            db_path=db_path,
        )
    else:
        raise ValueError(f"Unknown job kind '{kind}'.")
    for result in results:
        save_distribution(db_path, result)
    return {"runs": [_public(result) for result in results]}


def _public(result) -> dict:
    return {
        "run_id": result.details.get("run_id"),
        "method": result.method,
        "method_label": result.method_label,
        "scenario_id": result.scenario_id,
        "scenario_name": result.scenario_name,
        "valid_estimate": bool(result.valid_estimate),
        "currency": result.currency,
        "threshold": result.threshold,
        "seed": result.details.get("requested_seed", result.seed),
        "n_samples": result.n_samples,
        "runtime_seconds": result.runtime_seconds,
        "peak_memory_bytes": result.peak_memory_bytes,
        "n_evaluations": result.n_evaluations,
        "evaluation_unit": result.evaluation_unit,
        "metrics": {key: _number(value) for key, value in result.metrics.items()},
        "uncertainty": {key: _number(value) for key, value in result.uncertainty.items()},
        "country_expected_loss": {key: _number(value) for key, value in result.country_expected_loss.items()},
        "notes": list(result.notes),
        "details": json_ready(dict(result.details)),
    }


def _finish(db_path: Path, job_id: str, status: str, public: dict | None, error: str | None) -> None:
    run_ids = []
    if public and public.get("runs"):
        run_ids = [run.get("run_id") for run in public["runs"] if run.get("run_id")]
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE jobs
            SET status = ?, result_json = ?, error = ?, run_ids_json = ?
            WHERE job_id = ?
            """,
            (
                status,
                None if public is None else json.dumps(json_ready(public)),
                error,
                json.dumps(run_ids),
                job_id,
            ),
        )


def _ensure_jobs(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            created_utc TEXT NOT NULL,
            status TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            result_json TEXT,
            error TEXT,
            run_ids_json TEXT
        )
        """
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def job_payload(bundle, kind: str, **fields) -> dict:
    """Snapshot the active tables with the run settings."""
    payload = {"bundle": snapshot_bundle(bundle), "kind": kind}
    payload.update(fields)
    return payload
