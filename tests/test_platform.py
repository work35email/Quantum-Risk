"""Ingestion, general portfolios, tail-aware sampling, jobs and the Flask UI."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.classical import exact, tail_aware
from src.data_loader import read_json_bundle, read_workbook, snapshot_bundle
from src.jobs import enqueue, execute_pending, get_job, job_payload
from src.multimodel import build_general_portfolio
from src.scenarios import build_portfolio
from src.validation import validate_bundle
from tests.conftest import THRESHOLD


def test_general_engine_matches_the_shipped_fixture(bundle, exact_base):
    general = build_general_portfolio(bundle, "base")
    canonical = build_portfolio(bundle, "base")
    assert general.probability.shape == canonical.annual.probability.shape
    assert general.probability == pytest.approx(canonical.annual.probability, abs=1e-10)
    from src.experiment_tracker import run_experiment

    result = run_experiment(general, "exact", THRESHOLD, 1, 8, persist=False)
    for key, truth in exact_base.metrics.items():
        if "loss" in key:
            assert result.metrics[key] == pytest.approx(truth, abs=1.0)
        else:
            assert result.metrics[key] == pytest.approx(truth, abs=1e-8)


def test_third_country_is_accepted_and_adds_loss(bundle):
    hazards = pd.concat([bundle.hazards, bundle.hazards[bundle.hazards["country"] == "Tajikistan"].assign(country="Testland")], ignore_index=True)
    exposure = pd.concat(
        [bundle.exposure, bundle.exposure[bundle.exposure["country"] == "Tajikistan"].assign(country="Testland")],
        ignore_index=True,
    )
    contracts = pd.concat(
        [bundle.contracts, bundle.contracts[bundle.contracts["country"] == "Tajikistan"].assign(country="Testland")],
        ignore_index=True,
    )
    from src.data_loader import replace_tables

    updated = replace_tables(bundle, {"hazards": hazards, "exposure": exposure, "contracts": contracts})
    validation = validate_bundle(updated)
    assert validation.ok, validation.errors
    model = build_portfolio(updated, "base")
    assert model.countries == ("Kyrgyz Republic", "Tajikistan", "Testland")
    assert model.probability.shape == (4, 4, 4, 3, 3, 3)
    assert model.probability.sum() == pytest.approx(1.0)
    from src.experiment_tracker import run_experiment

    blocked = run_experiment(model, "exact", THRESHOLD, 1, 4, persist=False)
    assert not blocked.valid_estimate
    assert "500 entries" in " ".join(blocked.notes)
    sampled = run_experiment(model, "monte_carlo", THRESHOLD, 3, 800, persist=False)
    baseline = exact.estimate(build_portfolio(bundle, "base"), THRESHOLD)
    assert sampled.metrics["expected_loss"] > baseline.metrics["expected_loss"]


def test_json_and_excel_round_trip(bundle, tmp_path):
    payload = json.dumps(snapshot_bundle(bundle)).encode()
    restored = read_json_bundle(payload)
    assert validate_bundle(restored).ok
    excel = tmp_path / "bundle.xlsx"
    with pd.ExcelWriter(excel, engine="openpyxl") as writer:
        bundle.hazards.to_excel(writer, sheet_name="hazards", index=False)
        bundle.dependence.to_excel(writer, sheet_name="dependence", index=False)
        bundle.exposure.to_excel(writer, sheet_name="exposure", index=False)
        bundle.contracts.to_excel(writer, sheet_name="contracts", index=False)
        bundle.scenarios.to_excel(writer, sheet_name="scenarios", index=False)
    loaded = read_workbook(excel.read_bytes())
    assert validate_bundle(loaded).ok
    assert list(loaded.hazards["country"]) == list(bundle.hazards["country"])


def test_tail_aware_weights_are_unbiased(base_model, exact_base):
    result = tail_aware.estimate(base_model, THRESHOLD, seed=7, n_samples=6000)
    assert abs(result.details["mean_weight"] - 1.0) <= 4 * result.details["se_weight"] + 0.02
    assert result.details["proposal_shock_share"] > 0.5
    assert result.metrics["expected_loss"] == pytest.approx(exact_base.metrics["expected_loss"], rel=0.08, abs=2_000_000)


def test_job_runs_inline_and_exports(bundle, tmp_path):
    from app import create_app
    from src.exports import run_csv, run_json, run_pdf

    database = tmp_path / "experiments.sqlite"
    application = create_app(database, start_background=False)
    client = application.test_client()
    signed_in = client.post("/login", data={"username": "admin", "password": "demo-admin"})
    assert signed_in.status_code == 302
    page = client.get("/")
    assert page.status_code == 200
    assert b"Tensor train" in page.data
    queued = client.post("/run", data={"scenario_id": "base", "method": "exact", "n_samples": "32", "seed": "42", "threshold": "25000000"})
    assert queued.status_code == 302
    job_id = queued.headers["Location"].rstrip("/").split("/")[-1]
    assert execute_pending(database, limit=1) == 1
    job = get_job(database, job_id)
    assert job["status"] == "succeeded"
    run_id = job["result"]["runs"][0]["run_id"]
    assert job["result"]["runs"][0]["metrics"]["expected_loss"] == pytest.approx(23_227_734, abs=1)
    body = client.get(f"/jobs/{job_id}")
    assert b"23,227,734" in body.data
    assert run_json(database, run_id).startswith(b"{")
    assert b"expected_loss" in run_csv(database, run_id)
    assert run_pdf(database, run_id).startswith(b"%PDF")
    history = client.get("/exports/history.csv")
    assert history.status_code == 200
    assert b"base" in history.data


def test_reviewer_can_read_but_cannot_run_or_replace_data(tmp_path):
    from app import create_app

    application = create_app(tmp_path / "roles.sqlite", start_background=False)
    client = application.test_client()
    denied = client.get("/")
    assert denied.status_code == 302
    assert "/login" in denied.headers["Location"]

    reviewer = client.post("/login", data={"username": "reviewer", "password": "demo-reviewer"})
    assert reviewer.status_code == 302
    overview = client.get("/")
    assert overview.status_code == 200
    assert b"Tensor train" in overview.data
    assert b"Queue simulation" not in client.get("/run").data
    blocked = client.post("/run", data={"scenario_id": "base", "method": "exact", "n_samples": "8", "seed": "1", "threshold": "25000000"})
    assert blocked.status_code == 403
    assert client.get("/users").status_code == 403
    data_page = client.get("/data")
    assert b"Load data" not in data_page.data

    analyst = client.post("/login", data={"username": "analyst", "password": "demo-analyst"})
    assert analyst.status_code == 302
    queued = client.post("/run", data={"scenario_id": "base", "method": "exact", "n_samples": "8", "seed": "1", "threshold": "25000000"})
    assert queued.status_code == 302
    assert b"Load data" not in client.get("/data").data


def test_invalid_json_bundle_is_rejected():
    with pytest.raises(ValueError):
        read_json_bundle(b'{"hazards": []}')


def test_enqueue_payload_keeps_the_scenario_snapshot(bundle, tmp_path):
    database = tmp_path / "jobs.sqlite"
    job_id = enqueue(
        database,
        "run",
        job_payload(bundle, "run", scenario_id="independent", method="exact", n_samples=8, seed=1, threshold=THRESHOLD, shock_share_override=None, precip_scale_override=None),
    )
    assert get_job(database, job_id)["status"] == "queued"
    assert execute_pending(database) == 1
    done = get_job(database, job_id)
    assert done["status"] == "succeeded"
    assert done["result"]["runs"][0]["scenario_id"] == "independent"
