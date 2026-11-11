"""Local Flask dashboard for catastrophe-risk analytics.

Run with: python app.py
The interface is a light workspace with navy accents. Simulations run as
local background jobs and are stored in SQLite.
"""

from __future__ import annotations

import html
import secrets
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, redirect, render_template, request, send_file, session, url_for

from src.auth import DEMO_ACCOUNTS, add_user, authenticate, ensure_users, list_users, role_can
from src.data_loader import (
    OPTIONAL_TABLES,
    TABLE_FILES,
    load_synthetic,
    read_json_bundle,
    read_table_file,
    read_workbook,
    replace_tables,
)
from src.experiment_tracker import DEFAULT_DB, METHOD_LABELS, METHOD_ORDER, load_history
from src.exports import history_csv, run_csv, run_json, run_pdf
from src.jobs import enqueue, get_job, job_payload, list_jobs, load_distribution, start_worker
from src.risk_metrics import METRIC_LABELS, MONEY_METRICS, RunResult
from src.scenarios import DISCLAIMER, build_portfolio, list_scenarios, parameter_rows
from src.validation import validate_bundle
from src.visualisation import (
    comparison_figure,
    contract_composition_figure,
    country_loss_figure,
    exceedance_figure,
    exposure_share_figure,
    loss_distribution_figure,
    occurrence_figure,
    portfolio_cap,
    probability_comparison_figure,
    sample_spread_figure,
    scenario_comparison_figure,
    scenario_relationship_figure,
    severity_heatmap,
    truncation_figure,
)

NAV = (
    ("/", "Overview", "view"),
    ("/data", "Data", "view"),
    ("/scenarios", "Scenarios", "view"),
    ("/run", "Run", "run"),
    ("/compare", "Compare", "run"),
    ("/history", "History", "view"),
    ("/users", "Users", "manage_users"),
)

_WRITE_PERMISSION = {
    ("data_page", "POST"): "edit_book",
    ("scenarios_page", "POST"): "edit_book",
    ("run_page", "POST"): "run",
    ("compare_page", "POST"): "run",
    ("users_page", "GET"): "manage_users",
    ("users_page", "POST"): "manage_users",
}


def create_app(db_path: Path | None = None, start_background: bool = False) -> Flask:
    application = Flask(__name__)
    application.config["DB_PATH"] = str(db_path or DEFAULT_DB)
    application.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    application.secret_key = _session_secret(Path(application.config["DB_PATH"]))
    application.extensions["bundle"] = None
    ensure_users(Path(application.config["DB_PATH"]))
    if start_background:
        start_worker(Path(application.config["DB_PATH"]))

    @application.context_processor
    def _account():
        user = session.get("user")
        role = user["role"] if user else ""
        return {
            "current_user": user,
            "can_run": role_can(role, "run"),
            "can_edit": role_can(role, "edit_book"),
        }

    @application.before_request
    def _require_login():
        if request.endpoint in {None, "static", "login"}:
            return None
        user = session.get("user")
        if not user:
            target = request.full_path if request.query_string else request.path
            return redirect(url_for("login", next=_safe_next(target)))
        permission = _WRITE_PERMISSION.get((request.endpoint, request.method))
        if permission and not role_can(user["role"], permission):
            return (
                render_template(
                    "forbidden.html",
                    title="Not permitted",
                    active="",
                    nav=_nav(),
                ),
                403,
            )
        return None

    @application.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET" and session.get("user"):
            return redirect(_safe_next(request.args.get("next", "/")))
        error = None
        if request.method == "POST":
            user = authenticate(db(), request.form.get("username", ""), request.form.get("password", ""))
            if user is None:
                error = "Username or password was not recognised."
            else:
                session["user"] = user
                return redirect(_safe_next(request.form.get("next") or request.args.get("next", "/")))
        return render_template("login.html", title="Sign in", error=error, accounts=DEMO_ACCOUNTS, next=_safe_next(request.args.get("next", "/")))

    @application.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @application.route("/users", methods=["GET", "POST"])
    def users_page():
        error = None
        message = None
        if request.method == "POST":
            try:
                add_user(
                    db(),
                    request.form.get("username", ""),
                    request.form.get("display_name", ""),
                    request.form.get("role", ""),
                    request.form.get("password", ""),
                )
                message = "Account added."
            except Exception as exc:
                error = str(exc)
        return render_template(
            "users.html",
            title="Users",
            active="Users",
            nav=_nav(),
            users=list_users(db()),
            roles=("admin", "analyst", "reviewer"),
            message=message,
            error=error,
        )

    def db() -> Path:
        return Path(application.config["DB_PATH"])

    def bundle():
        if application.extensions["bundle"] is None:
            application.extensions["bundle"] = load_synthetic()
        return application.extensions["bundle"]

    def store(value) -> None:
        application.extensions["bundle"] = value

    @application.route("/")
    def overview():
        current = bundle()
        validation = validate_bundle(current)
        model = None
        parameters = pd.DataFrame(columns=["parameter", "value"])
        if validation.ok:
            model = build_portfolio(current, "base")
            parameters = parameter_rows(model)
        charts = []
        if validation.ok:
            figures = [
                exposure_share_figure(current.exposure),
                occurrence_figure(current.hazards),
                contract_composition_figure(current.contracts),
                severity_heatmap(current.hazards),
            ]
            for figure in figures:
                if figure is None:
                    continue
                charts.append(_figure_html(figure, include_js=not charts))
        return render_template(
            "overview.html",
            title="Overview",
            active="Overview",
            nav=_nav(),
            disclaimer=DISCLAIMER,
            errors=validation.errors,
            model=model,
            parameters_html=_frame_html(parameters.drop(columns=["group"], errors="ignore")) if not parameters.empty else "",
            source=_source_label(current.source),
            kpis=_overview_kpis(current, validation.ok),
            charts=charts,
        )

    @application.route("/data", methods=["GET", "POST"])
    def data_page():
        message = None
        current = bundle()
        if request.method == "POST":
            try:
                updated = _apply_upload(current, request.files)
                validation = validate_bundle(updated)
                if not validation.ok:
                    message = "Upload rejected. The previous tables are still active."
                    return _data_response(current, message, validation.errors, validation.warnings)
                store(updated)
                current = updated
                message = "Tables loaded."
            except Exception as exc:
                message = f"Upload rejected: {exc}"
        validation = validate_bundle(current)
        return _data_response(current, message, validation.errors, validation.warnings)

    def _data_response(current, message, errors, warnings):
        previews = []
        frames = {
            "hazards": current.hazards,
            "dependence": current.dependence,
            "exposure": current.exposure,
            "contracts": current.contracts,
            "scenarios": current.scenarios,
        }
        if current.peril_terms is not None and len(current.peril_terms):
            frames["peril_terms"] = current.peril_terms
        for name, frame in frames.items():
            shown = min(12, len(frame))
            previews.append((name, len(frame), shown, _frame_html(frame.head(12))))
        return render_template(
            "data.html",
            title="Data",
            active="Data",
            nav=_nav(),
            message=message,
            errors=errors,
            warnings=warnings,
            source=_source_label(current.source),
            tables=tuple(TABLE_FILES) + tuple(OPTIONAL_TABLES),
            previews=previews,
        )

    @application.route("/scenarios", methods=["GET", "POST"])
    def scenarios_page():
        message = None
        errors = []
        current = bundle()
        if request.method == "POST":
            try:
                updated = _add_scenario(current, request.form)
                validation = validate_bundle(updated)
                if not validation.ok:
                    errors = validation.errors
                else:
                    store(updated)
                    current = updated
                    message = "Scenario added to the active tables."
            except Exception as exc:
                errors = [str(exc)]
        return render_template(
            "scenarios.html",
            title="Scenarios",
            active="Scenarios",
            nav=_nav(),
            scenarios=list_scenarios(current),
            message=message,
            errors=errors,
        )

    @application.route("/run", methods=["GET", "POST"])
    def run_page():
        current = bundle()
        error = None
        if request.method == "POST":
            try:
                settings = _settings(request.form)
                build_portfolio(
                    current,
                    settings["scenario_id"],
                    shock_share_override=settings["shock_share_override"],
                    precip_scale_override=settings["precip_scale_override"],
                )
                job_id = enqueue(db(), "run", job_payload(current, "run", **settings))
                return redirect(url_for("job_page", job_id=job_id))
            except Exception as exc:
                error = str(exc)
        return render_template(
            "run.html",
            title="Run",
            active="Run",
            nav=_nav(),
            scenarios=list_scenarios(current),
            methods=list(METHOD_LABELS.items()),
            error=error,
        )

    @application.route("/compare", methods=["GET", "POST"])
    def compare_page():
        current = bundle()
        error = None
        if request.method == "POST":
            try:
                settings = _settings(request.form, require_method=request.form.get("kind") != "compare_methods")
                kind = request.form.get("kind", "")
                if kind not in {"compare_methods", "compare_scenarios"}:
                    raise ValueError("Choose a comparison.")
                if kind == "compare_methods":
                    build_portfolio(current, settings["scenario_id"])
                job_id = enqueue(db(), kind, job_payload(current, kind, **settings))
                return redirect(url_for("job_page", job_id=job_id))
            except Exception as exc:
                error = str(exc)
        return render_template(
            "compare.html",
            title="Compare",
            active="Compare",
            nav=_nav(),
            scenarios=list_scenarios(current),
            methods=[(key, METHOD_LABELS[key]) for key in METHOD_ORDER],
            error=error,
        )

    @application.route("/jobs/<job_id>")
    def job_page(job_id: str):
        job = get_job(db(), job_id)
        if job is None:
            abort(404)
        blocks = []
        runs = (job.get("result") or {}).get("runs") or []
        stubs = []
        for run in runs:
            block, stub = _run_block(db(), run, include_chart=len(runs) == 1)
            blocks.append(block)
            if stub is not None:
                stubs.append(stub)
        comparisons = []
        if len(stubs) > 1 and job.get("kind") == "compare_methods":
            comparisons = _chart_list(
                [
                    comparison_figure(stubs, "expected_loss"),
                    probability_comparison_figure(stubs, lambda result: result.method_label),
                ],
                include_js=not any(block.get("charts") for block in blocks),
            )
        elif len(stubs) > 1 and job.get("kind") == "compare_scenarios":
            comparisons = _chart_list(
                [
                    scenario_comparison_figure(stubs, "expected_loss"),
                    scenario_relationship_figure(stubs),
                ],
                include_js=not any(block.get("charts") for block in blocks),
            )
        return render_template(
            "job.html",
            title="Result",
            active="Run" if job.get("kind") == "run" else "Compare",
            nav=_nav(),
            refresh=job["status"] in {"queued", "running"},
            job=job,
            blocks=blocks,
            comparisons=comparisons,
        )

    @application.route("/history")
    def history_page():
        frame = load_history(db())
        if frame.empty:
            table = ""
        else:
            shown = frame[
                [
                    column
                    for column in (
                        "created_utc",
                        "scenario_name",
                        "method_label",
                        "expected_loss",
                        "joint_attachment_probability",
                        "runtime_seconds",
                        "run_id",
                    )
                    if column in frame.columns
                ]
            ].head(40)
            table = _frame_html(shown)
        return render_template(
            "history.html",
            title="History",
            active="History",
            nav=_nav(),
            jobs=list_jobs(db()),
            history=table,
        )

    @application.route("/exports/history.csv")
    def export_history():
        payload = history_csv(db())
        return _download(payload, "history.csv", "text/csv")

    @application.route("/exports/<run_id>.json")
    def export_json(run_id: str):
        try:
            payload = run_json(db(), run_id)
        except KeyError:
            abort(404)
        return _download(payload, f"{run_id}.json", "application/json")

    @application.route("/exports/<run_id>.csv")
    def export_csv(run_id: str):
        try:
            payload = run_csv(db(), run_id)
        except KeyError:
            abort(404)
        return _download(payload, f"{run_id}.csv", "text/csv")

    @application.route("/exports/<run_id>.pdf")
    def export_pdf(run_id: str):
        try:
            payload = run_pdf(db(), run_id)
        except KeyError:
            abort(404)
        return _download(payload, f"{run_id}.pdf", "application/pdf")

    return application


def _nav():
    role = (session.get("user") or {}).get("role", "")
    return [(href, label) for href, label, permission in NAV if role_can(role, permission)]


def _safe_next(value: str | None) -> str:
    text = (value or "/").strip()
    if text.startswith("/") and not text.startswith("//"):
        return text
    return "/"


def _session_secret(db_path: Path) -> str:
    path = db_path.parent / ".session_secret"
    if path.is_file():
        stored = path.read_text(encoding="utf-8").strip()
        if stored:
            return stored
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = secrets.token_hex(32)
    path.write_text(stored, encoding="utf-8")
    return stored


def _download(payload: bytes, filename: str, mime: str):
    import io

    return send_file(io.BytesIO(payload), mimetype=mime, as_attachment=True, download_name=filename)


def _apply_upload(current, files):
    bundle_file = files.get("bundle")
    if bundle_file and bundle_file.filename:
        payload = bundle_file.read()
        name = bundle_file.filename.lower()
        if name.endswith(".json"):
            return read_json_bundle(payload)
        if name.endswith(".xlsx"):
            return read_workbook(payload)
        raise ValueError("A bundle must be a .json object or an .xlsx workbook.")
    replacements = {}
    for table in tuple(TABLE_FILES) + tuple(OPTIONAL_TABLES):
        upload = files.get(table)
        if upload and upload.filename:
            replacements[table] = read_table_file(upload.filename, upload.read())
    if not replacements:
        raise ValueError("Choose a bundle or at least one table.")
    return replace_tables(current, replacements)


def _add_scenario(current, form):
    scenario_id = str(form.get("scenario_id", "")).strip().lower()
    if not scenario_id or not scenario_id.replace("_", "").isalnum():
        raise ValueError("Scenario id must use letters, numbers and underscores.")
    existing = {str(value).strip().lower() for value in current.scenarios["scenario_id"]}
    if scenario_id in existing:
        raise ValueError(f"Scenario '{scenario_id}' already exists.")
    mode = str(form.get("dependence_mode", "")).strip().lower()
    share = float(form["shock_share"])
    if mode == "independent" and share != 0.0:
        raise ValueError("Independent mode requires shock share 0.")
    row = {
        "scenario_id": scenario_id,
        "scenario_name": str(form.get("scenario_name", "")).strip(),
        "dependence_mode": mode,
        "shock_share": share,
        "precip_probability_scale": float(form["precip_probability_scale"]),
        "eq_probability_scale": float(form["eq_probability_scale"]),
        "description": str(form.get("description", "")).strip(),
        "assumption_label": "user_defined_demo_assumption",
    }
    if not row["scenario_name"] or not row["description"]:
        raise ValueError("Name and description are required.")
    frame = pd.concat([current.scenarios, pd.DataFrame([row])], ignore_index=True)
    return replace_tables(current, {"scenarios": frame})


def _settings(form, require_method: bool = True) -> dict:
    scenario_id = str(form.get("scenario_id") or "base").strip().lower()
    method = str(form.get("method") or "exact").strip()
    if require_method and method not in METHOD_LABELS:
        raise ValueError("Choose an estimator.")
    if not require_method:
        method = method if method in METHOD_LABELS else "exact"
    n_samples = int(form.get("n_samples") or 4096)
    seed = int(form.get("seed") or 42)
    threshold = float(form.get("threshold") or 25_000_000)
    if n_samples < 2:
        raise ValueError("Sample budget must be at least 2.")
    if threshold < 0:
        raise ValueError("Threshold must be non-negative.")
    shock_text = str(form.get("shock_share") or "").strip()
    precip_text = str(form.get("precip_scale") or "").strip()
    return {
        "scenario_id": scenario_id,
        "method": method,
        "n_samples": n_samples,
        "seed": seed,
        "threshold": threshold,
        "shock_share_override": None if shock_text == "" else float(shock_text),
        "precip_scale_override": None if precip_text == "" else float(precip_text),
    }


def _run_block(db_path: Path, run: dict, include_chart: bool) -> tuple[dict, RunResult | None]:
    metrics = []
    for key, label in METRIC_LABELS.items():
        value = run["metrics"].get(key)
        se = run["uncertainty"].get(key)
        metrics.append(
            {
                "label": label,
                "value": _format_metric(key, value),
                "detail": "" if not se else f"SE {_format_metric(key, se)}",
            }
        )
    stub = _stub(run)
    charts = []
    if include_chart and run.get("run_id") and stub is not None:
        distribution = load_distribution(db_path, run["run_id"])
        if distribution:
            _attach_distribution(stub, distribution)
        charts = _chart_list(
            [
                loss_distribution_figure(stub, portfolio_cap(stub)) if distribution else None,
                exceedance_figure(stub) if distribution else None,
                country_loss_figure(stub) if stub.country_expected_loss else None,
                sample_spread_figure(stub) if distribution else None,
                truncation_figure(stub),
            ]
        )
    return (
        {
            "title": f"{run.get('scenario_name')} — {run.get('method_label')}",
            "valid": bool(run.get("valid_estimate")),
            "metrics": metrics,
            "runtime": _round(run.get("runtime_seconds")),
            "memory": _memory(run.get("peak_memory_bytes")),
            "evaluations": run.get("n_evaluations"),
            "unit": run.get("evaluation_unit"),
            "run_id": run.get("run_id"),
            "notes": run.get("notes") or [],
            "charts": charts,
        },
        stub,
    )


def _stub(run: dict) -> RunResult | None:
    if not run.get("metrics"):
        return None
    return RunResult(
        method=run.get("method") or "",
        method_label=run.get("method_label") or "",
        scenario_id=run.get("scenario_id") or "",
        scenario_name=run.get("scenario_name") or "",
        seed=run.get("seed"),
        n_samples=run.get("n_samples"),
        n_evaluations=int(run.get("n_evaluations") or 0),
        evaluation_unit=run.get("evaluation_unit") or "",
        threshold=float(run.get("threshold") or 0.0),
        currency=run.get("currency") or "",
        model_version="",
        valid_estimate=bool(run.get("valid_estimate")),
        metrics=run.get("metrics") or {},
        uncertainty=run.get("uncertainty") or {},
        country_expected_loss=run.get("country_expected_loss") or {},
        country_expected_loss_se={key: 0.0 for key in (run.get("country_expected_loss") or {})},
        notes=list(run.get("notes") or []),
        details=run.get("details") or {},
    )


def _attach_distribution(result: RunResult, distribution: dict) -> None:
    losses = np.asarray(distribution.get("losses") or [], dtype=float)
    if distribution.get("kind") == "pmf":
        result.loss_pmf = (losses, np.asarray(distribution.get("probability") or [], dtype=float))
    elif len(losses):
        result.sample_losses = losses
        if distribution.get("weights") is not None:
            result.sample_weights = np.asarray(distribution["weights"], dtype=float)


def _figure_html(figure, include_js: bool) -> str:
    figure.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#142033", "family": "Segoe UI, Helvetica Neue, sans-serif"},
    )
    figure.update_xaxes(gridcolor="#e6edf5", zerolinecolor="#e6edf5", linecolor="#c5d0de")
    figure.update_yaxes(gridcolor="#e6edf5", zerolinecolor="#e6edf5", linecolor="#c5d0de")
    return figure.to_html(
        full_html=False,
        include_plotlyjs=include_js,
        config={"displaylogo": False, "displayModeBar": False, "responsive": True},
    )


def _chart_list(figures, include_js: bool = True) -> list[str]:
    html_blocks = []
    for figure in figures:
        if figure is None:
            continue
        html_blocks.append(_figure_html(figure, include_js=include_js and not html_blocks))
    return html_blocks


_CLIP_COLUMNS = {
    "notes",
    "description",
    "aggregation",
    "assumption_label",
    "data_label",
    "run_id",
}
_BADGE_COLUMNS = {"status", "dependence_mode", "data_quality_flag", "include_in_calculations", "valid_estimate"}


def _frame_html(frame: pd.DataFrame) -> str:
    """Render a table without letting long notes set the column width."""
    if frame is None or frame.empty:
        return "<p class=\"empty\">No rows in this table.</p>"
    numeric = {column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])}
    head = "".join(
        f"<th class=\"{'num' if column in numeric else ''}\">{html.escape(str(column).replace('_', ' '))}</th>"
        for column in frame.columns
    )
    body = []
    for record in frame.itertuples(index=False):
        cells = []
        for column, value in zip(frame.columns, record):
            css = "num" if column in numeric else ""
            cells.append(f"<td class=\"{css}\">{_cell(str(column), value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<div class=\"table-scroll\"><table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _cell(column: str, value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    text = str(value)
    if text in {"nan", "None", "<NA>"}:
        return ""
    escaped = html.escape(text)
    if column in _BADGE_COLUMNS:
        return f"<span class=\"badge badge-{_badge_kind(text)}\">{escaped}</span>"
    if column in _CLIP_COLUMNS or len(text) > 72:
        short = text if len(text) <= 64 else text[:61].rstrip() + "…"
        if len(text) <= 72:
            return f"<span class=\"clip\" title=\"{escaped}\">{html.escape(short)}</span>"
        return (
            "<details class=\"note\">"
            f"<summary title=\"{escaped}\">{html.escape(short)}</summary>"
            f"<p>{escaped}</p></details>"
        )
    return escaped


def _badge_kind(text: str) -> str:
    key = text.strip().lower()
    if key in {"yes", "valid", "succeeded", "ok", "true", "1"}:
        return "ok"
    if key in {"no", "invalid", "failed", "false", "0"}:
        return "warn"
    if key in {"queued", "running"}:
        return "info"
    return ""


def _source_label(source) -> str:
    text = str(source or "").strip()
    if not text:
        return "Active tables"
    path = Path(text)
    if "\\" in text or "/" in text:
        parts = [part for part in path.parts if part not in {"\\", "/"}]
        return " / ".join(parts[-2:]) if len(parts) >= 2 else path.name
    return text


def _overview_kpis(current, valid: bool) -> list[dict]:
    if not valid:
        return []
    exposure = current.exposure
    total = float(exposure["insured_exposure"].sum())
    currency = str(exposure["currency"].iloc[0]).strip().upper()
    countries = [str(value) for value in exposure["country"]]
    hazards = list(dict.fromkeys(str(value).strip().lower() for value in current.hazards["hazard"]))
    term = int(current.contracts["term_years"].iloc[0])
    return [
        {"label": "Insured exposure", "value": f"{total:,.0f} {currency}", "hint": "Sum of the loaded country exposures", "icon": "exposure"},
        {"label": "Countries", "value": str(len(countries)), "hint": ", ".join(countries), "icon": "countries"},
        {"label": "Contract term", "value": f"{term} years", "hint": f"Denominated in {currency}", "icon": "term"},
        {"label": "Hazards", "value": str(len(hazards)), "hint": ", ".join(hazard.replace("_", " ") for hazard in hazards), "icon": "hazard"},
        {"label": "Scenarios", "value": str(len(current.scenarios)), "hint": "Rows in the active scenario table", "icon": "scenario"},
    ]


def _memory(value) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "—"
    if number >= 1_048_576:
        return f"{number / 1_048_576:.1f} MB"
    if number >= 1024:
        return f"{number / 1024:.1f} KB"
    return f"{number} B"


def _format_metric(key: str, value) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "—"
    if key in MONEY_METRICS:
        return f"{number:,.0f} USD"
    return f"{number:.6f}"


def _round(value) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "—"


app = create_app(start_background=False)


def main() -> None:
    demo = Path(__file__).resolve().parent / "data" / "demo" / "quantumrisk_demo_bundle.xlsx"
    if demo.is_file():
        loaded = read_workbook(demo.read_bytes())
        loaded.source = str(demo)
        validation = validate_bundle(loaded)
        if not validation.ok:
            raise SystemExit("Demo workbook failed validation:\n- " + "\n- ".join(validation.errors))
        app.extensions["bundle"] = loaded
    start_worker(Path(app.config["DB_PATH"]))
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
