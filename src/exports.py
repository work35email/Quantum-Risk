"""CSV, JSON and PDF exports for saved runs."""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

from src.experiment_tracker import json_ready, load_history
from src.risk_metrics import METRIC_LABELS
from src.scenarios import DISCLAIMER


def run_record(db_path: Path, run_id: str) -> dict | None:
    path = Path(db_path)
    if not path.exists():
        return None
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    record = dict(row)
    if record.get("config_json"):
        record["config"] = json.loads(record["config_json"])
    return record


def history_csv(db_path: Path) -> bytes:
    frame = load_history(db_path)
    return frame.to_csv(index=False).encode("utf-8")


def run_json(db_path: Path, run_id: str) -> bytes:
    record = run_record(db_path, run_id)
    if record is None:
        raise KeyError(run_id)
    return json.dumps(json_ready(record), indent=2).encode("utf-8")


def run_csv(db_path: Path, run_id: str) -> bytes:
    record = run_record(db_path, run_id)
    if record is None:
        raise KeyError(run_id)
    lines = ["field,value"]
    fields = [
        "run_id",
        "created_utc",
        "scenario_id",
        "scenario_name",
        "method",
        "method_label",
        "seed",
        "n_samples",
        "threshold",
        "currency",
        "valid_estimate",
        "expected_loss",
        "joint_attachment_probability",
        "exceedance_probability",
        "expected_stop_loss",
        "both_exhausted_probability",
        "se_expected_loss",
        "se_joint_attachment_probability",
        "se_exceedance_probability",
        "se_expected_stop_loss",
        "se_both_exhausted_probability",
        "runtime_seconds",
        "peak_memory_bytes",
        "n_evaluations",
        "notes",
    ]
    for field in fields:
        lines.append(f"{field},{_csv_cell(record.get(field))}")
    lines.append(f"disclaimer,{_csv_cell(DISCLAIMER)}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def run_pdf(db_path: Path, run_id: str) -> bytes:
    """One-page PDF summary. White page, dark purple headings."""
    record = run_record(db_path, run_id)
    if record is None:
        raise KeyError(run_id)
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

    purple = colors.HexColor("#3b0764")
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitlePurple", parent=styles["Title"], textColor=purple, fontSize=16, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["Normal"], textColor=colors.HexColor("#1c1030"), fontSize=9, leading=12)
    heading = ParagraphStyle("HeadingPurple", parent=styles["Heading2"], textColor=purple, fontSize=12, spaceBefore=8)
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm)
    story = [
        Paragraph("QuantumRisk Analytics — simulation report", title),
        Paragraph(DISCLAIMER, body),
        Spacer(1, 4 * mm),
        Paragraph("Run", heading),
        Paragraph(
            f"Scenario: {record.get('scenario_name')} ({record.get('scenario_id')})<br/>"
            f"Estimator: {record.get('method_label')}<br/>"
            f"Seed: {record.get('seed')} · Samples: {record.get('n_samples')} · "
            f"Threshold: {record.get('threshold')} {record.get('currency')}<br/>"
            f"Runtime: {record.get('runtime_seconds')} s · Peak memory: {record.get('peak_memory_bytes')} bytes<br/>"
            f"Run id: {record.get('run_id')}",
            body,
        ),
        Paragraph("Risk metrics", heading),
    ]
    rows = [["Metric", "Estimate", "Standard error"]]
    for key, label in METRIC_LABELS.items():
        rows.append([label, _plain(record.get(key)), _plain(record.get(f"se_{key}"))])
    table = Table(rows, colWidths=[90 * mm, 40 * mm, 40 * mm])
    table.setStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), purple),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#1c1030")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.3, purple),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
    )
    story.append(table)
    story.append(Paragraph("Notes", heading))
    story.append(Paragraph(_escape(str(record.get("notes") or "")), body))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Synthetic demonstration. Not an official ADB payout schedule and not a quantum-advantage claim.", body))
    document.build(story)
    return buffer.getvalue()


def _plain(value) -> str:
    if value is None:
        return ""
    return str(value)


def _csv_cell(value) -> str:
    text = "" if value is None else str(value)
    if any(character in text for character in ",\"\n"):
        return '"' + text.replace('"', '""') + '"'
    return text


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
