"""Load synthetic fixtures and user-supplied CSV, Excel and JSON tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
REFERENCE_DIR = ROOT / "data" / "reference"

TABLE_FILES = {
    "hazards": "hazards.csv",
    "dependence": "dependence.csv",
    "exposure": "exposure.csv",
    "contracts": "contracts.csv",
    "scenarios": "scenarios.csv",
}
OPTIONAL_TABLES = {"peril_terms": "peril_terms.csv"}
TABLE_NAMES = tuple(TABLE_FILES) + ("peril_terms",)


@dataclass
class DataBundle:
    """Input tables that define a demo run."""

    hazards: pd.DataFrame
    dependence: pd.DataFrame
    exposure: pd.DataFrame
    contracts: pd.DataFrame
    scenarios: pd.DataFrame
    source: str = "synthetic"
    peril_terms: pd.DataFrame | None = None


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(column).strip() for column in out.columns]
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].apply(lambda value: value.strip() if isinstance(value, str) else value)
    return out


def read_csv_bytes(payload: bytes | str) -> pd.DataFrame:
    """Read a CSV from text or bytes and normalise column names."""
    if isinstance(payload, bytes):
        frame = pd.read_csv(BytesIO(payload))
    else:
        frame = pd.read_csv(StringIO(payload))
    return _normalise(frame)


def read_csv_path(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return _normalise(frame)


def load_synthetic(directory: Path | None = None) -> DataBundle:
    """Load the labelled synthetic fixtures shipped with the demo."""
    folder = directory or SYNTHETIC_DIR
    tables = {name: read_csv_path(folder / filename) for name, filename in TABLE_FILES.items()}
    peril_path = folder / OPTIONAL_TABLES["peril_terms"]
    peril_terms = read_csv_path(peril_path) if peril_path.exists() else None
    return DataBundle(source=str(folder), peril_terms=peril_terms, **tables)


def replace_tables(bundle: DataBundle, replacements: dict[str, pd.DataFrame]) -> DataBundle:
    """Return a copy of ``bundle`` with the named tables replaced."""
    unknown = set(replacements) - set(TABLE_NAMES)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown input tables: {names}.")
    data = {
        "hazards": bundle.hazards,
        "dependence": bundle.dependence,
        "exposure": bundle.exposure,
        "contracts": bundle.contracts,
        "scenarios": bundle.scenarios,
        "peril_terms": bundle.peril_terms,
    }
    data.update(replacements)
    source = bundle.source
    if replacements:
        source = f"{bundle.source} + upload ({', '.join(sorted(replacements))})"
    return DataBundle(source=source, **data)


def read_table_file(filename: str, payload: bytes) -> pd.DataFrame:
    """Read one uploaded CSV, Excel or JSON table."""
    name = filename.lower()
    if name.endswith(".csv"):
        return read_csv_bytes(payload)
    if name.endswith(".xlsx"):
        return _normalise(pd.read_excel(BytesIO(payload), engine="openpyxl"))
    if name.endswith(".json"):
        return _frame_from_json(json.loads(payload.decode("utf-8-sig")))
    raise ValueError(f"{filename}: upload a .csv, .xlsx or .json file.")


def read_workbook(payload: bytes) -> DataBundle:
    """Read a workbook whose sheet names are the input tables."""
    book = pd.ExcelFile(BytesIO(payload), engine="openpyxl")
    sheets = {str(sheet).strip().lower(): sheet for sheet in book.sheet_names}
    missing = [name for name in TABLE_FILES if name not in sheets]
    if missing:
        raise ValueError("Workbook is missing sheets: " + ", ".join(missing) + ".")
    tables = {name: _normalise(pd.read_excel(book, sheets[name])) for name in TABLE_FILES}
    peril = _normalise(pd.read_excel(book, sheets["peril_terms"])) if "peril_terms" in sheets else None
    return DataBundle(source="excel workbook", peril_terms=peril, **tables)


def read_json_bundle(payload: bytes | str) -> DataBundle:
    """Read a JSON object with one array of records per input table."""
    document = json.loads(payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload)
    if not isinstance(document, dict):
        raise ValueError("A JSON bundle must be an object with table names as keys.")
    missing = [name for name in TABLE_FILES if name not in document]
    if missing:
        raise ValueError("JSON bundle is missing tables: " + ", ".join(missing) + ".")
    tables = {name: _frame_from_json(document[name]) for name in TABLE_FILES}
    peril = _frame_from_json(document["peril_terms"]) if document.get("peril_terms") else None
    return DataBundle(source="json bundle", peril_terms=peril, **tables)


def snapshot_bundle(bundle: DataBundle) -> dict:
    """JSON-ready copy of the tables, used to rerun a job later."""

    def _records(frame: pd.DataFrame | None):
        if frame is None:
            return None
        clean = frame.astype(object).where(pd.notna(frame), None)
        return clean.to_dict(orient="records")

    return {
        "source": bundle.source,
        "hazards": _records(bundle.hazards),
        "dependence": _records(bundle.dependence),
        "exposure": _records(bundle.exposure),
        "contracts": _records(bundle.contracts),
        "scenarios": _records(bundle.scenarios),
        "peril_terms": _records(bundle.peril_terms),
    }


def bundle_from_snapshot(payload: dict) -> DataBundle:
    """Rebuild a bundle from :func:`snapshot_bundle`."""
    tables = {name: _frame_from_json(payload[name]) for name in TABLE_FILES}
    peril = _frame_from_json(payload["peril_terms"]) if payload.get("peril_terms") else None
    return DataBundle(source=str(payload.get("source", "snapshot")), peril_terms=peril, **tables)


def _frame_from_json(value) -> pd.DataFrame:
    if isinstance(value, dict) and "records" in value:
        value = value["records"]
    if not isinstance(value, list):
        raise ValueError("Each JSON table must be a list of row objects.")
    return _normalise(pd.DataFrame(value))
