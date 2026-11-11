"""Validate hazard, dependence, exposure, contract and scenario tables."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

COUNTRIES = ("Kyrgyz Republic", "Tajikistan")
HAZARDS = ("earthquake", "precipitation")
EQ_SEVERITIES = (1, 2, 3)
PRECIP_SEVERITIES = (1, 2)
DEPENDENCE_MODES = ("independent", "common_shock")
SCENARIO_IDS = ("base", "independent", "common_shock", "rain_stress", "rain_cover")
SCENARIO_NAMES = {
    "base": "Base",
    "independent": "Independent",
    "common_shock": "Common shock",
    "rain_stress": "Rain stress",
    "rain_cover": "Rain cover",
}
MIN_TERM_YEARS = 1
MAX_TERM_YEARS = 5
MAX_COUNTRIES = 6
MAX_HAZARDS = 4
MAX_ANNUAL_ATOMS = 300_000
PROBABILITY_SUM_TOLERANCE = 1e-8
PERIL_COLUMNS = (
    "country",
    "hazard",
    "severity",
    "payout_fraction",
    "principal",
    "cap",
    "aggregation",
    "country_cap",
)
PERIL_AGGREGATIONS = ("sum_capped", "max_once")

HAZARD_COLUMNS = (
    "country",
    "hazard",
    "annual_occurrence_probability",
    "severity_given_event",
    "conditional_probability",
)
DEPENDENCE_COLUMNS = (
    "dependence_mode",
    "hazard",
    "shock_share_of_min_marginal",
)
EXPOSURE_COLUMNS = ("country", "insured_exposure", "currency")
CONTRACT_COLUMNS = (
    "country",
    "term_years",
    "eq_principal",
    "precip_principal",
    "currency",
    "eq_severity_1_fraction",
    "eq_severity_2_fraction",
    "eq_severity_3_fraction",
    "precip_severity_1_fraction",
    "precip_severity_2_fraction",
    "eq_cap",
    "contract_cap",
)
SCENARIO_COLUMNS = (
    "scenario_id",
    "scenario_name",
    "dependence_mode",
    "shock_share",
    "precip_probability_scale",
    "eq_probability_scale",
    "description",
    "assumption_label",
)


@dataclass
class ValidationResult:
    """Collected data-quality errors and non-fatal warnings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_bundle(bundle) -> ValidationResult:
    """Validate a :class:`~src.data_loader.DataBundle`."""
    result = ValidationResult()
    _validate_hazards(bundle.hazards, result)
    _validate_dependence(bundle.dependence, result)
    _validate_exposure(bundle.exposure, result)
    _validate_contracts(bundle.contracts, result)
    _validate_scenarios(bundle.scenarios, result)
    peril_terms = getattr(bundle, "peril_terms", None)
    if peril_terms is not None and len(peril_terms):
        _validate_peril_terms(peril_terms, result)
    if result.errors:
        return result
    _validate_cross_tables(bundle, result)
    return result


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], table: str, result: ValidationResult) -> bool:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        result.errors.append(f"{table}: missing required columns: {', '.join(missing)}.")
        return False
    return True


def _require_values(frame: pd.DataFrame, columns: tuple[str, ...], table: str, result: ValidationResult) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        missing = frame.index[frame[column].isna()].tolist()
        if missing:
            shown = ", ".join(str(int(row) + 2) for row in missing[:8])
            result.errors.append(f"{table}: missing values in '{column}' at CSV data row(s) {shown}.")


def _label_warning(frame: pd.DataFrame, table: str, result: ValidationResult) -> None:
    if "data_label" not in frame.columns:
        result.warnings.append(
            f"{table}: no data_label column. Label synthetic inputs so they are not read as observed statistics."
        )
        return
    if frame["data_label"].isna().any() or (frame["data_label"].astype(str).str.strip() == "").any():
        result.errors.append(f"{table}: data_label is blank. Every row needs a source label.")


def _named_severities(hazard: str) -> tuple[int, ...] | None:
    if hazard == "earthquake":
        return EQ_SEVERITIES
    if hazard == "precipitation":
        return PRECIP_SEVERITIES
    return None


def _validate_hazards(frame: pd.DataFrame, result: ValidationResult) -> None:
    if not _require_columns(frame, HAZARD_COLUMNS, "hazards", result):
        return
    _require_values(frame, HAZARD_COLUMNS, "hazards", result)
    _label_warning(frame, "hazards", result)
    if result.errors:
        return

    seen: set[tuple[str, str, int]] = set()
    for row_number, row in frame.iterrows():
        csv_row = int(row_number) + 2
        hazard = str(row["hazard"]).strip().lower()
        if not hazard:
            result.errors.append(f"hazards row {csv_row}: hazard name is blank.")
        try:
            probability = float(row["annual_occurrence_probability"])
        except (TypeError, ValueError):
            result.errors.append(f"hazards row {csv_row}: annual_occurrence_probability is not numeric.")
            continue
        if not 0.0 <= probability <= 1.0:
            result.errors.append(
                f"hazards row {csv_row}: annual_occurrence_probability must be between 0 and 1 "
                f"(value {probability})."
            )
        try:
            severity = int(row["severity_given_event"])
            conditional = float(row["conditional_probability"])
        except (TypeError, ValueError):
            result.errors.append(f"hazards row {csv_row}: severity or conditional probability is not numeric.")
            continue
        if conditional < 0.0 or conditional > 1.0:
            result.errors.append(
                f"hazards row {csv_row}: conditional_probability must be between 0 and 1 (value {conditional})."
            )
        expected = _named_severities(hazard)
        if expected is not None and severity not in expected:
            result.errors.append(
                f"hazards row {csv_row}: {hazard} severity must be one of {expected} (value {severity})."
            )
        elif expected is None and severity < 1:
            result.errors.append(f"hazards row {csv_row}: severity must start at 1 (value {severity}).")
        key = (str(row["country"]), hazard, severity)
        if key in seen:
            result.errors.append(f"hazards row {csv_row}: duplicate country, hazard and severity {key}.")
        seen.add(key)

    grouped = frame.copy()
    grouped["hazard"] = grouped["hazard"].astype(str).str.strip().str.lower()
    for (country, hazard), group in grouped.groupby(["country", "hazard"], sort=False):
        occurrences = group["annual_occurrence_probability"].astype(float)
        if occurrences.nunique() != 1:
            result.errors.append(
                f"hazards: {country} {hazard} rows disagree on annual_occurrence_probability."
            )
        total = float(group["conditional_probability"].astype(float).sum())
        if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
            result.errors.append(
                f"hazards: {country} {hazard} conditional probabilities sum to {total:.8f}, not 1."
            )
        present = tuple(sorted(int(value) for value in group["severity_given_event"]))
        expected = _named_severities(str(hazard))
        if expected is None:
            expected = tuple(range(1, present[-1] + 1)) if present else ()
        if present != expected:
            result.errors.append(
                f"hazards: {country} {hazard} must list severities {expected}. Found {present}."
            )

    countries = list(dict.fromkeys(str(value) for value in frame["country"]))
    hazards = list(dict.fromkeys(str(value).strip().lower() for value in grouped["hazard"]))
    if not 1 <= len(countries) <= MAX_COUNTRIES:
        result.errors.append(f"hazards: use between 1 and {MAX_COUNTRIES} countries. Found {len(countries)}.")
    if not 1 <= len(hazards) <= MAX_HAZARDS:
        result.errors.append(f"hazards: use between 1 and {MAX_HAZARDS} hazards. Found {len(hazards)}.")
    covered = {(str(country), str(hazard)) for country, hazard in grouped.groupby(["country", "hazard"], sort=False).groups}
    for country in countries:
        for hazard in hazards:
            if (country, hazard) not in covered:
                result.errors.append(f"hazards: {country} is missing {hazard}. Every country needs every hazard.")


def _validate_dependence(frame: pd.DataFrame, result: ValidationResult) -> None:
    if not _require_columns(frame, DEPENDENCE_COLUMNS, "dependence", result):
        return
    _require_values(frame, DEPENDENCE_COLUMNS, "dependence", result)
    _label_warning(frame, "dependence", result)
    if result.errors:
        return
    modes = set()
    hazards = []
    for row_number, row in frame.iterrows():
        csv_row = int(row_number) + 2
        mode = str(row["dependence_mode"]).strip().lower()
        hazard = str(row["hazard"]).strip().lower()
        modes.add(mode)
        hazards.append(hazard)
        if mode not in DEPENDENCE_MODES:
            result.errors.append(
                f"dependence row {csv_row}: dependence_mode must be independent or common_shock."
            )
        if not hazard:
            result.errors.append(f"dependence row {csv_row}: hazard name is blank.")
        try:
            share = float(row["shock_share_of_min_marginal"])
        except (TypeError, ValueError):
            result.errors.append(f"dependence row {csv_row}: shock_share_of_min_marginal is not numeric.")
            continue
        if not 0.0 <= share <= 1.0:
            result.errors.append(
                f"dependence row {csv_row}: shock_share_of_min_marginal must be between 0 and 1 (value {share})."
            )
    if len(hazards) != len(set(hazards)):
        result.errors.append("dependence: provide one row per hazard.")
    if len(modes) > 1:
        result.warnings.append(
            "dependence: rows use different dependence modes. The selected scenario sets the mode used for a run."
        )


def _validate_exposure(frame: pd.DataFrame, result: ValidationResult) -> None:
    if not _require_columns(frame, EXPOSURE_COLUMNS, "exposure", result):
        return
    _require_values(frame, EXPOSURE_COLUMNS, "exposure", result)
    _label_warning(frame, "exposure", result)
    if result.errors:
        return
    countries = [str(value) for value in frame["country"]]
    if len(countries) != len(set(countries)):
        result.errors.append("exposure: each country may appear only once.")
    if not 1 <= len(set(countries)) <= MAX_COUNTRIES:
        result.errors.append(f"exposure: use between 1 and {MAX_COUNTRIES} countries.")
    currencies = {str(value).strip().upper() for value in frame["currency"]}
    if len(currencies) != 1:
        result.errors.append("exposure: every country must use the same currency.")
    for row_number, row in frame.iterrows():
        csv_row = int(row_number) + 2
        try:
            exposure = float(row["insured_exposure"])
        except (TypeError, ValueError):
            result.errors.append(f"exposure row {csv_row}: insured_exposure is not numeric.")
            continue
        if exposure <= 0:
            result.errors.append(f"exposure row {csv_row}: insured_exposure must be positive.")


def _validate_contracts(frame: pd.DataFrame, result: ValidationResult) -> None:
    if not _require_columns(frame, CONTRACT_COLUMNS, "contracts", result):
        return
    _require_values(frame, CONTRACT_COLUMNS, "contracts", result)
    _label_warning(frame, "contracts", result)
    if result.errors:
        return
    countries = [str(value) for value in frame["country"]]
    if len(countries) != len(set(countries)):
        result.errors.append("contracts: each country may appear only once.")
    if not 1 <= len(set(countries)) <= MAX_COUNTRIES:
        result.errors.append(f"contracts: use between 1 and {MAX_COUNTRIES} countries.")
    currencies = {str(value).strip().upper() for value in frame["currency"]}
    if len(currencies) != 1:
        result.errors.append("contracts: every country must use the same currency.")

    terms = []
    for row_number, row in frame.iterrows():
        csv_row = int(row_number) + 2
        try:
            term = int(row["term_years"])
            eq_principal = float(row["eq_principal"])
            precip_principal = float(row["precip_principal"])
            eq_cap = float(row["eq_cap"])
            contract_cap = float(row["contract_cap"])
            fractions = [
                float(row["eq_severity_1_fraction"]),
                float(row["eq_severity_2_fraction"]),
                float(row["eq_severity_3_fraction"]),
                float(row["precip_severity_1_fraction"]),
                float(row["precip_severity_2_fraction"]),
            ]
        except (TypeError, ValueError):
            result.errors.append(f"contracts row {csv_row}: a numeric contract field is not numeric.")
            continue
        terms.append(term)
        if term < MIN_TERM_YEARS or term > MAX_TERM_YEARS:
            result.errors.append(
                f"contracts row {csv_row}: term_years must be between {MIN_TERM_YEARS} and {MAX_TERM_YEARS}."
            )
        if eq_principal <= 0 or precip_principal <= 0:
            result.errors.append(f"contracts row {csv_row}: principals must be positive.")
        if not 0 < eq_cap <= eq_principal + 1e-6:
            result.errors.append(
                f"contracts row {csv_row}: eq_cap must be positive and no larger than eq_principal."
            )
        if not 0 < contract_cap <= eq_principal + precip_principal + 1e-6:
            result.errors.append(
                f"contracts row {csv_row}: contract_cap must be positive and no larger than the sum of the principals."
            )
        if contract_cap + 1e-6 < eq_cap:
            result.errors.append(f"contracts row {csv_row}: contract_cap is below eq_cap.")
        eq_fractions = fractions[:3]
        precip_fractions = fractions[3:]
        if any(fraction < 0.0 or fraction > 1.0 for fraction in fractions):
            result.errors.append(f"contracts row {csv_row}: payout fractions must be between 0 and 1.")
        if eq_fractions != sorted(eq_fractions):
            result.errors.append(f"contracts row {csv_row}: earthquake payout fractions must be non-decreasing.")
        if precip_fractions != sorted(precip_fractions):
            result.errors.append(f"contracts row {csv_row}: precipitation payout fractions must be non-decreasing.")
        for severity, fraction in enumerate(eq_fractions, start=1):
            step = int(round(fraction * eq_principal))
            if step > eq_cap + 1e-6:
                result.errors.append(
                    f"contracts row {csv_row}: earthquake severity {severity} pays {step}, above eq_cap {eq_cap:.0f}."
                )
        for severity, fraction in enumerate(precip_fractions, start=1):
            step = int(round(fraction * precip_principal))
            if step > precip_principal + 1e-6:
                result.errors.append(
                    f"contracts row {csv_row}: precipitation severity {severity} pays {step}, above the precip principal."
                )
        if contract_cap + 1e-6 < eq_cap + max(int(round(fraction * precip_principal)) for fraction in precip_fractions):
            result.warnings.append(
                f"contracts row {csv_row}: contract_cap can bind before both peril limits are reached."
            )
    if len(set(terms)) > 1:
        result.errors.append("contracts: every country must use the same term_years.")


def _validate_scenarios(frame: pd.DataFrame, result: ValidationResult) -> None:
    if not _require_columns(frame, SCENARIO_COLUMNS, "scenarios", result):
        return
    _require_values(frame, SCENARIO_COLUMNS, "scenarios", result)
    if result.errors:
        return
    identifiers = [str(value).strip().lower() for value in frame["scenario_id"]]
    missing = [scenario_id for scenario_id in SCENARIO_IDS if scenario_id not in identifiers]
    if missing or len(identifiers) != len(set(identifiers)):
        result.errors.append(
            "scenarios: include these scenario_id values once, and add further ids only if they are new: "
            + ", ".join(SCENARIO_IDS)
            + f". Found: {identifiers}."
        )
    names = {str(row["scenario_id"]).strip().lower(): str(row["scenario_name"]).strip() for _, row in frame.iterrows()}
    for scenario_id, expected_name in SCENARIO_NAMES.items():
        if scenario_id in names and names[scenario_id] != expected_name:
            result.errors.append(
                f"scenarios: scenario_id '{scenario_id}' must be named '{expected_name}'."
            )
    for row_number, row in frame.iterrows():
        csv_row = int(row_number) + 2
        mode = str(row["dependence_mode"]).strip().lower()
        if mode not in DEPENDENCE_MODES:
            result.errors.append(f"scenarios row {csv_row}: dependence_mode must be independent or common_shock.")
        try:
            share = float(row["shock_share"])
            precip_scale = float(row["precip_probability_scale"])
            eq_scale = float(row["eq_probability_scale"])
        except (TypeError, ValueError):
            result.errors.append(f"scenarios row {csv_row}: shock share or probability scale is not numeric.")
            continue
        if not 0.0 <= share <= 1.0:
            result.errors.append(f"scenarios row {csv_row}: shock_share must be between 0 and 1.")
        if not 0.0 < precip_scale <= 5.0 or not 0.0 < eq_scale <= 5.0:
            result.errors.append(f"scenarios row {csv_row}: probability scales must be in the interval (0, 5].")
        if mode == "independent" and share != 0.0:
            result.errors.append(f"scenarios row {csv_row}: independent mode requires shock_share 0.")
        for column in ("precip_severity_1_fraction", "precip_severity_2_fraction"):
            if column not in frame.columns or pd.isna(row[column]):
                continue
            try:
                fraction = float(row[column])
            except (TypeError, ValueError):
                result.errors.append(f"scenarios row {csv_row}: {column} is not numeric.")
                continue
            if not 0.0 <= fraction <= 1.0:
                result.errors.append(f"scenarios row {csv_row}: {column} must be between 0 and 1.")
        if (
            "precip_severity_1_fraction" in frame.columns
            and "precip_severity_2_fraction" in frame.columns
            and pd.notna(row["precip_severity_1_fraction"])
            and pd.notna(row["precip_severity_2_fraction"])
        ):
            if float(row["precip_severity_1_fraction"]) > float(row["precip_severity_2_fraction"]):
                result.errors.append(
                    f"scenarios row {csv_row}: precipitation cover fractions must be non-decreasing."
                )
        if not str(row["assumption_label"]).strip():
            result.errors.append(f"scenarios row {csv_row}: assumption_label is blank.")
        if not str(row["description"]).strip():
            result.errors.append(f"scenarios row {csv_row}: description is blank.")


def _validate_peril_terms(frame: pd.DataFrame, result: ValidationResult) -> None:
    if not _require_columns(frame, PERIL_COLUMNS, "peril_terms", result):
        return
    _require_values(frame, PERIL_COLUMNS, "peril_terms", result)
    if result.errors:
        return
    seen: set[tuple[str, str, int]] = set()
    for row_number, row in frame.iterrows():
        csv_row = int(row_number) + 2
        hazard = str(row["hazard"]).strip().lower()
        if hazard in HAZARDS:
            result.errors.append(
                f"peril_terms row {csv_row}: {hazard} is already defined by the contracts table."
            )
        aggregation = str(row["aggregation"]).strip().lower()
        if aggregation not in PERIL_AGGREGATIONS:
            result.errors.append(
                f"peril_terms row {csv_row}: aggregation must be sum_capped or max_once."
            )
        try:
            severity = int(row["severity"])
            fraction = float(row["payout_fraction"])
            principal = float(row["principal"])
            cap = float(row["cap"])
            country_cap = float(row["country_cap"])
        except (TypeError, ValueError):
            result.errors.append(f"peril_terms row {csv_row}: severity, fraction, principal or cap is not numeric.")
            continue
        if severity < 1:
            result.errors.append(f"peril_terms row {csv_row}: severity must start at 1.")
        if not 0.0 <= fraction <= 1.0:
            result.errors.append(f"peril_terms row {csv_row}: payout_fraction must be between 0 and 1.")
        if principal <= 0 or cap <= 0 or country_cap <= 0:
            result.errors.append(f"peril_terms row {csv_row}: principal, cap and country_cap must be positive.")
        if cap - principal > 1.0:
            result.errors.append(f"peril_terms row {csv_row}: cap cannot exceed principal.")
        key = (str(row["country"]), hazard, severity)
        if key in seen:
            result.errors.append(f"peril_terms row {csv_row}: duplicate country, hazard and severity {key}.")
        seen.add(key)


def _validate_cross_tables(bundle, result: ValidationResult) -> None:
    hazard_countries = list(dict.fromkeys(str(value) for value in bundle.hazards["country"]))
    exposure_countries = [str(value) for value in bundle.exposure["country"]]
    contract_countries = [str(value) for value in bundle.contracts["country"]]
    if set(hazard_countries) != set(exposure_countries) or set(hazard_countries) != set(contract_countries):
        result.errors.append(
            "Countries must match across hazards, exposure and contracts. "
            f"Hazards include {sorted(set(hazard_countries))}. "
            f"Exposure includes {sorted(set(exposure_countries))}. "
            f"Contracts include {sorted(set(contract_countries))}."
        )
        return

    exposure = {str(row["country"]): float(row["insured_exposure"]) for _, row in bundle.exposure.iterrows()}
    exposure_currency = {str(row["country"]): str(row["currency"]).strip().upper() for _, row in bundle.exposure.iterrows()}
    extra_principal = _extra_principals(bundle)
    for _, row in bundle.contracts.iterrows():
        country = str(row["country"])
        principal_sum = float(row["eq_principal"]) + float(row["precip_principal"]) + extra_principal.get(country, 0.0)
        if abs(exposure[country] - principal_sum) > 1.0:
            result.errors.append(
                f"{country}: insured exposure {exposure[country]:.0f} does not match "
                f"the sum of peril principals ({principal_sum:.0f})."
            )
        if exposure_currency[country] != str(row["currency"]).strip().upper():
            result.errors.append(f"{country}: exposure currency and contract currency differ.")

    hazard_names = list(dict.fromkeys(str(value).strip().lower() for value in bundle.hazards["hazard"]))
    dependence = {}
    for _, row in bundle.dependence.iterrows():
        dependence[str(row["hazard"]).strip().lower()] = row
    if set(dependence) != set(hazard_names):
        result.errors.append(
            "dependence hazards must match the hazard table. "
            f"Hazards: {sorted(hazard_names)}. Dependence: {sorted(dependence)}."
        )
    _require_extra_hazard_cover(bundle, hazard_names, hazard_countries, result)
    _check_state_space(bundle, hazard_names, hazard_countries, result)
    base_rows = bundle.scenarios[bundle.scenarios["scenario_id"].astype(str).str.strip().str.lower() == "base"]
    if base_rows.empty:
        return
    base = base_rows.iloc[0]
    base_mode = str(base["dependence_mode"]).strip().lower()
    file_modes = {str(row["dependence_mode"]).strip().lower() for _, row in bundle.dependence.iterrows()}
    if file_modes != {base_mode}:
        result.errors.append(
            "base scenario dependence_mode must match dependence.csv so the baseline has one definition."
        )
    for hazard, row in dependence.items():
        file_share = float(row["shock_share_of_min_marginal"])
        if abs(file_share - float(base["shock_share"])) > 1e-12:
            result.errors.append(
                f"base scenario shock_share must match dependence.csv for {hazard} "
                f"(file {file_share}, scenario {float(base['shock_share'])})."
            )


def _extra_principals(bundle) -> dict[str, float]:
    frame = getattr(bundle, "peril_terms", None)
    totals: dict[str, float] = {}
    if frame is None or not len(frame):
        return totals
    seen: set[tuple[str, str]] = set()
    for _, row in frame.iterrows():
        hazard = str(row["hazard"]).strip().lower()
        if hazard in HAZARDS:
            continue
        key = (str(row["country"]), hazard)
        if key in seen:
            continue
        seen.add(key)
        totals[key[0]] = totals.get(key[0], 0.0) + float(row["principal"])
    return totals


def _require_extra_hazard_cover(bundle, hazard_names: list[str], countries: list[str], result: ValidationResult) -> None:
    extra = [hazard for hazard in hazard_names if hazard not in HAZARDS]
    if not extra:
        return
    frame = getattr(bundle, "peril_terms", None)
    covered: set[tuple[str, str, int]] = set()
    if frame is not None and len(frame):
        for _, row in frame.iterrows():
            covered.add((str(row["country"]), str(row["hazard"]).strip().lower(), int(row["severity"])))
    grouped = bundle.hazards.copy()
    grouped["hazard"] = grouped["hazard"].astype(str).str.strip().str.lower()
    for hazard in extra:
        for country in countries:
            rows = grouped[(grouped["country"] == country) & (grouped["hazard"] == hazard)]
            for severity in rows["severity_given_event"]:
                if (country, hazard, int(severity)) not in covered:
                    result.errors.append(
                        f"peril_terms: {country} {hazard} severity {int(severity)} needs a payout fraction."
                    )


def _check_state_space(bundle, hazard_names: list[str], countries: list[str], result: ValidationResult) -> None:
    grouped = bundle.hazards.copy()
    grouped["hazard"] = grouped["hazard"].astype(str).str.strip().str.lower()
    size = 1
    for hazard in hazard_names:
        levels = grouped.loc[grouped["hazard"] == hazard, "severity_given_event"]
        width = int(levels.max()) + 1 if len(levels) else 1
        size *= width ** len(countries)
    if size > MAX_ANNUAL_ATOMS:
        result.errors.append(
            f"The annual probability table would have {size} entries, above the local limit of {MAX_ANNUAL_ATOMS}. "
            "Use fewer countries, hazards or severity steps."
        )
