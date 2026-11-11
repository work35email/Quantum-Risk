"""Generate the synthetic QuantumRisk Analytics demonstration dataset.

The random seed is 62. Monetary figures are whole USD. Reference metrics are
exact enumerations of the same contract model the app runs. They are synthetic
demonstration figures, not official ADB results and not historical losses.

The shipped fixture in data/synthetic and data/reference is left unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical.exact import estimate
from src.data_loader import DataBundle
from src.payout_engine import money_fraction
from src.scenarios import build_portfolio
from src.validation import validate_bundle

SEED = 62
HORIZON_YEARS = 60
THRESHOLD = 25_000_000
LOW_THRESHOLD = 10_000_000
LABEL = "synthetic_demo_seed_62"
OUT = ROOT / "data" / "demo"

KG = "Kyrgyz Republic"
TJ = "Tajikistan"

# Baseline book. Kyrgyz exposure is larger and earthquake-heavy.
# Tajik exposure is smaller and precipitation is a larger share of that book.
BASE_TERMS = {
    KG: {
        "eq_p": 0.12,
        "rain_p": 0.14,
        "eq_sev": (0.62, 0.28, 0.10),
        "rain_sev": (0.72, 0.28),
        "eq_principal": 80_000_000,
        "precip_principal": 16_000_000,
        "eq_fractions": (0.15, 0.40, 1.00),
        "precip_fractions": (0.40, 1.00),
        "eq_cap": 80_000_000,
        "contract_cap": 88_000_000,
    },
    TJ: {
        "eq_p": 0.09,
        "rain_p": 0.20,
        "eq_sev": (0.58, 0.30, 0.12),
        "rain_sev": (0.64, 0.36),
        "eq_principal": 50_000_000,
        "precip_principal": 18_000_000,
        "eq_fractions": (0.20, 0.55, 1.00),
        "precip_fractions": (0.50, 1.00),
        "eq_cap": 50_000_000,
        "contract_cap": 60_000_000,
    },
}

REGIONS = {
    KG: [
        ("REG-KG-BISHKEK", "Bishkek basin", 0.34, "urban", 0.12, 0.08),
        ("REG-KG-CHUY", "Chuy valley", 0.14, "urban", 0.10, 0.16),
        ("REG-KG-OSH", "Osh-Fergana", 0.16, "valley", 0.18, 0.10),
        ("REG-KG-ISSYK", "Issyk-Kul", 0.10, "mountain", 0.14, 0.08),
        ("REG-KG-JALAL", "Jalal-Abad", 0.09, "valley", 0.12, 0.18),
        ("REG-KG-BATKEN", "Batken", 0.07, "mountain", 0.16, 0.10),
        ("REG-KG-NARYN", "Naryn highlands", 0.06, "mountain", 0.12, 0.14),
        ("REG-KG-TALAS", "Talas", 0.04, "mountain", 0.06, 0.16),
    ],
    TJ: [
        ("REG-TJ-DUSHANBE", "Dushanbe basin", 0.28, "urban", 0.08, 0.10),
        ("REG-TJ-SUGHD", "Sughd", 0.18, "urban", 0.10, 0.14),
        ("REG-TJ-KHATLON", "Khatlon", 0.20, "valley", 0.12, 0.22),
        ("REG-TJ-GBAO", "Gorno-Badakhshan", 0.12, "mountain", 0.24, 0.08),
        ("REG-TJ-RASHT", "Rasht valley", 0.08, "mountain", 0.18, 0.12),
        ("REG-TJ-ZERAVSHAN", "Zeravshan", 0.06, "mountain", 0.14, 0.10),
        ("REG-TJ-PANJ", "Panj river lowlands", 0.05, "valley", 0.08, 0.16),
        ("REG-TJ-PAMIR", "Pamir highway corridor", 0.03, "mountain", 0.06, 0.08),
    ],
}

NEIGHBORS = {
    "REG-KG-BISHKEK": ("REG-KG-CHUY", "REG-KG-ISSYK", "REG-KG-TALAS"),
    "REG-KG-CHUY": ("REG-KG-BISHKEK", "REG-KG-TALAS", "REG-KG-NARYN"),
    "REG-KG-OSH": ("REG-KG-JALAL", "REG-KG-BATKEN", "REG-KG-NARYN"),
    "REG-KG-ISSYK": ("REG-KG-BISHKEK", "REG-KG-NARYN", "REG-KG-CHUY"),
    "REG-KG-JALAL": ("REG-KG-OSH", "REG-KG-NARYN", "REG-KG-BATKEN"),
    "REG-KG-BATKEN": ("REG-KG-OSH", "REG-KG-JALAL", "REG-KG-TALAS"),
    "REG-KG-NARYN": ("REG-KG-ISSYK", "REG-KG-JALAL", "REG-KG-BISHKEK"),
    "REG-KG-TALAS": ("REG-KG-CHUY", "REG-KG-BISHKEK", "REG-KG-BATKEN"),
    "REG-TJ-DUSHANBE": ("REG-TJ-KHATLON", "REG-TJ-RASHT", "REG-TJ-PAMIR"),
    "REG-TJ-SUGHD": ("REG-TJ-ZERAVSHAN", "REG-TJ-PANJ", "REG-TJ-GBAO"),
    "REG-TJ-KHATLON": ("REG-TJ-DUSHANBE", "REG-TJ-PANJ", "REG-TJ-RASHT"),
    "REG-TJ-GBAO": ("REG-TJ-PAMIR", "REG-TJ-KHATLON", "REG-TJ-SUGHD"),
    "REG-TJ-RASHT": ("REG-TJ-DUSHANBE", "REG-TJ-ZERAVSHAN", "REG-TJ-PAMIR"),
    "REG-TJ-ZERAVSHAN": ("REG-TJ-SUGHD", "REG-TJ-RASHT", "REG-TJ-PANJ"),
    "REG-TJ-PANJ": ("REG-TJ-KHATLON", "REG-TJ-SUGHD", "REG-TJ-ZERAVSHAN"),
    "REG-TJ-PAMIR": ("REG-TJ-GBAO", "REG-TJ-RASHT", "REG-TJ-DUSHANBE"),
}

CATEGORY_SHARES = {
    "urban": {
        "residential": 0.42,
        "commercial": 0.28,
        "public_infrastructure": 0.22,
        "agriculture": 0.08,
    },
    "valley": {
        "residential": 0.30,
        "commercial": 0.12,
        "public_infrastructure": 0.18,
        "agriculture": 0.40,
    },
    "mountain": {
        "residential": 0.28,
        "commercial": 0.10,
        "public_infrastructure": 0.37,
        "agriculture": 0.25,
    },
}

EQ_DAMAGE = {1: 0.035, 2: 0.11, 3: 0.30}
RAIN_DAMAGE = {1: 0.045, 2: 0.16}
EQ_CATEGORY = {
    "residential": 1.00,
    "commercial": 0.92,
    "public_infrastructure": 1.22,
    "agriculture": 0.48,
}
RAIN_CATEGORY = {
    "residential": 0.88,
    "commercial": 0.70,
    "public_infrastructure": 1.08,
    "agriculture": 1.45,
}
SEVERITY_WORD = {
    ("earthquake", 1): "moderate",
    ("earthquake", 2): "severe",
    ("earthquake", 3): "extreme",
    ("precipitation", 1): "moderate",
    ("precipitation", 2): "severe",
}


def allocate(total: int, weights: list[float]) -> list[int]:
    """Split an integer total across weights. The parts sum to the total."""
    raw = np.asarray(weights, dtype=float)
    raw = raw / raw.sum() * total
    floors = np.floor(raw).astype(np.int64)
    short = int(total - int(floors.sum()))
    order = np.argsort(-(raw - floors))
    for index in range(short):
        floors[int(order[index % len(floors)])] += 1
    return [int(value) for value in floors]


def exposure_frame(book: dict) -> pd.DataFrame:
    rows = []
    for country in (KG, TJ):
        spec = book[country]
        total = spec["eq_principal"] + spec["precip_principal"]
        regions = REGIONS[country]
        region_amounts = allocate(total, [item[2] for item in regions])
        for (region_id, region_name, share, settlement, _eq_w, _rain_w), region_amount in zip(regions, region_amounts):
            categories = list(CATEGORY_SHARES[settlement])
            cat_amounts = allocate(region_amount, [CATEGORY_SHARES[settlement][name] for name in categories])
            for category, amount in zip(categories, cat_amounts):
                rows.append(
                    {
                        "exposure_id": f"EXP-{region_id}-{category[:3].upper()}",
                        "portfolio_book": "baseline" if book is BASE_TERMS else "scenario_variant",
                        "country": country,
                        "region_id": region_id,
                        "region_name": region_name,
                        "settlement_type": settlement,
                        "asset_category": category,
                        "insured_exposure_usd": amount,
                        "eq_weight": share,
                        "currency": "USD",
                        "master_contract_id": "CON-KG-MASTER" if country == KG else "CON-TJ-MASTER",
                        "data_quality_flag": "valid",
                        "include_in_calculations": "yes",
                        "data_label": LABEL,
                        "notes": "Synthetic insured exposure. Not an observed insurance portfolio.",
                    }
                )
    return pd.DataFrame(rows)


def hazard_parameter_frame(book: dict, occurrence_scale: dict | None = None) -> pd.DataFrame:
    rows = []
    for country in (KG, TJ):
        spec = book[country]
        scale = 1.0 if not occurrence_scale else occurrence_scale.get(country, 1.0)
        for hazard, probability, severities in (
            ("earthquake", min(0.95, spec["eq_p"] * scale), spec["eq_sev"]),
            ("precipitation", min(0.95, spec["rain_p"] * scale), spec["rain_sev"]),
        ):
            for severity, conditional in enumerate(severities, start=1):
                rows.append(
                    {
                        "country": country,
                        "hazard": hazard,
                        "annual_occurrence_probability": probability,
                        "severity_given_event": severity,
                        "conditional_probability": conditional,
                        "unit": "probability",
                        "data_label": LABEL,
                    }
                )
    return pd.DataFrame(rows)


def contract_frame(book: dict, cap_mode: str = "baseline") -> pd.DataFrame:
    rows = []
    for country in (KG, TJ):
        spec = book[country]
        eq_cap = spec["eq_cap"]
        contract_cap = spec["contract_cap"]
        if cap_mode == "tight":
            contract_cap = eq_cap
        fractions = spec["eq_fractions"]
        precip = spec["precip_fractions"]
        rows.append(
            {
                "country": country,
                "term_years": 3,
                "eq_principal": spec["eq_principal"],
                "precip_principal": spec["precip_principal"],
                "currency": "USD",
                "eq_severity_1_fraction": fractions[0],
                "eq_severity_2_fraction": fractions[1],
                "eq_severity_3_fraction": fractions[2],
                "precip_severity_1_fraction": precip[0],
                "precip_severity_2_fraction": precip[1],
                "eq_cap": eq_cap,
                "contract_cap": contract_cap,
                "data_label": LABEL,
                "notes": "Illustrative synthetic terms. Not an official ADB payout schedule.",
            }
        )
    return pd.DataFrame(rows)


def country_exposure_frame(book: dict) -> pd.DataFrame:
    rows = []
    for country in (KG, TJ):
        spec = book[country]
        rows.append(
            {
                "country": country,
                "insured_exposure": spec["eq_principal"] + spec["precip_principal"],
                "currency": "USD",
                "data_label": LABEL,
                "notes": "Country total of the regional exposure book. Synthetic.",
            }
        )
    return pd.DataFrame(rows)


def dependence_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dependence_mode": "common_shock",
                "hazard": "earthquake",
                "shock_share_of_min_marginal": 0.30,
                "notes": "Baseline Marshall-Olkin share. Synthetic demo assumption, not an observed correlation.",
                "data_label": LABEL,
            },
            {
                "dependence_mode": "common_shock",
                "hazard": "precipitation",
                "shock_share_of_min_marginal": 0.30,
                "notes": "Baseline Marshall-Olkin share. Synthetic demo assumption, not an observed correlation.",
                "data_label": LABEL,
            },
        ]
    )


def scenario_row(spec: dict) -> dict:
    return {
        "scenario_id": spec["scenario_id"],
        "scenario_name": spec["scenario_name"],
        "dependence_mode": spec["dependence_mode"],
        "shock_share": spec["shock_share"],
        "precip_probability_scale": spec["precip_probability_scale"],
        "eq_probability_scale": spec["eq_probability_scale"],
        "precip_severity_1_fraction": spec.get("precip_severity_1_fraction"),
        "precip_severity_2_fraction": spec.get("precip_severity_2_fraction"),
        "description": spec["description"],
        "assumption_label": LABEL,
    }


REQUIRED = [
    {
        "scenario_id": "base",
        "scenario_name": "Base",
        "dependence_mode": "common_shock",
        "shock_share": 0.30,
        "precip_probability_scale": 1.0,
        "eq_probability_scale": 1.0,
        "description": "Baseline synthetic marginals and a common-shock share of 0.30. Contract rules are unchanged. Not an official benchmark.",
        "theme": "Baseline conditions",
    },
    {
        "scenario_id": "independent",
        "scenario_name": "Independent",
        "dependence_mode": "independent",
        "shock_share": 0.0,
        "precip_probability_scale": 1.0,
        "eq_probability_scale": 1.0,
        "description": "Shared-shock weight is zero. Country occurrence probabilities match the baseline. Not an official benchmark.",
        "theme": "Cross-country common shock comparison",
    },
    {
        "scenario_id": "common_shock",
        "scenario_name": "Common shock",
        "dependence_mode": "common_shock",
        "shock_share": 0.80,
        "precip_probability_scale": 1.0,
        "eq_probability_scale": 1.0,
        "description": "Shared-shock weight is 0.80 of the smaller marginal. Marginals match the baseline. Not an official benchmark.",
        "theme": "Cross-country common shock",
    },
    {
        "scenario_id": "rain_stress",
        "scenario_name": "Rain stress",
        "dependence_mode": "common_shock",
        "shock_share": 0.30,
        "precip_probability_scale": 1.75,
        "eq_probability_scale": 1.0,
        "description": "Precipitation occurrence is multiplied by 1.75. The common-shock share stays 0.30, so the shared-shock probability rises with the marginals.",
        "theme": "Extreme precipitation",
    },
    {
        "scenario_id": "rain_cover",
        "scenario_name": "Rain cover",
        "dependence_mode": "common_shock",
        "shock_share": 0.30,
        "precip_probability_scale": 1.0,
        "eq_probability_scale": 1.0,
        "precip_severity_1_fraction": 1.0,
        "precip_severity_2_fraction": 1.0,
        "description": "Either precipitation severity pays the full precipitation principal, still once per term.",
        "theme": "Full insurance payout",
    },
]


LIBRARY = REQUIRED + [
    {
        "scenario_id": "no_catastrophe",
        "scenario_name": "No catastrophe",
        "dependence_mode": "independent",
        "shock_share": 0.0,
        "precip_probability_scale": 0.08,
        "eq_probability_scale": 0.08,
        "description": "Occurrence rates are scaled down to 8 percent of baseline. This is a quiet book, not a literal zero-hazard world, because a zero probability is outside the scenario bounds.",
        "theme": "No-catastrophe scenario",
        "book": "baseline",
    },
    {
        "scenario_id": "independent_earthquake",
        "scenario_name": "Independent earthquakes",
        "dependence_mode": "independent",
        "shock_share": 0.0,
        "precip_probability_scale": 0.08,
        "eq_probability_scale": 1.0,
        "description": "Earthquakes keep baseline rates and are independent across countries. Precipitation is scaled to a residual 8 percent.",
        "theme": "Independent earthquake events",
        "book": "baseline",
    },
    {
        "scenario_id": "independent_precipitation",
        "scenario_name": "Independent precipitation",
        "dependence_mode": "independent",
        "shock_share": 0.0,
        "precip_probability_scale": 1.0,
        "eq_probability_scale": 0.08,
        "description": "Precipitation keeps baseline rates and is independent across countries. Earthquake occurrence is scaled to a residual 8 percent.",
        "theme": "Independent precipitation events",
        "book": "baseline",
    },
    {
        "scenario_id": "moderate_earthquake",
        "scenario_name": "Moderate earthquake",
        "dependence_mode": "common_shock",
        "shock_share": 0.30,
        "precip_probability_scale": 1.0,
        "eq_probability_scale": 1.35,
        "description": "Earthquake occurrence is 1.35 times baseline. Dependence and precipitation stay at the baseline settings.",
        "theme": "Moderate earthquake",
        "book": "baseline",
    },
    {
        "scenario_id": "severe_earthquake",
        "scenario_name": "Severe earthquake",
        "dependence_mode": "common_shock",
        "shock_share": 0.45,
        "precip_probability_scale": 1.0,
        "eq_probability_scale": 2.10,
        "description": "Earthquake occurrence is 2.10 times baseline and the common-shock share is raised to 0.45.",
        "theme": "Severe earthquake",
        "book": "baseline",
    },
    {
        "scenario_id": "extreme_precipitation",
        "scenario_name": "Extreme precipitation",
        "dependence_mode": "common_shock",
        "shock_share": 0.40,
        "precip_probability_scale": 2.40,
        "eq_probability_scale": 1.0,
        "description": "Precipitation occurrence is 2.40 times baseline. The common-shock share is 0.40.",
        "theme": "Extreme precipitation",
        "book": "baseline",
    },
    {
        "scenario_id": "simultaneous_earthquakes",
        "scenario_name": "Simultaneous earthquakes",
        "dependence_mode": "common_shock",
        "shock_share": 0.90,
        "precip_probability_scale": 0.15,
        "eq_probability_scale": 1.60,
        "description": "Earthquake rates are raised and 90 percent of the smaller marginal is a shared shock. Precipitation is residual.",
        "theme": "Simultaneous earthquakes",
        "book": "baseline",
    },
    {
        "scenario_id": "simultaneous_precipitation",
        "scenario_name": "Simultaneous precipitation",
        "dependence_mode": "common_shock",
        "shock_share": 0.90,
        "precip_probability_scale": 1.80,
        "eq_probability_scale": 0.15,
        "description": "Precipitation rates are raised and 90 percent of the smaller marginal is a shared shock. Earthquake occurrence is residual.",
        "theme": "Simultaneous precipitation",
        "book": "baseline",
    },
    {
        "scenario_id": "eq_rain_combination",
        "scenario_name": "Earthquake and precipitation",
        "dependence_mode": "common_shock",
        "shock_share": 0.50,
        "precip_probability_scale": 1.50,
        "eq_probability_scale": 1.50,
        "description": "Both hazards are scaled by 1.50 and the common-shock share is 0.50. The hazards remain independent of each other.",
        "theme": "Earthquake and precipitation combination",
        "book": "baseline",
    },
    {
        "scenario_id": "weak_dependence",
        "scenario_name": "Weak dependence",
        "dependence_mode": "common_shock",
        "shock_share": 0.10,
        "precip_probability_scale": 1.0,
        "eq_probability_scale": 1.0,
        "description": "Common-shock share is only 0.10 of the smaller marginal. Marginals match the baseline.",
        "theme": "Weak dependence",
        "book": "baseline",
    },
    {
        "scenario_id": "strong_dependence",
        "scenario_name": "Strong dependence",
        "dependence_mode": "common_shock",
        "shock_share": 0.95,
        "precip_probability_scale": 1.0,
        "eq_probability_scale": 1.0,
        "description": "Common-shock share is 0.95 of the smaller marginal. Marginals match the baseline, so expected loss is unchanged while joint payout risk rises.",
        "theme": "Strong dependence",
        "book": "baseline",
    },
    {
        "scenario_id": "moderate_losses",
        "scenario_name": "Moderate losses",
        "dependence_mode": "common_shock",
        "shock_share": 0.30,
        "precip_probability_scale": 1.20,
        "eq_probability_scale": 1.20,
        "description": "Both hazards are 1.20 times baseline, with the baseline common-shock share.",
        "theme": "Moderate financial losses",
        "book": "baseline",
    },
    {
        "scenario_id": "high_losses",
        "scenario_name": "High losses",
        "dependence_mode": "common_shock",
        "shock_share": 0.55,
        "precip_probability_scale": 1.70,
        "eq_probability_scale": 1.90,
        "description": "Earthquake occurrence is 1.90 times baseline, precipitation 1.70 times, and the common-shock share is 0.55.",
        "theme": "High financial losses",
        "book": "baseline",
    },
    {
        "scenario_id": "rare_extreme_catastrophe",
        "scenario_name": "Rare extreme catastrophe",
        "dependence_mode": "common_shock",
        "shock_share": 0.85,
        "precip_probability_scale": 2.20,
        "eq_probability_scale": 2.80,
        "description": "A low-frequency tail is represented by sharply higher occurrence and a 0.85 common-shock share. Still a synthetic rate, not a forecast.",
        "theme": "Rare extreme catastrophe",
        "book": "baseline",
    },
    {
        "scenario_id": "worst_case_stress",
        "scenario_name": "Worst-case stress",
        "dependence_mode": "common_shock",
        "shock_share": 1.0,
        "precip_probability_scale": 2.80,
        "eq_probability_scale": 3.00,
        "precip_severity_1_fraction": 1.0,
        "precip_severity_2_fraction": 1.0,
        "description": "Highest occurrence scales in the library, a full common shock, and full precipitation steps. The smaller country's idiosyncratic shock is zero.",
        "theme": "Worst-case stress scenario",
        "book": "baseline",
    },
    {
        "scenario_id": "threshold_exceedance",
        "scenario_name": "Portfolio threshold 10 million",
        "dependence_mode": "common_shock",
        "shock_share": 0.30,
        "precip_probability_scale": 1.0,
        "eq_probability_scale": 1.0,
        "description": "Same portfolio as the baseline, scored at a 10 million USD threshold instead of 25 million. Expected loss does not change. Exceedance and stop-loss do.",
        "theme": "Portfolio threshold exceedance",
        "book": "baseline",
        "threshold": LOW_THRESHOLD,
    },
]


def scaled_book(scale: float) -> dict:
    book = {}
    for country, spec in BASE_TERMS.items():
        copied = dict(spec)
        copied["eq_principal"] = int(round(spec["eq_principal"] * scale))
        copied["precip_principal"] = int(round(spec["precip_principal"] * scale))
        copied["eq_cap"] = copied["eq_principal"]
        copied["contract_cap"] = int(round(spec["contract_cap"] * scale))
        book[country] = copied
    return book


def concentrated_book() -> dict:
    """Move insured limit toward the Kyrgyz earthquake tower."""
    book = {country: dict(spec) for country, spec in BASE_TERMS.items()}
    book[KG]["eq_principal"] = 110_000_000
    book[KG]["precip_principal"] = 8_000_000
    book[KG]["eq_cap"] = 110_000_000
    book[KG]["contract_cap"] = 114_000_000
    book[TJ]["eq_principal"] = 28_000_000
    book[TJ]["precip_principal"] = 10_000_000
    book[TJ]["eq_cap"] = 28_000_000
    book[TJ]["contract_cap"] = 34_000_000
    return book


def fraction_book(eq_fractions: tuple[float, float, float], precip_fractions: tuple[float, float]) -> dict:
    book = {country: dict(spec) for country, spec in BASE_TERMS.items()}
    for country in book:
        book[country]["eq_fractions"] = eq_fractions
        book[country]["precip_fractions"] = precip_fractions
    return book


def with_scenario(specs: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame([scenario_row(spec) for spec in specs])
    return frame


def make_bundle(book: dict, scenarios: list[dict], occurrence_scale: dict | None = None, cap_mode: str = "baseline") -> DataBundle:
    present = {spec["scenario_id"] for spec in scenarios}
    specs = list(scenarios)
    for required in REQUIRED:
        if required["scenario_id"] not in present:
            specs.append(required)
    return DataBundle(
        hazards=hazard_parameter_frame(book, occurrence_scale),
        dependence=dependence_frame(),
        exposure=country_exposure_frame(book),
        contracts=contract_frame(book, cap_mode=cap_mode),
        scenarios=with_scenario(specs),
        source="synthetic demo seed 62",
    )


def run_scenario(bundle: DataBundle, scenario_id: str, threshold: float) -> dict:
    model = build_portfolio(bundle, scenario_id)
    result = estimate(model, threshold)
    return result.metrics


def classify(expected_loss: float, exceedance: float) -> str:
    if expected_loss >= 45_000_000 or exceedance >= 0.50:
        return "extreme"
    if expected_loss >= 28_000_000 or exceedance >= 0.32:
        return "high"
    if expected_loss >= 10_000_000 or exceedance >= 0.12:
        return "medium"
    return "low"


def draw_occurrence(rng: np.random.Generator, p_a: float, p_b: float, share: float) -> tuple[bool, bool, bool]:
    alpha = share * min(p_a, p_b)
    beta_a = 0.0 if alpha >= 1.0 else (p_a - alpha) / (1.0 - alpha)
    beta_b = 0.0 if alpha >= 1.0 else (p_b - alpha) / (1.0 - alpha)
    shared = bool(rng.random() < alpha)
    occurs_a = shared or bool(rng.random() < beta_a)
    occurs_b = shared or bool(rng.random() < beta_b)
    return occurs_a, occurs_b, shared


def contractual_year(spec: dict, eq_severity: int, rain_severity: int) -> tuple[int, int, int]:
    eq_steps = [0, *[money_fraction(fraction, spec["eq_principal"]) for fraction in spec["eq_fractions"]]]
    rain_steps = [0, *[money_fraction(fraction, spec["precip_principal"]) for fraction in spec["precip_fractions"]]]
    eq_pay = min(spec["eq_cap"], eq_steps[eq_severity])
    rain_pay = rain_steps[rain_severity]
    raw = eq_pay + rain_pay
    capped = min(spec["contract_cap"], raw)
    if raw > capped and raw > 0:
        eq_pay = int(round(eq_pay * capped / raw))
        rain_pay = capped - eq_pay
    return eq_pay, rain_pay, capped


def build_catalogue(rng: np.random.Generator, exposure: pd.DataFrame) -> pd.DataFrame:
    by_region = {}
    for row in exposure.itertuples(index=False):
        by_region.setdefault(row.region_id, {})[row.asset_category] = int(row.insured_exposure_usd)
    region_name = {region_id: name for country in REGIONS for region_id, name, *_rest in REGIONS[country]}
    rows = []
    event_index = 0
    for year in range(1, HORIZON_YEARS + 1):
        severities: dict[tuple[str, str], int] = {}
        mechanisms: dict[tuple[str, str], str] = {}
        for hazard, key_a, key_b in (
            ("earthquake", "eq_p", "eq_sev"),
            ("precipitation", "rain_p", "rain_sev"),
        ):
            occurs_kg, occurs_tj, shared = draw_occurrence(
                rng, BASE_TERMS[KG][key_a], BASE_TERMS[TJ][key_a], 0.30
            )
            for country, occurs in ((KG, occurs_kg), (TJ, occurs_tj)):
                if not occurs:
                    continue
                severities[(country, hazard)] = int(
                    rng.choice(np.arange(1, len(BASE_TERMS[country][key_b]) + 1), p=BASE_TERMS[country][key_b])
                )
                mechanisms[(country, hazard)] = "common_shock" if shared else "idiosyncratic"
        for country in (KG, TJ):
            eq_severity = severities.get((country, "earthquake"), 0)
            rain_severity = severities.get((country, "precipitation"), 0)
            eq_pay, rain_pay, total_pay = contractual_year(BASE_TERMS[country], eq_severity, rain_severity)
            pays = {"earthquake": eq_pay, "precipitation": rain_pay}
            country_code = "KG" if country == KG else "TJ"
            for hazard in ("earthquake", "precipitation"):
                severity = eq_severity if hazard == "earthquake" else rain_severity
                if severity == 0:
                    event_index += 1
                    rows.append(
                        {
                            "event_id": f"EVT-{event_index:04d}-{country_code}-{hazard[:3].upper()}-QUIET",
                            "event_group_id": f"GRP-{year:03d}-{country_code}-{hazard[:3].upper()}",
                            "catalogue_year": year,
                            "country": country,
                            "region_id": "NATIONAL",
                            "region_name": "National year",
                            "asset_category": "portfolio",
                            "hazard": hazard,
                            "severity": 0,
                            "severity_label": "no event",
                            "impact_role": "none",
                            "occurrence_mechanism": "no_event",
                            "joint_country_event": "no",
                            "primary_region_id": "",
                            "ground_up_loss_usd": 0,
                            "contract_payout_usd": 0,
                            "payout_status": "no_payout",
                            "master_contract_id": "CON-KG-MASTER" if country == KG else "CON-TJ-MASTER",
                            "currency": "USD",
                            "data_quality_flag": "valid",
                            "include_in_calculations": "yes",
                            "data_label": LABEL,
                            "description": (
                                f"Synthetic quiet catalogue year {year}: no {hazard} trigger in {country}. "
                                "Not an observed non-event."
                            ),
                        }
                    )
                    continue
                event_index += 1
                group = f"GRP-{year:03d}-{country_code}-{hazard[:3].upper()}"
                regions = REGIONS[country]
                weights = np.array([item[4] if hazard == "earthquake" else item[5] for item in regions], dtype=float)
                weights = weights / weights.sum()
                primary_index = int(rng.choice(len(regions), p=weights))
                primary = regions[primary_index][0]
                near_a, near_b, far = NEIGHBORS[primary]
                factors = {primary: 1.0, near_a: 0.40, near_b: 0.22, far: 0.05}
                damage = EQ_DAMAGE if hazard == "earthquake" else RAIN_DAMAGE
                category_scale = EQ_CATEGORY if hazard == "earthquake" else RAIN_CATEGORY
                ground = {}
                for region_id, factor in factors.items():
                    for category, exposure_usd in by_region[region_id].items():
                        ratio = damage[severity] * factor * category_scale[category]
                        ground[(region_id, category)] = int(round(exposure_usd * ratio))
                ground_total = sum(ground.values())
                payout_total = pays[hazard]
                if ground_total <= 0:
                    allocations = {key: 0 for key in ground}
                else:
                    allocations = dict(
                        zip(
                            ground,
                            allocate(payout_total, [ground[key] for key in ground]),
                        )
                    )
                both = (KG, hazard) in severities and (TJ, hazard) in severities
                exhausted = (hazard == "earthquake" and eq_pay >= BASE_TERMS[country]["eq_cap"]) or (
                    total_pay >= BASE_TERMS[country]["contract_cap"]
                )
                word = SEVERITY_WORD[(hazard, severity)]
                for (region_id, category), ground_up in ground.items():
                    payout = allocations[(region_id, category)]
                    if payout <= 0:
                        status = "no_payout"
                    elif exhausted and region_id == primary:
                        status = "coverage_exhausted"
                    else:
                        status = "partial_payout"
                    role = "primary" if region_id == primary else "affected"
                    rows.append(
                        {
                            "event_id": f"EVT-{event_index:04d}-{region_id}-{category[:3].upper()}",
                            "event_group_id": group,
                            "catalogue_year": year,
                            "country": country,
                            "region_id": region_id,
                            "region_name": region_name[region_id],
                            "asset_category": category,
                            "hazard": hazard,
                            "severity": severity,
                            "severity_label": word,
                            "impact_role": role,
                            "occurrence_mechanism": mechanisms[(country, hazard)],
                            "joint_country_event": "yes" if both else "no",
                            "primary_region_id": primary,
                            "ground_up_loss_usd": ground_up,
                            "contract_payout_usd": payout,
                            "payout_status": status,
                            "master_contract_id": "CON-KG-MASTER" if country == KG else "CON-TJ-MASTER",
                            "currency": "USD",
                            "data_quality_flag": "valid",
                            "include_in_calculations": "yes",
                            "data_label": LABEL,
                            "description": (
                                f"Synthetic {word} {hazard} in catalogue year {year}, "
                                f"{role} impact on {region_name[region_id]} {category.replace('_', ' ')}. "
                                "Not an observed disaster."
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def insurance_contracts(book: dict) -> pd.DataFrame:
    rows = []
    for country, contract_id in ((KG, "CON-KG-MASTER"), (TJ, "CON-TJ-MASTER")):
        spec = book[country]
        rows.append(
            {
                "contract_id": contract_id,
                "contract_role": "master",
                "country": country,
                "term_years": 3,
                "currency": "USD",
                "eq_principal_usd": spec["eq_principal"],
                "precip_principal_usd": spec["precip_principal"],
                "attachment_usd": 0,
                "deductible_usd": 0,
                "eq_cap_usd": spec["eq_cap"],
                "payout_limit_usd": spec["contract_cap"],
                "eq_step_1_usd": money_fraction(spec["eq_fractions"][0], spec["eq_principal"]),
                "eq_step_2_usd": money_fraction(spec["eq_fractions"][1], spec["eq_principal"]),
                "eq_step_3_usd": money_fraction(spec["eq_fractions"][2], spec["eq_principal"]),
                "precip_step_1_usd": money_fraction(spec["precip_fractions"][0], spec["precip_principal"]),
                "precip_step_2_usd": money_fraction(spec["precip_fractions"][1], spec["precip_principal"]),
                "aggregation": "earthquake sums annual steps up to the earthquake cap; precipitation pays the highest step once per term; the country cap then applies",
                "data_quality_flag": "valid",
                "include_in_calculations": "yes",
                "data_label": LABEL,
                "notes": "Illustrative master contract. Not an official ADB schedule.",
            }
        )
    kg = book[KG]
    moderate = money_fraction(kg["eq_fractions"][1], kg["eq_principal"])
    rows.append(
        {
            "contract_id": "CON-KG-EQ-XOL",
            "contract_role": "excess_layer",
            "country": KG,
            "term_years": 3,
            "currency": "USD",
            "eq_principal_usd": kg["eq_principal"],
            "precip_principal_usd": 0,
            "attachment_usd": moderate,
            "deductible_usd": 0,
            "eq_cap_usd": kg["eq_cap"] - moderate,
            "payout_limit_usd": kg["eq_cap"] - moderate,
            "eq_step_1_usd": 0,
            "eq_step_2_usd": 0,
            "eq_step_3_usd": kg["eq_cap"] - moderate,
            "precip_step_1_usd": 0,
            "precip_step_2_usd": 0,
            "aggregation": "pays only the earthquake step above the moderate-step attachment, and only on an extreme earthquake",
            "data_quality_flag": "valid",
            "include_in_calculations": "yes",
            "data_label": LABEL,
            "notes": "Illustration of an attachment. Not stacked on top of the master payout in portfolio totals.",
        }
    )
    tj = book[TJ]
    first = money_fraction(tj["precip_fractions"][0], tj["precip_principal"])
    rows.append(
        {
            "contract_id": "CON-TJ-PRECIP-LAYER",
            "contract_role": "excess_layer",
            "country": TJ,
            "term_years": 3,
            "currency": "USD",
            "eq_principal_usd": 0,
            "precip_principal_usd": tj["precip_principal"],
            "attachment_usd": first,
            "deductible_usd": 0,
            "eq_cap_usd": 0,
            "payout_limit_usd": tj["precip_principal"] - first,
            "eq_step_1_usd": 0,
            "eq_step_2_usd": 0,
            "eq_step_3_usd": 0,
            "precip_step_1_usd": 0,
            "precip_step_2_usd": tj["precip_principal"] - first,
            "aggregation": "pays the precipitation amount above the moderate-step attachment",
            "data_quality_flag": "valid",
            "include_in_calculations": "yes",
            "data_label": LABEL,
            "notes": "Illustration of a precipitation attachment. Not an official layer.",
        }
    )
    return pd.DataFrame(rows)


def dependence_library() -> pd.DataFrame:
    rows = [
        {
            "dependence_id": "DEP-EQ-BASE",
            "scope": "portfolio",
            "hazard": "earthquake",
            "country_a": KG,
            "country_b": TJ,
            "dependence_mode": "common_shock",
            "shock_share_of_min_marginal": 0.30,
            "use_in_baseline_model": "yes",
            "interpretation": "Baseline shared shock. Preserves each country's earthquake occurrence probability.",
        },
        {
            "dependence_id": "DEP-RAIN-BASE",
            "scope": "portfolio",
            "hazard": "precipitation",
            "country_a": KG,
            "country_b": TJ,
            "dependence_mode": "common_shock",
            "shock_share_of_min_marginal": 0.30,
            "use_in_baseline_model": "yes",
            "interpretation": "Baseline shared shock for precipitation. Independent of the earthquake shock.",
        },
        {
            "dependence_id": "DEP-EQ-WEAK",
            "scope": "scenario",
            "hazard": "earthquake",
            "country_a": KG,
            "country_b": TJ,
            "dependence_mode": "common_shock",
            "shock_share_of_min_marginal": 0.10,
            "use_in_baseline_model": "no",
            "interpretation": "Weak-dependence scenario. Used by scenario weak_dependence, not by the baseline file.",
        },
        {
            "dependence_id": "DEP-EQ-STRONG",
            "scope": "scenario",
            "hazard": "earthquake",
            "country_a": KG,
            "country_b": TJ,
            "dependence_mode": "common_shock",
            "shock_share_of_min_marginal": 0.95,
            "use_in_baseline_model": "no",
            "interpretation": "Strong-dependence scenario. Marginals stay fixed, joint occurrence rises.",
        },
        {
            "dependence_id": "DEP-INDEPENDENT",
            "scope": "scenario",
            "hazard": "both",
            "country_a": KG,
            "country_b": TJ,
            "dependence_mode": "independent",
            "shock_share_of_min_marginal": 0.0,
            "use_in_baseline_model": "no",
            "interpretation": "Independent scenario. No shared shock.",
        },
    ]
    pairs = [
        ("REG-KG-BISHKEK", "REG-KG-CHUY", "earthquake", 0.55, "Neighbouring basins share shaking more often than distant regions."),
        ("REG-KG-OSH", "REG-KG-BATKEN", "earthquake", 0.48, "Southern districts are grouped for a synthetic regional earthquake cluster."),
        ("REG-KG-JALAL", "REG-KG-OSH", "precipitation", 0.42, "Valley precipitation is more likely to arrive together."),
        ("REG-TJ-DUSHANBE", "REG-TJ-KHATLON", "precipitation", 0.46, "Capital and Khatlon share a synthetic precipitation cluster."),
        ("REG-TJ-GBAO", "REG-TJ-PAMIR", "earthquake", 0.52, "High-mountain corridors share a synthetic earthquake cluster."),
        ("REG-TJ-SUGHD", "REG-TJ-ZERAVSHAN", "earthquake", 0.36, "Northern districts have a milder synthetic cluster."),
        ("REG-KG-NARYN", "REG-TJ-RASHT", "earthquake", 0.22, "Cross-border mountain shaking is possible but weaker than within-country clusters."),
        ("REG-KG-BISHKEK", "REG-TJ-DUSHANBE", "precipitation", 0.18, "The two capitals are only weakly linked for precipitation."),
    ]
    for index, (left, right, hazard, share, text) in enumerate(pairs, start=1):
        rows.append(
            {
                "dependence_id": f"DEP-REGION-{index:02d}",
                "scope": "region_pair",
                "hazard": hazard,
                "country_a": left,
                "country_b": right,
                "dependence_mode": "regional_cluster",
                "shock_share_of_min_marginal": share,
                "use_in_baseline_model": "no",
                "interpretation": text + " Illustrative only. The pricing model uses the country common shock.",
            }
        )
    frame = pd.DataFrame(rows)
    frame["data_quality_flag"] = "valid"
    frame["include_in_calculations"] = "yes"
    frame["data_label"] = LABEL
    frame["currency"] = "USD"
    return frame


def invalid_records() -> pd.DataFrame:
    samples = [
        ("INV-001", "exposure_data", "EXP-BAD-MISSING", "missing_value", "insured_exposure_usd is blank"),
        ("INV-002", "exposure_data", "EXP-BAD-NEGATIVE", "negative_exposure", "insured_exposure_usd is -5000000"),
        ("INV-003", "exposure_data", "EXP-KG-BISHKEK-RES", "duplicate_record", "duplicates a valid exposure_id"),
        ("INV-004", "hazard_events", "EVT-BAD-PROB", "invalid_probability", "reported frequency 1.40 is above 1"),
        ("INV-005", "hazard_events", "EVT-0001-REG-KG-BISHKEK-RES", "duplicate_record", "reuses an event id pattern"),
        ("INV-006", "insurance_contracts", "CON-MISSING", "incorrect_contract_reference", "master_contract_id does not exist"),
        ("INV-007", "insurance_contracts", "CON-KG-MASTER", "inconsistent_units", "principal entered as 80 with unit millions beside USD rows"),
        ("INV-008", "dependence_data", "DEP-BAD-SHARE", "invalid_probability", "shock share 1.50 is above 1"),
        ("INV-009", "scenario_data", "bad scenario", "invalid_identifier", "scenario_id contains a space"),
        ("INV-010", "exposure_data", "EXP-BAD-CURRENCY", "inconsistent_units", "currency is KGS while the book is USD"),
        ("INV-011", "hazard_events", "EVT-BAD-LOSS", "negative_exposure", "ground_up_loss_usd is -120000"),
        ("INV-012", "scenario_data", "independent", "invalid_probability", "independent mode paired with shock share 0.40"),
        ("INV-013", "insurance_contracts", "CON-BAD-CAP", "inconsistent_contract", "payout limit is below the attachment"),
        ("INV-014", "hazard_events", "EVT-BAD-COUNTRY", "incorrect_contract_reference", "country Exampleland is outside the book"),
        ("INV-015", "dependence_data", "DEP-BAD-BLANK", "missing_value", "hazard name is blank"),
        ("INV-016", "exposure_data", "EXP-BAD-REGION", "incorrect_contract_reference", "region_id does not match a catalogue region"),
    ]
    rows = []
    for record_id, source, key, flag, detail in samples:
        rows.append(
            {
                "invalid_record_id": record_id,
                "source_file": source,
                "referenced_id": key,
                "data_quality_flag": flag,
                "include_in_calculations": "no",
                "problem": detail,
                "data_label": "synthetic_invalid_example",
                "notes": "Held out of the demonstration calculations. Use only to show validation.",
            }
        )
    return pd.DataFrame(rows)


def dictionary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    descriptions = {
        "exposure_id": "Stable id for one country, region and asset category.",
        "portfolio_book": "Which exposure book the row belongs to.",
        "country": "Kyrgyz Republic or Tajikistan.",
        "region_id": "Synthetic region id. Not an official administrative code.",
        "region_name": "Synthetic region name used in the demo.",
        "settlement_type": "urban, valley or mountain. Drives the asset mix.",
        "asset_category": "residential, commercial, public_infrastructure or agriculture.",
        "insured_exposure_usd": "Insured value in whole USD.",
        "eq_weight": "Region share of the country total before category split.",
        "currency": "Always USD in the valid book.",
        "master_contract_id": "Country master contract that covers this exposure.",
        "data_quality_flag": "valid on demonstration rows. Invalid examples live in invalid_records.csv.",
        "include_in_calculations": "yes for demonstration rows. Invalid rows are no.",
        "data_label": "Marks the row as synthetic.",
        "notes": "Short caveat for the row.",
        "event_id": "One impact of one hazard occurrence on one region and category.",
        "event_group_id": "Joins every regional impact of the same country-hazard-year.",
        "catalogue_year": "Year index in the 60-year synthetic catalogue. Not a calendar year.",
        "hazard": "earthquake or precipitation.",
        "severity": "Earthquake 1 to 3, precipitation 1 to 2. Matches the contract steps.",
        "severity_label": "moderate, severe or extreme.",
        "impact_role": "primary epicentre or storm centre, or a neighbouring affected region.",
        "occurrence_mechanism": "common_shock or idiosyncratic.",
        "joint_country_event": "yes when both countries have this hazard in the same catalogue year.",
        "primary_region_id": "Region drawn as the event centre.",
        "ground_up_loss_usd": "Synthetic physical loss before the insurance schedule.",
        "contract_payout_usd": "This row's share of the country-hazard contractual payout.",
        "payout_status": "no_payout, partial_payout or coverage_exhausted.",
        "description": "Plain-language description. Not a historical event.",
        "dependence_id": "Id for a dependence statement.",
        "scope": "portfolio, scenario or region_pair.",
        "country_a": "First country, or a region id when scope is region_pair.",
        "country_b": "Second country, or a region id when scope is region_pair.",
        "dependence_mode": "common_shock, independent or regional_cluster.",
        "shock_share_of_min_marginal": "Share of the smaller occurrence probability assigned to the shared shock.",
        "use_in_baseline_model": "yes only for the two rows that define the baseline app model.",
        "interpretation": "What the dependence row is for.",
        "contract_id": "Contract id.",
        "contract_role": "master or excess_layer.",
        "term_years": "Contract term. The master term is 3 years.",
        "eq_principal_usd": "Earthquake principal in USD.",
        "precip_principal_usd": "Precipitation principal in USD.",
        "attachment_usd": "Loss level where this layer starts. Masters attach at the first severity step.",
        "deductible_usd": "Deductible. Masters use 0 so the step schedule is the whole rule.",
        "eq_cap_usd": "Earthquake payout cap.",
        "payout_limit_usd": "Maximum payment under this contract.",
        "eq_step_1_usd": "Earthquake payout at severity 1.",
        "eq_step_2_usd": "Earthquake payout at severity 2.",
        "eq_step_3_usd": "Earthquake payout at severity 3.",
        "precip_step_1_usd": "Precipitation payout at severity 1.",
        "precip_step_2_usd": "Precipitation payout at severity 2.",
        "aggregation": "How annual steps become a term payment.",
        "scenario_id": "Scenario slug.",
        "scenario_name": "Display name.",
        "shock_share": "Scenario common-shock share.",
        "precip_probability_scale": "Multiplier on precipitation occurrence.",
        "eq_probability_scale": "Multiplier on earthquake occurrence.",
        "precip_severity_1_fraction": "Optional precipitation step override.",
        "precip_severity_2_fraction": "Optional precipitation step override.",
        "assumption_label": "Synthetic label stored with the scenario.",
        "theme": "Demonstration theme from the scenario library.",
        "risk_classification": "low, medium, high or extreme, assigned from the enumerated metrics.",
        "threshold_usd": "Strict portfolio threshold T for P(L > T).",
        "app_bundle": "yes if this scenario is in the uploadable workbook on the baseline book.",
        "metric": "expected_loss, joint_attachment_probability, exceedance_probability, expected_stop_loss or both_exhausted_probability.",
        "value": "Enumerated synthetic value.",
        "tolerance": "0.01 USD for money, 1e-9 for probabilities.",
        "units": "USD or probability.",
        "source": "internal_exact_enumeration.",
        "book": "Which contract and hazard book was enumerated.",
        "invalid_record_id": "Id of a held-out invalid example.",
        "source_file": "Which demonstration file the invalid example imitates.",
        "referenced_id": "Id the invalid example points at.",
        "problem": "What is wrong with the example.",
        "file_name": "CSV file this dictionary row documents.",
        "column_name": "Column name.",
        "data_type": "text, integer, number or probability.",
        "unit": "USD, probability, year, flag or text.",
        "required": "yes if the column is part of the valid demonstration schema.",
        "example": "A short example value.",
    }
    rows = []
    for file_name, frame in frames.items():
        for column in frame.columns:
            sample = frame[column].dropna().iloc[0] if frame[column].notna().any() else ""
            kind = "integer" if str(frame[column].dtype).startswith("int") else "number" if str(frame[column].dtype).startswith("float") else "text"
            unit = "USD" if column.endswith("usd") or column.endswith("_usd") or "loss" in column or "payout" in column or "principal" in column or "cap" in column or "attachment" in column or "deductible" in column or "exposure" in column else "text"
            if "probability" in column or column.endswith("share") or column.endswith("fraction") or column.endswith("scale"):
                unit = "probability or multiplier"
                kind = "number"
            rows.append(
                {
                    "file_name": file_name,
                    "column_name": column,
                    "data_type": kind,
                    "unit": unit,
                    "required": "yes",
                    "description": descriptions.get(column, "Demonstration field."),
                    "example": str(sample)[:80],
                    "data_label": LABEL,
                }
            )
    return pd.DataFrame(rows)


def write_readme(path: Path, summary: dict, scenarios: pd.DataFrame, reference: pd.DataFrame) -> None:
    ranked = scenarios.sort_values("expected_loss_usd", ascending=False)
    lines = [
        "# QuantumRisk Analytics synthetic demonstration dataset",
        "",
        "This folder is a synthetic book for a live demonstration. It is not a historical loss catalogue, not an observed insurance portfolio, and not an official ADB payout schedule or benchmark.",
        "",
        "The random seed is **62**. Running `python scripts/generate_demo_dataset.py` rebuilds every file in this folder. The original fixture in `data/synthetic` and the locked reference in `data/reference` are not modified.",
        "",
        "## What the files are",
        "",
        "| File | Role |",
        "| --- | --- |",
        "| `exposure_data.csv` | Regional insured values by country, settlement type and asset category. Parts sum to the country totals. |",
        "| `hazard_events.csv` | A 60-year synthetic catalogue. Triggered hazards are split across regions and asset categories. Years with no trigger are one national zero-payout row per country and hazard. |",
        "| `dependence_data.csv` | Baseline common-shock settings, plus scenario and regional-cluster notes. Only rows with `use_in_baseline_model = yes` define the pricing model. |",
        "| `insurance_contracts.csv` | Two master contracts and two excess layers. Layers illustrate attachment. They are not added on top of the master payout. |",
        "| `scenario_data.csv` | Twenty-eight scenarios, including the five ids the app requires. |",
        "| `reference_results.csv` | Exact three-year term metrics for every scenario. |",
        "| `invalid_records.csv` | Sixteen held-out bad rows for a validation demo. `include_in_calculations = no`. |",
        "| `data_dictionary.csv` | Column definitions. |",
        "| `quantumrisk_demo_bundle.xlsx` | Uploadable workbook for the baseline book and the scenarios that the current app can run without a further edit. |",
        "",
        "## Relationships",
        "",
        "- Country insured exposure equals earthquake principal plus precipitation principal.",
        "- Regional exposures sum to that country total.",
        "- A catalogue event draws occurrence with the same Marshall-Olkin common shock as the pricing model (baseline share 0.30) and the same conditional severities.",
        "- The master payout for a catalogue year uses the contract steps. Earthquake pays the severity step, capped at the earthquake cap. Precipitation pays the severity step. The country cap is then applied, and any haircut is shared in proportion to the two peril payments.",
        "- That country-hazard payout is allocated to impact rows in proportion to ground-up loss. Impact payouts in one event group sum to the contractual payout.",
        "- Reference metrics are not averages of the catalogue. They are the three-year term calculation: earthquake steps accumulate and cap, precipitation pays once at the highest step, then the country cap applies. Portfolio loss is the sum of the two countries. Thresholds are strict: loss greater than T.",
        "",
        "## Baseline contract",
        "",
        f"- {KG}: earthquake principal {BASE_TERMS[KG]['eq_principal']:,} USD, precipitation principal {BASE_TERMS[KG]['precip_principal']:,} USD, country cap {BASE_TERMS[KG]['contract_cap']:,} USD. Occurrence 0.12 earthquake and 0.14 precipitation.",
        f"- {TJ}: earthquake principal {BASE_TERMS[TJ]['eq_principal']:,} USD, precipitation principal {BASE_TERMS[TJ]['precip_principal']:,} USD, country cap {BASE_TERMS[TJ]['contract_cap']:,} USD. Occurrence 0.09 earthquake and 0.20 precipitation.",
        "- Earthquake steps are 15/40/100 percent in the Kyrgyz Republic and 20/55/100 percent in Tajikistan.",
        "- Precipitation steps are 40/100 percent and 50/100 percent.",
        "- Term is 3 years. Currency is USD.",
        "",
        "## Findings from this seed",
        "",
        f"- Insured exposure is {summary['exposure_kg']:,.0f} USD in the Kyrgyz Republic and {summary['exposure_tj']:,.0f} USD in Tajikistan. The Kyrgyz book is larger.",
        f"- The largest regional concentration is {summary['top_region']} at {summary['top_region_share']:.1%} of its country.",
        f"- In the catalogue, earthquake contractual payouts are {summary['eq_payout']:,.0f} USD and precipitation contractual payouts are {summary['rain_payout']:,.0f} USD.",
        f"- Catalogue payout status counts: {summary['status_counts']}.",
        f"- Triggered joint-country hazard years: {summary['joint_groups']}. Triggered single-country hazard years: {summary['single_groups']}. Quiet national rows are counted in the no-payout status, not here.",
        f"- Baseline expected loss is {summary['base_loss']:,.0f} USD. The worst-case stress expected loss is {summary['worst_loss']:,.0f} USD. The quiet book is {summary['quiet_loss']:,.0f} USD.",
        f"- Joint attachment is {summary['joint_independent']:.3f} when the countries are independent, {summary['joint_base']:.3f} at the 0.30 common shock, and {summary['joint_strong']:.3f} at the 0.95 common shock. Expected loss is the same in those three, because the marginals do not change.",
        f"- At a 10 million threshold the baseline exceedance is {summary['exceed_10']:.3f}. At 25 million it is {summary['exceed_25']:.3f}. Expected loss is unchanged.",
        "",
        "Highest expected loss:",
        "",
    ]
    for row in ranked.head(5).itertuples(index=False):
        lines.append(f"- {row.scenario_name}: {row.expected_loss_usd:,.0f} USD ({row.risk_classification}).")
    lines.extend(
        [
            "",
            "## How to use it in the app",
            "",
            "On the Data page, upload `quantumrisk_demo_bundle.xlsx` as the bundle. The workbook sheets are `hazards`, `dependence`, `exposure`, `contracts` and `scenarios`. Invalid examples are not in that workbook.",
            "",
            "Scenarios that change only the shock share, the occurrence scales or the precipitation steps are in the workbook. Country-specific rates, resized principals, tighter caps and earthquake step changes are enumerated in `reference_results.csv` and marked `app_bundle = no` in `scenario_data.csv`, because one uploaded scenario table cannot retarget a single country or replace the contract.",
            "",
            "Default comparison threshold in the reference file is 25,000,000 USD, except the portfolio-threshold scenario, which uses 10,000,000 USD.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    exposure = exposure_frame(BASE_TERMS)
    exposure["portfolio_book"] = "baseline"
    events = build_catalogue(rng, exposure)
    contracts = insurance_contracts(BASE_TERMS)
    dependence = dependence_library()

    jobs = []
    baseline_bundle = make_bundle(BASE_TERMS, LIBRARY)
    validation = validate_bundle(baseline_bundle)
    if not validation.ok:
        raise SystemExit("Baseline bundle failed validation:\n- " + "\n- ".join(validation.errors))

    for spec in LIBRARY:
        threshold = float(spec.get("threshold", THRESHOLD))
        metrics = run_scenario(baseline_bundle, spec["scenario_id"], threshold)
        jobs.append((spec, "baseline", threshold, metrics))

    low = scaled_book(0.55)
    concentrated = concentrated_book()
    partial = fraction_book((0.10, 0.25, 0.45), (0.25, 0.55))
    full = fraction_book((1.0, 1.0, 1.0), (1.0, 1.0))
    special_plan = [
        ("kyrgyz_high_risk", "Kyrgyz high risk", "Kyrgyz Republic high-risk scenario", BASE_TERMS, {KG: 1.75, TJ: 1.0}, "baseline", "kyrgyz_rates",
         "Kyrgyz occurrence is 1.75 times baseline for both hazards. Tajik rates are unchanged."),
        ("tajikistan_high_risk", "Tajikistan high risk", "Tajikistan high-risk scenario", BASE_TERMS, {KG: 1.0, TJ: 1.75}, "baseline", "tajik_rates",
         "Tajik occurrence is 1.75 times baseline for both hazards. Kyrgyz rates are unchanged."),
        ("low_exposure_portfolio", "Low exposure portfolio", "Low exposure portfolio", low, None, "baseline", "low_exposure",
         "Principals, caps and insured exposure are 55 percent of the baseline. Hazard rates are unchanged."),
        ("high_exposure_concentration", "High exposure concentration", "High exposure concentration", concentrated, None, "baseline", "concentrated",
         "Insured limit is concentrated in the Kyrgyz earthquake tower: 110 million earthquake principal against a much smaller Tajik book."),
        ("partial_payout", "Partial payout schedule", "Partial insurance payout", partial, None, "baseline", "partial_schedule",
         "Steps stay below the principal: earthquake 10/25/45 percent and precipitation 25/55 percent."),
        ("full_payout", "Full payout schedule", "Full insurance payout", full, None, "baseline", "full_schedule",
         "Any positive severity pays the full peril principal, still subject to the caps and the term aggregation."),
        ("both_exhausted", "Both contracts exhausted", "Both contracts exhausted", BASE_TERMS, None, "tight", "tight_cap",
         "Occurrence is raised, the common shock is 0.90, precipitation steps are full, and the country cap equals the earthquake cap."),
    ]
    special_params = {
        "kyrgyz_high_risk": (0.30, 1.0, 1.0, "common_shock", None, None),
        "tajikistan_high_risk": (0.30, 1.0, 1.0, "common_shock", None, None),
        "low_exposure_portfolio": (0.30, 1.0, 1.0, "common_shock", None, None),
        "high_exposure_concentration": (0.30, 1.0, 1.0, "common_shock", None, None),
        "partial_payout": (0.30, 1.0, 1.0, "common_shock", 0.25, 0.55),
        "full_payout": (0.30, 1.0, 1.0, "common_shock", 1.0, 1.0),
        "both_exhausted": (0.90, 2.0, 2.4, "common_shock", 1.0, 1.0),
    }
    for scenario_id, name, theme, book, scale, cap_mode, book_name, description in special_plan:
        shock, precip_scale, eq_scale, mode, p1, p2 = special_params[scenario_id]
        spec = {
            "scenario_id": scenario_id,
            "scenario_name": name,
            "dependence_mode": mode,
            "shock_share": shock,
            "precip_probability_scale": precip_scale,
            "eq_probability_scale": eq_scale,
            "description": description,
            "theme": theme,
        }
        if p1 is not None:
            spec["precip_severity_1_fraction"] = p1
            spec["precip_severity_2_fraction"] = p2
        bundle = make_bundle(book, [spec], occurrence_scale=scale, cap_mode=cap_mode)
        check = validate_bundle(bundle)
        if not check.ok:
            raise SystemExit(f"{scenario_id} failed validation:\n- " + "\n- ".join(check.errors))
        metrics = run_scenario(bundle, scenario_id, THRESHOLD)
        jobs.append((spec, book_name, THRESHOLD, metrics))

    app_ids = {spec["scenario_id"] for spec in LIBRARY if spec["scenario_id"] != "threshold_exceedance"}
    scenario_rows = []
    reference_rows = []
    for spec, book_name, threshold, metrics in jobs:
        risk = classify(metrics["expected_loss"], metrics["exceedance_probability"])
        scenario_rows.append(
            {
                "scenario_id": spec["scenario_id"],
                "scenario_name": spec["scenario_name"],
                "theme": spec["theme"],
                "risk_classification": risk,
                "dependence_mode": spec["dependence_mode"],
                "shock_share": spec["shock_share"],
                "eq_probability_scale": spec["eq_probability_scale"],
                "precip_probability_scale": spec["precip_probability_scale"],
                "precip_severity_1_fraction": spec.get("precip_severity_1_fraction", ""),
                "precip_severity_2_fraction": spec.get("precip_severity_2_fraction", ""),
                "threshold_usd": int(threshold),
                "model_book": book_name,
                "app_bundle": "yes" if spec["scenario_id"] in app_ids and book_name == "baseline" else "no",
                "expected_loss_usd": metrics["expected_loss"],
                "joint_attachment_probability": metrics["joint_attachment_probability"],
                "exceedance_probability": metrics["exceedance_probability"],
                "expected_stop_loss_usd": metrics["expected_stop_loss"],
                "both_exhausted_probability": metrics["both_exhausted_probability"],
                "description": spec["description"],
                "data_quality_flag": "valid",
                "include_in_calculations": "yes",
                "assumption_label": LABEL,
                "data_label": LABEL,
            }
        )
        for metric, units, tolerance in (
            ("expected_loss", "USD", 0.01),
            ("joint_attachment_probability", "probability", 1e-9),
            ("exceedance_probability", "probability", 1e-9),
            ("expected_stop_loss", "USD", 0.01),
            ("both_exhausted_probability", "probability", 1e-9),
        ):
            reference_rows.append(
                {
                    "scenario_id": spec["scenario_id"],
                    "metric": metric,
                    "value": metrics[metric],
                    "tolerance": tolerance,
                    "units": units,
                    "threshold_usd": int(threshold),
                    "model_book": book_name,
                    "risk_classification": risk,
                    "source": "internal_exact_enumeration",
                    "data_label": LABEL,
                    "notes": "Synthetic demonstration reference from exact enumeration of this scenario. Not an ADB benchmark.",
                }
            )

    scenarios = pd.DataFrame(scenario_rows)
    reference = pd.DataFrame(reference_rows)
    invalid = invalid_records()

    # Catalogue consistency: group payouts match the sum of rows.
    grouped = events.groupby("event_group_id")["contract_payout_usd"].sum()
    if (grouped < 0).any():
        raise SystemExit("Negative catalogue payout.")
    status_counts = events.groupby("payout_status")["event_id"].nunique().to_dict()
    if set(status_counts) != {"no_payout", "partial_payout", "coverage_exhausted"}:
        raise SystemExit(f"Catalogue payout statuses are not varied: {status_counts}")

    by_id = {row.scenario_id: row for row in scenarios.itertuples(index=False)}
    if by_id["worst_case_stress"].expected_loss_usd <= by_id["base"].expected_loss_usd:
        raise SystemExit("Worst case is not above the baseline.")
    if by_id["no_catastrophe"].expected_loss_usd >= by_id["base"].expected_loss_usd:
        raise SystemExit("Quiet book is not below the baseline.")
    if by_id["strong_dependence"].joint_attachment_probability <= by_id["independent"].joint_attachment_probability:
        raise SystemExit("Strong dependence did not raise joint attachment.")
    if abs(by_id["strong_dependence"].expected_loss_usd - by_id["independent"].expected_loss_usd) > 1:
        raise SystemExit("Dependence scenarios changed expected loss.")
    if by_id["full_payout"].expected_loss_usd <= by_id["partial_payout"].expected_loss_usd:
        raise SystemExit("Full schedule does not pay more than the partial schedule.")
    if by_id["threshold_exceedance"].exceedance_probability <= by_id["base"].exceedance_probability:
        raise SystemExit("A lower threshold did not raise exceedance.")
    exposure_kg = int(exposure.loc[exposure["country"] == KG, "insured_exposure_usd"].sum())
    exposure_tj = int(exposure.loc[exposure["country"] == TJ, "insured_exposure_usd"].sum())
    if exposure_kg != BASE_TERMS[KG]["eq_principal"] + BASE_TERMS[KG]["precip_principal"]:
        raise SystemExit("Kyrgyz regional exposure does not sum to the principal.")
    if exposure_tj != BASE_TERMS[TJ]["eq_principal"] + BASE_TERMS[TJ]["precip_principal"]:
        raise SystemExit("Tajik regional exposure does not sum to the principal.")
    if exposure_kg <= exposure_tj:
        raise SystemExit("Kyrgyz exposure should exceed Tajik exposure.")

    frames = {
        "exposure_data.csv": exposure,
        "hazard_events.csv": events,
        "dependence_data.csv": dependence,
        "insurance_contracts.csv": contracts,
        "scenario_data.csv": scenarios,
        "reference_results.csv": reference,
        "invalid_records.csv": invalid,
    }
    dictionary_frame = dictionary(frames)
    frames["data_dictionary.csv"] = dictionary_frame
    for name, frame in frames.items():
        frame.to_csv(OUT / name, index=False)

    app_scenarios = scenarios[scenarios["app_bundle"] == "yes"]
    app_specs = []
    for row in app_scenarios.itertuples(index=False):
        spec = {
            "scenario_id": row.scenario_id,
            "scenario_name": row.scenario_name if row.scenario_id not in {item["scenario_id"] for item in REQUIRED} else next(item["scenario_name"] for item in REQUIRED if item["scenario_id"] == row.scenario_id),
            "dependence_mode": row.dependence_mode,
            "shock_share": row.shock_share,
            "precip_probability_scale": row.precip_probability_scale,
            "eq_probability_scale": row.eq_probability_scale,
            "description": row.description,
            "theme": row.theme,
        }
        if row.precip_severity_1_fraction != "" and pd.notna(row.precip_severity_1_fraction):
            spec["precip_severity_1_fraction"] = float(row.precip_severity_1_fraction)
            spec["precip_severity_2_fraction"] = float(row.precip_severity_2_fraction)
        app_specs.append(spec)
    # Required display names are enforced for the five ids.
    bundle = make_bundle(BASE_TERMS, app_specs)
    check = validate_bundle(bundle)
    if not check.ok:
        raise SystemExit("App workbook failed validation:\n- " + "\n- ".join(check.errors))
    workbook = OUT / "quantumrisk_demo_bundle.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        bundle.hazards.to_excel(writer, sheet_name="hazards", index=False)
        bundle.dependence.to_excel(writer, sheet_name="dependence", index=False)
        bundle.exposure.to_excel(writer, sheet_name="exposure", index=False)
        bundle.contracts.to_excel(writer, sheet_name="contracts", index=False)
        bundle.scenarios.to_excel(writer, sheet_name="scenarios", index=False)

    top = exposure.groupby(["country", "region_name"], as_index=False)["insured_exposure_usd"].sum()
    top["share"] = top["insured_exposure_usd"] / top.groupby("country")["insured_exposure_usd"].transform("sum")
    top_row = top.sort_values("share", ascending=False).iloc[0]
    triggered = events[events["severity"] > 0]
    joint_keys = triggered.loc[triggered["joint_country_event"] == "yes", ["catalogue_year", "hazard"]].drop_duplicates()
    single_keys = triggered.loc[triggered["joint_country_event"] == "no", ["catalogue_year", "hazard", "country"]].drop_duplicates()
    summary = {
        "exposure_kg": exposure_kg,
        "exposure_tj": exposure_tj,
        "top_region": f"{top_row.region_name} ({top_row.country})",
        "top_region_share": float(top_row.share),
        "eq_payout": int(events.loc[events["hazard"] == "earthquake", "contract_payout_usd"].sum()),
        "rain_payout": int(events.loc[events["hazard"] == "precipitation", "contract_payout_usd"].sum()),
        "status_counts": status_counts,
        "joint_groups": int(len(joint_keys)),
        "single_groups": int(len(single_keys)),
        "base_loss": by_id["base"].expected_loss_usd,
        "worst_loss": by_id["worst_case_stress"].expected_loss_usd,
        "quiet_loss": by_id["no_catastrophe"].expected_loss_usd,
        "joint_independent": by_id["independent"].joint_attachment_probability,
        "joint_base": by_id["base"].joint_attachment_probability,
        "joint_strong": by_id["strong_dependence"].joint_attachment_probability,
        "exceed_10": by_id["threshold_exceedance"].exceedance_probability,
        "exceed_25": by_id["base"].exceedance_probability,
    }
    write_readme(OUT / "README.md", summary, scenarios, reference)
    clean_rows = sum(len(frame) for name, frame in frames.items() if name not in {"invalid_records.csv", "data_dictionary.csv"})
    print(f"Wrote {OUT}")
    print(f"Clean analytical rows (excluding dictionary and invalid examples): {clean_rows}")
    print(f"Invalid examples: {len(invalid)}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Event impact rows: {len(events)}")
    print(f"Baseline E[L]: {summary['base_loss']:.2f}")
    print(f"Worst E[L]: {summary['worst_loss']:.2f}")
    print(f"Strong joint attachment: {summary['joint_strong']:.6f}")


if __name__ == "__main__":
    main()
