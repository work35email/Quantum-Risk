# QuantumRisk Analytics synthetic demonstration dataset

This folder is a synthetic book for a live demonstration. It is not a historical loss catalogue, not an observed insurance portfolio, and not an official ADB payout schedule or benchmark.

The random seed is **62**. Running `python scripts/generate_demo_dataset.py` rebuilds every file in this folder. The original fixture in `data/synthetic` and the locked reference in `data/reference` are not modified.

## What the files are

| File | Role |
| --- | --- |
| `exposure_data.csv` | Regional insured values by country, settlement type and asset category. Parts sum to the country totals. |
| `hazard_events.csv` | A 60-year synthetic catalogue. Triggered hazards are split across regions and asset categories. Years with no trigger are one national zero-payout row per country and hazard. |
| `dependence_data.csv` | Baseline common-shock settings, plus scenario and regional-cluster notes. Only rows with `use_in_baseline_model = yes` define the pricing model. |
| `insurance_contracts.csv` | Two master contracts and two excess layers. Layers illustrate attachment. They are not added on top of the master payout. |
| `scenario_data.csv` | Twenty-eight scenarios, including the five ids the app requires. |
| `reference_results.csv` | Exact three-year term metrics for every scenario. |
| `invalid_records.csv` | Sixteen held-out bad rows for a validation demo. `include_in_calculations = no`. |
| `data_dictionary.csv` | Column definitions. |
| `quantumrisk_demo_bundle.xlsx` | Uploadable workbook for the baseline book and the scenarios that the current app can run without a further edit. |

## Relationships

- Country insured exposure equals earthquake principal plus precipitation principal.
- Regional exposures sum to that country total.
- A catalogue event draws occurrence with the same Marshall-Olkin common shock as the pricing model (baseline share 0.30) and the same conditional severities.
- The master payout for a catalogue year uses the contract steps. Earthquake pays the severity step, capped at the earthquake cap. Precipitation pays the severity step. The country cap is then applied, and any haircut is shared in proportion to the two peril payments.
- That country-hazard payout is allocated to impact rows in proportion to ground-up loss. Impact payouts in one event group sum to the contractual payout.
- Reference metrics are not averages of the catalogue. They are the three-year term calculation: earthquake steps accumulate and cap, precipitation pays once at the highest step, then the country cap applies. Portfolio loss is the sum of the two countries. Thresholds are strict: loss greater than T.

## Baseline contract

- Kyrgyz Republic: earthquake principal 80,000,000 USD, precipitation principal 16,000,000 USD, country cap 88,000,000 USD. Occurrence 0.12 earthquake and 0.14 precipitation.
- Tajikistan: earthquake principal 50,000,000 USD, precipitation principal 18,000,000 USD, country cap 60,000,000 USD. Occurrence 0.09 earthquake and 0.20 precipitation.
- Earthquake steps are 15/40/100 percent in the Kyrgyz Republic and 20/55/100 percent in Tajikistan.
- Precipitation steps are 40/100 percent and 50/100 percent.
- Term is 3 years. Currency is USD.

## Findings from this seed

- Insured exposure is 96,000,000 USD in the Kyrgyz Republic and 68,000,000 USD in Tajikistan. The Kyrgyz book is larger.
- The largest regional concentration is Bishkek basin (Kyrgyz Republic) at 34.0% of its country.
- In the catalogue, earthquake contractual payouts are 124,000,000 USD and precipitation contractual payouts are 247,800,000 USD.
- Catalogue payout status counts: {'coverage_exhausted': 4, 'no_payout': 212, 'partial_payout': 444}.
- Triggered joint-country hazard years: 5. Triggered single-country hazard years: 18. Quiet national rows are counted in the no-payout status, not here.
- Baseline expected loss is 23,436,842 USD. The worst-case stress expected loss is 67,346,883 USD. The quiet book is 2,015,817 USD.
- Joint attachment is 0.348 when the countries are independent, 0.387 at the 0.30 common shock, and 0.516 at the 0.95 common shock. Expected loss is the same in those three, because the marginals do not change.
- At a 10 million threshold the baseline exceedance is 0.625. At 25 million it is 0.332. Expected loss is unchanged.

Highest expected loss:

- Worst-case stress: 67,346,883 USD (extreme).
- Rare extreme catastrophe: 54,567,060 USD (extreme).
- Both contracts exhausted: 54,246,870 USD (extreme).
- Full payout schedule: 50,521,324 USD (extreme).
- High losses: 40,349,014 USD (extreme).

## How to use it in the app

On the Data page, upload `quantumrisk_demo_bundle.xlsx` as the bundle. The workbook sheets are `hazards`, `dependence`, `exposure`, `contracts` and `scenarios`. Invalid examples are not in that workbook.

Scenarios that change only the shock share, the occurrence scales or the precipitation steps are in the workbook. Country-specific rates, resized principals, tighter caps and earthquake step changes are enumerated in `reference_results.csv` and marked `app_bundle = no` in `scenario_data.csv`, because one uploaded scenario table cannot retarget a single country or replace the contract.

Default comparison threshold in the reference file is 25,000,000 USD, except the portfolio-threshold scenario, which uses 10,000,000 USD.
