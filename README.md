# QuantumRisk Analytics

Local research demo for catastrophe-risk scenarios. It loads a synthetic insurance book, turns earthquake and extreme-precipitation events into capped contract payouts, and estimates portfolio loss with classical simulation and a separate tensor-train check. Results appear on a local dashboard, with history and CSV, JSON and PDF exports.

The application demonstrates the calculation and the comparison of methods. It does not price, bind or pay a real policy.

## What it does

- Opens a local site at http://127.0.0.1:5000. On startup it loads `data/demo/quantumrisk_demo_bundle.xlsx` when that file is present. That book covers the Kyrgyz Republic and Tajikistan, earthquake and extreme precipitation, regional exposure, a common-shock dependence setting, and a three-year USD contract.
- Accepts a replacement book as CSV, Excel or JSON. A workbook or JSON object can replace the whole set of tables. Invalid files are rejected and the previous valid book stays in place.
- Builds a Marshall–Olkin common-shock model that keeps each country's own chance of a hazard. Earthquake and precipitation are treated as independent of each other. Years are independent.
- Converts severities into stepwise payouts. Earthquake can pay in each year, then the earthquake cap applies. Precipitation pays once per term, at the highest step reached. Each country loss is then capped, and the portfolio loss is the sum of the country losses.
- Scores five metrics: expected portfolio loss, the probability that both countries attach, the probability that portfolio loss is strictly greater than a threshold T, expected loss above T, and the probability that every contract reaches its cap. The default threshold is 25,000,000 USD.
- Runs seven estimators on the same book: exact enumeration, Monte Carlo, scrambled Sobol, stratified sampling, importance sampling, a tail-aware proposal aimed at rare joint disasters, and an experimental tensor train (a matrix-product state) built with Quimb. Cotengra chooses the contraction order. The tensor train is a classical calculation on this computer.
- Lets an administrator add scenarios that change dependence, the shared-shock share, and the earthquake or precipitation occurrence scale. A run can also override the shock share and the precipitation scale for that job.
- Compares every estimator on one scenario, or one estimator across scenarios. Charts and tables use the numbers from that run.
- Keeps each job in SQLite, including a snapshot of the tables, the seed, the sample budget, library versions, runtime and memory. A later upload does not change a finished run. A local background thread executes queued jobs.
- Signs users in with a local account and one of three roles:
  - **Administrator** can replace the book, add scenarios, run simulations, and create accounts.
  - **Analyst** can run simulations and comparisons, and can read the book.
  - **Reviewer** can read results, charts, history and exports.

The sign-in page lists the local demonstration accounts. They are created only when the account table is empty.

Tests use the smaller locked fixture in `data/synthetic/`, checked against `data/reference/reference_results.csv`. Starting the application with `python app.py` uses the larger demo workbook, not that fixture.

## Limitations

- Every shipped input is a synthetic demonstration assumption. The files are not observed disaster rates, not a real insurance portfolio, and not an official payout schedule or external benchmark. Catalogue rows in `data/demo/` are a drawn illustration. The reference metrics are the term calculation, not an average of that catalogue.
- A stochastic result is an estimate under the stated assumptions. It is not a guaranteed payout. A reported standard error describes that one run. A band of 1.96 standard errors is an approximate interval for the estimator, not a range of possible claim payments.
- Exact enumeration is the reference on the small discrete model. It is refused when the annual probability table has more than 500 entries. The tensor-train term metrics stop at the same size. Sampling can still run. A table larger than 300,000 entries is rejected at upload.
- The loader accepts at most 6 countries, 4 hazards and a term of 5 years. Hazards are the discrete earthquake and precipitation steps in the tables. The engine does not add further perils, continuous loss sizes, compound Poisson counts, copulas, or an exceedance curve supplied as an input.
- Scenario controls apply to the whole book. The demo folder also contains country-specific, exposure, fraction and tighter-cap cases. Those stay in `data/demo/scenario_data.csv` and `data/demo/reference_results.csv`. They are not in the workbook the application runs.
- Contract layers in the demo file illustrate attachment. The calculation uses the master steps and caps. It does not price a separate excess contract on top of the master payout.
- Because each country's loss depends only on its own hazard rates, raising the shared-shock share does not change expected portfolio loss. It does change joint attachment, exceedance, stop-loss and the chance that every contract is exhausted.
- The tensor train is a feasibility check. A full-bond result is shown only when it reproduces the annual table. On this small model that match is a validity check. It is not evidence that the method is faster or more accurate than enumeration, and it is not a quantum-advantage claim. No quantum hardware or quantum circuit is used.
- Accounts live in the same local SQLite file as the jobs. Passwords are for this machine. There is no organisation directory, approval workflow, model-drift monitor, cloud worker or separate production database.
- The site binds to this computer only (`127.0.0.1`, port 5000). It is a research dashboard, not an underwriting or claims system.

## Run it

Python 3.10 or newer. On this machine the default `python` may be older, so create the environment with the 3.11 launcher:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
python app.py
```

Then open http://127.0.0.1:5000 and sign in.

`pytest` checks validation, payout caps, the locked reference numbers, sampling error against exact enumeration, likelihood-ratio weights, the tensor-train reconstruction, file upload, background jobs, exports, and the role checks.
