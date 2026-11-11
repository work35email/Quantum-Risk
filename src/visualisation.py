"""Plotly charts for the local dashboard.

Chart choice follows the data: columns for rankings, lines for exceedance and
approximation error, a donut for exposure share, a stacked column for contract
composition, a heatmap for severity probabilities, and a box plot only for
samples drawn from the target distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.risk_metrics import METRIC_LABELS, MONEY_METRICS, RunResult, weighted_histogram

NAVY = "#0f2744"
BLUE = "#1d4e89"
SERIES = ("#0f2744", "#1d4e89", "#3d6f99", "#1f6b6b", "#6d8299", "#5c4d3a")
TARGET_SAMPLE_METHODS = {"monte_carlo", "sobol"}


def loss_distribution_figure(result: RunResult, portfolio_cap: float) -> go.Figure:
    """Histogram of portfolio loss. Exact runs use the probability mass directly."""
    figure = go.Figure()
    if result.loss_pmf is not None:
        losses, probability = result.loss_pmf
        figure.add_bar(
            x=losses,
            y=probability,
            name="Probability",
            marker_color=NAVY,
            hovertemplate="Loss %{x:,.0f}<br>Probability %{y:.4f}<extra></extra>",
        )
        figure.update_layout(yaxis_title="Probability", bargap=0.12)
        caption = "Exact probability mass of portfolio loss."
    elif result.sample_losses is not None and len(result.sample_losses):
        centres, heights = weighted_histogram(result.sample_losses, result.sample_weights, portfolio_cap)
        figure.add_bar(
            x=centres,
            y=heights,
            name="Estimated probability",
            marker_color=NAVY,
            hovertemplate="Loss %{x:,.0f}<br>Estimated probability %{y:.4f}<extra></extra>",
        )
        figure.update_layout(yaxis_title="Estimated probability", bargap=0.05)
        caption = _sample_caption(result)
    else:
        figure.add_annotation(text="No distribution was stored for this run", showarrow=False)
        caption = "This run did not produce a loss distribution."
    figure.update_layout(
        title=_title("Portfolio loss distribution", caption),
        xaxis_title=f"Portfolio loss ({result.currency})",
        showlegend=False,
    )
    return _theme(figure)


def exceedance_figure(result: RunResult) -> go.Figure | None:
    """Survival curve P(L > x) from the stored distribution. Not a new metric."""
    curve = _survival(result)
    if curve is None:
        return None
    losses, survival = curve
    figure = go.Figure(
        go.Scatter(
            x=losses,
            y=survival,
            mode="lines",
            name="P(L > x)",
            line={"color": BLUE, "width": 2},
            hovertemplate="Loss %{x:,.0f}<br>P(L > x) %{y:.4f}<extra></extra>",
        )
    )
    if result.threshold is not None and np.isfinite(result.threshold):
        figure.add_vline(
            x=float(result.threshold),
            line_dash="dot",
            line_color="#8a5a12",
            annotation_text="Threshold",
            annotation_position="top right",
            annotation_font_color="#8a5a12",
        )
    note = "Strict exceedance read from the stored distribution."
    if result.method in {"importance_sampling", "tail_aware"}:
        note = "Weighted empirical curve for display. The reported probability uses the unbiased ratio."
    figure.update_layout(
        title=_title("Portfolio exceedance curve", note),
        xaxis_title=f"Portfolio loss ({result.currency})",
        yaxis_title="P(L > x)",
        yaxis={"range": [0, 1.02]},
        showlegend=False,
    )
    return _theme(figure)


def country_loss_figure(result: RunResult) -> go.Figure:
    """Horizontal comparison of expected loss by country."""
    names = list(result.country_expected_loss)
    means = [result.country_expected_loss[name] for name in names]
    errors = [1.96 * result.country_expected_loss_se.get(name, 0.0) for name in names]
    figure = go.Figure(
        go.Bar(
            y=names,
            x=means,
            orientation="h",
            error_x={"type": "data", "array": errors, "visible": True, "color": "#5c6b80"},
            marker_color=NAVY,
            name="Expected loss",
            hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=_title("Expected loss by country", "Interval is 1.96 standard errors when the estimator has one."),
        xaxis_title=f"Expected loss ({result.currency})",
        showlegend=False,
    )
    figure.update_yaxes(autorange="reversed")
    return _theme(figure, height=max(280, 64 * max(len(names), 1) + 120))


def comparison_figure(results: list[RunResult], metric: str) -> go.Figure:
    """Rank one metric across estimators. Invalid experiments are omitted."""
    return _ranking(results, metric, label=lambda result: result.method_label, subtitle="Estimators on one scenario.")


def scenario_comparison_figure(results: list[RunResult], metric: str) -> go.Figure:
    """Rank one metric across scenarios for a single estimator."""
    return _ranking(results, metric, label=lambda result: result.scenario_name, subtitle="One estimator across scenarios.")


def probability_comparison_figure(results: list[RunResult], label) -> go.Figure | None:
    """Grouped columns for the three probability metrics."""
    usable = [result for result in results if result.valid_estimate]
    if len(usable) < 2:
        return None
    keys = (
        ("joint_attachment_probability", "Joint attachment"),
        ("exceedance_probability", "Exceedance"),
        ("both_exhausted_probability", "Both exhausted"),
    )
    figure = go.Figure()
    labels = [label(result) for result in usable]
    for index, (key, name) in enumerate(keys):
        figure.add_bar(
            name=name,
            x=labels,
            y=[float(result.metrics.get(key) or 0.0) for result in usable],
            marker_color=SERIES[index % len(SERIES)],
            hovertemplate="%{x}<br>" + name + " %{y:.4f}<extra></extra>",
        )
    figure.update_layout(
        barmode="group",
        title=_title("Probability metrics", "Joint attachment, threshold exceedance and contract exhaustion."),
        yaxis_title="Probability",
        yaxis={"range": [0, 1.05]},
        legend={"orientation": "h", "y": 1.14, "x": 0},
    )
    return _theme(figure, height=380)


def scenario_relationship_figure(results: list[RunResult]) -> go.Figure | None:
    """Expected loss against exceedance. Each point is one completed scenario."""
    usable = [
        result
        for result in results
        if result.valid_estimate and "expected_loss" in result.metrics and "exceedance_probability" in result.metrics
    ]
    if len(usable) < 2:
        return None
    figure = go.Figure(
        go.Scatter(
            x=[float(result.metrics["exceedance_probability"]) for result in usable],
            y=[float(result.metrics["expected_loss"]) for result in usable],
            mode="markers",
            name="Scenario",
            text=[result.scenario_name for result in usable],
            marker={"size": 11, "color": NAVY, "line": {"color": "#ffffff", "width": 1}},
            hovertemplate="%{text}<br>Exceedance %{x:.4f}<br>Expected loss %{y:,.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=_title("Expected loss and exceedance", "Each point is one scenario under the selected estimator."),
        xaxis_title="Portfolio exceedance probability",
        yaxis_title="Expected loss",
        showlegend=False,
    )
    return _theme(figure)


def sample_spread_figure(result: RunResult) -> go.Figure | None:
    """Box plot of target-distribution samples. Proposal samples are excluded."""
    if result.method not in TARGET_SAMPLE_METHODS:
        return None
    if result.sample_losses is None or len(result.sample_losses) < 8:
        return None
    figure = go.Figure(
        go.Box(
            y=result.sample_losses,
            name="Portfolio loss",
            marker_color=NAVY,
            line_color=NAVY,
            fillcolor="#e7eef6",
            hovertemplate="Loss %{y:,.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=_title("Sample loss spread", "Drawn from the target distribution. Quartiles and outliers of this run."),
        yaxis_title=f"Portfolio loss ({result.currency})",
        showlegend=False,
    )
    return _theme(figure, height=320)


def truncation_figure(result: RunResult) -> go.Figure | None:
    """Bond-dimension approximation error. This is not a risk estimate."""
    rows = result.details.get("truncation")
    if not rows:
        return None
    labels = [str(row["max_bond"]) for row in rows] + ["full"]
    errors = [row["l1_error"] for row in rows] + [result.details.get("l1_error", 0.0)]
    figure = go.Figure(
        go.Scatter(
            x=labels,
            y=errors,
            mode="lines+markers",
            name="L1 error",
            line={"color": BLUE, "width": 2},
            marker={"size": 8, "color": NAVY},
            hovertemplate="Bond %{x}<br>L1 error %{y:.3e}<extra></extra>",
        )
    )
    figure.update_layout(
        title=_title("Tensor-train approximation error", "Reconstruction error by bond dimension. Not a risk metric."),
        xaxis_title="Maximum bond dimension",
        yaxis_title="L1 reconstruction error",
        showlegend=False,
    )
    return _theme(figure)


def exposure_share_figure(exposure: pd.DataFrame) -> go.Figure | None:
    """Share of insured exposure by country."""
    if exposure is None or exposure.empty or "insured_exposure" not in exposure.columns:
        return None
    labels = [str(value) for value in exposure["country"]]
    values = [float(value) for value in exposure["insured_exposure"]]
    figure = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            sort=False,
            marker={"colors": list(SERIES[: len(labels)])},
            textinfo="percent",
            hovertemplate="%{label}<br>%{value:,.0f}<br>%{percent}<extra></extra>",
        )
    )
    figure.update_layout(title=_title("Insured exposure by country", "Share of the loaded exposure table."), showlegend=True)
    figure = _theme(figure, height=340)
    figure.update_layout(
        legend={"orientation": "h", "y": -0.15, "x": 0},
        margin={"t": 88, "r": 24, "b": 72, "l": 24},
    )
    return figure


def occurrence_figure(hazards: pd.DataFrame) -> go.Figure | None:
    """Grouped columns of annual occurrence probability by hazard and country."""
    required = {"country", "hazard", "annual_occurrence_probability"}
    if hazards is None or hazards.empty or not required.issubset(hazards.columns):
        return None
    frame = hazards.copy()
    frame["country"] = frame["country"].astype(str)
    frame["hazard"] = frame["hazard"].astype(str).str.strip().str.lower()
    summary = (
        frame.groupby(["hazard", "country"], sort=False)["annual_occurrence_probability"].first().reset_index()
    )
    hazards_order = list(dict.fromkeys(summary["hazard"]))
    countries = list(dict.fromkeys(summary["country"]))
    figure = go.Figure()
    for index, country in enumerate(countries):
        subset = summary[summary["country"] == country].set_index("hazard")
        figure.add_bar(
            name=country,
            x=[hazard.replace("_", " ") for hazard in hazards_order],
            y=[float(subset.loc[hazard, "annual_occurrence_probability"]) if hazard in subset.index else 0.0 for hazard in hazards_order],
            marker_color=SERIES[index % len(SERIES)],
            hovertemplate="%{x}<br>%{y:.4f}<extra>" + country + "</extra>",
        )
    figure.update_layout(
        barmode="group",
        title=_title("Annual occurrence probability", "One bar per country. Conditional severities are not included."),
        yaxis_title="Annual probability",
        xaxis_title="Hazard",
    )
    return _theme(figure)


def contract_composition_figure(contracts: pd.DataFrame) -> go.Figure | None:
    """Stacked principal by country and peril."""
    if contracts is None or contracts.empty or "country" not in contracts.columns:
        return None
    columns = [column for column in contracts.columns if str(column).endswith("_principal") or str(column).endswith("principal")]
    columns = [column for column in columns if pd.api.types.is_numeric_dtype(contracts[column])]
    if not columns:
        return None
    figure = go.Figure()
    countries = [str(value) for value in contracts["country"]]
    for index, column in enumerate(columns):
        label = str(column).replace("_", " ").replace("principal", "").strip().title() or str(column)
        figure.add_bar(
            name=label,
            x=countries,
            y=[float(value) for value in contracts[column]],
            marker_color=SERIES[index % len(SERIES)],
            hovertemplate="%{x}<br>" + label + " %{y:,.0f}<extra></extra>",
        )
    figure.update_layout(
        barmode="stack",
        title=_title("Contract principal by peril", "Stacked from the loaded contract table. Caps are not included."),
        yaxis_title="Principal",
        xaxis_title="Country",
    )
    return _theme(figure)


def severity_heatmap(hazards: pd.DataFrame) -> go.Figure | None:
    """Conditional severity probability. Blank cells are severities that are not defined."""
    required = {"country", "hazard", "severity_given_event", "conditional_probability"}
    if hazards is None or hazards.empty or not required.issubset(hazards.columns):
        return None
    frame = hazards.copy()
    frame["row"] = frame["country"].astype(str) + " — " + frame["hazard"].astype(str).str.strip().str.lower()
    frame["severity"] = frame["severity_given_event"].astype(int)
    rows = list(dict.fromkeys(frame["row"]))
    severities = sorted(set(int(value) for value in frame["severity"]))
    lookup = {
        (str(record.row), int(record.severity)): float(record.conditional_probability)
        for record in frame.itertuples(index=False)
    }
    matrix = [[lookup.get((row, severity), np.nan) for severity in severities] for row in rows]
    text = [["" if not np.isfinite(value) else f"{value:.2f}" for value in line] for line in matrix]
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[str(severity) for severity in severities],
            y=rows,
            text=text,
            texttemplate="%{text}",
            colorscale=[[0.0, "#f4f7fb"], [1.0, NAVY]],
            zmin=0,
            zmax=1,
            colorbar={"title": "Probability", "tickformat": ".1f"},
            hovertemplate="%{y}<br>Severity %{x}<br>%{z:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=_title("Conditional severity probability", "Probability of each severity given that the hazard occurs."),
        xaxis_title="Severity given the event",
        yaxis_title="",
    )
    figure.update_yaxes(autorange="reversed")
    return _theme(figure, height=max(320, 48 * len(rows) + 140))


def portfolio_cap(result: RunResult) -> float:
    if result.loss_pmf is not None and len(result.loss_pmf[0]):
        return float(max(result.loss_pmf[0].max(), 1.0))
    if result.sample_losses is not None and len(result.sample_losses):
        return float(max(np.max(result.sample_losses), 1.0))
    return 1.0


def _ranking(results: list[RunResult], metric: str, label, subtitle: str) -> go.Figure:
    usable = [result for result in results if result.valid_estimate and metric in result.metrics]
    labels = [label(result) for result in usable]
    values = [result.metrics[metric] for result in usable]
    errors = [1.96 * float(result.uncertainty.get(metric) or 0.0) for result in usable]
    money = metric in MONEY_METRICS
    hover = "%{y}<br>%{x:,.0f}<extra></extra>" if money else "%{y}<br>%{x:.4f}<extra></extra>"
    figure = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            error_x={"type": "data", "array": errors, "visible": True, "color": "#5c6b80"},
            marker_color=NAVY,
            hovertemplate=hover,
        )
    )
    figure.update_layout(
        title=_title(METRIC_LABELS.get(metric, metric), subtitle),
        xaxis_title="USD" if money else "Probability",
        showlegend=False,
    )
    figure.update_yaxes(autorange="reversed")
    return _theme(figure, height=max(300, 42 * max(len(labels), 1) + 120))


def _survival(result: RunResult) -> tuple[np.ndarray, np.ndarray] | None:
    if result.loss_pmf is not None and len(result.loss_pmf[0]):
        losses = np.asarray(result.loss_pmf[0], dtype=float)
        probability = np.asarray(result.loss_pmf[1], dtype=float)
    elif result.sample_losses is not None and len(result.sample_losses):
        losses = np.asarray(result.sample_losses, dtype=float)
        weights = result.sample_weights
        if weights is None:
            probability = np.full(len(losses), 1.0 / len(losses))
        else:
            probability = np.asarray(weights, dtype=float)
            total = float(probability.sum())
            if total <= 0:
                return None
            probability = probability / total
    else:
        return None
    order = np.argsort(losses)
    losses = losses[order]
    probability = probability[order]
    unique, inverse = np.unique(losses, return_inverse=True)
    mass = np.zeros(len(unique), dtype=float)
    np.add.at(mass, inverse, probability)
    survival = np.cumsum(mass[::-1])[::-1] - mass
    return unique, survival


def _sample_caption(result: RunResult) -> str:
    if result.method in {"importance_sampling", "tail_aware"}:
        return "Display histogram uses self-normalised weights. Reported metrics use unbiased ratios."
    if result.method == "stratified":
        return "Histogram uses stratified mixture weights, matching the reported estimator."
    if result.method == "sobol":
        return "Empirical distribution of the scrambled Sobol term scenarios."
    return "Empirical distribution of the simulated terms."


def _title(heading: str, subtitle: str) -> dict:
    return {
        "text": f"{heading}<br><span style='font-size:12px;color:#5c6b80;font-weight:400'>{subtitle}</span>",
        "x": 0,
        "xanchor": "left",
    }


def _theme(figure: go.Figure, height: int = 340) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Segoe UI, Helvetica Neue, sans-serif", "color": "#142033", "size": 12},
        title={"font": {"size": 15, "color": NAVY}},
        margin={"t": 88, "r": 20, "b": 48, "l": 56},
        height=height,
        legend={"orientation": "h", "y": 1.16, "x": 0, "font": {"size": 11}},
        hoverlabel={"bgcolor": "#ffffff", "bordercolor": "#d5deea", "font": {"color": "#142033", "size": 12}},
    )
    figure.update_xaxes(gridcolor="#e6edf5", zerolinecolor="#e6edf5", linecolor="#c5d0de", tickfont={"size": 11})
    figure.update_yaxes(gridcolor="#e6edf5", zerolinecolor="#e6edf5", linecolor="#c5d0de", tickfont={"size": 11})
    return figure
