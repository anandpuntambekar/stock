#!/usr/bin/env python3
"""Streamlit UI for the Nifty 50 EOD strategy research tool."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


DEFAULT_SYMBOL = "^NSEI"


@dataclass
class StrategyInputs:
    window_months: int
    buy_threshold: float
    sell_threshold: float
    prob_threshold: float
    bin_size: float


def download_nifty_data(start: str, end: str | None, symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    data = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        data = data.loc[:, ~data.columns.duplicated()]
    data = data.rename(
        columns={
            "Open": "BOD",
            "Close": "EOD",
            "High": "High",
            "Low": "Low",
        }
    )
    data = data[["BOD", "EOD", "High", "Low", "Volume"]].dropna()
    return data


@st.cache_data(show_spinner=False)
def load_data(start: str, end: str | None) -> pd.DataFrame:
    return download_nifty_data(start=start, end=end)


def median_top_five(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    sorted_values = np.sort(values)
    top_five = sorted_values[-5:]
    return float(np.median(top_five))


def median_bottom_five(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    sorted_values = np.sort(values)
    bottom_five = sorted_values[:5]
    return float(np.median(bottom_five))


def rolling_median(series: pd.Series, window_days: int, fn) -> pd.Series:
    return series.rolling(window_days, min_periods=5).apply(fn, raw=True)


def percent_from_reference(close: pd.Series, reference: pd.Series) -> pd.Series:
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(reference, pd.DataFrame):
        reference = reference.iloc[:, 0]
    return (close - reference) / reference * 100.0


def probability_bins(series: pd.Series, bin_size: float) -> List[Tuple[float, float, float]]:
    if series.empty:
        return []
    min_bin = np.floor(series.min() / bin_size) * bin_size
    max_bin = np.ceil(series.max() / bin_size) * bin_size
    bins = np.arange(min_bin, max_bin + bin_size, bin_size)
    binned = pd.cut(series, bins=bins, include_lowest=True)
    counts = binned.value_counts().sort_index()
    probabilities = counts / counts.sum()

    results: List[Tuple[float, float, float]] = []
    for interval, prob in probabilities.items():
        results.append((float(interval.left), float(interval.right), float(prob)))
    return results


def compute_probability_chart(series: pd.Series, window_days: int, bin_size: float) -> List[List[Tuple[float, float, float]]]:
    charts: List[List[Tuple[float, float, float]]] = []
    for idx in range(len(series)):
        window = series.iloc[max(0, idx - window_days + 1) : idx + 1].dropna()
        charts.append(probability_bins(window, bin_size))
    return charts


def build_features(data: pd.DataFrame, window_months: int, bin_size: float) -> pd.DataFrame:
    window_days = window_months * 21
    enriched = data.copy()
    enriched["HighMax"] = enriched["High"].rolling(window_days, min_periods=1).max()
    enriched["LowMin"] = enriched["Low"].rolling(window_days, min_periods=1).min()
    enriched["HighMedianTop5"] = rolling_median(enriched["High"], window_days, median_top_five)
    enriched["LowMedianBottom5"] = rolling_median(enriched["Low"], window_days, median_bottom_five)
    enriched["PctFromHighMedian"] = percent_from_reference(enriched["EOD"], enriched["HighMedianTop5"])
    enriched["PctFromLowMedian"] = percent_from_reference(enriched["EOD"], enriched["LowMedianBottom5"])
    enriched["ProbChartHigh"] = compute_probability_chart(
        enriched["PctFromHighMedian"], window_days, bin_size
    )
    enriched["ProbChartLow"] = compute_probability_chart(
        enriched["PctFromLowMedian"], window_days, bin_size
    )
    return enriched


def select_row_by_date(data: pd.DataFrame, target: date) -> pd.Series:
    matches = data.loc[data.index.date == target]
    if matches.empty:
        raise ValueError("Selected date not found in data.")
    return matches.iloc[-1]


def probability_below(chart: List[Tuple[float, float, float]], threshold: float) -> float:
    return float(np.sum([prob for low, _high, prob in chart if low <= threshold])) if chart else 0.0


def probability_above(chart: List[Tuple[float, float, float]], threshold: float) -> float:
    return float(np.sum([prob for low, _high, prob in chart if low >= threshold])) if chart else 0.0


def decide_signal(
    pct_from_high: float,
    pct_from_low: float,
    chart_high: List[Tuple[float, float, float]],
    chart_low: List[Tuple[float, float, float]],
    inputs: StrategyInputs,
) -> str:
    if np.isnan(pct_from_high) or np.isnan(pct_from_low):
        return "HOLD"

    prob_drop = probability_below(chart_high, inputs.buy_threshold)
    prob_rise = probability_above(chart_low, inputs.sell_threshold)

    if pct_from_high <= inputs.buy_threshold and prob_drop >= inputs.prob_threshold:
        return "BUY"
    if pct_from_low >= inputs.sell_threshold and prob_rise >= inputs.prob_threshold:
        return "SELL"
    return "HOLD"


def chart_to_frame(chart: List[Tuple[float, float, float]]) -> pd.DataFrame:
    rows = [
        {
            "Range": f"{low:.1f}% to {high:.1f}%",
            "Probability": prob,
        }
        for low, high, prob in chart
    ]
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Nifty 50 EOD Strategy", layout="wide")
    st.title("Nifty 50 End-of-Day Strategy Explorer")

    with st.sidebar:
        st.header("Data")
        start = st.date_input("Start date", value=date.today() - timedelta(days=365 * 15))
        end = st.date_input("End date", value=date.today() - timedelta(days=1))
        window_months = st.slider("Window (months)", min_value=2, max_value=6, value=3)
        bin_size = st.slider("Probability bin size (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

        st.header("Decision thresholds")
        buy_threshold = st.slider("Buy threshold (% from high)", -10.0, -0.5, -3.0, step=0.5)
        sell_threshold = st.slider("Sell threshold (% from low)", 0.5, 10.0, 3.0, step=0.5)
        prob_threshold = st.slider("Probability threshold", 0.05, 0.5, 0.15, step=0.05)

    data = load_data(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat())
    features = build_features(data, window_months, bin_size)

    selected_date = st.date_input(
        "Select a date", value=features.index[-1].date(), min_value=features.index[0].date(), max_value=features.index[-1].date()
    )
    selected_row = select_row_by_date(features, selected_date)

    inputs = StrategyInputs(
        window_months=window_months,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        prob_threshold=prob_threshold,
        bin_size=bin_size,
    )

    signal = decide_signal(
        selected_row["PctFromHighMedian"],
        selected_row["PctFromLowMedian"],
        selected_row["ProbChartHigh"],
        selected_row["ProbChartLow"],
        inputs,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Signal", signal)
    col2.metric("EOD Close", f"{selected_row['EOD']:.2f}")
    col3.metric("BOD Open", f"{selected_row['BOD']:.2f}")

    st.subheader("Window statistics")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    stats_col1.metric("High (window)", f"{selected_row['HighMax']:.2f}")
    stats_col2.metric("Low (window)", f"{selected_row['LowMin']:.2f}")
    stats_col3.metric("Median of top 5 highs", f"{selected_row['HighMedianTop5']:.2f}")
    stats_col4.metric("Median of bottom 5 lows", f"{selected_row['LowMedianBottom5']:.2f}")

    prob_drop = probability_below(selected_row["ProbChartHigh"], buy_threshold)
    prob_rise = probability_above(selected_row["ProbChartLow"], sell_threshold)

    st.subheader("Probability summary")
    prob_col1, prob_col2 = st.columns(2)
    prob_col1.metric("Probability of drop (from high median)", f"{prob_drop:.2%}")
    prob_col2.metric("Probability of rise (from low median)", f"{prob_rise:.2%}")

    tabs = st.tabs(["Probability Charts", "Decision Logic", "Raw Data"])

    with tabs[0]:
        st.markdown("### From High Median")
        chart_high = chart_to_frame(selected_row["ProbChartHigh"])
        st.bar_chart(chart_high.set_index("Range")["Probability"])

        st.markdown("### From Low Median")
        chart_low = chart_to_frame(selected_row["ProbChartLow"])
        st.bar_chart(chart_low.set_index("Range")["Probability"])

    with tabs[1]:
        st.markdown(
            """
            **How the probabilities are prepared**

            1. Pick a rolling window (default 3 months, ~21 trading days per month).
            2. Compute the median of the **top 5 highs** and **bottom 5 lows** in that window.
            3. Convert today’s close into percent change from those medians.
            4. Build a rolling probability chart using 1% bins over the same window.

            **Decision rules**
            - **BUY** if percent from high median <= buy threshold **and** the probability of being below
              that threshold is above the probability threshold.
            - **SELL** if percent from low median >= sell threshold **and** the probability of being above
              that threshold is above the probability threshold.
            - Otherwise **HOLD**.
            """
        )

    with tabs[2]:
        st.dataframe(features.tail(200))


if __name__ == "__main__":
    main()
