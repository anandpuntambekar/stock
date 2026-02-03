#!/usr/bin/env python3
"""Nifty 50 end-of-day strategy research tool.

This script downloads daily OHLC data, builds a rolling 3–5 month
probability chart of percent change from a rolling high, and produces
buy/sell/hold signals at end of day.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_SYMBOL = "^NSEI"


@dataclass
class StrategyConfig:
    window_months: int = 3
    buy_threshold: float = -3.0
    sell_threshold: float = -1.0
    prob_threshold: float = 0.15
    bin_size: float = 1.0


@dataclass
class BacktestConfig:
    fit_years: int = 5
    test_years: int = 1
    start_cash: float = 1_000_000.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    cagr: float


def download_nifty_data(
    start: str,
    end: str | None = None,
    symbol: str = DEFAULT_SYMBOL,
) -> pd.DataFrame:
    data = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False)
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


def median_top_five(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    sorted_values = np.sort(values)
    top_five = sorted_values[-5:]
    return float(np.median(top_five))


def rolling_high_median(data: pd.Series, window_days: int) -> pd.Series:
    return data.rolling(window_days, min_periods=5).apply(median_top_five, raw=True)


def percent_from_high(close: pd.Series, rolling_high: pd.Series) -> pd.Series:
    return (close - rolling_high) / rolling_high * 100.0


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


def compute_probability_chart(
    data: pd.DataFrame,
    window_days: int,
    bin_size: float,
) -> List[List[Tuple[float, float, float]]]:
    charts: List[List[Tuple[float, float, float]]] = []
    pct_from_high = data["PctFromHigh"]
    for idx in range(len(pct_from_high)):
        window = pct_from_high.iloc[max(0, idx - window_days + 1) : idx + 1].dropna()
        charts.append(probability_bins(window, bin_size))
    return charts


def generate_signals(
    data: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    window_days = config.window_months * 21
    data = data.copy()
    data["RollingHigh"] = rolling_high_median(data["High"], window_days)
    data["PctFromHigh"] = percent_from_high(data["EOD"], data["RollingHigh"])
    data["ProbChart"] = compute_probability_chart(data, window_days, config.bin_size)

    signals = []
    for pct, chart in zip(data["PctFromHigh"], data["ProbChart"]):
        if np.isnan(pct) or not chart:
            signals.append("HOLD")
            continue
        lower_bins = [prob for low, _high, prob in chart if low <= config.buy_threshold]
        prob_low = float(np.sum(lower_bins))

        if pct <= config.buy_threshold and prob_low >= config.prob_threshold:
            signals.append("BUY")
        elif pct >= config.sell_threshold:
            signals.append("SELL")
        else:
            signals.append("HOLD")

    data["Signal"] = signals
    return data


def simulate_backtest(
    data: pd.DataFrame,
    config: StrategyConfig,
    backtest_config: BacktestConfig,
) -> BacktestResult:
    data = generate_signals(data, config)
    cash = backtest_config.start_cash
    shares = 0.0
    trades = []
    equity_curve = []

    for idx in range(1, len(data)):
        today = data.iloc[idx]
        yesterday = data.iloc[idx - 1]
        signal = yesterday["Signal"]
        trade_price = today["BOD"]

        if signal == "BUY" and shares == 0:
            shares = cash / trade_price
            cash = 0.0
            trades.append({"Date": today.name, "Action": "BUY", "Price": trade_price})
        elif signal == "SELL" and shares > 0:
            cash = shares * trade_price
            shares = 0.0
            trades.append({"Date": today.name, "Action": "SELL", "Price": trade_price})

        equity = cash + shares * today["EOD"]
        equity_curve.append(equity)

    equity_series = pd.Series(equity_curve, index=data.index[1:])
    total_years = (equity_series.index[-1] - equity_series.index[0]).days / 365.25
    cagr = (equity_series.iloc[-1] / backtest_config.start_cash) ** (1 / total_years) - 1

    return BacktestResult(
        equity_curve=equity_series,
        trades=pd.DataFrame(trades),
        cagr=cagr,
    )


def train_and_test(
    data: pd.DataFrame,
    config: StrategyConfig,
    backtest_config: BacktestConfig,
) -> Tuple[BacktestResult, BacktestResult]:
    cutoff = data.index[-1] - pd.DateOffset(years=backtest_config.test_years)
    train_start = cutoff - pd.DateOffset(years=backtest_config.fit_years)
    train_data = data.loc[train_start:cutoff]
    test_data = data.loc[cutoff:]

    train_result = simulate_backtest(train_data, config, backtest_config)
    test_result = simulate_backtest(test_data, config, backtest_config)

    return train_result, test_result


def grid_search(
    data: pd.DataFrame,
    window_months: Iterable[int],
    buy_thresholds: Iterable[float],
    sell_thresholds: Iterable[float],
    prob_thresholds: Iterable[float],
    backtest_config: BacktestConfig,
) -> Tuple[StrategyConfig, BacktestResult]:
    best_config = None
    best_result = None

    for window in window_months:
        for buy in buy_thresholds:
            for sell in sell_thresholds:
                for prob in prob_thresholds:
                    config = StrategyConfig(
                        window_months=window,
                        buy_threshold=buy,
                        sell_threshold=sell,
                        prob_threshold=prob,
                    )
                    result = simulate_backtest(data, config, backtest_config)
                    if best_result is None or result.cagr > best_result.cagr:
                        best_result = result
                        best_config = config

    if best_config is None or best_result is None:
        raise ValueError("Grid search did not produce any results.")

    return best_config, best_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nifty 50 EOD strategy research tool.")
    parser.add_argument("--start", default="2009-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--window-months", type=int, default=3)
    parser.add_argument("--buy-threshold", type=float, default=-3.0)
    parser.add_argument("--sell-threshold", type=float, default=-1.0)
    parser.add_argument("--prob-threshold", type=float, default=0.15)
    parser.add_argument("--fit-years", type=int, default=5)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--grid-search", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = download_nifty_data(start=args.start, end=args.end)

    strategy_config = StrategyConfig(
        window_months=args.window_months,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        prob_threshold=args.prob_threshold,
    )
    backtest_config = BacktestConfig(
        fit_years=args.fit_years,
        test_years=args.test_years,
    )

    if args.grid_search:
        window_months = [3, 4, 5]
        buy_thresholds = [-2.0, -3.0, -4.0]
        sell_thresholds = [-1.0, -0.5]
        prob_thresholds = [0.1, 0.15, 0.2]
        best_config, best_result = grid_search(
            data,
            window_months,
            buy_thresholds,
            sell_thresholds,
            prob_thresholds,
            backtest_config,
        )
        print("Best config:", best_config)
        print(f"Best CAGR: {best_result.cagr:.2%}")
    else:
        train_result, test_result = train_and_test(data, strategy_config, backtest_config)
        print(f"Train CAGR: {train_result.cagr:.2%}")
        print(f"Test CAGR: {test_result.cagr:.2%}")

        print("Recent signals:")
        print(
            generate_signals(data, strategy_config)
            .tail(5)[["BOD", "EOD", "RollingHigh", "PctFromHigh", "Signal"]]
        )


if __name__ == "__main__":
    main()
