# Nifty 50 EOD Strategy Research

This repository contains a Python script to download Nifty 50 daily OHLC data,
compute an end-of-day decision framework, and back-test it using a rolling
3–5 month probability chart.

## What the script does

1. **Downloads daily OHLC data** for the Nifty 50 index using `yfinance`.
2. **Computes the rolling high** as the median of the top 5 highs in the last
   3–5 months (default is 3 months).
3. **Creates a probability chart** of percent changes from that rolling high
   in 1% bins (e.g., +1%, +2%, -1%, -2%).
4. **Generates EOD buy/sell/hold signals** based on the current percent change
   from the rolling high and the probability of being in the lower tail.
5. **Back-tests** the strategy with a training window (default 5 years) and
   a test window (default 1 year).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Download ~15 years of data and run the default strategy:

```bash
python nifty_strategy.py --start 2009-01-01
```

Run a simple grid search over window sizes and thresholds:

```bash
python nifty_strategy.py --start 2009-01-01 --grid-search
```

Adjust decision windows and thresholds:

```bash
python nifty_strategy.py \
  --start 2009-01-01 \
  --window-months 4 \
  --buy-threshold -4 \
  --sell-threshold -1 \
  --prob-threshold 0.2
```

## Notes

- Signals are generated **at end-of-day**, and trades execute at the next
  day open (BOD), which matches a daily decision workflow.
- This is a research tool. Please validate assumptions and incorporate
  realistic transaction costs/slippage before live trading.
