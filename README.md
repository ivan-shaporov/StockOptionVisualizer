# StockOptionVisualizer

Download historical option prices from [marketdata.app](https://marketdata.app) and visualize put/call price evolution over time across strikes and maturities.

> **Note:** This repository was 100% vibecoded.
>
> **Caveat:** Call options visualization is not fully supported.

## Contents

- `download_option_data.py` — fetch historical option chains for a ticker into CSV.
- `plot_option_prices.py` — interactive matplotlib viewer for the downloaded CSV (per-strike or all-strikes mode, maturity toggles, strike slider, underlying-price overlay).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in MARKETDATA_TOKEN, SYMBOL, SIDE, dates, etc.
```

## Usage

```bash
python download_option_data.py
python plot_option_prices.py data/AAPL/put/all_puts.csv --strike 180 --maturity 2024-03-15
```

See module docstrings for the full list of env vars and CLI options.
