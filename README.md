# Unfair HFT Toolkit

Minimal tooling for studying and monitoring **microprice** in live crypto order flow.

This repo focuses on three pieces:

- `microstructure/microprice/gui/microprice_gui.py`
  Live GUI for streaming bid/ask, midprice, microprice, adjustment, strategy state, and equity.
- `vpin/gui/vpin_gui.py`
  Standalone VPIN inspector for loading trade CSVs and checking that the toxicity signal behaves sensibly.
- `data_fetcher/fetch_hyperliquid_l2.py`
  Historical Hyperliquid L2 fetcher using AWS-backed archive access.
- `data_fetcher/gui_fetch_hyperliquid_l2.py`
  Simple GUI for downloading and decompressing historical Hyperliquid data.
- `microstructure/microprice/`
  Core microprice models, calibration utilities, and live streaming logic.
- `vpin/`
  Standalone VPIN library and GUI with the academic bulk-volume classification baseline plus `tick` and `aggressor` paths for HFT use.
- `avellaneda_stoikov/`
  Standalone Avellaneda-Stoikov market-making module with a paper-faithful quoting core, internal fill simulator, optional Hyperliquid execution adapter, and a GUI for previewing hypothetical bid/ask quotes.

The toolkit is built for lightweight experimentation:

- stream live order book updates
- compare midprice vs microprice in real time
- calibrate and load L1 or multilevel models
- run simple signal-driven strategies from microprice adjustment
- compute VPIN from trade data with configurable bucket sizing
- run incremental low-latency VPIN updates for trade streams
- compute classical Avellaneda-Stoikov reservation prices and optimal quotes
- preview hypothetical market-making quotes on live Hyperliquid books or internal demo paths
- simulate fills and inventory/PnL before enabling live order placement

It is intentionally small, practical, and centered on market microstructure rather than full trading infrastructure.

## Avellaneda-Stoikov

The `avellaneda_stoikov` package is separated from the existing microprice and VPIN tooling so it can be used independently by Hyperliquid market-making bots.

- `python -m avellaneda_stoikov --gui`
  Launch the quote monitor GUI. `Internal Demo` simulates fills and PnL. `Hyperliquid Preview` streams live top-of-book and shows theoretical quotes without sending orders.
- `python -m avellaneda_stoikov`
  Print demo quotes as JSON.

Live order placement is implemented through the official `hyperliquid-python-sdk`, but the dependency is optional so the simulator and GUI remain testable without credentials. The live execution wrapper is in `avellaneda_stoikov/hyperliquid.py`.
