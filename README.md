# Unfair HFT Toolkit

Minimal tooling for studying and monitoring **microprice** in live crypto order flow.

This repo focuses on three pieces:

- `microprice/gui/microprice_gui.py`
  Live GUI for streaming bid/ask, midprice, microprice, adjustment, strategy state, and equity.
- `data_fetcher/fetch_hyperliquid_l2.py`
  Historical Hyperliquid L2 fetcher using AWS-backed archive access.
- `data_fetcher/gui_fetch_hyperliquid_l2.py`
  Simple GUI for downloading and decompressing historical Hyperliquid data.
- `microprice/`
  Core microprice models, calibration utilities, and live streaming logic.

The toolkit is built for lightweight experimentation:

- stream live order book updates
- compare midprice vs microprice in real time
- calibrate and load L1 or multilevel models
- run simple signal-driven strategies from microprice adjustment

It is intentionally small, practical, and centered on market microstructure rather than full trading infrastructure.
