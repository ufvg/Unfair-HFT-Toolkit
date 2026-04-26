# Unfair HFT Toolkit

Minimal tooling for studying and monitoring **microprice** in live crypto order flow.

This repo focuses on three pieces:

- `apps/microprice_gui.py`
  Live GUI for streaming bid/ask, midprice, microprice, adjustment, strategy state, and equity.
- `fetch_hyperliquid_l2.py`
  Historical Hyperliquid L2 fetcher using AWS-backed archive access.
- `L1microprice.py`, `calibration.py`, `multilevel_microprice.py`, `multilevel_calibration.py`
  Core microprice models, calibration utilities, and live streaming logic.

The toolkit is built for lightweight experimentation:

- stream live order book updates
- compare midprice vs microprice in real time
- calibrate and load L1 or multilevel models
- run simple signal-driven strategies from microprice adjustment

It is intentionally small, practical, and centered on market microstructure rather than full trading infrastructure.
