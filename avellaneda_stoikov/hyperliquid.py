from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Callable

from microstructure.microprice.L1microprice import (
    HYPERLIQUID_MAINNET_WS_URL,
    HYPERLIQUID_TESTNET_WS_URL,
    _default_websocket_factory,
)

from .model import MarketSnapshot, QuoteDecision


def _normalize_book_message(message: dict[str, Any]) -> MarketSnapshot | None:
    if message.get("channel") != "l2Book":
        return None
    data = message.get("data")
    if not isinstance(data, dict):
        return None
    levels = data.get("levels")
    if not isinstance(levels, list) or len(levels) < 2:
        return None
    bids = levels[0] or []
    asks = levels[1] or []
    if not bids or not asks:
        return None
    try:
        return MarketSnapshot(
            timestamp=float(data["time"]) / 1000.0,
            best_bid=float(bids[0]["px"]),
            best_ask=float(asks[0]["px"]),
            bid_size=float(bids[0]["sz"]),
            ask_size=float(asks[0]["sz"]),
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return None


class HyperliquidBookFeed:
    def __init__(
        self,
        *,
        coin: str,
        url: str = HYPERLIQUID_MAINNET_WS_URL,
        websocket_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.coin = coin.upper()
        self.url = url
        self.websocket_factory = _default_websocket_factory if websocket_factory is None else websocket_factory

    def stream(self, on_snapshot: Callable[[MarketSnapshot], None], *, stop_condition: Callable[[], bool] | None = None) -> None:
        connection = self.websocket_factory(self.url)
        try:
            connection.send(
                json.dumps(
                    {
                        "method": "subscribe",
                        "subscription": {
                            "type": "l2Book",
                            "coin": self.coin,
                        },
                    }
                )
            )
            while True:
                if stop_condition is not None and stop_condition():
                    break
                raw = connection.recv()
                if raw is None:
                    break
                try:
                    parsed = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                snapshot = _normalize_book_message(parsed)
                if snapshot is not None:
                    on_snapshot(snapshot)
        finally:
            connection.close()


@dataclass(slots=True)
class HyperliquidExecutionConfig:
    account_address: str
    secret_key: str
    coin: str
    order_size: float
    vault_address: str | None = None
    use_testnet: bool = True
    tif: str = "Gtc"
    expires_after_ms: int | None = None

    @property
    def base_url(self) -> str:
        return "https://api.hyperliquid-testnet.xyz" if self.use_testnet else "https://api.hyperliquid.xyz"

    @property
    def ws_url(self) -> str:
        return HYPERLIQUID_TESTNET_WS_URL if self.use_testnet else HYPERLIQUID_MAINNET_WS_URL


class HyperliquidOrderExecutor:
    """Thin wrapper around the official Hyperliquid Python SDK.

    This module is intentionally optional so the library remains testable without
    live credentials or extra dependencies.
    """

    def __init__(self, config: HyperliquidExecutionConfig) -> None:
        try:
            import eth_account
            from hyperliquid.exchange import Exchange
            from hyperliquid.info import Info
        except ImportError as exc:
            raise RuntimeError(
                "Live order placement requires `hyperliquid-python-sdk` and `eth-account`."
            ) from exc
        self.config = config
        self._eth_account = eth_account
        self._exchange_cls = Exchange
        self._info_cls = Info
        self._wallet = eth_account.Account.from_key(config.secret_key)
        self.info = Info(config.base_url, True)
        self.exchange = Exchange(
            self._wallet,
            config.base_url,
            vault_address=config.vault_address,
            account_address=config.account_address,
        )
        if config.expires_after_ms is not None:
            self.exchange.set_expires_after(int(config.expires_after_ms))
        self.bid_order_id: int | None = None
        self.ask_order_id: int | None = None

    @staticmethod
    def _extract_resting_oid(result: dict[str, Any]) -> int | None:
        if result.get("status") != "ok":
            return None
        try:
            statuses = result["response"]["data"]["statuses"]
        except (KeyError, TypeError):
            return None
        for status in statuses:
            if isinstance(status, dict) and "resting" in status:
                resting = status["resting"]
                if isinstance(resting, dict) and "oid" in resting:
                    return int(resting["oid"])
        return None

    def _place_order(self, is_buy: bool, price: float, size: float) -> int | None:
        result = self.exchange.order(
            self.config.coin,
            is_buy,
            size,
            price,
            {"limit": {"tif": self.config.tif}},
        )
        return self._extract_resting_oid(result)

    def _modify_order(self, oid: int, is_buy: bool, price: float, size: float) -> int | None:
        result = self.exchange.modify_order(
            oid,
            self.config.coin,
            is_buy,
            size,
            price,
            {"limit": {"tif": self.config.tif}},
        )
        return self._extract_resting_oid(result) or oid

    def _cancel_order(self, oid: int | None) -> None:
        if oid is None:
            return
        self.exchange.cancel(self.config.coin, oid)

    def cancel_all(self) -> None:
        self._cancel_order(self.bid_order_id)
        self._cancel_order(self.ask_order_id)
        self.bid_order_id = None
        self.ask_order_id = None

    def sync_quotes(self, decision: QuoteDecision) -> dict[str, int | None]:
        size = self.config.order_size
        if decision.bid_intensity <= 0.0:
            self._cancel_order(self.bid_order_id)
            self.bid_order_id = None
        elif self.bid_order_id is None:
            self.bid_order_id = self._place_order(True, decision.bid_price, size)
        else:
            self.bid_order_id = self._modify_order(self.bid_order_id, True, decision.bid_price, size)

        if decision.ask_intensity <= 0.0:
            self._cancel_order(self.ask_order_id)
            self.ask_order_id = None
        elif self.ask_order_id is None:
            self.ask_order_id = self._place_order(False, decision.ask_price, size)
        else:
            self.ask_order_id = self._modify_order(self.ask_order_id, False, decision.ask_price, size)
        return {"bid_order_id": self.bid_order_id, "ask_order_id": self.ask_order_id}

    def heartbeat(self) -> None:
        nonce = int(time.time() * 1000)
        self.exchange.noop(nonce)
