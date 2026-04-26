from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np

from .calibration import FittedMicropriceModel, load_model
from .multilevel_calibration import FittedMultilevelMicropriceModel, load_multilevel_model

HYPERLIQUID_MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
HYPERLIQUID_TESTNET_WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"
GUI_POLL_MS = 1
ESTIMATOR_CHOICES = tuple([f"G{horizon}" for horizon in range(1, 7)] + ["G_star"])

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional runtime acceleration
    njit = None


def _compute_live_values_python(
    bid: float,
    ask: float,
    bid_size: float,
    ask_size: float,
    tick_size: float,
    n_spread: int,
    n_imb: int,
    imbalance_edges: np.ndarray,
    adjustment_vector: np.ndarray,
) -> tuple[float, float, int | None, int | None, int | None]:
    total = bid_size + ask_size
    imbalance = 0.5 if total <= 0.0 else bid_size / total
    spread = ask - bid
    if spread <= 0.0 or tick_size <= 0.0:
        return imbalance, 0.0, None, None, None
    spread_ticks = int(np.rint(spread / tick_size))
    if spread_ticks < 1:
        return imbalance, 0.0, None, None, None
    spread_ticks = min(max(spread_ticks, 1), n_spread)
    clipped = min(max(imbalance, 0.0), 1.0)
    imbalance_bucket = int(np.searchsorted(imbalance_edges[1:-1], clipped, side="right"))
    imbalance_bucket = min(max(imbalance_bucket, 0), n_imb - 1)
    spread_bucket = spread_ticks - 1
    state_index = spread_bucket * n_imb + imbalance_bucket
    return imbalance, float(adjustment_vector[state_index]), spread_ticks, spread_bucket, state_index


if njit is not None:

    @njit(cache=True)
    def _searchsorted_right(edges: np.ndarray, value: float) -> int:
        left = 0
        right = edges.shape[0]
        while left < right:
            mid = (left + right) // 2
            if value < edges[mid]:
                right = mid
            else:
                left = mid + 1
        return left


    @njit(cache=True)
    def _compute_live_values_numba(
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
        tick_size: float,
        n_spread: int,
        n_imb: int,
        imbalance_edges: np.ndarray,
        adjustment_vector: np.ndarray,
    ) -> tuple[float, float, int, int, int]:
        total = bid_size + ask_size
        imbalance = 0.5 if total <= 0.0 else bid_size / total
        spread = ask - bid
        if spread <= 0.0 or tick_size <= 0.0:
            return imbalance, 0.0, -1, -1, -1
        spread_ticks = int(np.rint(spread / tick_size))
        if spread_ticks < 1:
            return imbalance, 0.0, -1, -1, -1
        if spread_ticks > n_spread:
            spread_ticks = n_spread
        clipped = imbalance
        if clipped < 0.0:
            clipped = 0.0
        elif clipped > 1.0:
            clipped = 1.0
        imbalance_bucket = _searchsorted_right(imbalance_edges[1:-1], clipped)
        if imbalance_bucket < 0:
            imbalance_bucket = 0
        elif imbalance_bucket >= n_imb:
            imbalance_bucket = n_imb - 1
        spread_bucket = spread_ticks - 1
        state_index = spread_bucket * n_imb + imbalance_bucket
        return imbalance, adjustment_vector[state_index], spread_ticks, spread_bucket, state_index


else:
    _compute_live_values_numba = None


StreamingModel = FittedMicropriceModel | FittedMultilevelMicropriceModel


def _default_websocket_factory(url: str) -> Any:
    try:
        import websocket
    except ImportError as exc:
        raise RuntimeError(
            "Missing websocket client dependency. Install `websocket-client` in the project venv."
        ) from exc

    create_connection = getattr(websocket, "create_connection", None)
    if create_connection is None:
        module_path = getattr(websocket, "__file__", "<unknown>")
        raise RuntimeError(
            "The installed `websocket` module is not `websocket-client` and does not expose "
            f"`create_connection` ({module_path}). Install `websocket-client` in the project venv."
        )
    return create_connection(url)


def is_valid_estimator_name(estimator: str) -> bool:
    return estimator == "G_star" or re.fullmatch(r"G\d+", estimator) is not None


def build_subscription_message(coin: str, subscription_type: str = "l2Book") -> dict[str, Any]:
    return {
        "method": "subscribe",
        "subscription": {
            "type": subscription_type,
            "coin": coin,
        },
    }


def _normalize_l2book(data: dict[str, Any]) -> dict[str, float] | None:
    levels = data.get("levels")
    if not isinstance(levels, list) or len(levels) < 2:
        return None
    bids = levels[0] or []
    asks = levels[1] or []
    if not bids or not asks:
        return None
    try:
        best_bid = bids[0]
        best_ask = asks[0]
        return {
            "time": float(data["time"]),
            "bid": float(best_bid["px"]),
            "bs": float(best_bid["sz"]),
            "ask": float(best_ask["px"]),
            "as": float(best_ask["sz"]),
        }
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _normalize_l2book_multilevel(data: dict[str, Any], levels: int) -> dict[str, Any] | None:
    if levels <= 0:
        raise ValueError("levels must be positive.")
    book_levels = data.get("levels")
    if not isinstance(book_levels, list) or len(book_levels) < 2:
        return None
    bids = book_levels[0] or []
    asks = book_levels[1] or []
    if len(bids) < levels or len(asks) < levels:
        return None
    try:
        bid_prices = np.empty(levels, dtype=np.float64)
        bid_sizes = np.empty(levels, dtype=np.float64)
        ask_prices = np.empty(levels, dtype=np.float64)
        ask_sizes = np.empty(levels, dtype=np.float64)
        for level in range(levels):
            bid_prices[level] = float(bids[level]["px"])
            bid_sizes[level] = float(bids[level]["sz"])
            ask_prices[level] = float(asks[level]["px"])
            ask_sizes[level] = float(asks[level]["sz"])
        return {
            "time": float(data["time"]),
            "bid_prices": bid_prices,
            "bid_sizes": bid_sizes,
            "ask_prices": ask_prices,
            "ask_sizes": ask_sizes,
        }
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _normalize_bbo(data: dict[str, Any]) -> dict[str, float] | None:
    bbo = data.get("bbo")
    if not isinstance(bbo, list) or len(bbo) != 2:
        return None
    best_bid, best_ask = bbo
    if best_bid is None or best_ask is None:
        return None
    try:
        return {
            "time": float(data["time"]),
            "bid": float(best_bid["px"]),
            "bs": float(best_bid["sz"]),
            "ask": float(best_ask["px"]),
            "as": float(best_ask["sz"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _extract_book_levels_from_message(message: dict[str, Any]) -> tuple[list[dict[str, float]] | None, list[dict[str, float]] | None]:
    data = message.get("data")
    if not isinstance(data, dict):
        return None, None
    channel = message.get("channel")
    if channel == "l2Book":
        levels = data.get("levels")
        if not isinstance(levels, list) or len(levels) < 2:
            return None, None
        bids = levels[0] or []
        asks = levels[1] or []
        try:
            book_bids = [
                {"px": float(level["px"]), "sz": float(level["sz"])}
                for level in bids
                if isinstance(level, dict)
            ]
            book_asks = [
                {"px": float(level["px"]), "sz": float(level["sz"])}
                for level in asks
                if isinstance(level, dict)
            ]
        except (KeyError, TypeError, ValueError):
            return None, None
        return book_bids, book_asks
    if channel == "bbo":
        bbo = data.get("bbo")
        if not isinstance(bbo, list) or len(bbo) != 2 or bbo[0] is None or bbo[1] is None:
            return None, None
        try:
            return (
                [{"px": float(bbo[0]["px"]), "sz": float(bbo[0]["sz"])}],
                [{"px": float(bbo[1]["px"]), "sz": float(bbo[1]["sz"])}],
            )
        except (KeyError, TypeError, ValueError):
            return None, None
    return None, None


def normalize_hyperliquid_message(message: dict[str, Any]) -> dict[str, float] | None:
    if not isinstance(message.get("data"), dict):
        return None
    if message.get("channel") == "l2Book":
        return _normalize_l2book(message["data"])
    if message.get("channel") == "bbo":
        return _normalize_bbo(message["data"])
    return None


def load_streaming_model(path: str | Path) -> StreamingModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model_kind = payload.get("model_kind")
    if model_kind == "multilevel":
        return load_multilevel_model(path)
    if model_kind == "multilevel_state_conditioned":
        raise ValueError("State-conditioned multilevel models are research-only and are not supported in the live path.")
    if model_kind == "l1":
        return load_model(path)
    if {"levels", "decay_lambda", "intercept", "slope"}.issubset(payload):
        return load_multilevel_model(path)
    return load_model(path)


def _build_l1_payload(
    normalized: dict[str, float],
    model: FittedMicropriceModel,
    estimator: str | int,
) -> dict[str, Any]:
    imbalance_edges = np.asarray(model.imbalance_edges, dtype=np.float64)
    adjustment_vector = np.asarray(model._adjustment_vector(estimator), dtype=np.float64)
    compute_live_values = (
        _compute_live_values_numba if _compute_live_values_numba is not None else _compute_live_values_python
    )
    bid = normalized["bid"]
    ask = normalized["ask"]
    bid_size = normalized["bs"]
    ask_size = normalized["as"]
    midprice = (bid + ask) / 2.0
    imbalance, adjustment, spread_ticks, spread_bucket, state_index = compute_live_values(
        float(bid),
        float(ask),
        float(bid_size),
        float(ask_size),
        float(model.tick_size),
        int(model.n_spread),
        int(model.n_imb),
        imbalance_edges,
        adjustment_vector,
    )
    adjustment_ticks = None if model.tick_size <= 0.0 else float(adjustment / model.tick_size)
    state = {
        "imbalance": float(imbalance),
        "imbalance_bucket": None if state_index in (None, -1) else int(state_index % model.n_imb),
        "spread_ticks": None if spread_ticks in (None, -1) else int(spread_ticks),
        "spread_bucket": None if spread_bucket in (None, -1) else int(spread_bucket),
        "state_index": None if state_index in (None, -1) else int(state_index),
    }
    return {
        "timestamp": normalized["time"],
        "bid": bid,
        "ask": ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "midprice": midprice,
        "imbalance": float(imbalance),
        "state": state,
        "adjustment": float(adjustment),
        "adjustment_ticks": adjustment_ticks,
        "microprice": midprice + adjustment,
        "estimator": estimator,
        "model_dt": int(model.dt),
        "model_tick_size": float(model.tick_size),
        "model_kind": "l1",
        "book_bids": normalized.get("book_bids"),
        "book_asks": normalized.get("book_asks"),
    }


def _build_multilevel_payload(
    normalized: dict[str, Any],
    model: FittedMultilevelMicropriceModel,
) -> dict[str, Any]:
    bid_prices = normalized["bid_prices"]
    bid_sizes = normalized["bid_sizes"]
    ask_prices = normalized["ask_prices"]
    ask_sizes = normalized["ask_sizes"]
    bid = float(bid_prices[0])
    ask = float(ask_prices[0])
    bid_size = float(bid_sizes[0])
    ask_size = float(ask_sizes[0])
    midprice = 0.5 * (bid + ask)
    imbalance = 0.5 if bid_size + ask_size <= 0.0 else bid_size / (bid_size + ask_size)
    adjustment = model.adjustment_from_book(
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
    )
    adjustment_ticks = None if model.tick_size <= 0.0 else float(adjustment / model.tick_size)
    return {
        "timestamp": float(normalized["time"]),
        "bid": bid,
        "ask": ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "midprice": midprice,
        "imbalance": float(imbalance),
        "state": None,
        "adjustment": float(adjustment),
        "adjustment_ticks": adjustment_ticks,
        "microprice": midprice + adjustment,
        "estimator": "multilevel",
        "model_dt": int(model.dt),
        "model_tick_size": float(model.tick_size),
        "model_kind": "multilevel",
        "model_levels": int(model.levels),
        "model_decay_lambda": float(model.decay_lambda),
        "book_bids": normalized.get("book_bids"),
        "book_asks": normalized.get("book_asks"),
    }


def build_live_payload_from_message(
    message: dict[str, Any],
    model: StreamingModel,
    estimator: str | int = "G1",
) -> dict[str, Any] | None:
    if isinstance(model, FittedMultilevelMicropriceModel):
        if message.get("channel") != "l2Book" or not isinstance(message.get("data"), dict):
            return None
        normalized = _normalize_l2book_multilevel(message["data"], levels=int(model.levels))
        if normalized is None:
            return None
        payload = _build_multilevel_payload(normalized, model)
        book_bids, book_asks = _extract_book_levels_from_message(message)
        payload["book_bids"] = book_bids
        payload["book_asks"] = book_asks
        return payload

    normalized = normalize_hyperliquid_message(message)
    if normalized is None:
        return None
    payload = _build_l1_payload(normalized, model, estimator)
    book_bids, book_asks = _extract_book_levels_from_message(message)
    payload["book_bids"] = book_bids
    payload["book_asks"] = book_asks
    return payload


def _stream_messages(
    coin: str,
    on_update: Callable[[dict[str, Any]], None],
    url: str,
    subscription_type: str,
    websocket_factory: Callable[[str], Any],
    build_payload: Callable[[dict[str, Any]], dict[str, Any] | None],
    max_messages: int | None,
) -> None:
    connection = websocket_factory(url)
    processed = 0
    try:
        connection.send(json.dumps(build_subscription_message(coin, subscription_type)))
        while True:
            raw_message = connection.recv()
            if raw_message is None:
                break
            try:
                parsed = json.loads(raw_message)
            except (TypeError, json.JSONDecodeError):
                continue
            payload = build_payload(parsed)
            if payload is None:
                continue
            on_update(payload)
            processed += 1
            if max_messages is not None and processed >= max_messages:
                break
    finally:
        connection.close()


def _stream_l1_microprice(
    coin: str,
    model: FittedMicropriceModel,
    on_update: Callable[[dict[str, Any]], None],
    url: str,
    subscription_type: str,
    estimator: str | int,
    websocket_factory: Callable[[str], Any],
    max_messages: int | None,
) -> None:
    _stream_messages(
        coin=coin,
        on_update=on_update,
        url=url,
        subscription_type=subscription_type,
        websocket_factory=websocket_factory,
        build_payload=lambda parsed: build_live_payload_from_message(parsed, model, estimator=estimator),
        max_messages=max_messages,
    )


def _stream_multilevel_microprice(
    coin: str,
    model: FittedMultilevelMicropriceModel,
    on_update: Callable[[dict[str, Any]], None],
    url: str,
    subscription_type: str,
    websocket_factory: Callable[[str], Any],
    max_messages: int | None,
) -> None:
    if subscription_type != "l2Book":
        raise ValueError("Multilevel streaming requires subscription_type='l2Book'.")
    _stream_messages(
        coin=coin,
        on_update=on_update,
        url=url,
        subscription_type=subscription_type,
        websocket_factory=websocket_factory,
        build_payload=lambda parsed: build_live_payload_from_message(parsed, model, estimator="multilevel"),
        max_messages=max_messages,
    )


def stream_hyperliquid_microprice(
    coin: str,
    model: StreamingModel,
    on_update: Callable[[dict[str, Any]], None],
    url: str = HYPERLIQUID_MAINNET_WS_URL,
    subscription_type: str = "l2Book",
    estimator: str | int = "G1",
    websocket_factory: Callable[[str], Any] | None = None,
    max_messages: int | None = None,
) -> None:
    if websocket_factory is None:
        websocket_factory = _default_websocket_factory
    if isinstance(model, FittedMultilevelMicropriceModel):
        _stream_multilevel_microprice(
            coin=coin,
            model=model,
            on_update=on_update,
            url=url,
            subscription_type=subscription_type,
            websocket_factory=websocket_factory,
            max_messages=max_messages,
        )
        return
    _stream_l1_microprice(
        coin=coin,
        model=model,
        on_update=on_update,
        url=url,
        subscription_type=subscription_type,
        estimator=estimator,
        websocket_factory=websocket_factory,
        max_messages=max_messages,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream Hyperliquid top-of-book data and print live Stoikov microprice updates."
    )
    parser.add_argument("--coin", help="Hyperliquid coin symbol, for example BTC or ETH.")
    parser.add_argument("--model", help="Path to a calibrated model JSON file.")
    parser.add_argument(
        "--url",
        default=HYPERLIQUID_MAINNET_WS_URL,
        help="Websocket URL. Defaults to Hyperliquid mainnet.",
    )
    parser.add_argument(
        "--subscription-type",
        default="l2Book",
        choices=("l2Book", "bbo"),
        help="Subscription channel to consume.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        help="Stop after this many normalized market data messages.",
    )
    parser.add_argument(
        "--estimator",
        default="G1",
        help="Adjustment estimator to stream, for example G1, G4, G12, or G_star. Defaults to G1.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Tkinter GUI instead of printing JSON updates.",
    )
    return parser


def launch_gui() -> None:
    from microprice.gui.microprice_gui import launch_gui as gui_launch

    gui_launch()


def __getattr__(name: str) -> Any:
    if name == "MicropriceGui":
        from microprice.gui.microprice_gui import MicropriceGui

        return MicropriceGui
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ESTIMATOR_CHOICES",
    "GUI_POLL_MS",
    "HYPERLIQUID_MAINNET_WS_URL",
    "HYPERLIQUID_TESTNET_WS_URL",
    "_build_argument_parser",
    "build_subscription_message",
    "build_live_payload_from_message",
    "is_valid_estimator_name",
    "launch_gui",
    "load_model",
    "load_streaming_model",
    "normalize_hyperliquid_message",
    "stream_hyperliquid_microprice",
]


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()

    if args.gui or (not args.coin and not args.model):
        launch_gui()
        return

    if not args.coin or not args.model:
        parser.error("--coin and --model are required unless --gui is used.")
    if not is_valid_estimator_name(args.estimator):
        parser.error("--estimator must be G<k> for a positive integer horizon or G_star.")

    model = load_streaming_model(args.model)
    if isinstance(model, FittedMicropriceModel):
        if args.estimator not in model.available_estimators():
            parser.error(
                f"--estimator {args.estimator} is not available in this model. "
                f"Available estimators: {', '.join(model.available_estimators())}."
            )
    elif args.subscription_type != "l2Book":
        parser.error("--subscription-type must be l2Book for multilevel models.")

    def print_update(payload: dict[str, Any]) -> None:
        print(json.dumps(payload, sort_keys=True))

    stream_hyperliquid_microprice(
        coin=args.coin,
        model=model,
        on_update=print_update,
        url=args.url,
        subscription_type=args.subscription_type,
        estimator=args.estimator,
        max_messages=args.max_messages,
    )


if __name__ == "__main__":
    main()
