from __future__ import annotations

import csv
import json
from pathlib import Path

from .core import TradePrint, estimate_average_daily_volume


def load_trades_from_csv(
    path: str | Path,
    *,
    timestamp_column: str = "timestamp",
    price_column: str = "price",
    volume_column: str = "volume",
    side_column: str | None = None,
) -> tuple[TradePrint, ...]:
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {timestamp_column, price_column, volume_column}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(
                f"CSV must contain columns {sorted(required)}; found [{available}]"
            )
        if side_column is not None and side_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise ValueError(f"CSV is missing side column '{side_column}'; found [{available}]")
        rows = []
        for row in reader:
            side = row[side_column] if side_column is not None else None
            rows.append(
                TradePrint(
                    timestamp=row[timestamp_column],
                    price=float(row[price_column]),
                    volume=float(row[volume_column]),
                    side=side,
                )
            )
    if not rows:
        raise ValueError("CSV file did not contain any trade rows.")
    rows.sort(key=lambda trade: trade.timestamp)
    return tuple(rows)


def estimate_average_daily_volume_from_csv(
    path: str | Path,
    *,
    timestamp_column: str = "timestamp",
    price_column: str = "price",
    volume_column: str = "volume",
    side_column: str | None = None,
) -> float:
    trades = load_trades_from_csv(
        path,
        timestamp_column=timestamp_column,
        price_column=price_column,
        volume_column=volume_column,
        side_column=side_column,
    )
    average_daily_volume, _ = estimate_average_daily_volume(trades)
    return average_daily_volume


def load_adv_cache(path: str | Path) -> dict[str, dict[str, object]]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ADV cache must be a JSON object.")
    normalized: dict[str, dict[str, object]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            normalized[key.upper()] = dict(value)
    return normalized


def save_adv_cache(path: str | Path, payload: dict[str, dict[str, object]]) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
