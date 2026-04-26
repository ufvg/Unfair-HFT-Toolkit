from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import lz4.frame

BUCKET_NAME = "hyperliquid-archive"
KEY_TEMPLATE = "market_data/{date}/{hour}/l2Book/{coin}.lz4"


@dataclass
class FetchStats:
    downloaded: int = 0
    skipped: int = 0
    missing: int = 0
    bytes_downloaded: int = 0
    planned: int = 0
    local_paths: list[Path] | None = None


def parse_yyyymmdd(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}'. Expected YYYYMMDD.") from exc


def generate_archive_requests(
    coin: str,
    date_from: date,
    date_to: date,
    hour_from: int = 0,
    hour_to: int = 23,
) -> list[dict[str, str]]:
    if date_to < date_from:
        raise ValueError("date_to must be greater than or equal to date_from.")
    if not (0 <= hour_from <= 23 and 0 <= hour_to <= 23):
        raise ValueError("Hours must be between 0 and 23.")
    if hour_to < hour_from:
        raise ValueError("hour_to must be greater than or equal to hour_from.")

    requests: list[dict[str, str]] = []
    current = date_from
    while current <= date_to:
        date_str = current.strftime("%Y%m%d")
        for hour in range(hour_from, hour_to + 1):
            remote_hour = str(hour)
            local_hour = f"{hour:02d}"
            key = KEY_TEMPLATE.format(date=date_str, hour=remote_hour, coin=coin)
            requests.append(
                {
                    "date": date_str,
                    "hour": local_hour,
                    "remote_hour": remote_hour,
                    "key": key,
                    "coin": coin,
                }
            )
        current += timedelta(days=1)
    return requests


def local_output_path(out_dir: str | Path, coin: str, date_str: str, hour_str: str) -> Path:
    return Path(out_dir) / coin / date_str / f"{hour_str}.lz4"


def decompressed_output_path(path: str | Path) -> Path:
    archive_path = Path(path)
    if archive_path.suffix.lower() == ".lz4":
        return archive_path.with_suffix(".jsonl")
    return archive_path.with_name(archive_path.name + ".jsonl")


def create_s3_client() -> Any:
    return boto3.client("s3")


def _is_missing_object(error: ClientError) -> bool:
    code = error.response.get("Error", {}).get("Code")
    return code in {"404", "NoSuchKey", "NotFound"}


def download_archive_object(
    s3_client: Any,
    bucket: str,
    key: str,
    destination: str | Path,
    request_payer: str = "requester",
    overwrite: bool = False,
) -> dict[str, Any]:
    target = Path(destination)
    if target.exists() and not overwrite:
        return {"status": "skipped", "path": target, "bytes": 0}

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key, RequestPayer=request_payer)
    except ClientError as exc:
        if _is_missing_object(exc):
            return {"status": "missing", "path": target, "bytes": 0}
        raise
    except NoCredentialsError:
        raise RuntimeError(
            "AWS credentials are required to access the requester-pays Hyperliquid archive bucket."
        )

    body = response["Body"].read()
    target.write_bytes(body)
    return {"status": "downloaded", "path": target, "bytes": len(body)}


def decompress_archive_object(
    source_path: str | Path,
    destination: str | Path | None = None,
    overwrite: bool = False,
    remove_source: bool = False,
) -> Path:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Archive file does not exist: {source}")
    output_path = Path(destination) if destination is not None else decompressed_output_path(source)
    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    decompressed = lz4.frame.decompress(source.read_bytes())
    output_path.write_bytes(decompressed)
    if remove_source and source.exists():
        source.unlink()
    return output_path


def fetch_historical_l2(
    coin: str,
    date_from: date,
    date_to: date,
    out_dir: str | Path,
    hour_from: int = 0,
    hour_to: int = 23,
    overwrite: bool = False,
    dry_run: bool = False,
    request_payer: str = "requester",
    s3_client: Any | None = None,
    progress_callback: Any | None = None,
) -> FetchStats:
    requests = generate_archive_requests(
        coin=coin,
        date_from=date_from,
        date_to=date_to,
        hour_from=hour_from,
        hour_to=hour_to,
    )
    stats = FetchStats(planned=len(requests), local_paths=[])

    if s3_client is None and not dry_run:
        s3_client = create_s3_client()

    for index, item in enumerate(requests, start=1):
        destination = local_output_path(out_dir, coin=item["coin"], date_str=item["date"], hour_str=item["hour"])
        print(f"Checking {item['date']} {item['hour']}:00 -> {item['key']}")
        if dry_run:
            print(f"Would download to {destination}")
            if progress_callback is not None:
                progress_callback(
                    {
                        "current": index,
                        "planned": stats.planned,
                        "date": item["date"],
                        "hour": item["hour"],
                        "status": "dry-run",
                        "path": destination,
                    }
                )
            continue

        result = download_archive_object(
            s3_client=s3_client,
            bucket=BUCKET_NAME,
            key=item["key"],
            destination=destination,
            request_payer=request_payer,
            overwrite=overwrite,
        )

        status = result["status"]
        stats.local_paths.append(destination)
        if status == "downloaded":
            stats.downloaded += 1
            stats.bytes_downloaded += int(result["bytes"])
        elif status == "skipped":
            stats.skipped += 1
        elif status == "missing":
            stats.missing += 1

        print(
            f"Progress downloaded={stats.downloaded} skipped={stats.skipped} "
            f"missing={stats.missing} bytes={stats.bytes_downloaded}"
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "current": index,
                    "planned": stats.planned,
                    "date": item["date"],
                    "hour": item["hour"],
                    "status": status,
                    "path": destination,
                    "downloaded": stats.downloaded,
                    "skipped": stats.skipped,
                    "missing": stats.missing,
                    "bytes_downloaded": stats.bytes_downloaded,
                }
            )

    if not dry_run and stats.downloaded == 0 and stats.skipped == 0:
        raise FileNotFoundError("No archive objects were downloaded or found locally for the requested range.")

    return stats


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download official Hyperliquid historical l2Book archives for one coin over a date range."
    )
    parser.add_argument("--coin", required=True, help="Coin symbol, for example BTC or ETH.")
    parser.add_argument("--date-from", required=True, type=parse_yyyymmdd, help="Start date in YYYYMMDD.")
    parser.add_argument("--date-to", required=True, type=parse_yyyymmdd, help="End date in YYYYMMDD.")
    parser.add_argument("--out-dir", required=True, help="Local output directory for raw .lz4 files.")
    parser.add_argument("--hour-from", type=int, default=0, help="First hour to fetch each day, 0-23.")
    parser.add_argument("--hour-to", type=int, default=23, help="Last hour to fetch each day, 0-23.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing local files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without writing files.")
    parser.add_argument(
        "--request-payer",
        default="requester",
        help="RequestPayer value for S3 requests. Hyperliquid archive uses requester-pays.",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    stats = fetch_historical_l2(
        coin=args.coin,
        date_from=args.date_from,
        date_to=args.date_to,
        out_dir=args.out_dir,
        hour_from=args.hour_from,
        hour_to=args.hour_to,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        request_payer=args.request_payer,
    )
    print(
        "Fetch summary:"
        f" planned={stats.planned}"
        f" downloaded={stats.downloaded}"
        f" skipped={stats.skipped}"
        f" missing={stats.missing}"
        f" bytes_downloaded={stats.bytes_downloaded}"
    )


if __name__ == "__main__":
    main()
