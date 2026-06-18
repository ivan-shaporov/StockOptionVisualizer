"""Download historical option prices from marketdata.app.

Configuration is read from a .env file (or process env). Run:

    python download_option_data.py

Required env vars:
    MARKETDATA_TOKEN   API token
    SYMBOL             underlying ticker, e.g. AAPL
    SIDE               'put' or 'call'
    QUOTE_DAYS_BACK    calendar days before QUOTE_LATEST_DATE to include
    EXP_FROM           earliest expiration, YYYY-MM-DD
    EXP_TO             latest expiration, YYYY-MM-DD

Optional:
    QUOTE_LATEST_DATE (defaults to today),
    STRIKE, DELTA, RANGE (itm/otm/all),
    MIN_VOLUME, MIN_OPEN_INTEREST,
    LOOKBACK_STEP_DAYS,
    WRITE_CHUNK_FILES,
    OUT_DIR (default ./data)
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

API_ROOT = "https://api.marketdata.app/v1"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class MarketDataError(RuntimeError):
    pass


class LatestAvailableSessionError(MarketDataError):
    def __init__(self, url: str, latest_available_date: str, detail: str) -> None:
        super().__init__(f"HTTP 402 from {url}: {detail}")
        self.latest_available_date = latest_available_date


LATEST_AVAILABLE_DATE_PATTERN = re.compile(
    r"latest available is (\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Config:
    token: str
    symbol: str
    side: str
    quote_latest_date: str
    quote_days_back: int
    exp_from: str
    exp_to: str
    lookback_step_days: int
    out_dir_root: Path
    write_chunk_files: bool
    extra: dict[str, Any]

    @property
    def quote_start_date(self) -> str:
        return _quote_window_bounds(self.quote_latest_date, self.quote_days_back)[0]

    @property
    def quote_end_date_exclusive(self) -> str:
        return _quote_window_bounds(self.quote_latest_date, self.quote_days_back)[1]

    @classmethod
    def from_env(cls) -> "Config":
        env_path = Path(".env")
        if load_dotenv is not None:
            load_dotenv(dotenv_path=env_path)
        elif env_path.exists():
            sys.exit("error: install python-dotenv (pip install -r requirements.txt)")

        side = _require_env("SIDE").lower()
        if side not in ("put", "call"):
            sys.exit("error: SIDE must be 'put' or 'call'")

        extra: dict[str, Any] = {}
        if value := os.environ.get("STRIKE"):
            extra["strike"] = value
        if value := os.environ.get("DELTA"):
            extra["delta"] = value
        if value := os.environ.get("RANGE"):
            extra["range"] = value
        if (value := _optional_int_env("MIN_VOLUME")) is not None:
            extra["minVolume"] = value
        if (value := _optional_int_env("MIN_OPEN_INTEREST")) is not None:
            extra["minOpenInterest"] = value

        lookback_step_days = _optional_int_env("LOOKBACK_STEP_DAYS")
        if lookback_step_days is None:
            lookback_step_days = 7
        elif lookback_step_days <= 0:
            raise SystemExit("error: LOOKBACK_STEP_DAYS must be a positive integer")

        write_chunk_files = _optional_bool_env("WRITE_CHUNK_FILES")
        if write_chunk_files is None:
            write_chunk_files = False

        quote_latest_date = os.environ.get("QUOTE_LATEST_DATE", "").strip()
        if not quote_latest_date:
            quote_latest_date = date.today().isoformat()
        quote_days_back = _optional_int_env("QUOTE_DAYS_BACK")
        if quote_days_back is None:
            sys.exit("error: missing required env var QUOTE_DAYS_BACK (set in .env)")
        if quote_days_back < 0:
            raise SystemExit("error: QUOTE_DAYS_BACK must be zero or a positive integer")

        exp_from = _require_env("EXP_FROM")
        exp_to = _require_env("EXP_TO")
        if _parse_iso_date(exp_from, "EXP_FROM") > _parse_iso_date(exp_to, "EXP_TO"):
            raise SystemExit("error: EXP_FROM must be earlier than or equal to EXP_TO")

        return cls(
            token=_require_env("MARKETDATA_TOKEN"),
            symbol=_require_env("SYMBOL").upper(),
            side=side,
            quote_latest_date=quote_latest_date,
            quote_days_back=quote_days_back,
            exp_from=exp_from,
            exp_to=exp_to,
            lookback_step_days=lookback_step_days,
            out_dir_root=Path(os.environ.get("OUT_DIR", "./data")),
            write_chunk_files=write_chunk_files,
            extra=extra,
        )


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _get(
    url: str,
    token: str,
    params: dict[str, Any] | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    client = session or requests
    last_error: Exception | None = None
    response: requests.Response | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.get(
                url,
                headers=_auth_headers(token),
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES - 1:
                raise MarketDataError(f"Request failed for {url}: {exc}") from exc
            time.sleep(2 ** attempt)
            continue
        except requests.HTTPError as exc:
            error_response = exc.response or response
            if error_response is None:
                raise MarketDataError(f"HTTP error from {url}: {exc}") from exc
            raw_detail = error_response.text.strip().replace("\n", " ")
            latest_available_date = _extract_latest_available_date(raw_detail)
            detail = raw_detail
            if len(detail) > 200:
                detail = f"{detail[:200]}..."
            if error_response.status_code == 402 and latest_available_date is not None:
                raise LatestAvailableSessionError(
                    url,
                    latest_available_date,
                    detail,
                ) from exc
            raise MarketDataError(
                f"HTTP {error_response.status_code} from {url}: {detail}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise MarketDataError(f"Non-JSON response from {url}") from exc

        if not isinstance(data, dict):
            raise MarketDataError(
                f"Unexpected response type from {url}: {type(data).__name__}"
            )
        if data.get("s") == "no_data":
            return {"s": "no_data"}
        if data.get("s") != "ok":
            raise MarketDataError(f"API error from {url}: {data}")
        return data

    raise MarketDataError(f"Request failed repeatedly for {url}") from last_error


def _require_list_column(
    data: dict[str, Any], key: str, context: str
) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise MarketDataError(f"{context} is missing list column '{key}'")
    return value


def _quote_columns(
    data: dict[str, Any], keys: list[str], option_symbol: str
) -> dict[str, list[Any]]:
    context = f"quote history for {option_symbol}"
    updated = _require_list_column(data, "updated", context)
    expected_len = len(updated)
    columns: dict[str, list[Any]] = {"updated": updated}

    for key in keys[1:]:
        value = data.get(key)
        if value is None:
            columns[key] = [None] * expected_len
            continue
        if not isinstance(value, list):
            raise MarketDataError(f"{context} has non-list column '{key}'")
        if len(value) != expected_len:
            raise MarketDataError(
                f"{context} has mismatched column '{key}' ({len(value)} != {expected_len})"
            )
        columns[key] = value

    return columns


def fetch_chain_symbols(
    token: str,
    symbol: str,
    side: str,
    exp_from: str,
    exp_to: str,
    snapshot_date: str | None = None,
    extra: dict[str, Any] | None = None,
    session: requests.Session | None = None,
) -> list[str]:
    params: dict[str, Any] = {
        "side": side,
        "from": exp_from,
        "to": exp_to,
        "dateformat": "timestamp",
    }
    if snapshot_date:
        params["date"] = snapshot_date
    if extra:
        params.update(extra)
    data = _get(f"{API_ROOT}/options/chain/{symbol}/", token, params=params, session=session)
    if data.get("s") == "no_data":
        return []
    option_symbols = _require_list_column(data, "optionSymbol", f"chain response for {symbol}")
    if not all(isinstance(option_symbol, str) for option_symbol in option_symbols):
        raise MarketDataError(f"chain response for {symbol} has invalid optionSymbol values")
    return option_symbols


def fetch_quote_history(
    token: str,
    option_symbol: str,
    quote_start_date: str,
    quote_end_date_exclusive: str,
    session: requests.Session | None = None,
    underlying: str | None = None,
) -> list[dict[str, Any]]:
    strike_price, maturity_date = _contract_fields_from_symbol(option_symbol)
    data = _get(
        f"{API_ROOT}/options/quotes/{option_symbol}/",
        token,
        params={
            "from": quote_start_date,
            "to": quote_end_date_exclusive,
            "dateformat": "timestamp",
        },
        session=session,
    )
    if data.get("s") == "no_data":
        return []
    keys = [
        "updated", "bid", "bidSize", "mid", "ask", "askSize",
        "last", "volume", "openInterest",
        "underlyingPrice", "iv", "delta", "gamma", "theta", "vega", "rho",
        "intrinsicValue", "extrinsicValue",
    ]
    columns = _quote_columns(data, keys, option_symbol)
    n = len(columns["updated"])
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {
            "underlying": underlying,
            "optionSymbol": option_symbol,
            "strikePrice": strike_price,
            "maturityDate": maturity_date,
        }
        for k in keys:
            row[k] = columns[k][i]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"error: missing required env var {name} (set in .env)")
    return v


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"error: {name} must be an integer") from exc


def _optional_bool_env(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(
        f"error: {name} must be a boolean (1/0, true/false, yes/no, on/off)"
    )


def _extract_latest_available_date(detail: str) -> str | None:
    match = LATEST_AVAILABLE_DATE_PATTERN.search(detail)
    if match is None:
        return None

    latest_available_date = match.group(1)
    try:
        date.fromisoformat(latest_available_date)
    except ValueError:
        return None
    return latest_available_date


def _parse_iso_date(value: str, name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"error: {name} must be YYYY-MM-DD") from exc


def _quote_window_bounds(
    quote_latest_date: str,
    quote_days_back: int,
) -> tuple[str, str]:
    latest_day = _parse_iso_date(quote_latest_date, "QUOTE_LATEST_DATE")
    return (
        (latest_day - timedelta(days=quote_days_back)).isoformat(),
        (latest_day + timedelta(days=1)).isoformat(),
    )


def _contract_fields_from_symbol(option_symbol: str) -> tuple[float, str]:
    if len(option_symbol) < 15:
        raise MarketDataError(f"option symbol has unexpected format: {option_symbol}")

    expiry_text = option_symbol[-15:-9]
    strike_text = option_symbol[-8:]

    try:
        maturity_date = date(
            2000 + int(expiry_text[0:2]),
            int(expiry_text[2:4]),
            int(expiry_text[4:6]),
        ).isoformat()
        strike_price = int(strike_text) / 1000
    except ValueError as exc:
        raise MarketDataError(f"option symbol has unexpected format: {option_symbol}") from exc

    return strike_price, maturity_date


def collect_chain_symbols(
    token: str,
    symbol: str,
    side: str,
    quote_start_date: str,
    exp_from: str,
    exp_to: str,
    quote_latest_date: str,
    lookback_step_days: int,
    extra: dict[str, Any] | None = None,
    session: requests.Session | None = None,
    today: date | None = None,
) -> list[str]:
    if lookback_step_days <= 0:
        raise ValueError("lookback_step_days must be positive")

    floor = _parse_iso_date(quote_start_date, "derived quote start date")
    latest = _parse_iso_date(quote_latest_date, "QUOTE_LATEST_DATE")
    if floor > latest:
        raise MarketDataError(
            "derived quote start date must not be later than QUOTE_LATEST_DATE"
        )

    start = min(today or date.today(), latest)

    if start < floor:
        raise MarketDataError(
            "The requested quote window starts after the latest available chain "
            f"snapshot date ({start.isoformat()}). QUOTE_LATEST_DATE is the last "
            "quote day to include; keep future expirations in EXP_FROM/EXP_TO."
        )

    collected: list[str] = []
    seen: set[str] = set()
    snapshot_day = start

    while snapshot_day >= floor:
        snapshot_symbols = fetch_chain_symbols(
            token,
            symbol,
            side,
            exp_from,
            exp_to,
            snapshot_day.isoformat(),
            extra,
            session=session,
        )
        new_count = 0
        for option_symbol in snapshot_symbols:
            if option_symbol not in seen:
                seen.add(option_symbol)
                collected.append(option_symbol)
                new_count += 1
        print(f"  snapshot {snapshot_day.isoformat()}: +{new_count} new symbols")
        snapshot_day -= timedelta(days=lookback_step_days)

    return collected


def main() -> int:
    config = Config.from_env()
    effective_quote_latest_date = config.quote_latest_date

    while True:
        quote_start_date, quote_end_date_exclusive = _quote_window_bounds(
            effective_quote_latest_date,
            config.quote_days_back,
        )

        try:
            with requests.Session() as session:
                print(
                    f"Fetching {config.side} chain for {config.symbol}, "
                    f"quotes latest {effective_quote_latest_date} "
                    f"({config.quote_days_back} days back), "
                    f"exp {config.exp_from}..{config.exp_to}"
                )
                contracts = collect_chain_symbols(
                    config.token,
                    config.symbol,
                    config.side,
                    quote_start_date,
                    config.exp_from,
                    config.exp_to,
                    effective_quote_latest_date,
                    config.lookback_step_days,
                    config.extra,
                    session=session,
                )
                print(f"  {len(contracts)} contracts matched")
                if not contracts:
                    return 0

                out_dir = config.out_dir_root / config.symbol / config.side
                out_dir.mkdir(parents=True, exist_ok=True)

                combined: list[dict[str, Any]] = []
                for i, sym in enumerate(contracts, 1):
                    print(f"[{i}/{len(contracts)}] {sym}", end=" ", flush=True)
                    try:
                        rows = fetch_quote_history(
                            config.token,
                            sym,
                            quote_start_date,
                            quote_end_date_exclusive,
                            session=session,
                            underlying=config.symbol,
                        )
                    except MarketDataError as exc:
                        print(f"FAILED: {exc}", file=sys.stderr)
                        continue
                    print(f"-> {len(rows)} rows")
                    if rows:
                        if config.write_chunk_files:
                            write_csv(out_dir / f"{sym}.csv", rows)
                        combined.extend(rows)

                if combined:
                    combined_path = out_dir / f"all_{config.side}s.csv"
                    write_csv(combined_path, combined)
                    print(f"Wrote {len(combined)} rows to {combined_path}")
        except LatestAvailableSessionError as exc:
            if exc.latest_available_date >= effective_quote_latest_date:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(
                "warning: marketdata.app plan only allows fully-closed sessions; "
                f"retrying with QUOTE_LATEST_DATE={exc.latest_available_date}",
                file=sys.stderr,
            )
            effective_quote_latest_date = exc.latest_available_date
            continue
        except MarketDataError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
