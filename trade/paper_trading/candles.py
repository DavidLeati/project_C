from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


def klines_to_frame(rows: list[list[Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    return normalize_candles(frame)


def normalize_candles(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=False, errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=False, errors="coerce")
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return (
        df.sort_values("open_time")
        .drop_duplicates(subset="open_time", keep="last")
        .reset_index(drop=True)
    )


def closed_kline_event_to_row(event: dict[str, Any]) -> dict[str, Any] | None:
    payload = event.get("data", event)
    kline = payload.get("k")
    if not kline or not kline.get("x"):
        return None
    return {
        "open_time": int(kline["t"]),
        "open": kline["o"],
        "high": kline["h"],
        "low": kline["l"],
        "close": kline["c"],
        "volume": kline["v"],
        "close_time": int(kline["T"]),
        "quote_asset_volume": kline["q"],
        "number_of_trades": kline["n"],
        "taker_buy_base_asset_volume": kline["V"],
        "taker_buy_quote_asset_volume": kline["Q"],
        "ignore": kline.get("B", "0"),
    }


@dataclass
class CandleStore:
    max_rows: int = 1000

    def __post_init__(self) -> None:
        self._frames: dict[tuple[str, str], pd.DataFrame] = {}

    def set_history(self, symbol: str, interval: str, frame: pd.DataFrame) -> None:
        self._frames[(symbol.upper(), interval)] = normalize_candles(frame).tail(self.max_rows)

    def append_closed_kline(self, symbol: str, interval: str, row: dict[str, Any]) -> pd.DataFrame:
        key = (symbol.upper(), interval)
        old = self._frames.get(key, pd.DataFrame(columns=KLINE_COLUMNS))
        row_frame = normalize_candles(pd.DataFrame([row], columns=KLINE_COLUMNS))
        merged = pd.concat([old, row_frame], ignore_index=True)
        merged = normalize_candles(merged).tail(self.max_rows)
        self._frames[key] = merged
        return merged

    def get(self, symbol: str, interval: str) -> pd.DataFrame:
        key = (symbol.upper(), interval)
        if key not in self._frames:
            raise KeyError(f"Sem candles para {symbol.upper()} {interval}.")
        return self._frames[key].copy()
