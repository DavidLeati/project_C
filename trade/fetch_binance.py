import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from binance.client import Client
    from binance.enums import HistoricalKlinesType
except ImportError:
    raise ImportError("Please install python-binance: pip install python-binance")

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


def normalize_candles(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", errors="coerce")
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

    df = (
        df.sort_values("open_time")
        .drop_duplicates(subset="open_time", keep="last")
        .reset_index(drop=True)
    )
    return df


def fetch_binance_data(
    symbol: str, interval: str, days: int, market: str = "futures_usdm"
) -> pd.DataFrame:
    client = Client()
    start_str = f"{days} days ago UTC"

    print(f"\n[DATA] ----------------------------------------------------")
    print(f"[DATA] Iniciando download {symbol} {interval} ({days} dias)...")

    if market == "futures_usdm":
        klines = []
        for idx, kline in enumerate(
            client.get_historical_klines_generator(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                klines_type=HistoricalKlinesType.FUTURES,
            ),
            start=1,
        ):
            klines.append(kline)
            if idx % 5000 == 0:
                last_open = pd.to_datetime(kline[0], unit="ms", errors="coerce")
                print(f"[DATA]   Baixando: {idx} candles | último open_time={last_open}", flush=True)
    else:
        print("[DATA]   Usando endpoint spot...", flush=True)
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str,
        )

    print(f"[DATA]   Tamanho bruto retornado: {len(klines)} candles.")
    frame = pd.DataFrame(klines, columns=KLINE_COLUMNS)
    return normalize_candles(frame)


def save_data(df: pd.DataFrame, save_dir: Path, symbol: str, interval: str, days: int) -> Path:
    safe_symbol = symbol.upper()
    safe_interval = interval.replace("/", "_")
    path = save_dir / f"{safe_symbol}_{safe_interval}_{days}d.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[DATA]   Salvo com sucesso em: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Puxa dados da Binance no padrão do projeto HSAMA.")
    parser.add_argument("--symbol", type=str, default="SOLUSDT", help="Símbolo para baixar (ex: SOLUSDT, BTCUSDT)")
    parser.add_argument("--days", type=int, default=1825, help="Quantidade de dias (ex: 1825 para ~5 anos)")
    parser.add_argument("--market", type=str, default="futures_usdm", choices=["futures_usdm", "spot"])
    args = parser.parse_args()

    intervals = ["15m", "1h", "4h", "1d"]
    # Salvar em "data" no mesmo diretório deste script
    save_dir = Path(__file__).parent / "data"

    print(f"[DATA] O diretório de salvamento será: {save_dir}")

    for interval in intervals:
        df = fetch_binance_data(
            symbol=args.symbol,
            interval=interval,
            days=args.days,
            market=args.market,
        )
        save_data(df, save_dir, args.symbol, interval, args.days)

    print(f"\n[DATA] Todos os downloads concluídos!")


if __name__ == "__main__":
    main()
