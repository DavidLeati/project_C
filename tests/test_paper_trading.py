from __future__ import annotations

import argparse
import os
from decimal import Decimal

import pandas as pd
import torch

from trade.features import CryptoFeatureBuilder
from trade.paper_trading.candles import closed_kline_event_to_row
from trade.paper_trading.binance_client import BinanceCredentials
from trade.paper_trading.model import load_paper_trading_model
from trade.paper_trading.run import PaperTradingRunner, _load_env_file
from trade.paper_trading.sizing import SymbolFilters, plan_position_order


def _sample_candles(rows: int = 120) -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01")
    values = []
    for i in range(rows):
        close = 100.0 + i * 0.1
        values.append(
            {
                "open_time": base + pd.Timedelta(minutes=15 * i),
                "open": close - 0.05,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 1000 + i,
                "close_time": base + pd.Timedelta(minutes=15 * i + 14),
                "quote_asset_volume": 100000 + i,
                "number_of_trades": 100 + i,
                "taker_buy_base_asset_volume": 500 + i,
                "taker_buy_quote_asset_volume": 50000 + i,
            }
        )
    return pd.DataFrame(values)


def test_live_features_include_latest_row_without_future_target():
    builder = CryptoFeatureBuilder(window_size=48)
    frame = _sample_candles()

    live = builder.build_live_features_df(frame)
    train = builder.build_features_df(frame)

    assert live.iloc[-1]["open_time"] == frame.iloc[-1]["open_time"]
    assert train.iloc[-1]["open_time"] < frame.iloc[-1]["open_time"]
    assert list(live[builder.FEATURE_COLS].columns) == builder.FEATURE_COLS


def test_historical_live_features_match_training_feature_columns():
    builder = CryptoFeatureBuilder(window_size=48)
    frame = _sample_candles()

    live = builder.build_live_features_df(frame)
    train = builder.build_features_df(frame)
    common_open_time = train.iloc[-1]["open_time"]

    live_row = live[live["open_time"] == common_open_time].iloc[0]
    train_row = train[train["open_time"] == common_open_time].iloc[0]

    pd.testing.assert_series_equal(
        live_row[builder.FEATURE_COLS],
        train_row[builder.FEATURE_COLS],
        check_names=False,
    )


def test_checkpoint_loader_restores_monolithic_shapes():
    model = load_paper_trading_model("models/monolithic_20260505_201200.pt", device=torch.device("cpu"))

    assert set(model.predictors) == {"15m", "1h", "4h", "1d"}
    assert all(runtime.model.in_features == 28 for runtime in model.predictors.values())
    assert model.trader.model.in_features == 32
    assert tuple(model.normalizer.mu.shape) == (1, 4)
    assert tuple(model.normalizer.var.shape) == (1, 4)


def test_position_sizing_handles_long_short_flat_and_rounding():
    filters = SymbolFilters(
        quantity_step=Decimal("0.1"),
        min_quantity=Decimal("0.1"),
        min_notional=Decimal("5"),
    )

    long_plan = plan_position_order(
        model_position=0.5,
        margin_balance=Decimal("100"),
        mark_price=Decimal("20"),
        current_position_qty=Decimal("0"),
        filters=filters,
    )
    assert long_plan.side == "BUY"
    assert long_plan.quantity == Decimal("2.5")

    short_plan = plan_position_order(
        model_position=-0.5,
        margin_balance=Decimal("100"),
        mark_price=Decimal("20"),
        current_position_qty=Decimal("2"),
        filters=filters,
    )
    assert short_plan.side == "SELL"
    assert short_plan.quantity == Decimal("4.5")

    flat_plan = plan_position_order(
        model_position=0.01,
        margin_balance=Decimal("100"),
        mark_price=Decimal("20"),
        current_position_qty=Decimal("0"),
        filters=filters,
    )
    assert not flat_plan.should_order
    assert flat_plan.reason == "below_min_notional"


def test_closed_kline_event_parses_only_final_candle():
    event = {
        "data": {
            "s": "SOLUSDT",
            "k": {
                "t": 1,
                "T": 2,
                "i": "15m",
                "x": True,
                "o": "1",
                "h": "2",
                "l": "0.5",
                "c": "1.5",
                "v": "10",
                "q": "15",
                "n": 3,
                "V": "6",
                "Q": "9",
                "B": "0",
            },
        }
    }
    assert closed_kline_event_to_row(event)["close"] == "1.5"
    event["data"]["k"]["x"] = False
    assert closed_kline_event_to_row(event) is None


def test_load_env_file_overrides_existing_values_by_default(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "BINANCE_FUTURES_TESTNET_API_KEY='abc'",
                'BINANCE_FUTURES_TESTNET_SECRET_KEY="def"',
                "BINANCE_FUTURES_TESTNET_BASE_URL=https://testnet.binancefuture.com",
                "EXISTING=value_from_file",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("BINANCE_FUTURES_TESTNET_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_FUTURES_TESTNET_SECRET_KEY", raising=False)
    monkeypatch.delenv("BINANCE_FUTURES_TESTNET_BASE_URL", raising=False)
    monkeypatch.setenv("EXISTING", "manual")

    _load_env_file(env_path)

    assert os.environ["BINANCE_FUTURES_TESTNET_API_KEY"] == "abc"
    assert os.environ["BINANCE_FUTURES_TESTNET_SECRET_KEY"] == "def"
    assert os.environ["BINANCE_FUTURES_TESTNET_BASE_URL"] == "https://testnet.binancefuture.com"
    assert os.environ["EXISTING"] == "value_from_file"


def test_load_env_file_can_preserve_existing_values(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("EXISTING=value_from_file", encoding="utf-8")
    monkeypatch.setenv("EXISTING", "manual")

    _load_env_file(env_path, override=False)

    assert os.environ["EXISTING"] == "manual"


def test_runner_handles_mocked_closed_15m_event(tmp_path, monkeypatch):
    class DummyClient:
        def __init__(self, *_args, **_kwargs):
            self.order_payload = None

        def klines(self, symbol, interval, limit=500):
            return [
                [
                    int(row.open_time.timestamp() * 1000),
                    str(row.open),
                    str(row.high),
                    str(row.low),
                    str(row.close),
                    str(row.volume),
                    int(row.close_time.timestamp() * 1000),
                    str(row.quote_asset_volume),
                    int(row.number_of_trades),
                    str(row.taker_buy_base_asset_volume),
                    str(row.taker_buy_quote_asset_volume),
                    "0",
                ]
                for row in _sample_candles(limit).itertuples()
            ]

        def funding_rate(self, symbol, limit=100):
            return [{"symbol": symbol, "fundingRate": "0.0001", "fundingTime": 0}]

        def exchange_info(self, symbol):
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "filters": [
                            {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.1", "minQty": "0.1"},
                            {"filterType": "MIN_NOTIONAL", "notional": "5"},
                        ],
                    }
                ]
            }

        def account(self):
            return {"totalMarginBalance": "100"}

        def position_risk(self, symbol):
            return [{"symbol": symbol, "positionSide": "BOTH", "positionAmt": "0", "markPrice": "110"}]

        def new_market_order(self, **kwargs):
            self.order_payload = kwargs
            return {"status": "FILLED", **kwargs}

    class DummyDecision:
        position = 0.5
        action = "LONG"
        logits = 1.0
        edges = {"15m": 0.1, "1h": 0.1, "4h": 0.1, "1d": 0.1}

    class DummyModel:
        def __init__(self):
            self.device = torch.device("cpu")

        def decide(self, tensors):
            return DummyDecision()

    monkeypatch.setattr("trade.paper_trading.run.BinanceFuturesTestnetClient", DummyClient)
    monkeypatch.setattr("trade.paper_trading.run.load_paper_trading_model", lambda *_args, **_kwargs: DummyModel())
    monkeypatch.setattr(
        "trade.paper_trading.run._credentials_from_env",
        lambda: BinanceCredentials(api_key="abcd1234", secret_key="secret1234"),
    )

    args = argparse.Namespace(
        symbol="SOLUSDT",
        checkpoint="models/monolithic_20260505_201200.pt",
        mode="ledger_plus_testnet",
        artifact_dir=str(tmp_path),
        history_limit=120,
        paper_balance=Decimal("100"),
        device="cpu",
        base_url=None,
        env_file=None,
        no_env_override=False,
    )
    runner = PaperTradingRunner(args)
    event = {
        "data": {
            "s": "SOLUSDT",
            "k": {
                "t": int(pd.Timestamp("2026-01-02").timestamp() * 1000),
                "T": int(pd.Timestamp("2026-01-02 00:14").timestamp() * 1000),
                "i": "15m",
                "x": True,
                "o": "110",
                "h": "111",
                "l": "109",
                "c": "110",
                "v": "100",
                "q": "11000",
                "n": 10,
                "V": "50",
                "Q": "5500",
                "B": "0",
            },
        }
    }

    result = runner.handle_event(event)

    assert result["order_plan"]["side"] == "BUY"
    assert (tmp_path / "events.jsonl").exists()


def test_runner_degrades_to_ledger_when_auth_is_rejected(tmp_path, monkeypatch):
    class RejectingSignedClient:
        def __init__(self, credentials=None, base_url=None):
            self.base_url = base_url or "https://demo-fapi.binance.com"
            self.credentials = credentials

        def account(self):
            raise RuntimeError("auth rejected")

    class PublicClient:
        def __init__(self, credentials=None, base_url=None):
            self.base_url = base_url or "https://demo-fapi.binance.com"

        def klines(self, symbol, interval, limit=500):
            return [
                [
                    int(row.open_time.timestamp() * 1000),
                    str(row.open),
                    str(row.high),
                    str(row.low),
                    str(row.close),
                    str(row.volume),
                    int(row.close_time.timestamp() * 1000),
                    str(row.quote_asset_volume),
                    int(row.number_of_trades),
                    str(row.taker_buy_base_asset_volume),
                    str(row.taker_buy_quote_asset_volume),
                    "0",
                ]
                for row in _sample_candles(limit).itertuples()
            ]

        def funding_rate(self, symbol, limit=100):
            return [{"symbol": symbol, "fundingRate": "0.0001", "fundingTime": 0}]

        def exchange_info(self, symbol):
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "filters": [
                            {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.1", "minQty": "0.1"},
                            {"filterType": "MIN_NOTIONAL", "notional": "5"},
                        ],
                    }
                ]
            }

    class DummyModel:
        device = torch.device("cpu")

        def decide(self, tensors):
            class Decision:
                position = 0.5
                action = "LONG"
                logits = 1.0
                edges = {"15m": 0.1, "1h": 0.1, "4h": 0.1, "1d": 0.1}

            return Decision()

    monkeypatch.setattr("trade.paper_trading.run.BinanceFuturesTestnetClient", RejectingSignedClient)
    monkeypatch.setattr("trade.paper_trading.run._public_client", lambda base_url: PublicClient(base_url=base_url))
    monkeypatch.setattr("trade.paper_trading.run.load_paper_trading_model", lambda *_args, **_kwargs: DummyModel())
    monkeypatch.setattr(
        "trade.paper_trading.run._credentials_from_env",
        lambda: BinanceCredentials(api_key="abcd1234", secret_key="secret1234"),
    )

    args = argparse.Namespace(
        symbol="SOLUSDT",
        checkpoint="models/monolithic_20260505_201200.pt",
        mode="ledger_plus_testnet",
        artifact_dir=str(tmp_path),
        history_limit=120,
        paper_balance=Decimal("100"),
        device="cpu",
        base_url=None,
        env_file=None,
        no_env_override=False,
    )

    runner = PaperTradingRunner(args)

    assert runner.auth_available is False


def test_local_ledger_updates_virtual_position_and_pnl(tmp_path, monkeypatch):
    class PublicClient:
        def __init__(self, *_args, **_kwargs):
            self.base_url = "https://demo-fapi.binance.com"

        def klines(self, symbol, interval, limit=500):
            return [
                [
                    int(row.open_time.timestamp() * 1000),
                    str(row.open),
                    str(row.high),
                    str(row.low),
                    str(row.close),
                    str(row.volume),
                    int(row.close_time.timestamp() * 1000),
                    str(row.quote_asset_volume),
                    int(row.number_of_trades),
                    str(row.taker_buy_base_asset_volume),
                    str(row.taker_buy_quote_asset_volume),
                    "0",
                ]
                for row in _sample_candles(limit).itertuples()
            ]

        def funding_rate(self, symbol, limit=100):
            return [{"symbol": symbol, "fundingRate": "0.0001", "fundingTime": 0}]

        def exchange_info(self, symbol):
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "filters": [
                            {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.1", "minQty": "0.1"},
                            {"filterType": "MIN_NOTIONAL", "notional": "5"},
                        ],
                    }
                ]
            }

    class ShortModel:
        device = torch.device("cpu")

        def decide(self, tensors):
            class Decision:
                position = -1.0
                action = "SHORT"
                logits = -5.0
                edges = {"15m": -0.1, "1h": -0.1, "4h": -0.1, "1d": -0.1}

            return Decision()

    monkeypatch.setattr("trade.paper_trading.run.BinanceFuturesTestnetClient", PublicClient)
    monkeypatch.setattr("trade.paper_trading.run.load_paper_trading_model", lambda *_args, **_kwargs: ShortModel())
    monkeypatch.setattr("trade.paper_trading.run._credentials_from_env", lambda: None)

    args = argparse.Namespace(
        symbol="SOLUSDT",
        checkpoint="models/monolithic_20260505_201200.pt",
        mode="ledger_only",
        artifact_dir=str(tmp_path),
        history_limit=120,
        paper_balance=Decimal("100"),
        device="cpu",
        base_url=None,
        env_file=None,
        no_env_override=False,
    )
    runner = PaperTradingRunner(args)

    first = runner.decide_and_execute()
    second = runner.decide_and_execute()

    assert Decimal(str(first["order_plan"]["current_notional"])) == Decimal("0.0")
    assert runner.virtual_position_qty < 0
    assert Decimal(str(second["order_plan"]["current_notional"])) < 0
