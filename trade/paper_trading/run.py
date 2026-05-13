from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .binance_client import (
    TESTNET_REST_BASE_URL,
    BinanceClientError,
    BinanceCredentials,
    BinanceFuturesTestnetClient,
    make_combined_kline_stream_url,
)
from .candles import CandleStore, closed_kline_event_to_row, klines_to_frame
from .features import LiveFeatureFactory
from .ledger import LedgerEvent, PaperLedger
from .model import TIMEFRAMES, load_paper_trading_model
from .sizing import SymbolFilters, plan_position_order


DEFAULT_SYMBOLS = ("SOLUSDT", "BTCUSDT", "ETHUSDT")
KNOWN_SIGNED_BASE_URLS = (
    TESTNET_REST_BASE_URL,
    "https://testnet.binancefuture.com",
)


def _load_env_file(path: str | Path | None, *, override: bool = True) -> None:
    if path is None:
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    loaded = 0
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value
            loaded += 1
    mode = "sobrescrevendo ambiente" if override else "preservando ambiente existente"
    _log(f"env carregado de {env_path} ({loaded} variaveis, {mode})")


def _mask_secret(value: str | None) -> str:
    if not value:
        return "<ausente>"
    if len(value) <= 8:
        return f"<len={len(value)}>"
    return f"{value[:4]}...{value[-4:]} (len={len(value)})"


def _candidate_base_urls(preferred: str | None) -> list[str]:
    candidates = []
    if preferred:
        candidates.append(preferred.rstrip("/"))
    for base_url in KNOWN_SIGNED_BASE_URLS:
        normalized = base_url.rstrip("/")
        if normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _validated_client(
    credentials: BinanceCredentials | None,
    *,
    preferred_base_url: str | None,
    require_signed: bool,
) -> BinanceFuturesTestnetClient:
    if not require_signed:
        return BinanceFuturesTestnetClient(credentials, base_url=preferred_base_url) if preferred_base_url else BinanceFuturesTestnetClient(credentials)

    errors = []
    for base_url in _candidate_base_urls(preferred_base_url):
        client = BinanceFuturesTestnetClient(credentials, base_url=base_url)
        _log(f"validando credenciais testnet em {base_url}")
        try:
            client.account()
            _log(f"credenciais aceitas em {base_url}")
            return client
        except BinanceClientError as exc:
            errors.append(f"{base_url}: {exc}")
            _log(f"credenciais rejeitadas em {base_url}: {exc}")
    joined = "\n  - ".join(errors)
    raise RuntimeError(
        "Credenciais rejeitadas em todos os endpoints testnet conhecidos.\n"
        f"  - {joined}\n"
        "Use a API key do mesmo ambiente do endpoint ou rode com --mode ledger_only para validar sinais sem ordens."
    )


def _public_client(preferred_base_url: str | None) -> BinanceFuturesTestnetClient:
    return BinanceFuturesTestnetClient(None, base_url=preferred_base_url) if preferred_base_url else BinanceFuturesTestnetClient(None)


def _credentials_from_env() -> BinanceCredentials | None:
    api_key = os.getenv("BINANCE_FUTURES_TESTNET_API_KEY")
    secret_key = os.getenv("BINANCE_FUTURES_TESTNET_SECRET_KEY")
    if not api_key or not secret_key:
        return None
    return BinanceCredentials(api_key=api_key.strip().strip('"').strip("'"), secret_key=secret_key.strip().strip('"').strip("'"))


def _log(message: str) -> None:
    print(f"[paper] {message}", flush=True)


def _funding_frame(client: BinanceFuturesTestnetClient, symbol: str) -> pd.DataFrame:
    rows = client.funding_rate(symbol, limit=100)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])
    frame["fundingTime"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=False, errors="coerce")
    frame["fundingRate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
    return frame.sort_values("fundingTime").reset_index(drop=True)


def _load_histories(client: BinanceFuturesTestnetClient, store: CandleStore, limit: int) -> None:
    for symbol in DEFAULT_SYMBOLS:
        for timeframe in TIMEFRAMES:
            _log(f"baixando historico {symbol} {timeframe} limit={limit}")
            store.set_history(symbol, timeframe, klines_to_frame(client.klines(symbol, timeframe, limit=limit)))


def _symbol_filters(client: BinanceFuturesTestnetClient, symbol: str) -> SymbolFilters:
    exchange_info = client.exchange_info(symbol)
    symbol_info = next(item for item in exchange_info["symbols"] if item["symbol"] == symbol.upper())
    return SymbolFilters.from_exchange_symbol(symbol_info)


def _current_position_qty(positions: list[dict[str, Any]], symbol: str) -> Decimal:
    for position in positions:
        if position.get("symbol") == symbol.upper() and position.get("positionSide", "BOTH") == "BOTH":
            return Decimal(str(position.get("positionAmt", "0")))
    return Decimal("0")


def _mark_price(positions: list[dict[str, Any]], fallback_price: Decimal, symbol: str) -> Decimal:
    for position in positions:
        if position.get("symbol") == symbol.upper() and position.get("markPrice"):
            value = Decimal(str(position["markPrice"]))
            if value > 0:
                return value
    return fallback_price


def _account_margin_balance(account: dict[str, Any]) -> Decimal:
    return Decimal(str(account.get("totalMarginBalance") or account.get("availableBalance") or "0"))


def _latest_feature_tensors(factory: LiveFeatureFactory, store: CandleStore, funding: pd.DataFrame, symbol: str, device: torch.device):
    tensors = {}
    latest_rows = {}
    for tf in TIMEFRAMES:
        tensor, latest = factory.latest_tensor(
            target=store.get(symbol, tf),
            btc=store.get("BTCUSDT", tf),
            eth=store.get("ETHUSDT", tf),
            funding=funding,
            timeframe=tf,
            device=device,
        )
        tensors[tf] = tensor
        latest_rows[tf] = latest
    return tensors, latest_rows


class PaperTradingRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        _load_env_file(getattr(args, "env_file", ".env"), override=not getattr(args, "no_env_override", False))
        self.credentials = _credentials_from_env()
        if args.mode == "ledger_plus_testnet" and self.credentials is None:
            raise RuntimeError("Defina BINANCE_FUTURES_TESTNET_API_KEY e BINANCE_FUTURES_TESTNET_SECRET_KEY.")

        base_url = getattr(args, "base_url", None) or os.getenv("BINANCE_FUTURES_TESTNET_BASE_URL")
        _log(f"iniciando symbol={args.symbol} mode={args.mode} checkpoint={args.checkpoint}")
        self.auth_available = False
        try:
            self.client = _validated_client(
                self.credentials,
                preferred_base_url=base_url,
                require_signed=args.mode == "ledger_plus_testnet",
            )
            self.auth_available = args.mode == "ledger_plus_testnet"
        except RuntimeError as exc:
            if args.mode != "ledger_plus_testnet":
                raise
            _log(str(exc))
            _log("continuando em ledger local sem ordens testnet; corrija a API key para habilitar execucao")
            self.client = _public_client(base_url)
        _log(f"REST base_url={getattr(self.client, 'base_url', base_url or 'default')}")
        if self.credentials:
            _log(f"API key carregada {_mask_secret(self.credentials.api_key)}")
        self.store = CandleStore(max_rows=max(args.history_limit, 300))
        _load_histories(self.client, self.store, args.history_limit)
        _log(f"baixando funding {args.symbol}")
        self.funding = _funding_frame(self.client, args.symbol)
        _log(f"carregando filtros {args.symbol}")
        self.filters = _symbol_filters(self.client, args.symbol)

        self.device = torch.device(args.device)
        _log(f"carregando modelo em {self.device}")
        self.model = load_paper_trading_model(args.checkpoint, device=self.device)
        self.factory = LiveFeatureFactory(target_symbol=args.symbol)

        # --- Warmup: popula history buffers dos GRU multi-escala ---
        _log("warmup: alimentando history buffers com dados historicos...")
        warmup_tensors: dict[str, torch.Tensor] = {}
        for tf in TIMEFRAMES:
            warmup_tensors[tf] = self.factory.history_tensor(
                target=self.store.get(args.symbol, tf),
                btc=self.store.get("BTCUSDT", tf),
                eth=self.store.get("ETHUSDT", tf),
                funding=self.funding,
                timeframe=tf,
                device=self.device,
            )
            _log(f"  warmup {tf}: {warmup_tensors[tf].shape[0]} amostras")
        self.model.warmup(warmup_tensors)
        _log("warmup concluido")

        self.ledger = PaperLedger(Path(args.artifact_dir))
        self.last_close_price: Decimal | None = None
        self.local_equity = Decimal(str(args.paper_balance))
        self.virtual_position_qty = Decimal("0")
        _log(f"pronto; ledger em {Path(args.artifact_dir)}")

    def decide_and_execute(self) -> dict[str, Any]:
        tensors, latest_rows = _latest_feature_tensors(
            self.factory, self.store, self.funding, self.args.symbol, self.device
        )
        _log(f"decidindo candle fechado {latest_rows['15m']['close_time']}")
        decision = self.model.decide(tensors)

        close_price = Decimal(str(latest_rows["15m"]["close"]))
        if self.auth_available:
            account = self.client.account()
            positions = self.client.position_risk(self.args.symbol)
            current_qty = _current_position_qty(positions, self.args.symbol)
        else:
            account = {"totalMarginBalance": str(self.local_equity)}
            positions = []
            current_qty = self.virtual_position_qty
        margin_balance = _account_margin_balance(account)
        # Cap margin_balance by paper_balance for clean sizing control
        paper_cap = Decimal(str(self.args.paper_balance))
        if paper_cap > 0 and margin_balance > paper_cap:
            margin_balance = paper_cap

        gross_pnl = Decimal("0")
        if self.last_close_price is not None:
            gross_pnl = current_qty * (close_price - self.last_close_price)
            if not self.auth_available:
                self.local_equity += gross_pnl
                margin_balance = self.local_equity

        mark_price = _mark_price(positions, close_price, self.args.symbol)
        plan = plan_position_order(
            model_position=decision.position,
            margin_balance=margin_balance,
            mark_price=mark_price,
            current_position_qty=current_qty,
            filters=self.filters,
        )

        order_response = None
        error = None
        if self.args.mode == "ledger_plus_testnet" and self.auth_available and plan.should_order:
            try:
                _log(f"enviando ordem {plan.side} qty={format(plan.quantity, 'f')} reduceOnly={plan.reduce_only}")
                order_response = self.client.new_market_order(
                    symbol=self.args.symbol,
                    side=plan.side or "",
                    quantity=format(plan.quantity, "f"),
                    reduce_only=plan.reduce_only,
                    new_client_order_id=f"project-c-{int(time.time())}",
                )
            except Exception as exc:
                error = str(exc)
                _log(f"erro ao enviar ordem: {error}")

        estimated_cost = abs(plan.delta_notional) * Decimal("0.0005")
        if not self.auth_available and plan.should_order:
            signed_quantity = plan.quantity if plan.side == "BUY" else -plan.quantity
            self.virtual_position_qty += signed_quantity
            self.local_equity -= estimated_cost
            margin_balance = self.local_equity

        self.ledger.append(
            LedgerEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                symbol=self.args.symbol,
                close_time=str(latest_rows["15m"]["close_time"]),
                model_position=decision.position,
                action=decision.action,
                close_price=float(close_price),
                target_notional=str(plan.target_notional),
                current_notional=str(plan.current_notional),
                delta_notional=str(plan.delta_notional),
                order_side=plan.side,
                order_quantity=str(plan.quantity),
                reduce_only=plan.reduce_only,
                local_equity=float(margin_balance),
                gross_pnl=float(gross_pnl),
                estimated_cost=float(estimated_cost),
                order_response=order_response,
                error=error,
            )
        )
        self.last_close_price = close_price
        _log(f"decisao {decision.action} pos={decision.position:+.4f} close={close_price} plan={plan.reason}")
        return {"decision": decision.__dict__, "order_plan": plan.__dict__, "error": error}

    def handle_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        row = closed_kline_event_to_row(event)
        if row is None:
            return None
        payload = event.get("data", event)
        symbol = payload["s"].upper()
        interval = payload["k"]["i"]
        self.store.append_closed_kline(symbol, interval, row)
        _log(f"candle fechado {symbol} {interval} close={row['close']}")
        if symbol == self.args.symbol.upper() and interval == "15m":
            return self.decide_and_execute()
        return None

    def run_forever(self) -> None:
        try:
            import websocket
        except ImportError as exc:
            raise RuntimeError("Instale websocket-client para o loop continuo: pip install websocket-client") from exc

        url = make_combined_kline_stream_url(DEFAULT_SYMBOLS, TIMEFRAMES)
        backoff = 1
        max_backoff = 60
        while True:
            _log(f"conectando websocket {url}")
            try:
                ws = websocket.create_connection(
                    url,
                    timeout=90,
                    enable_multithread=False,
                )
            except Exception as exc:
                _log(f"falha ao conectar websocket: {exc}; retry em {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue

            backoff = 1  # reset on successful connect
            try:
                _log("websocket conectado; aguardando candles fechados")
                while True:
                    try:
                        message = ws.recv()
                    except (
                        websocket.WebSocketTimeoutException,
                        websocket.WebSocketConnectionClosedException,
                        ConnectionError,
                        OSError,
                    ) as exc:
                        _log(f"websocket erro recv: {type(exc).__name__}: {exc}")
                        break  # sai do inner loop, reconecta no outer loop
                    if not message:
                        continue  # pong ou frame vazio, ignorar
                    try:
                        result = self.handle_event(json.loads(message))
                        if result is not None:
                            print(json.dumps(result, default=str), flush=True)
                    except Exception as exc:
                        _log(f"erro ao processar evento: {exc}")
            except Exception as exc:
                _log(f"erro inesperado no loop ws: {exc}")
            finally:
                try:
                    ws.close()
                except Exception:
                    pass
                _log(f"websocket desconectado; reconectando em {backoff}s")
                time.sleep(backoff)


def run_once(args: argparse.Namespace) -> None:
    runner = PaperTradingRunner(args)
    result = runner.decide_and_execute()
    print(json.dumps(result, default=str))


def run_forever(args: argparse.Namespace) -> None:
    runner = PaperTradingRunner(args)
    runner.run_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper Trading HSAMA na Binance Futures Testnet.")
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--checkpoint", default="models/monolithic_20260505_201200.pt")
    parser.add_argument("--mode", choices=["ledger_plus_testnet", "ledger_only"], default="ledger_plus_testnet")
    parser.add_argument("--artifact-dir", default="artifacts/paper_trading")
    parser.add_argument("--history-limit", type=int, default=1500)
    parser.add_argument("--paper-balance", type=Decimal, default=Decimal("1000"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--base-url", default=None, help="Sobrescreve o endpoint REST testnet da Binance Futures.")
    parser.add_argument("--env-file", default=".env", help="Arquivo .env com credenciais e endpoint testnet.")
    parser.add_argument("--no-env-override", action="store_true", help="Nao sobrescreve variaveis ja existentes no processo.")
    parser.add_argument("--once", action="store_true", help="Executa uma decisao com REST bootstrap e encerra.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.once:
        run_once(args)
    else:
        run_forever(args)


if __name__ == "__main__":
    main()
