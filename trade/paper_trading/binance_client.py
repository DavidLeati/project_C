from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


TESTNET_REST_BASE_URL = "https://demo-fapi.binance.com"
MARKET_WS_BASE_URL = "wss://fstream.binance.com/market"


class BinanceClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class BinanceCredentials:
    api_key: str
    secret_key: str


class BinanceFuturesTestnetClient:
    def __init__(
        self,
        credentials: BinanceCredentials | None = None,
        *,
        base_url: str = TESTNET_REST_BASE_URL,
        recv_window: int = 5000,
        timeout: float = 15.0,
    ):
        self.credentials = credentials
        self.base_url = base_url.rstrip("/")
        self.recv_window = int(recv_window)
        self.timeout = float(timeout)

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        signed: bool = False,
    ) -> Any:
        params = dict(params or {})
        headers = {"User-Agent": "project-c-paper-trading/1.0"}

        if signed:
            if self.credentials is None:
                raise BinanceClientError("Credenciais da Binance Futures Testnet nao configuradas.")
            params.setdefault("recvWindow", self.recv_window)
            params["timestamp"] = int(time.time() * 1000)
            query = urllib.parse.urlencode(params, doseq=True)
            signature = hmac.new(
                self.credentials.secret_key.encode("utf-8"),
                query.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            query = f"{query}&signature={signature}"
            headers["X-MBX-APIKEY"] = self.credentials.api_key
        else:
            query = urllib.parse.urlencode(params, doseq=True)

        url = f"{self.base_url}{path}"
        data = None
        if method.upper() in {"GET", "DELETE"}:
            if query:
                url = f"{url}?{query}"
        else:
            data = query.encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise BinanceClientError(f"Binance HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise BinanceClientError(f"Erro de conexao com Binance: {exc}") from exc

        if not payload:
            return None
        return json.loads(payload)

    def ping(self) -> Any:
        return self._request("GET", "/fapi/v1/ping")

    def exchange_info(self, symbol: str | None = None) -> dict[str, Any]:
        params = {"symbol": symbol.upper()} if symbol else {}
        return self._request("GET", "/fapi/v1/exchangeInfo", params)

    def klines(self, symbol: str, interval: str, *, limit: int = 500) -> list[list[Any]]:
        return self._request(
            "GET",
            "/fapi/v1/klines",
            {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)},
        )

    def funding_rate(self, symbol: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            "/fapi/v1/fundingRate",
            {"symbol": symbol.upper(), "limit": int(limit)},
        )

    def account(self) -> dict[str, Any]:
        try:
            return self._request("GET", "/fapi/v3/account", signed=True)
        except BinanceClientError as exc:
            fallback_error = exc
        try:
            return self._request("GET", "/fapi/v2/account", signed=True)
        except BinanceClientError as exc:
            raise BinanceClientError(f"{fallback_error}; fallback /fapi/v2/account falhou: {exc}") from exc

    def position_risk(self, symbol: str) -> list[dict[str, Any]]:
        try:
            return self._request("GET", "/fapi/v3/positionRisk", {"symbol": symbol.upper()}, signed=True)
        except BinanceClientError as exc:
            fallback_error = exc
        try:
            return self._request("GET", "/fapi/v2/positionRisk", {"symbol": symbol.upper()}, signed=True)
        except BinanceClientError as exc:
            raise BinanceClientError(f"{fallback_error}; fallback /fapi/v2/positionRisk falhou: {exc}") from exc

    def new_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: str,
        reduce_only: bool = False,
        new_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "MARKET",
            "quantity": quantity,
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        if new_client_order_id:
            params["newClientOrderId"] = new_client_order_id[:36]
        return self._request("POST", "/fapi/v1/order", params, signed=True)


def make_combined_kline_stream_url(symbols: Iterable[str], intervals: Iterable[str]) -> str:
    streams = [
        f"{symbol.lower()}@kline_{interval}"
        for symbol in symbols
        for interval in intervals
    ]
    return f"{MARKET_WS_BASE_URL}/stream?streams={'/'.join(streams)}"
