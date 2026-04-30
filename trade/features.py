import pandas as pd
import numpy as np
import torch
from typing import Tuple


class CryptoFeatureBuilder:
    """
    Construtor de Features Temporais Imune a Look-ahead Bias.

    Features geradas (24 no total):

    Grupo A — Momentum (4):
      ret_1, ret_4, ret_16, ret_64

    Grupo B — Osciladores / Bandas / Volatilidade (4):
      rsi, bb_pct, atr_ratio, ema_ratio

    Grupo C — Volume base (3):
      log_volume, vol_trend, vol_surge

    Grupo D — Microestrutura do Candle (4):
      candle_body   — retorno open->close (direcional)
      upper_wick    — rejeicao de topo  (high - max(open,close)) / close
      lower_wick    — suporte de fundo  (min(open,close) - low)  / close
      open_gap      — gap de abertura   (open - close_prev)      / close_prev

    Grupo E — Order Flow / Fluxo de Ordens Binance (4):
      taker_buy_ratio  — % volume comprado agressivamente [0,1]
      taker_buy_delta  — CVD normalizado 2*ratio-1 em [-1,+1]
      trade_intensity  — log(number_of_trades)
      avg_trade_size   — volume / number_of_trades (z-score rolante)

    Grupo F — Inter-mercado (2, opcionais):
      btc_ret_1  — log-retorno BTC no mesmo timeframe
      eth_ret_1  — log-retorno ETH no mesmo timeframe
      (zerado se colunas btc_close/eth_close ausentes no DataFrame)

    Grupo G — Sessao UTC (2):
      hour_sin, hour_cos  — sin/cos do hour-of-day (encoding ciclico)
      (zerado se open_time ausente ou nao-datetime)

    Grupo H — Range intracandle (1, herdado):
      high_low_range

    Todas as features passam por Z-score EMA-rolling (sem look-ahead global)
    e sao clipadas em [-5, 5].
    """

    FEATURE_COLS = [
        # A — Momentum
        "ret_1", "ret_4", "ret_16", "ret_64",
        # B — Osciladores / Bandas
        "rsi", "bb_pct", "atr_ratio", "ema_ratio",
        # C — Volume base
        "log_volume", "vol_trend", "vol_surge",
        # D — Microestrutura do candle
        "candle_body", "upper_wick", "lower_wick", "open_gap",
        # E — Order flow Binance
        "taker_buy_ratio", "taker_buy_delta", "trade_intensity", "avg_trade_size",
        # F — Inter-mercado (opcionais, zerado se ausente)
        "btc_ret_1", "eth_ret_1",
        # G — Sessao UTC
        "hour_sin", "hour_cos",
        # H — Range intracandle
        "high_low_range",
    ]

    def __init__(self, window_size: int = 48):
        # Janela de rolling historica (proporcional ao timeframe)
        self.window_size = window_size

    # ------------------------------------------------------------------
    # Helpers de calculo sem look-ahead
    # ------------------------------------------------------------------

    def _rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """RSI causal via EMA (Wilder). Normalizado para [-1, 1]."""
        delta    = series.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs  = avg_gain / (avg_loss + 1e-8)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return (rsi - 50.0) / 50.0   # [-1, 1] aproximado

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR via EMA de Wilder, normalizado pelo close."""
        hl  = df["high"] - df["low"]
        hcp = (df["high"] - df["close"].shift(1)).abs()
        lcp = (df["low"]  - df["close"].shift(1)).abs()
        tr  = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        return atr / (df["close"] + 1e-8)

    def _zscore_ema(self, series: pd.Series) -> pd.Series:
        """Z-score EMA-rolling causal, clipado em [-5, 5]."""
        mu  = series.ewm(span=self.window_size, adjust=False).mean()
        std = series.ewm(span=self.window_size, adjust=False).std()
        return ((series - mu) / (std + 1e-8)).clip(-5.0, 5.0)

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def build_features_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera as features no DataFrame original e retorna o DataFrame limpo de NaNs.
        """
        df = df.copy()

        # ---- Target (Y) ----
        # log-retorno do candle SEGUINTE (t -> t+1), sem look-ahead
        df["target_return"] = np.log(df["close"].shift(-1) / df["close"])

        # ================================================================
        # GRUPO A — Momentum (4 horizontes)
        # ================================================================
        df["ret_1"]  = np.log(df["close"] / df["close"].shift(1))
        df["ret_4"]  = np.log(df["close"] / df["close"].shift(4))
        df["ret_16"] = np.log(df["close"] / df["close"].shift(16))
        df["ret_64"] = np.log(df["close"] / df["close"].shift(64))

        # ================================================================
        # GRUPO B — Osciladores / Bandas / Volatilidade / Tendencia
        # ================================================================
        df["rsi"] = self._rsi(df["close"], period=14)

        bb_mid      = df["close"].rolling(window=self.window_size).mean()
        bb_std      = df["close"].rolling(window=self.window_size).std()
        df["bb_pct"] = (df["close"] - bb_mid) / (2.0 * bb_std + 1e-8)

        df["atr_ratio"] = self._atr(df, period=14)

        span_short   = max(4, self.window_size // 12)
        ema_short    = df["close"].ewm(span=span_short,       adjust=False).mean()
        ema_long     = df["close"].ewm(span=self.window_size, adjust=False).mean()
        df["ema_ratio"] = (ema_short - ema_long) / (ema_long + 1e-8)

        # ================================================================
        # GRUPO C — Volume base
        # ================================================================
        df["log_volume"] = np.log(df["volume"] + 1e-8)
        ema_vol          = df["log_volume"].ewm(span=self.window_size, adjust=False).mean()
        df["vol_trend"]  = df["log_volume"] - ema_vol
        df["vol_surge"]  = df["log_volume"] - df["log_volume"].rolling(window=4).mean()

        # ================================================================
        # GRUPO D — Microestrutura do Candle
        # ================================================================
        close_prev    = df["close"].shift(1)
        candle_top    = df[["close", "open"]].max(axis=1)
        candle_bottom = df[["close", "open"]].min(axis=1)

        # Body direcional: positivo = candle de alta, negativo = baixa
        df["candle_body"] = (df["close"] - df["open"]) / (df["close"].abs() + 1e-8)

        # Wicks: magnitude de rejeicao (sempre nao-negativos)
        df["upper_wick"] = (df["high"] - candle_top)    / (df["close"] + 1e-8)
        df["lower_wick"] = (candle_bottom - df["low"])  / (df["close"] + 1e-8)

        # Gap de abertura em relacao ao fechamento anterior
        df["open_gap"] = (df["open"] - close_prev) / (close_prev + 1e-8)

        # Range intracandle (herdado do builder original)
        df["high_low_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)

        # ================================================================
        # GRUPO E — Order Flow (colunas Binance opcionais)
        # ================================================================
        has_orderflow = all(c in df.columns for c in [
            "taker_buy_base_asset_volume", "number_of_trades", "quote_asset_volume"
        ])

        if has_orderflow:
            taker_buy = df["taker_buy_base_asset_volume"]
            n_trades  = df["number_of_trades"].clip(lower=1)
            vol       = df["volume"]

            df["taker_buy_ratio"] = taker_buy / (vol + 1e-8)            # [0, 1]
            df["taker_buy_delta"] = 2.0 * df["taker_buy_ratio"] - 1.0   # [-1, +1]
            df["trade_intensity"] = np.log(n_trades + 1e-8)
            df["avg_trade_size"]  = vol / n_trades                       # unidades por trade
        else:
            df["taker_buy_ratio"] = 0.5
            df["taker_buy_delta"] = 0.0
            df["trade_intensity"] = 0.0
            df["avg_trade_size"]  = 0.0

        # ================================================================
        # GRUPO F — Inter-mercado (btc_close / eth_close opcionais)
        # ================================================================
        if "btc_close" in df.columns:
            df["btc_ret_1"] = np.log(df["btc_close"] / (df["btc_close"].shift(1) + 1e-8))
        else:
            df["btc_ret_1"] = 0.0

        if "eth_close" in df.columns:
            df["eth_ret_1"] = np.log(df["eth_close"] / (df["eth_close"].shift(1) + 1e-8))
        else:
            df["eth_ret_1"] = 0.0

        # ================================================================
        # GRUPO G — Sessao UTC (sin/cos do hour-of-day)
        # ================================================================
        if "open_time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["open_time"]):
            hour           = df["open_time"].dt.hour.astype(float)
            df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
            df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        else:
            df["hour_sin"] = 0.0
            df["hour_cos"] = 0.0

        # ================================================================
        # Z-Score EMA-Rolling para colunas em escala bruta
        # ================================================================
        COLS_TO_ZSCORE = [
            "ret_1", "ret_4", "ret_16", "ret_64",
            "rsi", "bb_pct", "atr_ratio", "ema_ratio",
            "log_volume", "vol_trend", "vol_surge",
            "candle_body", "upper_wick", "lower_wick", "open_gap",
            "high_low_range",
            "trade_intensity", "avg_trade_size",
            "btc_ret_1", "eth_ret_1",
        ]
        for col in COLS_TO_ZSCORE:
            df[col] = self._zscore_ema(df[col])

        # Remove NaNs criados pelos shifts e janelas
        df = df.dropna(subset=self.FEATURE_COLS + ["target_return"]).copy()
        return df

    def transform(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recebe DataFrame com colunas minimas, gera as features e retorna os tensores.
        """
        df_feats = self.build_features_df(df)
        X = df_feats[self.FEATURE_COLS].values
        Y = df_feats[["target_return"]].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
