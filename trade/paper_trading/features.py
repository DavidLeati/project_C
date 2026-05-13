from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from trade.features import CryptoFeatureBuilder


TIMEFRAME_WINDOWS = {
    "15m": 48,
    "1h": 24,
    "4h": 42,
    "1d": 30,
}


@dataclass
class LiveFeatureFactory:
    target_symbol: str = "SOLUSDT"

    def __post_init__(self) -> None:
        self.builders = {
            tf: CryptoFeatureBuilder(window_size=window)
            for tf, window in TIMEFRAME_WINDOWS.items()
        }

    def build_frame(
        self,
        *,
        target: pd.DataFrame,
        btc: pd.DataFrame | None,
        eth: pd.DataFrame | None,
        funding: pd.DataFrame | None,
        timeframe: str,
    ) -> pd.DataFrame:
        df = target.copy().sort_values("open_time")
        if btc is not None and not btc.empty:
            btc_df = btc[["open_time", "close"]].rename(columns={"close": "btc_close"})
            df = pd.merge_asof(
                df.sort_values("open_time"),
                btc_df.sort_values("open_time"),
                on="open_time",
                direction="backward",
            )
        if eth is not None and not eth.empty:
            eth_df = eth[["open_time", "close"]].rename(columns={"close": "eth_close"})
            df = pd.merge_asof(
                df.sort_values("open_time"),
                eth_df.sort_values("open_time"),
                on="open_time",
                direction="backward",
            )
        if funding is not None and not funding.empty:
            fund = funding[["fundingTime", "fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
            df = pd.merge_asof(
                df.sort_values("open_time"),
                fund.sort_values("fundingTime"),
                left_on="open_time",
                right_on="fundingTime",
                direction="backward",
            )
            df["funding_rate"] = df["funding_rate"].fillna(0.0)
        else:
            df["funding_rate"] = 0.0
        return self.builders[timeframe].build_live_features_df(df)

    def history_tensor(
        self,
        *,
        target: pd.DataFrame,
        btc: pd.DataFrame | None,
        eth: pd.DataFrame | None,
        funding: pd.DataFrame | None,
        timeframe: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the full history as a [N, F] tensor for warmup."""
        frame = self.build_frame(
            target=target,
            btc=btc,
            eth=eth,
            funding=funding,
            timeframe=timeframe,
        )
        if frame.empty:
            raise ValueError(f"Historico insuficiente para features live em {timeframe}.")
        values = frame[CryptoFeatureBuilder.FEATURE_COLS].to_numpy(dtype="float32")
        return torch.tensor(values, dtype=torch.float32, device=device)

    def latest_tensor(
        self,
        *,
        target: pd.DataFrame,
        btc: pd.DataFrame | None,
        eth: pd.DataFrame | None,
        funding: pd.DataFrame | None,
        timeframe: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, pd.Series]:
        frame = self.build_frame(
            target=target,
            btc=btc,
            eth=eth,
            funding=funding,
            timeframe=timeframe,
        )
        if frame.empty:
            raise ValueError(f"Historico insuficiente para features live em {timeframe}.")
        latest = frame.iloc[-1]
        values = latest[CryptoFeatureBuilder.FEATURE_COLS].to_numpy(dtype="float32")
        return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0), latest
