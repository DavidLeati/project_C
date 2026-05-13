from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.hsama import HSAMA
from src.runtime.online import HSAMAOnlineRuntime, HSAMARuntimeConfig
from src.runtime.multiscale import MultiScaleT0Config, T0ScaleSpec
from trade.loss import position_from_logits


TIMEFRAMES = ("15m", "1h", "4h", "1d")
PREDICTOR_SIGMA_FLOOR = 1e-4
PREDICTOR_SIGMA_OFFSET = 8.0
PREDICTOR_EDGE_CLIP = 2.0
TRADER_POSITION_DEADZONE = 0.15


def _make_predictor_config(tf: str) -> MultiScaleT0Config:
    scale_map = {
        "15m": [
            T0ScaleSpec("fast", stride=1, history_length=16, hidden_dim=16),
            T0ScaleSpec("slow", stride=4, history_length=24, hidden_dim=16),
        ],
        "1h": [
            T0ScaleSpec("fast", stride=1, history_length=12, hidden_dim=16),
            T0ScaleSpec("slow", stride=6, history_length=24, hidden_dim=16),
        ],
        "4h": [
            T0ScaleSpec("fast", stride=1, history_length=7, hidden_dim=16),
            T0ScaleSpec("slow", stride=6, history_length=14, hidden_dim=16),
        ],
        "1d": [
            T0ScaleSpec("fast", stride=1, history_length=7, hidden_dim=16),
            T0ScaleSpec("slow", stride=4, history_length=14, hidden_dim=16),
        ],
    }
    return MultiScaleT0Config(scales=scale_map[tf])


def _make_trade_config() -> MultiScaleT0Config:
    return MultiScaleT0Config(
        scales=[
            T0ScaleSpec("micro", stride=1, history_length=8, hidden_dim=16),
            T0ScaleSpec("short", stride=4, history_length=12, hidden_dim=16),
        ]
    )


def _predictor_mu_sigma(preds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = preds[:, 0:1]
    sigma = F.softplus(preds[:, 1:2] - PREDICTOR_SIGMA_OFFSET) + PREDICTOR_SIGMA_FLOOR
    return mu, sigma


class FrozenSignalNormalizer:
    def __init__(self, state: dict, device: torch.device):
        self.mu = state.get("mu")
        self.var = state.get("var")
        self.decay = float(state.get("decay", 0.99))
        self._initialized = bool(state.get("initialized", self.mu is not None and self.var is not None))
        if self.mu is not None:
            self.mu = self.mu.to(device)
        if self.var is not None:
            self.var = self.var.to(device)

    def normalize_frozen(self, signals: torch.Tensor) -> torch.Tensor:
        if not self._initialized or self.mu is None or self.var is None:
            return signals
        std = self.var.sqrt().clamp(min=1e-6)
        return ((signals - self.mu) / std).clamp(-3.0, 3.0)


@dataclass(frozen=True)
class PaperTradingDecision:
    position: float
    action: str
    logits: float
    edges: dict[str, float]


class PaperTradingModel:
    def __init__(
        self,
        *,
        predictors: Dict[str, HSAMAOnlineRuntime],
        trader: HSAMAOnlineRuntime,
        normalizer: FrozenSignalNormalizer,
        device: torch.device,
    ):
        self.predictors = predictors
        self.trader = trader
        self.normalizer = normalizer
        self.device = device
        self.step = 0
        self.prev_position = torch.zeros((1, 1), device=self.device)
        for runtime in [*predictors.values(), trader]:
            runtime.model.eval()
            if runtime.multi_scale_builder is not None:
                runtime.multi_scale_builder.eval()

    def _runtime_forward(self, runtime: HSAMAOnlineRuntime, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            context, _, _, _ = runtime._prepare_context_batch(x, step=self.step)
            policy = runtime.model.build_policy_from_context(context, surprisal=0.0, use_exploration=False)
            pred, _ = runtime.model.execute_policy(x, policy, raw_output=True)
        return pred

    def warmup(self, features_by_tf: dict[str, torch.Tensor]) -> None:
        """Feed historical data through the full pipeline (predictors + trader)
        so that all multi-scale GRU history buffers are populated before the
        first live decision."""
        lengths = [features_by_tf[tf].size(0) for tf in TIMEFRAMES]
        if not lengths:
            return
        warmup_len = min(lengths)
        for i in range(warmup_len):
            step_features = {tf: features_by_tf[tf][i : i + 1].to(self.device) for tf in TIMEFRAMES}
            # 1) Feed predictors — populates their history buffers
            edges = self.predict_edges(step_features)
            # 2) Feed trader — populates its history buffer
            pred_signals_raw = torch.cat([edges[tf] for tf in TIMEFRAMES], dim=1)
            pred_signals = self.normalizer.normalize_frozen(pred_signals_raw)
            bx_trade = torch.cat([step_features["15m"], pred_signals, self.prev_position], dim=1)
            pred_trade = self._runtime_forward(self.trader, bx_trade)
            self.prev_position = position_from_logits(pred_trade, deadzone=0.0).detach()
            self.step += 1

    def predict_edges(self, latest_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        edges: dict[str, torch.Tensor] = {}
        for tf in TIMEFRAMES:
            pred = self._runtime_forward(self.predictors[tf], latest_features[tf].to(self.device))
            mu, sigma = _predictor_mu_sigma(pred)
            edges[tf] = (mu / sigma).clamp(-PREDICTOR_EDGE_CLIP, PREDICTOR_EDGE_CLIP)
        return edges

    def decide(self, latest_features: dict[str, torch.Tensor]) -> PaperTradingDecision:
        edges = self.predict_edges(latest_features)
        pred_signals_raw = torch.cat([edges[tf] for tf in TIMEFRAMES], dim=1)
        pred_signals = self.normalizer.normalize_frozen(pred_signals_raw)
        bx_trade = torch.cat([latest_features["15m"].to(self.device), pred_signals, self.prev_position], dim=1)
        pred_trade = self._runtime_forward(self.trader, bx_trade)
        
        position = position_from_logits(
            pred_trade,
            deadzone=TRADER_POSITION_DEADZONE,
        )
        self.prev_position = position_from_logits(pred_trade, deadzone=0.0).detach()
        pos_value = float(position[-1, 0].detach().cpu().item())
        self.step += 1
        action = "LONG" if pos_value > 0.05 else ("SHORT" if pos_value < -0.05 else "FLAT")
        return PaperTradingDecision(
            position=pos_value,
            action=action,
            logits=float(pred_trade[-1, 0].detach().cpu().item()),
            edges={tf: float(edges[tf][-1, 0].detach().cpu().item()) for tf in TIMEFRAMES},
        )


def _build_predictor(in_features: int, tf: str, device: torch.device) -> HSAMAOnlineRuntime:
    model = HSAMA(
        in_features=in_features,
        out_features=2,
        num_nodes=8,
        state_dim=16,
        max_hops=2,
        context_dim=16,
        output_scale_init=(0.01, 1.0),
        learnable_output_scale=True,
    )
    runtime = HSAMAOnlineRuntime(
        model,
        config=HSAMARuntimeConfig(
            replay_config=None,
            multi_scale_t0_config=_make_predictor_config(tf),
            raw_output=True,
        ),
    )
    runtime.model.to(device)
    runtime.multi_scale_builder.to(device)
    return runtime


def _build_trader(in_features: int, device: torch.device) -> HSAMAOnlineRuntime:
    model = HSAMA(
        in_features=in_features,
        out_features=1,
        num_nodes=8,
        state_dim=16,
        max_hops=2,
        context_dim=16,
        output_scale_init=(1.0,),
        learnable_output_scale=True,
    )
    runtime = HSAMAOnlineRuntime(
        model,
        config=HSAMARuntimeConfig(
            replay_config=None,
            multi_scale_t0_config=_make_trade_config(),
            raw_output=True,
        ),
    )
    runtime.model.to(device)
    runtime.multi_scale_builder.to(device)
    return runtime


def load_paper_trading_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device | None = None,
) -> PaperTradingModel:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    predictors: Dict[str, HSAMAOnlineRuntime] = {}
    for tf in TIMEFRAMES:
        state = checkpoint["predictors"][tf]
        runtime = _build_predictor(int(state["in_features"]), tf, device)
        runtime.model.load_state_dict(state["model"])
        runtime.multi_scale_builder.load_state_dict(state["multi_scale_builder"])
        predictors[tf] = runtime

    trader_state = checkpoint["trader"]
    trader = _build_trader(int(trader_state["in_features"]), device)
    trader.model.load_state_dict(trader_state["model"])
    trader.multi_scale_builder.load_state_dict(trader_state["multi_scale_builder"])
    normalizer = FrozenSignalNormalizer(checkpoint.get("sig_normalizer", {}), device)
    return PaperTradingModel(predictors=predictors, trader=trader, normalizer=normalizer, device=device)
