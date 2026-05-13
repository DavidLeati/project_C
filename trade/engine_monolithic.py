"""
MonolithicEngine -- Motor de Backtest End-to-End Monolítico com Memória

Arquitetura: 4 HSAMA previsores + 1 HSAMA trader, todos treinados
com um único otimizador AdamW via backprop end-to-end.

Incorpora a Solução Completa:
1. Memória Multi-Escala (GRUs) do HSAMAOnlineRuntime acoplada
   manualmente aos modelos para reter o grafo end-to-end.
2. Loss Auxiliar na Fase 1-B para evitar o esquecimento (catastrophic forgetting)
   dos sinais de predição sob a pressão exclusiva do PnL.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple

from src.models.hsama import HSAMA
from src.runtime.surprisal import EMASurprisalEstimator
from src.runtime.online import HSAMAOnlineRuntime, HSAMARuntimeConfig
from src.runtime.multiscale import MultiScaleT0Config, T0ScaleSpec

try:
    from .dataset import CryptoDataLoader
    from .loss import TriplexTradingLoss, position_from_logits, predictor_directional_loss
except ImportError:
    from trade.dataset import CryptoDataLoader  # type: ignore[no-redef]
    from trade.loss import TriplexTradingLoss, position_from_logits, predictor_directional_loss  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Hiperparâmetros globais
# ---------------------------------------------------------------------------
PREDICTOR_SIGMA_FLOOR  = 1e-4
PREDICTOR_SIGMA_OFFSET = 8.0
PREDICTOR_EDGE_CLIP    = 2.0
TRADER_POSITION_DEADZONE = 0.15
TRADING_COST_BPS       = 0.0005

PREDICTOR_WARMUP_EPOCHS = 5    # warm-up leve: sai do ruído puro antes do trader entrar
JOINT_EPOCHS            = 15   # fase principal: gradiente end-to-end remodela os previsores


# ---------------------------------------------------------------------------
# Helpers de Configuração Multi-Escala (Mesmos da engine modular)
# ---------------------------------------------------------------------------

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
            T0ScaleSpec("fast", stride=1, history_length=7,  hidden_dim=16),
            T0ScaleSpec("slow", stride=6, history_length=14, hidden_dim=16),
        ],
        "1d": [
            T0ScaleSpec("fast", stride=1, history_length=7,  hidden_dim=16),
            T0ScaleSpec("slow", stride=4, history_length=14, hidden_dim=16),
        ],
    }
    return MultiScaleT0Config(scales=scale_map[tf])

def _make_trade_config() -> MultiScaleT0Config:
    return MultiScaleT0Config(
        scales=[
            T0ScaleSpec("micro", stride=1, history_length=8,  hidden_dim=16),
            T0ScaleSpec("short", stride=4, history_length=12, hidden_dim=16),
        ]
    )

# ---------------------------------------------------------------------------
# Helpers de Loss e Sinais
# ---------------------------------------------------------------------------

def _predictor_mu_sigma(preds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu    = preds[:, 0:1]
    sigma = F.softplus(preds[:, 1:2] - PREDICTOR_SIGMA_OFFSET) + PREDICTOR_SIGMA_FLOOR
    return mu, sigma

def _predictor_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return predictor_directional_loss(
        preds, targets,
        sigma_offset=PREDICTOR_SIGMA_OFFSET,
        sigma_floor=PREDICTOR_SIGMA_FLOOR,
        direction_weight=0.5,
    )

# ---------------------------------------------------------------------------
# Normalizador EMA causal dos sinais dos previsores
# ---------------------------------------------------------------------------

class EMASignalNormalizer:
    def __init__(self, num_signals: int, decay: float = 0.99):
        self.decay        = decay
        self.num_signals  = num_signals
        self.mu:  Optional[torch.Tensor] = None
        self.var: Optional[torch.Tensor] = None
        self._initialized = False

    def normalize(self, signals: torch.Tensor) -> torch.Tensor:
        batch_mu  = signals.mean(dim=0, keepdim=True).detach()
        batch_var = signals.var( dim=0, keepdim=True).clamp(min=1e-12).detach()

        if not self._initialized:
            self.mu  = batch_mu
            self.var = batch_var
            self._initialized = True
            std = batch_var.sqrt().clamp(min=1e-6)
            return ((signals - batch_mu) / std).clamp(-3.0, 3.0)

        std  = self.var.sqrt().clamp(min=1e-6)
        norm = ((signals - self.mu) / std).clamp(-3.0, 3.0)

        self.mu  = self.decay * self.mu  + (1.0 - self.decay) * batch_mu
        self.var = self.decay * self.var + (1.0 - self.decay) * batch_var
        return norm

    def normalize_frozen(self, signals: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
            return signals
        std = self.var.sqrt().clamp(min=1e-6)
        return ((signals - self.mu) / std).clamp(-3.0, 3.0)


# ---------------------------------------------------------------------------
# Motor principal
# ---------------------------------------------------------------------------

class MonolithicEngine:
    TIMEFRAMES = ["15m", "1h", "4h", "1d"]

    def __init__(self, data_dir: str = "trade/data"):
        self.loader       = CryptoDataLoader(data_dir=data_dir)
        self.batch_size   = 32      # gradiente por amostra mais informativo
        self.max_samples  = 60_000  # 42k train / 18k test (~187 dias OOS)
        self.report_every = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Monolithic Engine] Dispositivo: {self.device}", flush=True)

    def run_backtest(self):
        print(f"\n{'='*65}")
        print(" INICIANDO BACKTEST MONOLÍTICO END-TO-END (C/ MEMÓRIA MULTI-ESCALA)")
        print(f"{'='*65}")

        # 1. Carrega dados
        data = self.loader.load_multi_timeframe_sol(
            train_ratio=0.7,
            max_samples=self.max_samples,
        )

        data_dev: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for tf in self.TIMEFRAMES:
            Xtr, Ytr, Xte, Yte = data[tf]
            data_dev[tf] = (
                Xtr.to(self.device),
                Ytr.to(self.device),
                Xte.to(self.device),
                Yte.to(self.device),
            )

        X_train_15m = data_dev["15m"][0]
        Y_train_15m = data_dev["15m"][1]
        X_test_15m  = data_dev["15m"][2]
        Y_test_15m  = data_dev["15m"][3]

        train_len = X_train_15m.shape[0]
        test_len  = X_test_15m.shape[0]

        print(f"\n Timeframes:")
        for tf in self.TIMEFRAMES:
            Xtr, _, Xte, _ = data_dev[tf]
            print(f"   [{tf}] features={Xtr.shape[1]} | train={Xtr.shape[0]} | test={Xte.shape[0]}")

        # 2. Instancia modelos (Envolvidos no HSAMAOnlineRuntime para ter os GRUs e Memória)
        # Bypassamos o método .observe() para usar apenas o _prepare_context_batch()
        # IMPORTANTE: o HSAMAOnlineRuntime requer um HSAMA pré-construído + HSAMARuntimeConfig
        predictors: Dict[str, HSAMAOnlineRuntime] = {}
        for tf in self.TIMEFRAMES:
            pred_model = HSAMA(
                in_features=data_dev[tf][0].shape[1],
                out_features=2,
                num_nodes=8,
                state_dim=16,
                max_hops=2,
                context_dim=16,
                output_scale_init=(0.01, 1.0),
                learnable_output_scale=True,
            )
            pred_config = HSAMARuntimeConfig(
                replay_config=None,  # Sem replay independente, gradiente será global
                multi_scale_t0_config=_make_predictor_config(tf),
                raw_output=True,
            )
            predictors[tf] = HSAMAOnlineRuntime(pred_model, config=pred_config)
            predictors[tf].model.to(self.device)
            predictors[tf].multi_scale_builder.to(self.device)

        predictor_signal_features = len(self.TIMEFRAMES)
        trader_features = X_train_15m.shape[1] + predictor_signal_features + 1
        trader_model = HSAMA(
            in_features=trader_features,
            out_features=2,
            num_nodes=8,
            state_dim=16,
            max_hops=2,
            context_dim=16,
            output_scale_init=(1.0, 1.0),
            learnable_output_scale=True,
        )
        trader_config = HSAMARuntimeConfig(
            replay_config=None,
            multi_scale_t0_config=_make_trade_config(),
            raw_output=True,
        )
        trader = HSAMAOnlineRuntime(trader_model, config=trader_config)
        trader.model.to(self.device)
        trader.multi_scale_builder.to(self.device)

        surprisal_estimator = EMASurprisalEstimator()

        trading_loss_fn = TriplexTradingLoss(
            cost_bps=TRADING_COST_BPS,
            trade_weight=5.0,
            return_scale=100.0,
            bias_weight=0.5,
            stagnation_threshold=0.7,
            stagnation_weight=0.5,
            sharpe_weight=0.5,
            position_deadzone=TRADER_POSITION_DEADZONE,
            gamma=0.0,
        )

        sig_normalizer = EMASignalNormalizer(num_signals=predictor_signal_features, decay=0.99)

        # ---------------------------------------------------------------
        # FASE 1-A: Warm-up exclusivo dos previsores
        # ---------------------------------------------------------------
        print(f"\n{'-'*65}")
        print(f" [Fase 1-A] Warm-up dos 4 Previsores ({PREDICTOR_WARMUP_EPOCHS} épocas)")
        print(f"{'-'*65}")

        pred_params: list[torch.nn.Parameter] = []
        for p in predictors.values():
            pred_params.extend(p.model.parameters())
            pred_params.extend(p.multi_scale_builder.parameters())
        pred_optimizer = torch.optim.AdamW(pred_params, lr=1e-3)

        for ep in range(PREDICTOR_WARMUP_EPOCHS):
            # O history_buffer NÃO é resetado entre épocas:
            # os GRU encoders precisam de histórico acumulado para produzir
            # contexto útil. Limpar o buffer forçaria ~93 batches de contexto
            # lixo no início de cada época, zerando o aprendizado.
            for p in predictors.values():
                p.model.train()
                p.multi_scale_builder.train()

            total_batches = (train_len + self.batch_size - 1) // self.batch_size
            ep_pnl = {tf: 0.0 for tf in self.TIMEFRAMES}
            ep_dir_ok = {tf: 0 for tf in self.TIMEFRAMES}
            ep_dir_tot = {tf: 0 for tf in self.TIMEFRAMES}
            t_start = time.time()

            for batch_idx, i in enumerate(range(0, train_len, self.batch_size)):
                end_i = min(i + self.batch_size, train_len)
                B = end_i - i
                global_step = ep * train_len + i

                surprisal = surprisal_estimator.current(
                    batch_size=B, device=self.device, dtype=X_train_15m.dtype
                )

                pred_optimizer.zero_grad()
                total_pred_loss = torch.tensor(0.0, device=self.device)
                commits = []

                for tf in self.TIMEFRAMES:
                    rt = predictors[tf]
                    bx = data_dev[tf][0][i:end_i]
                    by = data_dev[tf][1][i:end_i]
                    
                    # Extraindo forward pass com memória
                    context, latest, due, _ = rt._prepare_context_batch(bx, step=global_step)
                    policy = rt.model.build_policy_from_context(context, surprisal=surprisal)
                    pred, _ = rt.model.execute_policy(bx, policy, raw_output=True)
                    
                    loss = _predictor_loss(pred, by)
                    total_pred_loss = total_pred_loss + loss.mean()
                    commits.append((rt, latest, due))

                    with torch.no_grad():
                        mu, _ = _predictor_mu_sigma(pred)
                        mu_v = mu.squeeze(-1)
                        tgt_v = by[:mu_v.shape[0], 0]
                        ep_pnl[tf] += (mu_v * tgt_v).sum().item()
                        ep_dir_ok[tf]  += int(((mu_v > 0) == (tgt_v > 0)).sum().item())
                        ep_dir_tot[tf] += int(tgt_v.shape[0])

                total_pred_loss.backward()
                torch.nn.utils.clip_grad_norm_(pred_params, 1.0)
                pred_optimizer.step()

                # Atualiza buffers de histórico e métricas de scale
                for rt, latest, due in commits:
                    rt.multi_scale_builder.commit_contexts(latest, due_scale_names=due, step=global_step + B)

                surprisal_estimator.observe(total_pred_loss.detach())

                if (batch_idx + 1) % self.report_every == 0 or (batch_idx + 1) == total_batches:
                    pct   = 100.0 * (batch_idx + 1) / total_batches
                    eta_s = (time.time() - t_start) / (batch_idx + 1) * (total_batches - batch_idx - 1)
                    pnl_str = " | ".join(f"{tf}:{ep_pnl[tf]:+.3f}" for tf in self.TIMEFRAMES)
                    print(
                        f"  WarmUp {ep+1}/{PREDICTOR_WARMUP_EPOCHS}"
                        f" [{batch_idx+1:>{len(str(total_batches))}}/{total_batches}]"
                        f" {pct:5.1f}%  ETA {eta_s:4.0f}s  [{pnl_str}]",
                        flush=True,
                    )

            print(f"\n  WarmUp {ep+1}/{PREDICTOR_WARMUP_EPOCHS} concluido em {time.time()-t_start:.1f}s")
            for tf in self.TIMEFRAMES:
                acc = 100.0 * ep_dir_ok[tf] / max(1, ep_dir_tot[tf])
                print(f"   T0_{tf} PnL Bruto: {ep_pnl[tf]:+.4f}  DirAcc: {acc:.1f}%")

        # ---------------------------------------------------------------
        # FASE 1-B: Treino end-to-end com Trader e Loss Auxiliar
        # ---------------------------------------------------------------
        print(f"\n{'-'*65}")
        print(f" [Fase 1-B] Treino End-to-End c/ Loss Auxiliar ({JOINT_EPOCHS} épocas)")
        print(f"{'-'*65}")

        # Otimizadores SEPARADOS para trader e previsores:
        # Evita que o gradient clipping da aux_loss (magnitude ~2-4) suprima
        # os gradientes do trader (magnitude ~0.02) quando clipados juntos.
        trader_params = list(trader.model.parameters()) + list(trader.multi_scale_builder.parameters())
        pred_params_joint: list[torch.nn.Parameter] = []
        for p in predictors.values():
            pred_params_joint.extend(p.model.parameters())
            pred_params_joint.extend(p.multi_scale_builder.parameters())
        trader_optimizer = torch.optim.AdamW(trader_params, lr=1e-3)
        pred_optimizer_joint = torch.optim.AdamW(pred_params_joint, lr=1e-3)

        last_position = 0.0  # posição final do batch anterior para custo de borda

        for ep in range(JOINT_EPOCHS):
            trader.model.train()
            trader.multi_scale_builder.train()
            for p in predictors.values():
                p.model.train()
                p.multi_scale_builder.train()

            total_batches = (train_len + self.batch_size - 1) // self.batch_size
            ep_pnl_trade = 0.0
            t_start = time.time()

            for batch_idx, i in enumerate(range(0, train_len, self.batch_size)):
                end_i = min(i + self.batch_size, train_len)
                bx_15m = X_train_15m[i:end_i]
                by_15m = Y_train_15m[i:end_i]
                B = bx_15m.size(0)
                global_step = (PREDICTOR_WARMUP_EPOCHS + ep) * train_len + i

                surprisal = surprisal_estimator.current(
                    batch_size=B, device=self.device, dtype=bx_15m.dtype
                )

                edges = []
                commits = []
                total_aux_loss = torch.tensor(0.0, device=self.device)

                # sem detach: gradiente flui end-to-end pelos previsores
                for tf in self.TIMEFRAMES:
                    rt = predictors[tf]
                    bx_tf = data_dev[tf][0][i:end_i]
                    by_tf = data_dev[tf][1][i:end_i]
                    
                    context, latest, due, _ = rt._prepare_context_batch(bx_tf, step=global_step)
                    policy = rt.model.build_policy_from_context(context, surprisal=surprisal)
                    pred, _ = rt.model.execute_policy(bx_tf, policy, raw_output=True)
                    
                    # 1. Extração do Edge (Sinal)
                    mu, sigma = _predictor_mu_sigma(pred)
                    edge = (mu / sigma).clamp(-PREDICTOR_EDGE_CLIP, PREDICTOR_EDGE_CLIP)
                    edges.append(edge)
                    commits.append((rt, latest, due))
                    
                    # 2. Loss Auxiliar (Evita Catastrophic Forgetting)
                    loss_p = _predictor_loss(pred, by_tf)
                    total_aux_loss = total_aux_loss + loss_p.mean()

                pred_signals_raw = torch.cat(edges, dim=1)
                pred_signals = sig_normalizer.normalize(pred_signals_raw)

                # -------------------------------------------------------------
                # Pseudo-Autoregressive Position Feedback (2-Pass Trick)
                # -------------------------------------------------------------
                with torch.no_grad():
                    # Pass 1: Estimate trajectory without valid past position
                    dummy_prev = torch.full((B, 1), last_position, device=self.device)
                    bx_trade_dummy = torch.cat([bx_15m, pred_signals, dummy_prev], dim=1)
                    trader_ctx_d, _, _, _ = trader._prepare_context_batch(bx_trade_dummy, step=global_step)
                    pol_d = trader.model.build_policy_from_context(trader_ctx_d, surprisal=surprisal)
                    pred_d, _ = trader.model.execute_policy(bx_trade_dummy, pol_d, raw_output=True)
                    pos_estim = position_from_logits(pred_d, deadzone=0.0)
                    
                    prev_positions = torch.roll(pos_estim, shifts=1, dims=0)
                    prev_positions[0] = last_position

                bx_trade = torch.cat([bx_15m, pred_signals, prev_positions], dim=1)

                trading_loss_fn.set_previous_position(last_position)

                # Forward do Trader com Memória
                trader_context, trader_latest, trader_due, _ = trader._prepare_context_batch(bx_trade, step=global_step)
                trader_policy = trader.model.build_policy_from_context(trader_context, surprisal=surprisal)
                pred_trade, _ = trader.model.execute_policy(bx_trade, trader_policy, raw_output=True)

                loss_pnl = trading_loss_fn(pred_trade, by_15m)
                
                # A loss final combina o PnL do Trader + Loss Auxiliar dos Previsores.
                # Peso 0.1 (era 0.5): aux_loss serve apenas como guardrail de catastrophic
                # forgetting, não deve competir com o sinal de PnL do trader.
                total_loss = loss_pnl.mean() + 0.1 * total_aux_loss

                trader_optimizer.zero_grad()
                pred_optimizer_joint.zero_grad()
                total_loss.backward()
                # Clip SEPARADO: impede que aux_loss domine o sinal do trader
                torch.nn.utils.clip_grad_norm_(trader_params, 1.0)
                torch.nn.utils.clip_grad_norm_(pred_params_joint, 1.0)
                trader_optimizer.step()
                pred_optimizer_joint.step()

                # Commits de histórico GRU
                for rt, latest, due in commits:
                    rt.multi_scale_builder.commit_contexts(latest, due_scale_names=due, step=global_step + B)
                trader.multi_scale_builder.commit_contexts(trader_latest, due_scale_names=trader_due, step=global_step + B)

                surprisal_estimator.observe(loss_pnl.detach())

                with torch.no_grad():
                    logits   = pred_trade.detach()
                    posicoes = position_from_logits(logits, deadzone=TRADER_POSITION_DEADZONE)
                    
                    # For logging stats, just use the directional part
                    pos_raw = torch.tanh(logits[:, 0:1])
                    
                    last_position = float(posicoes[-1].item())
                    ep_pnl_trade += (posicoes * by_15m[:posicoes.shape[0]]).sum().item()

                    n_batch  = posicoes.shape[0]
                    n_long   = int((posicoes >  TRADER_POSITION_DEADZONE).sum().item())
                    n_short  = int((posicoes < -TRADER_POSITION_DEADZONE).sum().item())
                    n_flat   = n_batch - n_long - n_short
                    pos_mean = float(posicoes.mean().item())
                    logit_mean = float(logits[:, 0].mean().item())
                    logit_std  = float(logits[:, 0].std().item())
                    raw_abs    = float(pos_raw.abs().mean().item())
                    pos_abs    = float(posicoes.abs().mean().item())

                if (batch_idx + 1) % self.report_every == 0 or (batch_idx + 1) == total_batches:
                    pct   = 100.0 * (batch_idx + 1) / total_batches
                    eta_s = (time.time() - t_start) / (batch_idx + 1) * (total_batches - batch_idx - 1)
                    print(
                        f"  Ep {ep+1}/{JOINT_EPOCHS}"
                        f" [{batch_idx+1:>{len(str(total_batches))}}/{total_batches}]"
                        f" {pct:5.1f}%  ETA {eta_s:4.0f}s"
                        f"  PnL={ep_pnl_trade:+.4f}  pos_media={pos_mean:+.3f}"
                        f"  L/S/F={n_long}/{n_short}/{n_flat}\n"
                        f"    logit_mean={logit_mean:+.4f} logit_std={logit_std:.4f}"
                        f"  raw_abs={raw_abs:.4f} pos_abs={pos_abs:.4f}",
                        flush=True,
                    )

            print(f"\n  Ep {ep+1}/{JOINT_EPOCHS} concluido em {time.time()-t_start:.1f}s")
            print(f"   T0_trade PnL Bruto: {ep_pnl_trade:+.4f}")

        # Salva um checkpoint temporário logo após o treino, para evitar perda caso o OOS quebre
        os.makedirs("models", exist_ok=True)
        tmp_ckpt_path = f"models/monolithic_temp_pre_oos.pt"
        torch.save({
            "trader": trader.model.state_dict(),
            "predictors": {tf: predictors[tf].model.state_dict() for tf in self.TIMEFRAMES}
        }, tmp_ckpt_path)
        print(f"\n[Backup de Segurança] Pesos salvos provisoriamente em {tmp_ckpt_path} antes do OOS!")

        # ---------------------------------------------------------------
        # FASE 2: Inferência OOS em batch
        # ---------------------------------------------------------------
        print(f"\n{'-'*65}")
        print(" [Fase 2] Inferência OOS")
        print(f"{'-'*65}")

        trader.model.eval()
        trader.multi_scale_builder.eval()
        for p in predictors.values():
            p.model.eval()
            p.multi_scale_builder.eval()

        with torch.no_grad():
            test_start_step = (PREDICTOR_WARMUP_EPOCHS + JOINT_EPOCHS) * train_len
            B_oos = X_test_15m.size(0)
            pred_trade_oos_list = []
            curr_pos = torch.full((1, 1), last_position, device=self.device)
            
            print("   -> Simulando OOS step-by-step (Autoregressivo)...")
            for i in range(B_oos):
                if i % max(1, B_oos // 10) == 0:
                    print(f"      OOS Progresso: {100.0 * i / B_oos:.1f}%")
                
                # Predictors
                edges_i = []
                for tf in self.TIMEFRAMES:
                    rt = predictors[tf]
                    bx_i = data_dev[tf][2][i:i+1]
                    ctx_i, latest_i, due_i, _ = rt._prepare_context_batch(bx_i, step=test_start_step + i)
                    pol_i = rt.model.build_policy_from_context(ctx_i, surprisal=0.0)
                    pred_i, _ = rt.model.execute_policy(bx_i, pol_i, raw_output=True)
                    rt.multi_scale_builder.commit_contexts(latest_i, due_scale_names=due_i, step=test_start_step + i + 1)
                    
                    mu, sigma = _predictor_mu_sigma(pred_i)
                    edge = (mu / sigma).clamp(-PREDICTOR_EDGE_CLIP, PREDICTOR_EDGE_CLIP)
                    edges_i.append(edge)
                
                sig_raw_i = torch.cat(edges_i, dim=1)
                sig_i = sig_normalizer.normalize_frozen(sig_raw_i)
                
                # Trader
                bx_trade_i = torch.cat([X_test_15m[i:i+1], sig_i, curr_pos], dim=1)
                t_ctx_i, t_latest_i, t_due_i, _ = trader._prepare_context_batch(bx_trade_i, step=test_start_step + i)
                t_pol_i = trader.model.build_policy_from_context(t_ctx_i, surprisal=0.0)
                t_pred_i, _ = trader.model.execute_policy(bx_trade_i, t_pol_i, raw_output=True)
                trader.multi_scale_builder.commit_contexts(t_latest_i, due_scale_names=t_due_i, step=test_start_step + i + 1)
                
                pred_trade_oos_list.append(t_pred_i)
                curr_pos = position_from_logits(t_pred_i, deadzone=0.0)
            
            print("      OOS Progresso: 100.0%")
            pred_trade_oos = torch.cat(pred_trade_oos_list, dim=0)

        posicoes_teste = position_from_logits(
            pred_trade_oos, deadzone=TRADER_POSITION_DEADZONE
        ).cpu()

        # ---------------------------------------------------------------
        # Métricas OOS
        # ---------------------------------------------------------------
        pos_np   = posicoes_teste.view(-1).numpy()
        ret_np   = Y_test_15m.cpu().view(-1).numpy()
        gross_np = pos_np * ret_np

        pos_shift_np    = np.roll(pos_np, 1); pos_shift_np[0] = 0.0
        cost_np         = np.abs(pos_np - pos_shift_np) * TRADING_COST_BPS
        net_stream      = gross_np - cost_np
        agent_equity    = net_stream.cumsum()
        bh_equity       = ret_np.cumsum()

        discrete_pos = np.zeros_like(pos_np)
        discrete_pos[pos_np >  0.05] =  1.0
        discrete_pos[pos_np < -0.05] = -1.0
        pos_change   = discrete_pos[1:] != discrete_pos[:-1]
        n_trades     = int(pos_change.sum())
        avg_holding  = test_len / max(n_trades, 1)

        long_pct  = float((discrete_pos ==  1.0).mean()) * 100
        short_pct = float((discrete_pos == -1.0).mean()) * 100
        flat_pct  = float((discrete_pos ==  0.0).mean()) * 100

        candles_per_year = 252 * 96
        sharpe = 0.0
        if net_stream.std() > 0:
            sharpe = (net_stream.mean() / net_stream.std()) * np.sqrt(candles_per_year)

        cum_max = np.maximum.accumulate(agent_equity)
        max_dd  = float(np.min(agent_equity - cum_max)) * 100

        total_gross  = float(gross_np.sum())
        total_costs  = float(cost_np.sum())
        friction_pct = (total_costs / total_gross * 100) if total_gross > 0 else float("inf")

        print(f"\n[Resultado OOS Monolítico]")
        print(f"B&H Retorno Bruto:               {bh_equity[-1]:+.4f}")
        print(f"Agente Retorno Líquido:          {agent_equity[-1]:+.4f}")
        print(f"Sharpe Anualizado (15m):         {sharpe:+.3f}")
        print(f"Max Drawdown:                    {max_dd:.1f}%")
        print(f"Número de Trades:                {n_trades}")
        print(f"Holding Médio (candles):         {avg_holding:.1f}")
        print(f"Distribuição: LONG {long_pct:.1f}% | SHORT {short_pct:.1f}% | FLAT {flat_pct:.1f}%")
        if friction_pct != float("inf"):
            print(f"Atrito de Transação:             {friction_pct:.1f}% do Gross PnL")

        # CSV de trades
        os.makedirs("artifacts", exist_ok=True)
        records = []
        cumul = 0.0
        for step in range(test_len):
            net_val = float(net_stream[step])
            cumul  += net_val
            records.append({
                "step":          step,
                "position":      round(float(pos_np[step]), 6),
                "action":        "LONG" if pos_np[step] > 0.05 else ("SHORT" if pos_np[step] < -0.05 else "FLAT"),
                "market_return": round(float(ret_np[step]), 6),
                "gross_pnl":     round(float(gross_np[step]), 6),
                "cost":          round(float(cost_np[step]), 6),
                "net_pnl":       round(net_val, 6),
                "cumul_net_pnl": round(cumul, 6),
            })
        df = pd.DataFrame(records)
        csv_path = "artifacts/SOL_monolithic_trades_oos.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nCSV salvo em: {csv_path}  ({len(df)} linhas)")

        # Gráfico
        os.makedirs("artifacts/plots", exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

        axes[0].plot(bh_equity,    label="Buy & Hold",        color="#aaaaaa", alpha=0.8)
        axes[0].plot(agent_equity, label="Monolithic Trader",  color="#00c4ff", linewidth=2)
        axes[0].set_title("SOL/USDT -- Equity Curve OOS | Monolítico End-to-End (Memória GRU)")
        axes[0].set_ylabel("Retorno Cumulativo (Log)")
        axes[0].legend(); axes[0].grid(True, alpha=0.2)

        axes[1].plot(pos_np, color="#ff7a00", linewidth=0.7, alpha=0.85)
        axes[1].axhline( 0.05, color="green", linewidth=0.5, linestyle="--", alpha=0.5)
        axes[1].axhline(-0.05, color="red",   linewidth=0.5, linestyle="--", alpha=0.5)
        axes[1].set_title("Posição do Agente")
        axes[1].set_ylabel("Posição [-1, 1]")
        axes[1].set_xlabel("Passos 15m (OOS)")
        axes[1].grid(True, alpha=0.2)

        plt.tight_layout()
        plot_path = "artifacts/plots/SOL_monolithic_equity.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Gráfico salvo em: {plot_path}")

        # Salva checkpoint completo
        os.makedirs("models", exist_ok=True)
        ckpt_ts   = time.strftime("%Y%m%d_%H%M%S")
        ckpt_path = f"models/monolithic_{ckpt_ts}.pt"
        checkpoint = {
            # Metadados da run
            "timestamp":        ckpt_ts,
            "max_samples":      self.max_samples,
            "train_len":        int(train_len),
            "test_len":         int(test_len),
            # Hiperparâmetros relevantes
            "hyperparams": {
                "PREDICTOR_WARMUP_EPOCHS": PREDICTOR_WARMUP_EPOCHS,
                "JOINT_EPOCHS":            JOINT_EPOCHS,
                "TRADING_COST_BPS":        TRADING_COST_BPS,
                "TRADER_POSITION_DEADZONE": TRADER_POSITION_DEADZONE,
            },
            # Pesos dos 4 previsores
            "predictors": {
                tf: {
                    "model":              predictors[tf].model.state_dict(),
                    "multi_scale_builder": predictors[tf].multi_scale_builder.state_dict(),
                    "runtime":            predictors[tf].state_dict(),
                    "in_features":        data_dev[tf][0].shape[1],
                }
                for tf in self.TIMEFRAMES
            },
            # Pesos do trader
            "trader": {
                "model":               trader.model.state_dict(),
                "multi_scale_builder": trader.multi_scale_builder.state_dict(),
                "runtime":             trader.state_dict(),
                "in_features":         int(trader_features),
            },
            # Estado do normalizador EMA (necessário para inferência)
            "sig_normalizer": {
                "mu":           sig_normalizer.mu,
                "var":          sig_normalizer.var,
                "decay":        sig_normalizer.decay,
                "initialized":  sig_normalizer._initialized,
            },
            # Métricas OOS para referência
            "oos_metrics": {
                "agent_net_return": float(agent_equity[-1]),
                "bh_return":        float(bh_equity[-1]),
                "sharpe":           float(sharpe),
                "max_dd_pct":       float(max_dd),
                "n_trades":         int(n_trades),
                "avg_holding":      float(avg_holding),
                "long_pct":         float(long_pct),
                "short_pct":        float(short_pct),
                "flat_pct":         float(flat_pct),
                "friction_pct":     float(friction_pct) if friction_pct != float("inf") else None,
            },
        }
        torch.save(checkpoint, ckpt_path)
        print(f"\nCheckpoint salvo em: {ckpt_path}")

        return {
            "agent_equity": agent_equity,
            "bh_equity":    bh_equity,
            "net_returns":  net_stream,
            "trades_df":    df,
            "checkpoint":   ckpt_path,
        }


if __name__ == "__main__":
    engine = MonolithicEngine()
    engine.run_backtest()
