"""
SolanaMultiTFEngine -- Motor de Backtest Multi-Timeframe para Solana (SOL)

Arquitetura de 5 T0s independentes:
  * T0_15m  -> preve log-retorno do proximo candle 15m
  * T0_1h   -> preve log-retorno do proximo candle 1h
  * T0_4h   -> preve log-retorno do proximo candle 4h
  * T0_1d   -> preve log-retorno do proximo candle 1d
  * T0_trade -> recebe as 4 previsoes dos T0s acima como features adicionais
               e decide posicao long/short via TriplexTradingLoss
"""

import sys
import os

# Guard para execucao direta: garante que a raiz do projeto esta no sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

from src.models.hsama import HSAMA
from src.runtime.online import HSAMAOnlineRuntime, HSAMARuntimeConfig
from src.runtime.replay import SurprisalBufferConfig, AdaptiveQuantileThreshold
from src.runtime.multiscale import MultiScaleT0Config, T0ScaleSpec

# Imports do pacote trade -- suportam execucao direta e como modulo
try:
    from .dataset import CryptoDataLoader
    from .loss import TriplexTradingLoss, position_from_logits
except ImportError:
    from trade.dataset import CryptoDataLoader  # type: ignore[no-redef]
    from trade.loss import TriplexTradingLoss, position_from_logits    # type: ignore[no-redef]


PREDICTOR_SIGMA_FLOOR = 1e-4
PREDICTOR_SIGMA_OFFSET = 8.0
PREDICTOR_EDGE_CLIP = 5.0
TRADER_POSITION_DEADZONE = 0.05
TRADING_COST_BPS = 0.0005


def _predictor_mu_sigma(preds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Converte saida do previsor em retorno esperado e volatilidade positiva."""
    mu = preds[:, 0:1]
    sigma = F.softplus(preds[:, 1:2] - PREDICTOR_SIGMA_OFFSET) + PREDICTOR_SIGMA_FLOOR
    return mu, sigma


def _predictor_probabilistic_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mu, sigma = _predictor_mu_sigma(preds)
    standardized_error = ((targets - mu) / sigma).pow(2)
    calibration_penalty = torch.log(sigma / PREDICTOR_SIGMA_FLOOR)
    return (0.5 * standardized_error + calibration_penalty).reshape(preds.size(0), -1).mean(dim=1)


def _build_trade_signal_features(predictions: list[torch.Tensor]) -> torch.Tensor:
    """Empacota [mu, sigma, mu/sigma] por timeframe para o trader."""
    features = []
    for pred in predictions:
        mu, sigma = _predictor_mu_sigma(pred)
        edge = (mu / sigma).clamp(-PREDICTOR_EDGE_CLIP, PREDICTOR_EDGE_CLIP)
        features.extend((mu, sigma, edge))
    return torch.cat(features, dim=1)


# ---------------------------------------------------------------------------
# Configuracoes de escala multi-temporal por agente
# ---------------------------------------------------------------------------

def _make_predictor_config(tf: str) -> MultiScaleT0Config:
    """
    Retorna o MultiScaleT0Config adequado para cada timeframe de previsao.
    Cada preditor tem 2 escalas internas (fast/slow) calibradas para o
    horizonte temporal do seu timeframe.
    """
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
    """Config do agente trader: micro + short para capturar momentum."""
    return MultiScaleT0Config(
        scales=[
            T0ScaleSpec("micro", stride=1, history_length=8,  hidden_dim=16),
            T0ScaleSpec("short", stride=4, history_length=12, hidden_dim=16),
        ]
    )


def _move_runtime_to_device(runtime: HSAMAOnlineRuntime, device: torch.device) -> None:
    """
    Move o modelo HSAMA e o MultiScaleT0Builder (scale_encoders + context_projection)
    para o device especificado. O HSAMAOnlineRuntime nao expoe .to() diretamente,
    entao movemos os sub-modulos nn.Module manualmente.
    """
    runtime.model.to(device)
    if runtime.multi_scale_builder is not None:
        runtime.multi_scale_builder.to(device)


# ---------------------------------------------------------------------------
# Normalizador EMA causal para sinais dos previsores
# ---------------------------------------------------------------------------

class EMASignalNormalizer:
    """
    Normaliza os sinais dos 4 previsores de forma causal — sem look-ahead.

    Cada batch e normalizado usando estatisticas (media e variancia EMA)
    acumuladas EXCLUSIVAMENTE dos batches anteriores.
    Apos cada batch, o estado EMA e atualizado com as estatisticas do batch atual.

    Para o primeiro batch nao ha historico disponivel — usa as proprias
    estatisticas do batch (unico caso inevitavel, impacto negligivel).

    Durante inferencia OOS, o estado e congelado no final do treino:
    usa normalize_frozen() para aplicar a mesma escala sem atualizar o EMA.
    """

    def __init__(self, num_signals: int, decay: float = 0.99):
        self.decay       = decay
        self.num_signals = num_signals
        self.mu          : Optional[torch.Tensor] = None  # [1, num_signals]
        self.var         : Optional[torch.Tensor] = None  # [1, num_signals]
        self._initialized = False

    def normalize(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Normaliza signals [B, num_signals] com estatisticas ANTERIORES ao batch,
        depois atualiza o EMA com as estatisticas deste batch.
        """
        batch_mu  = signals.mean(dim=0, keepdim=True).detach()
        batch_var = signals.var( dim=0, keepdim=True).clamp(min=1e-12).detach()

        if not self._initialized:
            # Primeiro batch: sem historico — inicializa e normaliza com o proprio batch
            self.mu  = batch_mu
            self.var = batch_var
            self._initialized = True
            std = batch_var.sqrt().clamp(min=1e-6)
            return ((signals - batch_mu) / std).clamp(-3.0, 3.0)

        # Normaliza com estatisticas ANTERIORES (causal, sem look-ahead)
        std  = self.var.sqrt().clamp(min=1e-6)
        norm = ((signals - self.mu) / std).clamp(-3.0, 3.0)

        # Atualiza EMA com o batch atual (para o proximo batch usar)
        self.mu  = self.decay * self.mu  + (1.0 - self.decay) * batch_mu
        self.var = self.decay * self.var + (1.0 - self.decay) * batch_var

        return norm

    def normalize_frozen(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Normaliza usando o estado final do treino, sem atualizar o EMA.
        Usado durante inferencia OOS para garantir consistencia de escala.
        """
        if not self._initialized:
            return signals  # fallback: sem historico, retorna bruto
        std = self.var.sqrt().clamp(min=1e-6)
        return ((signals - self.mu) / std).clamp(-3.0, 3.0)


# ---------------------------------------------------------------------------
# Motor principal
# ---------------------------------------------------------------------------

class SolanaMultiTFEngine:
    """
    Motor de backtest com 5 T0s independentes para Solana (SOL/USDT).

      4 T0s de previsao (15m, 1h, 4h, 1d) -- cada um preve o log-retorno
      do proximo candle no seu respectivo timeframe.

      1 T0 de trading -- recebe as 4 previsoes concatenadas ao vetor de
      features do 15m e decide posicao long/short.

    Suporte a CUDA: usa GPU automaticamente se disponivel.
    """

    TIMEFRAMES = ["15m", "1h", "4h", "1d"]

    def __init__(self, data_dir: str = "trade/data"):
        self.loader = CryptoDataLoader(data_dir=data_dir)
        self.batch_size   = 32
        self.max_samples  = 20_000   # Voltando para amostra pequena para iteracao rapida
        self.report_every = 100      # Mais feedback no terminal

        # Device auto-detect
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Engine] Dispositivo selecionado: {self.device}", flush=True)
        if self.device.type == "cuda":
            print(f"         GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ------------------------------------------------------------------
    # Inicializacao de agentes
    # ------------------------------------------------------------------

    def _init_predictor(self, tf: str, features_count: int) -> HSAMAOnlineRuntime:
        """Inicializa um agente previsor para um timeframe especifico."""
        model = HSAMA(
            in_features=features_count,
            out_features=2,       # [retorno_esperado, volatilidade_esperada_bruta]
            num_nodes=8,
            state_dim=16,
            max_hops=1,
            context_dim=16,
        )
        config = HSAMARuntimeConfig(
            observe_mode="batch",
            raw_output=True,
            replay_config=SurprisalBufferConfig(
                capacity=1200,
                replay_ratio=0.25,
                threshold=AdaptiveQuantileThreshold(quantile=0.75),
            ),
            multi_scale_t0_config=_make_predictor_config(tf),
        )
        runtime = HSAMAOnlineRuntime(model, config=config)
        runtime._per_sample_loss = _predictor_probabilistic_loss

        # Move para GPU
        _move_runtime_to_device(runtime, self.device)
        return runtime

    def _init_trader(self, features_count: int) -> HSAMAOnlineRuntime:
        """
        Inicializa o agente de trading.
        features_count = features_15m_dim + 4  (4 previsoes dos T0s de previsao)
        """
        model = HSAMA(
            in_features=features_count,
            out_features=2,       # [retorno_esperado, posicao_trade]
            num_nodes=8,
            state_dim=16,
            max_hops=1,
            context_dim=16,
            output_scale_init=(1.0, 4.0),
            learnable_output_scale=True,
        )
        config = HSAMARuntimeConfig(
            observe_mode="batch",
            raw_output=True,
            replay_config=SurprisalBufferConfig(
                capacity=1500,
                replay_ratio=0.3,
                threshold=AdaptiveQuantileThreshold(quantile=0.8),
            ),
            multi_scale_t0_config=_make_trade_config(),
        )
        runtime = HSAMAOnlineRuntime(model, config=config)
        trading_loss = TriplexTradingLoss(
            cost_bps=TRADING_COST_BPS,
            trade_weight=5.0,
            return_scale=100.0,
            entropy_weight=0.0,
            bias_weight=0.25,
            position_deadzone=TRADER_POSITION_DEADZONE,
            gamma=0.0,
        )
        runtime._per_sample_loss = trading_loss

        # Move para GPU
        _move_runtime_to_device(runtime, self.device)
        return runtime

    # ------------------------------------------------------------------
    # Backtest principal
    # ------------------------------------------------------------------

    def run_backtest(self):
        print(f"\n{'='*65}")
        print(" INICIANDO BACKTEST MULTI-TIMEFRAME -- SOLANA (SOL)")
        print(f" Arquitetura: 4 T0s de Previsao + 1 T0 de Trading")
        print(f"{'='*65}")

        # 1. Carrega e alinha todos os timeframes
        print("\n[Carregamento] Alinhando 4 timeframes da SOL ao grid 15m...")
        data = self.loader.load_multi_timeframe_sol(
            train_ratio=0.7,
            max_samples=self.max_samples,
        )

        X_train_15m, Y_train_15m, X_test_15m, Y_test_15m = data["15m"]
        features_15m = X_train_15m.shape[1]
        train_len = X_train_15m.shape[0]
        test_len  = X_test_15m.shape[0]

        # Move todos os tensores para o device de uma vez
        print(f"\n[Device] Movendo tensores para {self.device}...")
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

        print(f"\n Timeframes carregados:")
        for tf in self.TIMEFRAMES:
            X_tr, _, X_te, _ = data_dev[tf]
            print(f"   [{tf}] features={X_tr.shape[1]} | train={X_tr.shape[0]} | test={X_te.shape[0]}")

        # 2. Inicializa os 4 previsores
        print("\n[Init] Criando 4 T0s de previsao...")
        predictors: Dict[str, HSAMAOnlineRuntime] = {}
        for tf in self.TIMEFRAMES:
            X_tr, _, _, _ = data_dev[tf]
            predictors[tf] = self._init_predictor(tf, features_count=X_tr.shape[1])
            print(f"   T0_{tf} -> in_features={X_tr.shape[1]}")

        # 3. Inicializa o agente trader
        #    in_features = features_15m + 12 sinais dos previsores: mu, sigma e edge por timeframe
        predictor_signal_features = len(self.TIMEFRAMES) * 3
        trader_features = features_15m + predictor_signal_features
        print(f"\n[Init] Criando T0_trade (in_features={trader_features})...")
        trader = self._init_trader(features_count=trader_features)

        # ---------------------------------------------------------------
        # FASE 1-A -- Warm-up exclusivo dos Previsores (3 epocas)
        # O trader so entra depois que os previsores tiverem sinal proprio.
        # ---------------------------------------------------------------
        print(f"\n{'-'*65}")
        print(" [Fase 1-A] Warm-up dos 4 Previsores (sem trader)")
        print(f"{'-'*65}")

        import time
        PREDICTOR_WARMUP_EPOCHS = 5
        for ep in range(PREDICTOR_WARMUP_EPOCHS):
            ep_pnl_pred  = {tf: 0.0 for tf in self.TIMEFRAMES}
            total_batches = (train_len + self.batch_size - 1) // self.batch_size
            t_ep_start = time.time()

            for batch_idx, i in enumerate(range(0, train_len, self.batch_size)):
                end_i = min(i + self.batch_size, train_len)
                for tf in self.TIMEFRAMES:
                    bx = data_dev[tf][0][i:end_i]
                    by = data_dev[tf][1][i:end_i]
                    res = predictors[tf].observe(bx, by)
                    pred_mu, _ = _predictor_mu_sigma(res.live_prediction.detach())
                    pred_val = pred_mu.squeeze(-1)
                    ep_pnl_pred[tf] += (pred_val * by[:pred_val.shape[0], 0]).sum().item()

                if (batch_idx + 1) % self.report_every == 0 or (batch_idx + 1) == total_batches:
                    pct     = 100.0 * (batch_idx + 1) / total_batches
                    elapsed = time.time() - t_ep_start
                    eta_s   = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)
                    pred_str = " | ".join(f"{tf}:{ep_pnl_pred[tf]:+.3f}" for tf in self.TIMEFRAMES)
                    print(
                        f"  WarmUp {ep+1}/{PREDICTOR_WARMUP_EPOCHS}"
                        f" [{batch_idx+1:>{len(str(total_batches))}}/{total_batches}]"
                        f" {pct:5.1f}%  ETA {eta_s:4.0f}s  [{pred_str}]",
                        flush=True,
                    )

            print(f"\n  WarmUp {ep+1}/{PREDICTOR_WARMUP_EPOCHS} concluido em {time.time()-t_ep_start:.1f}s")
            for tf in self.TIMEFRAMES:
                print(f"   T0_{tf} PnL Bruto: {ep_pnl_pred[tf]:+.4f}")

        # ---------------------------------------------------------------
        # FASE 1-B -- Treino do Trader com sinais normalizados (2 epocas)
        # Os sinais dos previsores sao z-score normalizados intra-batch
        # antes de entrar no trader, evitando que magnitudes minusculas
        # (~0.005) sejam ignoradas pelo bias da rede.
        # ---------------------------------------------------------------
        print(f"\n{'-'*65}")
        print(" [Fase 1-B] Treino do Trader (sinais normalizados)")
        print(f"{'-'*65}")

        # Normalizer EMA causal: cada batch usa estatisticas dos batches ANTERIORES
        # Elimina o data leak do z-score intra-batch que usava amostras futuras do batch
        sig_normalizer = EMASignalNormalizer(num_signals=predictor_signal_features, decay=0.99)

        TRADER_EPOCHS = 5
        for ep in range(TRADER_EPOCHS):
            # Reinicia o normalizer a cada epoca para estadisticas capazes de adaptar
            # (o estado do EMA persiste entre epocas — nao resetamos aqui intencionalmente:
            # a 2a epoca usa o EMA aquecido da 1a, que e mais estavel)
            ep_pnl_trade = 0.0
            total_batches = (train_len + self.batch_size - 1) // self.batch_size
            t_ep_start = time.time()

            for batch_idx, i in enumerate(range(0, train_len, self.batch_size)):
                end_i = min(i + self.batch_size, train_len)

                # Continua treinando previsores em paralelo e captura a previsao exata
                # que acabou de ser feita (sem chamar predict() de novo para nao
                # corromper o historico temporal com duplicatas)
                signals = []
                for tf in self.TIMEFRAMES:
                    bx = data_dev[tf][0][i:end_i]
                    by = data_dev[tf][1][i:end_i]
                    res = predictors[tf].observe(bx, by)
                    signals.append(res.live_prediction.detach())
                
                # Tensor [B, 12] com mu, sigma e edge ajustado por risco dos 4 T0s
                predictor_signals = _build_trade_signal_features(signals)
                
                # Normalizacao EMA causal: usa estatisticas dos batches anteriores
                predictor_signals_norm = sig_normalizer.normalize(predictor_signals)

                bx_15m   = X_train_15m[i:end_i]
                by_15m   = Y_train_15m[i:end_i]
                bx_trade = torch.cat([bx_15m, predictor_signals_norm], dim=1)

                loss_obj = trader._per_sample_loss
                if hasattr(loss_obj, "set_previous_position"):
                    prev_pos = getattr(self, "_last_train_position", 0.0)
                    loss_obj.set_previous_position(prev_pos)

                res_trade = trader.observe(bx_trade, by_15m)

                logits = res_trade.live_prediction[:, 1:2].detach()
                pos_raw = torch.tanh(logits)
                
                posicoes  = position_from_logits(
                    logits,
                    deadzone=TRADER_POSITION_DEADZONE,
                ).detach()
                self._last_train_position = float(posicoes[-1].item())
                ep_pnl_trade += (posicoes * by_15m[:posicoes.shape[0]]).sum().item()

                if (batch_idx + 1) % self.report_every == 0 or (batch_idx + 1) == total_batches:
                    pct     = 100.0 * (batch_idx + 1) / total_batches
                    elapsed = time.time() - t_ep_start
                    eta_s   = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)
                    # Posicao media do batch para diagnostico
                    pos_mean = float(posicoes.mean().item())
                    
                    logit_mean = logits.mean().item()
                    logit_std = logits.std().item()
                    raw_abs = pos_raw.abs().mean().item()
                    pos_abs = posicoes.abs().mean().item()

                    print(
                        f"  Trader Ep {ep+1}/{TRADER_EPOCHS}"
                        f" [{batch_idx+1:>{len(str(total_batches))}}/{total_batches}]"
                        f" {pct:5.1f}%  ETA {eta_s:4.0f}s"
                        f"  PnL={ep_pnl_trade:+.4f}  pos_media={pos_mean:+.3f}\n"
                        f"    logit_mean={logit_mean:+.4f} logit_std={logit_std:.4f} "
                        f"raw_abs={raw_abs:.4f} pos_abs={pos_abs:.4f}",
                        flush=True,
                    )

            print(f"\n  Trader Ep {ep+1}/{TRADER_EPOCHS} concluido em {time.time()-t_ep_start:.1f}s")
            print(f"   T0_trade PnL Bruto: {ep_pnl_trade:+.4f}")


        # ---------------------------------------------------------------
        # FASE 2 -- Hold-out cego (OOS Inference)
        # ---------------------------------------------------------------
        print(f"\n{'-'*65}")
        print(" [Fase 2] Execucao Hold-out Cega (Teste Isolado OOS)")
        print(f"{'-'*65}")

        # Coloca todos os modelos em modo eval
        for tf in self.TIMEFRAMES:
            predictors[tf].model.eval()
        trader.model.eval()

        print(" Executando predicoes OOS em uma unica passagem (para preservar cronologia da GRU)...")
        with torch.no_grad():
            # 1. Inferencia dos 4 previsores para todos os 6000 passos de uma vez.
            # O predict() vai processar sequencialmente garantindo integridade da GRU.
            signals = []
            for tf in self.TIMEFRAMES:
                bx_all = data_dev[tf][2]  # [test_len, features_dim]
                pred = predictors[tf].predict(bx_all, raw_output=True)  # [test_len, 2]
                signals.append(pred)
            
            pred_signals_oos = _build_trade_signal_features(signals)  # [test_len, 12]
            
            # Normalizacao com estado EMA congelado
            pred_signals_norm_oos = sig_normalizer.normalize_frozen(pred_signals_oos)

            # 2. Inferencia do Trader (Step-by-step causal para calcular Surprisal e extrair Logs do Grafo)
            bx_trade_oos = torch.cat([X_test_15m, pred_signals_norm_oos], dim=1)
            by_trade_oos = Y_test_15m  # Usaremos para calcular a loss do step t-1 e alimentar o surprisal
            
            pred_trade_list = []
            dna_list = []
            temp_list = []

            # Clonamos o history buffer do trader para nao sujar o estado de treino
            temp_history = None
            if trader.multi_scale_builder is not None and trader.history_buffer is not None:
                temp_history = trader.history_buffer.clone()

            for i in range(test_len):
                bx_single = bx_trade_oos[i:i+1]
                
                # Prepara o contexto multi-escala
                if temp_history is None or trader.multi_scale_builder is None:
                    context = trader.model.encode_context(bx_single)
                else:
                    temp_history.append(bx_single.squeeze(0))
                    contexts, _ = trader.multi_scale_builder.prepare_contexts(
                        temp_history, step=0, device=bx_single.device, dtype=bx_single.dtype,
                    )
                    context, _ = trader.multi_scale_builder.compose_context(
                        contexts, step=0, device=bx_single.device, dtype=bx_single.dtype,
                    )

                # Surprisal Causal: Pânico baseado na loss do passo (i-1)
                if i == 0:
                    current_surprisal = 0.0
                else:
                    prev_pred = pred_trade_list[-1]
                    prev_y = by_trade_oos[i-1:i]
                    # Calcula a perda (TriplexTradingLoss) que o agente obteve na barra passada
                    per_sample_loss = trader._per_sample_loss(prev_pred, prev_y)
                    # Atualiza o estimador EMA e pega o Z-Score do Surprisal
                    obs = trader.global_surprisal_estimator.observe(per_sample_loss)
                    current_surprisal = float(obs.priority.mean().item())

                # Constroi a politica usando o surprisal calculado
                policy = trader.model.build_policy_from_context(
                    context, surprisal=current_surprisal, use_exploration=False
                )
                # Executa o grafo com o DNA dinamico
                pred, _ = trader.model.execute_policy(bx_single, policy, raw_output=True)
                
                pred_trade_list.append(pred)
                dna_list.append(policy.edge_dnas.detach().cpu())
                temp_list.append(policy.temperature.detach().cpu())

            pred_trade_oos = torch.cat(pred_trade_list, dim=0)  # [test_len, 2]
            dnas_oos = torch.cat(dna_list, dim=0)  # [test_len, num_edges, dna_dim]
            temps_oos = torch.cat(temp_list, dim=0)  # [test_len, 1, 1]

        pred_return_oos = pred_trade_oos[:, 0:1].cpu()  # [test_len, 1]
        posicoes_teste  = position_from_logits(
            pred_trade_oos[:, 1:2],
            deadzone=TRADER_POSITION_DEADZONE,
        ).cpu()  # [test_len, 1]
        pred_signals_oos = pred_signals_oos.cpu()
        Y_test_15m_cpu  = Y_test_15m.cpu()

        # ---------------------------------------------------------------
        # Metricas e Relatorio
        # ---------------------------------------------------------------
        pos_np    = posicoes_teste.view(-1).numpy()
        ret_np    = Y_test_15m_cpu.view(-1).numpy()
        gross_np  = pos_np * ret_np

        pos_shift_np      = np.roll(pos_np, 1)
        pos_shift_np[0]   = 0.0
        cost_np           = np.abs(pos_np - pos_shift_np) * TRADING_COST_BPS
        net_return_stream = gross_np - cost_np

        agent_equity = net_return_stream.cumsum()
        bh_equity    = ret_np.cumsum()

        # Metricas adicionais
        
        # Discretiza a posicao para calcular numero REAL de trades
        discrete_pos = np.zeros_like(pos_np)
        discrete_pos[pos_np > 0.05] = 1.0
        discrete_pos[pos_np < -0.05] = -1.0
        
        # Um trade real ocorre apenas quando a classe da acao muda (LONG/SHORT/FLAT)
        pos_change = discrete_pos[1:] != discrete_pos[:-1]
        n_trades = int(pos_change.sum())
        avg_holding = test_len / max(n_trades, 1)
        win_rate      = float((net_return_stream > 0).mean()) * 100
        avg_win       = float(net_return_stream[net_return_stream > 0].mean()) if (net_return_stream > 0).any() else 0.0
        avg_loss      = float(net_return_stream[net_return_stream < 0].mean()) if (net_return_stream < 0).any() else 0.0
        correct_dir   = float(((pos_np > 0) == (ret_np > 0)).mean()) * 100

        candles_per_year  = 252 * 96
        sharpe = 0.0
        if net_return_stream.std() > 0:
            sharpe = (net_return_stream.mean() / net_return_stream.std()) * np.sqrt(candles_per_year)

        # --- NOVAS METRICAS DE DEBUG DA ARQUITETURA ---
        
        # 1. Metricas dos Previsores (Acuracia Direcional, MSE e incerteza)
        pred_sig_np = pred_signals_oos.numpy()
        tf_metrics = {}
        for idx, tf in enumerate(self.TIMEFRAMES):
            y_real = data_dev[tf][3].cpu().numpy().squeeze()  # [test_len]
            base_col = idx * 3
            y_pred = pred_sig_np[:, base_col]
            y_sigma = pred_sig_np[:, base_col + 1]
            y_edge = pred_sig_np[:, base_col + 2]
            dir_acc = float(((y_pred > 0) == (y_real > 0)).mean()) * 100
            mse = float(np.mean((y_pred - y_real)**2))
            
            # Correlacao com a decisao final do trader
            corr_pos = float(np.corrcoef(y_edge, pos_np)[0, 1]) if np.std(y_edge) > 0 else 0.0
            tf_metrics[tf] = {
                "acc": dir_acc,
                "mse": mse,
                "corr": corr_pos,
                "sigma": float(np.mean(y_sigma)),
            }

        # 2. Critic Check (Retorno Esperado do Trader vs Posicao)
        pred_ret_np = pred_return_oos.numpy().squeeze()
        critic_corr = float(np.corrcoef(pred_ret_np, pos_np)[0, 1]) if np.std(pred_ret_np) > 0 else 0.0

        # 2.5 Metricas do Grafo (Plasticidade e Regime)
        dnas_oos_np = dnas_oos.numpy()
        temps_oos_np = temps_oos.numpy().squeeze()
        
        dna_diffs = np.linalg.norm(dnas_oos_np[1:] - dnas_oos_np[:-1], axis=(1, 2))
        avg_dna_volatility = float(np.mean(dna_diffs))
        dna_sparsity = float(np.mean(np.abs(dnas_oos_np) < 0.01)) * 100
        
        avg_temp = float(np.mean(temps_oos_np))
        max_temp = float(np.max(temps_oos_np))

        # 3. Comportamento e Saturacao
        long_pct = float((discrete_pos == 1.0).mean()) * 100
        short_pct = float((discrete_pos == -1.0).mean()) * 100
        flat_pct = float((discrete_pos == 0.0).mean()) * 100
        mean_conf = float(np.abs(pos_np).mean())
        
        # 4. Atrito e Drawdown
        total_gross = float(gross_np.sum())
        total_costs = float(cost_np.sum())
        friction_pct = (total_costs / total_gross * 100) if total_gross > 0 else float('inf')
        
        cum_max = np.maximum.accumulate(agent_equity)
        max_dd = float(np.min(agent_equity - cum_max)) * 100  # aproximação em % para log-returns

        # Print do relatorio turbinado
        print(f"\n>>> METRICAS ESTRUTURAIS & DE ARQUITETURA <<<")
        acc_str = " | ".join([f"{tf}: {tf_metrics[tf]['acc']:.1f}%" for tf in self.TIMEFRAMES])
        print(f"[Sinais OOS] Acuracia Direcional: {acc_str}")
        corr_str = " | ".join([f"{tf}: {tf_metrics[tf]['corr']:+.2f}" for tf in self.TIMEFRAMES])
        sigma_str = " | ".join([f"{tf}: {tf_metrics[tf]['sigma']:.5f}" for tf in self.TIMEFRAMES])
        print(f"[Volatilidade Prevista] Media: {sigma_str}")
        print(f"[Atencao do Trader] Correlacao Edge(mu/sigma) c/ Posicao: {corr_str}")
        print(f"[Critic Check] Correlacao (Ret Previsto x Posicao): {critic_corr:+.2f}")
        
        print(f"\n>>> LOGS DO GRAFO & DETECCAO DE REGIME <<<")
        print(f"Temperatura do Surprisal: Media = {avg_temp:.3f} | Max = {max_temp:.3f} (Pico de Panico)")
        print(f"Plasticidade do DNA (Delta L2/passo): {avg_dna_volatility:.4f}")
        print(f"Esparsidade Dinamica (Edge Pruning):  {dna_sparsity:.1f}% das arestas ~inativas")
        
        print(f"\n>>> COMPORTAMENTO DO AGENTE (Loss Diagnostics) <<<")
        print(f"Distribuicao de Tempo:  LONG: {long_pct:.1f}% | SHORT: {short_pct:.1f}% | FLAT: {flat_pct:.1f}%")
        print(f"Confianca Media (|tanh|): {mean_conf:.3f} (esperado 0.3 - 0.7)")
        friction_str = f"{friction_pct:.1f}% do Gross PnL" if friction_pct != float('inf') else "S/A (Gross PnL Negativo)"
        print(f"Atrito de Transacao:    Custos comeram {friction_str}")

        print(f"\n>>> DADOS FINANCEIROS OUT-OF-SAMPLE ({test_len} passos - 15m) <<<")
        print(f"Baseline B&H Retorno Bruto Acumulado:    {bh_equity[-1]:+.4f}")
        print(f"HSAMA Agente Retorno Liquido Acumulado:  {agent_equity[-1]:+.4f}")
        print(f"Sharpe Ratio Anualizado (15m):           {sharpe:+.3f}")
        print(f"Max Drawdown:                            {max_dd:.1f}%")
        print(f"Numero de Rebalanceamentos (Trades):     {n_trades}")
        print(f"Holding medio (candles 15m):             {avg_holding:.1f}")
        print(f"Win Rate (por passo):                    {win_rate:.1f}%")
        print(f"Avg Win | Avg Loss:                      {avg_win:+.6f} | {avg_loss:+.6f}")

        # ---------------------------------------------------------------
        # CSV de Trades
        # ---------------------------------------------------------------
        os.makedirs("artifacts", exist_ok=True)

        # Classificacao da acao por threshold
        def _classify_action(p: float) -> str:
            if   p >  0.05: return "LONG"
            elif p < -0.05: return "SHORT"
            else:           return "FLAT"

        pred_sig_np = pred_signals_oos.numpy()  # [test_len, 12] -> mu, sigma, edge por timeframe

        records = []
        cumul = 0.0
        for step in range(test_len):
            pos_val    = float(pos_np[step])
            ret_val    = float(ret_np[step])
            gross_val  = float(gross_np[step])
            cost_val   = float(cost_np[step])
            net_val    = float(net_return_stream[step])
            cumul     += net_val
            pos_change = float(cost_np[step] / TRADING_COST_BPS) if cost_np[step] > 0 else 0.0  # |delta_pos|

            records.append({
                "step":              step,
                "position":          round(pos_val, 6),
                "action":            _classify_action(pos_val),
                "position_change":   round(abs(pos_np[step] - pos_shift_np[step]), 6),
                "pred_return":       round(float(pred_return_oos[step, 0]), 6),
                "pred_15m":          round(float(pred_sig_np[step, 0]), 6),
                "vol_15m":           round(float(pred_sig_np[step, 1]), 6),
                "edge_15m":          round(float(pred_sig_np[step, 2]), 6),
                "pred_1h":           round(float(pred_sig_np[step, 3]), 6),
                "vol_1h":            round(float(pred_sig_np[step, 4]), 6),
                "edge_1h":           round(float(pred_sig_np[step, 5]), 6),
                "pred_4h":           round(float(pred_sig_np[step, 6]), 6),
                "vol_4h":            round(float(pred_sig_np[step, 7]), 6),
                "edge_4h":           round(float(pred_sig_np[step, 8]), 6),
                "pred_1d":           round(float(pred_sig_np[step, 9]), 6),
                "vol_1d":            round(float(pred_sig_np[step, 10]), 6),
                "edge_1d":           round(float(pred_sig_np[step, 11]), 6),
                "market_return":     round(ret_val,   6),
                "gross_pnl":         round(gross_val, 6),
                "cost":              round(cost_val,  6),
                "net_pnl":           round(net_val,   6),
                "cumul_net_pnl":     round(cumul,     6),
                "correct_direction": int((pos_val > 0) == (ret_val > 0)),
            })

        df_trades = pd.DataFrame(records)
        csv_path  = "artifacts/SOL_trades_oos.csv"
        df_trades.to_csv(csv_path, index=False)
        print(f"\nCSV de trades salvo em: {csv_path}  ({len(df_trades)} linhas)")

        # ---------------------------------------------------------------
        # Grafico
        # ---------------------------------------------------------------
        os.makedirs("artifacts/plots", exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

        # Painel 1: Equity curves
        axes[0].plot(bh_equity,    label="Buy & Hold (Baseline)", color="#aaaaaa", alpha=0.8, linewidth=1.2)
        axes[0].plot(agent_equity, label="HSAMA 5-T0 Trader",     color="#00c4ff", linewidth=2)
        axes[0].set_title("SOL/USDT -- Equity Curve OOS | 4 T0s Previsao + 1 T0 Trade")
        axes[0].set_ylabel("Retorno Cumulativo (Log)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.2)

        # Painel 2: Posicao contínua + coloracao de acoes
        axes[1].plot(pos_np, color="#ff7a00", linewidth=0.7, alpha=0.85)
        axes[1].axhline( 0.05, color="green", linewidth=0.5, linestyle="--", alpha=0.5)
        axes[1].axhline(-0.05, color="red",   linewidth=0.5, linestyle="--", alpha=0.5)
        axes[1].axhline( 0,    color="white", linewidth=0.4, linestyle=":",  alpha=0.3)
        axes[1].set_title("Posicao do Agente (dashed: thresholds LONG/SHORT +/-0.02)")
        axes[1].set_ylabel("Posicao [-1, 1]")
        axes[1].grid(True, alpha=0.2)

        # Painel 3: Sinais dos 4 previsores
        colors_tf = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373"]
        for idx, tf in enumerate(self.TIMEFRAMES):
            axes[2].plot(pred_sig_np[:, idx * 3 + 2], label=f"T0_{tf}",
                         color=colors_tf[idx], linewidth=0.7, alpha=0.75)
        axes[2].axhline(0, color="white", linewidth=0.4, linestyle=":", alpha=0.3)
        axes[2].set_title("Sinais dos 4 T0s de Previsao (edge = retorno / volatilidade)")
        axes[2].set_ylabel("Edge Ajustado")
        axes[2].set_xlabel("Passos de 15m (OOS)")
        axes[2].legend(loc="upper right", fontsize=8)
        axes[2].grid(True, alpha=0.2)

        plt.tight_layout()
        plot_path = "artifacts/plots/SOL_multitf_equity.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Grafico salvo em: {plot_path}")

        return {
            "agent_equity": agent_equity,
            "bh_equity":    bh_equity,
            "net_returns":  net_return_stream,
            "trades_df":    df_trades,
        }

    def run_walk_forward(self, sample_windows: Optional[list[int]] = None):
        """
        Executa uma validacao walk-forward inicial por janelas de tamanho crescente.

        Observacao: cada janela usa o sufixo cronologico mais recente de tamanho N
        e reaproveita o pipeline completo de run_backtest(). Isso cria uma primeira
        barreira contra overfit de um unico hold-out sem reescrever o loader para
        janelas deslizantes arbitrarias.
        """
        sample_windows = sample_windows or [8_000, 12_000, 20_000]
        original_max_samples = self.max_samples
        summaries = []
        try:
            for window in sample_windows:
                self.max_samples = int(window)
                print(f"\n[WalkForward] Janela max_samples={self.max_samples}")
                result = self.run_backtest()
                net_returns = result["net_returns"]
                trades_df = result["trades_df"]
                final_net = float(net_returns.sum())
                summaries.append(
                    {
                        "max_samples": self.max_samples,
                        "final_net": final_net,
                        "trades": int((trades_df["position_change"] > 0).sum()),
                        "flat_pct": float((trades_df["action"] == "FLAT").mean()),
                    }
                )
        finally:
            self.max_samples = original_max_samples
        return summaries


if __name__ == "__main__":
    engine = SolanaMultiTFEngine()
    engine.run_backtest()
