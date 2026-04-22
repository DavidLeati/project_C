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
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

from src.models.hsama import HSAMA
from src.runtime.online import HSAMAOnlineRuntime, HSAMARuntimeConfig
from src.runtime.replay import SurprisalBufferConfig, AdaptiveQuantileThreshold
from src.runtime.multiscale import MultiScaleT0Config, T0ScaleSpec

# Imports do pacote trade -- suportam execucao direta e como modulo
try:
    from .dataset import CryptoDataLoader
    from .loss import TriplexTradingLoss
except ImportError:
    from trade.dataset import CryptoDataLoader  # type: ignore[no-redef]
    from trade.loss import TriplexTradingLoss    # type: ignore[no-redef]


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
        self.batch_size = 32

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
            out_features=1,       # preve 1 valor: log-retorno do proximo candle
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
        # Previsores usam MSE puro (sem custo de transacao)
        runtime._per_sample_loss = lambda preds, targets: F.mse_loss(
            preds, targets, reduction="none"
        ).reshape(preds.size(0), -1).mean(dim=1)

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
        runtime._per_sample_loss = TriplexTradingLoss(cost_bps=0.0005, trade_weight=2.0)

        # Move para GPU
        _move_runtime_to_device(runtime, self.device)
        return runtime

    # ------------------------------------------------------------------
    # Pipeline de inferencia dos previsores
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_predictor_signals(
        self,
        predictors: Dict[str, HSAMAOnlineRuntime],
        x_slices: Dict[str, torch.Tensor],
        batch_indices: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Para um batch [start:end], executa a inferencia dos 4 previsores
        e retorna um tensor de shape [B, 4] com as previsoes de cada timeframe.
        Os tensores de entrada ja estao no device correto.
        """
        start, end = batch_indices
        signals = []
        for tf in self.TIMEFRAMES:
            agent = predictors[tf]
            bx = x_slices[tf][start:end]  # ja no device
            pred = agent.predict(bx, raw_output=True)  # [B, 1]
            signals.append(pred)
        return torch.cat(signals, dim=1)  # [B, 4]

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
        data = self.loader.load_multi_timeframe_sol(train_ratio=0.7)

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
        #    in_features = features_15m + 4 sinais dos previsores
        trader_features = features_15m + len(self.TIMEFRAMES)
        print(f"\n[Init] Criando T0_trade (in_features={trader_features})...")
        trader = self._init_trader(features_count=trader_features)

        # ---------------------------------------------------------------
        # FASE 1 -- Treino Online (Warm-up historico com 2 epocas)
        # ---------------------------------------------------------------
        print(f"\n{'-'*65}")
        print(" [Fase 1] Treino Online Continuo (Simulando Vida Passada...)")
        print(f"{'-'*65}")

        for ep in range(2):
            ep_pnl_pred = {tf: 0.0 for tf in self.TIMEFRAMES}
            ep_pnl_trade = 0.0

            for i in range(0, train_len, self.batch_size):
                end_i = min(i + self.batch_size, train_len)

                # --- Treina os 4 previsores no seu proprio target ---
                for tf in self.TIMEFRAMES:
                    bx = data_dev[tf][0][i:end_i]  # X_train no device
                    by = data_dev[tf][1][i:end_i]  # Y_train no device
                    res = predictors[tf].observe(bx, by)
                    pred_val = res.live_prediction.detach().squeeze(-1)
                    ep_pnl_pred[tf] += (pred_val * by[:pred_val.shape[0], 0]).sum().item()

                # --- Constroi features do trader ---
                predictor_signals = self._get_predictor_signals(
                    predictors,
                    x_slices={tf: data_dev[tf][0] for tf in self.TIMEFRAMES},
                    batch_indices=(i, end_i),
                )
                # Concatena features 15m + 4 sinais dos previsores (tudo no device)
                bx_15m  = X_train_15m[i:end_i]
                by_15m  = Y_train_15m[i:end_i]
                bx_trade = torch.cat([bx_15m, predictor_signals], dim=1)

                # --- Treina o agente trader ---
                res_trade = trader.observe(bx_trade, by_15m)
                posicoes = torch.tanh(res_trade.live_prediction[:, 1:2]).detach()
                ep_pnl_trade += (posicoes * by_15m[:posicoes.shape[0]]).sum().item()

            print(f"\n Epoca {ep+1}/2:")
            for tf in self.TIMEFRAMES:
                print(f"   T0_{tf} PnL Bruto: {ep_pnl_pred[tf]:+.4f}")
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

        all_positions: list[torch.Tensor] = []

        with torch.no_grad():
            for i in range(0, test_len, 256):
                end_i = min(i + 256, test_len)

                # Inferencia dos 4 previsores (dados ja no device)
                signals = []
                for tf in self.TIMEFRAMES:
                    bx = data_dev[tf][2][i:end_i]
                    pred = predictors[tf].predict(bx, raw_output=True)  # [B, 1]
                    signals.append(pred)
                predictor_signals = torch.cat(signals, dim=1)  # [B, 4]

                # Inferencia do trader
                bx_15m   = X_test_15m[i:end_i]
                bx_trade = torch.cat([bx_15m, predictor_signals], dim=1)
                trade_pred = trader.predict(bx_trade, raw_output=True)
                pos = torch.tanh(trade_pred[:, 1:2])  # posicao in (-1, 1)
                all_positions.append(pos)

        # Consolida e move para CPU para metricas/plot
        posicoes_teste = torch.cat(all_positions, dim=0).cpu()  # [test_len, 1]
        Y_test_15m_cpu = Y_test_15m.cpu()

        # ---------------------------------------------------------------
        # Metricas e Relatorio
        # ---------------------------------------------------------------
        gross_pnl = posicoes_teste * Y_test_15m_cpu
        pos_shift = torch.roll(posicoes_teste, shifts=1, dims=0)
        pos_shift[0] = posicoes_teste[0]
        custos = torch.abs(posicoes_teste - pos_shift) * 0.0005
        net_return_stream = (gross_pnl - custos).view(-1).numpy()
        bh_stream = Y_test_15m_cpu.view(-1).numpy()

        agent_equity = net_return_stream.cumsum(axis=0)
        bh_equity    = bh_stream.cumsum(axis=0)

        print(f"\n>>> DADOS DO OUT-OF-SAMPLE ({test_len} passos - 15m) <<<")
        print(f"Baseline B&H Retorno Bruto Acumulado:    {bh_equity[-1]:+.4f}")
        print(f"HSAMA Agente Retorno Liquido Acumulado:  {agent_equity[-1]:+.4f}")

        # Sharpe ratio anualizado (252 dias * 96 candles 15m/dia)
        candles_per_year = 252 * 96
        if net_return_stream.std() > 0:
            sharpe = (net_return_stream.mean() / net_return_stream.std()) * np.sqrt(candles_per_year)
            print(f"Sharpe Ratio Anualizado (15m):           {sharpe:+.3f}")

        # Grafico
        os.makedirs("artifacts/plots", exist_ok=True)
        plt.figure(figsize=(14, 7))

        plt.subplot(2, 1, 1)
        plt.plot(bh_equity,    label="Buy & Hold (Baseline)", color="gray",    alpha=0.7, linewidth=1.5)
        plt.plot(agent_equity, label="HSAMA 5-T0 Trader",     color="#00c4ff", linewidth=2)
        plt.title("SOL/USDT -- Equity Curve OOS | 4 T0s Previsao + 1 T0 Trade")
        plt.ylabel("Retorno Cumulativo (Log)")
        plt.legend()
        plt.grid(True, alpha=0.25)

        plt.subplot(2, 1, 2)
        plt.plot(posicoes_teste.numpy(), color="#ff7a00", linewidth=0.8, alpha=0.8)
        plt.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
        plt.title("Posicao do Agente Trader ao longo do OOS")
        plt.ylabel("Posicao (-1=Short, +1=Long)")
        plt.xlabel("Passos de 15m")
        plt.grid(True, alpha=0.25)

        plt.tight_layout()
        plot_path = "artifacts/plots/SOL_multitf_equity.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nGrafico salvo em: {plot_path}")

        return {
            "agent_equity": agent_equity,
            "bh_equity":    bh_equity,
            "net_returns":  net_return_stream,
        }


if __name__ == "__main__":
    engine = SolanaMultiTFEngine()
    engine.run_backtest()
