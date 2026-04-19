import sys
import os
import torch
import matplotlib.pyplot as plt

from src.models.hsama import HSAMA
from src.runtime.online import HSAMAOnlineRuntime, HSAMARuntimeConfig
from src.runtime.replay import SurprisalBufferConfig, AdaptiveQuantileThreshold
from src.runtime.multiscale import MultiScaleT0Config, T0ScaleSpec

from .dataset import CryptoDataLoader
from .loss import TriplexTradingLoss

class TradeBacktestEngine:
    def __init__(self, data_dir: str = "trade/data"):
        self.loader = CryptoDataLoader(data_dir=data_dir)
        self.batch_size = 32
        
    def init_agent(self, features_count: int) -> HSAMAOnlineRuntime:
        model = HSAMA(
            in_features=features_count,
            out_features=2, 
            num_nodes=8,
            state_dim=16,
            max_hops=1,
            context_dim=16
        )
        
        config = HSAMARuntimeConfig(
            observe_mode="batch",
            raw_output=True,
            replay_config=SurprisalBufferConfig(
                capacity=1500,
                replay_ratio=0.3,
                threshold=AdaptiveQuantileThreshold(quantile=0.8) # Maior rigor para a memoria ser focada
            ),
            multi_scale_t0_config=MultiScaleT0Config(
                scales=[
                    T0ScaleSpec("fast", stride=1, history_length=16, hidden_dim=16),
                    T0ScaleSpec("slow", stride=4, history_length=24, hidden_dim=16),
                ]
            )
        )
        
        runtime = HSAMAOnlineRuntime(model, config=config)
        runtime._per_sample_loss = TriplexTradingLoss(cost_bps=0.0005, trade_weight=2.0)
        return runtime

    def run_backtest(self, asset_filename: str):
        print(f"\n{'='*60}")
        print(f" INICIANDO ONLINE TRADE STREAMING: {asset_filename}")
        print(f"{'='*60}")

        # 1. Carrega via Sliding Generator sem Vazamentos
        X_train, Y_train, X_test, Y_test = self.loader.load_asset(asset_filename, train_ratio=0.7)
        features_dim = X_train.shape[1]
        
        # 2. Inicia os Meta-Bots
        agent = self.init_agent(features_count=features_dim)
        
        # 3. Stream de Validação / Treino Inicial (Vida Passada)
        print("\n[Fase 1] Treinamento Online Contínuo (Simulando Vida Passada...)")
        train_len = X_train.shape[0]
        
        # Faz apenas 2 "Épocas" online, para simular a janela histórica vista pelo bot
        for ep in range(2):
            ep_pnl = 0.0
            # Chunk processing via generator batching avoids exploding RAM globally
            for i in range(0, train_len, self.batch_size):
                bx = X_train[i:min(i + self.batch_size, train_len)]
                br = Y_train[i:min(i + self.batch_size, train_len)] 
                
                # O processamento de gradiente acorre por step! Mágica do Active Meta Learning.
                res = agent.observe(bx, br)
                
                # Accounting Off-Graph
                posicoes = torch.tanh(res.live_prediction[:, 1:2]).detach()
                ep_pnl += (posicoes * br).sum().item()

            print(f" Época de WarmUp {ep+1}/2 | PnL Bruto Gerado: {ep_pnl:+.4f}")
            
        print("\n[Fase 2] Execução Hold-out cega (Teste Isolado com Inference Realtime)")
        test_len = X_test.shape[0]
        agent.model.eval()
        
        # O Predit usa processamento global com o clone de `temp_history` q adicionamos.
        # Caso o teste seja MUITO longo, faremos um loop iterativo pra prevenir RAM blowout tmb:
        all_test_positions = []
        with torch.no_grad():
            for i in range(0, test_len, 256):
                bx = X_test[i:min(i + 256, test_len)]
                preds = agent.predict(bx, raw_output=True)
                pos = torch.tanh(preds[:, 1:2])
                all_test_positions.append(pos)
                
        posicoes_teste = torch.cat(all_test_positions, dim=0)
        
        # ---- Métricas Finais e Relatório ----
        gross_pnl = posicoes_teste * Y_test
        pos_shift = torch.roll(posicoes_teste, shifts=1, dims=0)
        pos_shift[0] = posicoes_teste[0]
        
        custos = torch.abs(posicoes_teste - pos_shift) * 0.0005
        net_return_stream = (gross_pnl - custos).view(-1).numpy()
        
        bh_stream = Y_test.view(-1).numpy() # Buy and hold puro base
        
        agent_equity = net_return_stream.cumsum(axis=0)
        bh_equity = bh_stream.cumsum(axis=0)
        
        print(f"\n>>> DADOS DO OUT-OF-SAMPLE ({test_len} passos - 15m) <<<")
        print(f"Baseline B&H Retorno Bruto Acumulado: {bh_equity[-1]:+.4f}")
        print(f"HSAMA Agente Retorno Líquido Acumulado: {agent_equity[-1]:+.4f}")
        
        # Plot Graph Data 
        os.makedirs("artifacts/plots", exist_ok=True)
        plt.figure(figsize=(12,6))
        plt.plot(bh_equity, label="Buy & Hold (Baseline)", color="gray", alpha=0.7)
        plt.plot(agent_equity, label="HSAMA Actor (Net PnL)", color="blue")
        plt.title(f"Acurácia Financeira Real OOS - {asset_filename}")
        plt.xlabel("15m Time Steps")
        plt.ylabel("Retornos Cumulativos (Log)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_name = asset_filename.split('.')[0]
        plot_path = f"artifacts/plots/{plot_name}_equity.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Gráfico gerado em {plot_path}")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, default="BTCUSDT_15m_1825d.parquet")
    args = parser.parse_args()
    
    engine = TradeBacktestEngine()
    engine.run_backtest(args.asset)
