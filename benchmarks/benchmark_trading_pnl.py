import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.models.hsama import HSAMA
from src.runtime.online import HSAMAOnlineRuntime, HSAMARuntimeConfig
from src.runtime.replay import SurprisalBufferConfig
from src.runtime.multiscale import MultiScaleT0Config, T0ScaleSpec
from src.runtime.replay import AdaptiveQuantileThreshold

def generate_market_data(samples=2000, features=8):
    """Gera uma série temporal sintética de retornos com ruído e tendência de onda."""
    torch.manual_seed(42)
    time_steps = torch.linspace(0, 4 * math.pi, samples)
    
    # Onda limpa (Sinal oculto)
    hidden_signal = torch.sin(time_steps) + 0.5 * torch.sin(2 * time_steps)
    
    # Adicionamos ruído aos retornos simulando o mercado
    returns = hidden_signal.unsqueeze(1) * 0.05 + torch.randn(samples, 1) * 0.02
    
    # Features artificiais (momentum, volatilidade em janela, etc. - aqui simpex)
    X = torch.randn(samples, features) * 0.1
    # vazamos o sinal do mercado defasado em T-1 para as features
    X[1:, 0] = returns[:-1, 0] 
    
    return X, returns

def run_trading_benchmark():
    print("-" * 60)
    print("BENCHMARK: OTIMIZAÇÃO DE PNL DIFERENCIÁVEL DE MÚLTIPLAS CABEÇAS")
    print("-" * 60)

    # 1. Dataset
    X_data, r_data = generate_market_data(samples=1500)
    train_size = 1200
    X_train, r_train = X_data[:train_size], r_data[:train_size]
    X_test, r_test = X_data[train_size:], r_data[train_size:]

    print(f"[Ambiente] {train_size} amostras de treino, {len(X_test)} de teste (Stream).")
    
    # 2. Configurando HSAMA Megazord (T0)
    model = HSAMA(
        in_features=8,
        out_features=2, # Setor 0 = Predição | Setor 1 = Trading
        num_nodes=8,
        state_dim=16,
        max_hops=1,
        context_dim=16
    )

    # 3. Configurando Runtime (T1)
    config = HSAMARuntimeConfig(
        observe_mode="batch",
        raw_output=True,
        replay_config=SurprisalBufferConfig(
            capacity=800,
            replay_ratio=0.5,
            threshold=AdaptiveQuantileThreshold(quantile=0.75)
        ),
        multi_scale_t0_config=MultiScaleT0Config(
            scales=[
                T0ScaleSpec("fast", stride=1, history_length=16, hidden_dim=16),
                T0ScaleSpec("slow", stride=4, history_length=16, hidden_dim=16),
            ]
        )
    )
    runtime = HSAMAOnlineRuntime(model, config=config)

    # 4. Injetando a Função de Utilidade Dificenciável (Trade Loss)
    cost_bps = 0.001 # Custo de 10 bps de corretagem/spread (0.1%)

    def trading_triplex_loss(preds, targets):
        """
        targets são os 'Retornos de Mercado (t+1)'
        preds: [Batch, 2] -> Setor 0: Previsão do Retorno | Setor 1: Posição Contínua
        """
        # --- MSE do Setor 0 (Predição Tradicional de Regressão) ---
        pred_return = preds[:, 0:1]
        loss_mse = F.mse_loss(pred_return, targets, reduction='none')

        # --- PnL Dificenciável do Setor 1 (Trade Ativo) ---
        # Tanh para garantir limites entre Full Short (-1) e Full Long (+1)
        position = torch.tanh(preds[:, 1:2]) 
        
        # O retorno capturado é a posição atual * o retorno futuro
        gross_pnl = position * targets 
        
        # Penaliza as mudanças de posição e adiciona custo transacional
        # Como o per_sample_loss do runtime opera no batch isoladamente (B=n), 
        # aproximamos a variação de posição olhando pra frente no lote.
        shifted_position = torch.roll(position, shifts=1, dims=0)
        shifted_position[0] = position[0] # Borda assume s/ custo
        
        position_diff = torch.abs(position - shifted_position)
        costs = position_diff * cost_bps
        
        net_pnl = gross_pnl - costs
        
        # O Sharp Ratio aproximação: Se queremos maximizar o PnL, invertemos a Loss para o otimizador
        # O modelo vai minimizar a Perda, caindo o valor da função para o limite negativo
        loss_trade = -net_pnl 
        
        # Equilibrar as importâncias
        return (loss_mse + loss_trade).squeeze(1)

    # Assinando a Lógica Financeira no Runtime
    runtime._per_sample_loss = trading_triplex_loss

    # 5. LOOP DE TREINO CRONOLÓGICO
    epochs = 30
    batch_size = 16
    
    cumulative_pnl_history = []

    print("\nIniciando Treinamento e Aprendizado por Reforço Implícito...")
    for ep in range(epochs):
        ep_pnl = 0.0
        
        # Stream simulando o tempo passando sem shuffle!
        for i in range(0, train_size, batch_size):
            bx = X_train[i:min(i + batch_size, train_size)]
            br = r_train[i:min(i + batch_size, train_size)] # br = Retornos T+1
            
            # Observe o runtime processando a loss customizada sem nem saber!
            res = runtime.observe(bx, br)
            
            # Cálculo de Monitoramento PnL (Offline / Accounting Real)
            posicoes = torch.tanh(res.live_prediction[:, 1:2]).detach()
            pnl_bruto = (posicoes * br).sum().item()
            ep_pnl += pnl_bruto
            
        cumulative_pnl_history.append(ep_pnl)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f" Época {ep+1:02d} | PnL Bruto Gerado do Semestre de Treino: {ep_pnl:+.4f} (Métrica Financeira)")

    print(f"\n[Fim] Treino Finalizado. Simulando no Set de Teste Invisto...")

    # 6. AVALIAÇÃO DE INFERÊNCIA NO TESTE C/ CLONE CONTEXT
    runtime.model.eval()
    with torch.no_grad():
        preds_test = runtime.predict(X_test, raw_output=True)
        
        # Calcular Retorno Livre de Trade (Accounting)
        posicoes_teste = torch.tanh(preds_test[:, 1:2])
        pnl_teste_gross = posicoes_teste * r_test
        
        # Custos
        pos_shift = torch.roll(posicoes_teste, shifts=1, dims=0)
        pos_shift[0] = posicoes_teste[0]
        custos_teste = torch.abs(posicoes_teste - pos_shift) * cost_bps
        pnl_liquido_teste = (pnl_teste_gross - custos_teste).cumsum(dim=0).numpy()

        pnl_base_hold = r_test.cumsum(dim=0).numpy() # Buy and Hold Baseline

    print(f"Retorno Buy&Hold Acumulado: {pnl_base_hold[-1][0]:+.4f}")
    print(f"Retorno do HSAMA Trade Agente: {pnl_liquido_teste[-1][0]:+.4f}\n")

    return True

if __name__ == "__main__":
    run_trading_benchmark()
