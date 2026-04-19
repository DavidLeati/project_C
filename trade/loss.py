import torch
import torch.nn.functional as F

class TriplexTradingLoss:
    """
    Função Multi-Objetivo (Actor-Critic Style) para a Rede HSAMA de Trade.
    Setor 0: MSE Padrão sobre o log-return para ancorar as representações matemáticas (Contexto/GRU)
    Setor 1: Diferencial de PnL para forçar os Hypernets da Action Node a aprender as posições que maximizam ganhos.
    """
    def __init__(self, cost_bps: float = 0.0005, trade_weight: float = 2.0):
        # 0.0005 = 5 bps (Base points) de custodia/spread em exchanges padrao
        self.cost_bps = cost_bps
        # Peso da agressividade do modelo em lucrar versus o peso em ser certinho previsor de preço
        self.trade_weight = trade_weight

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        targets são os 'Retornos de Mercado (t+1)' 
        preds: [Batch, 2] -> Setor 0: Previsão Estocastica | Setor 1: Output Direcional
        """
        # Setor 0 Crítico: Força a GRU a conhecer a direção primária
        pred_return = preds[:, 0:1]
        loss_mse = F.mse_loss(pred_return, targets, reduction='none')

        # Setor 1 Ator: Alocação Contínua [-1 até +1]
        position = torch.tanh(preds[:, 1:2]) 
        
        # PnL Bruto do Trade Capturado no step (t)->(t+1)
        gross_pnl = position * targets 
        
        # Cálculo vetorial da virada de posição (Custo Corretagem)
        # Shift captura a posição em x[0] -> x[1], como o loss opera em array batch (sem estado global intra-batch), 
        # essa aproximação do batch compensa pra rede no backprop.
        shifted_position = torch.roll(position, shifts=1, dims=0)
        shifted_position[0] = position[0] 
        
        position_diff = torch.abs(position - shifted_position)
        costs = position_diff * self.cost_bps
        
        net_pnl = gross_pnl - costs
        
        # O modelo sempre vai Minimizar a Loss Final.
        # Portanto, o PnL Net Negativo é nossa Função de Perda Objetiva.
        loss_trade = -net_pnl 
        
        # Superposição Customizada de Loss Agregando as Duas Cabeças!
        total_sample_loss = loss_mse + (loss_trade * self.trade_weight)
        
        # Remove empty inner dim e retorna 1D vector (Exigência do Runtime)
        return total_sample_loss.squeeze(1)
