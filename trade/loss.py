import torch
import torch.nn.functional as F


def position_from_logits(logits: torch.Tensor, deadzone: float = 0.05) -> torch.Tensor:
    """Mapeia logits para posicao e zera exposicao pequena para evitar giro em ruido."""
    raw_position = torch.tanh(logits)
    if deadzone <= 0.0:
        return raw_position
    abs_position = raw_position.abs()
    active = F.relu(abs_position - deadzone) / (1.0 - deadzone)
    return raw_position.sign() * active


class TriplexTradingLoss:
    """
    Funcao Multi-Objetivo (Actor-Critic Style) para a Rede HSAMA de Trade.

    Setor 0 (Critico): MSE sobre o log-return — ancora as representacoes
                       matematicas da GRU/contexto.

    Setor 1 (Ator):    Objetivo Sharpe intra-batch — maximiza a razao
                       PnL_medio / PnL_std dentro do batch.
                       Ao contrario do PnL bruto medio, o Sharpe nao pode
                       ser maximizado com "always long/short": uma posicao
                       fixa tem PnL_std proporcional ao std dos retornos,
                       mas PnL_medio proporcional ao retorno medio do periodo.
                       O gradiente obriga o modelo a encontrar timing real:
                       entrar certo e sair certo para ter alta media e baixo std.

    Regularizacao:     Penalidade suave sobre posicoes extremas (|pos| > 0.8).
                       Permite posicoes moderadas livremente, so penaliza
                       saturacao perto de +-1.
    """

    def __init__(
        self,
        cost_bps: float       = 0.0025,   # Treina com custo alto (25bps) para filtrar ruido (ankle weights)
        trade_weight: float   = 50.0,     # Priorizar PnL real
        return_scale: float   = 100.0,
        entropy_weight: float = 0.1,      # Penaliza indecisao
        bias_weight: float    = 0.1,      # Mantido baixo para permitir surfar tendencias
        position_deadzone: float = 0.05,
        gamma: float          = 0.0,
        previous_position: float | torch.Tensor = 0.0,
    ):
        self.cost_bps       = cost_bps
        self.trade_weight   = trade_weight
        self.return_scale   = return_scale
        self.entropy_weight = entropy_weight
        self.bias_weight    = bias_weight
        self.position_deadzone = float(position_deadzone)
        self.gamma          = gamma
        self.previous_position = previous_position

    def set_previous_position(self, previous_position: float | torch.Tensor) -> None:
        """Define a posicao anterior usada para custo na borda do proximo batch."""
        self.previous_position = previous_position

    def _previous_position_tensor(self, position: torch.Tensor) -> torch.Tensor:
        prev = torch.as_tensor(
            self.previous_position,
            device=position.device,
            dtype=position.dtype,
        )
        if prev.numel() == 1:
            return prev.reshape(1).expand(position.size(1))
        return prev.reshape(-1)[: position.size(1)]

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds:   [Batch, 2]  -> col 0: previsao de retorno | col 1: logit de posicao
        targets: [Batch, 1]  -> log-retorno real do passo t+1
        """
        pred_return = preds[:, 0:1]
        loss_mse = F.mse_loss(pred_return, targets, reduction="none")  # [B, 1]

        position = position_from_logits(preds[:, 1:2], deadzone=self.position_deadzone)  # [-1, +1]

        # --- HORIZONTE ALONGADO (Discounted Future Returns) ---
        # Em vez de olhar apenas para o target do próximo candle (t+1),
        # somamos os retornos futuros do batch com um fator de decaimento (gamma).
        # Isso permite que a rede enxergue o lucro de uma tendência longa,
        # justificando pagar a taxa de transação (cost_bps).
        if self.gamma > 0.0 and targets.size(0) > 1:
            pnl_targets = torch.zeros_like(targets)
            pnl_targets[-1] = targets[-1]
            for t in range(targets.size(0) - 2, -1, -1):
                pnl_targets[t] = targets[t] + self.gamma * pnl_targets[t+1]
        else:
            pnl_targets = targets

        scaled_targets = pnl_targets * self.return_scale
        gross_pnl = position * scaled_targets

        shifted_position    = torch.roll(position, shifts=1, dims=0)
        shifted_position[0] = self._previous_position_tensor(position).detach()
        costs   = torch.abs(position - shifted_position) * self.cost_bps * self.return_scale
        net_pnl = gross_pnl - costs  # [B, 1]

        # Minimizar loss_trade == maximizar net_pnl
        loss_trade = -net_pnl  # [B, 1]

        # Penalidade de Indecisao: penaliza posicoes no "meio termo" (|pos| entre 0.0 e 1.0)
        # Isso forca o modelo a tomar uma decisao clara: ficar FLAT (0) ou fully LONG/SHORT (1/-1).
        loss_entropy = (position.abs() * (1.0 - position.abs())).expand_as(net_pnl) # [B, 1]

        # Penalidade de Vies Direcional.
        # Se o modelo fica so LONG (+1) o batch inteiro, a media e +1 e a perda e alta.
        # Obriga o modelo a alternar e encontrar os verdadeiros sinais,
        # impedindo a histerese do custo de transacao.
        loss_bias = position.mean().pow(2).expand_as(net_pnl) # [B, 1]

        total = loss_mse + self.trade_weight * loss_trade + self.entropy_weight * loss_entropy + self.bias_weight * loss_bias

        return total.squeeze(1)
