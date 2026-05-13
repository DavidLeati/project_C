import torch
import torch.nn.functional as F


def position_from_logits(logits: torch.Tensor, deadzone: float = 0.02, max_leverage: float = 20.0) -> torch.Tensor:
    """Mapeia logits para posicao e alavancagem, zerando exposicao pequena para evitar giro em ruido."""
    if logits.shape[-1] == 1:
        raw_position = torch.tanh(logits)
        leverage = torch.ones_like(raw_position)
    else:
        raw_position = torch.tanh(logits[..., 0:1])
        # Leverage goes from 1.0 to max_leverage smoothly
        leverage = 1.0 + (max_leverage - 1.0) * torch.sigmoid(logits[..., 1:2])

    if deadzone <= 0.0:
        return raw_position * leverage

    abs_position = raw_position.abs()
    active = F.relu(abs_position - deadzone) / (1.0 - deadzone)
    return raw_position.sign() * active * leverage


# ---------------------------------------------------------------------------
# Loss do Preditor: Gaussian NLL + Incentivo Direcional
# ---------------------------------------------------------------------------

def predictor_directional_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    sigma_offset: float = 8.0,
    sigma_floor: float = 1e-4,
    direction_weight: float = 0.5,
) -> torch.Tensor:
    """
    Combina Gaussian NLL com incentivo direcional.

    O NLL puro tende a convergir para mu=0 e sigma grande (aposta segura).
    O componente direcional recompensa sign(mu) == sign(target), incentivando
    o preditor a pelo menos acertar a direcao, mesmo com magnitude imprecisa.

    preds:   [B, 2] -> col 0: mu, col 1: log_sigma_raw
    targets: [B, 1] -> retorno real
    """
    mu = preds[:, 0:1]
    sigma = F.softplus(preds[:, 1:2] - sigma_offset) + sigma_floor

    # Gaussian NLL
    standardized_error = ((targets - mu) / sigma).pow(2)
    calibration_penalty = torch.log(sigma / sigma_floor)
    nll = (0.5 * standardized_error + calibration_penalty)

    # Incentivo Direcional: penaliza quando sign(mu) != sign(target)
    # Usa uma proxy suave: -mu * sign(target) (negativo quando acerta, positivo quando erra)
    # Normalizada pelo sigma para escalar adequadamente
    direction_signal = -mu * targets.sign() / sigma.detach()
    # Clipa para evitar gradientes explosivos
    direction_loss = direction_signal.clamp(-3.0, 3.0)

    total = nll + direction_weight * direction_loss
    return total.reshape(preds.size(0), -1).mean(dim=1)


# ---------------------------------------------------------------------------
# Loss do Trader: PnL-Driven com controle de risco por Sharpe
# ---------------------------------------------------------------------------

class TriplexTradingLoss:
    """
    Loss de trading para o agente de posicao.

    O modelo agora tem out_features=1 (apenas logit de posicao).
    A loss maximiza PnL liquido com regularizacao de vies, estagnacao e
    qualidade risco-retorno do fluxo de PnL.

    Componentes:
      1. PnL Liquido: -net_pnl (maximiza retorno apos custos)
      2. Penalidade de Vies: posicao media do batch^2 (impede always-long/short)
      3. Penalidade de Estagnacao: penaliza |pos_mean| > threshold
      4. Sharpe do batch: -mean(net_pnl) / std(net_pnl), clipado para estabilidade

    Nota: A penalidade de diversidade (1/var) foi REMOVIDA pois causava
    flipping constante entre +1 e -1 no OOS, gerando custos catastroficos.
    O custo de transacao elevado (ankle weights) naturalmente desincentiva
    mudancas de posicao desnecessarias.
    """

    def __init__(
        self,
        cost_bps: float       = 0.002,
        trade_weight: float   = 5.0,
        return_scale: float   = 100.0,
        bias_weight: float    = 3.0,
        stagnation_threshold: float = 0.6,
        stagnation_weight: float = 2.0,
        sharpe_weight: float = 0.0,
        sharpe_eps: float = 1e-6,
        sharpe_clip: float = 3.0,
        position_deadzone: float = 0.02,
        gamma: float          = 0.0,
        previous_position: float | torch.Tensor = 0.0,
    ):
        self.cost_bps       = cost_bps
        self.trade_weight   = trade_weight
        self.return_scale   = return_scale
        self.bias_weight    = bias_weight
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_weight = stagnation_weight
        self.sharpe_weight = sharpe_weight
        self.sharpe_eps = sharpe_eps
        self.sharpe_clip = sharpe_clip
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
        preds:   [Batch, 1 ou 2]  -> logit de posicao (e opcionalmente alavancagem)
        targets: [Batch, 1]  -> log-retorno real do passo t+1
        """
        # IMPORTANTE: deadzone=0 aqui para manter fluxo de gradientes.
        # O deadzone hard e aplicado apenas na engine para metricas/logs.
        # position_from_logits ja lida com a alavancagem se preds tiver dimensao 2
        position = position_from_logits(preds, deadzone=0.0)

        # --- HORIZONTE ALONGADO (Discounted Future Returns) ---
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

        # 1. Minimizar loss_trade == maximizar net_pnl
        loss_trade = -net_pnl  # [B, 1]

        # Regularizacoes batch-level: so fazem sentido com B > 1.
        # No OOS step-by-step (B=1), aplicar apenas o PnL puro.
        batch_size = position.size(0)
        if batch_size > 1:
            if preds.shape[-1] == 2:
                # Extrai apenas a direcionalidade [-1, 1] para a penalidade
                raw_pos = torch.tanh(preds[..., 0:1])
            else:
                raw_pos = position

            # 2. Penalidade de Vies Direcional
            pos_mean = raw_pos.mean()
            loss_bias = pos_mean.pow(2).expand_as(net_pnl)

            # 3. Penalidade de Estagnacao
            excess = F.relu(pos_mean.abs() - self.stagnation_threshold)
            loss_stagnation = excess.pow(2).expand_as(net_pnl)

            pnl_std = net_pnl.std(unbiased=False).clamp_min(self.sharpe_eps)
            sharpe = (net_pnl.mean() / pnl_std).clamp(-self.sharpe_clip, self.sharpe_clip)
            loss_sharpe = (-sharpe).expand_as(net_pnl)

            total = (self.trade_weight * loss_trade
                     + self.bias_weight * loss_bias
                     + self.stagnation_weight * loss_stagnation
                     + self.sharpe_weight * loss_sharpe)
        else:
            total = self.trade_weight * loss_trade

        return total.squeeze(1)
