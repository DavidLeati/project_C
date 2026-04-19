"""
src/learning/objectives.py

Funções de objetivo meta modulares para o meta-otimizador HSAMA.

Princípios de Design
-------------------
1. Nomeado ``objective_fn`` / ``BaseMetaObjective`` — NÃO "reward proxy".
   Este é um objetivo meta diferenciável geral; pode conter termos semelhantes a
   recompensas, mas não carrega semântica específica de RL.

2. Retorna :class:`ObjectiveOutput` com ``total_loss`` (escalar) **e**
   ``components`` (dict de escalares destacados por termo). O chamador nunca
   recebe um escalar opaco; o registro por componente é obrigatório.

3. A normalização é real, não cosmética. Toda função auxiliar retorna um
   **tensor por amostra** de formato ``[B]`` para que :func:`_safe_normalize` possa
   atuar na dimensão do lote antes da redução ponderada para um escalar. Isso
   garante que todos os termos entrem na mistura com variância comparável,
   independentemente de sua escala intrínseca.

4. ``metadata`` é consumido concretamente. Chaves suportadas:
     - ``"sample_weight"``  : FloatTensor ``[B]``  — peso de importância por amostra.
     - ``"regime_mask"``    : BoolTensor  ``[B]``  — True = evento anômalo / raro.
       Quando fornecido, amostras de tail-penalty são amplificadas adicionalmente por
       ``regime_tail_multiplier``.
     - ``"tail_multiplier"`` : float (padrão 2.0) — amplificador escalar aplicado
       a amostras de cauda que *também* são marcadas por ``regime_mask``.

5. Limitação (documentada, não oculta): o sinal de gradiente usado para construir
   recursos de bloco no meta-otimizador atualmente vem do MSE simples (veja
   ``HSAMAMetaOptimizer.step``). O objetivo, portanto, governa *quão fortemente
   seguir* a direção derivada do MSE, e não a direção em si. Esta é uma
   escolha deliberada da v1; versões futuras podem expor o direcionamento do gradiente.
"""

from __future__ import annotations

import abc
import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class ObjectiveOutput:
    """Saída estruturada de qualquer :class:`BaseMetaObjective`.

    Atributos:
        total_loss: Tensor escalar adequado para ``.backward()``.
        components: Valores nomeados por termo **antes** da ponderação, mantidos como
                     escalares destacados. Nunca colapse estes em
                     ``total_loss`` antes de apresentá-los ao chamador —
                     observabilidade é o objetivo.
    """
    total_loss: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)

    def log_dict(self) -> dict[str, float]:
        """Dicionário de floats destacados adequado para loggers / rastreadores de métricas."""
        out: dict[str, float] = {"total_loss": self.total_loss.detach().item()}
        for k, v in self.components.items():
            out[k] = v.detach().item() if v.numel() == 1 else v.detach().mean().item()
        return out


# ---------------------------------------------------------------------------
# Base contract
# ---------------------------------------------------------------------------

class BaseMetaObjective(abc.ABC):
    """Interface para todos os meta-objetivos HSAMA.

    Os implementadores recebem ``predictions`` and ``targets`` (ambos tensores
    diferenciáveis do motor de desenrolamento) e um dicionário ``metadata`` opcional
    que pode carregar sinalizadores de regime, pesos de amostra ou qualquer contexto auxiliar de nível de lote.
    Deve retornar um :class:`ObjectiveOutput`.
    """

    @abc.abstractmethod
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> ObjectiveOutput:
        ...

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> ObjectiveOutput:
        return self.forward(predictions, targets, metadata)

    def peek(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> ObjectiveOutput:
        """Avalia o objetivo e restaura imediatamente seu estado interno.

        Objetivos com estado devem expor todo o estado rolante mutável através de
        ``state_dict`` / ``load_state_dict``. Os chamadores usam este auxiliador quando
        precisam de gradientes do objetivo sem consumir uma observação de treinamento real
        para o histórico próprio do objetivo.
        """
        snapshot = copy.deepcopy(self.state_dict())
        output = self.forward(predictions, targets, metadata)
        self.load_state_dict(snapshot)
        return output

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: Optional[dict]):
        return self


# ---------------------------------------------------------------------------
# Batch-level utilities
# ---------------------------------------------------------------------------

def _safe_normalize(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalização média-zero / desvio-padrão unitário sobre a dimensão do **lote** (batch).

    A entrada deve ser um tensor 1-D por amostra ``[B]``. A saída é limitada a
    ``[-3, 3]`` (≈ 99,7% de uma Gaussiana) para que amostras discrepantes não
    dominem a mistura antes que a ponderação por termo seja aplicada.

    Args:
        t: Tensor por amostra de formato ``[B]``.
        eps: Protetor de estabilidade numérica para o desvio padrão.

    Returns:
        Tensor normalizado de formato ``[B]``.
    """
    mean = t.mean()
    std = t.std(unbiased=False).clamp(min=eps)
    return ((t - mean) / std).clamp(-3.0, 3.0)


def _resolve_sample_weight(
    metadata: Optional[dict],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Extrai ``sample_weight`` dos metadados ou retorna pesos uniformes.

    Retorna um tensor de pesos normalizado de formato ``[B]`` que soma 1.0,
    para que a média ponderada esteja na mesma escala que uma média não ponderada.
    """
    if metadata is not None and "sample_weight" in metadata:
        w = metadata["sample_weight"].to(device=device, dtype=dtype)
        w = w.clamp(min=0.0)
        total = w.sum()
        return w / total.clamp(min=1e-8)
    # Uniforme: cada amostra contribui igualmente
    return torch.full((batch_size,), 1.0 / batch_size, device=device, dtype=dtype)


def _resolve_regime_multiplier(
    metadata: Optional[dict],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    base_multiplier: float = 2.0,
) -> torch.Tensor:
    """Constrói um amplificador de regime por amostra de formato ``[B]``.

    Amostras onde ``metadata["regime_mask"]`` é True recebem
    ``metadata.get("tail_multiplier", base_multiplier)``, outras recebem 1.0.
    """
    multiplier = torch.ones(batch_size, device=device, dtype=dtype)
    if metadata is not None and "regime_mask" in metadata:
        mask = metadata["regime_mask"].to(device=device)
        amp = float(metadata.get("tail_multiplier", base_multiplier))
        multiplier[mask] = amp
    return multiplier


# ---------------------------------------------------------------------------
# Per-sample component helpers
# Todos os auxiliadores retornam formato [B] para que _safe_normalize possa atuar na dimensão do lote antes da redução ponderada para um escalar.
# ---------------------------------------------------------------------------

def _directional_samples(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Proxy de precisão direcional por amostra — formato ``[B]``.

    Usa ``-tanh(pred * softsign(target) / temperature)``:
      - Direção correta → negativo (perda minimizada ↔ boa direção).
      - Direção errada   → positivo.
      - Suave em todos os lugares, sem limiares rígidos.

    O chamador deve achatar ``predictions`` e ``targets`` para ``[B]``
    antes de passá-los se eles tiverem dimensões de saída extras.
    """
    pred_flat = predictions.flatten(start_dim=1).mean(dim=1) if predictions.dim() > 1 else predictions.squeeze(-1)
    tgt_flat  = targets.flatten(start_dim=1).mean(dim=1)     if targets.dim() > 1    else targets.squeeze(-1)
    soft_sign = tgt_flat / (tgt_flat.abs() + eps)
    alignment = (pred_flat * soft_sign) / temperature
    return -torch.tanh(alignment)   # [B]


def _tail_samples_from_threshold(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: torch.Tensor,
    regime_multiplier: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalidade de cauda (tail-penalty) por amostra — formato ``[B]``.

    Amostras cujo ``|target|`` excede o ``threshold`` são "eventos de cauda".
    Seu erro quadrático de previsão é retornado diretamente; amostras que não são de cauda recebem 0.
    O ``regime_multiplier`` opcional (formato ``[B]``) amplifica ainda mais as amostras
    marcadas como eventos raros pelo chamador.
    """
    pred_flat = predictions.flatten(start_dim=1).mean(dim=1) if predictions.dim() > 1 else predictions.squeeze(-1)
    tgt_flat  = targets.flatten(start_dim=1).mean(dim=1)     if targets.dim() > 1    else targets.squeeze(-1)

    abs_targets = tgt_flat.abs()
    tail_mask   = (abs_targets >= threshold).float()   # [B]

    squared_err = (pred_flat - tgt_flat).pow(2)        # [B]
    per_sample  = squared_err * tail_mask               # [B]: 0 for non-tail

    if regime_multiplier is not None:
        per_sample = per_sample * regime_multiplier

    return per_sample   # [B]


def _tail_samples(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantile: float = 0.9,
    regime_multiplier: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    abs_targets = targets.flatten(start_dim=1).mean(dim=1).abs() if targets.dim() > 1 else targets.squeeze(-1).abs()
    threshold = torch.quantile(abs_targets, quantile)
    return _tail_samples_from_threshold(
        predictions,
        targets,
        threshold,
        regime_multiplier,
    )


def _drawdown_proxy_samples(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Proxy de exposição a drawdown local por amostra — formato ``[B]``.

    Penaliza previsões que têm a mesma direção positiva que alvos negativos
    (ou seja, o modelo "entrou em massa" em um movimento de queda).

    Nota: este é um **proxy local do lote**, não um drawdown sequencial real.
    O drawdown real requer acumulação temporal. O nome da classe
    ``DrawdownProxyObjective`` torna isso explícito.
    """
    pred_flat = predictions.flatten(start_dim=1).mean(dim=1) if predictions.dim() > 1 else predictions.squeeze(-1)
    tgt_flat  = targets.flatten(start_dim=1).mean(dim=1)     if targets.dim() > 1    else targets.squeeze(-1)

    negative_mask = (tgt_flat < 0).float()                   # [B]
    exposure      = (pred_flat * negative_mask).clamp(min=0.0)  # positive = bad
    return exposure   # [B]


def _weighted_mean(
    per_sample: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Média ponderada de um tensor por amostra usando pesos pré-normalizados."""
    return (per_sample * sample_weight).sum()


def _weighted_variance(
    per_sample: torch.Tensor,
    sample_weight: torch.Tensor,
    mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Variância de lote ponderada usando pesos pré-normalizados."""
    if mean is None:
        mean = _weighted_mean(per_sample, sample_weight)
    centered = per_sample - mean
    return (sample_weight * centered.pow(2)).sum()


def _return_samples(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metadata: Optional[dict],
    *,
    position_temperature: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constrói retornos por amostra diferenciáveis e as posições implícitas.

    O caminho primário usa ``targets`` como o sinal de retorno realizado, que é
    o contrato de regime de mercado atual. Se os metadados expuserem um tensor
    explícito de ``realized_return`` / ``returns``, esse sinal terá precedência.
    """
    pred_scalar = predictions.flatten(start_dim=1).mean(dim=1) if predictions.dim() > 1 else predictions.reshape(-1)
    realized = targets.flatten(start_dim=1).mean(dim=1) if targets.dim() > 1 else targets.reshape(-1)
    if metadata is not None:
        for key in ("realized_return", "returns", "return"):
            value = metadata.get(key)
            if isinstance(value, torch.Tensor):
                realized = value.to(device=predictions.device, dtype=predictions.dtype).reshape(-1)
                break
    scale = max(float(position_temperature), eps)
    position = torch.tanh(pred_scalar / scale)
    return position * realized, position


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class MSEObjective(BaseMetaObjective):
    """Objetivo de Erro Quadrático Médio (MSE).

    Implementa o critério padrão para regressão, suportando ``sample_weight`` nos metadados.
    """

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> ObjectiveOutput:
        B = predictions.shape[0]
        w = _resolve_sample_weight(metadata, B, predictions.device, predictions.dtype)
        per_sample = F.mse_loss(predictions, targets, reduction="none").flatten(start_dim=1).mean(dim=1)
        loss = _weighted_mean(per_sample, w)
        return ObjectiveOutput(
            total_loss=loss,
            components={"mse": loss.detach()},
        )


class MAEObjective(BaseMetaObjective):
    """Objetivo de erro médio absoluto ponderado."""

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> ObjectiveOutput:
        batch_size = predictions.shape[0]
        weights = _resolve_sample_weight(metadata, batch_size, predictions.device, predictions.dtype)
        per_sample = F.l1_loss(predictions, targets, reduction="none").flatten(start_dim=1).mean(dim=1)
        loss = _weighted_mean(per_sample, weights)
        return ObjectiveOutput(
            total_loss=loss,
            components={"mae": loss.detach()},
        )


class HuberObjective(BaseMetaObjective):
    """Perda Huber ponderada para regressão mais robusta a discrepâncias (outliers)."""

    def __init__(self, delta: float = 1.0):
        self.delta = float(delta)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> ObjectiveOutput:
        batch_size = predictions.shape[0]
        weights = _resolve_sample_weight(metadata, batch_size, predictions.device, predictions.dtype)
        per_sample = F.huber_loss(
            predictions,
            targets,
            reduction="none",
            delta=self.delta,
        ).flatten(start_dim=1).mean(dim=1)
        loss = _weighted_mean(per_sample, weights)
        return ObjectiveOutput(
            total_loss=loss,
            components={"huber": loss.detach()},
        )


class AsymmetricHuberObjective(BaseMetaObjective):
    """Regressão no estilo Huber que penaliza a subestimação mais do que a superestimação."""

    def __init__(
        self,
        delta: float = 1.0,
        underestimation_weight: float = 2.0,
        overestimation_weight: float = 1.0,
    ):
        self.delta = float(delta)
        self.underestimation_weight = float(underestimation_weight)
        self.overestimation_weight = float(overestimation_weight)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> ObjectiveOutput:
        batch_size = predictions.shape[0]
        weights = _resolve_sample_weight(metadata, batch_size, predictions.device, predictions.dtype)
        per_sample_huber = F.huber_loss(
            predictions,
            targets,
            reduction="none",
            delta=self.delta,
        ).flatten(start_dim=1).mean(dim=1)
        pred_scalar = predictions.flatten(start_dim=1).mean(dim=1)
        target_scalar = targets.flatten(start_dim=1).mean(dim=1)
        asymmetry = torch.where(
            pred_scalar < target_scalar,
            per_sample_huber.new_full((batch_size,), self.underestimation_weight),
            per_sample_huber.new_full((batch_size,), self.overestimation_weight),
        )
        per_sample = per_sample_huber * asymmetry
        loss = _weighted_mean(per_sample, weights)
        return ObjectiveOutput(
            total_loss=loss,
            components={"asymmetric_huber": loss.detach()},
        )


class TradingCompositeObjective(BaseMetaObjective):
    """Meta-objetivo composto para cenários de trading.

    Mistura três termos, cada um **normalizado sobre a dimensão do lote** antes
    da ponderação:

    1. ``directional_term``  — proxy de precisão da direção do sinal diferenciável.
    2. ``tail_penalty``      — erro amplificado em alvos de magnitude extrema.
    3. ``drawdown_proxy``    — penalidade local para exposição direcional à queda.

    Procedimento de normalização por termo
    -------------------------------------
    Cada auxiliador retorna um tensor por amostra ``[B]``.
    :func:`_safe_normalize` é aplicado **antes** da redução ponderada para um
    escalar, para que cada termo entre na mistura com média zero e variância unitária —
    tornando os pesos ``weight_*`` genuinamente comparáveis.

    Chaves de ``metadata`` suportadas
    --------------------------------
    ``"sample_weight"``   : FloatTensor ``[B]`` — peso de importância por amostra.
    ``"regime_mask"``     : BoolTensor  ``[B]`` — True = evento anômalo/raro;
                            amplifica essas amostras na penalidade de cauda.
    ``"tail_multiplier"`` : float — multiplicador para amostras de cauda com máscara de regime
                            (padrão 2.0).

    Args:
        weight_directional:    Peso de mistura para o termo direcional.
        weight_tail:           Peso de mistura para a penalidade de cauda.
        weight_drawdown_proxy: Peso de mistura para o proxy de drawdown.
        tail_quantile:         Limiar de quantil que define amostras de "cauda".
        directional_temperature: Suavidade do sinal direcional tanh.
        component_clip:        Se definido, cada termo normalizado é transformado em
                                ``[-component_clip, component_clip]``.
        eps:                   Epsilon de estabilidade numérica.
    """

    def __init__(
        self,
        weight_directional:    float = 1.0,
        weight_tail:           float = 0.5,
        weight_drawdown_proxy: float = 0.3,
        tail_quantile:         float = 0.9,
        tail_quantile_mode:    str = "adaptive_rolling",
        tail_quantile_window_batches: int = 32,
        tail_small_batch_threshold: int = 32,
        directional_temperature: float = 1.0,
        component_clip:        Optional[float] = 3.0,
        eps: float = 1e-8,
    ):
        self.weight_directional    = weight_directional
        self.weight_tail           = weight_tail
        self.weight_drawdown_proxy = weight_drawdown_proxy
        self.tail_quantile         = tail_quantile
        self.tail_quantile_mode    = tail_quantile_mode
        self.tail_quantile_window_batches = max(1, int(tail_quantile_window_batches))
        self.tail_small_batch_threshold = max(1, int(tail_small_batch_threshold))
        self.directional_temperature = directional_temperature
        self.component_clip        = component_clip
        self.eps                   = eps
        self._tail_history = deque(maxlen=self.tail_quantile_window_batches)

    def _clip(self, t: torch.Tensor) -> torch.Tensor:
        if self.component_clip is not None:
            return t.clamp(-self.component_clip, self.component_clip)
        return t

    def _history_tensor(
        self,
        abs_targets: torch.Tensor,
    ) -> torch.Tensor:
        history = [entry.to(device=abs_targets.device, dtype=abs_targets.dtype) for entry in self._tail_history]
        if history:
            return torch.cat(history + [abs_targets.detach()], dim=0)
        return abs_targets.detach()

    def _tail_threshold(self, abs_targets: torch.Tensor) -> torch.Tensor:
        if self.tail_quantile_mode not in {"adaptive_rolling", "batch"}:
            raise ValueError(
                "tail_quantile_mode must be one of {'adaptive_rolling', 'batch'}."
            )
        if (
            self.tail_quantile_mode == "batch"
            or int(abs_targets.numel()) >= self.tail_small_batch_threshold
        ):
            threshold_source = abs_targets
        else:
            threshold_source = self._history_tensor(abs_targets)
        return torch.quantile(threshold_source, self.tail_quantile)

    def _observe_tail_batch(self, abs_targets: torch.Tensor):
        self._tail_history.append(abs_targets.detach().cpu())

    def state_dict(self) -> dict:
        return {
            "tail_quantile_mode": self.tail_quantile_mode,
            "tail_quantile_window_batches": self.tail_quantile_window_batches,
            "tail_small_batch_threshold": self.tail_small_batch_threshold,
            "tail_history": [entry.clone() for entry in self._tail_history],
        }

    def load_state_dict(self, state: Optional[dict]):
        state = dict(state or {})
        self.tail_quantile_mode = str(state.get("tail_quantile_mode", self.tail_quantile_mode))
        self.tail_quantile_window_batches = max(
            1,
            int(state.get("tail_quantile_window_batches", self.tail_quantile_window_batches)),
        )
        self.tail_small_batch_threshold = max(
            1,
            int(state.get("tail_small_batch_threshold", self.tail_small_batch_threshold)),
        )
        self._tail_history = deque(
            (entry.detach().clone().cpu() for entry in state.get("tail_history", [])),
            maxlen=self.tail_quantile_window_batches,
        )
        return self

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> ObjectiveOutput:
        B      = predictions.shape[0]
        device = predictions.device
        dtype  = predictions.dtype

        # --- Resolve metadata ---------------------------------------------------
        sample_weight    = _resolve_sample_weight(metadata, B, device, dtype)
        regime_multiplier = _resolve_regime_multiplier(metadata, B, device, dtype)

        # --- 1. Per-sample raw components [B] ----------------------------------
        raw_dir  = _directional_samples(predictions, targets, self.directional_temperature, self.eps)
        abs_targets = targets.flatten(start_dim=1).mean(dim=1).abs() if targets.dim() > 1 else targets.squeeze(-1).abs()
        tail_threshold = self._tail_threshold(abs_targets)
        raw_tail = _tail_samples_from_threshold(
            predictions,
            targets,
            tail_threshold,
            regime_multiplier,
        )
        raw_draw = _drawdown_proxy_samples(predictions, targets)
        self._observe_tail_batch(abs_targets)

        # --- 2. Normalise over batch dimension (where B > 1) -------------------
        # _safe_normalize requires at least 2 samples to be meaningful
        if B > 1:
            norm_dir  = _safe_normalize(raw_dir,  self.eps)
            norm_tail = _safe_normalize(raw_tail, self.eps)
            norm_draw = _safe_normalize(raw_draw, self.eps)
        else:
            norm_dir, norm_tail, norm_draw = raw_dir, raw_tail, raw_draw

        # --- 3. Clip each normalised component ---------------------------------
        dir_term  = self._clip(norm_dir)
        tail_term = self._clip(norm_tail)
        draw_term = self._clip(norm_draw)

        # --- 4. Weighted reduction to scalar (respects sample_weight) ----------
        dir_scalar  = _weighted_mean(dir_term,  sample_weight)
        tail_scalar = _weighted_mean(tail_term, sample_weight)
        draw_scalar = _weighted_mean(draw_term, sample_weight)

        # --- 5. Blend with explicit term weights --------------------------------
        total = (
            self.weight_directional    * dir_scalar
            + self.weight_tail         * tail_scalar
            + self.weight_drawdown_proxy * draw_scalar
        )

        return ObjectiveOutput(
            total_loss=total,
            components={
                "directional_term": dir_scalar.detach(),
                "tail_penalty":     tail_scalar.detach(),
                "drawdown_proxy":   draw_scalar.detach(),
            },
        )


class AnchoredTradingCompositeObjective(TradingCompositeObjective):
    """Composto de trading com uma âncora de regressão explícita.

    Adiciona uma âncora de regressão positiva que mantém o objetivo meta vinculado à 
    qualidade da previsão absoluta. A âncora é reduzida sem normalização por lote
    para que erros de grande magnitude permaneçam caros, complementando os termos 
    de trading normalizados.
    """

    def __init__(
        self,
        *,
        weight_anchor: float = 1.0,
        anchor_loss: str = "mse",
        anchor_delta: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        normalized_anchor_loss = str(anchor_loss).lower()
        if normalized_anchor_loss not in {"mse", "huber"}:
            raise ValueError("anchor_loss deve ser um de {'mse', 'huber'}.")
        self.weight_anchor = float(weight_anchor)
        self.anchor_loss = normalized_anchor_loss
        self.anchor_delta = float(anchor_delta)

    def _anchor_per_sample(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if self.anchor_loss == "huber":
            return F.huber_loss(
                predictions,
                targets,
                reduction="none",
                delta=self.anchor_delta,
            ).flatten(start_dim=1).mean(dim=1)
        return F.mse_loss(
            predictions,
            targets,
            reduction="none",
        ).flatten(start_dim=1).mean(dim=1)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> ObjectiveOutput:
        trading_output = super().forward(predictions, targets, metadata)
        batch_size = predictions.shape[0]
        sample_weight = _resolve_sample_weight(
            metadata,
            batch_size,
            predictions.device,
            predictions.dtype,
        )
        anchor_per_sample = self._anchor_per_sample(predictions, targets)
        anchor_scalar = _weighted_mean(anchor_per_sample, sample_weight)
        total = trading_output.total_loss + (self.weight_anchor * anchor_scalar)
        return ObjectiveOutput(
            total_loss=total,
            components={
                "anchor_loss": anchor_scalar.detach(),
                **trading_output.components,
            },
        )


class SharpeRatioObjective(BaseMetaObjective):
    """Objetivo Sharpe diferenciável em nível de lote.

    Este objetivo trata a saída do modelo como um sinal de posição assinado e
    computa um índice Sharpe ponderado sobre os retornos por amostra resultantes.
    A posição é comprimida através de ``tanh`` para que a alavancagem permaneça limitada e o
    objetivo permaneça numericamente estável.

    Suporte a metadados:
      - ``sample_weight``: mesmo contrato que outros objetivos.
      - ``realized_return`` / ``returns`` / ``return``: tensor opcional que
        substitui ``targets`` como o fluxo de retorno realizado.

    Limitação: este ainda é local do lote em vez de um backtest sequencial completo.
    O objetivo otimiza um proxy Sharpe diferenciável sobre a distribuição do lote observado.
    """

    def __init__(
        self,
        position_temperature: float = 1.0,
        volatility_floor: float = 1e-4,
        eps: float = 1e-8,
    ):
        self.position_temperature = float(position_temperature)
        self.volatility_floor = float(volatility_floor)
        self.eps = float(eps)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> ObjectiveOutput:
        batch_size = predictions.shape[0]
        weights = _resolve_sample_weight(metadata, batch_size, predictions.device, predictions.dtype)
        per_sample_return, position = _return_samples(
            predictions,
            targets,
            metadata,
            position_temperature=self.position_temperature,
            eps=self.eps,
        )
        mean_return = _weighted_mean(per_sample_return, weights)
        variance = _weighted_variance(per_sample_return, weights, mean=mean_return)
        volatility = variance.clamp(min=self.volatility_floor**2 + self.eps).sqrt()
        sharpe = mean_return / volatility
        total_loss = -sharpe
        mean_abs_position = _weighted_mean(position.abs(), weights)
        return ObjectiveOutput(
            total_loss=total_loss,
            components={
                "sharpe_ratio": sharpe.detach(),
                "mean_return": mean_return.detach(),
                "return_volatility": volatility.detach(),
                "mean_abs_position": mean_abs_position.detach(),
            },
        )


class DrawdownProxyObjective(BaseMetaObjective):
    """Meta-objetivo defensivo que minimiza a exposição local a drawdown.

    Adequado quando a principal preocupação é a proteção de capital em vez da
    maximização do retorno. Combina precisão direcional com um proxy de drawdown amplificado.

    **Ressalva do Proxy**: o drawdown real é uma métrica cumulativa sequencial. Este
    objetivo opera no nível do lote. O sufixo "Proxy" no nome da classe
    é intencional e documenta esta limitação explicitamente.

    Suporta as mesmas chaves de ``metadata`` que :class:`TradingCompositeObjective`:
    ``"sample_weight"`` e ``"regime_mask"`` / ``"tail_multiplier"``.
    """

    def __init__(
        self,
        weight_directional:    float = 0.5,
        weight_drawdown_proxy: float = 1.0,
        component_clip:        Optional[float] = 3.0,
        eps: float = 1e-8,
    ):
        self.weight_directional    = weight_directional
        self.weight_drawdown_proxy = weight_drawdown_proxy
        self.component_clip        = component_clip
        self.eps                   = eps

    def _clip(self, t: torch.Tensor) -> torch.Tensor:
        if self.component_clip is not None:
            return t.clamp(-self.component_clip, self.component_clip)
        return t

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> ObjectiveOutput:
        B      = predictions.shape[0]
        device = predictions.device
        dtype  = predictions.dtype

        sample_weight = _resolve_sample_weight(metadata, B, device, dtype)

        raw_dir  = _directional_samples(predictions, targets, eps=self.eps)
        raw_draw = _drawdown_proxy_samples(predictions, targets)

        if B > 1:
            norm_dir  = _safe_normalize(raw_dir,  self.eps)
            norm_draw = _safe_normalize(raw_draw, self.eps)
        else:
            norm_dir, norm_draw = raw_dir, raw_draw

        dir_scalar  = _weighted_mean(self._clip(norm_dir),  sample_weight)
        draw_scalar = _weighted_mean(self._clip(norm_draw), sample_weight)

        total = (
            self.weight_directional    * dir_scalar
            + self.weight_drawdown_proxy * draw_scalar
        )

        return ObjectiveOutput(
            total_loss=total,
            components={
                "directional_term": dir_scalar.detach(),
                "drawdown_proxy":   draw_scalar.detach(),
            },
        )
