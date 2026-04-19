from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .surprisal import EMASurprisalEstimator
from .temporal import LogicalClockScheduler, ScaleLatentState, TemporalHistoryBuffer


@dataclass(frozen=True)
class T0ScaleSpec:
    name: str
    stride: int
    history_length: int
    hidden_dim: Optional[int] = None
    surprisal_decay: Optional[float] = None


@dataclass
class MultiScaleT0Config:
    scales: tuple[T0ScaleSpec, ...]
    aggregation: str = "concat"
    append_scale_surprisal: bool = False
    attention_heads: int = 1

    def __post_init__(self):
        self.scales = tuple(self.scales)
        if not self.scales:
            raise ValueError("MultiScaleT0Config requer pelo menos uma escala.")
        if self.aggregation not in {"concat", "attention"}:
            raise ValueError("aggregation deve ser 'concat' ou 'attention'.")


class ScaleEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, context_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_dim, context_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(sequence)
        return self.projection(hidden[-1])


class MultiScaleT0Builder(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        context_dim: int,
        config: MultiScaleT0Config,
        surprisal_beta_mu: float = 0.95,
        surprisal_beta_var: float = 0.95,
        surprisal_eps: float = 1e-8,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.context_dim = int(context_dim)
        self.config = config
        self.scheduler = LogicalClockScheduler()
        self.scale_specs = {spec.name: spec for spec in self.config.scales}
        self.scale_order = [spec.name for spec in self.config.scales]
        self.scale_encoders = nn.ModuleDict(
            {
                spec.name: ScaleEncoder(
                    in_features=self.in_features,
                    hidden_dim=spec.hidden_dim or self.context_dim,
                    context_dim=self.context_dim,
                )
                for spec in self.config.scales
            }
        )
        self.scale_states = {spec.name: ScaleLatentState() for spec in self.config.scales}
        self.scale_estimators = {
            spec.name: EMASurprisalEstimator(
                beta_mu=surprisal_beta_mu,
                beta_var=surprisal_beta_var,
                eps=surprisal_eps,
            )
            for spec in self.config.scales
        }

        if self.config.aggregation == "attention":
            self.attention_query = nn.Parameter(torch.randn(1, 1, self.context_dim) * 0.02)
            self.attention = nn.MultiheadAttention(
                embed_dim=self.context_dim,
                num_heads=self.config.attention_heads,
                batch_first=True,
            )
            projection_in = self.context_dim
        else:
            projection_in = self.context_dim * len(self.scale_order)

        if self.config.append_scale_surprisal:
            projection_in += 2 * len(self.scale_order)

        self.context_projection = nn.Sequential(
            nn.Linear(projection_in, self.context_dim),
            nn.SiLU(),
            nn.Linear(self.context_dim, self.context_dim),
        )

    def required_history(self) -> int:
        return max(spec.stride * (spec.history_length - 1) + 1 for spec in self.config.scales)

    def shared_parameters(self):
        params = list(self.context_projection.parameters())
        if self.config.aggregation == "attention":
            params += list(self.attention.parameters()) + [self.attention_query]
        return params

    def scale_parameters(self, name: str):
        return list(self.scale_encoders[name].parameters())

    def due_scale_names(self, step: int) -> list[str]:
        return [
            spec.name
            for spec in self.config.scales
            if self.scale_states[spec.name].context is None
            or self.scheduler.is_due(step, spec.stride)
        ]

    def prepare_contexts(
        self,
        history: TemporalHistoryBuffer,
        *,
        step: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        contexts: dict[str, torch.Tensor] = {}
        due_scale_names = self.due_scale_names(step)
        for spec in self.config.scales:
            state = self.scale_states[spec.name]
            if spec.name in due_scale_names:
                sequence = history.get_sequence(
                    stride=spec.stride,
                    history_length=spec.history_length,
                    device=device,
                    dtype=dtype,
                )
                contexts[spec.name] = self.scale_encoders[spec.name](sequence)
            else:
                contexts[spec.name] = state.context.to(device=device, dtype=dtype)
        return contexts, due_scale_names

    def compose_context(
        self,
        contexts: dict[str, torch.Tensor],
        *,
        step: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        context_sequence = torch.stack([contexts[name] for name in self.scale_order], dim=1)
        if self.config.aggregation == "attention":
            query = self.attention_query.to(device=device, dtype=dtype).expand(
                context_sequence.size(0), -1, -1
            )
            aggregated, _ = self.attention(query, context_sequence, context_sequence)
            aggregated = aggregated.squeeze(1)
        else:
            aggregated = context_sequence.reshape(context_sequence.size(0), -1)

        metadata = {
            "scale_contexts": context_sequence,
        }
        if self.config.append_scale_surprisal:
            scale_surprisal = []
            scale_age = []
            for name in self.scale_order:
                spec = self.scale_specs[name]
                state = self.scale_states[name]
                decay_steps = spec.surprisal_decay or spec.stride
                scale_surprisal.append(state.live_surprisal(step, decay_steps))
                scale_age.append(min(state.age(step) / max(spec.stride, 1), 1.0))
            scale_surprisal_t = torch.tensor(
                scale_surprisal,
                device=device,
                dtype=dtype,
            ).view(1, -1).expand(aggregated.size(0), -1)
            scale_age_t = (
                torch.tensor(scale_age, device=device, dtype=dtype)
                .view(1, -1)
                .expand(aggregated.size(0), -1)
            )
            aggregated = torch.cat([aggregated, scale_surprisal_t, scale_age_t], dim=-1)
            metadata["scale_surprisal"] = scale_surprisal_t
            metadata["scale_age"] = scale_age_t
        final_context = self.context_projection(aggregated)
        return final_context, metadata

    def commit_contexts(
        self,
        contexts: dict[str, torch.Tensor],
        *,
        due_scale_names: list[str],
        step: int,
    ):
        for name in due_scale_names:
            self.scale_states[name].context = contexts[name].detach().clone().cpu()
            self.scale_states[name].last_context_step = int(step)

    def update_scale_surprisal(
        self,
        due_scale_names: list[str],
        *,
        losses: torch.Tensor | float,
        step: int,
    ):
        for name in due_scale_names:
            observation = self.scale_estimators[name].observe(losses)
            self.scale_states[name].surprisal_last = float(observation.priority.mean().item())
            self.scale_states[name].last_surprisal_step = int(step)

    def runtime_state_dict(self) -> dict:
        return {
            "scale_states": {
                name: state.state_dict()
                for name, state in self.scale_states.items()
            },
            "scale_estimators": {
                name: estimator.state_dict()
                for name, estimator in self.scale_estimators.items()
            },
        }

    def load_runtime_state_dict(self, state: Optional[dict]):
        state = dict(state or {})
        for name, scale_state in state.get("scale_states", {}).items():
            if name in self.scale_states:
                self.scale_states[name].load_state_dict(scale_state)
        for name, estimator_state in state.get("scale_estimators", {}).items():
            if name in self.scale_estimators:
                self.scale_estimators[name].load_state_dict(estimator_state)
        return self
