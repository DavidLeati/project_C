from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

import torch


@dataclass
class ScaleLatentState:
    context: Optional[torch.Tensor] = None
    last_context_step: int = -1
    surprisal_last: float = 0.0
    last_surprisal_step: int = -1

    def live_surprisal(self, current_step: int, decay_steps: float) -> float:
        if self.last_surprisal_step < 0:
            return 0.0
        age = max(0, int(current_step) - int(self.last_surprisal_step))
        decay = math.exp(-float(age) / max(float(decay_steps), 1e-8))
        return float(self.surprisal_last) * decay

    def age(self, current_step: int) -> int:
        if self.last_surprisal_step < 0:
            return max(0, int(current_step) + 1)
        return max(0, int(current_step) - int(self.last_surprisal_step))

    def state_dict(self) -> dict:
        return {
            "context": None if self.context is None else self.context.detach().clone().cpu(),
            "last_context_step": self.last_context_step,
            "surprisal_last": self.surprisal_last,
            "last_surprisal_step": self.last_surprisal_step,
        }

    def load_state_dict(self, state: Optional[dict]):
        state = dict(state or {})
        context = state.get("context")
        self.context = None if context is None else torch.as_tensor(context).detach().clone().cpu()
        self.last_context_step = int(state.get("last_context_step", self.last_context_step))
        self.surprisal_last = float(state.get("surprisal_last", self.surprisal_last))
        self.last_surprisal_step = int(
            state.get("last_surprisal_step", self.last_surprisal_step)
        )
        return self


class TemporalHistoryBuffer:
    def __init__(self, max_events: int):
        self.max_events = max(1, int(max_events))
        self._events: deque[torch.Tensor] = deque(maxlen=self.max_events)

    def __len__(self) -> int:
        return len(self._events)

    def append(self, x: torch.Tensor):
        if x.ndim != 1:
            raise ValueError("TemporalHistoryBuffer espera vetores de características por evento.")
        self._events.append(x.detach().clone().cpu())

    def clone(self) -> 'TemporalHistoryBuffer':
        """Cópia profunda do buffer para preservar o histórico de curto prazo entre os modos."""
        new_buffer = TemporalHistoryBuffer(max_events=self.max_events)
        new_buffer._events.extend([x.clone() for x in self._events])
        return new_buffer

    def get_sequence(
        self,
        *,
        stride: int,
        history_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if not self._events:
            raise RuntimeError("Não é possível construir uma sequência temporal a partir de um buffer de histórico vazio.")
        stride = max(1, int(stride))
        history_length = max(1, int(history_length))
        events = list(self._events)
        indices = [max(0, len(events) - 1 - offset * stride) for offset in range(history_length)]
        indices.reverse()
        sequence = torch.stack([events[index] for index in indices], dim=0)
        if device is not None or dtype is not None:
            sequence = sequence.to(device=device or sequence.device, dtype=dtype or sequence.dtype)
        return sequence.unsqueeze(0)

    def state_dict(self) -> dict:
        return {
            "max_events": self.max_events,
            "events": [event.clone() for event in self._events],
        }

    def load_state_dict(self, state: Optional[dict]):
        state = dict(state or {})
        self.max_events = max(1, int(state.get("max_events", self.max_events)))
        self._events = deque(
            (torch.as_tensor(event).detach().clone().cpu() for event in state.get("events", [])),
            maxlen=self.max_events,
        )
        return self


class LogicalClockScheduler:
    def is_due(self, step: int, stride: int) -> bool:
        stride = max(1, int(stride))
        return int(step) % stride == 0

    def due_scales(self, step: int, scales: Iterable) -> list:
        return [scale for scale in scales if self.is_due(step, getattr(scale, "stride"))]
