from __future__ import annotations

import abc
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SurprisalBufferConfig:
    capacity: int
    lag_k: int = 0
    threshold: Optional["BaseSurprisalThreshold"] = None
    replay_ratio: float = 1.0
    replay_samples: Optional[int] = None
    importance_weighting: bool = False
    sample_with_replacement: bool = True


@dataclass
class BufferRecord:
    x: torch.Tensor
    y: torch.Tensor
    priority_surprisal: float
    raw_loss: float
    event_id: int


@dataclass
class LaggedEvent:
    x: torch.Tensor
    y: torch.Tensor
    event_id: int


class BaseSurprisalThreshold(abc.ABC):
    @abc.abstractmethod
    def current_value(self) -> float:
        raise NotImplementedError

    def accept(self, priority_surprisal: float) -> bool:
        return float(priority_surprisal) > float(self.current_value())

    def observe(self, priority_surprisal: float):
        return None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: Optional[dict]):
        return self


class FixedSurprisalThreshold(BaseSurprisalThreshold):
    def __init__(self, tau: float):
        self.tau = float(tau)

    def current_value(self) -> float:
        return self.tau

    def state_dict(self) -> dict:
        return {"tau": self.tau}

    def load_state_dict(self, state: Optional[dict]):
        state = dict(state or {})
        self.tau = float(state.get("tau", self.tau))
        return self


class AdaptiveQuantileThreshold(BaseSurprisalThreshold):
    def __init__(self, window_size: int = 128, quantile: float = 0.9):
        self.window_size = max(1, int(window_size))
        self.quantile = float(quantile)
        self._history = deque(maxlen=self.window_size)

    def current_value(self) -> float:
        if not self._history:
            return math.inf
        history = torch.tensor(list(self._history), dtype=torch.float32)
        return float(torch.quantile(history, self.quantile).item())

    def observe(self, priority_surprisal: float):
        self._history.append(float(priority_surprisal))

    def state_dict(self) -> dict:
        return {
            "window_size": self.window_size,
            "quantile": self.quantile,
            "history": list(self._history),
        }

    def load_state_dict(self, state: Optional[dict]):
        state = dict(state or {})
        self.window_size = max(1, int(state.get("window_size", self.window_size)))
        self.quantile = float(state.get("quantile", self.quantile))
        self._history = deque(
            (float(v) for v in state.get("history", [])),
            maxlen=self.window_size,
        )
        return self


class PrioritizedSurprisalBuffer:
    def __init__(
        self,
        capacity: int,
        *,
        threshold: Optional[BaseSurprisalThreshold] = None,
        sample_with_replacement: bool = True,
    ):
        self.capacity = max(1, int(capacity))
        self.threshold = threshold
        self.sample_with_replacement = bool(sample_with_replacement)
        self._records: list[BufferRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> tuple[BufferRecord, ...]:
        return tuple(self._records)

    def insert(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        priority_surprisal: float,
        raw_loss: float,
        event_id: int,
    ) -> bool:
        priority_value = float(priority_surprisal)
        accepted = True
        if self.threshold is not None:
            accepted = self.threshold.accept(priority_value)
            self.threshold.observe(priority_value)
        if not accepted:
            return False

        # Normaliza para o formato [F] / [out] — remove a dimensão de lote inicial se
        # o chamador passou um tensor de amostra única de formato [1, F].
        x_store = x.detach().clone().cpu()
        y_store = y.detach().clone().cpu()
        if x_store.ndim > 1 and x_store.size(0) == 1:
            x_store = x_store.squeeze(0)
        if y_store.ndim > 1 and y_store.size(0) == 1:
            y_store = y_store.squeeze(0)
        record = BufferRecord(
            x=x_store,
            y=y_store,
            priority_surprisal=priority_value,
            raw_loss=float(raw_loss),
            event_id=int(event_id),
        )
        if len(self._records) < self.capacity:
            self._records.append(record)
            return True

        min_index = min(
            range(len(self._records)),
            key=lambda index: self._records[index].priority_surprisal,
        )
        if record.priority_surprisal <= self._records[min_index].priority_surprisal:
            return False
        self._records[min_index] = record
        return True

    def sample(
        self,
        num_samples: int,
        *,
        importance_weighting: bool = False,
        device: Optional[torch.device] = None,
    ) -> Optional[dict[str, torch.Tensor]]:
        if not self._records or num_samples <= 0:
            return None
        num_samples = min(int(num_samples), len(self._records)) if not self.sample_with_replacement else int(num_samples)
        if num_samples <= 0:
            return None
        probabilities = torch.full((len(self._records),), 1.0 / len(self._records))
        sample_indices = torch.multinomial(
            probabilities,
            num_samples=num_samples,
            replacement=self.sample_with_replacement,
        )
        chosen = [self._records[int(index)] for index in sample_indices.tolist()]
        x = torch.stack([record.x for record in chosen], dim=0)
        y = torch.stack([record.y for record in chosen], dim=0)
        priority = torch.tensor([record.priority_surprisal for record in chosen], dtype=x.dtype)
        raw_loss = torch.tensor([record.raw_loss for record in chosen], dtype=x.dtype)
        event_ids = torch.tensor([record.event_id for record in chosen], dtype=torch.long)
        if importance_weighting:
            sample_weight = priority / priority.sum().clamp(min=1e-8)
        else:
            sample_weight = torch.full((num_samples,), 1.0 / num_samples, dtype=x.dtype)
        if device is not None:
            x = x.to(device=device)
            y = y.to(device=device)
            priority = priority.to(device=device)
            raw_loss = raw_loss.to(device=device)
            event_ids = event_ids.to(device=device)
            sample_weight = sample_weight.to(device=device)
        return {
            "x": x,
            "y": y,
            "priority_surprisal": priority,
            "raw_loss": raw_loss,
            "event_id": event_ids,
            "sample_weight": sample_weight,
        }

    def update_priorities(
        self,
        event_ids: list[int],
        new_priorities: list[float],
    ) -> int:
        """Reavalia as prioridades armazenadas usando a perda (loss) **atual** do modelo.

        Após o modelo ser atualizado em um conjunto de amostras de replay, sua
        prioridade de inserção pode não refletir mais sua relevância 
        para o modelo atual. Chamar este método substitui a prioridade obsoleta 
        pelo valor da perda atual.

        Isso evita o "envenenamento de prioridade": amostras com alta perda inicial
        manteriam uma prioridade permanentemente alta que as manteria no buffer 
        mesmo após o modelo tê-las aprendido. A reavaliação permite que o 
        mecanismo de expulsão desloque amostras já aprendidas em favor de 
        novas amostras genuinamente surpreendentes.

        Parâmetros
        ----------
        event_ids:
            Lista de valores ``event_id`` retornados por uma chamada anterior a ``sample()``
            (``batch["event_id"].tolist()``).
        new_priorities:
            Perdas do modelo atual por amostra correspondentes (escalares), normalmente
            obtidas de ``per_sample_loss[B:B+R].detach()`` onde B é o tamanho do lote vivo
            e R é o número de amostras de replay.

        Retorna
        -------
        int
            Número de registros cuja prioridade foi atualizada com sucesso. Um
            registro pode não ser encontrado se tiver sido expulso entre a chamada de ``sample()``
            e esta chamada (raro, mas possível).
        """
        if not event_ids:
            return 0
        # Constrói uma busca rápida: event_id → índice da lista.
        # (O(N) uma vez por chamada) para evitar manter um dicionário separado que
        # precisaria ficar em sincronia com cada operação de inserção/expulsão.
        id_to_index: dict[int, int] = {
            record.event_id: idx for idx, record in enumerate(self._records)
        }
        updated = 0
        for eid, new_prio in zip(event_ids, new_priorities):
            idx = id_to_index.get(int(eid))
            if idx is not None:
                self._records[idx].priority_surprisal = max(0.0, float(new_prio))
                updated += 1
        return updated

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "sample_with_replacement": self.sample_with_replacement,
            "threshold": None if self.threshold is None else self.threshold.state_dict(),
            "threshold_class": None if self.threshold is None else self.threshold.__class__.__name__,
            "records": [
                {
                    "x": record.x.clone(),
                    "y": record.y.clone(),
                    "priority_surprisal": record.priority_surprisal,
                    "raw_loss": record.raw_loss,
                    "event_id": record.event_id,
                }
                for record in self._records
            ],
        }

    def load_state_dict(self, state: Optional[dict]):
        state = dict(state or {})
        self.capacity = max(1, int(state.get("capacity", self.capacity)))
        self.sample_with_replacement = bool(
            state.get("sample_with_replacement", self.sample_with_replacement)
        )
        self._records = [
            BufferRecord(
                x=torch.as_tensor(entry["x"]).detach().clone().cpu(),
                y=torch.as_tensor(entry["y"]).detach().clone().cpu(),
                priority_surprisal=float(entry["priority_surprisal"]),
                raw_loss=float(entry["raw_loss"]),
                event_id=int(entry["event_id"]),
            )
            for entry in state.get("records", [])
        ]
        if self.threshold is not None:
            self.threshold.load_state_dict(state.get("threshold"))
        return self


class LagEventQueue:
    def __init__(self, lag_k: int = 0):
        self.lag_k = max(0, int(lag_k))
        self._queue: deque[LaggedEvent] = deque()

    def push(self, x: torch.Tensor, y: torch.Tensor, event_id: int):
        self._queue.append(
            LaggedEvent(
                x=x.detach().clone().cpu(),
                y=y.detach().clone().cpu(),
                event_id=int(event_id),
            )
        )

    def pop_ready(self, current_event_id: int) -> list[LaggedEvent]:
        ready: list[LaggedEvent] = []
        threshold_event_id = int(current_event_id) - self.lag_k
        while self._queue and self._queue[0].event_id < threshold_event_id:
            self._queue.popleft()
        while self._queue and self._queue[0].event_id == threshold_event_id:
            ready.append(self._queue.popleft())
        return ready

    def state_dict(self) -> dict:
        return {
            "lag_k": self.lag_k,
            "queue": [
                {
                    "x": event.x.clone(),
                    "y": event.y.clone(),
                    "event_id": event.event_id,
                }
                for event in self._queue
            ],
        }

    def load_state_dict(self, state: Optional[dict]):
        state = dict(state or {})
        self.lag_k = max(0, int(state.get("lag_k", self.lag_k)))
        self._queue = deque(
            LaggedEvent(
                x=torch.as_tensor(entry["x"]).detach().clone().cpu(),
                y=torch.as_tensor(entry["y"]).detach().clone().cpu(),
                event_id=int(entry["event_id"]),
            )
            for entry in state.get("queue", [])
        )
        return self
