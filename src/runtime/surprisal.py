from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import torch


def _as_loss_vector(losses: torch.Tensor | float) -> torch.Tensor:
    loss_t = torch.as_tensor(losses)
    if loss_t.ndim == 0:
        return loss_t.view(1)
    if loss_t.ndim == 1:
        return loss_t
    return loss_t.reshape(loss_t.size(0), -1).mean(dim=1)


@dataclass
class SurprisalObservation:
    priority: torch.Tensor
    raw_loss: torch.Tensor
    ready: bool


class BaseSurprisalEstimator(abc.ABC):
    def __init__(self):
        self._last_priority: Optional[torch.Tensor] = None

    @abc.abstractmethod
    def observe(self, losses: torch.Tensor | float) -> SurprisalObservation:
        raise NotImplementedError

    @abc.abstractmethod
    def estimate(self, losses: torch.Tensor | float) -> torch.Tensor:
        """Retorna a prioridade escalonada estritamente sem mutar o estado do estimador."""
        raise NotImplementedError

    def current(
        self,
        batch_size: int = 1,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if self._last_priority is None:
            current = torch.zeros(1)
        else:
            current = self._last_priority.detach().view(1)
        if device is not None or dtype is not None:
            current = current.to(device=device or current.device, dtype=dtype or current.dtype)
        return current.expand(batch_size)

    def state_dict(self) -> dict:
        return {
            "last_priority": None
            if self._last_priority is None
            else self._last_priority.detach().clone().cpu(),
        }

    def load_state_dict(self, state: Optional[dict]):
        state = dict(state or {})
        last_priority = state.get("last_priority")
        self._last_priority = None if last_priority is None else torch.as_tensor(last_priority)
        return self


class EMASurprisalEstimator(BaseSurprisalEstimator):
    def __init__(self, beta_mu: float = 0.95, beta_var: float = 0.95, eps: float = 1e-8):
        super().__init__()
        self.beta_mu = float(beta_mu)
        self.beta_var = float(beta_var)
        self.eps = float(eps)
        self.mu: Optional[torch.Tensor] = None
        self.var: Optional[torch.Tensor] = None

    @property
    def ready(self) -> bool:
        return self.mu is not None and self.var is not None

    def observe(self, losses: torch.Tensor | float) -> SurprisalObservation:
        raw_loss = _as_loss_vector(losses).detach()
        mean_loss = raw_loss.mean()
        if not self.ready:
            centered = raw_loss - mean_loss
            self.mu = mean_loss.detach()
            self.var = centered.pow(2).mean().detach()
            self._last_priority = torch.zeros(1, dtype=raw_loss.dtype, device=raw_loss.device)
            return SurprisalObservation(
                priority=torch.zeros_like(raw_loss),
                raw_loss=raw_loss,
                ready=False,
            )

        mu_prev = self.mu.to(device=raw_loss.device, dtype=raw_loss.dtype)
        var_prev = self.var.to(device=raw_loss.device, dtype=raw_loss.dtype)
        priority = torch.relu((raw_loss - mu_prev) / torch.sqrt(var_prev + self.eps))
        self.mu = (
            self.beta_mu * mu_prev + (1.0 - self.beta_mu) * mean_loss
        ).detach().cpu()
        mean_sq = (raw_loss - mu_prev).pow(2).mean()
        self.var = (
            self.beta_var * var_prev + (1.0 - self.beta_var) * mean_sq
        ).detach().cpu()
        self._last_priority = priority.mean().detach().view(1).cpu()
        return SurprisalObservation(priority=priority, raw_loss=raw_loss, ready=True)

    def estimate(self, losses: torch.Tensor | float) -> torch.Tensor:
        raw_loss = _as_loss_vector(losses).detach()
        if not self.ready:
            return torch.zeros_like(raw_loss)
        mu_prev = self.mu.to(device=raw_loss.device, dtype=raw_loss.dtype)
        var_prev = self.var.to(device=raw_loss.device, dtype=raw_loss.dtype)
        return torch.relu((raw_loss - mu_prev) / torch.sqrt(var_prev + self.eps))

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update(
            {
                "beta_mu": self.beta_mu,
                "beta_var": self.beta_var,
                "eps": self.eps,
                "mu": None if self.mu is None else self.mu.detach().clone().cpu(),
                "var": None if self.var is None else self.var.detach().clone().cpu(),
            }
        )
        return state

    def load_state_dict(self, state: Optional[dict]):
        super().load_state_dict(state)
        state = dict(state or {})
        self.beta_mu = float(state.get("beta_mu", self.beta_mu))
        self.beta_var = float(state.get("beta_var", self.beta_var))
        self.eps = float(state.get("eps", self.eps))
        mu = state.get("mu")
        var = state.get("var")
        self.mu = None if mu is None else torch.as_tensor(mu)
        self.var = None if var is None else torch.as_tensor(var)
        return self


class RawLossSurprisalEstimator(BaseSurprisalEstimator):
    def observe(self, losses: torch.Tensor | float) -> SurprisalObservation:
        raw_loss = _as_loss_vector(losses).detach()
        priority = raw_loss.clamp_min(0.0)
        self._last_priority = priority.mean().detach().view(1).cpu()
        return SurprisalObservation(priority=priority, raw_loss=raw_loss, ready=True)

    def estimate(self, losses: torch.Tensor | float) -> torch.Tensor:
        raw_loss = _as_loss_vector(losses).detach()
        return raw_loss.clamp_min(0.0)
