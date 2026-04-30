from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from ..models.hsama import HSAMAPolicySnapshot, HSAMA
from .multiscale import MultiScaleT0Builder, MultiScaleT0Config
from .replay import LagEventQueue, PrioritizedSurprisalBuffer, SurprisalBufferConfig
from .surprisal import BaseSurprisalEstimator, EMASurprisalEstimator
from .temporal import TemporalHistoryBuffer


@dataclass
class HSAMARuntimeConfig:
    replay_config: Optional[SurprisalBufferConfig] = None
    multi_scale_t0_config: Optional[MultiScaleT0Config] = None
    optimizer_t1_kwargs: dict = field(default_factory=lambda: {"lr": 1e-3})
    optimizer_t0_context_kwargs: dict = field(default_factory=lambda: {"lr": 1e-3})
    optimizer_t0_model_kwargs: dict = field(default_factory=lambda: {"lr": 1e-3})
    optimizer_t0_builder_shared_kwargs: dict = field(default_factory=lambda: {"lr": 1e-3})
    optimizer_t0_scale_kwargs: dict = field(default_factory=lambda: {"lr": 1e-3})
    use_exploration: Optional[bool] = None
    observe_mode: str = "online"
    """Controla como as entradas em lote são processadas durante ``observe()``.

    ``"online"``  – (padrão) cada amostra no lote é processada
        individualmente em ordem cronológica. Um passo de gradiente é realizado
        *por amostra*, o que maximiza a granularidade temporal, mas produz
        gradientes de alta variância para lotes grandes.

    ``"batch"``   – o lote inteiro é tratado como um único passo lógico.
        Uma passagem para frente (forward) e uma atualização de gradiente cobrem todas as amostras vivas
        (mais quaisquer amostras de replay). O contexto é computado por amostra para o
        codificador simples e compartilhado (a partir do último estado do histórico) para o
        construtor multi-escala. Este modo corresponde à frequência de atualização de um
        baseline SGD convencional de mini-lote.
    """
    raw_output: bool = True
    """Ativação de saída aplicada consistentemente em **todos** os caminhos de tempo de execução
    (treinamento, replay e inferência via ``predict()``).

    Quando *True* (padrão), a projeção linear final é retornada como está,
    o que é apropriado para problemas de regressão cujos alvos podem ser
    negativos ou ilimitados.

    Quando *False*, ``F.softplus`` é aplicado, restringindo as previsões a
    valores estritamente positivos.

    .. important::
        Este valor deve corresponder ao que for usado nas chamadas de ``model.forward()`` /
        ``execute_policy()`` feitas **fora** do tempo de execução (runtime) (ex: em
        loops de avaliação personalizados). Misturar ``raw_output=True`` no momento do treinamento
        com ``raw_output=False`` no momento da avaliação — ou vice-versa
        — causa uma incompatibilidade sistemática no espaço de saída que impede a
        convergência. Se você chamar ``runtime.predict()`` para avaliação,
        isso é tratado automaticamente.
    """


@dataclass
class RuntimeStepResult:
    event_id: int
    live_prediction: torch.Tensor
    live_projection: torch.Tensor
    total_loss: torch.Tensor
    live_loss: torch.Tensor
    priority_surprisal: torch.Tensor
    replay_size: int
    buffer_size: int
    buffer_inserted: bool
    due_scale_names: tuple[str, ...]
    sample_weight: torch.Tensor
    scale_metadata: dict[str, torch.Tensor]


class HSAMAOnlineRuntime:
    def __init__(
        self,
        model: HSAMA,
        *,
        config: Optional[HSAMARuntimeConfig] = None,
        loss_fn: Optional[nn.Module] = None,
        surprisal_estimator: Optional[BaseSurprisalEstimator] = None,
        optimizer_cls=torch.optim.AdamW,
    ):
        self.model = model
        self.config = config or HSAMARuntimeConfig()
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss(reduction="none")
        self.global_surprisal_estimator = (
            surprisal_estimator if surprisal_estimator is not None else EMASurprisalEstimator()
        )
        self.global_step = 0
        self._replay_ratio_carry = 0.0

        self.replay_buffer = None
        self.lag_queue = None
        if self.config.replay_config is not None:
            replay_config = self.config.replay_config
            self.replay_buffer = PrioritizedSurprisalBuffer(
                capacity=replay_config.capacity,
                threshold=replay_config.threshold,
                sample_with_replacement=replay_config.sample_with_replacement,
            )
            self.lag_queue = LagEventQueue(lag_k=replay_config.lag_k)

        self.multi_scale_builder = None
        self.history_buffer = None
        if self.config.multi_scale_t0_config is not None:
            self.multi_scale_builder = MultiScaleT0Builder(
                in_features=self.model.in_features,
                context_dim=self.model.context_dim,
                config=self.config.multi_scale_t0_config,
            )
            self.history_buffer = TemporalHistoryBuffer(
                max_events=self.multi_scale_builder.required_history()
            )

        self.optimizer_t1 = optimizer_cls(
            self.model.get_t1_parameters(),
            **self.config.optimizer_t1_kwargs,
        )
        self.optimizer_t0_context = None
        if self.multi_scale_builder is None:
            self.optimizer_t0_context = optimizer_cls(
                self.model.get_t0_context_parameters(),
                **self.config.optimizer_t0_context_kwargs,
            )
        self.optimizer_t0_model = optimizer_cls(
            self.model.get_t0_policy_parameters(),
            **self.config.optimizer_t0_model_kwargs,
        )
        self.optimizer_t0_builder_shared = None
        self.optimizer_t0_scale: dict[str, torch.optim.Optimizer] = {}
        if self.multi_scale_builder is not None:
            self.optimizer_t0_builder_shared = optimizer_cls(
                self.multi_scale_builder.shared_parameters(),
                **self.config.optimizer_t0_builder_shared_kwargs,
            )
            for name in self.multi_scale_builder.scale_order:
                self.optimizer_t0_scale[name] = optimizer_cls(
                    self.multi_scale_builder.scale_parameters(name),
                    **self.config.optimizer_t0_scale_kwargs,
                )

    def _zero_all_gradients(self):
        self.optimizer_t1.zero_grad(set_to_none=True)
        if self.optimizer_t0_context is not None:
            self.optimizer_t0_context.zero_grad(set_to_none=True)
        self.optimizer_t0_model.zero_grad(set_to_none=True)
        if self.optimizer_t0_builder_shared is not None:
            self.optimizer_t0_builder_shared.zero_grad(set_to_none=True)
        for optimizer in self.optimizer_t0_scale.values():
            optimizer.zero_grad(set_to_none=True)

    def _normalize_batch(
        self,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        if y_t.ndim == 0:
            y_t = y_t.view(1, 1)
        elif y_t.ndim == 1:
            if x_t.size(0) == 1:
                y_t = y_t.unsqueeze(0)
            else:
                y_t = y_t.unsqueeze(-1)
        if x_t.size(0) != y_t.size(0):
            raise ValueError("x_t e y_t devem compartilhar a mesma dimensão de lote.")
        return x_t, y_t

    def _per_sample_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(predictions, targets)
        if not torch.is_tensor(loss):
            raise TypeError("loss_fn deve retornar um tensor.")
        if loss.ndim == 0:
            return loss.view(1).expand(predictions.size(0))
        if loss.ndim == 1:
            return loss
        return loss.reshape(loss.size(0), -1).mean(dim=1)

    def _expand_policy(self, policy: HSAMAPolicySnapshot, batch_size: int) -> HSAMAPolicySnapshot:
        if policy.context.size(0) == batch_size:
            return policy
        return HSAMAPolicySnapshot(
            edge_dnas=policy.edge_dnas.expand(batch_size, -1, -1),
            context=policy.context.expand(batch_size, -1),
            surprisal=policy.surprisal.expand(batch_size, -1, -1),
            temperature=policy.temperature.expand(batch_size, -1, -1),
            scale=policy.scale.expand(batch_size, -1, -1),
            use_exploration=policy.use_exploration,
            exploration_noise=policy.exploration_noise.expand(batch_size, -1, -1),
        )

    def _replay_sample_count(self, live_batch_size: int) -> int:
        if self.replay_buffer is None or self.config.replay_config is None:
            return 0
        if len(self.replay_buffer) == 0:
            return 0
        replay_samples = self.config.replay_config.replay_samples
        if replay_samples is not None:
            return max(0, int(replay_samples))
        replay_ratio = max(0.0, float(self.config.replay_config.replay_ratio))
        if replay_ratio <= 0.0:
            return 0
        target_replay = (replay_ratio * max(1, int(live_batch_size))) + self._replay_ratio_carry
        count = int(math.floor(target_replay))
        self._replay_ratio_carry = target_replay - count
        return count

    def _build_training_batch(
        self,
        x_live: torch.Tensor,
        y_live: torch.Tensor,
        policy: HSAMAPolicySnapshot,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, HSAMAPolicySnapshot, int]:
        replay_size = self._replay_sample_count(x_live.size(0))
        replay_batch = None
        if self.replay_buffer is not None and replay_size > 0:
            replay_batch = self.replay_buffer.sample(
                replay_size,
                importance_weighting=self.config.replay_config.importance_weighting,
                device=x_live.device,
            )

        if replay_batch is None:
            sample_weight = torch.full(
                (x_live.size(0),),
                1.0 / x_live.size(0),
                device=x_live.device,
                dtype=x_live.dtype,
            )
            # Garanta sempre que a dimensão do lote da política corresponda a x_live (necessário para
            # o modo batch, onde a política é construída com batch_size=1, mas x_live
            # pode ter B > 1 amostras).
            expanded_policy = self._expand_policy(policy, x_live.size(0))
            return x_live, y_live, sample_weight, expanded_policy, 0, None

        train_x = torch.cat([x_live, replay_batch["x"].to(dtype=x_live.dtype)], dim=0)
        train_y = torch.cat([y_live, replay_batch["y"].to(dtype=y_live.dtype)], dim=0)
        if self.config.replay_config.importance_weighting:
            sample_weight = torch.cat(
                [
                    torch.ones(
                        x_live.size(0),
                        device=x_live.device,
                        dtype=x_live.dtype,
                    ),
                    replay_batch["sample_weight"].to(device=x_live.device, dtype=x_live.dtype),
                ],
                dim=0,
            )
            sample_weight = sample_weight / sample_weight.sum().clamp(min=1e-8)
        else:
            sample_weight = torch.full(
                (train_x.size(0),),
                1.0 / train_x.size(0),
                device=x_live.device,
                dtype=x_live.dtype,
            )
        expanded_policy = self._expand_policy(policy, train_x.size(0))
        return train_x, train_y, sample_weight, expanded_policy, int(replay_batch["x"].size(0)), replay_batch

    def _prepare_context(
        self,
        x_live: torch.Tensor,
        *,
        step: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[str], dict[str, torch.Tensor]]:
        if self.multi_scale_builder is None:
            return self.model.encode_context(x_live), {}, [], {}

        if self.history_buffer is None:
            raise RuntimeError("history_buffer deve existir quando multi_scale_builder está habilitado.")

        self.history_buffer.append(x_live.squeeze(0))
        contexts, due_scale_names = self.multi_scale_builder.prepare_contexts(
            self.history_buffer,
            step=step,
            device=x_live.device,
            dtype=x_live.dtype,
        )
        context, scale_metadata = self.multi_scale_builder.compose_context(
            contexts,
            step=step,
            device=x_live.device,
            dtype=x_live.dtype,
        )
        return context, contexts, due_scale_names, scale_metadata

    def _prepare_context_batch(
        self,
        x_batch: torch.Tensor,
        *,
        step: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[str], dict[str, torch.Tensor]]:
        """Prepara o contexto para um lote inteiro em um único passo lógico.

        Codificador simples: cada amostra recebe seu próprio vetor de contexto
        (política por amostra), idêntico em qualidade ao modo online.

        Construtor multi-escala: o buffer de histórico é atualizado sequencialmente
        com todas as amostras do lote, então um *único* contexto compartilhado é construído
        a partir do estado de histórico resultante e expandido por todo o lote.
        Esta é uma aproximação — todas as amostras recebem a mesma política T0
        — mas evita passos para trás (backward passes) O(B) e é semanticamente coerente
        porque o contexto T0 resume o *histórico visto até agora*.
        """
        if self.multi_scale_builder is None:
            # Cada amostra recebe um contexto independente — fidelidade total.
            context = self.model.encode_context(x_batch)  # [B, context_dim]
            return context, {}, [], {}

        if self.history_buffer is None:
            raise RuntimeError("history_buffer deve existir quando multi_scale_builder está habilitado.")

        batch_size = x_batch.size(0)
        for i in range(batch_size):
            self.history_buffer.append(x_batch[i])

        contexts, due_scale_names = self.multi_scale_builder.prepare_contexts(
            self.history_buffer,
            step=step,
            device=x_batch.device,
            dtype=x_batch.dtype,
        )
        context, scale_metadata = self.multi_scale_builder.compose_context(
            contexts,
            step=step,
            device=x_batch.device,
            dtype=x_batch.dtype,
        )
        # contexto: [1, context_dim] → replicado por todo o lote
        context = context.expand(batch_size, -1)
        return context, contexts, due_scale_names, scale_metadata

    def _insert_ready_events(
        self,
        *,
        x_live: torch.Tensor,
        y_live: torch.Tensor,
        step: int,
        priority_surprisal: torch.Tensor,
        raw_loss: torch.Tensor,
        ready: bool,
    ) -> bool:
        if self.lag_queue is None or self.replay_buffer is None:
            return False

        self.lag_queue.push(x_live.squeeze(0), y_live.squeeze(0), step)
        if not ready:
            return False

        inserted = False
        ready_events = self.lag_queue.pop_ready(step)
        priority_value = float(priority_surprisal.mean().item())
        raw_loss_value = float(raw_loss.mean().item())
        for event in ready_events:
            inserted = (
                self.replay_buffer.insert(
                    x=event.x,
                    y=event.y,
                    priority_surprisal=priority_value,
                    raw_loss=raw_loss_value,
                    event_id=event.event_id,
                )
                or inserted
            )
        return inserted

    def _observe_single(self, x_live: torch.Tensor, y_live: torch.Tensor) -> RuntimeStepResult:
        step = self.global_step
        context, contexts, due_scale_names, scale_metadata = self._prepare_context(
            x_live,
            step=step,
        )
        global_surprisal = self.global_surprisal_estimator.current(
            batch_size=1,
            device=x_live.device,
            dtype=x_live.dtype,
        )
        policy = self.model.build_policy_from_context(
            context,
            surprisal=global_surprisal,
            use_exploration=self.config.use_exploration,
        )
        train_x, train_y, sample_weight, train_policy, replay_size, replay_batch = self._build_training_batch(
            x_live,
            y_live,
            policy,
        )

        self._zero_all_gradients()
        batch_prediction, batch_projection = self.model.execute_policy(
            train_x,
            train_policy,
            raw_output=self.config.raw_output,
        )
        per_sample_loss = self._per_sample_loss(batch_prediction, train_y)
        total_loss = (per_sample_loss * sample_weight).sum()
        total_loss.backward()

        # Re-prioriza amostras de replay com a perda atual do modelo (evita envenenamento de prioridade).
        if replay_size > 0 and self.replay_buffer is not None and replay_batch is not None:
            replay_current_loss = per_sample_loss[1 : 1 + replay_size].detach()
            new_priorities = self.global_surprisal_estimator.estimate(replay_current_loss).tolist()
            self.replay_buffer.update_priorities(
                replay_batch["event_id"].tolist(),
                new_priorities,
            )

        self.optimizer_t1.step()
        if self.optimizer_t0_context is not None:
            self.optimizer_t0_context.step()
        self.optimizer_t0_model.step()
        if self.optimizer_t0_builder_shared is not None:
            self.optimizer_t0_builder_shared.step()
        for name in due_scale_names:
            self.optimizer_t0_scale[name].step()

        if self.multi_scale_builder is not None and due_scale_names:
            self.multi_scale_builder.commit_contexts(
                contexts,
                due_scale_names=due_scale_names,
                step=step,
            )

        live_loss = per_sample_loss[:1].detach()
        global_observation = self.global_surprisal_estimator.observe(live_loss)
        if self.multi_scale_builder is not None and due_scale_names:
            self.multi_scale_builder.update_scale_surprisal(
                due_scale_names,
                losses=live_loss,
                step=step,
            )

        buffer_inserted = self._insert_ready_events(
            x_live=x_live,
            y_live=y_live,
            step=step,
            priority_surprisal=global_observation.priority,
            raw_loss=global_observation.raw_loss,
            ready=global_observation.ready,
        )

        self.global_step += 1
        return RuntimeStepResult(
            event_id=step,
            live_prediction=batch_prediction[:1].detach(),
            live_projection=batch_projection[:1].detach(),
            total_loss=total_loss.detach(),
            live_loss=live_loss.detach(),
            priority_surprisal=global_observation.priority.detach(),
            replay_size=replay_size,
            buffer_size=0 if self.replay_buffer is None else len(self.replay_buffer),
            buffer_inserted=buffer_inserted,
            due_scale_names=tuple(due_scale_names),
            sample_weight=sample_weight.detach(),
            scale_metadata={
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in scale_metadata.items()
            },
        )

    def _observe_batch(self, x_live: torch.Tensor, y_live: torch.Tensor) -> RuntimeStepResult:
        """Passo de gradiente único cobrindo todo o lote vivo mais quaisquer amostras de replay.

        Este é o caminho ``observe_mode="batch"``. Ele corresponde à frequência de
        atualização esperada por um baseline SGD de mini-lote: um passo do otimizador
        para cada chamada a ``observe()``, independentemente do tamanho do lote.

        ``live_prediction`` / ``live_projection`` no resultado retornado
        têm o formato ``[B, out_features]`` (todas as amostras vivas, sem linhas de replay).
        """
        batch_size = x_live.size(0)
        step = self.global_step

        # --- Contexto e política T0 ---
        context, contexts, due_scale_names, scale_metadata = self._prepare_context_batch(
            x_live, step=step
        )
        # Colapsa os contextos por amostra em um único contexto representativo
        # para que a política resultante (edge_dnas formato [1, E, D]) possa ser
        # expandida por _expand_policy para cobrir o lote completo (vivo + replay).
        # Para o codificador simples, esta é a média das codificações por amostra;
        # para o construtor multi-escala, o contexto já é [1, D] (compartilhado).
        context_for_policy = context.mean(dim=0, keepdim=True)  # [1, context_dim]
        global_surprisal = self.global_surprisal_estimator.current(
            batch_size=1,
            device=x_live.device,
            dtype=x_live.dtype,
        )
        policy = self.model.build_policy_from_context(
            context_for_policy,
            surprisal=global_surprisal,
            use_exploration=self.config.use_exploration,
        )

        # --- Combina o lote vivo com amostras de replay ---
        train_x, train_y, sample_weight, train_policy, replay_size, replay_batch = self._build_training_batch(
            x_live, y_live, policy
        )

        # --- Passo único de forward + backward + otimizador ---
        self._zero_all_gradients()
        batch_prediction, batch_projection = self.model.execute_policy(
            train_x, train_policy, raw_output=self.config.raw_output
        )
        per_sample_loss = self._per_sample_loss(batch_prediction, train_y)
        total_loss = (per_sample_loss * sample_weight).sum()
        total_loss.backward()

        # --- Re-prioriza amostras de replay usando a perda atual do modelo ---
        # A prioridade de inserção pode não refletir mais a relevância da amostra
        # para o modelo atual. Atualizá-la garante que amostras já dominadas recebam
        # prioridade inferior, mantendo o buffer focado em amostras informativas.
        if replay_size > 0 and self.replay_buffer is not None and replay_batch is not None:
            replay_event_ids = replay_batch["event_id"].tolist()
            replay_current_loss = per_sample_loss[batch_size : batch_size + replay_size].detach()
            new_priorities = self.global_surprisal_estimator.estimate(replay_current_loss).tolist()
            self.replay_buffer.update_priorities(replay_event_ids, new_priorities)

        self.optimizer_t1.step()
        if self.optimizer_t0_context is not None:
            self.optimizer_t0_context.step()
        self.optimizer_t0_model.step()
        if self.optimizer_t0_builder_shared is not None:
            self.optimizer_t0_builder_shared.step()
        for name in due_scale_names:
            self.optimizer_t0_scale[name].step()

        if self.multi_scale_builder is not None and due_scale_names:
            self.multi_scale_builder.commit_contexts(
                contexts, due_scale_names=due_scale_names, step=step
            )

        # --- Atualização de surprisal (apenas amostras vivas, sem linhas de replay) ---
        live_loss = per_sample_loss[:batch_size].detach()
        global_observation = self.global_surprisal_estimator.observe(live_loss)

        if self.multi_scale_builder is not None and due_scale_names:
            self.multi_scale_builder.update_scale_surprisal(
                due_scale_names, losses=live_loss, step=step
            )

        buffer_inserted = False
        if self.replay_buffer is not None and global_observation.ready:
            priority_value = float(global_observation.priority.mean().item())
            raw_loss_value = float(global_observation.raw_loss.mean().item())
            for i in range(batch_size):
                inserted = self.replay_buffer.insert(
                    x=x_live[i : i + 1],
                    y=y_live[i : i + 1],
                    priority_surprisal=priority_value,
                    raw_loss=raw_loss_value,
                    event_id=step * batch_size + i,
                )
                buffer_inserted = buffer_inserted or inserted

        self.global_step += 1
        return RuntimeStepResult(
            event_id=step,
            live_prediction=batch_prediction[:batch_size].detach(),
            live_projection=batch_projection[:batch_size].detach(),
            total_loss=total_loss.detach(),
            live_loss=live_loss,
            priority_surprisal=global_observation.priority.detach(),
            replay_size=replay_size,
            buffer_size=0 if self.replay_buffer is None else len(self.replay_buffer),
            buffer_inserted=buffer_inserted,
            due_scale_names=tuple(due_scale_names),
            sample_weight=sample_weight.detach(),
            scale_metadata={
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in scale_metadata.items()
            },
        )

    def observe(self, x_t: torch.Tensor, y_t: torch.Tensor):
        """Processa uma nova observação (ou mini-lote de observações).

        Despacha para ``_observe_single`` (modo online) ou
        ``_observe_batch`` (modo batch) com base em ``config.observe_mode``.

        Retorna
        -------
        RuntimeStepResult
            Sempre um único objeto de resultado. No modo online com B > 1,
            é retornada uma *lista* de resultados por amostra.
        """
        x_batch, y_batch = self._normalize_batch(x_t, y_t)
        batch_size = x_batch.size(0)

        if batch_size == 1 or self.config.observe_mode == "online":
            results = [
                self._observe_single(
                    x_batch[index : index + 1],
                    y_batch[index : index + 1],
                )
                for index in range(batch_size)
            ]
            return results[0] if len(results) == 1 else results

        return self._observe_batch(x_batch, y_batch)

    def predict(self, x_t: torch.Tensor, *, raw_output: Optional[bool] = None) -> torch.Tensor:
        """Passagem de inferência pura: sem atualizações de gradiente, sem mutação do estado de treinamento.

        Processa amostras sequencialmente através do **mesmo pipeline de contexto**
        usado durante o treinamento — os codificadores GRU multi-escala quando
        ``multi_scale_t0_config`` está ativo, ou o simples ``context_encoder``
        caso contrário. Um ``TemporalHistoryBuffer`` temporário é usado para que
        ``self.history_buffer`` e ``self.scale_states`` nunca sejam tocados.

        Este é o caminho de avaliação correto para execuções com um construtor
        Multi-Escala: chamar ``model.forward()`` diretamente usaria o
        ``context_encoder`` (uma camada linear simples que nunca recebe atualizações
        de gradiente quando o construtor Multi-Escala está ativo), produzindo vetores de
        contexto fora da distribuição e previsões quase constantes.

        Parâmetros
        ----------
        x_t:
            Tensor de entrada de formato ``(B, in_features)`` ou ``(in_features,)``.
        raw_output:
            Sobrescrita para a ativação de saída. Quando *None* (padrão), o
            valor de ``config.raw_output`` é usado, que é o comportamento
            recomendado. Passar um valor explícito que difira de
            ``config.raw_output`` emite um ``UserWarning`` porque
            cria uma incompatibilidade entre treinamento/previsão.

        Retorna
        -------
        torch.Tensor
            Previsões de formato ``(B, out_features)``.
        """
        import warnings
        if raw_output is None:
            raw_output = self.config.raw_output
        elif raw_output != self.config.raw_output:
            warnings.warn(
                f"predict() called with raw_output={raw_output!r}, which differs from "
                f"config.raw_output={self.config.raw_output!r}.  "
                "This creates a train/inference output-space mismatch and will "
                "produce biased predictions.  Set config.raw_output to the "
                "desired value instead of overriding it per-call.",
                UserWarning,
                stacklevel=2,
            )
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        x_batch = x_t

        temp_history: Optional[TemporalHistoryBuffer] = None
        if self.multi_scale_builder is not None and self.history_buffer is not None:
            temp_history = self.history_buffer.clone()

        self.model.eval()
        predictions: list[torch.Tensor] = []

        with torch.no_grad():
            for i in range(x_batch.size(0)):
                x_single = x_batch[i : i + 1]

                if temp_history is None or self.multi_scale_builder is None:
                    # Caminho simples: context_encoder é o codificador ativo.
                    context = self.model.encode_context(x_single)
                else:
                    temp_history.append(x_single.squeeze(0))
                    # step=0 força todas as escalas a recalcular a partir do
                    # histórico temporário em cada chamada, evitando contextos
                    # comprometidos desatualizados da fase de treinamento.
                    contexts, _ = self.multi_scale_builder.prepare_contexts(
                        temp_history,
                        step=0,
                        device=x_batch.device,
                        dtype=x_batch.dtype,
                    )
                    context, _ = self.multi_scale_builder.compose_context(
                        contexts,
                        step=0,
                        device=x_batch.device,
                        dtype=x_batch.dtype,
                    )

                policy = self.model.build_policy_from_context(
                    context, surprisal=0.0, use_exploration=False
                )
                pred, _ = self.model.execute_policy(x_single, policy, raw_output=raw_output)
                predictions.append(pred)

        self.model.train()
        return torch.cat(predictions, dim=0)

    def state_dict(self) -> dict:
        return {
            "global_step": self.global_step,
            "global_surprisal_estimator": self.global_surprisal_estimator.state_dict(),
            "replay_buffer": None if self.replay_buffer is None else self.replay_buffer.state_dict(),
            "lag_queue": None if self.lag_queue is None else self.lag_queue.state_dict(),
            "multi_scale_builder": None
            if self.multi_scale_builder is None
            else self.multi_scale_builder.state_dict(),
            "multi_scale_builder_runtime": None
            if self.multi_scale_builder is None
            else self.multi_scale_builder.runtime_state_dict(),
            "history_buffer": None
            if self.history_buffer is None
            else self.history_buffer.state_dict(),
            "optimizer_t1": self.optimizer_t1.state_dict(),
            "optimizer_t0_context": None
            if self.optimizer_t0_context is None
            else self.optimizer_t0_context.state_dict(),
            "optimizer_t0_model": self.optimizer_t0_model.state_dict(),
            "optimizer_t0_builder_shared": None
            if self.optimizer_t0_builder_shared is None
            else self.optimizer_t0_builder_shared.state_dict(),
            "optimizer_t0_scale": {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizer_t0_scale.items()
            },
            "replay_ratio_carry": self._replay_ratio_carry,
        }

    def load_state_dict(self, state: dict):
        self.global_step = int(state.get("global_step", self.global_step))
        self.global_surprisal_estimator.load_state_dict(state.get("global_surprisal_estimator"))
        if self.replay_buffer is not None:
            self.replay_buffer.load_state_dict(state.get("replay_buffer"))
        if self.lag_queue is not None:
            self.lag_queue.load_state_dict(state.get("lag_queue"))
        if self.multi_scale_builder is not None:
            self.multi_scale_builder.load_state_dict(state.get("multi_scale_builder", {}))
            self.multi_scale_builder.load_runtime_state_dict(
                state.get("multi_scale_builder_runtime")
            )
        if self.history_buffer is not None:
            self.history_buffer.load_state_dict(state.get("history_buffer"))
        self.optimizer_t1.load_state_dict(state.get("optimizer_t1", {}))
        if self.optimizer_t0_context is not None:
            self.optimizer_t0_context.load_state_dict(state.get("optimizer_t0_context", {}))
        self.optimizer_t0_model.load_state_dict(state.get("optimizer_t0_model", {}))
        if self.optimizer_t0_builder_shared is not None:
            self.optimizer_t0_builder_shared.load_state_dict(
                state.get("optimizer_t0_builder_shared", {})
            )
        for name, optimizer_state in state.get("optimizer_t0_scale", {}).items():
            if name in self.optimizer_t0_scale:
                self.optimizer_t0_scale[name].load_state_dict(optimizer_state)
        self._replay_ratio_carry = float(state.get("replay_ratio_carry", self._replay_ratio_carry))
        return self
