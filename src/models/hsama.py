from dataclasses import dataclass
import math
from dataclasses import dataclass
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import InputDelegator, KRegularGraph
# importações kan removidas para ablação


class RMSNorm(nn.Module):
    """Normalização de Camada Root Mean Square (RMS).

    Ao contrário do LayerNorm, o RMSNorm não centraliza as ativações (sem subtração
    da média). Isso preserva a informação de magnitude enquanto ainda
    normaliza a variância — importante para a passagem de mensagens em grafos onde
    a magnitude da mensagem agregada carrega sinal.
    """

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


@dataclass
class HSAMAPolicySnapshot:
    edge_dnas: torch.Tensor
    context: torch.Tensor
    surprisal: torch.Tensor
    temperature: torch.Tensor
    scale: torch.Tensor
    use_exploration: bool
    exploration_noise: torch.Tensor

    def with_edge_dnas(self, edge_dnas):
        return HSAMAPolicySnapshot(
            edge_dnas=edge_dnas,
            context=self.context,
            surprisal=self.surprisal,
            temperature=self.temperature,
            scale=self.scale,
            use_exploration=self.use_exploration,
            exploration_noise=self.exploration_noise,
        )


class HyperNetwork(nn.Module):
    """
    Módulo HyperNetwork para meta-configuração top-down.

    Gera formulações estruturais dinâmicas (vetores DNA) para modular
    configurações de arestas dentro do grafo de execução com base no contexto ambiental.
    """

    def __init__(self, context_dim, target_dna_dim, hidden_dim=None):
        super().__init__()
        hidden = hidden_dim if hidden_dim is not None else max(64, target_dna_dim // 16)
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, target_dna_dim),
        )

    def forward(self, context):
        return self.net(context)


class HSAMA(nn.Module):
    """
    Meta-Arquitetura Algébrica Simbólica Hierárquica (H-SAMA).

    Topologia arquitetural apresentando um ciclo de execução de duas fases:
    - Fase T0: Meta-configuração construindo deltas de DNA por aresta via hiper-redes contextuais.
    - Fase T1: Execução matemática base utilizando parâmetros vivos modulados dinamicamente pelo DNA T0.
    """

    def __init__(
        self,
        in_features,
        out_features=1,
        num_nodes=50,
        k=6,
        state_dim=1,
        spline_order=3,
        grid_size=5,
        projection_dim=16,
        num_sectors=4,
        max_hops=2,
        graph_seed=0,
        context_dim=8,
        num_entry_nodes=4,
        dropout_p=0.1,
        explore_noise_std=0.05,
        dna_base_scale=0.9,
        dna_temp_scale=0.2,
        heaviside_sharpness=50.0,
        hyper_hidden_dim=None,
        ensure_connected=True,
        max_rewire_attempts=16,
        hop_scale_init=1.0,
        rewire_prob=0.1,
        pade_eps=1e-4,
        output_scale_init=None,
        learnable_output_scale=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.state_dim = state_dim
        self.max_hops = max_hops
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.graph_seed = graph_seed
        self.context_dim = context_dim
        self.explore_noise_std = explore_noise_std
        self.dna_base_scale = dna_base_scale
        self.dna_temp_scale = dna_temp_scale
        self.heaviside_sharpness = heaviside_sharpness
        self.hyper_hidden_dim = hyper_hidden_dim
        self.ensure_connected = bool(ensure_connected)
        self.max_rewire_attempts = max(1, int(max_rewire_attempts))

        self.hop_scale = nn.Parameter(torch.tensor(float(hop_scale_init)))

        self._set_graph(
            KRegularGraph(
                num_nodes=num_nodes,
                k=k,
                rewire_prob=rewire_prob,
                seed=graph_seed,
                num_entry_nodes=num_entry_nodes,
                ensure_connected=self.ensure_connected,
                max_rewire_attempts=self.max_rewire_attempts,
            )
        )
        self.delegator = InputDelegator(
            in_features,
            num_nodes=num_nodes,
            num_entry_nodes=num_entry_nodes,
            state_dim=state_dim,
            semantic_groups=self.graph.semantic_groups,
            context_features=self.context_dim,
        )
        num_edges = self.graph.num_edges

        bound = 1.0 / math.sqrt(state_dim)
        self.mlp_w1 = nn.Parameter(
            torch.empty(num_edges, state_dim, state_dim).uniform_(-bound, bound)
        )
        self.mlp_w2 = nn.Parameter(
            torch.empty(num_edges, state_dim, state_dim).uniform_(-bound, bound)
        )

        self.dna_dim = 2 * state_dim
        self.dna_scale = nn.Parameter(torch.tensor(1.0))

        self.num_sectors = num_sectors
        self.edges_per_sector = math.ceil(self.graph.num_edges / self.num_sectors)

        self.context_encoder = nn.Linear(in_features, self.context_dim)
        self.sector_hypernets = nn.ModuleList(
            [
                HyperNetwork(
                    self.context_dim,
                    self.dna_dim * self.edges_per_sector,
                    hidden_dim=self.hyper_hidden_dim,
                )
                for _ in range(self.num_sectors)
            ]
        )

        self.projection_head = nn.Sequential(
            nn.Linear(state_dim, projection_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(projection_dim, projection_dim),
        )

        self.output_layer = nn.Linear(state_dim, out_features)
        if output_scale_init is None:
            self.output_log_scale = None
        else:
            scale = torch.as_tensor(output_scale_init, dtype=torch.float32)
            if scale.ndim == 0:
                scale = scale.expand(out_features).clone()
            if scale.shape != (out_features,):
                raise ValueError(
                    "output_scale_init deve ser escalar ou ter formato (out_features,)."
                )
            if torch.any(scale <= 0):
                raise ValueError("output_scale_init deve conter apenas valores positivos.")
            log_scale = scale.log()
            if learnable_output_scale:
                self.output_log_scale = nn.Parameter(log_scale)
            else:
                self.register_buffer("output_log_scale", log_scale)
        self.rms_norm = RMSNorm(state_dim)

    def _apply_output_scale(self, raw_state):
        if self.output_log_scale is None:
            return raw_state
        return raw_state * self.output_log_scale.exp().to(
            device=raw_state.device,
            dtype=raw_state.dtype,
        )

    @staticmethod
    def _dedupe_parameters(parameters):
        seen = set()
        unique = []
        for parameter in parameters:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            unique.append(parameter)
        return unique

    def get_t0_parameters(self):
        return self._dedupe_parameters(
            self.get_t0_context_parameters() + self.get_t0_policy_parameters()
        )

    def get_t0_context_parameters(self):
        return self._dedupe_parameters(list(self.context_encoder.parameters()))

    def get_t0_policy_parameters(self):
        return self._dedupe_parameters(
            list(self.sector_hypernets.parameters()) + [self.dna_scale]
        )

    def get_t1_parameters(self):
        t0_ids = {id(parameter) for parameter in self.get_t0_parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in t0_ids]

    def _set_graph(self, graph: KRegularGraph):
        self.graph = graph
        src_nodes = torch.tensor([u for u, _ in self.graph.edges], dtype=torch.long)
        dst_nodes = torch.tensor([v for _, v in self.graph.edges], dtype=torch.long)
        if "src_nodes" in self._buffers:
            self.src_nodes = src_nodes
            self.dst_nodes = dst_nodes
        else:
            self.register_buffer("src_nodes", src_nodes)
            self.register_buffer("dst_nodes", dst_nodes)
        if hasattr(self, "delegator"):
            semantic_groups = self.graph.semantic_groups
            self.delegator.entry_indices = torch.tensor(
                semantic_groups.get("entry", [0]),
                dtype=torch.long,
                device=self.delegator.entry_indices.device,
            )
            self.delegator.memory_indices = torch.tensor(
                semantic_groups.get("memory", [0]),
                dtype=torch.long,
                device=self.delegator.memory_indices.device,
            )
            self.delegator.context_indices = torch.tensor(
                semantic_groups.get("context", [0]),
                dtype=torch.long,
                device=self.delegator.context_indices.device,
            )
            self.delegator.global_indices = torch.tensor(
                semantic_groups.get("global", [0]),
                dtype=torch.long,
                device=self.delegator.global_indices.device,
            )

    def export_checkpoint_state(self) -> dict:
        return {
            "state_dict": super().state_dict(),
            "graph_state": self.graph.export_metadata(),
        }

    def load_checkpoint_state(self, state):
        if not isinstance(state, dict) or "state_dict" not in state:
            super().load_state_dict(state)
            return self

        graph_state = state.get("graph_state")
        if graph_state is None:
            if state.get("requires_graph_topology", False):
                raise ValueError(
                    "Este checkpoint HSAMA precede a serialização da topologia do grafo e não "
                    "registrou uma semente (graph_seed) explícita. O layout das arestas não pode ser restaurado com segurança."
                )
        else:
            graph = KRegularGraph.from_metadata(graph_state)
            if graph.num_edges != self.graph.num_edges:
                raise ValueError(
                    "A topologia do grafo do checkpoint é incompatível com a arquitetura HSAMA "
                    "instanciada: divergência na contagem de arestas."
                )
            self._set_graph(graph)
        sd = state["state_dict"]
        remapped = {}
        for key, value in sd.items():
            new_key = key.replace("layer_norm.", "rms_norm.") if key.startswith("layer_norm.") else key
            remapped[new_key] = value
        super().load_state_dict(remapped)
        return self

    def _prepare_surprisal(self, surprisal, batch_size, device, dtype):
        surp_t = torch.as_tensor(surprisal, device=device, dtype=dtype)
        if surp_t.ndim == 0:
            surp_t = surp_t.expand(batch_size)
        elif surp_t.ndim == 1 and surp_t.numel() == 1:
            surp_t = surp_t.expand(batch_size)
        elif surp_t.ndim == 1 and surp_t.numel() == batch_size:
            pass
        elif surp_t.ndim == 2 and surp_t.shape == (batch_size, 1):
            surp_t = surp_t.squeeze(-1)
        else:
            raise ValueError(
                "surprisal deve ser um escalar, formato (B,) ou formato (B, 1)."
            )
        return surp_t.view(batch_size, 1, 1)

    def _resolve_exploration_mode(self, use_exploration):
        if use_exploration is None:
            return self.training
        return bool(use_exploration)

    def _should_reduce_hops(self, surprisal_variance, delta):
        if surprisal_variance is None:
            return False
        phi_tensor = torch.as_tensor(surprisal_variance)
        return bool(torch.all(phi_tensor < delta).item())

    def encode_context(self, x):
        return self.context_encoder(x)

    def build_policy_from_context(self, context, surprisal=0.0, use_exploration=None):
        """
        Constrói um snapshot da política T0 a partir de um tensor de contexto pré-computado.
        """
        use_exploration = self._resolve_exploration_mode(use_exploration)
        batch_size = context.size(0)
        surprisal_tensor = self._prepare_surprisal(
            surprisal,
            batch_size=batch_size,
            device=context.device,
            dtype=context.dtype,
        )
        temperature = torch.sigmoid(surprisal_tensor)
        scale = self.dna_base_scale + self.dna_temp_scale * temperature

        flat_dna_chunks = []
        for hypernet in self.sector_hypernets:
            sector_dna = hypernet(context)
            flat_dna_chunks.append(sector_dna.view(batch_size, -1, self.dna_dim))

        all_dna = torch.cat(flat_dna_chunks, dim=1)[:, : self.graph.num_edges, :]
        dna_norm = all_dna.norm(dim=-1, keepdim=True).clamp(min=1.0)
        all_dna = (all_dna / dna_norm) * self.dna_scale

        if use_exploration:
            exploration_noise = (
                torch.randn_like(all_dna) * self.explore_noise_std * temperature
            )
        else:
            exploration_noise = torch.zeros_like(all_dna)

        edge_dnas = all_dna * scale + exploration_noise
        return HSAMAPolicySnapshot(
            edge_dnas=edge_dnas,
            context=context,
            surprisal=surprisal_tensor,
            temperature=temperature,
            scale=scale,
            use_exploration=use_exploration,
            exploration_noise=exploration_noise,
        )

    def build_policy(self, x, surprisal=0.0, use_exploration=None):
        """
        Constrói um snapshot da política T0 que pode ser executado repetidamente por T1.
        """
        context = self.encode_context(x)
        return self.build_policy_from_context(
            context,
            surprisal=surprisal,
            use_exploration=use_exploration,
        )

    def _batched_kan_forward(self, x, edge_dnas):
        # ABLAÇÃO C: Aresta MLP com Viés Dinâmico
        if x.ndim != 3 or edge_dnas.ndim != 3:
            raise ValueError("x e edge_dnas devem ser tensores de ordem 3.")
        batch_size, num_edges, state_dim = x.shape

        # Formato do DNA: (B, E, 2 * S)
        # Particionamos o DNA em dois vieses
        b1 = edge_dnas[:, :, :state_dim]
        b2 = edge_dnas[:, :, state_dim:]

        # Camada 1
        h = torch.einsum('bes,est->bet', x, self.mlp_w1)
        h = F.silu(h + b1)

        # Camada 2
        out = torch.einsum('bet,ets->bes', h, self.mlp_w2)
        out = out + b2
        
        return out

    def execute_policy(
        self,
        x,
        policy,
        surprisal_variance=None,
        delta=1e-3,
        raw_output=False,
        use_local_greedy=False,
        detach_local_greedy=False,
        phi_variance=None,
    ):
        """
        Executa um snapshot da política construído através de T1.
        """
        if surprisal_variance is None and phi_variance is not None:
            warnings.warn(
                "'phi_variance' está obsoleto; use 'surprisal_variance' em seu lugar.",
                DeprecationWarning,
                stacklevel=2,
            )
            surprisal_variance = phi_variance
        if not isinstance(policy, HSAMAPolicySnapshot):
            raise TypeError("policy deve ser um HSAMAPolicySnapshot.")

        batch_size = x.size(0)
        device = x.device
        node_states = torch.zeros(
            batch_size,
            self.num_nodes,
            self.state_dim,
            device=device,
            dtype=x.dtype,
        )
        node_states = self.delegator(
            x,
            node_states,
            context=policy.context,
            surprisal=policy.temperature.squeeze(-1).squeeze(-1),
        )

        current_hops = 1 if self._should_reduce_hops(surprisal_variance, delta) else self.max_hops
        hop_predictions = []

        for _ in range(current_hops):
            source_states = node_states[:, self.src_nodes, :]
            messages = self._batched_kan_forward(source_states, policy.edge_dnas)

            new_states = torch.zeros(
                batch_size,
                self.num_nodes,
                self.state_dim,
                device=device,
                dtype=x.dtype,
            )
            dst_expanded = self.dst_nodes.view(1, -1, 1).expand(
                batch_size, -1, self.state_dim
            )
            new_states.scatter_add_(1, dst_expanded, messages)
            new_states = self.rms_norm(new_states)

            if use_local_greedy:
                if detach_local_greedy:
                    warnings.warn(
                        "'detach_local_greedy=True' está obsoleto; a execução greedy local "
                        "agora preserva gradientes por padrão.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    node_states = node_states.detach()
                node_states = node_states + (self.hop_scale * new_states)

                mean_state = node_states.mean(dim=1)
                raw_state = self._apply_output_scale(self.output_layer(mean_state))
                hop_global = raw_state if raw_output else F.softplus(raw_state)
                hop_proj = self.projection_head(mean_state)
                hop_predictions.append((hop_global, hop_proj))
            else:
                node_states = node_states + (self.hop_scale * new_states)

        if use_local_greedy:
            return hop_predictions

        mean_state = node_states.mean(dim=1)
        raw_state = self._apply_output_scale(self.output_layer(mean_state))
        global_state = raw_state if raw_output else F.softplus(raw_state)
        projected_state = self.projection_head(mean_state)
        return global_state, projected_state

    def forward(
        self,
        x,
        surprisal_variance=None,
        surprisal=0.0,
        use_exploration=None,
        raw_output=False,
        use_local_greedy=False,
        detach_local_greedy=False,
        phi_variance=None,
    ):
        if surprisal_variance is None and phi_variance is not None:
            warnings.warn(
                "'phi_variance' está obsoleto; use 'surprisal_variance' em seu lugar.",
                DeprecationWarning,
                stacklevel=2,
            )
            surprisal_variance = phi_variance
        policy = self.build_policy(
            x,
            surprisal=surprisal,
            use_exploration=use_exploration,
        )
        return self.execute_policy(
            x,
            policy=policy,
            surprisal_variance=surprisal_variance,
            raw_output=raw_output,
            use_local_greedy=use_local_greedy,
            detach_local_greedy=detach_local_greedy,
        )
