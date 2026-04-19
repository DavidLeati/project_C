import random

import networkx as nx
import torch
import torch.nn as nn


class KRegularGraph:
    """
    Representa a geometria estrutural fixa do grafo de computação base do H-SAMA.

    Implementa uma topologia K-Regular aprimorada com religação Small-World (Watts-Strogatz).
    Esta topografia facilita a passagem de mensagens eficiente e a diversidade estrutural.
    """

    def __init__(
        self,
        num_nodes,
        k=6,
        rewire_prob=0.1,
        seed=0,
        num_entry_nodes=4,
        ensure_connected=True,
        max_rewire_attempts=16,
    ):
        assert k % 2 == 0, "O parâmetro 'k' deve ser um número inteiro par para um grafo regular em anel"
        assert num_nodes > k, "O número de nós deve exceder estritamente o valor de 'k'"

        self.num_nodes = num_nodes
        self.k = k
        self.rewire_prob = rewire_prob
        self.seed = seed
        self.num_entry_nodes = min(num_entry_nodes, num_nodes)
        self.ensure_connected = bool(ensure_connected)
        self.max_rewire_attempts = max(1, int(max_rewire_attempts))
        self._rng = random.Random(seed)

        self.adj_list = self._build_adjacency()
        self.edges = self._build_edges()
        self.num_edges = len(self.edges)
        self.semantic_groups = self._build_semantic_groups()
        self.node_roles = self._build_node_roles()

    def _build_ring_adjacency(self):
        adj_list = {i: [] for i in range(self.num_nodes)}
        for i in range(self.num_nodes):
            for j in range(1, self.k // 2 + 1):
                forward_neighbor = (i + j) % self.num_nodes
                backward_neighbor = (i - j) % self.num_nodes

                if forward_neighbor not in adj_list[i]:
                    adj_list[i].append(forward_neighbor)
                if backward_neighbor not in adj_list[i]:
                    adj_list[i].append(backward_neighbor)
        return adj_list

    def _build_edges(self):
        edges = []
        for u in range(self.num_nodes):
            for v in self.adj_list[u]:
                edges.append((u, v))
        return edges

    def _is_connected(self, adj_list) -> bool:
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_nodes))
        for u, neighbors in adj_list.items():
            for v in neighbors:
                graph.add_edge(u, v)
        return nx.is_connected(graph)

    def _build_adjacency(self):
        base_adj = self._build_ring_adjacency()
        if self.rewire_prob <= 0.0:
            return base_adj

        attempt_limit = self.max_rewire_attempts if self.ensure_connected else 1
        for attempt in range(attempt_limit):
            candidate = {node: list(neighbors) for node, neighbors in base_adj.items()}
            attempt_seed = (self.seed if self.seed is not None else 0) + attempt
            self._rewire(candidate, random.Random(attempt_seed))
            if not self.ensure_connected or self._is_connected(candidate):
                return candidate
        return base_adj

    def _build_semantic_groups(self):
        step = max(1, self.num_nodes // self.num_entry_nodes)
        entry_nodes = []
        for idx in range(self.num_entry_nodes):
            node_idx = min(idx * step, self.num_nodes - 1)
            if node_idx not in entry_nodes:
                entry_nodes.append(node_idx)

        remaining_nodes = [i for i in range(self.num_nodes) if i not in entry_nodes]
        if not remaining_nodes:
            remaining_nodes = entry_nodes[:]

        global_nodes = [remaining_nodes[-1]]
        memory_nodes = remaining_nodes[: max(1, min(2, len(remaining_nodes) - 1 or 1))]
        context_nodes = [
            node
            for node in remaining_nodes
            if node not in global_nodes and node not in memory_nodes
        ]
        if not context_nodes:
            context_nodes = entry_nodes[:1]

        return {
            "entry": entry_nodes,
            "memory": memory_nodes,
            "context": context_nodes,
            "global": global_nodes,
        }

    def _build_node_roles(self):
        roles = ["latent"] * self.num_nodes
        for role, indices in self.semantic_groups.items():
            for index in indices:
                roles[index] = role
        return roles

    def _rewire(self, adj_list, rng):
        for u in range(self.num_nodes):
            neighbors = list(adj_list[u])
            for v in neighbors:
                if u < v and rng.random() < self.rewire_prob:
                    w = rng.randint(0, self.num_nodes - 1)
                    while w == u or w in adj_list[u]:
                        w = rng.randint(0, self.num_nodes - 1)

                    adj_list[u].remove(v)
                    adj_list[v].remove(u)
                    adj_list[u].append(w)
                    adj_list[w].append(u)

    def export_metadata(self) -> dict:
        return {
            "num_nodes": self.num_nodes,
            "k": self.k,
            "rewire_prob": self.rewire_prob,
            "seed": self.seed,
            "num_entry_nodes": self.num_entry_nodes,
            "ensure_connected": self.ensure_connected,
            "max_rewire_attempts": self.max_rewire_attempts,
            "edges": [list(edge) for edge in self.edges],
            "semantic_groups": {
                key: list(value)
                for key, value in self.semantic_groups.items()
            },
            "node_roles": list(self.node_roles),
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> "KRegularGraph":
        graph = cls.__new__(cls)
        graph.num_nodes = int(metadata["num_nodes"])
        graph.k = int(metadata["k"])
        graph.rewire_prob = float(metadata.get("rewire_prob", 0.1))
        graph.seed = metadata.get("seed", 0)
        graph.num_entry_nodes = int(metadata.get("num_entry_nodes", 4))
        graph.ensure_connected = bool(metadata.get("ensure_connected", True))
        graph.max_rewire_attempts = max(1, int(metadata.get("max_rewire_attempts", 16)))
        graph._rng = random.Random(graph.seed)
        graph.edges = [tuple(edge) for edge in metadata.get("edges", [])]
        graph.adj_list = {i: [] for i in range(graph.num_nodes)}
        for u, v in graph.edges:
            graph.adj_list[int(u)].append(int(v))
        graph.num_edges = len(graph.edges)
        graph.semantic_groups = {
            str(key): [int(value) for value in values]
            for key, values in metadata.get("semantic_groups", {}).items()
        } or graph._build_semantic_groups()
        graph.node_roles = list(metadata.get("node_roles", [])) or graph._build_node_roles()
        return graph

    def to_networkx(self):
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        G.add_edges_from(self.edges)
        nx.set_node_attributes(
            G,
            {idx: role for idx, role in enumerate(self.node_roles)},
            name="role",
        )
        return G


class InputDelegator(nn.Module):
    """
    Roteia sinais de entrada heterogêneos para nós âncora semanticamente distintos.
    """

    def __init__(
        self,
        in_features,
        num_nodes,
        num_entry_nodes=4,
        state_dim=1,
        semantic_groups=None,
        context_features=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_nodes = num_nodes
        self.num_entry_nodes = min(num_entry_nodes, num_nodes)
        self.context_features = in_features if context_features is None else context_features

        if semantic_groups is None:
            graph = KRegularGraph(
                num_nodes=num_nodes,
                k=2 if num_nodes <= 4 else 4,
                num_entry_nodes=self.num_entry_nodes,
            )
            semantic_groups = graph.semantic_groups

        self._register_index_buffer("entry_indices", semantic_groups.get("entry", []))
        self._register_index_buffer("memory_indices", semantic_groups.get("memory", []))
        self._register_index_buffer("context_indices", semantic_groups.get("context", []))
        self._register_index_buffer("global_indices", semantic_groups.get("global", []))

        hidden = max(16, state_dim * 2)
        self.gradient_projection = nn.Linear(in_features, state_dim)
        self.weight_projection = nn.Linear(in_features, state_dim)
        self.stability_projection = nn.Linear(in_features + 1, state_dim)
        self.global_projection = nn.Linear(self.context_features + 1, state_dim)
        self.router = nn.Sequential(
            nn.Linear(in_features + self.context_features + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4),
        )
        self.role_embeddings = nn.Parameter(torch.randn(4, state_dim) * 0.02)
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, state_dim) * 0.1)

    def _register_index_buffer(self, name, values):
        if not values:
            values = [0]
        self.register_buffer(name, torch.tensor(values, dtype=torch.long))

    def _resolve_context(self, batch_x, context):
        if context is None:
            if batch_x.size(-1) == self.context_features:
                return batch_x
            return torch.zeros(
                batch_x.size(0),
                self.context_features,
                device=batch_x.device,
                dtype=batch_x.dtype,
            )
        return context

    def _resolve_surprisal(self, batch_x, surprisal):
        if surprisal is None:
            return torch.zeros(
                batch_x.size(0),
                1,
                device=batch_x.device,
                dtype=batch_x.dtype,
            )
        surprisal = torch.as_tensor(
            surprisal,
            device=batch_x.device,
            dtype=batch_x.dtype,
        )
        if surprisal.ndim == 0:
            surprisal = surprisal.expand(batch_x.size(0))
        elif surprisal.ndim == 1 and surprisal.numel() == 1:
            surprisal = surprisal.expand(batch_x.size(0))
        elif surprisal.ndim == 2 and surprisal.shape[-1] == 1:
            surprisal = surprisal.squeeze(-1)
        return surprisal.view(batch_x.size(0), 1)

    def _inject(self, node_states, indices, signal, role_index):
        node_states[:, indices, :] += signal.unsqueeze(1) + self.role_embeddings[role_index]
        return node_states

    def forward(self, batch_x, node_states, context=None, surprisal=None):
        context = self._resolve_context(batch_x, context)
        surprisal = self._resolve_surprisal(batch_x, surprisal)

        router_input = torch.cat([batch_x, context, surprisal], dim=-1)
        routing = torch.softmax(self.router(router_input), dim=-1)

        gradient_signal = torch.tanh(self.gradient_projection(batch_x)) * routing[:, 0:1]
        weight_signal = torch.tanh(self.weight_projection(batch_x)) * routing[:, 1:2]
        stability_signal = torch.tanh(
            self.stability_projection(torch.cat([batch_x, surprisal], dim=-1))
        ) * routing[:, 2:3]
        global_signal = torch.tanh(
            self.global_projection(torch.cat([context, surprisal], dim=-1))
        ) * routing[:, 3:4]

        node_states = self._inject(node_states, self.entry_indices, gradient_signal, 0)
        node_states = self._inject(node_states, self.context_indices, weight_signal, 1)
        node_states = self._inject(node_states, self.memory_indices, stability_signal, 2)
        node_states = self._inject(node_states, self.global_indices, global_signal, 3)
        node_states = node_states + self.node_embeddings.unsqueeze(0)
        return node_states
