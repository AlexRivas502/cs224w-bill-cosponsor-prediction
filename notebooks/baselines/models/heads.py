"""Shared model pieces (DESIGN section 4): TypeEncoder, LinkHead, NodeHead and the GraphModel base class.

Every model exposes forward(graph) -> {"leg": [NL, d], "tbill": [NB, d], ...}, score_t1(emb, part) -> [nb, M]
and score_t2(emb, part) -> [n]. GNN subclasses override `message_passing(h, graph)`; the base class does no
message passing, which is exactly PairMLP (= G1 with 0 layers).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NODE_TYPES = ("leg", "hbill", "comm", "tbill")
STATE_EMB = 8
COMM_EMB = 16


def mlp(dims: list[int], dropout: float = 0.0, last_act: bool = False) -> nn.Sequential:
    """Linear-ReLU(-Dropout) stack over `dims` (no activation after the last layer unless last_act)."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class TypeEncoder(nn.Module):
    """Per node type Linear(F_type [+ embedding] -> d) + LayerNorm + ReLU (L09 type-specific pre-processing).

    leg rows append an 8-d state embedding (`state_idx`), comm rows a 16-d committee embedding (`emb_idx`,
    'unk' row for committees unseen in supervision). Input sizes come from `meta["x_dims"]`."""

    def __init__(self, meta: dict, d: int, types=NODE_TYPES):
        super().__init__()
        self.types = tuple(types)
        self.state_emb = nn.Embedding(meta["n_states"] + 1, STATE_EMB)
        self.comm_emb = nn.Embedding(meta["n_comm"], COMM_EMB)
        extra = {"leg": STATE_EMB, "comm": COMM_EMB}
        self.lin = nn.ModuleDict({t: nn.Linear(meta["x_dims"][t] + extra.get(t, 0), d) for t in self.types})
        self.norm = nn.ModuleDict({t: nn.LayerNorm(d) for t in self.types})

    def forward(self, graph) -> dict[str, torch.Tensor]:
        """Encoded node states {type: [n_type, d]}."""
        out = {}
        for t in self.types:
            x = graph[t].x
            if t == "leg":
                x = torch.cat([x, self.state_emb(graph["leg"].state_idx)], 1)
            elif t == "comm":
                x = torch.cat([x, self.comm_emb(graph["comm"].emb_idx)], 1)
            out[t] = F.relu(self.norm[t](self.lin[t](x)))
        return out


class LinkHead(nn.Module):
    """Task-1 scorer s(b, l) = <q_b, k_l> / 8 + g(phi_bl) with q_b = MLP([h_b ; h_sponsor]), k_l = MLP(h_l)
    (two-tower, sponsor-conditioned as in Russo et al. 2023) and g = Linear(F -> 32) -> ReLU -> Linear(32 -> 1)."""

    def __init__(self, d: int, phi_dim: int, tower: int = 64, pair_hidden: int = 32, use_dot: bool = True,
                 use_pair: bool = True, dropout: float = 0.0):
        super().__init__()
        self.use_dot, self.use_pair, self.tower = use_dot, use_pair, tower
        self.q = mlp([2 * d, d, tower], dropout)
        self.k = mlp([d, tower], 0.0)
        self.g = mlp([phi_dim, pair_hidden, 1], 0.0)

    def forward(self, h_b: torch.Tensor, h_s: torch.Tensor, h_l: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """h_b [nb, d], h_s [nb, d] (zeros without sponsor), h_l [M, d], phi [nb, M, F] -> scores [nb, M]."""
        s = torch.zeros(phi.shape[:2], device=phi.device)
        if self.use_dot:
            s = s + self.q(torch.cat([h_b, h_s], 1)) @ self.k(h_l).T / (self.tower ** 0.5)
        if self.use_pair:
            s = s + self.g(phi).squeeze(-1)
        return s


class NodeHead(nn.Module):
    """Task-2 head MLP([h_b ; h_sponsor ; x_b]) -> 128 -> 1 logit."""

    def __init__(self, d: int, xb_dim: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = mlp([2 * d + xb_dim, hidden, 1], dropout)

    def forward(self, h_b, h_s, x_b) -> torch.Tensor:
        """Logits [n]."""
        return self.net(torch.cat([h_b, h_s, x_b], 1)).squeeze(-1)


def gather_sponsor(h_leg: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
    """Sponsor states h_leg[sp] with zeros where sp < 0."""
    hs = h_leg[sp.clamp(min=0)]
    return hs * (sp >= 0).unsqueeze(1).to(hs.dtype)


class GraphModel(nn.Module):
    """Base model: TypeEncoder -> message_passing (identity here) -> LinkHead / NodeHead.

    With the identity message passing this is PairMLP: target bills only see their own x_b, legislators their
    own features, and Task-1 scores come from the two towers plus g(phi)."""

    def __init__(self, cfg, meta: dict):
        super().__init__()
        self.cfg, self.meta = cfg, meta
        d = cfg.d
        self.encoder = TypeEncoder(meta, d)
        self.link = LinkHead(d, meta["phi_dim"], cfg.tower, cfg.pair_hidden, cfg.use_dot, cfg.use_pair, cfg.dropout)
        self.node = NodeHead(d, meta["xb_dim"], 128, cfg.dropout)

    def message_passing(self, h: dict, graph) -> dict:
        """Override in GNN subclasses; identity here (PairMLP)."""
        return h

    def forward(self, graph) -> dict[str, torch.Tensor]:
        """Node states for one graph: {"leg": [NL, d], "tbill": [NB, d], ...}."""
        return self.message_passing(self.encoder(graph), graph)

    def score_t1(self, emb: dict, part) -> torch.Tensor:
        """Scores [nb, M] for a Task-1 part (tb, sp, cand, phi standardised)."""
        h_b = emb["tbill"][part.tb]
        return self.link(h_b, gather_sponsor(emb["leg"], part.sp), emb["leg"][part.cand], part.phi)

    def score_t2(self, emb: dict, part) -> torch.Tensor:
        """Logits [n] for a Task-2 part (tb, sp, xb standardised)."""
        return self.node(emb["tbill"][part.tb], gather_sponsor(emb["leg"], part.sp), part.xb)
