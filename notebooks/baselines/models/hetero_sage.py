"""G1 HeteroSAGE: heterogeneous GraphSAGE over the leak-free per-Congress graph (DESIGN section 4, G1).

Encoder: shared TypeEncoder -> `layers` x HeteroConv({relation: SAGEConv((d, d), d, mean, root weight)}, sum)
with a residual, dropout and LayerNorm per node type -> the shared LinkHead (Task 1) / NodeHead (Task 2).

Lecture / paper grounding:
- GraphSAGE (Hamilton et al. 2017; L04): inductive, h_v <- W_root h_v + W_nbr mean_{u in N(v)} h_u, so new
  legislators and every (new) target bill get embeddings from features alone.
- Heterogeneous message passing (L09 p.50-54): one SAGEConv per relation (relation-specific weights), then a
  sum over the relations that point into a node type (two-stage aggregation, PyG `HeteroConv`).
- Residual + LayerNorm (L05 skip connections); LayerNorm instead of BatchNorm because degree statistics differ
  per Congress graph. DropEdge on history edges during training only (L05 graph augmentation).
- 2 layers = receptive field leg -> bill -> leg (L05; Schlichtkrull et al. 2018 and Russo et al. 2023 use 2).

Leak-freedom is structural: graphs.EDGE_TYPES never has `tbill` as a source, so a target bill only reads its
own incoming edges (sponsor, day-<=t0 cosponsors, referrals dated <= cutoff) and cannot influence legislators,
committees or other target bills. With `layers = 0` this class is exactly PairMLP (the base GraphModel).
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

from ..graphs import EDGE_TYPES, NODE_TYPES, TARGET_RELS
from .heads import GraphModel

HIST_EDGE_TYPES = tuple(et for et in EDGE_TYPES if et not in TARGET_RELS)
OUTPUT_TYPES = ("leg", "tbill")          # the only node states the heads read


def drop_edges(edge_index: torch.Tensor, p: float) -> torch.Tensor:
    """DropEdge: keep each edge independently with probability 1 - p (training only)."""
    if p <= 0 or edge_index.shape[1] == 0:
        return edge_index
    keep = torch.rand(edge_index.shape[1], device=edge_index.device) >= p
    return edge_index[:, keep]


class HeteroSAGE(GraphModel):
    """G1: TypeEncoder -> L x [HeteroConv(SAGEConv per relation, sum) + residual + dropout + LayerNorm] -> heads.

    The last layer only computes relations whose destination is `leg` or `tbill` (the heads never read hbill or
    comm states after the final layer), which roughly halves its cost without changing the model."""

    def __init__(self, cfg, meta: dict):
        super().__init__(cfg, meta)
        d, L = cfg.d, int(cfg.layers)
        aggr = getattr(cfg, "aggr", "mean")
        self.layer_rels = []
        convs, norms = [], []
        for i in range(L):
            rels = EDGE_TYPES if i < L - 1 else tuple(et for et in EDGE_TYPES if et[2] in OUTPUT_TYPES)
            self.layer_rels.append(tuple(rels))
            with warnings.catch_warnings():   # last layer: hbill / comm are intentionally not destinations
                warnings.filterwarnings("ignore", message="There exist node types")
                convs.append(HeteroConv({et: SAGEConv((d, d), d, aggr=aggr, root_weight=True) for et in rels},
                                        aggr="sum"))
            dst = sorted({et[2] for et in rels})
            norms.append(nn.ModuleDict({t: nn.LayerNorm(d) for t in dst}))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.dropout = float(cfg.dropout)
        self.drop_edge = float(getattr(cfg, "drop_edge", 0.0))

    def edge_dict(self, graph, rels) -> dict:
        """Edge indices of `rels`, with DropEdge applied to history (and reverse) edges in training mode."""
        out = {}
        for et in rels:
            ei = graph[et].edge_index
            if self.training and et in HIST_EDGE_TYPES:
                ei = drop_edges(ei, self.drop_edge)
            out[et] = ei
        return out

    def message_passing(self, h: dict, graph) -> dict:
        """h_t <- LayerNorm(h_t + Dropout(ReLU(sum_r SAGEConv_r(h_src, h_t)))) for every destination type t."""
        for conv, norm, rels in zip(self.convs, self.norms, self.layer_rels):
            ed = self.edge_dict(graph, rels)
            m = conv(h, ed)
            new = dict(h)
            for t, mt in m.items():
                new[t] = norm[t](h[t] + F.dropout(F.relu(mt), self.dropout, self.training))
            h = new
        return h
