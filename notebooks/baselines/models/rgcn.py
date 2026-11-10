"""G2 R-GCN with basis decomposition over the leak-free per-Congress graph (DESIGN section 4, G2).

Encoder: shared TypeEncoder -> `layers` x RGCN layer (11 relations, `num_bases` shared bases, mean
normalisation) with ReLU, dropout, residual and a per-type LayerNorm -> the shared LinkHead / NodeHead.

Lecture / paper grounding:
- Schlichtkrull et al. 2018, eq. 2: h_i' = W_0 h_i + sum_r sum_{j in N_i^r} (1 / c_{i,r}) W_r h_j with
  c_{i,r} = |N_i^r| (per-relation mean) and a self-connection W_0 (L09 p.25-26).
- Eq. 3, basis decomposition: W_r = sum_b a_{rb} V_b, so the 11 relations share `num_bases` matrices and rare
  relations (e.g. referred_t) borrow statistical strength from frequent ones (L09 p.27-28: "regularisation of
  the relation weights"). Block-diagonal decomposition (Schlichtkrull eq. 4, L09 p.27) is an option.
- Russo et al. 2023 used a relational GCN over sponsor / cosponsor relations for cosponsorship prediction,
  conditioned on [legislator ; bill ; sponsor]; the shared LinkHead keeps that conditioning.
- Residual + LayerNorm (L05) and DropEdge on history edges, exactly as in G1, so G1 vs G2 isolates the encoder:
  relation-specific transforms of a *shared* basis vs. one free SAGEConv per relation.

Two numerically equivalent implementations (`RGCNConfig.impl`):
- "pyg": torch_geometric `RGCNConv(aggr="mean")` on `graphs.homogeneous_view(graph)` (the reference).
- "manual" (default): the same equation computed per relation in local (per-type) coordinates. Each relation
  only transforms the rows it needs and applies W_r before the gather when the source type is smaller than the
  destination type (mean(x_j) W = mean(x_j W)). It is unit-tested against RGCNConv (`check_equivalence`) and
  is several times faster because PyG's RGCNConv multiplies the full node matrix by every W_r.

Leak-freedom is structural: no relation has `tbill` as its source (asserted on the homogeneous view), so a
target bill reads only its own incoming edges and cannot influence legislators, committees or other bills.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graphs import EDGE_TYPES, NODE_TYPES, TARGET_RELS, homogeneous_view
from .heads import GraphModel

HIST_EDGE_TYPES = tuple(et for et in EDGE_TYPES if et not in TARGET_RELS)
OUTPUT_TYPES = ("leg", "tbill")          # the only node states the heads read after the last layer
NUM_RELATIONS = len(EDGE_TYPES)


class RelWeights(nn.Module):
    """Relation weights W_r (r = 0..R-1) with basis (eq. 3), block-diagonal (eq. 4) or no decomposition,
    plus the self-connection W_0 and a bias. Initialisation follows PyG's RGCNConv (Glorot)."""

    def __init__(self, d: int, num_relations: int, num_bases: int, decomposition: str = "basis"):
        super().__init__()
        self.d, self.R, self.decomposition = d, num_relations, decomposition
        if decomposition == "basis":
            self.basis = nn.Parameter(torch.empty(num_bases, d, d))
            self.comp = nn.Parameter(torch.empty(num_relations, num_bases))
            _glorot(self.basis)
            _glorot(self.comp)
        elif decomposition == "block":
            if d % num_bases:
                raise ValueError(f"block decomposition needs d ({d}) divisible by num_bases ({num_bases})")
            self.nb = num_bases
            self.blocks = nn.Parameter(torch.empty(num_relations, num_bases, d // num_bases, d // num_bases))
            _glorot(self.blocks)
        elif decomposition == "none":
            self.full = nn.Parameter(torch.empty(num_relations, d, d))
            _glorot(self.full)
        else:
            raise ValueError(f"unknown decomposition {decomposition!r} (basis | block | none)")
        self.root = nn.Parameter(torch.empty(d, d))
        _glorot(self.root)
        self.bias = nn.Parameter(torch.zeros(d))

    def dense(self) -> torch.Tensor | None:
        """[R, d, d] relation matrices for basis / none (None for block, which is applied blockwise)."""
        if self.decomposition == "basis":
            return (self.comp @ self.basis.view(self.basis.shape[0], -1)).view(self.R, self.d, self.d)
        if self.decomposition == "none":
            return self.full
        return None

    def apply(self, x: torch.Tensor, r: int, W: torch.Tensor | None) -> torch.Tensor:
        """x @ W_r for rows x [n, d]."""
        if W is not None:
            return x @ W[r]
        n = x.shape[0]
        return torch.einsum("nbi,bij->nbj", x.view(n, self.nb, -1), self.blocks[r]).reshape(n, self.d)


def _glorot(t: torch.Tensor) -> None:
    a = math.sqrt(6.0 / (t.size(-2) + t.size(-1)))
    with torch.no_grad():
        t.uniform_(-a, a)


def mean_aggregate(x_src: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, n_dst: int) -> torch.Tensor:
    """Per-destination mean of x_src[src] (zeros for destinations with no edge): (1/|N_i^r|) sum_j x_j."""
    out = torch.zeros(n_dst, x_src.shape[1], dtype=x_src.dtype, device=x_src.device)
    if src.numel() == 0:
        return out
    out.index_add_(0, dst, x_src.index_select(0, src))
    deg = torch.bincount(dst, minlength=n_dst).clamp(min=1).to(x_src.dtype)
    return out / deg.unsqueeze(1)


def drop_mask(n: int, p: float, device) -> torch.Tensor | None:
    """Boolean keep mask for DropEdge (None when p == 0)."""
    if p <= 0 or n == 0:
        return None
    return torch.rand(n, device=device) >= p


def rgcn_layer(conv: RelWeights, h: dict, graph, dst_types, drop_edge: float = 0.0, training: bool = False) -> dict:
    """One R-GCN layer in local (per-type) coordinates (Schlichtkrull eq. 2):
    out_i = W_0 h_i + bias + sum_r mean_{j in N_i^r} W_r h_j, for nodes of `dst_types`. DropEdge (training only)
    removes history / reverse edges; target edges are always kept."""
    W = conv.dense()
    out = {t: h[t] @ conv.root + conv.bias for t in dst_types}
    for r, et in enumerate(EDGE_TYPES):
        s, _, t = et
        if t not in dst_types:
            continue
        ei = graph[et].edge_index
        if training and et in HIST_EDGE_TYPES:
            keep = drop_mask(ei.shape[1], drop_edge, ei.device)
            if keep is not None:
                ei = ei[:, keep]
        n_src, n_dst = h[s].shape[0], h[t].shape[0]
        if n_src <= n_dst:          # transform the (fewer) source rows first, then average
            out[t] = out[t] + mean_aggregate(conv.apply(h[s], r, W), ei[0], ei[1], n_dst)
        else:                       # average first, then transform the destination rows
            out[t] = out[t] + conv.apply(mean_aggregate(h[s], ei[0], ei[1], n_dst), r, W)
    return out


class RGCN(GraphModel):
    """G2: TypeEncoder -> L x [R-GCN layer + ReLU + dropout + residual + LayerNorm per type] -> shared heads.

    Every layer shares one self-connection W_0 across node types (as in Schlichtkrull et al.); type identity is
    carried by the per-type TypeEncoder and LayerNorm. The last layer only computes destinations `leg` and
    `tbill` (the heads read nothing else), which does not change the model's outputs."""

    def __init__(self, cfg, meta: dict):
        super().__init__(cfg, meta)
        d, L = cfg.d, int(cfg.layers)
        self.impl = getattr(cfg, "impl", "manual")
        if self.impl not in ("manual", "pyg"):
            raise ValueError(f"unknown R-GCN impl {self.impl!r} (manual | pyg)")
        decomp = getattr(cfg, "decomposition", "basis")
        if self.impl == "pyg":
            from torch_geometric.nn import RGCNConv
            kw = dict(num_bases=cfg.num_bases) if decomp == "basis" else (
                dict(num_blocks=cfg.num_bases) if decomp == "block" else {})
            self.convs = nn.ModuleList([RGCNConv(d, d, NUM_RELATIONS, aggr="mean", **kw) for _ in range(L)])
        else:
            self.convs = nn.ModuleList([RelWeights(d, NUM_RELATIONS, cfg.num_bases, decomp) for _ in range(L)])
        self.layer_dst = [NODE_TYPES if i < L - 1 else OUTPUT_TYPES for i in range(L)]
        self.norms = nn.ModuleList([nn.ModuleDict({t: nn.LayerNorm(d) for t in dst}) for dst in self.layer_dst])
        self.dropout = float(cfg.dropout)
        self.drop_edge = float(getattr(cfg, "drop_edge", 0.0))

    def _conv_pyg(self, conv, h: dict, graph) -> dict:
        """Reference path: RGCNConv on the cached homogeneous view (all node types)."""
        hv = homogeneous_view(graph)
        assert not bool((hv.edge_index[0] >= hv.offsets["tbill"]).any()), "tbill source in homogeneous view"
        x = torch.cat([h[t] for t in NODE_TYPES], 0)
        ei, et = hv.edge_index.to(x.device), hv.edge_type.to(x.device)
        if self.training:
            keep = drop_mask(ei.shape[1], self.drop_edge, x.device)
            if keep is not None:
                keep = keep | ~hv.is_hist.to(x.device)
                ei, et = ei[:, keep], et[keep]
        y = conv(x, ei, et)
        return {t: y[hv.offsets[t]:hv.offsets[t] + h[t].shape[0]] for t in NODE_TYPES}

    def message_passing(self, h: dict, graph) -> dict:
        """h_t <- LayerNorm(h_t + Dropout(ReLU(RGCN(h)_t))) for each destination type of the layer."""
        for conv, norm, dst in zip(self.convs, self.norms, self.layer_dst):
            m = (rgcn_layer(conv, h, graph, dst, self.drop_edge, self.training) if self.impl == "manual"
                 else self._conv_pyg(conv, h, graph))
            new = dict(h)
            for t in dst:
                new[t] = norm[t](h[t] + F.dropout(F.relu(m[t]), self.dropout, self.training))
            h = new
        return h


def check_equivalence(graph, d: int = 16, num_bases: int = 4, decomposition: str = "basis", seed: int = 0,
                      atol: float = 1e-4) -> float:
    """Unit test: the manual relation loop equals torch_geometric RGCNConv(aggr="mean") with the same weights
    on `graph` (random inputs, no dropout / DropEdge). Returns the max absolute difference (asserted <= atol)."""
    from torch_geometric.nn import RGCNConv
    g = torch.Generator().manual_seed(seed)
    h = {t: torch.randn(graph[t].num_nodes, d, generator=g) for t in NODE_TYPES}
    kw = dict(num_bases=num_bases) if decomposition == "basis" else dict(num_blocks=num_bases)
    ref = RGCNConv(d, d, NUM_RELATIONS, aggr="mean", **kw)
    mine = RelWeights(d, NUM_RELATIONS, num_bases, decomposition)
    with torch.no_grad():
        if decomposition == "basis":
            mine.basis.copy_(ref.weight)
            mine.comp.copy_(ref.comp)
        else:
            mine.blocks.copy_(ref.weight)
        mine.root.copy_(ref.root)
        mine.bias.copy_(torch.randn(d, generator=g))
        ref.bias.copy_(mine.bias)
    hv = homogeneous_view(graph)
    with torch.no_grad():
        y = ref(torch.cat([h[t] for t in NODE_TYPES], 0), hv.edge_index, hv.edge_type)
        m = rgcn_layer(mine, h, graph, NODE_TYPES)
    diff = max(float((y[hv.offsets[t]:hv.offsets[t] + h[t].shape[0]] - m[t]).abs().max()) for t in NODE_TYPES)
    assert diff <= atol, f"manual R-GCN differs from RGCNConv by {diff}"
    return diff
