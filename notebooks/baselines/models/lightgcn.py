"""G3 LightGCN-feat: parameter-free GCN-normalised diffusion of inductive features + MLPs (DESIGN section 4, G3).

Precomputed once per (standardised) graph, with nothing learned in the propagation:
- A = symmetric, binarised history adjacency over leg-hbill (sponsor, original and later cosponsor edges) and
  hbill-comm (referrals); no self-loops; target bills are NOT part of A.
- A_hat = D^-1/2 A D^-1/2 (the GCN / LightGCN normalisation; L03, L11 p.45-51).
- X = block-diagonal standardised node features [x_leg | x_hbill | x_comm] (one column block per type).
- P_j = A_hat^j X for j = 0..K by repeated sparse matmul (L11 p.47; SGC, Wu et al. 2019). Only the leg and
  comm rows are kept, and only the column blocks that are structurally reachable in j hops (e.g. a legislator's
  1-hop block is the hbill block: the degree-normalised sum of the titles / outcomes of the bills they sponsored
  or cosponsored; its 2-hop block holds co-sponsoring legislators and the committees of those bills).
- Target bills attach parameter-free and incoming-only: t_j(b) = [mean P_j over the sponsor ; over the day-<=t0
  cosponsors C_b ; over the referral committees dated <= cutoff] for j = 0..K.

Learned part (SIGN-style, Rossi et al. 2020; LightGCN layer combination, He et al. 2020):
- h0_v = shared TypeEncoder(x_v) (Linear + LayerNorm + ReLU, with the state / committee embeddings)
- combine = "concat": z_v = LN(h0_v + MLP([h0_v ; LN(P_1[v]) ; ... ; LN(P_K[v])])), one LayerNorm per hop block
  ("combine the hops" rather than "last hop only", because A_hat^j oscillates on a bipartite graph)
- combine = "mean": the LightGCN layer average alpha_j = 1 / (K + 1) over the shared block-diagonal space,
  z_v = LN(h0_v + MLP([h0_v ; LN(mean_j P_j[v])]))
- target bills: z_b = LN(h0_b + MLP([h0_b ; LN(t_0) ; ... ; LN(t_K)])) per (group, hop) block
- the shared LinkHead / NodeHead of every model (models/heads.py)

Why this is cheap: there is no message passing per step, so the forward is a few small MLPs; target-bill rows
are encoded lazily (only the rows a mini-batch needs), so mini-batches can mix Congresses (TrainConfig
mix_congresses=True). Leak-freedom is structural: A holds history edges only (Congresses W_k, all finished
before k starts) and target bills only read their own incoming edges, exactly like G1 / G2.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from ..graphs import HIST_RELS
from .heads import COMM_EMB, STATE_EMB, GraphModel, TypeEncoder, mlp

DIFF_TYPES = ("leg", "hbill", "comm")                  # node types of the history adjacency A
TYPE_NBRS = {"leg": ("hbill",), "hbill": ("leg", "comm"), "comm": ("hbill",)}
TBILL_GROUPS = (("sponsor", ("leg", "sponsors_t", "tbill")), ("cosp", ("leg", "cosp_t", "tbill")),
                ("comm", ("comm", "referred_t", "tbill")))


def reachable_blocks(K: int) -> dict[tuple[str, int], tuple[str, ...]]:
    """Column blocks of A_hat^j X that can be non-zero at rows of each type, for j = 0..K (structural, so the
    feature dimensions never depend on the data): blocks(t, 0) = {t}, blocks(t, j) = U_{t' ~ t} blocks(t', j-1)."""
    out = {(t, 0): (t,) for t in DIFF_TYPES}
    for j in range(1, K + 1):
        for t in DIFF_TYPES:
            s = set()
            for u in TYPE_NBRS[t]:
                s |= set(out[(u, j - 1)])
            out[(t, j)] = tuple(b for b in DIFF_TYPES if b in s)
    return out


def block_dims(x_dims: dict, K: int, combine: str) -> dict:
    """Input dimensions of every diffused block: {"leg": [dims per kept hop], "comm": [...], "tbill": [...]}.
    concat: leg / comm hops 1..K, tbill (group, hop) for hops 0..K; mean: one averaged block each."""
    rb = reachable_blocks(K)

    def dim(t, j):
        return sum(int(x_dims[b]) for b in rb[(t, j)])

    def mean_dim(t):
        return sum(int(x_dims[b]) for b in DIFF_TYPES if any(b in rb[(t, j)] for j in range(K + 1)))

    if combine == "concat":
        node = {t: [dim(t, j) for j in range(1, K + 1)] for t in ("leg", "comm")}
        tb = [dim(src, j) for _, (src, _, _) in TBILL_GROUPS for j in range(K + 1)]
    elif combine == "mean":
        node = {t: ([mean_dim(t)] if K > 0 else []) for t in ("leg", "comm")}
        tb = [mean_dim(src) for _, (src, _, _) in TBILL_GROUPS]
    else:
        raise ValueError(f"unknown combine {combine!r} (use 'concat' or 'mean')")
    return dict(node, tbill=tb)


def _row_mean(n_rows: int, n_cols: int, rows: np.ndarray, cols: np.ndarray) -> sp.csr_matrix:
    """Sparse row-normalised (mean) aggregation matrix; empty rows stay zero."""
    M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_rows, n_cols))
    M.data[:] = 1.0                                    # duplicates count once
    deg = np.asarray(M.sum(1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return sp.diags(inv) @ M


def diffuse(graph, K: int, combine: str = "concat") -> dict[str, list[torch.Tensor]]:
    """Parameter-free diffusion of one standardised graph (float64 in scipy, stored as float32 tensors).

    Returns {"leg": [blocks], "comm": [blocks], "tbill": [blocks]} in the order of `block_dims`. Reads only the
    history relations (HIST_RELS) for A and the target relations as incoming mean aggregations."""
    n = {t: int(graph[t].num_nodes) for t in DIFF_TYPES}
    off, N = {}, 0
    for t in DIFF_TYPES:
        off[t] = N
        N += n[t]
    r, c = [], []
    for s, rel, d in HIST_RELS:                        # history only; reverses give the symmetric part
        ei = graph[s, rel, d].edge_index.cpu().numpy()
        r.append(ei[0] + off[s])
        c.append(ei[1] + off[d])
    r, c = np.concatenate(r), np.concatenate(c)
    A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(N, N))
    A = (A + A.T).tocsr()
    A.data[:] = 1.0                                    # binarise (a sponsor is never also a cosponsor; safety)
    A.setdiag(0)
    A.eliminate_zeros()
    deg = np.asarray(A.sum(1)).ravel()
    dinv = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)
    Ahat = (sp.diags(dinv) @ A @ sp.diags(dinv)).tocsr()
    cols, F0 = {}, 0
    for t in DIFF_TYPES:
        f = int(graph[t].x.shape[1])
        cols[t] = (F0, F0 + f)
        F0 += f
    X = np.zeros((N, F0))
    for t in DIFF_TYPES:
        X[off[t]:off[t] + n[t], cols[t][0]:cols[t][1]] = graph[t].x.detach().cpu().numpy().astype(np.float64)
    P = [X]
    for _ in range(K):
        P.append(Ahat @ P[-1])
    rb = reachable_blocks(K)

    def take(j, t, blocks):
        rows = P[j][off[t]:off[t] + n[t]]
        return np.concatenate([rows[:, cols[b][0]:cols[b][1]] for b in blocks], 1)

    def node_blocks(t):
        if combine == "concat":
            return [take(j, t, rb[(t, j)]) for j in range(1, K + 1)]
        if K == 0:
            return []
        union = tuple(b for b in DIFF_TYPES if any(b in rb[(t, j)] for j in range(K + 1)))
        return [np.mean([take(j, t, union) for j in range(K + 1)], 0)]

    out = {t: node_blocks(t) for t in ("leg", "comm")}
    nb = int(graph["tbill"].num_nodes)
    tb = []
    for _, et in TBILL_GROUPS:
        src = et[0]
        ei = graph[et].edge_index.cpu().numpy()
        M = _row_mean(nb, n[src], ei[1], ei[0])
        if combine == "concat":
            tb += [M @ take(j, src, rb[(src, j)]) for j in range(K + 1)]
        else:
            union = tuple(b for b in DIFF_TYPES if any(b in rb[(src, j)] for j in range(K + 1)))
            tb.append(M @ np.mean([take(j, src, union) for j in range(K + 1)], 0))
    out["tbill"] = tb
    dev = graph["leg"].x.device
    return {t: [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev) for a in v] for t, v in out.items()}


class LazyRows:
    """Stand-in for an [n, d] embedding tensor that computes only the indexed rows (emb["tbill"][idx])."""

    def __init__(self, fn, n: int, d: int):
        self.fn, self.n, self.d = fn, int(n), int(d)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.fn(idx)

    @property
    def shape(self) -> tuple:
        return (self.n, self.d)

    def __len__(self) -> int:
        return self.n


class _BlockMLP(nn.Module):
    """z = LayerNorm(h0 + MLP([h0 ; LN(block_1) ; ... ; LN(block_m)])) (residual keeps the hop-0 path)."""

    def __init__(self, d: int, dims: list[int], hidden: tuple, dropout: float):
        super().__init__()
        self.lns = nn.ModuleList([nn.LayerNorm(int(x)) for x in dims])
        self.net = mlp([d + int(sum(dims)), *[int(h) for h in hidden], d], dropout)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, h0: torch.Tensor, blocks: list[torch.Tensor]) -> torch.Tensor:
        """h0 [n, d], blocks [n, dims_i] -> [n, d]."""
        x = torch.cat([h0] + [ln(b) for ln, b in zip(self.lns, blocks)], 1)
        return self.out_norm(h0 + self.net(x))


class LightGCNFeat(GraphModel):
    """G3: TypeEncoder (hop 0) + precomputed A_hat^j X blocks -> per-type MLPs -> shared heads.

    `prepare(data, congresses)` precomputes the diffusion for the given FoldData graphs; graphs that were not
    prepared (e.g. the test Congress, or a perturbed graph in checks.perturbation_test) are diffused on first
    use. The cache lives on the graph object (`graph._g3`), keyed by (K, combine), so the validation fit and the
    refit (different standardisation, hence different graph objects) never share entries."""

    def __init__(self, cfg, meta: dict):
        super().__init__(cfg, meta)
        d = int(cfg.d)
        hidden = tuple(cfg.mlp[:-1]) if len(cfg.mlp) > 1 else ()
        if cfg.mlp and int(cfg.mlp[-1]) != d:
            raise ValueError(f"LightGCNConfig.mlp must end in d = {d}, got {cfg.mlp}")
        self.K, self.combine = int(cfg.K), str(cfg.combine)
        # hbill states are never read by the heads, so the encoder only covers leg / comm / tbill
        self.encoder = TypeEncoder(meta, d, types=("leg", "comm", "tbill"))
        dims = block_dims(meta["x_dims"], self.K, self.combine)
        self.dims = dims
        self.mix = nn.ModuleDict({t: _BlockMLP(d, dims[t], hidden, float(cfg.dropout)) for t in ("leg", "comm", "tbill")})
        self.d = d

    # --- precomputation -----------------------------------------------------------------------------------
    def blocks(self, graph) -> dict[str, list[torch.Tensor]]:
        """Diffused blocks of `graph` (computed once per graph object and (K, combine))."""
        cache = getattr(graph, "_g3", None)
        if cache is None:
            cache = {}
            graph._g3 = cache
        key = (self.K, self.combine)
        # the entry also holds the feature tensors it was computed from, so a shallow graph copy with replaced
        # features (graphs.standardized) can never hit a stale entry
        xs = tuple(graph[t].x for t in DIFF_TYPES + ("tbill",))
        hit = cache.get(key)
        if hit is None or any(a is not b for a, b in zip(hit[0], xs)):
            blocks = diffuse(graph, self.K, self.combine)
            for t, v in blocks.items():
                assert [int(b.shape[1]) for b in v] == list(self.dims[t]), (t, [b.shape for b in v], self.dims[t])
            hit = cache[key] = (xs, blocks)
        return hit[1]

    def prepare(self, data, congresses) -> None:
        """Precompute A_hat^j X for the FoldData graphs of `congresses` before training."""
        for k in congresses:
            self.blocks(data.graphs[k])

    # --- encoders ------------------------------------------------------------------------------------------
    def _h0(self, t: str, graph, idx=None) -> torch.Tensor:
        """TypeEncoder output for node type t (rows idx, or all rows)."""
        enc = self.encoder
        x = graph[t].x if idx is None else graph[t].x[idx]
        if t == "leg":
            si = graph["leg"].state_idx if idx is None else graph["leg"].state_idx[idx]
            x = torch.cat([x, enc.state_emb(si)], 1)
        elif t == "comm":
            ci = graph["comm"].emb_idx if idx is None else graph["comm"].emb_idx[idx]
            x = torch.cat([x, enc.comm_emb(ci)], 1)
        return torch.relu(enc.norm[t](enc.lin[t](x)))

    def _encode_tbill(self, graph, blocks, idx=None) -> torch.Tensor:
        h0 = self._h0("tbill", graph, idx)
        bl = blocks["tbill"] if idx is None else [b[idx] for b in blocks["tbill"]]
        return self.mix["tbill"](h0, bl)

    def forward(self, graph) -> dict:
        """{"leg": [NL, d], "comm": [NC, d], "tbill": [NB, d]}; in training mode "tbill" is a LazyRows that encodes
        only the indexed target bills (each mini-batch reads a few hundred of the ~10k rows)."""
        bl = self.blocks(graph)
        out = {t: self.mix[t](self._h0(t, graph), bl[t]) for t in ("leg", "comm")}
        nb = int(graph["tbill"].num_nodes)
        if self.training:
            out["tbill"] = LazyRows(lambda idx: self._encode_tbill(graph, bl, idx), nb, self.d)
        else:
            out["tbill"] = self._encode_tbill(graph, bl)
        return out

    def message_passing(self, h: dict, graph) -> dict:
        """Not used: G3 has no learned message passing (forward reads the precomputed diffusion)."""
        return h
