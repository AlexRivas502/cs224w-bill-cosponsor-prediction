"""Leak-free heterogeneous graphs, one per target Congress k (DESIGN section 3.2).

Node types: leg (all legislators, index = legislator code), hbill (every bill of W_k), comm (40 full committees
+ 'unk'), tbill (law-capable bills of k). History relations (finished before k starts) come with reverses;
target relations point INTO tbill only, so target bills never see each other (the Task-2 "plain" rule) and
cannot change leg / comm states. Label-window cosponsors and committee reports are never edges.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .data import Corpus, T1Block, hist_window
from .features import (Standardizer, TitleSVD, comm_features, day0_cosponsors, hbill_features, leg_features,
                       referrals_upto, tbill_features)

NODE_TYPES = ("leg", "hbill", "comm", "tbill")
HIST_RELS = [("leg", "sponsors", "hbill"), ("leg", "cosp_orig", "hbill"), ("leg", "cosp_later", "hbill"),
             ("hbill", "referred", "comm")]
REV_RELS = [("hbill", "rev_sponsors", "leg"), ("hbill", "rev_cosp_orig", "leg"), ("hbill", "rev_cosp_later", "leg"),
            ("comm", "rev_referred", "hbill")]
TARGET_RELS = [("leg", "sponsors_t", "tbill"), ("leg", "cosp_t", "tbill"), ("comm", "referred_t", "tbill")]
EDGE_TYPES = HIST_RELS + REV_RELS + TARGET_RELS          # 11 relations, fixed order
_GRAPH_CACHE: dict = {}


def _t(x) -> torch.Tensor:
    """Tensor from a (possibly read-only, pandas copy-on-write) array, copying it."""
    return torch.as_tensor(np.array(x))


def _ei(src, dst) -> torch.Tensor:
    return _t(np.vstack([np.asarray(src, np.int64), np.asarray(dst, np.int64)]))


def build_graph(corpus: Corpus, k: int, t0: int, H: int, title: TitleSVD) -> HeteroData:
    """HeteroData for target Congress k at prediction time t0 with history window H and a title fit.

    Raw (unstandardised) features are stored as `x`; `leg.state_idx` / `comm.code_idx` feed embeddings.
    Attributes: graph.k, t0, H, window, tbill_codes, hbill_codes; target edges carry `edge_day` (event day)
    and `cut_day` (the bill's cutoff) for check_graph. Cached in memory per (k, t0, H, title fit)."""
    key = (corpus.token, int(k), int(t0), int(H), title.max_congress, float(np.abs(title.emb[:5]).sum()))
    if key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]
    g = HeteroData()
    W = hist_window(k, H)
    Xl, nl, sidx = leg_features(corpus, k, H)
    Xh, nh, hcodes = hbill_features(corpus, k, H, title)
    Xc, nc = comm_features(corpus, k, H)
    Xt, nt, tcodes = tbill_features(corpus, k, t0, H, title)
    g["leg"].x = torch.from_numpy(Xl)
    g["leg"].state_idx = torch.from_numpy(sidx)
    g["hbill"].x = torch.from_numpy(Xh)
    g["hbill"].congress = _t(corpus.bills.congress.to_numpy()[hcodes])
    g["comm"].x = torch.from_numpy(Xc)
    g["comm"].code_idx = torch.arange(Xc.shape[0])
    g["tbill"].x = torch.from_numpy(Xt)
    g.feature_names = {"leg": nl, "hbill": nh, "comm": nc, "tbill": nt}
    CD = corpus.CD
    hpos = pd.Index(hcodes)
    # sponsors (history)
    spl = corpus.leg_index.get_indexer(corpus.bills.sponsor_bioguide_id.iloc[hcodes].fillna("?"))
    ok = spl >= 0
    g["leg", "sponsors", "hbill"].edge_index = _ei(spl[ok], np.flatnonzero(ok))
    # cosponsorships (history), original vs later
    m = np.isin(CD["cong"], W) & (CD["leg"] >= 0)
    hb = hpos.get_indexer(CD["bill"][m])
    okb = hb >= 0
    lg, hb, orig = CD["leg"][m][okb], hb[okb], CD["orig"][m][okb]
    g["leg", "cosp_orig", "hbill"].edge_index = _ei(lg[orig], hb[orig])
    g["leg", "cosp_later", "hbill"].edge_index = _ei(lg[~orig], hb[~orig])
    # referrals (history), full committees only
    ref = corpus.referrals
    rr = ref[ref.congress.isin(W)].drop_duplicates(["code", "comm"])
    rh = hpos.get_indexer(rr.code.to_numpy())
    okr = rh >= 0
    g["hbill", "referred", "comm"].edge_index = _ei(rh[okr], rr.comm.to_numpy()[okr])
    for (s, r, d), (s2, r2, d2) in zip(HIST_RELS, REV_RELS):
        g[s2, r2, d2].edge_index = g[s, r, d].edge_index.flip(0).contiguous()
    # target relations: incoming only, dated <= cut
    tb = corpus.bills.iloc[tcodes]
    tday = tb.introduced_date.to_numpy().astype("datetime64[D]").astype(np.int64)
    sp_t = corpus.leg_index.get_indexer(tb.sponsor_bioguide_id.fillna("?"))
    ok = sp_t >= 0
    e = g["leg", "sponsors_t", "tbill"]
    e.edge_index = _ei(sp_t[ok], np.flatnonzero(ok))
    e.edge_day, e.cut_day = _t(tday[ok]), _t(tday[ok] + t0)
    rp, lgs = day0_cosponsors(corpus, tcodes, t0)
    pos = pd.Index(tcodes).get_indexer(CD["bill"])
    mm = (pos >= 0) & (CD["lag"] <= t0) & (CD["leg"] >= 0)
    e = g["leg", "cosp_t", "tbill"]
    e.edge_index = _ei(lgs, rp)
    e.edge_lag = _t(CD["lag"][mm])
    e.edge_day = _t(CD["day"][mm])
    e.cut_day = _t(tday[rp] + t0)
    rp2, cm, dd = referrals_upto(corpus, tcodes, t0)
    e = g["comm", "referred_t", "tbill"]
    e.edge_index = _ei(cm, rp2)
    e.edge_day, e.cut_day = _t(dd), _t(tday[rp2] + t0)
    g.k, g.t0, g.H, g.window = int(k), int(t0), int(H), tuple(W)
    g.tbill_codes, g.hbill_codes = tcodes, hcodes
    g.tbill_day = tday
    _GRAPH_CACHE[key] = g
    return g


@dataclass
class HomView:
    """Homogeneous view of a HeteroData in a fixed type / relation order (built once per graph, for R-GCN).

    offsets[type] = first global node id of the type; edge_type = relation id in EDGE_TYPES order;
    is_hist = True for history (and reverse) edges, the only edges DropEdge may drop."""
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    is_hist: torch.Tensor
    offsets: dict
    num_nodes: int
    node_types: tuple = NODE_TYPES
    edge_types: tuple = tuple(EDGE_TYPES)


def homogeneous_view(graph: HeteroData) -> HomView:
    """Concatenate all relations into one edge_index with per-type node offsets (cached on the graph)."""
    if getattr(graph, "_hom", None) is not None:
        return graph._hom
    offsets, n = {}, 0
    for t in NODE_TYPES:
        offsets[t] = n
        n += graph[t].num_nodes
    eis, ets, hist = [], [], []
    for r, et in enumerate(EDGE_TYPES):
        s, _, d = et
        assert s != "tbill", "tbill must never be a message source"
        ei = graph[et].edge_index
        eis.append(torch.stack([ei[0] + offsets[s], ei[1] + offsets[d]]))
        ets.append(torch.full((ei.shape[1],), r, dtype=torch.long))
        hist.append(torch.full((ei.shape[1],), et not in TARGET_RELS, dtype=torch.bool))
    hv = HomView(torch.cat(eis, 1), torch.cat(ets), torch.cat(hist), offsets, n)
    tb0 = offsets["tbill"]
    assert not bool((hv.edge_index[0] >= tb0).any()), "homogeneous view has a tbill source"
    graph._hom = hv
    return hv


def check_graph(graph: HeteroData, corpus: Corpus, k: int, t0: int, blocks: list[T1Block] | None = None) -> dict:
    """Leak checks (DESIGN check 3): no tbill sources; cosp_t lag <= t0 and dated <= cut; referred_t dated <= cut;
    hbill Congresses in W_k; no target edge (l, b) with l a Task-1 positive of b; full-committee referrals only.
    Returns a summary dict; raises AssertionError on any violation."""
    W = hist_window(k, graph.H)
    for et in graph.edge_types:
        assert et[0] != "tbill", f"edge type {et} has tbill as source"
    homogeneous_view(graph)
    e = graph["leg", "cosp_t", "tbill"]
    assert bool((e.edge_lag <= t0).all()), "cosp_t edge with lag > t0"
    assert bool((e.edge_day <= e.cut_day).all()), "cosp_t edge dated after the cutoff"
    e = graph["comm", "referred_t", "tbill"]
    assert bool((e.edge_day <= e.cut_day).all()), "referred_t edge dated after the cutoff"
    e = graph["leg", "sponsors_t", "tbill"]
    assert bool((e.edge_day <= e.cut_day).all())
    hc = graph["hbill"].congress.numpy()
    assert set(np.unique(hc)) <= set(W), f"hbill Congresses {set(np.unique(hc))} outside W_k {W}"
    assert int(graph["comm"].num_nodes) == len(corpus.committees) + 1
    for rel in (("comm", "referred_t", "tbill"), ("hbill", "referred", "comm")):
        src = graph[rel].edge_index[0 if rel[0] == "comm" else 1]
        assert src.numel() == 0 or int(src.max()) < len(corpus.committees), "referral to the 'unk' committee row"
    n_pos_edge = 0
    if blocks:
        for rel in (("leg", "cosp_t", "tbill"), ("leg", "sponsors_t", "tbill")):
            ei = graph[rel].edge_index.numpy()
            edge_set = pd.MultiIndex.from_arrays([graph.tbill_codes[ei[1]], ei[0]])
            for b in blocks:
                pr, pc = np.nonzero(b.Pos)
                pos_set = pd.MultiIndex.from_arrays([b.codes[pr], b.cand[pc]])
                n_pos_edge += int(edge_set.isin(pos_set).sum())
        assert n_pos_edge == 0, f"{n_pos_edge} target edges point from a label-window positive"
    return dict(k=k, t0=t0, window=W, n_edges={"/".join(et): int(graph[et].edge_index.shape[1]) for et in graph.edge_types},
                n_nodes={t: int(graph[t].num_nodes) for t in NODE_TYPES}, pos_edges=n_pos_edge)


def describe_graph(graph: HeteroData) -> str:
    """Text schematic of node and edge types with counts."""
    lines = [f"Graph for target Congress {graph.k} (t0 = {graph.t0}, history window {list(graph.window)})", "  nodes:"]
    for t in NODE_TYPES:
        lines.append(f"    {t:6s} {graph[t].num_nodes:>8,}  x: {tuple(graph[t].x.shape)}")
    lines.append("  edges (src -> dst):")
    for et in EDGE_TYPES:
        tag = "target, incoming only" if et in TARGET_RELS else ("reverse" if et in REV_RELS else "history")
        lines.append(f"    {et[0]:5s} -[{et[1]}]-> {et[2]:6s} {graph[et].edge_index.shape[1]:>9,}  ({tag})")
    return "\n".join(lines)


def fit_node_standardizers(graphs: dict, sup_congresses) -> dict[str, Standardizer]:
    """Per node type Standardizer fitted on the supervision Congress graphs only (title block included)."""
    out = {}
    for t in NODE_TYPES:
        arrs = [graphs[k][t].x.numpy() for k in sup_congresses]
        out[t] = Standardizer().fit(arrs, sup_congresses, graphs[sup_congresses[0]].feature_names[t])
    return out


def standardized(graph: HeteroData, stds: dict[str, Standardizer], comm_seen: np.ndarray | None = None) -> HeteroData:
    """Shallow copy of a graph whose node features are z-scored with the given fit; committees not seen in the
    supervision Congresses map to the 'unk' embedding row."""
    g = graph.__copy__()                       # new stores; tensors shared, x replaced below
    for t in NODE_TYPES:
        g[t].x = torch.from_numpy(stds[t].transform(graph[t].x.numpy()))
    if comm_seen is not None:
        idx = graph["comm"].code_idx.clone()
        unk = graph["comm"].num_nodes - 1
        idx[~_t(comm_seen)] = unk
        g["comm"].emb_idx = idx
    else:
        g["comm"].emb_idx = graph["comm"].code_idx
    for a in ("k", "t0", "H", "window", "tbill_codes", "hbill_codes", "tbill_day", "feature_names"):
        setattr(g, a, getattr(graph, a))
    g._hom = getattr(graph, "_hom", None)
    return g
