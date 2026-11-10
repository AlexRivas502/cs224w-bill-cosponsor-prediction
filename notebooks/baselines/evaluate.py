"""Ranking and classification metrics, cluster bootstrap CIs and paired differences.

Ported from 40_splits.py (`_rank_metrics`, `_boot_weights`, `_auc_ap`, `_t1_summarize`, `_t2_summarize`,
`_pdiff`). Depends only on numpy / scipy / pandas and `data`.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from .data import Corpus, T1Block

TOPK = 20
BOOT_B = 500
T1_METRIC_NAMES = ("Recall@10", "Recall@20", "NDCG@20")
MIDX = {"Recall@10": 0, "Recall@20": 1, "NDCG@20": 2}


def boot_weights(clusters, key, B: int = BOOT_B) -> np.ndarray:
    """(B+1, n) resample-count matrix over clusters (row 0 = ones = point estimate), seeded by crc32(str(key))."""
    # ported from 40_splits.py:_boot_weights
    u, inv = np.unique(np.asarray(clusters), return_inverse=True)
    K = len(u)
    rng = np.random.default_rng(zlib.crc32(str(key).encode()))
    idx = rng.integers(0, K, (B, K))
    cnt = np.bincount((idx + np.arange(B)[:, None] * K).ravel(), minlength=B * K).reshape(B, K)
    return np.vstack([np.ones(len(inv)), cnt[:, inv]]).astype(float)


def rank_metrics(S: np.ndarray, Pos: np.ndarray, return_top: bool = False):
    """Per-row [Recall@10, Recall@20, NDCG@20] for scores S (ineligible = -inf) and positives Pos."""
    # ported from 40_splits.py:_rank_metrics
    K = TOPK
    top = np.argpartition(-S, K, axis=1)[:, :K]
    top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), axis=1), 1)
    hits = np.take_along_axis(Pos, top, 1).astype(float)
    npos = Pos.sum(1)
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    idcg = np.cumsum(disc)[np.minimum(npos, K) - 1]
    res = np.c_[hits[:, :10].sum(1) / npos, hits.sum(1) / npos, (hits @ disc) / idcg]
    return (res, top) if return_top else res


def block_metrics(S: np.ndarray, block: T1Block, tiebreak: float = 0.0, return_top: bool = False):
    """rank_metrics of a score matrix on a block: Excl -> -inf, plus `tiebreak` x block.tieb (learned models
    use 1e-9)."""
    S = np.array(S, dtype=float)
    if tiebreak:
        S = S + tiebreak * block.tieb[None, :]
    S[block.Excl] = -np.inf
    return rank_metrics(S, block.Pos, return_top=return_top)


def blocks_metrics(scores: list, blocks: list[T1Block], tiebreak: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-bill metrics over several blocks; returns (per_bill [n, 3], bill codes [n])."""
    pb = [block_metrics(S, b, tiebreak) for S, b in zip(scores, blocks)]
    return np.vstack(pb), np.concatenate([b.codes for b in blocks])


def auc_ap(s, y, W) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted AUROC, AP and positive rate for every row of W (tie-aware)."""
    # ported from 40_splits.py:_auc_ap
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    W = np.atleast_2d(W)
    uniq, inv = np.unique(-s, return_inverse=True)
    A = sparse.csr_matrix((np.ones(len(s)), (np.arange(len(s)), inv)), shape=(len(s), len(uniq)))
    Pg = np.asarray((A.T @ (W * y).T).T)
    Ng = np.asarray((A.T @ (W * (1 - y)).T).T)
    P, N = Pg.sum(1), Ng.sum(1)
    cP, cN = np.cumsum(Pg, 1), np.cumsum(Ng, 1)
    auc = (Pg * (N[:, None] - cN + 0.5 * Ng)).sum(1) / (P * N)
    ap = (Pg * cP / np.maximum(cP + cN, 1e-12)).sum(1) / P
    return auc, ap, P / (P + N)


def auroc(s, y) -> float:
    """Point AUROC (tie-aware)."""
    return float(auc_ap(s, y, np.ones((1, len(y))))[0][0])


def bill_clusters(corpus: Corpus, codes) -> np.ndarray:
    """Sponsor cluster of each bill code (sponsor-less bills are their own cluster)."""
    return corpus.BL.cluster.reindex(np.asarray(codes)).to_numpy()


def summarize_t1(per_bill: dict, codes: np.ndarray, corpus: Corpus, key, B: int = BOOT_B) -> list[dict]:
    """Point estimate plus cluster (sponsor) and bill bootstrap 95% CIs per model and metric, paired across
    models (same resamples)."""
    # ported from 40_splits.py:_t1_summarize
    n = len(codes)
    cl = bill_clusters(corpus, codes)
    Wc = boot_weights(cl, key, B)
    Wb = boot_weights(np.arange(n), (key, "bill"), B)
    res = []
    for b, v in per_bill.items():
        ec = (Wc @ v) / Wc.sum(1)[:, None]
        eb = (Wb @ v) / Wb.sum(1)[:, None]
        lo, hi = np.percentile(ec[1:], [2.5, 97.5], axis=0)
        lob, hib = np.percentile(eb[1:], [2.5, 97.5], axis=0)
        for j, mname in enumerate(T1_METRIC_NAMES):
            res.append(dict(model=b, metric=mname, value=float(ec[0, j]), ci_low=float(lo[j]), ci_high=float(hi[j]),
                            ci_low_bill=float(lob[j]), ci_high_bill=float(hib[j]), n_eval=n,
                            n_clusters=int(len(np.unique(cl)))))
    return res


def summarize_t2(scores: dict, rows: pd.DataFrame, key, B: int = BOOT_B) -> list[dict]:
    """AUROC, AP and AP lift with cluster and bill bootstrap CIs per model (paired resamples)."""
    # ported from 40_splits.py:_t2_summarize
    y = rows.y.to_numpy()
    Wc = boot_weights(rows.cluster.to_numpy(), key, B)
    Wb = boot_weights(np.arange(len(rows)), (key, "bill"), B)
    res = []
    for b, s in scores.items():
        auc, ap, prev = auc_ap(s, y, Wc)
        aucb, apb, prevb = auc_ap(s, y, Wb)
        for mname, est, estb in (("AUROC", auc, aucb), ("AP", ap, apb), ("AP_lift", ap / prev, apb / prevb)):
            lo, hi = np.percentile(est[1:], [2.5, 97.5])
            lob, hib = np.percentile(estb[1:], [2.5, 97.5])
            res.append(dict(model=b, metric=mname, value=float(est[0]), ci_low=float(lo), ci_high=float(hi),
                            ci_low_bill=float(lob), ci_high_bill=float(hib), n_eval=len(rows),
                            n_clusters=int(rows.cluster.nunique())))
    return res


def paired_diff(per_bill_a: np.ndarray, per_bill_b: np.ndarray, clusters, key, metric_idx: int = 1,
                B: int = BOOT_B) -> dict:
    """Paired cluster-bootstrap difference mean(a) - mean(b) of one Task-1 metric column on the same bills."""
    # ported from 40_splits.py:_pdiff (Task 1 branch)
    W = boot_weights(clusters, key, B)
    est = W @ (per_bill_a[:, metric_idx] - per_bill_b[:, metric_idx]) / W.sum(1)
    lo, hi = np.percentile(est[1:], [2.5, 97.5])
    return dict(diff=float(est[0]), ci_low=float(lo), ci_high=float(hi), half_width=float((hi - lo) / 2))


def paired_diff_t2(s_a, s_b, rows: pd.DataFrame, key, metric: str = "AUROC", B: int = BOOT_B) -> dict:
    """Paired cluster-bootstrap difference of AUROC (or AP_lift / AP) between two score vectors on the same rows."""
    # ported from 40_splits.py:_pdiff (Task 2 branch)
    y, W = rows.y.to_numpy(), boot_weights(rows.cluster.to_numpy(), key, B)
    ra, rb = auc_ap(s_a, y, W), auc_ap(s_b, y, W)
    if metric == "AUROC":
        est = ra[0] - rb[0]
    elif metric == "AP":
        est = ra[1] - rb[1]
    else:
        est = ra[1] / ra[2] - rb[1] / rb[2]
    lo, hi = np.percentile(est[1:], [2.5, 97.5])
    return dict(diff=float(est[0]), ci_low=float(lo), ci_high=float(hi), half_width=float((hi - lo) / 2))


def new_legislator_hits(scores: list, blocks: list[T1Block], is_new: np.ndarray, tiebreak: float = 0.0) -> dict:
    """Hit@20 share over positives whose legislator is new (is_new[leg] True) vs returning, pooled over blocks."""
    h = {"new": [0, 0], "returning": [0, 0]}
    for S, b in zip(scores, blocks):
        _, top = block_metrics(S, b, tiebreak, return_top=True)
        intop = np.zeros(b.Pos.shape, bool)
        intop[np.arange(len(top))[:, None], top] = True
        pr, pc = np.nonzero(b.Pos)
        new = is_new[b.cand[pc]]
        for name, m in (("new", new), ("returning", ~new)):
            h[name][0] += int(intop[pr[m], pc[m]].sum())
            h[name][1] += int(m.sum())
    return {k: dict(hits=v[0], positives=v[1], hit_rate=(v[0] / v[1]) if v[1] else float("nan")) for k, v in h.items()}


# --- Result containers and the row schema of report.Row ---------------------------------------------
@dataclass
class EvalResult:
    """Per-bill output of one model on one evaluation unit (fold, role, task, t0, protocol).

    Task 1: `per_bill` [n, 3] metrics and `codes` (bill codes, identical across models of a unit). Task 2:
    `scores` [n] and `rows` (the _ADV rows, for labels and clusters). `kind` is one of heuristic /
    reference_leaky / learned_ref / gnn. `selection` is clean / future_tuned / contaminated."""
    task: str
    fold: int
    role: str
    t0: int
    protocol: str
    model: str
    codes: np.ndarray
    per_bill: np.ndarray | None = None
    scores: np.ndarray | None = None
    rows: pd.DataFrame | None = None
    kind: str = "heuristic"
    config_set: str = "reference"
    selection: str = "clean"
    sup: str = ""
    seed: int | None = None
    runtime_s: float = float("nan")
    extra: dict = field(default_factory=dict)

    @property
    def eval_key(self) -> tuple:
        """Bootstrap key shared by every model of the same evaluation unit (paired resamples)."""
        return (int(self.fold), self.task, int(self.t0), "eval", self.role)


def summarize_results(results: list[EvalResult], corpus: Corpus, B: int = BOOT_B) -> list[dict]:
    """Row dicts (report.Row schema) with cluster-bootstrap CIs for a list of EvalResults."""
    out = []
    for r in results:
        if r.task == "t1":
            stats = summarize_t1({r.model: r.per_bill}, r.codes, corpus, r.eval_key, B)
        else:
            stats = summarize_t2({r.model: r.scores}, r.rows, r.eval_key, B)
        for st in stats:
            out.append(dict(fold=r.fold, role=r.role, task=r.task, t0=r.t0, protocol=r.protocol, model=r.model,
                            kind=r.kind, config_set=r.config_set, selection=r.selection, sup=r.sup, seed=r.seed,
                            metric=st["metric"], value=st["value"], ci_low=st["ci_low"], ci_high=st["ci_high"],
                            ci_low_bill=st["ci_low_bill"], ci_high_bill=st["ci_high_bill"], n_eval=st["n_eval"],
                            runtime_s=r.runtime_s))
    return out


def paired_vs(a: EvalResult, b: EvalResult, corpus: Corpus, metric: str | None = None, B: int = BOOT_B) -> dict:
    """Paired cluster-bootstrap difference a - b on the same evaluation unit (asserts identical bills)."""
    assert a.task == b.task and np.array_equal(a.codes, b.codes), "paired difference needs identical bills"
    if a.task == "t1":
        metric = metric or "Recall@20"
        d = paired_diff(a.per_bill, b.per_bill, bill_clusters(corpus, a.codes), (a.eval_key, "paired"),
                        MIDX[metric], B)
    else:
        metric = metric or "AUROC"
        d = paired_diff_t2(a.scores, b.scores, a.rows, (a.eval_key, "paired"), metric, B)
    return dict(a=a.model, a_protocol=a.protocol, b=b.model, b_protocol=b.protocol, metric=metric, **d)


def new_leg_mask(corpus: Corpus, k: int) -> np.ndarray:
    """[NL] True for legislators with no cosponsorship in 108..k-1 (the 'new legislator' stratum)."""
    m = (corpus.CD["cong"] < k) & (corpus.CD["leg"] >= 0)
    return np.bincount(corpus.CD["leg"][m], minlength=corpus.NL) == 0
