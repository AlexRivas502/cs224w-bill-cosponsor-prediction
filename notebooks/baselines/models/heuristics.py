"""Heuristic and tabular references for both tasks.

Task 1 (ported from 40_splits.py:_t1_eval): popularity, party+pop, sponsor_hist, rw2hop (RW_BETA = 0.3),
in plain and +stream form, plus sh_hl1 and svd_r512_hl1 (plain only, from `_learn_unit`) and the new
co-support reference `cosup` (depth-consistent, from `features`). The ports use the setup caucus and
108..k-1 counts exactly as the split study did, because the reproduction asserts depend on them.

Task 2 (ported from 40_splits.py:_t2_scores / _grp): prior, chamber_maj, eb_rate (uses policy area: leaky
upper reference), eb_rate_t0safe (policy area hidden for a PA_MISS_T0 share), and the new PA-free
eb_committee. TabGBDT / TabLogReg fit on the tabular bill features x_b.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..counts import counts, rw_parts, rw_scores, rw_stream, stream_mats, svd_factors
from ..data import ADV_WINDOW_DAYS, D_BIG, Corpus, T1Block, t1_blocks
from ..evaluate import block_metrics, new_leg_mask, new_legislator_hits

RW_BETA = 0.3
T1_PLAIN = ("popularity", "party+pop", "sponsor_hist", "rw2hop", "sh_hl1", "svd_r512_hl1")
T1_STREAM = ("popularity", "party+pop", "sponsor_hist", "rw2hop")
T2_NAMES = ("prior", "chamber_maj", "eb_rate", "eb_rate_t0safe", "eb_committee")
REFERENCE_ONLY = {"eb_rate": "uses policy area at t0 = 0: leaky upper reference, not a baseline"}


class HistCounts:
    """Count tables of the training Congresses for the Task-1 heuristic ports (108..k-1 convention)."""

    def __init__(self, corpus: Corpus, hist_congresses):
        self.congresses = tuple(int(c) for c in hist_congresses)
        mask = np.isin(corpus.CD["cong"], list(self.congresses))
        self.pop, self.hist = counts(corpus, mask)
        self.pop1, self.hist1 = counts(corpus, mask, 1)
        self.rwp = rw_parts(self.hist)
        self._svd = None
        self._corpus = corpus

    def svd(self):
        """(U*s, Vt) rank-512 factors of log1p(hl-1 counts) (svd_r512_hl1)."""
        if self._svd is None:
            self._svd = svd_factors(self._corpus, self.congresses, 1)
        return self._svd


def t1_block_scores(corpus: Corpus, block: T1Block, hc: HistCounts, stream: bool,
                    names=None) -> dict[str, np.ndarray]:
    """Score matrices [nb, M] of the Task-1 heuristics on one block (40_splits.py:_t1_eval inner loop).

    stream=True adds the evaluation Congress's events dated <= intro + t0 (Pst, Hst exact daily; rw2hop with
    monthly snapshots). svd_r512_hl1 and sh_hl1 are plain-only references."""
    names = tuple(names or (T1_STREAM if stream else T1_PLAIN))
    g, cand, t0, c = block, block.cand, block.t0, block.congress
    spv = g.spon
    hs, spi = spv >= 0, np.where(spv >= 0, spv, 0)
    same = (block.cparty[None, :] == g.sp_caucus[:, None])
    if stream:
        Pst, Hst = stream_mats(corpus, c, g.day, spv, cand, t0)
    else:
        Pst, Hst = 0.0, 0.0
    tieb = block.tieb
    P = hc.pop[cand][None, :] + Pst + tieb[None, :]
    H = hc.hist[spi][:, cand] * hs[:, None] + Hst
    out = {}
    if "popularity" in names:
        out["popularity"] = np.broadcast_to(P, block.Pos.shape)
    if "party+pop" in names:
        out["party+pop"] = same * 1e7 + P
    if "sponsor_hist" in names:
        out["sponsor_hist"] = H * 1e6 + P
    if "rw2hop" in names:
        if stream:
            rw1, rw2 = rw_stream(corpus, c, g.sm, cand, hc.hist, spi, hs)
        else:
            rw1, rw2 = rw_scores(hc.rwp, spi, hs, cand)
        out["rw2hop"] = rw1 + RW_BETA * rw2 + P * 1e-16
    if "sh_hl1" in names:
        P1 = hc.pop1[cand][None, :] + Pst + tieb[None, :]
        out["sh_hl1"] = (hc.hist1[spi][:, cand] * hs[:, None] + Hst) * 1e6 + P1
    if "svd_r512_hl1" in names:
        US, Vt = hc.svd()
        us, inv = np.unique(spi, return_inverse=True)
        out["svd_r512_hl1"] = (US[us] @ Vt[:, cand])[inv] * hs[:, None] + P * 1e-12
    return out


def t1_heuristics(corpus: Corpus, hist_congresses, eval_congress: int, t0: int, stream: bool,
                  names=None, extra_scores=None, return_scores: bool = False):
    """Per-bill Task-1 metrics of every heuristic on Congress `eval_congress` with count history
    `hist_congresses` (108..k-1 convention). Returns (per_bill {name: [n, 3]}, codes [n]) and, if asked,
    the per-block score matrices {name: [S_block, ...]}.

    extra_scores: optional callable(block) -> {name: S} (e.g. the cosup reference from features)."""
    hc = HistCounts(corpus, hist_congresses)
    blocks = t1_blocks(corpus, eval_congress, t0)
    per, codes, keep = {}, [], {}
    for b in blocks:
        sc = t1_block_scores(corpus, b, hc, stream, names)
        if extra_scores is not None:
            sc.update(extra_scores(b))
        for n, S in sc.items():
            per.setdefault(n, []).append(block_metrics(S, b))
            if return_scores:
                keep.setdefault(n, []).append(np.asarray(S, float))
        codes.append(b.codes)
    per = {n: np.vstack(v) for n, v in per.items()}
    codes = np.concatenate(codes)
    return (per, codes, keep) if return_scores else (per, codes)


# --- Task 2 ------------------------------------------------------------------------------------------
def _grp(corpus: Corpus, tr: pd.DataFrame, te: pd.DataFrame, col: str, stream: bool, w=None):
    """(sum of labels, count) of te's group in training rows (optionally weighted)."""
    # ported from 40_splits.py:_grp
    NG = corpus.NG
    s = np.bincount(tr[col], weights=tr.y if w is None else tr.y * w, minlength=NG[col])[te[col]]
    n = np.bincount(tr[col], weights=w, minlength=NG[col])[te[col]].astype(float)
    if stream:
        g, res = te[col].to_numpy(), te.day.to_numpy() + ADV_WINDOW_DAYS
        o = np.lexsort((res, g))
        key = g[o] * D_BIG + res[o]
        cy = np.r_[0, np.cumsum(te.y.to_numpy()[o])]
        st = np.searchsorted(key, g * D_BIG, "left")
        en = np.searchsorted(key, g * D_BIG + te.day.to_numpy(), "right")
        s, n = s + cy[en] - cy[st], n + en - st
    return s, n


def first_referral_committee(corpus: Corpus, codes: np.ndarray, t0: int = 0) -> np.ndarray:
    """Committee index of each bill's first full-committee referral dated <= intro + t0 (-1 if none)."""
    ref = corpus.referrals
    intro = corpus.BL.day.reindex(ref.code).to_numpy()
    r = ref[(ref.day.to_numpy() <= intro + t0)]
    r = r.sort_values(["code", "day", "comm"], kind="stable").drop_duplicates("code")
    return pd.Series(r.comm.to_numpy(), index=r.code.to_numpy()).reindex(np.asarray(codes)).fillna(-1).astype(int).to_numpy()


def t2_heuristics(corpus: Corpus, train_congresses, eval_rows: pd.DataFrame, t0: int = 0) -> dict[str, np.ndarray]:
    """Task-2 reference scores for eval_rows, fit on _ADV rows of train_congresses (plain protocol).

    prior, chamber_maj, eb_rate (policy area: leaky upper reference), eb_rate_t0safe (40_splits.py:_t2_scores)
    and eb_committee: EB rate by (first referral committee <= cut, chamber, sponsor in majority) shrunk to
    chamber_maj with m = 20."""
    tr = corpus.ADV[corpus.ADV.congress.isin(list(train_congresses))]
    te = eval_rows
    p0 = float(np.average(tr.y))
    hid = te.pa_hidden.to_numpy()
    s, n = _grp(corpus, tr, te, "g1", False)
    s1 = (s + 20 * p0) / (n + 20)
    s, n = _grp(corpus, tr, te, "g2", False)
    eb = (s + 20 * s1) / (n + 20)
    out = {"prior": np.full(len(te), p0), "chamber_maj": s1, "eb_rate": eb, "eb_rate_t0safe": np.where(hid, s1, eb)}
    ctr = first_referral_committee(corpus, tr.code.to_numpy(), t0)
    cte = first_referral_committee(corpus, te.code.to_numpy(), t0)
    ktr = (ctr + 1) * 4 + tr.g1.to_numpy()      # (committee, chamber x majority) group; g1 has <= 4 levels
    kte = (cte + 1) * 4 + te.g1.to_numpy()
    nk = int(max(ktr.max(initial=0), kte.max(initial=0))) + 1
    sc = np.bincount(ktr, weights=tr.y.to_numpy(), minlength=nk)[kte]
    nc = np.bincount(ktr, minlength=nk)[kte].astype(float)
    out["eb_committee"] = (sc + 20 * s1) / (nc + 20)
    return out


def t2_tabular(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """TabGBDT (HistGradientBoosting, DESIGN section 7) and TabLogReg (standardised, C = 1) on x_b."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True,
                                        validation_fraction=0.1, random_state=seed)
    gb.fit(x_train, y_train)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    lr.fit(x_train, y_train)
    return {"TabGBDT": gb.predict_proba(x_eval)[:, 1], "TabLogReg": lr.predict_proba(x_eval)[:, 1]}


# --- Runners: EvalResults for the notebook ---------------------------------------------------------------
def _span(cs) -> str:
    cs = list(cs)
    return f"{min(cs)}-{max(cs)}"


def t1_reference_results(corpus: Corpus, c: int, t0: int = 0, protocols=("stream", "plain"), roles=("val", "test"),
                         H: int = 3, with_cosup: bool = True) -> list:
    """Task-1 heuristic EvalResults of fold c: role 'val' scores Congress c-1 with history 108..c-2, role
    'test' scores c with 108..c-1 (split-study convention). Includes cosup (depth-consistent co-support)."""
    import time

    from ..evaluate import EvalResult
    from ..features import PairContext, cosup_reference
    out = []
    for role in roles:
        ev = c - 1 if role == "val" else c
        hist = list(range(108, ev))
        ctx = PairContext(corpus, ev, H) if with_cosup else None
        for prot in protocols:
            ts = time.time()
            stream = prot == "stream"
            extra = (lambda b, s=stream: {"cosup": cosup_reference(corpus, b, s, H, ctx)}) if with_cosup else None
            per, codes, S = t1_heuristics(corpus, hist, ev, t0, stream, extra_scores=extra, return_scores=True)
            secs = time.time() - ts
            blocks = t1_blocks(corpus, ev, t0)
            is_new = new_leg_mask(corpus, ev)
            for name, pb in per.items():
                out.append(EvalResult("t1", c, role, t0, prot, name, codes, per_bill=pb, kind="heuristic",
                                      sup=_span(hist), runtime_s=secs,
                                      extra={"new_hits": new_legislator_hits(S[name], blocks, is_new)}))
    return out


def t2_reference_results(corpus: Corpus, c: int, roles=("val", "test")) -> list:
    """Task-2 heuristic EvalResults of fold c (fit on 108..c-2 for val, 108..c-1 for test). eb_rate is tagged
    kind='reference_leaky' (it uses the policy area, hidden at t0 = 0)."""
    import time

    from ..data import t2_rows
    from ..evaluate import EvalResult
    out = []
    for role in roles:
        ev = c - 1 if role == "val" else c
        tr = list(range(108, ev))
        rows = t2_rows(corpus, [ev])
        ts = time.time()
        sc = t2_heuristics(corpus, tr, rows)
        secs = time.time() - ts
        for name, s in sc.items():
            out.append(EvalResult("t2", c, role, 0, "plain", name, rows.code.to_numpy(), scores=s, rows=rows,
                                  kind="reference_leaky" if name in REFERENCE_ONLY else "heuristic",
                                  sup=_span(tr), runtime_s=secs))
    return out


def t2_tabular_results(corpus: Corpus, c: int, sup_start: int = 109, H: int = 3, use_title: bool = True,
                       seed: int = 0, roles=("val", "test")) -> list:
    """TabGBDT / TabLogReg EvalResults on x_b (tbill_features at t0 = 0). Val: fit sup_start..c-2 (title fit
    <= c-2), score c-1. Test: fit sup_start..c-1 (title fit <= c-1), score c."""
    import time

    from ..data import t2_rows
    from ..evaluate import EvalResult
    from ..features import fit_title_svd, tbill_features
    out = []
    for role in roles:
        ev = c - 1 if role == "val" else c
        tr = list(range(sup_start, ev))
        title = fit_title_svd(corpus, ev - 1, use_title=use_title)
        ts = time.time()
        X, y, R = {}, {}, {}
        for k in tr + [ev]:
            Xk, names, codes = tbill_features(corpus, k, 0, H, title)
            rows = corpus.ADV[corpus.ADV.congress == k]
            pos = np.searchsorted(codes, rows.code.to_numpy())
            X[k], y[k], R[k] = Xk[pos], rows.y.to_numpy(), rows
        sc = t2_tabular(np.vstack([X[k] for k in tr]), np.concatenate([y[k] for k in tr]), X[ev], seed)
        secs = time.time() - ts
        rows = t2_rows(corpus, [ev])
        for name, s in sc.items():
            out.append(EvalResult("t2", c, role, 0, "plain", name, rows.code.to_numpy(), scores=s, rows=rows,
                                  kind="learned_ref", sup=_span(tr), seed=seed, runtime_s=secs,
                                  config_set="apriori"))
    return out
