"""Counting primitives shared by the heuristic ports and the leak-free pair features.

Ported verbatim from 40_splits.py (`_counts`, `_rownorm`, `_rw_parts`, `_rw_scores`, `_ev_congress`,
`_stream_hist`, `_stream_mats`, `_rw_stream`, `_svd_factors`). All functions take the Corpus explicitly.
Stream functions only read events of the evaluation Congress dated <= each bill's cutoff (exact daily) or
dated before the cutoff month's first day (monthly snapshots).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import D_BIG, INTMAX, Corpus

_EV_CACHE: dict = {}
_SVD_CACHE: dict = {}


def counts(corpus: Corpus, train_mask: np.ndarray, hl: float | None = None, cw: np.ndarray | None = None):
    """(pop [NL], hist [NL, NL]) from the events in train_mask: legislator popularity and sponsor x
    legislator counts, optionally recency-decayed (half-life hl Congresses) and/or weighted per Congress."""
    # ported from 40_splits.py:_counts
    CD, NL = corpus.CD, corpus.NL
    leg, spon, cg = CD["leg"][train_mask], CD["spon"][train_mask], CD["cong"][train_mask]
    w = None if hl is None else 0.5 ** ((cg.max() - cg) / hl)
    if cw is not None:
        w = cw[cg] if w is None else w * cw[cg]
    pop = np.bincount(leg, weights=w, minlength=NL).astype(float)
    ok = spon >= 0
    hist = np.bincount(spon[ok] * NL + leg[ok], weights=None if w is None else w[ok],
                       minlength=NL * NL).reshape(NL, NL).astype(float)
    return pop, hist


def rownorm(X: np.ndarray) -> np.ndarray:
    """Row-normalise a non-negative matrix (rows summing to 0 stay 0)."""  # ported from 40_splits.py:_rownorm
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12)


def rw_parts(hist: np.ndarray):
    """rw2hop matrices: row-normalised sponsor->cosponsor counts and the symmetric walk matrix."""
    # ported from 40_splits.py:_rw_parts
    return rownorm(hist), rownorm(hist + hist.T)


def rw_scores(parts, spi: np.ndarray, has_sp: np.ndarray, cand: np.ndarray):
    """Step-1 and step-2 walk mass from each bill's sponsor to every candidate (nb x M each)."""
    # ported from 40_splits.py:_rw_scores
    Hn, A = parts
    us, inv = np.unique(spi, return_inverse=True)
    R = Hn[us]
    s1, s2 = R[:, cand][inv], (R @ A[:, cand])[inv]
    return s1 * has_sp[:, None], s2 * has_sp[:, None]


def ev_congress(corpus: Corpus, c: int) -> dict:
    """Dated events of Congress c sorted by day, by (sponsor, day), by (cosponsor, day) for streaming."""
    # ported from 40_splits.py:_ev_congress (+ the cosponsor-keyed order used by recip_st)
    key_ = (corpus.token, int(c))
    if key_ in _EV_CACHE:
        return _EV_CACHE[key_]
    CD, NL = corpus.CD, corpus.NL
    m = (CD["cong"] == c) & (CD["day"] < INTMAX)
    leg, spon, day = CD["leg"][m], CD["spon"][m], CD["day"][m]
    o = np.argsort(day, kind="stable")
    ok = spon >= 0
    key = spon[ok] * D_BIG + day[ok]
    o2 = np.argsort(key, kind="stable")
    o3 = np.argsort(day[ok], kind="stable")
    keyc = leg[ok] * D_BIG + day[ok]
    o4 = np.argsort(keyc, kind="stable")
    ev = dict(leg_d=leg[o], day_d=day[o], key=key[o2], leg_k=leg[ok][o2],
              day_e=day[ok][o3], pair_e=(spon[ok] * NL + leg[ok])[o3],
              keyc=keyc[o4], spon_c=spon[ok][o4], bill_d=CD["bill"][m][o])
    _EV_CACHE[key_] = ev
    return ev


def stream_hist(corpus: Corpus, c: int, cut_day: int) -> np.ndarray:
    """Sponsor x legislator counts of Congress c's events dated before cut_day (monthly snapshot)."""
    # ported from 40_splits.py:_stream_hist
    ev = ev_congress(corpus, c)
    n = np.searchsorted(ev["day_e"], cut_day, "left")
    NL = corpus.NL
    return np.bincount(ev["pair_e"][:n], minlength=NL * NL).reshape(NL, NL).astype(float)


def _keyed_counts(keys, vals, who: np.ndarray, thr: np.ndarray, colpos: np.ndarray, nb: int, M: int) -> np.ndarray:
    """For each row r: counts over candidate columns of `vals` among sorted events keyed (who[r], day <= thr[r])."""
    st = np.searchsorted(keys, who * D_BIG, "left")
    en = np.searchsorted(keys, who * D_BIG + thr, "right")
    ln = np.where(who >= 0, en - st, 0)
    rows = np.repeat(np.arange(nb), ln)
    idx = np.repeat(st, ln) + (np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln))
    col = colpos[vals[idx]]
    ok = col >= 0
    return np.bincount(rows[ok] * M + col[ok], minlength=nb * M).reshape(nb, M)


def stream_mats(corpus: Corpus, c: int, day: np.ndarray, spon: np.ndarray, cand: np.ndarray, t0: int,
                cum_day_offset: int = 0):
    """Eval-Congress counts known at each bill's prediction time (events dated <= day + t0): per bill x
    candidate popularity Pst and sponsor-history Hst. `cum_day_offset` shifts the popularity cutoff
    (e.g. -30 gives counts dated <= cut - 30)."""
    # ported from 40_splits.py:_stream_mats (takes day / spon arrays instead of the _BL frame)
    ev = ev_congress(corpus, c)
    NL = corpus.NL
    M, nb = len(cand), len(day)
    colpos = np.full(NL, -1)
    colpos[cand] = np.arange(M)
    thr = np.asarray(day) + t0
    dcol = colpos[ev["leg_d"]]
    okc = dcol >= 0
    ud, di = np.unique(ev["day_d"][okc], return_inverse=True)
    cum = np.bincount(di * M + dcol[okc], minlength=len(ud) * M).reshape(len(ud), M).cumsum(0)
    ix = np.searchsorted(ud, thr + cum_day_offset, "right") - 1
    Pst = np.where((ix >= 0)[:, None], cum[np.maximum(ix, 0)], 0)
    Hst = _keyed_counts(ev["key"], ev["leg_k"], np.asarray(spon), thr, colpos, nb, M)
    return Pst.astype(float), Hst.astype(float)


def stream_recip(corpus: Corpus, c: int, day: np.ndarray, spon: np.ndarray, cand: np.ndarray, t0: int) -> np.ndarray:
    """[nb, M] number of Congress-c bills sponsored by candidate l that the bill's sponsor cosponsored, dated
    <= cut (the role-swapped stream_mats)."""
    ev = ev_congress(corpus, c)
    M, nb = len(cand), len(day)
    colpos = np.full(corpus.NL, -1)
    colpos[cand] = np.arange(M)
    return _keyed_counts(ev["keyc"], ev["spon_c"], np.asarray(spon), np.asarray(day) + t0, colpos, nb, M).astype(float)


def rw_stream(corpus: Corpus, c: int, sm: np.ndarray, cand: np.ndarray, hist: np.ndarray, spi: np.ndarray,
              has_sp: np.ndarray):
    """rw2hop parts with the eval Congress's graph as of the first day of each bill's cutoff month."""
    # ported from 40_splits.py:_rw_stream (takes the sm{t0} array instead of the _BL frame)
    s1, s2 = np.zeros((len(sm), len(cand))), np.zeros((len(sm), len(cand)))
    for d in np.unique(sm):
        r = sm == d
        a, b = rw_scores(rw_parts(hist + stream_hist(corpus, c, d)), spi[r], has_sp[r], cand)
        s1[r], s2[r] = a, b
    return s1, s2


def svd_factors(corpus: Corpus, train: tuple, hl: float | None, rank: int = 512):
    """Rank-`rank` factors (U*s, Vt) of log1p(sponsor x cosponsor counts) of the training Congresses."""
    # ported from 40_splits.py:_svd_factors (SVD_RANKS max = 512)
    key = (corpus.token, tuple(train), hl, rank)
    if key not in _SVD_CACHE:
        _, hist = counts(corpus, np.isin(corpus.CD["cong"], list(train)), hl)
        U, sv, Vt = np.linalg.svd(np.log1p(hist), full_matrices=False)
        _SVD_CACHE[key] = (U[:, :rank] * sv[:rank], Vt[:rank])
    return _SVD_CACHE[key]
