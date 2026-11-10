"""Leak-free, depth-consistent node features and Task-1 pair features (DESIGN section 3.3 / 3.4).

Depth rule: every learned history feature of target Congress k reads only W_k = max(108, k-H)..k-1, weights
Congress k' by 0.5 ** (k-1-k') and divides by the summed weights of W_k (a per-Congress average), and
`hist_depth = len(W_k)` enters as a one-hot. Stream columns (protocol "stream") read only Congress-k events
dated <= the bill's cutoff (exact daily) or before the cutoff month's first day (monthly snapshots).

Target-bill features x_b use an allowlist; FORBIDDEN_PATTERNS is asserted against every name list.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from .counts import counts, rw_parts, rw_scores, rw_stream, stream_mats, stream_recip
from .data import (ALL_BILL_TYPES, CHAMBERS, INTMAX, LAW_TYPES, NONVOTING_TYPES, Corpus, T1Block, caucus_pit,
                   hist_window, party_pit)

FEATURE_VERSION = "2"
FORBIDDEN_PATTERNS = ("policy_area", "policy", "subject", "summar", "text_version", "related", "latest_action",
                      "n_cosponsors", "n_committees", "n_actions", "n_subjects", "label_alt", "reported_date")
EB_M = 20.0
TITLE_DIM = 64
CAUCUS3 = ("D", "R", "other")
PHI_HIST = ("sh_w", "recip_w", "pop_w", "rw1_w", "rw2_w", "cosup_sum", "cosup_max", "same_caucus", "same_state",
            "same_both", "cand_majority", "cand_new", "cand_delegate", "hist_depth_2", "hist_depth_3")
# cosup_sum / cosup_st are stored as log1p(sum); every other count column is log1p(count) as in DESIGN 3.4.
PHI_STREAM = ("sh_st", "pop_st", "recip_st", "act30_st", "rw1_st", "rw2_st", "cosup_st")
PHI_ALL = PHI_HIST + PHI_STREAM


def assert_allowed(names) -> None:
    """Raise if any feature name matches FORBIDDEN_PATTERNS (allowlist check, DESIGN check 5)."""
    bad = [n for n in names if any(p in n.lower() for p in FORBIDDEN_PATTERNS)]
    if bad:
        raise AssertionError(f"forbidden feature names: {bad}")


def phi_columns(protocol: str) -> list[int]:
    """Indices of PHI_ALL used by a protocol: "stream" = all, "plain" = the history part only."""
    if protocol == "stream":
        return list(range(len(PHI_ALL)))
    if protocol == "plain":
        return list(range(len(PHI_HIST)))
    raise ValueError(f"unknown protocol {protocol!r}")


# --- Window weights and per-Congress count tables -----------------------------------------------------
def window_weights(k: int, H: int) -> dict[int, float]:
    """{k': w} over W_k with w = 0.5 ** (k-1-k') normalised to sum 1 (depth-normalised, hl = 1)."""
    W = hist_window(k, H)
    w = np.array([0.5 ** (k - 1 - kk) for kk in W])
    return {kk: float(x) for kk, x in zip(W, w / w.sum())}


_PC_CACHE: dict = {}


def _per_congress(corpus: Corpus, kk: int) -> dict:
    """Finished-Congress count tables of Congress kk (all bill types): pop, sponsor x cosponsor, sponsored,
    co-cosponsorship (incl. the sponsor, diagonal zero), cross-caucus counts and EB report stats."""
    key = (corpus.token, int(kk))
    if key in _PC_CACHE:
        return _PC_CACHE[key]
    CD, NL, bills = corpus.CD, corpus.NL, corpus.bills
    m = CD["cong"] == kk
    pop, hist = counts(corpus, m)
    bk = bills[bills.congress == kk]
    bcode = corpus.billix.get_indexer(bk.bill_id)
    bspon = corpus.leg_index.get_indexer(bk.sponsor_bioguide_id.fillna("?"))
    spons = np.bincount(bspon[bspon >= 0], minlength=NL).astype(float)
    # co-cosponsorship: bill x leg incidence (cosponsors + sponsor)
    rows = np.r_[CD["bill"][m], bcode[bspon >= 0]]
    cols = np.r_[CD["leg"][m], bspon[bspon >= 0]]
    ok = cols >= 0
    _, rr = np.unique(rows[ok], return_inverse=True)
    B = sparse.csr_matrix((np.ones(ok.sum()), (rr, cols[ok])), shape=(rr.max() + 1 if len(rr) else 0, NL))
    B.data[:] = 1.0                                   # duplicates were summed on construction -> binary
    co = (B.T @ B).toarray()
    np.fill_diagonal(co, 0.0)
    # setup caucus of Congress kk (finished, legal), for cross-caucus shares
    cau = _setup_caucus(corpus, kk)
    cl, cs = CD["leg"][m], CD["spon"][m]
    okx = (cl >= 0) & (cs >= 0)
    a, b = cau[cl[okx]], cau[cs[okx]]
    known = (a != "") & (b != "")
    cross = (a != b) & known
    cross_n = np.bincount(cl[okx][cross], minlength=NL).astype(float)
    cross_t = np.bincount(cl[okx][known], minlength=NL).astype(float)
    # EB report stats: law-capable, uncensored bills of kk
    adv = corpus.adv[(corpus.adv.congress == kk) & ~corpus.adv.censored]
    asp = corpus.leg_index.get_indexer(corpus.bills.set_index("bill_id").sponsor_bioguide_id.reindex(adv.bill_id).fillna("?"))
    ok2 = asp >= 0
    rep_n = np.bincount(asp[ok2], minlength=NL).astype(float)
    rep_y = np.bincount(asp[ok2], weights=adv.label.to_numpy()[ok2], minlength=NL).astype(float)
    ch_rate = adv.groupby("chamber").label.mean().to_dict()
    ch_n = adv.groupby("chamber").size().to_dict()
    out = dict(pop=pop, hist=hist, spons=spons, co=co, cross_n=cross_n, cross_t=cross_t, rep_n=rep_n, rep_y=rep_y,
               ch_rate=ch_rate, ch_n=ch_n, adv_codes=corpus.billix.get_indexer(adv.bill_id), adv_y=adv.label.to_numpy())
    _PC_CACHE[key] = out
    return out


_CAU_CACHE: dict = {}


def _setup_caucus(corpus: Corpus, kk: int) -> np.ndarray:
    """Setup caucus (members.caucus) per legislator code in Congress kk ('' if no row). Legal for kk < k."""
    key = (corpus.token, int(kk))
    if key not in _CAU_CACHE:
        m = corpus.members[corpus.members.congress == kk].drop_duplicates("bioguide_id")
        out = np.full(corpus.NL, "", dtype=object)
        out[corpus.leg_index.get_indexer(m.bioguide_id)] = m.caucus.to_numpy()
        _CAU_CACHE[key] = out
    return _CAU_CACHE[key]


@dataclass
class Windowed:
    """A depth-normalised windowed count table plus the Congresses (and weights) it read."""
    mat: np.ndarray
    congresses: tuple
    weights: dict


def windowed_counts(corpus: Corpus, k: int, H: int, kind: str) -> Windowed:
    """Depth-normalised hl-1 weighted average over W_k of a per-Congress table: kind in {"pop", "sc"
    (sponsor x cosponsor), "spons", "co" (co-cosponsorship), "cross_n", "cross_t", "rep_n", "rep_y"}."""
    ww = window_weights(k, H)
    mat = None
    for kk, w in ww.items():
        x = _per_congress(corpus, kk)["hist" if kind == "sc" else kind] * w
        mat = x if mat is None else mat + x
    return Windowed(mat, tuple(ww), ww)


# --- Title SVD ---------------------------------------------------------------------------------------
@dataclass
class TitleSVD:
    """TF-IDF (1-2 grams, min_df 5, sublinear, 50k) + TruncatedSVD(64) of introduced official titles, fitted on
    bills of Congresses <= max_congress. `emb` holds the transform of every bill (rows = bill codes)."""
    max_congress: int
    dim: int
    emb: np.ndarray
    fitted_on: tuple = ()

    def transform(self, codes) -> np.ndarray:
        """Title embedding rows of the given bill codes."""
        return self.emb[np.asarray(codes)]


def fit_title_svd(corpus: Corpus, max_congress: int, dim: int = TITLE_DIM, use_title: bool = True) -> TitleSVD:
    """Fit (or load from cache) the title TF-IDF + SVD on Congresses 108..max_congress; zeros if use_title=False."""
    n = len(corpus.billix)
    cong = corpus.bills.congress.to_numpy()
    fitted = tuple(int(c) for c in sorted(set(cong[cong <= max_congress])) if c >= 108)
    if not use_title:
        return TitleSVD(max_congress, dim, np.zeros((n, dim), np.float32), fitted)
    path = corpus.cache_dir / f"title_svd_v{FEATURE_VERSION}_{corpus.info['data_key']}_le{max_congress}_d{dim}.npy"
    if path.exists():
        return TitleSVD(max_congress, dim, np.load(path), fitted)
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    txt = corpus.titles.to_numpy()
    m = (cong <= max_congress) & (cong >= 108)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True, max_features=50000)
    X = vec.fit_transform(txt[m])
    svd = TruncatedSVD(dim, random_state=0).fit(X)
    emb = svd.transform(vec.transform(txt)).astype(np.float32)
    np.save(path, emb)
    return TitleSVD(max_congress, dim, emb, fitted)


# --- Node features -----------------------------------------------------------------------------------
class _Cols:
    """Small helper that accumulates named feature columns."""

    def __init__(self, n: int):
        self.n, self.cols, self.names = n, [], []

    def add(self, name: str, v) -> None:
        """Append one column (scalar values are broadcast)."""
        v = np.asarray(v, dtype=np.float32)
        if v.ndim == 0:
            v = np.full(self.n, float(v), np.float32)
        self.cols.append(v.reshape(self.n))
        self.names.append(name)

    def onehot(self, prefix: str, values, levels) -> None:
        """Append one indicator column per level, named <prefix>_<level>."""
        values = np.asarray(values, dtype=object)
        for lv in levels:
            self.add(f"{prefix}_{lv}", values == lv)

    def done(self) -> tuple[np.ndarray, list[str]]:
        """(float32 matrix, names); asserts the allowlist."""
        X = np.stack(self.cols, 1) if self.cols else np.zeros((self.n, 0), np.float32)
        assert_allowed(self.names)
        return X.astype(np.float32), list(self.names)


def _members_k(corpus: Corpus, k: int) -> pd.DataFrame:
    return corpus.members[corpus.members.congress == k].drop_duplicates("bioguide_id")


_STATES: dict = {}


def state_index(corpus: Corpus) -> pd.Index:
    """Global state order (members.state values, sorted); index len() = the 'unk' row."""
    if corpus.token not in _STATES:
        _STATES[corpus.token] = pd.Index(sorted(corpus.members.state.dropna().unique()))
    return _STATES[corpus.token]


def leg_state(corpus: Corpus, k: int) -> np.ndarray:
    """State per legislator code as of Congress k (its row in k, else its latest earlier row; '' if none)."""
    m = corpus.members[corpus.members.congress <= k].sort_values("congress").drop_duplicates("bioguide_id", keep="last")
    out = np.full(corpus.NL, "", dtype=object)
    out[corpus.leg_index.get_indexer(m.bioguide_id)] = m.state.fillna("").to_numpy()
    return out


def leg_static(corpus: Corpus, k: int, H: int) -> dict:
    """Point-in-time legislator attributes for Congress k used by leg / tbill / pair features.

    `in_office` and `chamber` come from the Congress-k members table, so they mean "serves at some point in
    Congress k" (a mid-Congress entrant is marked from day 0). They are node attributes only: labels and the
    Task-1 eligible pool use the day-resolution point-in-time roster at intro + t0."""
    NL = corpus.NL
    mk = _members_k(corpus, k)
    ik = corpus.leg_index.get_indexer(mk.bioguide_id)
    chamber = np.full(NL, "none", dtype=object)
    chamber[ik] = mk.chamber.fillna("none").to_numpy()
    in_office = np.zeros(NL, bool)
    in_office[ik] = True
    t = corpus.terms[(corpus.terms.congress == k) & corpus.terms.member_type.isin(NONVOTING_TYPES)]
    delegate = np.zeros(NL, bool)
    di = corpus.leg_index.get_indexer(t.bioguide_id)
    delegate[di[di >= 0]] = True
    W = hist_window(k, H)
    mw = corpus.members[corpus.members.congress.isin(W)].drop_duplicates(["bioguide_id", "congress"])
    nwin = np.bincount(corpus.leg_index.get_indexer(mw.bioguide_id), minlength=NL).astype(float)
    cau = caucus_pit(corpus, k, H)
    maj = np.array([corpus.maj(k, ch) if ch != "none" else "" for ch in chamber], dtype=object)
    return dict(party=party_pit(corpus, k), caucus=cau, chamber=chamber, in_office=in_office, delegate=delegate,
                tenure_win=nwin / max(len(W), 1), new=nwin == 0, in_majority=(cau == maj) & (maj != ""),
                state=leg_state(corpus, k), depth=len(W), window=tuple(W))


def leg_features(corpus: Corpus, k: int, H: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(X [NL, F], names, state_idx [NL]) legislator features as of Congress k (DESIGN 3.3 'leg')."""
    st = leg_static(corpus, k, H)
    NL = corpus.NL
    c = _Cols(NL)
    c.onehot("party", np.where(np.isin(st["party"], ["D", "R"]), st["party"], "other"), CAUCUS3)
    c.onehot("caucus", st["caucus"], CAUCUS3)
    c.onehot("chamber", np.array([{"House of Representatives": "house", "Senate": "senate"}.get(x, "none")
                                  for x in st["chamber"]], dtype=object), ("house", "senate", "none"))
    c.onehot("hist_depth", np.full(NL, st["depth"]), (1, 2, 3))
    c.add("delegate", st["delegate"])
    c.add("in_office", st["in_office"])
    c.add("in_majority", st["in_majority"])
    c.add("new_member", st["new"])
    c.add("tenure_win", st["tenure_win"])
    r = leg_rates(corpus, k, H, st)
    for n in ("log_sponsored_pc", "log_cosponsored_pc", "cross_share_eb", "sponsor_eb_rate"):
        c.add(n, r[n])
    X, names = c.done()
    sidx = state_index(corpus).get_indexer(st["state"])
    sidx = np.where(sidx >= 0, sidx, len(state_index(corpus)))
    return X, names, sidx.astype(np.int64)


def leg_rates(corpus: Corpus, k: int, H: int, st: dict | None = None) -> dict:
    """Depth-normalised windowed legislator rates: log1p sponsored / cosponsored per Congress, cross-caucus
    cosponsorship share (EB to 0.5, m = 20), sponsor EB report rate (uncensored, to the chamber rate)."""
    st = st or leg_static(corpus, k, H)
    ww = window_weights(k, H)
    spons = windowed_counts(corpus, k, H, "spons").mat
    pop = windowed_counts(corpus, k, H, "pop").mat
    xn = windowed_counts(corpus, k, H, "cross_n").mat
    xt = windowed_counts(corpus, k, H, "cross_t").mat
    rn = windowed_counts(corpus, k, H, "rep_n").mat
    ry = windowed_counts(corpus, k, H, "rep_y").mat
    ch_rate = chamber_rates(corpus, k, H)
    prior = np.array([ch_rate.get(ch, ch_rate["all"]) for ch in st["chamber"]])
    return dict(log_sponsored_pc=np.log1p(spons), log_cosponsored_pc=np.log1p(pop),
                cross_share_eb=(xn + EB_M * 0.5) / (xt + EB_M), sponsor_eb_rate=(ry + EB_M * prior) / (rn + EB_M),
                window=tuple(ww))


def chamber_rates(corpus: Corpus, k: int, H: int) -> dict:
    """Weighted EB report rate per chamber over W_k (uncensored law-capable bills), plus 'all'."""
    ww = window_weights(k, H)
    num, den = {}, {}
    for kk, w in ww.items():
        pc = _per_congress(corpus, kk)
        for ch, r in pc["ch_rate"].items():
            n = pc["ch_n"][ch]
            num[ch] = num.get(ch, 0.0) + w * r * n
            den[ch] = den.get(ch, 0.0) + w * n
    out = {ch: num[ch] / den[ch] for ch in num}
    out["all"] = sum(num.values()) / max(sum(den.values()), 1e-12)
    return out


def hbill_features(corpus: Corpus, k: int, H: int, title: TitleSVD) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(X, names, codes) for every bill (8 types) of W_k: type, chamber, Congress lag, sponsor in majority,
    log1p final cosponsors, cross-caucus cosponsor share, advancement label + censored / not-law-capable flags,
    title SVD. All of these are finished before Congress k starts."""
    W = hist_window(k, H)
    b = corpus.bills[corpus.bills.congress.isin(W)]
    codes = corpus.billix.get_indexer(b.bill_id)
    n = len(b)
    c = _Cols(n)
    c.onehot("btype", b.bill_type.to_numpy(), ALL_BILL_TYPES)
    c.add("is_senate", (b.chamber == "Senate").to_numpy())
    lag = k - b.congress.to_numpy()
    c.onehot("lag", lag, tuple(range(1, H + 1)))
    spl = corpus.leg_index.get_indexer(b.sponsor_bioguide_id.fillna("?"))
    spc = np.full(n, "", dtype=object)
    for kk in W:
        m = b.congress.to_numpy() == kk
        cau = _setup_caucus(corpus, kk)
        spc[m] = np.where(spl[m] >= 0, cau[np.maximum(spl[m], 0)], "")
    maj = np.array([corpus.maj(kk, ch) for kk, ch in zip(b.congress, b.chamber)], dtype=object)
    c.add("sponsor_majority", (spc == maj) & (spc != ""))
    CD = corpus.CD
    mcd = np.isin(CD["cong"], W)
    ncs = np.bincount(CD["bill"][mcd], minlength=len(corpus.billix))[codes]
    c.add("log_cosp_final", np.log1p(ncs))
    # cross-caucus share of each bill's cosponsors (setup caucus of that Congress)
    bl, lg, cg, sp = CD["bill"][mcd], CD["leg"][mcd], CD["cong"][mcd], CD["spon"][mcd]
    cross = np.zeros(len(bl), bool)
    known = np.zeros(len(bl), bool)
    for kk in W:
        mm = cg == kk
        cau = _setup_caucus(corpus, kk)
        a = np.where(lg[mm] >= 0, cau[np.maximum(lg[mm], 0)], "")
        s = np.where(sp[mm] >= 0, cau[np.maximum(sp[mm], 0)], "")
        known[mm] = (a != "") & (s != "")
        cross[mm] = known[mm] & (a != s)
    xc = np.bincount(bl[cross], minlength=len(corpus.billix))[codes]
    xk = np.bincount(bl[known], minlength=len(corpus.billix))[codes]
    c.add("cross_cosp_share", np.where(xk > 0, xc / np.maximum(xk, 1), 0.0))
    a = corpus.adv.set_index("bill_id")
    lab = a.label.reindex(b.bill_id).fillna(0).to_numpy(float)
    cen = a.censored.reindex(b.bill_id).fillna(False).to_numpy(bool)
    lawc = b.law_capable.to_numpy()
    c.add("adv_label", np.where(lawc & ~cen, lab, 0.0))   # censored -> 0 plus the flag
    c.add("adv_censored", lawc & cen)
    c.add("not_law_capable", ~lawc)
    X, names = c.done()
    T = title.transform(codes)
    names = names + [f"title_{i}" for i in range(T.shape[1])]
    return np.hstack([X, T]).astype(np.float32), names, codes


def comm_features(corpus: Corpus, k: int, H: int) -> tuple[np.ndarray, list[str]]:
    """(X [NC+1, F], names) per full committee (+ 'unk' row): chamber, log1p referrals per Congress over W_k,
    EB report rate over W_k (uncensored law-capable bills, m = 20, shrunk to the chamber rate)."""
    comms = corpus.committees
    NC = len(comms)
    ref = corpus.referrals
    ww = window_weights(k, H)
    ch = ref.drop_duplicates("comm").set_index("comm").comm_chamber.reindex(range(NC)).to_numpy()
    nref = np.zeros(NC + 1)
    rn = np.zeros(NC + 1)
    ry = np.zeros(NC + 1)
    for kk, w in ww.items():
        r = ref[ref.congress == kk].drop_duplicates(["code", "comm"])
        nref[:NC] += w * np.bincount(r.comm, minlength=NC)
        pc = _per_congress(corpus, kk)
        lab = pd.Series(pc["adv_y"], index=pc["adv_codes"])
        r2 = r[r.code.isin(lab.index)]
        rn[:NC] += w * np.bincount(r2.comm, minlength=NC)
        ry[:NC] += w * np.bincount(r2.comm, weights=lab.reindex(r2.code).to_numpy(float), minlength=NC)
    cr = chamber_rates(corpus, k, H)
    prior = np.array([cr.get(x, cr["all"]) for x in ch] + [cr["all"]])
    c = _Cols(NC + 1)
    c.add("is_senate", np.r_[ch == "Senate", False])
    c.add("is_unk", np.r_[np.zeros(NC, bool), True])
    c.add("log_referrals_pc", np.log1p(nref))
    c.add("comm_eb_rate", (ry + EB_M * prior) / (rn + EB_M))
    return c.done()


def tbill_codes(corpus: Corpus, k: int) -> np.ndarray:
    """Codes of the law-capable bills of Congress k (target nodes), sorted."""
    return np.sort(corpus.BL.index[corpus.BL.congress == k].to_numpy())


def day0_cosponsors(corpus: Corpus, codes: np.ndarray, t0: int) -> tuple[np.ndarray, np.ndarray]:
    """(row position, leg code) of every cosponsorship with lag <= t0 of the given bills (the Excl set)."""
    CD = corpus.CD
    pos = pd.Index(codes).get_indexer(CD["bill"])
    m = (pos >= 0) & (CD["lag"] <= t0) & (CD["leg"] >= 0)
    return pos[m], CD["leg"][m]


def referrals_upto(corpus: Corpus, codes: np.ndarray, t0: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row position, committee index, day) of full-committee referrals dated <= intro + t0 (Eastern date)."""
    ref = corpus.referrals
    pos = pd.Index(codes).get_indexer(ref.code.to_numpy())
    intro = corpus.bills.introduced_date.to_numpy().astype("datetime64[D]").astype(np.int64)
    ok = pos >= 0
    cut = intro[ref.code.to_numpy()] + t0
    m = ok & (ref.day.to_numpy() <= cut)
    r = pd.DataFrame({"pos": pos[m], "comm": ref.comm.to_numpy()[m], "day": ref.day.to_numpy()[m]}).drop_duplicates(["pos", "comm"])
    return r.pos.to_numpy(), r.comm.to_numpy(), r.day.to_numpy()


def tbill_features(corpus: Corpus, k: int, t0: int, H: int, title: TitleSVD) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(x_b [nb, F], names, codes) for the law-capable bills of Congress k; every value is known at the bill's
    cutoff intro + t0 (DESIGN 3.3 'tbill' allowlist)."""
    codes = tbill_codes(corpus, k)
    b = corpus.bills.iloc[codes]
    n = len(codes)
    st = leg_static(corpus, k, H)
    rates = leg_rates(corpus, k, H, st)
    sp = corpus.leg_index.get_indexer(b.sponsor_bioguide_id.fillna("?"))
    hs = sp >= 0
    spi = np.maximum(sp, 0)
    c = _Cols(n)
    c.onehot("btype", b.bill_type.to_numpy(), LAW_TYPES)
    c.add("is_senate", (b.chamber == "Senate").to_numpy())
    c.add("by_request", b.sponsor_is_by_request.fillna(False).astype(bool).to_numpy())
    doc = b.day_of_congress.to_numpy(float)
    c.add("day_of_congress", doc / 730.0)
    c.add("second_session", doc >= 365)
    c.onehot("hist_depth", np.full(n, st["depth"]), (1, 2, 3))
    maj = np.array([corpus.maj(k, ch) for ch in b.chamber], dtype=object)
    spc = np.where(hs, st["caucus"][spi], "")
    c.add("sp_majority", (spc == maj) & hs)
    c.add("sp_tenure_win", np.where(hs, st["tenure_win"][spi], 0))
    c.add("sp_eb_rate", np.where(hs, rates["sponsor_eb_rate"][spi], 0))
    c.add("sp_log_sponsored_pc", np.where(hs, rates["log_sponsored_pc"][spi], 0))
    c.add("sp_new", np.where(hs, st["new"][spi], 0))
    rp, lg = day0_cosponsors(corpus, codes, t0)
    ncos = np.bincount(rp, minlength=n)
    lc = st["caucus"][lg]
    xcs = np.bincount(rp, weights=(lc != spc[rp]) & (spc[rp] != ""), minlength=n)
    mjs = np.bincount(rp, weights=lc == maj[rp], minlength=n)
    c.add("log_cosp_t0", np.log1p(ncos))
    c.add("has_cosp_t0", ncos > 0)
    c.add("cosp_t0_cross_share", np.where(ncos > 0, xcs / np.maximum(ncos, 1), 0))
    c.add("cosp_t0_majority_share", np.where(ncos > 0, mjs / np.maximum(ncos, 1), 0))
    cf, _ = comm_features(corpus, k, H)
    ebc = cf[:, 3]
    rp2, cm, _ = referrals_upto(corpus, codes, t0)
    nref = np.bincount(rp2, minlength=n)
    mx = np.zeros(n)
    np.maximum.at(mx, rp2, ebc[cm])
    mean = np.bincount(rp2, weights=ebc[cm], minlength=n) / np.maximum(nref, 1)
    c.add("n_referral_comms_t0", np.log1p(nref))
    c.add("has_referral_t0", nref > 0)
    c.add("ref_comm_eb_max", mx)
    c.add("ref_comm_eb_mean", mean)
    X, names = c.done()
    T = title.transform(codes)
    names = names + [f"title_{i}" for i in range(T.shape[1])]
    assert_allowed(names)
    return np.hstack([X, T]).astype(np.float32), names, codes


# --- Pair features phi(b, l) ------------------------------------------------------------------------
class PairContext:
    """Per (Congress k, H) inputs of the pair features: windowed counts, rw parts, log co-support, static
    legislator attributes. Built once per Congress and reused by both chambers."""

    def __init__(self, corpus: Corpus, k: int, H: int):
        self.k, self.H = k, H
        self.Hw = windowed_counts(corpus, k, H, "sc")
        self.popw = windowed_counts(corpus, k, H, "pop").mat
        self.rwp = rw_parts(self.Hw.mat)
        self.Lco = np.log1p(windowed_counts(corpus, k, H, "co").mat)
        self.st = leg_static(corpus, k, H)
        self.window = self.Hw.congresses


def _cosup(L: np.ndarray, rows: np.ndarray, legs: np.ndarray, nb: int, cand: np.ndarray):
    """(sum, max) over each bill's day-0 cosponsors m of L[m, cand] ([nb, M] each; 0 without cosponsors)."""
    M = len(cand)
    s = np.zeros((nb, M))
    mx = np.zeros((nb, M))
    if len(rows) == 0:
        return s, mx
    o = np.argsort(rows, kind="stable")
    rows, legs = rows[o], legs[o]
    V = L[legs][:, cand]
    np.add.at(s, rows, V)
    starts = np.r_[0, np.flatnonzero(np.diff(rows)) + 1]
    mx[rows[starts]] = np.maximum.reduceat(V, starts, axis=0)
    return s, mx


def _stream_cosup(corpus: Corpus, k: int, block: T1Block, rows: np.ndarray, legs: np.ndarray) -> np.ndarray:
    """cosup_st: sum over day-0 cosponsors m of log1p Co_k[m, l], Congress-k participations (cosponsorships
    and sponsorships) dated before the cutoff month's first day."""
    nb, cand = block.nb, block.cand
    out = np.zeros((nb, len(cand)))
    if len(rows) == 0:
        return out
    CD, NL = corpus.CD, corpus.NL
    m = (CD["cong"] == k) & (CD["day"] < INTMAX) & (CD["leg"] >= 0)
    pb, pl, pd_ = CD["bill"][m], CD["leg"][m], CD["day"][m]
    bk = corpus.bills[corpus.bills.congress == k]
    bs = corpus.leg_index.get_indexer(bk.sponsor_bioguide_id.fillna("?"))
    okb = bs >= 0
    pb = np.r_[pb, corpus.billix.get_indexer(bk.bill_id)[okb]]
    pl = np.r_[pl, bs[okb]]
    pd_ = np.r_[pd_, bk.introduced_date.to_numpy().astype("datetime64[D]").astype(np.int64)[okb]]
    ub, bi = np.unique(pb, return_inverse=True)
    for d in np.unique(block.sm):
        r = block.sm == d
        sel = r[rows]
        if not sel.any():
            continue
        e = pd_ < d
        B = sparse.csr_matrix((np.ones(e.sum()), (bi[e], pl[e])), shape=(len(ub), NL))
        B.data[:] = 1.0                               # binary incidence
        U = np.unique(legs[sel])
        Co = (B[:, U].T @ B).toarray()
        Co[np.arange(len(U)), U] = 0.0                # diagonal zero
        Lsub = np.log1p(Co[:, cand])
        ui = pd.Index(U).get_indexer(legs[sel])
        np.add.at(out, rows[sel], Lsub[ui])
    return out


def pair_features(corpus: Corpus, block: T1Block, protocol: str = "stream", H: int = 3,
                  ctx: PairContext | None = None, debug: dict | None = None) -> tuple[np.ndarray, list[str]]:
    """phi [nb, M, F] float16 for one T1Block (columns PHI_HIST, + PHI_STREAM if protocol == "stream").

    History columns read W_k only (depth-normalised hl-1); stream columns read Congress-k events dated <= the
    bill's cutoff (sh_st, pop_st, recip_st, act30_st) or before the cutoff month (rw*_st, cosup_st)."""
    k, t0, cand = block.congress, block.t0, block.cand
    ctx = ctx or PairContext(corpus, k, H)
    st = ctx.st
    nb, M = block.nb, block.M
    spv = block.spon
    hs = spv >= 0
    spi = np.where(hs, spv, 0)
    Hw = ctx.Hw.mat
    F = {}
    F["sh_w"] = np.log1p(Hw[spi][:, cand]) * hs[:, None]
    F["recip_w"] = np.log1p(Hw[cand][:, spi].T) * hs[:, None]
    F["pop_w"] = np.broadcast_to(np.log1p(ctx.popw[cand])[None, :], (nb, M))
    r1, r2 = rw_scores(ctx.rwp, spi, hs, cand)
    F["rw1_w"], F["rw2_w"] = np.log1p(100 * r1), np.log1p(100 * r2)
    rows, legs = day0_cosponsors(corpus, block.codes, t0)
    cs_, cm_ = _cosup(ctx.Lco, rows, legs, nb, cand)
    F["cosup_sum"], F["cosup_max"] = np.log1p(cs_), cm_          # log1p of the sum (heavy-tailed; monotone)
    cc, sc = st["caucus"][cand], np.where(hs, st["caucus"][spi], "?")
    same_c = (cc[None, :] == sc[:, None]) & hs[:, None]
    cs, ss = st["state"][cand], np.where(hs, st["state"][spi], "?")
    same_s = (cs[None, :] == ss[:, None]) & (cs[None, :] != "") & hs[:, None]
    F["same_caucus"], F["same_state"], F["same_both"] = same_c, same_s, same_c & same_s
    maj = corpus.maj(k, block.chamber)
    for n, v in (("cand_majority", cc == maj), ("cand_new", st["new"][cand]), ("cand_delegate", st["delegate"][cand])):
        F[n] = np.broadcast_to(np.asarray(v, float)[None, :], (nb, M))
    F["hist_depth_2"] = np.full((nb, M), float(st["depth"] == 2))
    F["hist_depth_3"] = np.full((nb, M), float(st["depth"] == 3))
    names = list(PHI_HIST)
    if protocol == "stream":
        Pst, Hst = stream_mats(corpus, k, block.day, spv, cand, t0)
        P30, _ = stream_mats(corpus, k, block.day, spv, cand, t0, cum_day_offset=-30)
        F["sh_st"], F["pop_st"] = np.log1p(Hst), np.log1p(Pst)
        F["recip_st"] = np.log1p(stream_recip(corpus, k, block.day, spv, cand, t0))
        F["act30_st"] = np.log1p(Pst - P30)
        s1, s2 = rw_stream(corpus, k, block.sm, cand, Hw, spi, hs)
        F["rw1_st"], F["rw2_st"] = np.log1p(100 * s1), np.log1p(100 * s2)
        F["cosup_st"] = np.log1p(_stream_cosup(corpus, k, block, rows, legs))
        names += list(PHI_STREAM)
        if debug is not None:
            debug["max_stream_day_minus_cut"] = _max_stream_day(corpus, block)
    phi = np.stack([np.asarray(F[n], np.float32) for n in names], -1).astype(np.float16)
    assert_allowed(names)
    return phi, names


def _max_stream_day(corpus: Corpus, block: T1Block) -> int:
    """Max over bills of (latest Congress-k event day <= cut) - cut; <= 0 by construction (debug check)."""
    from .counts import ev_congress
    ev = ev_congress(corpus, block.congress)
    ix = np.searchsorted(ev["day_d"], block.cut, "right") - 1
    used = np.where(ix >= 0, ev["day_d"][np.maximum(ix, 0)], -10 ** 9)
    return int((used - block.cut).max())


def cosup_reference(corpus: Corpus, block: T1Block, stream: bool, H: int = 3, ctx: PairContext | None = None) -> np.ndarray:
    """The `cosup` heuristic: cosup_sum (+ cosup_st under stream) + 1e-6 * pop (+Pst) + tie-break."""
    ctx = ctx or PairContext(corpus, block.congress, H)
    rows, legs = day0_cosponsors(corpus, block.codes, block.t0)
    s, _ = _cosup(ctx.Lco, rows, legs, block.nb, block.cand)
    pop = ctx.popw[block.cand][None, :]
    if stream:
        s = s + _stream_cosup(corpus, block.congress, block, rows, legs)
        Pst, _ = stream_mats(corpus, block.congress, block.day, block.spon, block.cand, block.t0)
        pop = pop + Pst
    return s + 1e-6 * pop + 1e-9 * block.tieb[None, :]


_PHI_MEM: dict = {}


def block_phi(corpus: Corpus, block: T1Block, H: int, ctx: PairContext | None = None) -> np.ndarray:
    """Full (stream) phi of a block, memoised in RAM and cached as .npy under the cache dir (keyed by the
    block's bills and candidates and FEATURE_VERSION; bump the version after changing any feature). Only the
    corpus returned by load_corpus should use this; tests on modified corpora call pair_features directly."""
    key = (corpus.token, block.congress, block.chamber, block.t0, H)
    if key in _PHI_MEM:
        return _PHI_MEM[key]
    h = hashlib.sha1(block.codes.tobytes() + block.cand.tobytes()).hexdigest()[:10]
    ch = "H" if block.chamber.startswith("House") else "S"
    path = corpus.cache_dir / f"phi_v{FEATURE_VERSION}_{corpus.info['data_key']}_{block.congress}{ch}_t{block.t0}_H{H}_{h}.npy"
    if path.exists():
        phi = np.load(path)
    else:
        phi, _ = pair_features(corpus, block, "stream", H, ctx)
        np.save(path, phi)
    _PHI_MEM[key] = phi
    return phi


# --- Standardisation and drift ----------------------------------------------------------------------
@dataclass
class Standardizer:
    """Column z-scoring fitted on supervision Congresses only; `fitted_on` records them."""
    mean: np.ndarray = None
    std: np.ndarray = None
    fitted_on: tuple = ()
    names: list = field(default_factory=list)

    def fit(self, arrays: list[np.ndarray], congresses, names=None) -> "Standardizer":
        """Fit on row-stacked arrays (each [n_i, F]) of the given supervision Congresses."""
        X = np.vstack([np.asarray(a, np.float64) for a in arrays])
        self.mean = X.mean(0)
        sd = X.std(0)
        self.std = np.where(sd > 1e-8, sd, 1.0)
        self.fitted_on = tuple(int(c) for c in congresses)
        self.names = list(names or [])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """(X - mean) / std as float32."""
        return ((np.asarray(X, np.float64) - self.mean) / self.std).astype(np.float32)


def phi_moments(phis: list[np.ndarray], excls: list[np.ndarray], max_rows: int = 2_000_000, seed: int = 0):
    """(mean, std) of pair features over eligible (non-Excl) pairs of supervision blocks (subsampled)."""
    rng = np.random.default_rng(seed)
    parts = []
    for phi, ex in zip(phis, excls):
        v = phi[~ex]
        parts.append(v)
    X = np.concatenate(parts, 0)
    if len(X) > max_rows:
        X = X[rng.choice(len(X), max_rows, replace=False)]
    X = X.astype(np.float64)
    mu, sd = X.mean(0), X.std(0)
    return mu.astype(np.float32), np.where(sd > 1e-8, sd, 1.0).astype(np.float32)


def drift_table(feats_by_congress: dict, sup_congresses, test_congress: int, names: list[str]) -> pd.DataFrame:
    """Per-Congress means of standardised columns, plus the test Congress's shift in supervision sd units.

    feats_by_congress: {k: standardised array [n_k, F]}. Columns 'shift_sd' = (mean_test - mean_sup) / sd_sup,
    'flag' = '>1sd' / '>0.5sd' / ''."""
    means = pd.DataFrame({k: np.asarray(v, np.float64).mean(0) for k, v in feats_by_congress.items()}, index=names).T
    sup = np.vstack([np.asarray(feats_by_congress[k], np.float64) for k in sup_congresses if k in feats_by_congress])
    mu, sd = sup.mean(0), sup.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    shift = (means.loc[test_congress].to_numpy() - mu) / sd if test_congress in means.index else np.full(len(names), np.nan)
    out = means.T
    out["shift_sd"] = shift
    out["flag"] = np.where(np.abs(shift) > 1, ">1sd", np.where(np.abs(shift) > 0.5, ">0.5sd", ""))
    return out
