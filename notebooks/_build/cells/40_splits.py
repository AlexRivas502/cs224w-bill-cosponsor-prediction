# %% [markdown]
# ## D. Candidate splits and baseline backtest
#
# Sections A-C test the *preconditions* of a temporal split (similar distributions, node overlap,
# no leakage). This section tests the *consequence*: if we actually train simple, transparent
# baselines under each candidate split, does the split do its job? A split is useful for model
# selection if (1) validation picks the model that is best on test *among close candidates* (regret),
# (2) the test set is large enough to resolve **paired** model-vs-model differences, (3) the validation
# score level predicts the test level (for early stopping and reporting), and (4) its protocol choices
# (expanding vs sliding window, streaming same-Congress history, refit on train + val) do not cost
# accuracy. We also quantify how optimistic the leaky random split is and whether a within-Congress
# split behaves differently, and (D27) what it costs to keep Congress 118 as a locked final test that
# no model choice ever sees. One minimum detectable effect per task (`D_MDE` = 0.01 Recall@20 / 0.01
# AUROC) is used by every selection, optimism and sizing check.
#
# **Protocol.** Every split comes from `SPLIT_DEFS` / `iter_eval_units()` and `SPLIT_VARIANTS` (setup).
# Each validation unit is scored with baselines fit on its training Congresses; each test unit with
# baselines refit on train + validation ("select on val, refit, report test"; D19 checks that refit).
# The baselines are counts, smoothed rates, one graph-structural score (`rw2hop`, a 2-step walk on
# the time-masked cosponsorship graph) and one **learned, transductive embedding model** (`svd`: a
# truncated SVD of the sponsor x cosponsor training matrix, rank and recency chosen on validation), the
# closest cheap stand-in for a GNN with per-legislator embeddings (it can go stale and is cold for new
# members). The goal is to measure the split, not to win: conclusions that hold for count, graph and
# learned models are the robust ones.
#
# **Prediction times.** Task 1 is scored at t0 = 0 and 7 days after introduction (task definition).
# Task 2 has no prediction time in the task definition; **this section assumes prediction at
# introduction** (same t0 as Task 1). At introduction the CRS policy area is usually missing, so the
# deployable Task 2 baseline is `eb_rate_t0safe`; `eb_rate` (policy area always present) is reported as
# the "predict about a month after introduction" variant. Task 2 selection (D12b) and test-set sizing
# (D9) are judged on the t0-safe variant; the policy-area variant is shown for reference.

# %%
# D.0 Protocol variants this section scores on top of SPLIT_DEFS (defined in setup as SPLIT_VARIANTS).
D_VARIANTS = SPLIT_VARIANTS                       # legacy alias; the definitions live in setup
print("Protocol variants (SPLIT_VARIANTS): " + " | ".join(f"{k}: {v['desc']}" for k, v in SPLIT_VARIANTS.items()))

# %% [markdown]
# ### Baselines, metrics and the backtest engine
#
# **Task 1 (cosponsor ranking).** For each evaluable bill (law-capable, window not censored, at least
# one new cosponsor in *(t0, t0+30]*) we rank the eligible same-chamber legislators: voting members
# **in office in the month of introduced_date + t0** (`ROSTER_ASOF_ALL`, the shared point-in-time pool,
# delegates and resident commissioners included as the task definition requires) minus
# the sponsor and anyone who already cosponsored by *t0*.
# `popularity` = legislator's number of cosponsorships in the training events;
# `party+pop` = same caucus as the sponsor first (as-of-Congress caucus, `members.caucus`), then popularity;
# `sponsor_hist` = times the legislator cosponsored this sponsor's bills in the training events (ties
# broken by popularity); `rw2hop` = a **graph-structural** score closer to what a GNN computes: a two-step
# random walk from the sponsor on the time-masked cosponsorship graph, `h1 + 0.3 * h1 @ A`, where `h1` is
# the sponsor's row of the sponsor->cosponsor count matrix (row-normalised) and `A` the row-normalised
# symmetric graph (H + H^T). It rewards "cosponsors of my cosponsors" (multi-hop structure), not just direct
# history; it is a fixed-weight version of the propagation step of LightGCN / SGC-style GNNs. Streaming
# variants use the eval Congress's graph as of the first day of the month of *introduced_date + t0*.
# `svd_r{rank}[_hl1]` = learned embeddings: rank-r truncated SVD of log(1 + sponsor x cosponsor counts) in
# the training events (optionally recency-weighted, half-life 1 Congress), score = reconstructed affinity
# sponsor -> candidate. Legislators absent from training have a zero embedding (transductive cold start).
# Metrics: Recall@10, Recall@20, NDCG@20 per bill, averaged.
# **Task 2 (advancement).** `prior` = constant training rate (AUROC 0.5 by construction);
# `chamber_maj` = smoothed rate by chamber x sponsor-in-majority (`majority_party()`, sponsor caucus);
# `eb_rate` = empirical-Bayes rate by policy area x chamber x sponsor-in-majority, shrunk toward
# `chamber_maj` (20 pseudo-bills); `eb_rate_t0safe` = `eb_rate`, but bills whose policy area would not
# yet be assigned at introduction fall back to `chamber_maj` (rate measured on fresh 119th bills).
# Metrics: AUROC, average precision (AP) and **AP lift** = AP / positive rate (AP alone is not
# comparable across Congresses whose positive rate differs).
# **Uncertainty.** 95% percentile intervals from a **cluster bootstrap by sponsor** (500 resamples;
# bills of the same sponsor share coalitions, so they are resampled together). `ci_low_bill` /
# `ci_high_bill` give the naive bill-level bootstrap for comparison. The same resamples are used for
# every baseline of a unit, so **paired differences** (model A - model B on the same bills) get their own,
# much tighter, CIs.
# **Near-tie families** (for model selection, D11-D12, D21): Task 1 `sponsor_hist` with recency decay
# (half-life 1, 2, 4 Congresses or none) and a same-caucus tie-break variant; Task 1 `rw2hop` with
# second-hop weight in {0.1, 0.3, 1} and recency half-life 1 (weights 0.1, 0.3); Task 2 `eb_rate` with shrinkage m in
# {5, 20, 80} and a variant without sponsor-majority, and the same four members in their t0-safe form
# (`*_t0s`, policy area hidden for fresh bills); Task 1 `svd` with rank in {8, ..., 512} (powers of 2) x recency
# half-life {none, 1} (D22-D25).

# %%
# D.1 Vectorized backtest engine: integer-coded arrays, Task 1 ranking metrics, Task 2 AUROC/AP, cluster bootstrap.
import time
import zlib
from scipy import sparse

_D_B = 500
_D_TOPK = 20
_D_BIG = 10 ** 6                                   # key = group * BIG + day (days since epoch < 1e5)
_D_PRIMARY_T1 = ("cosponsor_t0", "Recall@20")
_D_PRIMARY_T2 = ("advancement", "AUROC")
T1_BASELINES = ["popularity", "party+pop", "sponsor_hist", "rw2hop"]
T2_BASELINES = ["prior", "chamber_maj", "eb_rate", "eb_rate_t0safe"]
T1_FAMILY = {"sh_none": (None, False), "sh_hl4": (4, False), "sh_hl2": (2, False), "sh_hl1": (1, False),
             "sh_none+caucus": (None, True)}            # (recency half-life in Congresses, caucus tie-break)
RW_BETA = 0.3                                         # second-hop weight of the main rw2hop baseline
RW_FAMILY = {"rw_b0.1": (0.1, None), "rw_b0.3": (0.3, None), "rw_b1": (1.0, None),
             "rw_b0.1_hl1": (0.1, 1), "rw_b0.3_hl1": (0.3, 1)}   # (second-hop weight, recency half-life)
T2_FAMILY = {"eb_m5": (5.0, True), "eb_m20": (20.0, True), "eb_m80": (80.0, True), "eb_nomaj_m20": (20.0, False)}
T2S_FAMILY = {f"{k}_t0s": v for k, v in T2_FAMILY.items()}   # same members, policy area hidden for fresh bills
SVD_RANKS = (8, 16, 32, 64, 128, 256, 512)
SVD_FAMILY = {f"svd_r{r}" + ("" if hl is None else f"_hl{hl}"): (r, hl) for hl in (None, 1) for r in SVD_RANKS}
FAMILIES = {"T1 sponsor_hist family": list(T1_FAMILY), "T1 rw2hop family": list(RW_FAMILY),
            "T2 eb_rate family": list(T2_FAMILY), "T2 eb_rate_t0safe family": list(T2S_FAMILY)}
_FAM_OF = {m: f for f, ms in FAMILIES.items() for m in ms}
_INTMAX = np.iinfo(np.int64).max
# ONE minimum detectable effect per task, used by every selection, optimism and test-set sizing check below: the
# smallest test-score difference worth acting on when choosing a model or protocol (a policy choice, not a data
# estimate; see the D7 cell for how it compares with the data). Regret tolerance for selection checks = MDE / 2.
D_MDE = {"cosponsor_t0": 0.01, "cosponsor_t7": 0.01, "advancement": 0.01}
D_TOL = {k: v / 2 for k, v in D_MDE.items()}
_LOCK = max(FOLD_C)                                   # rolling fold reserved as the locked final test (D27)
_DEV = [c for c in FOLD_C if c != _LOCK]              # development folds


def _bs(d):
    """by_split with an explicit key for every split in ALL_SPLITS (None = not judged; reason in na_reason)."""
    return {s: (None if d.get(s) is None else d[s]) for s in ALL_SPLITS}


_NA_FAM = ("near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = S3 folds 117 / 118, "
           "reported as a metric); stream variants, S1pool, S4 and S5 are not run through the selection backtest")

_LEG = pd.Index(pd.unique(pd.concat([members.bioguide_id, cosp.bioguide_id, bills.sponsor_bioguide_id.dropna()])))
_NL = len(_LEG)
_BILLIX = pd.Index(bills.bill_id)
_cdate = cosp["sponsorship_date"]
_CD = dict(leg=_LEG.get_indexer(cosp.bioguide_id), spon=_LEG.get_indexer(cosp.sponsor_bioguide_id),
           cong=cosp.congress.to_numpy(), bill=_BILLIX.get_indexer(cosp.bill_id),
           lag=cosp.lag_days.to_numpy(dtype=float),
           day=np.where(_cdate.notna(), _cdate.values.astype("datetime64[D]").astype(np.int64), _INTMAX))


def _mstart_day(dates):
    """Day number (since epoch) of the first day of each date's month."""
    return pd.to_datetime(dates).dt.to_period("M").dt.to_timestamp().values.astype("datetime64[D]").astype(np.int64)


# Caucus as of each Congress (independents assigned by setup); sponsors without a member row keep their party code.
_CAU = members.drop_duplicates(["bioguide_id", "congress"]).set_index(["bioguide_id", "congress"])["caucus"]
_BL = bills[bills.law_capable & bills.congress.isin(CONGRESSES)][
    ["bill_id", "congress", "chamber", "sponsor_bioguide_id", "sponsor_party", "introduced_date"]].copy()
_BL["sp_caucus"] = pd.MultiIndex.from_arrays([_BL.sponsor_bioguide_id, _BL.congress]).map(_CAU)
_BL["sp_caucus"] = _BL.sp_caucus.where(_BL.sp_caucus.isin(["D", "R"]), _BL.sponsor_party)
_BL["code"] = _BILLIX.get_indexer(_BL.bill_id)
_BL["spon"] = _LEG.get_indexer(_BL.sponsor_bioguide_id.fillna("?"))
_BL["day"] = _BL.introduced_date.values.astype("datetime64[D]").astype(np.int64)
for _t0 in (0, 7):
    _ref = _BL.introduced_date + pd.Timedelta(days=_t0)
    _BL[f"sm{_t0}"] = _mstart_day(_ref)          # streaming graph snapshot: events dated before this month start
    _BL[f"pm{_t0}"] = _mstart_day(_ref.clip(upper=_BL.congress.map(congress_end) - pd.Timedelta(days=ROSTER_REF_DAY + 5)))
_BL = _BL.set_index("code")
_BL["cluster"] = np.where(_BL.spon >= 0, _BL.spon, -1 - np.arange(len(_BL)))   # sponsor-less bills = own cluster
_CENS = {t0: window_targets(t0).assign(code=lambda d: _BILLIX.get_indexer(d.bill_id)).set_index("code")["censored"]
         for t0 in (0, 7)}

# Candidate pools. Legacy: everyone who served in the chamber that Congress. Point-in-time (default): the union
# of the Congress roster and ROSTER_ASOF as candidate columns, masked per bill to members in office in the
# month of introduced_date + t0 (same rule as window_targets(point_in_time=True)).
_mm = members.dropna(subset=["chamber"]).drop_duplicates(["congress", "chamber", "bioguide_id"])
_ROSTER = {k: (_LEG.get_indexer(g.bioguide_id), g.caucus.to_numpy()) for k, g in _mm.groupby(["congress", "chamber"])}
_RA = ROSTER_ASOF_ALL[ROSTER_ASOF_ALL.congress.isin(CONGRESSES)].assign(
    leg=lambda d: _LEG.get_indexer(d.bioguide_id), mday=lambda d: d.month.values.astype("datetime64[D]").astype(np.int64))
assert set(_RA.chamber.unique()) <= set(_mm.chamber.unique()), "ROSTER_ASOF chamber labels differ from members"
_ROSTER_PIT = {}
for (_c, _ch), _g in _RA[_RA.leg >= 0].groupby(["congress", "chamber"]):
    _cand = pd.unique(np.r_[_ROSTER.get((_c, _ch), (np.array([], int),))[0], _g.leg.to_numpy()])
    _cau = pd.MultiIndex.from_arrays([_LEG[_cand], np.full(len(_cand), _c)]).map(_CAU)
    _cau = np.asarray(pd.Series(_cau).fillna("other"))
    _md = np.sort(_g.mday.unique())
    _cp = pd.Series(np.arange(len(_cand)), index=_cand)
    _io = np.zeros((len(_md), len(_cand)), bool)
    _io[np.searchsorted(_md, _g.mday.to_numpy()), _cp.reindex(_g.leg).to_numpy()] = True
    _ROSTER_PIT[(_c, _ch)] = (_cand, _cau, _md, _io)
_D_LOG = {"unmatched_pos": 0, "pos_total": 0, "pos_not_in_office": 0, "pos_pit_total": 0, "pool_pit": [], "pool_legacy": [],
          "stream_s": 0.0}


def _boot_weights(clusters, key, B=_D_B):
    """(B+1, n) resample-count matrix over clusters (row 0 = ones = point estimate). Seeded by key."""
    u, inv = np.unique(np.asarray(clusters), return_inverse=True)
    K = len(u)
    rng = np.random.default_rng(zlib.crc32(str(key).encode()))
    idx = rng.integers(0, K, (B, K))
    cnt = np.bincount((idx + np.arange(B)[:, None] * K).ravel(), minlength=B * K).reshape(B, K)
    return np.vstack([np.ones(len(inv)), cnt[:, inv]]).astype(float)


def _rank_metrics(S, Pos, return_top=False):
    """Per-row Recall@10, Recall@20, NDCG@20 given scores S (ineligible = -inf) and positives Pos
    (and the top-K column indices if asked)."""
    K = _D_TOPK
    top = np.argpartition(-S, K, axis=1)[:, :K]
    top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), axis=1), 1)
    hits = np.take_along_axis(Pos, top, 1).astype(float)
    npos = Pos.sum(1)
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    idcg = np.cumsum(disc)[np.minimum(npos, K) - 1]
    res = np.c_[hits[:, :10].sum(1) / npos, hits.sum(1) / npos, (hits @ disc) / idcg]
    return (res, top) if return_top else res


def _counts(train_mask, hl=None, cw=None):
    """Legislator popularity and sponsor x legislator history from training events, optionally recency-decayed
    (half-life hl Congresses) and/or weighted per Congress (cw: array indexed by Congress number)."""
    leg, spon, cg = _CD["leg"][train_mask], _CD["spon"][train_mask], _CD["cong"][train_mask]
    w = None if hl is None else 0.5 ** ((cg.max() - cg) / hl)
    if cw is not None:
        w = cw[cg] if w is None else w * cw[cg]
    pop = np.bincount(leg, weights=w, minlength=_NL).astype(float)
    ok = spon >= 0
    hist = np.bincount(spon[ok] * _NL + leg[ok], weights=None if w is None else w[ok],
                       minlength=_NL * _NL).reshape(_NL, _NL).astype(float)
    return pop, hist


def _rownorm(X):
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12)


def _rw_parts(hist):
    """rw2hop matrices: row-normalised sponsor->cosponsor counts (step 1) and symmetric walk matrix (step 2)."""
    return _rownorm(hist), _rownorm(hist + hist.T)


def _rw_scores(parts, spi, has_sp, cand):
    """Step-1 and step-2 walk mass from each bill's sponsor to every candidate (nb x M each)."""
    Hn, A = parts
    us, inv = np.unique(spi, return_inverse=True)
    R = Hn[us]
    s1, s2 = R[:, cand][inv], (R @ A[:, cand])[inv]
    return s1 * has_sp[:, None], s2 * has_sp[:, None]


@lru_cache(maxsize=None)
def _ev_congress(c):
    """Dated cosponsorship events of Congress c, sorted two ways (by day; by sponsor then day) for streaming."""
    m = (_CD["cong"] == c) & (_CD["day"] < _INTMAX)
    leg, spon, day = _CD["leg"][m], _CD["spon"][m], _CD["day"][m]
    o = np.argsort(day, kind="stable")
    ok = spon >= 0
    key = spon[ok] * _D_BIG + day[ok]
    o2 = np.argsort(key, kind="stable")
    o3 = np.argsort(day[ok], kind="stable")
    return dict(leg_d=leg[o], day_d=day[o], key=key[o2], leg_k=leg[ok][o2],
                day_e=day[ok][o3], pair_e=(spon[ok] * _NL + leg[ok])[o3])


def _stream_hist(c, cut_day):
    """Sponsor x legislator counts of Congress c's events dated before cut_day (streaming graph snapshot)."""
    ev = _ev_congress(c)
    n = np.searchsorted(ev["day_e"], cut_day, "left")
    return np.bincount(ev["pair_e"][:n], minlength=_NL * _NL).reshape(_NL, _NL).astype(float)


def _stream_mats(c, g, cand, t0):
    """Eval-Congress counts known at each bill's prediction time (events dated <= introduced + t0):
    per bill x candidate popularity increment and sponsor-history increment."""
    ts = time.time()
    ev = _ev_congress(c)
    M, nb = len(cand), len(g)
    colpos = np.full(_NL, -1)
    colpos[cand] = np.arange(M)
    thr = g.day.to_numpy() + t0
    dcol = colpos[ev["leg_d"]]
    okc = dcol >= 0
    ud, di = np.unique(ev["day_d"][okc], return_inverse=True)
    cum = np.bincount(di * M + dcol[okc], minlength=len(ud) * M).reshape(len(ud), M).cumsum(0)
    ix = np.searchsorted(ud, thr, "right") - 1
    Pst = np.where((ix >= 0)[:, None], cum[np.maximum(ix, 0)], 0)
    sp = g.spon.to_numpy()
    st = np.searchsorted(ev["key"], sp * _D_BIG, "left")
    en = np.searchsorted(ev["key"], sp * _D_BIG + thr, "right")
    ln = np.where(sp >= 0, en - st, 0)
    rows = np.repeat(np.arange(nb), ln)
    idx = np.repeat(st, ln) + (np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln))
    col = colpos[ev["leg_k"][idx]]
    ok = col >= 0
    Hst = np.bincount(rows[ok] * M + col[ok], minlength=nb * M).reshape(nb, M)
    _D_LOG["stream_s"] += time.time() - ts
    return Pst.astype(float), Hst.astype(float)


def _rw_stream(c, g, cand, t0, hist, spi, has_sp):
    """rw2hop parts with the eval Congress's graph as of the first day of the month of introduced + t0
    (monthly snapshots: lags exact streaming by < 1 month, never leaks)."""
    ts = time.time()
    s1, s2 = np.zeros((len(g), len(cand))), np.zeros((len(g), len(cand)))
    sm = g[f"sm{t0}"].to_numpy()
    for d in np.unique(sm):
        r = sm == d
        a, b = _rw_scores(_rw_parts(hist + _stream_hist(c, d)), spi[r], has_sp[r], cand)
        s1[r], s2[r] = a, b
    _D_LOG["stream_s"] += time.time() - ts
    return s1, s2


def _t1_eval(train_mask, eval_codes, t0, stream=False, family=False, pit=True, cnt=None, extra=None, hit_sink=None):
    """Per-bill Task 1 metrics for every baseline (and the near-tie families if asked); returns (dict, codes).
    pit=True ranks the point-in-time pool (in office in the month of introduced + t0); False = legacy roster.
    extra(cand=, spi=, hs=, P=, same=, c=, ch=) -> {name: score matrix} adds models; hit_sink = {model: []} collects,
    for the models named in it, one (bill code, legislator, hit@20) row per positive."""
    cnt = dict(cnt or {})
    cnt.setdefault(None, _counts(train_mask))
    hls = {None} | ({v[0] for v in T1_FAMILY.values()} | {v[1] for v in RW_FAMILY.values()} if family else set())
    for hl in hls - set(cnt):
        cnt[hl] = _counts(train_mask, hl)
    rwp = {hl: _rw_parts(cnt[hl][1]) for hl in ({None} | ({v[1] for v in RW_FAMILY.values()} if family else set()))}
    eb = _BL.loc[np.asarray(eval_codes)]
    eb = eb[~_CENS[t0].reindex(eb.index).fillna(True).to_numpy(bool)]
    rows = np.isin(_CD["bill"], eb.index.to_numpy())
    r_bill, r_leg, r_lag = _CD["bill"][rows], _CD["leg"][rows], _CD["lag"][rows]
    out, code_out = {}, []
    rng = np.random.default_rng(t0 + 17)
    for (c, ch), g in eb.groupby(["congress", "chamber"]):
        if (c, ch) not in (_ROSTER_PIT if pit else _ROSTER):
            continue
        if pit:
            cand, cparty, mdays, inoff = _ROSTER_PIT[(c, ch)]
        else:
            cand, cparty = _ROSTER[(c, ch)]
        M, nb = len(cand), len(g)
        colpos = np.full(_NL, -1)
        colpos[cand] = np.arange(M)
        rowpos = pd.Index(g.index).get_indexer(r_bill)
        sel = rowpos >= 0
        rp, cp, lg = rowpos[sel], colpos[r_leg[sel]], r_lag[sel]
        win = (lg > t0) & (lg <= t0 + WINDOW_DAYS)
        Pos = np.zeros((nb, M), bool)
        Excl = np.zeros((nb, M), bool)
        Pos[rp[win & (cp >= 0)], cp[win & (cp >= 0)]] = True
        seen = (lg <= t0) & (cp >= 0)
        Excl[rp[seen], cp[seen]] = True
        spv = g.spon.to_numpy()
        sp = colpos[np.where(spv >= 0, spv, 0)]
        has_sp = (spv >= 0) & (sp >= 0)
        Excl[np.arange(nb)[has_sp], sp[has_sp]] = True
        if not stream and t0 == 0:
            _D_LOG["pos_total"] += int(win.sum())
            _D_LOG["unmatched_pos"] += int((win & (cp < 0)).sum())
        if pit:                                                   # not in office in the bill's pool month
            mi = np.minimum(np.searchsorted(mdays, g[f"pm{t0}"].to_numpy()), len(mdays) - 1)
            okm = mdays[mi] == g[f"pm{t0}"].to_numpy()
            out_pool = okm[:, None] & ~inoff[mi]
            if not stream and t0 == 0:
                _D_LOG["pos_pit_total"] += int((Pos & ~Excl).sum())
                _D_LOG["pos_not_in_office"] += int((Pos & ~Excl & out_pool).sum())
                _D_LOG["pool_pit"].append((~(Excl | out_pool)).sum(1))
            Excl |= out_pool
        elif not stream and t0 == 0:
            _D_LOG["pool_legacy"].append((~Excl).sum(1))
        Pos &= ~Excl
        keep = Pos.sum(1) > 0
        tieb = rng.random(len(cand)) * 0.5                        # random tie-break, fixed per roster
        if not keep.any():
            continue
        Pos, Excl, g = Pos[keep], Excl[keep], g[keep]
        spv = g.spon.to_numpy()
        hs, spi = spv >= 0, np.where(spv >= 0, spv, 0)
        same = (cparty[None, :] == g.sp_caucus.to_numpy()[:, None])
        Pst, Hst = _stream_mats(c, g, cand, t0) if stream else (0.0, 0.0)

        def ph(hl):
            pop, hist = cnt[hl]
            return pop[cand][None, :] + Pst + tieb[None, :], hist[spi][:, cand] * hs[:, None] + Hst

        P, H = ph(None)
        rw1, rw2 = _rw_stream(c, g, cand, t0, cnt[None][1], spi, hs) if stream else _rw_scores(rwp[None], spi, hs, cand)
        tbr = P * 1e-16                                           # popularity tie-break far below any walk mass
        scores = {"popularity": np.broadcast_to(P, Pos.shape), "party+pop": same * 1e7 + P, "sponsor_hist": H * 1e6 + P,
                  "rw2hop": rw1 + RW_BETA * rw2 + tbr}
        if family:
            for name, (hl, tb) in T1_FAMILY.items():
                Pf, Hf = (P, H) if hl is None else ph(hl)
                scores[name] = Hf * 1e8 + same * 1e6 + Pf if tb else Hf * 1e6 + Pf
            rwc = {None: (rw1, rw2)}
            for name, (beta, hl) in RW_FAMILY.items():
                if hl not in rwc:
                    rwc[hl] = _rw_scores(rwp[hl], spi, hs, cand)
                a, b = rwc[hl]
                scores[name] = a + beta * b + tbr
        if extra is not None:
            scores.update(extra(cand=cand, spi=spi, hs=hs, P=P, same=same, c=c, ch=ch))
        pr, pc = np.nonzero(Pos)
        for b, S in scores.items():
            S = np.array(S, dtype=float)
            S[Excl] = -np.inf
            if hit_sink is None or b not in hit_sink:
                out.setdefault(b, []).append(_rank_metrics(S, Pos))
                continue
            m_, top = _rank_metrics(S, Pos, return_top=True)
            out.setdefault(b, []).append(m_)
            intop = np.zeros(Pos.shape, bool)
            intop[np.arange(len(top))[:, None], top] = True
            hit_sink.setdefault(b, []).append(np.c_[g.index.to_numpy()[pr], cand[pc], intop[pr, pc]])
        code_out.append(g.index.to_numpy())
    if hit_sink is not None:
        for b in list(hit_sink):
            hit_sink[b] = np.vstack(hit_sink[b]) if len(hit_sink[b]) else np.zeros((0, 3), np.int64)
    return {b: np.vstack(v) for b, v in out.items()}, np.concatenate(code_out)


def _t1_summarize(per_bill, codes, key, mask=None, baselines=None):
    """Point estimate + cluster (by sponsor) and bill bootstrap CIs, paired across baselines."""
    if mask is not None:
        codes = codes[mask]
    n = len(codes)
    cl = _BL.cluster.reindex(codes).to_numpy()
    Wc = _boot_weights(cl, key)
    Wb = _boot_weights(np.arange(n), (key, "bill"))
    res = []
    for b in (baselines or per_bill):
        v = per_bill[b] if mask is None else per_bill[b][mask]
        ec = (Wc @ v) / Wc.sum(1)[:, None]
        eb = (Wb @ v) / Wb.sum(1)[:, None]
        lo, hi = np.percentile(ec[1:], [2.5, 97.5], axis=0)
        lob, hib = np.percentile(eb[1:], [2.5, 97.5], axis=0)
        for j, mname in enumerate(["Recall@10", "Recall@20", "NDCG@20"]):
            res.append(dict(baseline=b, metric=mname, value=ec[0, j], ci_low=lo[j], ci_high=hi[j],
                            ci_low_bill=lob[j], ci_high_bill=hib[j], n_eval=n, n_clusters=int(len(np.unique(cl)))))
    return res


# --- Task 2 ---
# Share of fresh bills without a CRS policy area at prediction time: measured on the in-progress 119th Congress
# (the only one observed before CRS catches up). Proxy: bills at most a week old at the data snapshot.
# Shared estimate from setup (PA_MISSING_AT_T0 by age bucket at the snapshot).
D_PA_FRESH_DAYS = int(PA_MISSING_AT_T0.loc["0-7d", "age_hi"])
D_PA_MISS_T0, _D_PA_N = PA_MISS_T0, PA_MISS_T0_N
_D_PA_MISS_30 = float(PA_MISSING_AT_T0.loc["8-30d", "share_missing"])

_ADV = adv[~adv.censored & adv.congress.isin(CONGRESSES)].merge(MAJORITY[["congress", "chamber", "majority"]],
                                                                on=["congress", "chamber"], how="left")
_ADV["code"] = _BILLIX.get_indexer(_ADV.bill_id)
_ADV["sp_caucus"] = _BL.sp_caucus.reindex(_ADV.code).to_numpy()
_ADV["maj"] = (_ADV.sp_caucus == _ADV.majority).astype(int)
_ADV["policy_area"] = _ADV.policy_area.fillna("(none)")
_ADV["day"] = _ADV.introduced_date.values.astype("datetime64[D]").astype(np.int64)
_ADV["y"] = _ADV.label.astype(float)
_ADV["cluster"] = _BL.cluster.reindex(_ADV.code).to_numpy()
_ADV["g_ch"] = pd.factorize(_ADV.chamber)[0]
_ADV["g1"] = pd.factorize(_ADV.chamber + "|" + _ADV.maj.astype(str))[0]
_ADV["g2"] = pd.factorize(_ADV.policy_area + "|" + _ADV.chamber + "|" + _ADV.maj.astype(str))[0]
_ADV["g3"] = pd.factorize(_ADV.policy_area + "|" + _ADV.chamber)[0]
_ADV["pa_hidden"] = np.random.default_rng(1180).random(len(_ADV)) < D_PA_MISS_T0   # fixed random mask (t0-safe)
_ADV = _ADV.reset_index(drop=True)
_NG = {k: int(_ADV[k].max()) + 1 for k in ("g_ch", "g1", "g2", "g3")}


def _grp(tr, te, col, stream, w=None):
    """(sum of labels, count) of te's group in training rows (optionally row-weighted by w), plus eval-Congress
    bills already resolved (introduced >= 180 days before the scored bill) when stream=True."""
    s = np.bincount(tr[col], weights=tr.y if w is None else tr.y * w, minlength=_NG[col])[te[col]]
    n = np.bincount(tr[col], weights=w, minlength=_NG[col])[te[col]].astype(float)
    if stream:
        g, res = te[col].to_numpy(), te.day.to_numpy() + ADV_WINDOW_DAYS
        o = np.lexsort((res, g))
        key = g[o] * _D_BIG + res[o]
        cy = np.r_[0, np.cumsum(te.y.to_numpy()[o])]
        st = np.searchsorted(key, g * _D_BIG, "left")
        en = np.searchsorted(key, g * _D_BIG + te.day.to_numpy(), "right")
        s, n = s + cy[en] - cy[st], n + en - st
    return s, n


def _t2_scores(tr, te, stream=False, family=False, w=None):
    """Fit Task 2 baselines (and the near-tie families) on rows tr (optional row weights w, normalised to mean 1),
    score rows te."""
    if w is not None:
        w = np.asarray(w, float) / np.mean(w)
    p0 = np.average(tr.y, weights=w)
    hid = te.pa_hidden.to_numpy()
    s, n = _grp(tr, te, "g1", stream, w)
    s1 = (s + 20 * p0) / (n + 20)
    s, n = _grp(tr, te, "g2", stream, w)
    eb = (s + 20 * s1) / (n + 20)
    out = {"prior": np.full(len(te), p0), "chamber_maj": s1, "eb_rate": eb, "eb_rate_t0safe": np.where(hid, s1, eb)}
    if family:
        sc, nc = _grp(tr, te, "g_ch", stream, w)
        sch = (sc + 20 * p0) / (nc + 20)
        s3, n3 = _grp(tr, te, "g3", stream, w)
        for name, (m, use_maj) in T2_FAMILY.items():
            out[name] = (s + m * s1) / (n + m) if use_maj else (s3 + m * sch) / (n3 + m)
            out[name + "_t0s"] = np.where(hid, s1 if use_maj else sch, out[name])   # fallback without policy area
    return out


def _auc_ap(s, y, W):
    """Weighted AUROC, AP and positive rate for every row of W (tie-aware, grouped by distinct score)."""
    uniq, inv = np.unique(-s, return_inverse=True)
    A = sparse.csr_matrix((np.ones(len(s)), (np.arange(len(s)), inv)), shape=(len(s), len(uniq)))
    Pg = np.asarray((A.T @ (W * y).T).T)
    Ng = np.asarray((A.T @ (W * (1 - y)).T).T)
    P, N = Pg.sum(1), Ng.sum(1)
    cP, cN = np.cumsum(Pg, 1), np.cumsum(Ng, 1)
    auc = (Pg * (N[:, None] - cN + 0.5 * Ng)).sum(1) / (P * N)
    ap = (Pg * cP / np.maximum(cP + cN, 1e-12)).sum(1) / P
    return auc, ap, P / (P + N)


def _t2_summarize(scores, te, key, baselines=None):
    y = te.y.to_numpy()
    Wc = _boot_weights(te.cluster.to_numpy(), key)
    Wb = _boot_weights(np.arange(len(te)), (key, "bill"))
    res = []
    for b in (baselines or scores):
        auc, ap, prev = _auc_ap(scores[b], y, Wc)
        aucb, apb, prevb = _auc_ap(scores[b], y, Wb)
        for mname, est, estb in (("AUROC", auc, aucb), ("AP", ap, apb), ("AP_lift", ap / prev, apb / prevb)):
            lo, hi = np.percentile(est[1:], [2.5, 97.5])
            lob, hib = np.percentile(estb[1:], [2.5, 97.5])
            res.append(dict(baseline=b, metric=mname, value=est[0], ci_low=lo, ci_high=hi, ci_low_bill=lob,
                            ci_high_bill=hib, n_eval=len(te), n_clusters=int(te.cluster.nunique())))
    return res


print(f"engine ready: {_NL:,} legislators coded, {len(_BL):,} law-capable bills in {CONGRESSES[0]}-{CONGRESSES[-1]}, "
      f"{len(_ADV):,} uncensored advancement labels")
print(f"policy area missing among 119th bills <= {D_PA_FRESH_DAYS} days old at the snapshot (setup PA_MISS_T0): "
      f"{D_PA_MISS_T0:.0%} of {_D_PA_N} bills (8-30 days: {_D_PA_MISS_30:.0%}); eb_rate_t0safe hides policy area for that "
      "share of evaluation bills (Task 2 prediction time = introduction)")

# %%
# D.2 Run every unit of iter_eval_units() (+ SPLIT_VARIANTS) through both tasks -> SPLIT_RESULTS.
_t_start = time.time()
_rows, _fam_rows, _CACHE = [], [], {}
_S5_assign = pd.Series(S5_ROLE.to_numpy(), index=_BILLIX.get_indexer(S5_ROLE.index))   # shared draw from setup
_S4_CUTDAYS = {c: {q: int(np.datetime64(d.date(), "D").astype(np.int64)) for q, d in v.items()}
               for c, v in S4_CUT_DATES.items()}


def _unit(train, ev, stream=False, pit=True):
    """train: ('congs', tuple) | ('within', c, cutoff_day) | ('random', excluded_roles)
       ev:    ('congs', tuple) | ('window', c, lo_day, hi_day) | ('random', role).
    Returns per-bill Task 1 metrics and Task 2 scores (cached; family members only for by-Congress units)."""
    key = (train, ev, stream, pit)
    if key in _CACHE:
        return _CACHE[key]
    fam = train[0] == "congs" and not stream and pit
    if train[0] == "congs":
        tmask = np.isin(_CD["cong"], train[1])
        tr2 = _ADV[_ADV.congress.isin(train[1])]
    elif train[0] == "within":
        c, cut = train[1], train[2]
        tmask = (_CD["cong"] < c) | ((_CD["cong"] == c) & (_CD["day"] < cut))
        tr2 = _ADV[(_ADV.congress < c) | ((_ADV.congress == c) & (_ADV.day + ADV_WINDOW_DAYS <= cut))]
    else:
        held = _S5_assign[_S5_assign.isin(train[1])].index.to_numpy()
        tmask = ~np.isin(_CD["bill"], held)
        tr2 = _ADV[~_ADV.code.isin(held)]
    if ev[0] == "congs":
        codes = _BL.index[_BL.congress.isin(ev[1])].to_numpy()
    elif ev[0] == "window":
        codes = _BL.index[(_BL.congress == ev[1]) & (_BL.day >= ev[2]) & (_BL.day < ev[3])].to_numpy()
    else:
        codes = _S5_assign.index[_S5_assign == ev[1]].to_numpy()
    out = {"fam": fam}
    cnt = {None: _counts(tmask)}                                  # shared by both t0
    for t0 in (0, 7):
        out[f"cosponsor_t{t0}"] = _t1_eval(tmask, codes, t0, stream=stream, family=fam and t0 == 0, pit=pit, cnt=cnt)
    if pit:
        te2 = _ADV[_ADV.code.isin(codes)]
        ok2 = len(te2) and te2.label.nunique() == 2
        out["advancement"] = (_t2_scores(tr2, te2, stream=stream, family=fam), te2) if ok2 else None
    _CACHE[key] = out
    return out


def _add(split, fold, role, train, ev, train_label, eval_congress, stream=False):
    u = _unit(train, ev, stream)
    key = (train, ev, stream)
    meta = dict(split=split, fold=fold, train_congresses=train_label, eval_congress=eval_congress, eval_role=role)
    for t0 in (0, 7):
        pb, codes = u[f"cosponsor_t{t0}"]
        for r in _t1_summarize(pb, codes, (key, t0), baselines=T1_BASELINES):
            _rows.append(dict(meta, task=f"cosponsor_t{t0}", **r))
        if u["fam"] and t0 == 0 and split in ("S2", "S3"):
            for r in _t1_summarize(pb, codes, (key, t0), baselines=list(T1_FAMILY) + list(RW_FAMILY)):
                _fam_rows.append(dict(meta, task="cosponsor_t0", family=_FAM_OF[r["baseline"]], **r))
    if u["advancement"] is not None:
        sc, te2 = u["advancement"]
        for r in _t2_summarize(sc, te2, key, baselines=T2_BASELINES):
            _rows.append(dict(meta, task="advancement", **r))
        if u["fam"] and split in ("S2", "S3"):
            for r in _t2_summarize(sc, te2, key, baselines=list(T2_FAMILY) + list(T2S_FAMILY)):
                _fam_rows.append(dict(meta, task="advancement", family=_FAM_OF[r["baseline"]], **r))


def _unit_keys(u):
    """Map an iter_eval_units() dict onto the engine's (train, ev) keys."""
    if u["within"] is not None:
        c, tq, lo, hi = u["within"]
        cd = _S4_CUTDAYS[c]
        return ("within", c, cd[tq]), ("window", c, cd[lo], cd[hi])
    if u.get("random") is not None:
        role = u["random"][1]
        return ("random", ("val", "test") if role == "val" else ("test",)), ("random", role)
    return ("congs", tuple(u["train"])), ("congs", (u["eval_congress"],))


_MIDX = {"Recall@10": 0, "Recall@20": 1, "NDCG@20": 2}


def _fold_unit(split, c, role="test", **kw):
    """Engine unit of fold c: rolling 'S3', sliding 'S2', or 'S3norefit' (test Congress c, fit on 108..c-2 = no refit)."""
    ev = c if role == "test" else c - 1
    tr = {"S3": range(CONGRESSES[0], ev), "S2": range(ev - SLIDE_K, ev), "S3norefit": range(CONGRESSES[0], c - 1)}[split]
    return _unit(("congs", tuple(tr)), ("congs", (ev,)), **kw)


def _pdiff(uA, a, uB, b, task, metric, key):
    """Paired cluster-bootstrap difference score(uA, a) - score(uB, b) on the same evaluation bills
    (same resamples for both): (point, ci_low, ci_high, half-width)."""
    if task == "advancement":
        (sa, te), (sb, teb) = uA[task], uB[task]
        assert np.array_equal(te.code.to_numpy(), teb.code.to_numpy())
        y, W = te.y.to_numpy(), _boot_weights(te.cluster.to_numpy(), key)
        ra, rb = _auc_ap(sa[a], y, W), _auc_ap(sb[b], y, W)
        est = ra[0] - rb[0] if metric == "AUROC" else ra[1] / ra[2] - rb[1] / rb[2]
    else:
        (pa, ca), (pb_, cb) = uA[task], uB[task]
        assert np.array_equal(ca, cb)
        W = _boot_weights(_BL.cluster.reindex(ca).to_numpy(), key)
        est = W @ (pa[a][:, _MIDX[metric]] - pb_[b][:, _MIDX[metric]]) / W.sum(1)
    lo, hi = np.percentile(est[1:], [2.5, 97.5])
    return float(est[0]), float(lo), float(hi), float((hi - lo) / 2)


def _n_test_c(s):
    """Number of test Congresses a by-Congress split (or variant) reports; None for S4 / S5."""
    return len(split_eval_congresses(s, roles=("test",))) if split_kind(s) in BY_CONGRESS_KINDS else None


for u in iter_eval_units():
    tr, ev = _unit_keys(u)
    ec = u["eval_congress"] if u["eval_congress"] is not None else pd.NA
    _add(u["split"], u["fold"], u["role"], tr, ev, u["train_label"], ec)
    if u["split"] + "+stream" in SPLIT_VARIANTS:              # streaming variant: same pools, eval-Congress history
        _add(u["split"] + "+stream", u["fold"], u["role"], tr, ev, u["train_label"] + " + stream", ec, stream=True)
_pv = SPLIT_VARIANTS["S1pool"]
_pa, _pval = _pv["train"], _pv["val"][0]
_add("S1pool", "S1pool", "val", ("congs", tuple(_pa)), ("congs", (_pval,)), _rng_label(_pa), _pval)
_add("S1pool", "S1pool", "test", ("congs", tuple(_pa + [_pval])), ("congs", tuple(_pv["test"])), _rng_label(_pa + [_pval]), pd.NA)

SPLIT_RESULTS = pd.DataFrame(_rows)
SPLIT_RESULTS["eval_congress"] = SPLIT_RESULTS["eval_congress"].astype("Int64")
D_FAMILY_RESULTS = pd.DataFrame(_fam_rows)
_d_secs = time.time() - _t_start
_pp, _pl = np.concatenate(_D_LOG["pool_pit"]), None
print(f"backtest: {SPLIT_RESULTS.groupby(['split', 'fold', 'eval_role']).ngroups} split/fold/role units, "
      f"{len(SPLIT_RESULTS):,} result rows (+{len(D_FAMILY_RESULTS):,} near-tie family rows), {_d_secs:.0f}s "
      f"(streaming {_D_LOG['stream_s']:.0f}s); no subsampling (all evaluable bills used)")
print(f"Task 1 (t0=0) positives missing from the Congress roster (dropped): {_D_LOG['unmatched_pos']:,} of "
      f"{_D_LOG['pos_total']:,} ({_D_LOG['unmatched_pos'] / max(_D_LOG['pos_total'], 1):.2%}); positives outside the "
      f"point-in-time pool (dropped): {_D_LOG['pos_not_in_office']:,} of {_D_LOG['pos_pit_total']:,} "
      f"({_D_LOG['pos_not_in_office'] / max(_D_LOG['pos_pit_total'], 1):.2%}); median eligible pool {np.median(_pp):.0f}")


def _sel(task, metric, baseline=None, df=None, **kw):
    d = SPLIT_RESULTS if df is None else df
    d = d[(d.task == task) & (d.metric == metric)]
    if baseline is not None:
        d = d[d.baseline == baseline]
    for k, v in kw.items():
        d = d[d[k] == v]
    return d


_ov = SPLIT_RESULTS[SPLIT_RESULTS.eval_role == "test"]
_ov = _ov[((_ov.task == _D_PRIMARY_T1[0]) & (_ov.metric == _D_PRIMARY_T1[1]))
          | ((_ov.task == _D_PRIMARY_T2[0]) & _ov.metric.isin(["AUROC", "AP_lift"]) & (_ov.baseline != "prior"))]
print("\nMean test score per split (Task 1 Recall@20 at t0=0; Task 2 AUROC and AP lift = AP / positive rate):")
print(_ov.pivot_table(index=["task", "metric", "baseline"], columns="split", values="value").round(3).to_string())

# %% [markdown]
# ### D1-D2, D20. Does the validation score predict the test score *level*?
#
# The validation Congress stands in for the test Congress when we tune and early-stop. If val and test
# scores differ a lot (Congress *c* is systematically easier or harder than *c-1*), loss-based early
# stopping and score thresholds are tuned on a different target. We report the gap (test - val) per
# split and fold; a gap is *significant* when it exceeds the 95% half-width of the difference under the
# cluster bootstrap (sqrt(hw_val^2 + hw_test^2)). **Calibration:** instead of a fixed relative tolerance,
# a split FAILS when its gaps are larger than sampling noise explains: chi-square test of
# sum (gap / se_diff)^2 over its folds (df = number of folds; a single-fold split reduces to |z| > 1.96).
# Val and test share many sponsors, so se_diff is conservative. We also print the noise-expected mean
# |relative gap| (sqrt(2/pi) * se_diff / val) next to the observed one. Note that a pure **level shift** does not harm model
# *selection* as long as the ranking of candidates holds; rankings and selection regret are measured
# directly in D11-D12, D17 and D21, so the Task 2 level gap (D2) is informational. D20 repeats D1 for
# the graph-structural `rw2hop` baseline. In the chart each row is a split/fold, the two dots are val and
# test with 95% cluster-bootstrap CIs; long segments mean the validation Congress is a poor proxy.

# %%
# D1/D2/D20: validation-to-test gaps per split and fold for the primary metrics of both tasks.
def _gaps(task, metric, df=None):
    d = _sel(task, metric, df=df).pivot_table(index=["split", "fold", "baseline"], columns="eval_role",
                                              values=["value", "ci_low", "ci_high"])
    g = pd.DataFrame({"val": d[("value", "val")], "test": d[("value", "test")],
                      "hw_val": (d[("ci_high", "val")] - d[("ci_low", "val")]) / 2,
                      "hw_test": (d[("ci_high", "test")] - d[("ci_low", "test")]) / 2}).reset_index()
    g["gap"] = g.test - g.val
    g["rel_gap"] = g.gap / g.val
    g["hw_diff"] = np.sqrt(g.hw_val ** 2 + g.hw_test ** 2)
    g["signif"] = g.gap.abs() > g.hw_diff
    return g


_g1 = _gaps(*_D_PRIMARY_T1)
_g1b = _gaps("cosponsor_t7", "Recall@20")
_g2 = _gaps(*_D_PRIMARY_T2)
_t1v = _sel(*_D_PRIMARY_T1, split="S1b", eval_role="val")
_best1 = _t1v[_t1v.baseline != "rw2hop"].sort_values("value").baseline.iloc[-1]   # best count baseline; rw2hop gets D20
_best2 = _sel(*_D_PRIMARY_T2, split="S1b", eval_role="val").sort_values("value").baseline.iloc[-1]


def _gap_table(g, best):
    t = g[g.baseline == best].copy()
    return t.groupby("split").agg(mean_abs_gap=("gap", lambda x: x.abs().mean()),
                                  mean_abs_rel_gap=("rel_gap", lambda x: x.abs().mean()),
                                  max_abs_rel_gap=("rel_gap", lambda x: x.abs().max()),
                                  mean_hw_diff=("hw_diff", "mean"), share_signif=("signif", "mean"),
                                  folds=("fold", "nunique"))


_gt1, _gtr, _gt2 = _gap_table(_g1, _best1), _gap_table(_g1, "rw2hop"), _gap_table(_g2, _best2)
print(f"Mean |test - val| per split (Task 1 Recall@20 at t0=0: '{_best1}', 'rw2hop'; Task 2 AUROC: '{_best2}'):")
print(pd.DataFrame({"folds": _gt1.folds, f"T1 {_best1}": _gt1.mean_abs_gap, "T1 max|rel|": _gt1.max_abs_rel_gap,
                    "T1 rw2hop": _gtr.mean_abs_gap, f"T2 {_best2}": _gt2.mean_abs_gap, "T2 hw_diff": _gt2.mean_hw_diff,
                    "T2 share_signif": _gt2.share_signif}).round(3).to_string())

_dd = _sel(*_D_PRIMARY_T1, baseline=_best1)
_dd = _dd[_dd.split.isin(["S1a", "S1b", "S1b+stream", "S4", "S5"]) | ((_dd.split == "S2") & _dd.fold.isin(["c=117", "c=118"]))]
_dd = _dd.assign(row=_dd.split + " " + _dd.fold.where(_dd.split.isin(["S2", "S4"]), ""))[
    ["row", "eval_role", "value", "ci_low", "ci_high"]]
_DUMBBELL_JS = r"""
const {width, height, title, subtitle, xlabel} = opts;
const m = {top: 48, right: 24, bottom: 52, left: 110};
const svg = d3.select(el).append('svg').attr('width', width).attr('height', height);
viz.title(svg, title, subtitle);
const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
const W = width - m.left - m.right, H = height - m.top - m.bottom;
const rows = Array.from(new Set(data.map(d => d.row)));
const ys = d3.scaleBand(rows, [0, H]).padding(0.3);
const xs = d3.scaleLinear([d3.min(data, d => d.ci_low), d3.max(data, d => d.ci_high)], [0, W]).nice();
const col = {val: viz.palette[0], test: viz.palette[1]};
g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xs).ticks(6).tickSize(-H).tickFormat(d3.format('.2f')))
  .call(a => a.selectAll('line').attr('stroke', viz.grid)).call(a => a.select('.domain').remove())
  .call(a => a.selectAll('text').attr('fill', viz.ink2).style('font', '11px sans-serif'));
g.append('g').call(d3.axisLeft(ys).tickSize(0)).call(viz.axis).call(a => a.select('.domain').remove());
g.append('text').attr('x', W).attr('y', H + 32).attr('text-anchor', 'end').attr('fill', viz.ink2).style('font', '11px sans-serif').text(xlabel);
rows.forEach(r => {
  const v = data.find(d => d.row === r && d.eval_role === 'val'), t = data.find(d => d.row === r && d.eval_role === 'test');
  if (v && t) g.append('line').attr('x1', xs(v.value)).attr('x2', xs(t.value)).attr('y1', ys(r) + ys.bandwidth() / 2)
    .attr('y2', ys(r) + ys.bandwidth() / 2).attr('stroke', '#b5b4ae').attr('stroke-width', 2);
});
const tip = viz.tooltip(el);
const off = d => ys(d.row) + ys.bandwidth() / 2 + (d.eval_role === 'val' ? -3 : 3);
g.selectAll(null).data(data).join('line').attr('x1', d => xs(d.ci_low)).attr('x2', d => xs(d.ci_high))
  .attr('y1', off).attr('y2', off).attr('stroke', d => col[d.eval_role]).attr('stroke-width', 1.5).attr('opacity', 0.6);
g.selectAll(null).data(data).join('circle').attr('cx', d => xs(d.value)).attr('cy', off).attr('r', 5)
  .attr('fill', d => col[d.eval_role]).attr('stroke', '#fff').attr('stroke-width', 1.5)
  .on('mousemove', (e, d) => tip.show(`<b>${d.row} · ${d.eval_role}</b><br>${d3.format('.3f')(d.value)} [${d3.format('.3f')(d.ci_low)}, ${d3.format('.3f')(d.ci_high)}]`, e))
  .on('mouseleave', () => tip.hide());
viz.legend(svg, [['validation', col.val], ['test', col.test]], m.left, height - 10);
"""
show_d3(_DUMBBELL_JS, _dd, height=60 + 34 * _dd.row.nunique() + 50, width=720,
        opts=dict(title=f"Validation vs test Recall@20 ({_best1}, t0=0) per split",
                  subtitle="dots = point estimate, whiskers = 95% cluster-bootstrap CI (by sponsor); S5 is the leaky random split",
                  xlabel="Recall@20"), name="D_val_test_dumbbell")

# Per-fold gaps of the one-step-ahead protocol (S3 folds; S1a/S1b are the folds c=117 and c=118).
_s3g1 = _g1[(_g1.split == "S3") & (_g1.baseline == _best1)].set_index("fold")
_s3g1b = _g1b[(_g1b.split == "S3") & (_g1b.baseline == _best1)].set_index("fold")
_s3gr = _g1[(_g1.split == "S3") & (_g1.baseline == "rw2hop")].set_index("fold")
_s3g2 = _g2[(_g2.split == "S3") & (_g2.baseline == _best2)].set_index("fold")
_pf = pd.DataFrame({"T1_val": _s3g1.val, "T1_test": _s3g1.test, "T1_gap": _s3g1.gap, "T1_hw_diff": _s3g1.hw_diff,
                    "T1rw_gap": _s3gr.gap, "T2_val": _s3g2.val, "T2_test": _s3g2.test, "T2_gap": _s3g2.gap,
                    "T2_hw_diff": _s3g2.hw_diff})
print(f"\nPer-fold gaps, S3 (test c, val c-1): Task 1 Recall@20 '{_best1}' and rw2hop (t0=0), Task 2 AUROC '{_best2}':")
print(_pf.round(3).to_string())


def _gap_list(g, best, splits=("S1a", "S1b")):
    t = g[(g.baseline == best) & g.split.isin(splits)]
    return ", ".join(f"{r.split} {r.gap:+.3f}{' (signif.)' if r.signif else ''}" for r in t.itertuples())


def _gap_test(g, b):
    """Per split: chi-square test that val-test gaps are sampling noise (sum z^2, df = folds), observed and
    noise-expected mean |relative gap|."""
    t = g[(g.baseline == b) & g.split.isin(ALL_SPLITS)].copy()
    t["z2"] = (t.gap / (t.hw_diff / 1.96)) ** 2
    t["exp_rel"] = np.sqrt(2 / np.pi) * (t.hw_diff / 1.96) / t.val
    r = t.groupby("split").agg(folds=("z2", "size"), chi2=("z2", "sum"), mean_abs_rel=("rel_gap", lambda x: x.abs().mean()),
                               noise_rel=("exp_rel", "mean"))
    r["p"] = stats.chi2.sf(r.chi2, r.folds)
    r["ok"] = r.p >= 0.05
    return r


def _gap_check(cid, cond, g, b, splits=None, quantity=None, extra="", kind="validity", t0_task2=None, na=None):
    r = _gap_test(g, b)
    if splits is not None:
        r = r[r.index.isin(splits)]
    main = "S3+stream" if "S3+stream" in r.index and splits is not None else "S3"
    m = r.loc[main]
    record_check("D", cid, cond, bool(m.ok), metric=float(m.mean_abs_rel), quantity=quantity, kind=kind, t0_task2=t0_task2,
                 threshold="FAIL if gaps exceed cluster-bootstrap noise: chi-square p < 0.05 (sum of z^2 over the split's folds)",
                 by_split=_bs({k: bool(v) for k, v in r.ok.items()}),
                 na_reason=na or "this model / protocol is not scored with a val and a test unit under that split",
                 detail=f"'{b}', {main}: mean |rel gap| {m.mean_abs_rel:.3f} vs {m.noise_rel:.3f} expected from sampling noise alone, "
                        f"chi2 {m.chi2:.1f} on {int(m.folds)} df, p={m.p:.3f}. Per split p: "
                        + ", ".join(f"{k} {v:.2f}" for k, v in r.p.items()) + ". " + extra)
    return r


_gt_d1 = _gap_check("D1", "Task 1: validation Congress predicts the test score level (val-test gaps within sampling noise, S3 folds)",
                    _g1, _best1, quantity="val_test_gap",
                    extra=f"t0=7 mean |rel gap| {_s3g1b.rel_gap.abs().mean():.3f}; S1 gaps: {_gap_list(_g1, _best1)}. A gap beyond "
                          "noise means early stopping / thresholds tune for a different Congress than the one tested.")
_d1 = float(_s3g1.rel_gap.abs().mean())
_d2 = float(_s3g2.gap.abs().mean())
_d2s = float(_s3g2.signif.mean())
record_check("D", "D2", "Task 2: validation AUROC level vs test AUROC level (mean |val-test gap| over one-step-ahead folds; informational)",
             None, metric=_d2, threshold="informational: level shifts do not harm selection if rankings hold (see D12/D17)",
             by_split=_bs({k: round(float(v), 4) for k, v in _gt2.mean_abs_gap.items() if k in ALL_SPLITS}), kind="drift",
             t0_task2=30, na_reason="no val / test pair under that split",
             detail=f"'{_best2}' over S3 folds: {_d2s:.0%} of folds have |gap| beyond the 95% cluster-bootstrap half-width of the "
                    f"difference (mean hw_diff {_s3g2.hw_diff.mean():.3f}); largest gap at {_s3g2.gap.abs().idxmax()} "
                    f"({_s3g2.gap.abs().max():.3f}). S1 gaps: {_gap_list(_g2, _best2)}. Task 2 levels move from Congress to "
                    "Congress (base-rate drift, section A), so a single val Congress mis-states the expected test AUROC; this "
                    "matters for reporting and calibration, not for picking models (D12/D17 test rankings directly). Report "
                    "Task 2 scores per test Congress. t0-safe variant (prediction at introduction): mean |gap| "
                    f"{_g2[(_g2.split == 'S3') & (_g2.baseline == 'eb_rate_t0safe')].gap.abs().mean():.3f}.")
_gap_check("D2b", "Task 2 at introduction (eb_rate_t0safe): val Congress predicts the test AUROC level within noise "
                  "[calibration and reporting only, not model selection]", _g2, "eb_rate_t0safe", kind="drift", t0_task2=0,
           extra="Scope: thresholds, calibration and the expected score level, NOT the choice of model (rankings transfer: "
                 "D12b). A FAIL under a split means Task-2 decision thresholds / probabilities tuned on its val Congress do not "
                 "carry over: recalibrate on the eval Congress's already-resolved bills (streaming, labels known 180 days after "
                 "introduction) and report Task-2 scores per test Congress.")
_gap_check("D20", "Task 1, graph-structural rw2hop: validation Congress predicts the test score level (gaps within noise, S3)",
           _g1, "rw2hop", extra=f"S1 gaps: {_gap_list(_g1, 'rw2hop')}. rw2hop behaves almost exactly like sponsor_hist (D9), so "
                                "the learned-embedding repeat of this check is D22.")

# %% [markdown]
# ### D3-D4. How much does the score *level* move from one Congress to the next?
#
# The rolling-origin folds (S3) repeat the one-step-ahead protocol for every test Congress from
# 112 to 118. If a fixed baseline's score swings across folds more than sampling noise allows
# (between-fold std vs the cluster-bootstrap standard error, hw/1.96), then a single test Congress
# (as in S1) gives a Congress-specific number: report scores per Congress and their mean over rolling
# folds. This concerns the *reported level*; comparisons between models on the same test bills are
# paired and much more stable (D9). Lines are per baseline; shaded bands are 95% cluster-bootstrap CIs.

# %%
# D3/D4: fold-to-fold variability of test scores across rolling-origin folds (S3), both tasks.
_BAND_JS = r"""
const {width, height, title, subtitle, ylabel, yfmt} = opts;
const m = {top: 48, right: 120, bottom: 52, left: 56};
const svg = d3.select(el).append('svg').attr('width', width).attr('height', height);
viz.title(svg, title, subtitle);
const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
const W = width - m.left - m.right, H = height - m.top - m.bottom;
const keys = Array.from(new Set(data.map(d => d.series)));
const color = d3.scaleOrdinal(keys, viz.palette);
const xs = d3.scaleLinear(d3.extent(data, d => d.x), [0, W]);
const ys = d3.scaleLinear([d3.min(data, d => d.lo), d3.max(data, d => d.hi)], [H, 0]).nice();
g.append('g').call(d3.axisLeft(ys).ticks(5).tickSize(-W).tickFormat(d3.format(yfmt)))
  .call(a => a.selectAll('line').attr('stroke', viz.grid)).call(a => a.select('.domain').remove())
  .call(a => a.selectAll('text').attr('fill', viz.ink2).style('font', '11px sans-serif'));
g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xs).ticks(new Set(data.map(d => d.x)).size).tickFormat(d3.format('d'))).call(viz.axis);
g.append('text').attr('x', -m.left + 4).attr('y', -8).attr('fill', viz.ink2).style('font', '11px sans-serif').text(ylabel);
g.append('text').attr('x', W).attr('y', H + 32).attr('text-anchor', 'end').attr('fill', viz.ink2).style('font', '11px sans-serif').text('test Congress');
const tip = viz.tooltip(el);
keys.forEach(k => {
  const rows = data.filter(d => d.series === k).sort((a, b) => a.x - b.x);
  g.append('path').datum(rows).attr('fill', color(k)).attr('opacity', 0.15)
    .attr('d', d3.area().x(d => xs(d.x)).y0(d => ys(d.lo)).y1(d => ys(d.hi)));
  g.append('path').datum(rows).attr('fill', 'none').attr('stroke', color(k)).attr('stroke-width', 2)
    .attr('d', d3.line().x(d => xs(d.x)).y(d => ys(d.y)));
  g.selectAll(null).data(rows).join('circle').attr('cx', d => xs(d.x)).attr('cy', d => ys(d.y)).attr('r', 4)
    .attr('fill', color(k)).attr('stroke', '#fff').attr('stroke-width', 1.5)
    .on('mousemove', (e, d) => tip.show(`<b>${k} · ${d.x}</b><br>${d3.format(yfmt)(d.y)} [${d3.format(yfmt)(d.lo)}, ${d3.format(yfmt)(d.hi)}]<br>n=${d.n}`, e))
    .on('mouseleave', () => tip.hide());
  const last = rows[rows.length - 1];
  g.append('text').attr('x', xs(last.x) + 8).attr('y', ys(last.y) + 4).attr('fill', viz.ink2).style('font', '11px sans-serif').text(k);
});
viz.legend(svg, keys.map(k => [k, color(k)]), m.left, height - 10);
"""


def _band(task, metric, split="S3", baselines=None, label=None):
    d = _sel(task, metric, split=split, eval_role="test")
    if baselines:
        d = d[d.baseline.isin(baselines)]
    return pd.DataFrame({"x": d.eval_congress.astype(int), "y": d.value, "lo": d.ci_low, "hi": d.ci_high,
                         "n": d.n_eval, "series": d.baseline if label is None else label}).reset_index(drop=True)


show_d3(_BAND_JS, _band(*_D_PRIMARY_T1), height=330, width=720,
        opts=dict(title="Task 1 test Recall@20 across rolling-origin folds (S3, t0=0)",
                  subtitle="each point: baselines refit on 108..c-1, evaluated on Congress c; bands = cluster-bootstrap 95% CI",
                  ylabel="Recall@20", yfmt=".2f"), name="D_t1_rolling")
show_d3(_BAND_JS, _band(*_D_PRIMARY_T2, baselines=["chamber_maj", "eb_rate", "eb_rate_t0safe"]), height=330, width=720,
        opts=dict(title="Task 2 test AUROC across rolling-origin folds (S3)",
                  subtitle="prior omitted (AUROC = 0.5 by construction); eb_rate_t0safe hides policy area for fresh bills",
                  ylabel="AUROC", yfmt=".2f"), name="D_t2_rolling")


def _fold_var(task, metric, split="S3"):
    d = _sel(task, metric, split=split, eval_role="test").sort_values("eval_congress")
    out = []
    for b, g in d.groupby("baseline"):
        hw = ((g.ci_high - g.ci_low) / 2).to_numpy()
        v = g.value.to_numpy()
        out.append(dict(baseline=b, mean=v.mean(), std=v.std(ddof=1), cv=v.std(ddof=1) / v.mean(),
                        min=v.min(), max=v.max(), mean_ci_hw=hw.mean(),
                        std_over_se=v.std(ddof=1) / (hw.mean() / 1.96) if hw.mean() > 0 else np.nan))
    return pd.DataFrame(out).set_index("baseline")


_fv1, _fv2 = _fold_var(*_D_PRIMARY_T1), _fold_var(*_D_PRIMARY_T2)
print("Test score across S3 folds (T1 Recall@20 at t0=0, T2 AUROC; std_over_se = between-fold std / cluster-bootstrap SE):")
print(pd.concat({"T1": _fv1, "T2": _fv2.drop(index="prior")}).round(3).to_string())
_r = _fv1.loc[_best1]
_r2 = _fv2.loc[_best2]
record_check("D", "D3", "Task 1: test score level across rolling folds (CV of best baseline across S3 folds; informational)",
             None, metric=float(_r.cv), threshold="informational (CV <= 0.10 = level stable)", quantity="score_stability",
             detail=f"'{_best1}' Recall@20 ranges {_r['min']:.3f}-{_r['max']:.3f} over test Congresses {FOLD_C[0]}-{FOLD_C[-1]}; "
                    f"between-fold std is {_r.std_over_se:.1f}x the cluster-bootstrap SE (>1.5x = real Congress-to-Congress "
                    f"change beyond sampling noise); rw2hop CV {_fv1.loc['rw2hop', 'cv']:.3f}. Affects the reported level only; "
                    "paired model comparisons are sized in D9. Consecutive folds share most training Congresses.")
record_check("D", "D4", "Task 2: test AUROC level across rolling folds (std of best baseline across S3 folds; informational)",
             None, metric=float(_r2["std"]), threshold="informational (std > 0.03 and > 1.5x SE = level moves)", kind="validity",
             t0_task2=30,
             detail=f"'{_best2}' AUROC ranges {_r2['min']:.3f}-{_r2['max']:.3f}; between-fold std = {_r2.std_over_se:.1f}x the "
                    "cluster-bootstrap SE of one fold, so the Task 2 level genuinely changes between Congresses. Implication: report "
                    "Task 2 test scores per Congress and their mean over several rolling folds; a single test Congress (S1) is a "
                    "Congress-specific number. Model comparisons are paired and far more stable (see D9 for the paired sizing).")

# %% [markdown]
# ### D5-D6, D11-D12, D17, D21. Does validation pick the model that is best on test?
#
# The real job of a validation set is choosing among **close** candidates (hyperparameters, decay
# rates, regularization). D11, D12 and D21 use near-tie families and measure, per rolling fold, whether
# the validation-best candidate is also test-best (hit rate), how much test score is lost by trusting
# validation (**regret** = test score of the test-best minus test score of the val-chosen candidate) and
# the Kendall tau between the val and test orderings. Family members are often statistically
# indistinguishable on one Congress, so a regret only counts as **significant** when the paired
# cluster-bootstrap CI of (test-best minus val pick) on that test Congress excludes 0 (this is liberal:
# the test-best is itself chosen on test). A check FAILs only if mean regret > D_TOL = MDE / 2 (0.005) **and** at least
# 2 of 7 folds have significant regret (2+ false alarms out of 7 at the 5% level has probability ~4%).
# D17 repeats the selection using the mean val score of the last three rolling folds. D5-D6 compare
# the main baselines (informational: most pairs are far apart). D12b repeats D12 for the t0-safe Task 2
# family (the deployable one if Task 2 predicts at introduction). **Single-fold splits** (S1a, S1b) cannot
# meet a "2 of 7 folds" rule, so their `by_split` entry is the regret measured on their own fold (a
# metric, not a verdict), and the detail says where it falls in the S3 distribution of per-fold regret.

# %%
# D5/D6 (informational): val vs test ordering of the main baselines; D11/D12/D17/D21: near-tie family selection.
def _order_agree(task, metric, split="S3", baselines=None, df=None):
    d = _sel(task, metric, df=df, split=split)
    if baselines:
        d = d[d.baseline.isin(baselines)]
    piv = d.pivot_table(index=["fold", "eval_role"], columns="baseline", values="value")
    folds = list(piv.index.get_level_values(0).unique())
    rows = []
    for f in folds:
        v, t = piv.loc[(f, "val")], piv.loc[(f, "test")]
        # alternative: pick on the mean val score of this and the 2 previous folds (val Congresses c-3..c-1; no test data)
        prev = [x for x in folds if int(x[2:]) in range(int(f[2:]) - 2, int(f[2:]) + 1)]
        vm = pd.concat([piv.loc[(x, "val")] for x in prev], axis=1).mean(axis=1)
        rows.append(dict(fold=f, same_order=list(v.sort_values().index) == list(t.sort_values().index),
                         tau=stats.kendalltau(v.values, t.values)[0], val_pick=v.idxmax(), pick_3val=vm.idxmax(),
                         test_best=t.idxmax(), regret=t.max() - t[v.idxmax()], regret_3val=t.max() - t[vm.idxmax()],
                         test_spread=t.max() - t.min()))
    return pd.DataFrame(rows).set_index("fold")


def _regret_sig(o, task, metric, split, pick_col="val_pick"):
    """Per fold: half-width and significance of the paired difference (test-best minus pick) on the test Congress."""
    hw, sig = [], []
    for f, r in o.iterrows():
        if r[pick_col] == r.test_best:
            hw.append(0.0), sig.append(False)
            continue
        u = _fold_unit(split, int(f[2:]))
        _, lo, _, h = _pdiff(u, r.test_best, u, r[pick_col], task, metric, ("regret", split, f, task, metric, pick_col))
        hw.append(h), sig.append(lo > 0)
    return np.array(hw), np.array(sig)


_o1 = _order_agree(*_D_PRIMARY_T1)
_o2 = _order_agree(*_D_PRIMARY_T2, baselines=["prior", "chamber_maj", "eb_rate"])
_gapmin1 = _sel(*_D_PRIMARY_T1, split="S3", eval_role="test").pivot_table(index="fold", columns="baseline", values="value")
_close1 = _gapmin1[["rw2hop", "sponsor_hist"]].diff(axis=1).iloc[:, 1]
record_check("D", "D5", "Task 1: main baselines keep their order from validation to test (share of S3 folds; informational)",
             None, metric=float(_o1.same_order.mean()), threshold="informational (see D11/D21 for close candidates)",
             detail=f"Kendall tau(val, test) of the 4 main baselines averages {_o1.tau.mean():.2f}. The closest pair is "
                    f"sponsor_hist vs rw2hop (test difference {_close1.mean():+.3f} on average, sign flips in "
                    f"{(np.sign(_close1) != np.sign(_close1.mean())).mean():.0%} of folds); its paired precision is in D9.")
record_check("D", "D6", "Task 2: main baselines keep their order from validation to test (share of S3 folds; informational)",
             None, metric=float(_o2.same_order.mean()), threshold="informational (candidates far apart)", t0_task2=30,
             detail="prior < chamber_maj < eb_rate by wide margins; see D12 for selection among close candidates.")

_sel_rows, _ORD, _SIGS = [], {}, {}
for fam, task, metric in (("T1 sponsor_hist family", *_D_PRIMARY_T1), ("T1 rw2hop family", *_D_PRIMARY_T1),
                          ("T2 eb_rate family", "advancement", "AUROC"), ("T2 eb_rate family", "advancement", "AP_lift"),
                          ("T2 eb_rate_t0safe family", "advancement", "AUROC")):
    for split in ("S3", "S2"):
        o = _order_agree(task, metric, split=split, df=D_FAMILY_RESULTS[D_FAMILY_RESULTS.family == fam])
        hw = _sel(task, metric, df=D_FAMILY_RESULTS[D_FAMILY_RESULTS.family == fam], split=split, eval_role="test")
        if metric != "AP_lift":
            o["regret_hw"], o["regret_sig"] = _regret_sig(o, task, metric, split)
            o["hw3"], o["sig3"] = _regret_sig(o, task, metric, split, "pick_3val")
        _ORD[(fam, metric, split)] = o
        _sel_rows.append(dict(family=fam, metric=metric, split=split, hit_rate=o.val_pick.eq(o.test_best).mean(),
                              mean_regret=o.regret.mean(), max_regret=o.regret.max(),
                              regret_hw=o.get("regret_hw", pd.Series(np.nan)).mean(),
                              n_sig=int(o.get("regret_sig", pd.Series(0)).sum()), regret_3val=o.regret_3val.mean(),
                              n_sig_3val=int(o.get("sig3", pd.Series(0)).sum()), mean_tau=o.tau.mean(),
                              test_spread=o.test_spread.mean(), test_hw=((hw.ci_high - hw.ci_low) / 2).mean(),
                              picks=", ".join(f"{k}:{v}" for k, v in o.val_pick.value_counts().items())))
_selt = pd.DataFrame(_sel_rows)
print("Near-tie family selection on rolling folds (regret = test-best minus test of val pick; regret_hw = mean paired CI\n"
      "half-width of that difference; n_sig = folds (of 7) where it is significant; *_3val = pick on mean of last 3 val folds):")
print(_selt[_selt.metric != "AP_lift"].drop(columns=["picks"]).round(4).to_string(index=False))
print("Val picks (S3): " + " | ".join(f"{r.family.replace(' family', '')} {r.metric}: {r.picks}"
                                     for r in _selt[_selt.split == "S3"].itertuples()))
_regc = pd.concat([pd.DataFrame({"fold": o.index, "regret": o.regret.values, "family": f"{k[0]} ({k[1]})"})
                   for k, o in _ORD.items() if k[2] == "S3" and k[1] != "AP_lift"])
bar_chart(_regc, x="fold", y="regret", series="family", yfmt=".4f", ylabel="regret",
          title="Model-selection regret per rolling fold (S3, near-tie families)",
          subtitle="test score lost by picking the validation-best candidate instead of the test-best (0 = val picked right)",
          height=300, name="D_selection_regret")


def _sel_verdict(fam, metric, split, col="regret", sig="regret_sig", folds=None):
    o = _ORD[(fam, metric, split)]
    if folds is not None:
        o = o[o.index.isin([f"c={c}" for c in folds])]
    tol = D_TOL["advancement" if fam.startswith("T2") else "cosponsor_t0"]
    return bool(not ((o[col].mean() > tol) and (o[sig].sum() >= 2)))


def _fold_regret(fam, metric, c):
    """Single-Congress splits (S1a = S3 fold 117, S1b = fold 118): the regret on that fold (a metric, not a verdict)."""
    return round(float(_ORD[(fam, metric, "S3")].loc[f"c={c}", "regret"]), 4)


def _fold_note(fam, metric):
    o = _ORD[(fam, metric, "S3")]
    parts = []
    for s_ in ("S1a", "S1b"):
        f = f"c={SPLIT_DEFS[s_]['test'][0]}"
        pct = (o.regret <= o.loc[f, "regret"]).mean()
        parts.append(f"{s_} fold regret {o.loc[f, 'regret']:.4f} ({'significant' if o.loc[f, 'regret_sig'] else 'not significant'}, "
                     f"at the {pct:.0%} point of the 7 S3 fold regrets)")
    return "; ".join(parts)


def _sel_check(cid, fam, metric, cond, unit, t0_task2=None):
    r = _selt[(_selt.family == fam) & (_selt.metric == metric) & (_selt.split == "S3")].iloc[0]
    r2 = _selt[(_selt.family == fam) & (_selt.metric == metric) & (_selt.split == "S2")].iloc[0]
    tol = D_TOL["advancement" if fam.startswith("T2") else "cosponsor_t0"]
    record_check("D", cid, cond, _sel_verdict(fam, metric, "S3"), metric=float(r.mean_regret), kind="validity", t0_task2=t0_task2,
                 threshold=f"FAIL if mean regret > D_TOL = MDE/2 = {tol} {unit} AND regret significant (paired cluster CI) in >= 2 of 7 folds",
                 by_split=_bs({"S3": _sel_verdict(fam, metric, "S3"), "S2": _sel_verdict(fam, metric, "S2"),
                               "S1a": _fold_regret(fam, metric, SPLIT_DEFS["S1a"]["test"][0]),
                               "S1b": _fold_regret(fam, metric, SPLIT_DEFS["S1b"]["test"][0])}), na_reason=_NA_FAM,
                 detail=f"Family {FAMILIES[fam]}: hit rate {r.hit_rate:.0%}, max regret {r.max_regret:.4f}, mean regret "
                        f"{r.mean_regret:.4f} vs mean paired half-width {r.regret_hw:.4f}; significant in {r.n_sig}/7 folds; mean "
                        f"tau(val,test) {r.mean_tau:.2f}; family spread on test {r.test_spread:.3f} vs single-score CI hw "
                        f"{r.test_hw:.3f}. S2 folds: regret {r2.mean_regret:.4f} ({r2.n_sig}/7 significant). Picking on the mean of "
                        f"the last 3 val folds: regret {r.regret_3val:.4f} ({r.n_sig_3val}/7 significant). {_fold_note(fam, metric)} "
                        "(by_split S1a/S1b = that regret, a metric). Tolerance = half the section-wide MDE (smallest improvement worth "
                        "acting on); the significance rule keeps churn among statistically tied members from causing a FAIL.")


_sel_check("D11", "T1 sponsor_hist family", "Recall@20",
           "Task 1: validation picks the test-best among close count-history candidates (regret vs paired noise, S3 folds)", "Recall@20")
_sel_check("D12", "T2 eb_rate family", "AUROC",
           "Task 2: validation picks the test-best among close candidates (regret vs paired noise, S3 folds, AUROC)", "AUROC", t0_task2=30)
_sel_check("D12b", "T2 eb_rate_t0safe family", "AUROC",
           "Task 2 at introduction (t0-safe family, policy area hidden for fresh bills): validation picks the test-best "
           "(regret vs paired noise, S3)", "AUROC", t0_task2=0)
_sel_check("D21", "T1 rw2hop family", "Recall@20",
           "Task 1, graph-structural rw2hop family: validation picks the test-best (regret vs paired noise, S3 folds)", "Recall@20")
_v3 = {(f, m, sp): _sel_verdict(f, m, sp, "regret_3val", "sig3") for f, m in (("T1 sponsor_hist family", "Recall@20"),
       ("T1 rw2hop family", "Recall@20"), ("T2 eb_rate family", "AUROC"), ("T2 eb_rate_t0safe family", "AUROC"))
       for sp in ("S3", "S2")}
_r3 = _selt[(_selt.metric != "AP_lift") & (_selt.split == "S3")].set_index("family")
record_check("D", "D17", "Selecting on the mean of the last 3 rolling val folds picks near-test-best candidates (all families, S3)",
             all(v for k, v in _v3.items() if k[2] == "S3"), metric=float(_r3.regret_3val.max()),
             threshold="per family: FAIL if mean regret > D_TOL (MDE/2 = 0.005) AND significant in >= 2 of 7 folds", kind="validity",
             by_split=_bs({sp: all(v for k, v in _v3.items() if k[2] == sp) for sp in ("S3", "S2")}), na_reason=_NA_FAM,
             detail="Regret with 3-fold val selection (single val in brackets): " + "; ".join(
                 f"{f}: {r.regret_3val:.4f} ({r.n_sig_3val}/7 signif.) [{r.mean_regret:.4f}, {r.n_sig}/7]" for f, r in _r3.iterrows())
             + ". Val Congresses c-3..c-1, each scored with its own one-step-ahead training set, so no test data is used. "
               "Only rolling/sliding splits can do this (S1a/S1b have one val Congress).")

# %% [markdown]
# ### D22-D25. A learned, transductive embedding model: selection, staleness and cold start
#
# Count heuristics cannot overfit and have no per-legislator parameters, so they cannot show the two
# failure modes that matter most for a GNN under a by-Congress split: (1) a **high-capacity model's
# validation choice may not carry over to test** (it can fit Congress-specific structure), and (2)
# **learned per-legislator embeddings go stale** across an election (a member's role changes: chamber or
# party switch, majority status flip, new committee portfolio), which a count heuristic hides. We fit a
# truncated SVD of log(1 + sponsor x cosponsor counts) on each fold's training Congresses (ranks 8-128,
# typical GNN sizes, plus 256 / 512, close to memorised history; each with or without recency weighting:
# 14 candidates; rank = capacity), choose the candidate on the validation Congress and refit on train + val for test, exactly as the S3 protocol prescribes. D22 is
# the val-test level check (as D1), D23 the selection regret (as D11), D24 reports Recall@20 separately
# for positives from **new** legislators (zero embedding) and **returning** ones, and D25 tests
# staleness: among returning legislators, is the learned model's recall loss on members whose role
# changed larger than the count heuristic's loss on the same positives (difference in differences,
# cluster-bootstrap CI)? The member-committee roster is not in the data, so "committee shift" is a proxy:
# cosine similarity < 0.5 between the committee-referral profiles of the member's sponsored bills in their
# last training Congress and in the eval Congress (both with >= 5 referred bills); it is used only to
# stratify the evaluation, never as a feature.

# %%
# D22-D25: truncated-SVD embeddings per S3 fold (val selection, refit, level gap), per-positive hits by legislator status.
_SVD_CACHE, _LU = {}, {}
_TR = lambda lo, hi: tuple(range(lo, hi))
_CH1 = {"House of Representatives": "H", "Senate": "S"}
_REG = MAJORITY.assign(ch=MAJORITY.chamber.map(_CH1)).pivot(index="congress", columns="ch", values="majority")
_REG = {int(c): f"{r.H}H/{r.S}S" for c, r in _REG.iterrows()}
_MAJ = MAJORITY.set_index(["congress", "chamber"]).majority


def _svd_factors(train, hl):
    """Rank-max(SVD_RANKS) factors (U*s, Vt) of log(1 + sponsor x cosponsor counts) of the training Congresses."""
    if (train, hl) not in _SVD_CACHE:
        _, hist = _counts(np.isin(_CD["cong"], train), hl)
        U, sv, Vt = np.linalg.svd(np.log1p(hist), full_matrices=False)
        r = max(SVD_RANKS)
        _SVD_CACHE[(train, hl)] = (U[:, :r] * sv[:r], Vt[:r])
    return _SVD_CACHE[(train, hl)]


def _regime_cw(train, ec, factor=2.0):
    """Per-Congress training weights: x factor for Congresses with the eval Congress's House/Senate majorities."""
    cw = np.ones(max(CONGRESSES) + 2)
    for c in train:
        if _REG.get(c) == _REG.get(ec):
            cw[c] = factor
    return cw


def _learn_unit(train, ec, svd_names=tuple(SVD_FAMILY), weighted=False, hit_names=()):
    """Task 1 per-bill metrics of the baselines + SVD members (+ regime / recency weighted sponsor_hist) fit on
    `train`, evaluated on Congress ec at t0=0; weighted=True also fits Task 2 eb_rate(_t0safe) under the weights."""
    key = (train, ec, tuple(svd_names), weighted, tuple(hit_names))
    if key in _LU:
        return _LU[key]
    tmask = np.isin(_CD["cong"], train)
    facs = {hl: _svd_factors(train, hl) for hl in {SVD_FAMILY[n][1] for n in svd_names}}
    cw = _regime_cw(train, ec)
    wc = ({"sh_recency_hl1": _counts(tmask, 1), "sh_regime_x2": _counts(tmask, None, cw),
           "sh_regime_x2_hl1": _counts(tmask, 1, cw)} if weighted else {})

    def extra(cand, spi, hs, P, **_):
        out = {}
        us, inv = np.unique(spi, return_inverse=True)
        for n in svd_names:
            r, hl = SVD_FAMILY[n]
            US, Vt = facs[hl]
            out[n] = (US[us, :r] @ Vt[:r][:, cand])[inv] * hs[:, None] + P * 1e-12   # popularity breaks exact ties
        for n, (_pop, hist) in wc.items():
            out[n] = hist[spi][:, cand] * hs[:, None] * 1e9 + P
        return out

    sink = {n: [] for n in hit_names} if hit_names else None
    u = {"cosponsor_t0": _t1_eval(tmask, _BL.index[_BL.congress == ec].to_numpy(), 0, extra=extra, hit_sink=sink), "hits": sink}
    if weighted:
        tr2, te2 = _ADV[_ADV.congress.isin(train)], _ADV[_ADV.congress == ec]
        rec, reg = 0.5 ** (max(train) - tr2.congress.to_numpy()), cw[tr2.congress.to_numpy()]
        sc = {}
        for n, w in (("unweighted", None), ("recency_hl1", rec), ("regime_x2", reg), ("regime_x2_hl1", rec * reg)):
            o = _t2_scores(tr2, te2, w=w)
            sc[f"eb_rate|{n}"], sc[f"eb_rate_t0safe|{n}"] = o["eb_rate"], o["eb_rate_t0safe"]
        u["advancement"] = (sc, te2)
    _LU[key] = u
    return u


_ts_l = time.time()
_SVDN = list(SVD_FAMILY)
_SVD_FIX = "svd_r32_hl1"                                # fixed GNN-sized embedding (rank 32) for the staleness sensitivity
_LV, _SVD_PICK, _lrows = {}, {}, []
for c in [FOLD_C[0] - 1] + FOLD_C:                      # T(c): fit on 108..c-1, evaluate c (= val unit of fold c+1)
    hn = ()
    if c in FOLD_C:
        vpb = _LV[c - 1]["cosponsor_t0"][0]
        _SVD_PICK[c] = pd.Series({n: vpb[n][:, 1].mean() for n in _SVDN}).idxmax()
        hn = tuple(dict.fromkeys((_SVD_PICK[c], _SVD_FIX, "sponsor_hist")))
    _LV[c] = _learn_unit(_TR(CONGRESSES[0], c), c, weighted=c in FOLD_C, hit_names=hn)
for c in FOLD_C:
    vpb, (tpb, tcodes) = _LV[c - 1]["cosponsor_t0"][0], _LV[c]["cosponsor_t0"]
    v = pd.Series({n: vpb[n][:, 1].mean() for n in _SVDN})
    t = pd.Series({n: tpb[n][:, 1].mean() for n in _SVDN})
    pick, best = v.idxmax(), t.idxmax()
    _, lo, _, hw = _pdiff(_LV[c], best, _LV[c], pick, "cosponsor_t0", "Recall@20", ("svd_regret", c)) if pick != best else (0, 0, 0, 0)
    dsh = _pdiff(_LV[c], pick, _LV[c], "sponsor_hist", "cosponsor_t0", "Recall@20", ("svd_vs_sh", c))
    lev = {role: next(r for r in _t1_summarize(pb, cd, ("svd_lvl", c, role), baselines=[pick]) if r["metric"] == "Recall@20")
           for role, (pb, cd) in (("val", _LV[c - 1]["cosponsor_t0"]), ("test", (tpb, tcodes)))}
    _lrows.append(dict(fold=f"c={c}", val_pick=pick, test_best=best, regret=t[best] - t[pick], regret_hw=hw,
                       regret_sig=pick != best and lo > 0, tau=stats.kendalltau(v.values, t.values)[0],
                       val=lev["val"]["value"], test=lev["test"]["value"],
                       hw_val=(lev["val"]["ci_high"] - lev["val"]["ci_low"]) / 2,
                       hw_test=(lev["test"]["ci_high"] - lev["test"]["ci_low"]) / 2,
                       svd_minus_sh=dsh[0], svd_minus_sh_lo=dsh[1], svd_minus_sh_hi=dsh[2]))
_LSEL = pd.DataFrame(_lrows).set_index("fold")
_cap = pd.DataFrame([dict(congress=c, member=n, rank=SVD_FAMILY[n][0], recency="hl1" if SVD_FAMILY[n][1] else "none",
                          val=_LV[c - 1]["cosponsor_t0"][0][n][:, 1].mean(), test=_LV[c]["cosponsor_t0"][0][n][:, 1].mean())
                     for c in FOLD_C for n in _SVDN])
print(f"SVD embeddings: {len(_SVD_CACHE)} factorizations, {time.time() - _ts_l:.0f}s. Per S3 fold (val = Congress c-1 with "
      "fit on 108..c-2; test = c with refit on 108..c-1), Recall@20 at t0=0:")
print(_LSEL[["val_pick", "test_best", "regret", "regret_hw", "regret_sig", "tau", "val", "test", "svd_minus_sh"]].round(4).to_string())
line_chart(_cap.groupby(["rank", "recency"])[["test"]].mean().reset_index().assign(series=lambda d: "test, recency " + d.recency),
           x="rank", y="test", series="series", yfmt=".3f", ylabel="mean test Recall@20", xlabel="SVD rank (capacity)",
           title="Learned embeddings: test Recall@20 by capacity (mean over S3 folds 112-118)",
           subtitle=f"sponsor_hist reference {_fv1.loc['sponsor_hist', 'mean']:.3f}; the val-selected member per fold is in the table",
           height=300, name="D_svd_capacity")

# D22 (level gap) and D23 (selection regret) for the learned model.
_gsvd = _LSEL.reset_index()[["fold", "val", "test", "hw_val", "hw_test"]].assign(baseline="svd_pick")
_gsvd = pd.concat([_gsvd.assign(split="S3")] + [_gsvd[_gsvd.fold == f"c={SPLIT_DEFS[s_]['test'][0]}"].assign(split=s_)
                                                 for s_ in ("S1a", "S1b")])
_gsvd = _gsvd.assign(gap=_gsvd.test - _gsvd.val, rel_gap=(_gsvd.test - _gsvd.val) / _gsvd.val,
                     hw_diff=np.sqrt(_gsvd.hw_val ** 2 + _gsvd.hw_test ** 2))
_gap_check("D22", "Task 1, learned SVD embeddings (val-selected): validation Congress predicts the test score level (S3)",
           _gsvd, "svd_pick", na=_NA_FAM, extra=f"Repeats D1 for a model with per-legislator parameters; val picks: "
                                    f"{', '.join(f'{k}:{v}' for k, v in _LSEL.val_pick.value_counts().items())}.")
_l_ok = bool(not ((_LSEL.regret.mean() > D_TOL["cosponsor_t0"]) and (_LSEL.regret_sig.sum() >= 2)))
record_check("D", "D23", "Task 1, learned SVD embeddings: validation picks the test-best rank / recency (regret vs paired noise, S3)",
             _l_ok, metric=float(_LSEL.regret.mean()), quantity="learned_model_selection", kind="validity",
             threshold="FAIL if mean regret > D_TOL (MDE/2 = 0.005) Recall@20 AND regret significant (paired cluster CI) in >= 2 of 7 folds (as D11)",
             by_split=_bs({"S3": _l_ok, **{s_: round(float(_LSEL.loc[f"c={SPLIT_DEFS[s_]['test'][0]}", "regret"]), 4) for s_ in ("S1a", "S1b")}}),
             na_reason=_NA_FAM,
             detail=f"Hit rate {(_LSEL.val_pick == _LSEL.test_best).mean():.0%}, max regret {_LSEL.regret.max():.4f}, significant in "
                    f"{int(_LSEL.regret_sig.sum())}/7 folds, mean tau(val, test) over the {len(_SVDN)} members {_LSEL.tau.mean():.2f}. Picks: "
                    f"{', '.join(f'{k}:{v}' for k, v in _LSEL.val_pick.value_counts().items())}. Val-picked SVD minus sponsor_hist on "
                    f"test: mean {_LSEL.svd_minus_sh.mean():+.4f}, significant in {int(((_LSEL.svd_minus_sh_lo > 0) | (_LSEL.svd_minus_sh_hi < 0)).sum())}/7 "
                    "folds. by_split S1a/S1b = regret on their single fold (metric). A learned model whose capacity is chosen on "
                    "the previous Congress keeps working on the next one if this passes.")

# %%
# D24/D25: Recall@20 per positive by legislator status (new / returning with or without a role change), SVD vs sponsor_hist.
_first_c = pd.Series(_CD["cong"]).groupby(_CD["leg"]).min()                 # first Congress with a cosponsorship
_mem = members[members.congress.isin(CONGRESSES)].assign(leg=lambda d: _LEG.get_indexer(d.bioguide_id))
_mem = _mem.sort_values("congress")[["leg", "congress", "chamber", "party_code_asof", "caucus"]]
_bc = load("bill_committees", ("bill_id", "committee_code", "parent_committee_code"))
_bc = _bc[_bc.parent_committee_code.isna()].merge(_BL.reset_index()[["bill_id", "spon", "congress"]], on="bill_id")
_bc = _bc[_bc.spon >= 0].drop_duplicates(["bill_id", "committee_code"])
_cprof = _bc.groupby(["spon", "congress", "committee_code"]).size()        # committee-referral profile per sponsor-Congress
_nref = _bc.drop_duplicates("bill_id").groupby(["spon", "congress"]).size()
_COS_MIN, _NREF_MIN = 0.5, 5


def _cos_profile(pairs):
    """Cosine similarity of committee-referral profiles for rows (leg, c_prev, c_now); NaN if < _NREF_MIN bills."""
    out = np.full(len(pairs), np.nan)
    for i, (l, c0, c1) in enumerate(pairs):
        if _nref.get((l, c0), 0) < _NREF_MIN or _nref.get((l, c1), 0) < _NREF_MIN:
            continue
        a, b = _cprof.loc[(l, c0)], _cprof.loc[(l, c1)]
        a, b = a.align(b, fill_value=0)
        out[i] = float(a @ b / np.sqrt((a @ a) * (b @ b)))
    return out


_STRATA = ["new (no training history)", "returning: chamber or party switch", "returning: majority status flipped",
           "returning: committee shift (proxy)", "returning: unchanged"]
_H = []
for c in FOLD_C:
    h, pk = _LV[c]["hits"], _SVD_PICK[c]
    a, b, f = h[pk], h["sponsor_hist"], h[_SVD_FIX]
    assert np.array_equal(a[:, :2], b[:, :2]) and np.array_equal(f[:, :2], b[:, :2])
    d = pd.DataFrame({"code": a[:, 0], "leg": a[:, 1], "hit_svd": a[:, 2], "hit_sh": b[:, 2], "hit_r32": f[:, 2], "congress": c})
    d["chamber"] = _BL.chamber.reindex(d.code).to_numpy()
    prev = _mem[_mem.congress < c].groupby("leg").last()
    cur = _mem[_mem.congress == c].groupby("leg").last()
    pv, cu = prev.reindex(d.leg), cur.reindex(d.leg)
    new = ~(_first_c.reindex(d.leg).to_numpy() < c)
    sw = (pv.chamber.to_numpy() != d.chamber.to_numpy()) | (pv.party_code_asof.to_numpy() != cu.party_code_asof.to_numpy())
    in_maj_prev = pv.caucus.to_numpy() == _MAJ.reindex(pd.MultiIndex.from_arrays([pv.congress.fillna(0).astype(int), pv.chamber])).to_numpy()
    in_maj_now = cu.caucus.to_numpy() == _MAJ.reindex(pd.MultiIndex.from_arrays([np.full(len(d), c), d.chamber])).to_numpy()
    ul = d[["leg"]].assign(c0=pv.congress.fillna(0).astype(int).to_numpy()).drop_duplicates()
    ul["cos"] = _cos_profile(list(zip(ul.leg, ul.c0, np.full(len(ul), c))))
    cs_ = d[["leg"]].merge(ul.drop(columns="c0"), on="leg", how="left").cos.to_numpy()
    d["stratum"] = np.select([new, pv.chamber.isna().to_numpy() | sw, in_maj_prev != in_maj_now, cs_ < _COS_MIN],
                             _STRATA[:4], _STRATA[4])
    _H.append(d)
_H = pd.concat(_H, ignore_index=True)
_H["changed"] = _H.stratum.isin(_STRATA[1:4])
_ubill, _H["bpos"] = np.unique(_H.code.to_numpy(), return_inverse=True)
_Wh = _boot_weights(_BL.cluster.reindex(_ubill).to_numpy(), "D_stale")


def _rec(mask, col, W=_Wh, bpos=None):
    """Micro Recall@20 over positives in mask, per bootstrap row of W (cluster bootstrap by sponsor)."""
    bp = _H.bpos.to_numpy() if bpos is None else bpos
    nb = W.shape[1]
    num = W @ np.bincount(bp[mask], weights=_H[col].to_numpy()[mask].astype(float), minlength=nb)
    den = W @ np.bincount(bp[mask], minlength=nb).astype(float)
    return num / np.maximum(den, 1e-12)


def _ci(x):
    return x[0], *np.percentile(x[1:], [2.5, 97.5])


_strt = []
for st in _STRATA:
    m = (_H.stratum == st).to_numpy()
    rs, rh = _rec(m, "hit_svd"), _rec(m, "hit_sh")
    _strt.append(dict(stratum=st, n_pos=int(m.sum()), share=m.mean(), svd=rs[0], svd_r32=_rec(m, "hit_r32")[0], sponsor_hist=rh[0],
                      svd_minus_sh=_ci(rs - rh)[0], ci_lo=_ci(rs - rh)[1], ci_hi=_ci(rs - rh)[2]))
_strt = pd.DataFrame(_strt).set_index("stratum")
print(f"Recall@20 per positive by legislator status, S3 test folds {FOLD_C[0]}-{FOLD_C[-1]} pooled (svd = val-picked member per "
      "fold; CI = cluster bootstrap by sponsor):")
print(_strt.round(4).to_string())
bar_chart(_strt.reset_index().melt(id_vars="stratum", value_vars=["svd", "sponsor_hist"], var_name="model", value_name="recall"),
          x="stratum", y="recall", series="model", yfmt=".2f", ylabel="Recall@20 (per positive)",
          title="Recall@20 by legislator status: learned embeddings vs count history (S3 test folds pooled)",
          subtitle="new = no cosponsorship in the training Congresses (zero embedding); changes measured vs the member's last training Congress",
          height=320, name="D_staleness")


def _did(mask_ch, mask_un, W=_Wh, bpos=None, col="hit_svd"):
    """(unchanged - changed) recall gap of the svd model minus the same gap of sponsor_hist, per bootstrap row."""
    g_s = _rec(mask_un, col, W, bpos) - _rec(mask_ch, col, W, bpos)
    g_h = _rec(mask_un, "hit_sh", W, bpos) - _rec(mask_ch, "hit_sh", W, bpos)
    return g_s - g_h


_un = (_H.stratum == _STRATA[4]).to_numpy()
_did_all = _ci(_did(_H.changed.to_numpy(), _un))
_did_r32 = _ci(_did(_H.changed.to_numpy(), _un, col="hit_r32"))
_did_type = {st: _ci(_did((_H.stratum == st).to_numpy(), _un)) for st in _STRATA[1:4]}
_did_fold = []
for c in FOLD_C:
    fm = (_H.congress == c).to_numpy()
    ub, bp = np.unique(_H.code.to_numpy()[fm], return_inverse=True)
    bpos = np.zeros(len(_H), int)
    bpos[fm] = bp
    Wf = _boot_weights(_BL.cluster.reindex(ub).to_numpy(), ("D_stale", c))
    _did_fold.append((c, *_ci(_did(fm & _H.changed.to_numpy(), fm & _un, Wf, bpos)),
                      _ci(_did(fm & _H.changed.to_numpy(), fm & _un, Wf, bpos, "hit_r32"))[1]))
_did_fold = pd.DataFrame(_did_fold, columns=["congress", "did", "lo", "hi", "lo_r32"])
_n_sig_st, _n_sig_r32 = int((_did_fold.lo > 0).sum()), int((_did_fold.lo_r32 > 0).sum())
_new = _strt.loc[_STRATA[0]]
_ret = _H.stratum != _STRATA[0]
record_check("D", "D24", "Task 1: Recall@20 for new vs returning legislators, learned embeddings vs count history (S3 test, informational)",
             None, metric=float(_new.svd), threshold="informational (cold start for a transductive model; section B sizes it)",
             detail=f"New legislators: {_new.share:.1%} of test positives; Recall@20 svd {_new.svd:.3f} vs sponsor_hist "
                    f"{_new.sponsor_hist:.3f} (zero embedding and zero training popularity: they are ranked below every member "
                    "with history, so they are never in the top 20). Returning: svd "
                    f"{_rec(_ret.to_numpy(), 'hit_svd')[0]:.3f}, sponsor_hist {_rec(_ret.to_numpy(), 'hit_sh')[0]:.3f}. A transductive GNN "
                    "has no embedding for new members: give it inductive node features (party, state, chamber, committee) or score "
                    "new members separately; S3 / S1 test Congresses contain the realistic share of new members, a random split does not.")
_d25_ok = not (((_did_all[1] > 0) and (_n_sig_st >= 2)) or ((_did_r32[1] > 0) and (_n_sig_r32 >= 2)))
record_check("D", "D25", "Task 1: learned embeddings do not go stale for returning legislators whose role changed (DiD vs count history, S3)",
             bool(_d25_ok), metric=float(_did_all[0]), quantity="embedding_staleness", kind="drift",
             threshold="FAIL if, for the val-picked or the fixed rank-32 SVD, the pooled difference in differences (svd recall loss "
                       "on changed vs unchanged returning members, minus the same loss for sponsor_hist) has CI > 0 AND is "
                       "significant in >= 2 of 7 folds",
             by_split=_bs({"S3": bool(_d25_ok)}), na_reason="measured on the pooled S3 test folds 112-118 (S1a / S1b folds are part of that pool)",
             detail=f"DiD pooled {_did_all[0]:+.4f} [{_did_all[1]:+.4f}, {_did_all[2]:+.4f}]; by change type: " + "; ".join(
                 f"{k.split(': ')[1]} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}] (n={int((_H.stratum == k).sum()):,})" for k, v in _did_type.items())
             + f"; significant in {_n_sig_st}/7 folds. Fixed rank-32 embedding ({_SVD_FIX}): DiD {_did_r32[0]:+.4f} "
               f"[{_did_r32[1]:+.4f}, {_did_r32[2]:+.4f}], significant in {_n_sig_r32}/7 folds. Positive = the learned model loses more on members whose role changed than "
               "the count heuristic does on the same positives. Committee shift is a proxy (referral profile of sponsored bills, "
               "eval Congress used for stratification only). If this fails, the GNN needs Congress-specific node features or "
               "embeddings refreshed with the val Congress (refit, D19), and only forward-in-time splits (S1/S3) can reveal it.")

# %% [markdown]
# ### D7. How optimistic is the random bill split?
#
# A random split mixes bills of the same Congress (and of future Congresses) across train and test,
# so the baselines see the very legislators, coalitions and sponsor relationships they are tested on.
# We compare the random split's test bills of each Congress with the temporal (S3) test score of
# that same Congress, **on the same bills** (the S3 model scores the S5 test bills of Congress c too), so
# the optimism has a paired cluster-bootstrap CI. Calibration: optimism counts when its mean exceeds the
# section-wide minimum detectable effect `D_MDE` (0.01 Recall@20 / 0.01 AUROC, the smallest difference worth
# acting on; the cell first compares it with the data) and it is significant in at least 2 of 7 Congresses.
#
# **Why one fixed MDE.** Earlier rounds used the median validation margin between the two best near-tie
# candidates as the MDE. That margin measures the *spacing of our candidate grid* (e.g. shrinkage 5 vs
# 20 pseudo-bills), not what matters in practice, and made almost every Task-2 pair "relevant". The cell
# prints that margin next to the MDE: when the grid margin is far below the MDE, validation is choosing
# among ties, which is harmless and is exactly what the regret tolerance (MDE / 2) allows. A positive gap confirms
# the random split overstates performance and must only be used as a leaky upper-bound reference.

# %%
# Diagnostic: median validation margin (val pick minus runner-up) over S3 folds and near-tie families = candidate-grid
# spacing; compared with the fixed section-wide D_MDE (defined with the engine constants).
def _val_margins(df, metric):
    v = df[(df.split == "S3") & (df.eval_role == "val") & (df.metric == metric)]
    return v.groupby(["family", "fold"]).value.apply(lambda x: float(np.diff(np.sort(x.to_numpy())[-2:])[0]))


_mg1 = pd.concat([_val_margins(D_FAMILY_RESULTS[D_FAMILY_RESULTS.task == "cosponsor_t0"], "Recall@20"),
                  _cap.groupby("congress").val.apply(lambda x: float(np.diff(np.sort(x.to_numpy())[-2:])[0]))])
_mg2 = _val_margins(D_FAMILY_RESULTS[D_FAMILY_RESULTS.task == "advancement"], "AUROC")
D_VAL_MARGIN = {"cosponsor_t0": float(_mg1.median()), "advancement": float(_mg2.median())}
_hwpair = _sel(*_D_PRIMARY_T1, split="S3", eval_role="test").assign(hw=lambda d: (d.ci_high - d.ci_low) / 2).hw.median()
print(f"MDE used by every check: Task 1 {D_MDE['cosponsor_t0']} Recall@20 ({D_MDE['cosponsor_t0'] / _fv1.loc[_best1, 'mean']:.0%} of "
      f"'{_best1}' mean test level), Task 2 {D_MDE['advancement']} AUROC; regret tolerance MDE/2. Median val margin pick vs "
      f"runner-up (grid spacing, not practical importance): Task 1 {D_VAL_MARGIN['cosponsor_t0']:.4f} (IQR {_mg1.quantile(.25):.4f}-"
      f"{_mg1.quantile(.75):.4f}), Task 2 {D_VAL_MARGIN['advancement']:.4f} (IQR {_mg2.quantile(.25):.4f}-{_mg2.quantile(.75):.4f}) "
      "-> most family choices are ties below the MDE")

# D7: random-split optimism = S5 test score minus S3 (temporal) test score on the same bills of Congress c.
_opt = []
_s5key = _unit_keys(next(u for u in iter_eval_units("S5") if u["role"] == "test"))
for t0, task in ((0, "cosponsor_t0"), (7, "cosponsor_t7")):
    pb, codes = _unit(*_s5key)[task]
    cong = _BL.congress.reindex(codes).to_numpy()
    for c in FOLD_C:
        msk = cong == c
        if msk.sum() < 30:
            continue
        p3, c3 = _fold_unit("S3", c)[task]
        ix = pd.Index(c3).get_indexer(codes[msk])
        assert (ix >= 0).all()
        W = _boot_weights(_BL.cluster.reindex(codes[msk]).to_numpy(), ("D7", c, t0))
        for b in T1_BASELINES:
            r5, r3 = pb[b][msk, 1], p3[b][ix, 1]
            est = W @ (r5 - r3) / W.sum(1)
            lo, hi = np.percentile(est[1:], [2.5, 97.5])
            _opt.append(dict(task=task, baseline=b, congress=c, random=r5.mean(), temporal=r3.mean(), optimism=est[0],
                             lo=lo, hi=hi, n_random=int(msk.sum())))
_opt = pd.DataFrame(_opt)
print(f"Random (S5) minus temporal (S3) Recall@20 on the same S5 test bills, mean over Congresses {FOLD_C[0]}-{FOLD_C[-1]}:")
print(_opt[_opt.task == "cosponsor_t0"].groupby(["task", "baseline"]).agg(mean_random=("random", "mean"), mean_temporal=("temporal", "mean"),
                                             mean_optimism=("optimism", "mean"), n_sig=("lo", lambda x: int((x > 0).sum())))
      .round(3).to_string())
_s5t2 = _sel(*_D_PRIMARY_T2, split="S5", eval_role="test").set_index("baseline").value
_s3t2 = _sel(*_D_PRIMARY_T2, split="S3", eval_role="test").groupby("baseline").value.mean()
print(f"Task 2 AUROC eb_rate: random split test {_s5t2['eb_rate']:.3f} vs mean S3 test {_s3t2['eb_rate']:.3f}")
_oc = _opt[(_opt.task == "cosponsor_t0") & (_opt.baseline == _best1)]
line_chart(pd.concat([_oc.assign(split="temporal (S3)", score=_oc.temporal),
                      _oc.assign(split="random (S5)", score=_oc.random)]),
           x="congress", y="score", series="split", yfmt=".2f", ylabel="Recall@20", xlabel="test Congress",
           title=f"Random vs temporal test Recall@20 per Congress ({_best1}, t0=0)",
           subtitle="random split = same Congress's bills seen in training; temporal = only earlier Congresses",
           height=320, name="D_random_optimism")
_d7 = float(_oc.optimism.mean())
_d7_ok = not ((_d7 > D_MDE["cosponsor_t0"]) and ((_oc.lo > 0).sum() >= 2))
record_check("D", "D7", "Random bill split does not inflate scores relative to the temporal split (paired optimism on the same bills, Task 1)",
             bool(_d7_ok), metric=_d7, threshold=f"FAIL if mean optimism > D_MDE ({D_MDE['cosponsor_t0']:.4f} Recall@20) AND "
             "significant (paired cluster CI) in >= 2 of 7 Congresses", by_split=_bs({"S5": bool(_d7_ok)}), kind="validity",
             na_reason="D7 judges only the random split S5; the temporal S3 test score of the same bills is the reference",
             detail=f"Mean S5-minus-S3 Recall@20 for '{_best1}' over Congresses {FOLD_C[0]}-{FOLD_C[-1]}: {_d7:+.3f} "
                    f"({_d7 / _oc.temporal.mean():+.0%} relative, {_d7 / D_MDE['cosponsor_t0']:.0f}x the MDE); significant in "
                    f"{int((_oc.lo > 0).sum())}/7 Congresses. Task 2 eb_rate AUROC random "
                    f"{_s5t2.get('eb_rate', np.nan):.3f} vs temporal {_s3t2.get('eb_rate', np.nan):.3f}. FAIL means the random split "
                    "leaks same-Congress information and overstates performance: use it only as a leaky upper bound. Part of the "
                    "gap is legitimate same-Congress history (see D13: the streaming protocol recovers it without leakage).")

# %% [markdown]
# ### D8, D13-D14. Same-Congress history: within-Congress split (S4) and the streaming protocol
#
# A temporal GNN deployed during a Congress can see that Congress's cosponsorships up to the
# prediction time. S4 tests this on the last 20% of each Congress's bills. The **streaming** variants
# (`S1a+stream`, `S1b+stream`, `S3+stream`) keep the by-Congress pools but let the baselines also
# count every eval-Congress cosponsorship dated on or before *introduced_date + t0* of the bill being
# scored (Task 2: eval-Congress bills whose 180-day label had resolved by the scored bill's
# introduction). This is leakage-free by construction and is how a temporal model would be used. If
# streaming adds a lot, the by-Congress split should be evaluated in streaming mode (and the cold-start
# problem from section B is largely an artefact of forbidding same-Congress history).

# %%
# D8: S4 vs S3 on the same Congress; D13/D14: gain and val-test gaps of the streaming by-Congress protocol.
_w = []
for c in SPLIT_DEFS["S4"]["folds"]:
    for task, metric, b in ((*_D_PRIMARY_T1, _best1), ("cosponsor_t7", "Recall@20", _best1), (*_D_PRIMARY_T2, _best2)):
        a = _sel(task, metric, b, split="S4", fold=f"c={c}", eval_role="test")
        z = _sel(task, metric, b, split="S3", eval_role="test", eval_congress=c)
        if len(a) and len(z):
            _w.append(dict(congress=c, task=task, baseline=b, S4_test=float(a.value.iloc[0]),
                           S3_test=float(z.value.iloc[0]), n_S4=int(a.n_eval.iloc[0]), n_S3=int(z.n_eval.iloc[0])))
_w = pd.DataFrame(_w)
_w["diff"] = _w.S4_test - _w.S3_test
print("S4 (last 20% of bills, same-Congress history) minus S3 test score on the same Congress, best baselines:")
print(_w.pivot_table(index="congress", columns="task", values=["S4_test", "S3_test", "diff"]).round(3).to_string())
_w1 = _w[_w.task == _D_PRIMARY_T1[0]]
_w2 = _w[_w.task == _D_PRIMARY_T2[0]]
record_check("D", "D8", "Within-Congress split (S4) gives scores comparable to by-Congress split (mean S4-S3 test diff, Task 1)",
             None, metric=float(_w1["diff"].mean()), threshold="informational", by_split=_bs({"S4": round(float(_w1["diff"].mean()), 4)}),
             na_reason="S4-only comparison (S4 test score minus the S3 test score of the same Congress)",
             detail=f"Task 1 '{_best1}' S4-S3 Recall@20 diff by Congress: " + ", ".join(f"{r.congress}: {r.diff:+.3f}" for r in _w1.itertuples())
                    + f". Task 2 S4 test sizes {_w2.n_S4.tolist()} vs S3 {_w2.n_S3.tolist()} uncensored bills "
                    f"(diff {_w2['diff'].round(3).tolist()}). S4 mixes two effects (same-Congress history and a late-bill population); "
                    "the streaming variant (D13) isolates the history effect on the full Congress.")

_st = []
for task, metric, b in ((*_D_PRIMARY_T1, _best1), ("cosponsor_t7", "Recall@20", _best1), (*_D_PRIMARY_T1, "popularity"),
                        (*_D_PRIMARY_T1, "rw2hop"), (*_D_PRIMARY_T2, _best2)):
    for role in ("val", "test"):
        p = _sel(task, metric, b, split="S3", eval_role=role).set_index("fold")
        s = _sel(task, metric, b, split="S3+stream", eval_role=role).set_index("fold")
        for f in p.index:
            _st.append(dict(series=f"{task} {metric} ({b})", fold=f, role=role, congress=int(p.eval_congress[f]),
                            plain=p.value[f], stream=s.value[f], gain=s.value[f] - p.value[f]))
_st = pd.DataFrame(_st)
print("\nStreaming (S3+stream) vs plain S3, test folds:")
print(_st[_st.role == "test"].groupby("series").agg(mean_plain=("plain", "mean"), mean_stream=("stream", "mean"),
                                                    mean_gain=("gain", "mean"), min_gain=("gain", "min"),
                                                    max_gain=("gain", "max")).round(4).to_string())
_gs1 = _gaps(*_D_PRIMARY_T1)
_gs1 = _gs1[(_gs1.split == "S3+stream") & (_gs1.baseline == _best1)]
_gs2 = _g2[(_g2.split == "S3+stream") & (_g2.baseline == _best2)]
print("Val-test gaps: " + "; ".join(f"{n}: mean |rel gap| {g.rel_gap.abs().mean():.3f}, signif {g.signif.mean():.0%}"
                                     for n, g in (("T1 S3", _s3g1), ("T1 S3+stream", _gs1), ("T2 S3", _s3g2), ("T2 S3+stream", _gs2))))
_stc = _st[(_st.role == "test") & (_st.series == f"{_D_PRIMARY_T1[0]} {_D_PRIMARY_T1[1]} ({_best1})")]
line_chart(pd.concat([_stc.assign(protocol="S3 (train Congresses only)", score=_stc.plain),
                      _stc.assign(protocol="S3+stream (+ eval-Congress history to t0)", score=_stc.stream)]),
           x="congress", y="score", series="protocol", yfmt=".2f", ylabel="Recall@20", xlabel="test Congress",
           title=f"Streaming same-Congress history: test Recall@20 per Congress ({_best1}, t0=0)",
           subtitle="streaming counts eval-Congress cosponsorships dated on or before each bill's prediction time",
           height=320, name="D_stream_gain")
_sg1 = _stc.gain
_sg2 = _st[(_st.role == "test") & (_st.series == f"{_D_PRIMARY_T2[0]} {_D_PRIMARY_T2[1]} ({_best2})")].gain
_sg1b = _st[(_st.role == "test") & (_st.series == f"cosponsor_t7 Recall@20 ({_best1})")].gain
_sgr = _st[(_st.role == "test") & (_st.series == f"{_D_PRIMARY_T1[0]} {_D_PRIMARY_T1[1]} (rw2hop)")].gain
_sgS1 = {v: float(_sel(*_D_PRIMARY_T1, _best1, split=v, eval_role="test").value.iloc[0]
                  - _sel(*_D_PRIMARY_T1, _best1, split=v[:-7], eval_role="test").value.iloc[0]) for v in ("S1a+stream", "S1b+stream")}
record_check("D", "D13", "Streaming eval-Congress history changes test scores (mean gain S3+stream - S3, Task 1 Recall@20)",
             None, metric=float(_sg1.mean()), threshold="informational",
             by_split=_bs({"S3+stream": round(float(_sg1.mean()), 4), **{k: round(v, 4) for k, v in _sgS1.items()}}),
             na_reason="gain is defined only for the streaming variants (vs their plain split)",
             detail=f"Task 1 '{_best1}' gain t0=0 {_sg1.mean():+.3f} (range {_sg1.min():+.3f} to {_sg1.max():+.3f}), t0=7 "
                    f"{_sg1b.mean():+.3f}; rw2hop gain {_sgr.mean():+.3f} (monthly graph snapshots, so slightly conservative); "
                    f"Task 2 '{_best2}' AUROC gain {_sg2.mean():+.3f}. A positive Task 1 gain means the model must "
                    "be evaluated with within-Congress history up to t0 (streaming), otherwise the by-Congress split understates "
                    "a temporal GNN and overstates cold start. The baselines simply add counts; a GNN can weight recent edges more.")
_gap_check("D14", "Task 1: under the streaming protocol the val Congress still predicts the test score (gaps within noise, S3+stream)",
           _g1, _best1, splits=("S1a+stream", "S1b+stream", "S3+stream"), na="plain splits are judged in D1",
           extra=f"Plain S3 mean |rel gap| {_d1:.3f}. Task 2 S3+stream mean |gap| {_gs2.gap.abs().mean():.3f} AUROC (plain "
                 f"{_d2:.3f}). PASS means the streaming protocol is as usable for model selection as the plain by-Congress split.")

# %% [markdown]
# ### D9, D15. Are the test sets large enough for what the split is used for?
#
# A split is used to **compare** models and protocols on the same test bills, so the relevant noise is
# that of the *paired difference*: Congress-level difficulty shifts move both models together and
# cancel. For each decision-relevant pair we compute, on every rolling test Congress, the paired
# difference and its cluster-bootstrap CI. Two different questions follow:
# (a) **single-Congress risk**: how often would ONE test Congress (S1a / S1b) fail to confirm the
# direction of a real difference? `p_miss_1C` = share of the 7 test Congresses whose paired CI does not
# exclude 0 in the direction of the mean (wrong sign or not significant); `p_flip_1C` = share with the
# opposite sign. More test Congresses do not reduce the Congress-to-Congress heterogeneity of an effect,
# so the deployment Congress can still differ: `pi_next` = 95% prediction half-width of the effect in a
# new Congress (t, 6 df). (b) **average effect**: the CI half-width of the mean effect over k = 1, 2, 7
# test Congresses, 1.96 * max(sd_between, se_1C) / sqrt(k).
# A pair is *relevant* if |mean difference| >= `D_MDE` (0.01, the same MDE as every other check here);
# smaller differences are not worth acting on. The `by_split` verdict: a split holds if, for every relevant
# model / near-tie pair, its miss risk is <= 20% (80% power): empirical `p_miss_1C` for one test Congress,
# the normal approximation Phi(|mean| / (sd / sqrt(k)) - 1.96) for k = 2 (S1pool) and k = 7 (S2 / S3).
# **Caveat: these are different quantities.** The one-Congress risk includes Congress-to-Congress
# heterogeneity of the effect; the k-Congress risk is about the *average* effect, so multi-fold splits
# pass more easily by construction. The deployment-relevant risk, `pi_next` relative to the MDE (how far
# the effect can move in the next Congress), does not shrink with k; D9b records it for every by-Congress
# split, also from the recent folds 115-118 only (the early folds 112-115 are least like 119). Task 2
# pairs are judged on the **t0-safe** family; policy-area pairs (kind `ref@d30`) are shown for reference.

# %%
# D9: paired-difference sizing per pair; D15 (info): absolute-level spread and pooled 117+118 test sets.
_S3u = lambda c: _fold_unit("S3", c)
_LVu = lambda c: _LV[c]
_PAIRS = [  # (kind, task, metric, label, unit A, a, unit B, b); a / b may be callables c -> member name
    ("model", *_D_PRIMARY_T1, "rw2hop - sponsor_hist", _S3u, "rw2hop", _S3u, "sponsor_hist"),
    ("model", *_D_PRIMARY_T1, "sponsor_hist - party+pop", _S3u, "sponsor_hist", _S3u, "party+pop"),
    ("model", *_D_PRIMARY_T1, "svd(val pick) - sponsor_hist", _LVu, _SVD_PICK.get, _LVu, "sponsor_hist"),
    ("near-tie", *_D_PRIMARY_T1, "sh_hl1 - sh_hl2", _S3u, "sh_hl1", _S3u, "sh_hl2"),
    ("near-tie", *_D_PRIMARY_T1, "sh_hl1 - sh_none+caucus", _S3u, "sh_hl1", _S3u, "sh_none+caucus"),
    ("near-tie", *_D_PRIMARY_T1, "rw_b0.1 - rw_b0.3", _S3u, "rw_b0.1", _S3u, "rw_b0.3"),
    ("near-tie", *_D_PRIMARY_T1, "svd_r32 - svd_r128", _LVu, "svd_r32", _LVu, "svd_r128"),
    ("protocol", *_D_PRIMARY_T1, "expanding - sliding (sponsor_hist)", _S3u, "sponsor_hist",
     lambda c: _fold_unit("S2", c), "sponsor_hist"),
    ("protocol", *_D_PRIMARY_T1, "stream - plain (sponsor_hist)", lambda c: _fold_unit("S3", c, stream=True),
     "sponsor_hist", _S3u, "sponsor_hist"),
    ("model", *_D_PRIMARY_T2, "eb_rate_t0safe - chamber_maj", _S3u, "eb_rate_t0safe", _S3u, "chamber_maj"),
    ("near-tie", *_D_PRIMARY_T2, "eb_m5_t0s - eb_m20_t0s", _S3u, "eb_m5_t0s", _S3u, "eb_m20_t0s"),
    ("near-tie", *_D_PRIMARY_T2, "eb_m20_t0s - eb_m80_t0s", _S3u, "eb_m20_t0s", _S3u, "eb_m80_t0s"),
    ("near-tie", *_D_PRIMARY_T2, "eb_m20_t0s - eb_nomaj_m20_t0s", _S3u, "eb_m20_t0s", _S3u, "eb_nomaj_m20_t0s"),
    ("ref@d30", *_D_PRIMARY_T2, "eb_rate - chamber_maj", _S3u, "eb_rate", _S3u, "chamber_maj"),
    ("ref@d30", *_D_PRIMARY_T2, "eb_m20 - eb_nomaj_m20", _S3u, "eb_m20", _S3u, "eb_nomaj_m20"),
    ("protocol", *_D_PRIMARY_T2, "expanding - sliding (eb_rate_t0safe)", _S3u, "eb_rate_t0safe",
     lambda c: _fold_unit("S2", c), "eb_rate_t0safe"),
]
_nm = lambda x, c: x if isinstance(x, str) else x(c)
_K_SPLIT = {s_: _n_test_c(s_) for s_ in ALL_SPLITS if _n_test_c(s_) is not None}
_tq = stats.t.ppf(0.975, len(FOLD_C) - 1) * np.sqrt(1 + 1 / len(FOLD_C))
_tq6 = stats.t.ppf(0.975, len(_DEV) - 1) * np.sqrt(1 + 1 / len(_DEV))          # development folds only (D27)
_RECENT = [c for c in FOLD_C if c >= _LOCK - 3]                                  # 115-118: closest to the deployment Congress
_tqr = stats.t.ppf(0.975, len(_RECENT) - 1) * np.sqrt(1 + 1 / len(_RECENT))
_prs, _pd_rows = [], []
for kind, task, metric, lab, uA, a, uB, b in _PAIRS:
    d = pd.DataFrame([dict(zip(("diff", "lo", "hi", "hw"), _pdiff(uA(c), _nm(a, c), uB(c), _nm(b, c), task, metric,
                                                                  ("pair", lab, c))), congress=c) for c in FOLD_C])
    _pd_rows.append(d.assign(pair=lab))
    mu, sd, hw1 = d["diff"].mean(), d["diff"].std(ddof=1), d.hw.mean()
    sd_eff = max(sd, hw1 / 1.96)
    sgn = np.sign(mu)
    n_dir = int(((d.lo > 0) if sgn > 0 else (d.hi < 0)).sum())
    row = dict(kind=kind, task="T1" if task.startswith("cos") else "T2", pair=lab, mean_diff=mu, sd_between=sd, hw_1C=hw1,
               n_sig_dir=n_dir, p_miss_1C=1 - n_dir / len(FOLD_C), p_flip_1C=float((np.sign(d["diff"]) == -sgn).mean()),
               pi_next=_tq * sd, relevant=bool(abs(mu) >= D_MDE[task]))
    for k in (1, 2, 7):
        row[f"hw_mean_k{k}"] = 1.96 * sd_eff / np.sqrt(k)
        row[f"miss_k{k}"] = float(1 - stats.norm.cdf(abs(mu) / (sd_eff / np.sqrt(k)) - 1.96))
    row["pi_rel"] = row["pi_next"] / D_MDE[task]                                  # deployment risk in MDE units
    row["pi_rel_recent"] = _tqr * d[d.congress.isin(_RECENT)]["diff"].std(ddof=1) / D_MDE[task]
    dv, lk = d[d.congress.isin(_DEV)], d[d.congress == _LOCK].iloc[0]           # D27: development folds vs locked fold
    mu6, sd6 = dv["diff"].mean(), dv["diff"].std(ddof=1)
    sd6e, k6 = max(sd6, dv.hw.mean() / 1.96), len(_DEV)
    row.update(mean_dev=mu6, relevant_dev=bool(abs(mu6) >= D_MDE[task]), hw_mean_k6=1.96 * sd6e / np.sqrt(k6),
               miss_k6=float(1 - stats.norm.cdf(abs(mu6) / (sd6e / np.sqrt(k6)) - 1.96)), pi_dev=_tq6 * sd6e,
               lock_diff=float(lk["diff"]), lock_lo=float(lk.lo), lock_hi=float(lk.hi),
               lock_in_pi=bool(abs(lk["diff"] - mu6) <= _tq6 * sd6e))
    _prs.append(row)
_prs = pd.DataFrame(_prs)
print(f"Paired differences on S3 test Congresses {FOLD_C[0]}-{FOLD_C[-1]} (MDE: T1 {D_MDE['cosponsor_t0']:.4f}, T2 "
      f"{D_MDE['advancement']:.4f}; p_miss_1C = share of single test Congresses not confirming the direction):")
print(_prs[["kind", "task", "pair", "mean_diff", "sd_between", "hw_1C", "relevant", "p_miss_1C", "p_flip_1C", "pi_next",
            "pi_rel", "hw_mean_k2", "hw_mean_k7", "miss_k2", "miss_k7"]].round(4).to_string(index=False))
_dec_pairs = _prs[_prs.kind.isin(["model", "near-tie"]) & _prs.relevant]


def _split_risk(k):
    """Max miss risk over relevant decision pairs for a split with k test Congresses ((a): empirical for k = 1)."""
    if not len(_dec_pairs):
        return 0.0, "none"
    r = _dec_pairs.p_miss_1C if k == 1 else _dec_pairs[f"miss_k{min(k, 7) if k in (1, 2, 7) else 7}"]
    return float(r.max()), _dec_pairs.pair.iloc[int(np.argmax(r.to_numpy()))]


_risk = {s_: _split_risk(1 if s_ in ("S1a", "S1b", "S1a+stream", "S1b+stream") else (2 if s_ == "S1pool" else 7))
         for s_ in _K_SPLIT}
_d9_split = {s_: bool(v[0] <= 0.2) for s_, v in _risk.items()}
_PREC_JS = r"""
const {width, height, title, subtitle, mde} = opts;
const m = {top: 48, right: 24, bottom: 56, left: 230};
const svg = d3.select(el).append('svg').attr('width', width).attr('height', height);
viz.title(svg, title, subtitle);
const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
const W = width - m.left - m.right, H = height - m.top - m.bottom;
const rows = Array.from(new Set(data.map(d => d.pair)));
const keys = Array.from(new Set(data.map(d => d.measure)));
const color = d3.scaleOrdinal(keys, viz.palette);
const y0 = d3.scaleBand(rows, [0, H]).padding(0.25), y1 = d3.scaleBand(keys, [0, y0.bandwidth()]).padding(0.1);
const xs = d3.scaleLinear([0, Math.max(mde * 1.2, d3.max(data, d => d.value))], [0, W]).nice();
g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xs).ticks(6).tickSize(-H).tickFormat(d3.format('.2f')))
  .call(a => a.selectAll('line').attr('stroke', viz.grid)).call(a => a.select('.domain').remove())
  .call(a => a.selectAll('text').attr('fill', viz.ink2).style('font', '11px sans-serif'));
g.append('g').call(d3.axisLeft(y0).tickSize(0)).call(viz.axis).call(a => a.select('.domain').remove());
g.append('line').attr('x1', xs(mde)).attr('x2', xs(mde)).attr('y1', 0).attr('y2', H).attr('stroke', viz.ink)
  .attr('stroke-width', 1.5).attr('stroke-dasharray', '4,3');
g.append('text').attr('x', xs(mde) + 4).attr('y', -6).attr('fill', viz.ink2).style('font', '11px sans-serif').text(opts.mlabel);
const tip = viz.tooltip(el);
g.selectAll(null).data(data).join('rect').attr('x', 0).attr('y', d => y0(d.pair) + y1(d.measure))
  .attr('height', y1.bandwidth()).attr('width', d => Math.max(xs(d.value), 1)).attr('rx', 2).attr('fill', d => color(d.measure))
  .on('mousemove', (e, d) => tip.show(`<b>${d.pair}</b><br>${d.measure}: ${d3.format('.2f')(d.value)}<br>mean paired difference: ${d3.format('+.4f')(d.mean_diff)}`, e))
  .on('mouseleave', () => tip.hide());
viz.legend(svg, keys.map(k => [k, color(k)]), m.left, height - 10);
"""
_szc = pd.concat([_dec_pairs.assign(measure="one test Congress (empirical, S1a / S1b)", value=_dec_pairs.p_miss_1C),
                  _dec_pairs.assign(measure="mean of 2 test Congresses (normal approx.)", value=_dec_pairs.miss_k2),
                  _dec_pairs.assign(measure="mean of 7 rolling test Congresses (normal approx.)", value=_dec_pairs.miss_k7)])
_szc = _szc.assign(pair=_szc.task + " " + _szc.pair)[["pair", "measure", "value", "mean_diff"]]
show_d3(_PREC_JS, _szc, height=110 + 42 * max(len(_dec_pairs), 1), width=760,
        opts=dict(title="Risk that the test set fails to confirm a real model difference (relevant pairs, S3 test Congresses)",
                  subtitle="share of evaluations whose paired CI misses the true direction; dashed line = 20% (80% power)",
                  mde=0.2, mlabel="20% miss risk"), name="D_paired_precision")
_dp_all = _prs[_prs.kind.isin(["model", "near-tie"])]
record_check("D", "D9", "Test-set size: a single test Congress confirms the direction of every relevant model / near-tie difference (paired, S3)",
             _d9_split.get("S1b"), metric=round(_risk["S1b"][0], 3),
             threshold="per split: max over relevant pairs (|mean diff| >= D_MDE) of the miss risk <= 0.20 (empirical share of "
                       "single test Congresses for 1-Congress splits; normal approximation for k = 2 / 7)",
             by_split=_bs(_d9_split), kind="validity", t0_task2=0,
             na_reason="S4 / S5 test sets are within-Congress samples, not whole Congresses: sized in the D15 table",
             detail=f"Relevant decision pairs: {len(_dec_pairs)} of {len(_dp_all)} (MDE T1 {D_MDE['cosponsor_t0']:.4f}, T2 "
                    f"{D_MDE['advancement']:.4f}). Miss risk by split: " + ", ".join(f"{k} {v[0]:.2f} ({v[1]})" for k, v in _risk.items())
                    + ". Worst single-Congress pairs: " + "; ".join(f"{r.task} {r.pair}: mean {r.mean_diff:+.4f}, p_miss_1C "
                    f"{r.p_miss_1C:.2f}, p_flip {r.p_flip_1C:.2f}, next-Congress PI +/-{r.pi_next:.4f}"
                    for r in _dec_pairs.sort_values("p_miss_1C").tail(3).itertuples())
                    + f". Average-effect CI half-width (median over relevant pairs) k=1 {_dec_pairs.hw_mean_k1.median():.4f}, k=2 "
                    f"{_dec_pairs.hw_mean_k2.median():.4f}, k=7 {_dec_pairs.hw_mean_k7.median():.4f}. More test Congresses narrow the "
                    "CI of the average effect but not the next-Congress prediction interval (Congress x model interaction; D9b: "
                    f"up to {_dec_pairs.pi_rel.max():.1f}x the MDE). The k = 1 and k = 7 risks are different quantities (single-Congress "
                    "direction vs average effect), so S2 / S3 pass partly by construction. Task 2 "
                    "pairs use the t0-safe family (prediction at introduction); policy-area pairs ('ref@d30') are in the table only.")
_pr_dec = _dec_pairs.set_index(_dec_pairs.task + " " + _dec_pairs.pair)
_pimax, _pimax_r = float(_pr_dec.pi_rel.max()), float(_pr_dec.pi_rel_recent.max())
record_check("D", "D9b", "Deployment risk: how far a relevant model difference can move in the next Congress (95% PI half-width / MDE; "
             "informational)", None, metric=round(_pimax, 2), kind="drift", t0_task2=0,
             threshold="informational: > 1 means the next Congress can reverse or erase an MDE-sized advantage seen on test",
             by_split=_bs({s_: round(_pimax, 2) for s_ in _K_SPLIT}),
             na_reason="S4 / S5 do not evaluate whole future Congresses",
             detail="pi_next / MDE per relevant pair (all 7 folds; recent folds 115-118 in brackets): " + "; ".join(
                 f"{k} {r.pi_rel:.1f} [{r.pi_rel_recent:.1f}]" for k, r in _pr_dec.iterrows())
             + f". Same value for every by-Congress split: the prediction interval for a NEW Congress does not shrink with the "
               f"number of test Congresses (it is Congress x model heterogeneity, t with {len(FOLD_C) - 1} df). The 7 S3 test "
               "Congresses include 112-115, which differ most from 119 (sections A / C); the recent-fold value uses only "
               f"{_RECENT[0]}-{_RECENT[-1]} (t with {len(_RECENT) - 1} df, hence wide). Read with D9: a split can pin down the average "
               "difference and still leave the 119 outcome uncertain.")

# Absolute level (informational): CI half-widths of the score itself, between-fold std, pooled 117+118 tests.
_hw = SPLIT_RESULTS[(SPLIT_RESULTS.eval_role == "test") & SPLIT_RESULTS.split.isin(["S1b", "S3", "S1pool", "S4"])
                    & (((SPLIT_RESULTS.task == _D_PRIMARY_T1[0]) & (SPLIT_RESULTS.metric == _D_PRIMARY_T1[1]) & (SPLIT_RESULTS.baseline == _best1))
                       | ((SPLIT_RESULTS.task == _D_PRIMARY_T2[0]) & (SPLIT_RESULTS.metric == "AUROC")
                          & (SPLIT_RESULTS.baseline == _best2)))].copy()
_hw["hw_cluster"] = (_hw.ci_high - _hw.ci_low) / 2
_hw["hw_bill"] = (_hw.ci_high_bill - _hw.ci_low_bill) / 2
_hwt = _hw.groupby(["task", "split"]).agg(n_eval=("n_eval", "mean"), n_clusters=("n_clusters", "mean"),
                                          value=("value", "mean"), hw_bill=("hw_bill", "mean"), hw_cluster=("hw_cluster", "mean"))
_hwt["cluster/bill"] = _hwt.hw_cluster / _hwt.hw_bill
print(f"\nAbsolute-score CI half-widths ('{_best1}' Recall@20, '{_best2}' AUROC; bill vs sponsor-cluster bootstrap; S3/S4 = mean over folds):")
print(_hwt.round(4).to_string())
_bstd1, _bstd2 = float(_fv1.loc[_best1, "std"]), float(_fv2.loc[_best2, "std"])
_k_abs = {k: int(np.ceil((1.96 * sd / thr) ** 2)) for k, sd, thr in (("T1", _bstd1, 2 * D_MDE["cosponsor_t0"]),
                                                                       ("T2", _bstd2, 2 * D_MDE["advancement"]))}
_sp2 = _hwt.loc[("advancement", "S1pool")]
record_check("D", "D15", "Absolute score level: test Congresses needed for a precise reported score (informational)",
             None, metric=f"T1 {_k_abs['T1']}, T2 {_k_abs['T2']} Congresses",
             threshold="informational: report test scores per Congress plus the mean over rolling folds",
             detail=f"Between-fold std of the absolute score (S3): T1 {_bstd1:.4f}, T2 {_bstd2:.4f}; Congresses for 1.96*std/sqrt(k) "
                    f"<= 2 x MDE ({2 * D_MDE['cosponsor_t0']:.2f} R@20 / {2 * D_MDE['advancement']:.2f} AUROC; a reported level is coarser than a paired comparison): T1 {_k_abs['T1']}, T2 {_k_abs['T2']}. Single-Congress cluster hw: T1 "
                    f"{_hwt.loc[(_D_PRIMARY_T1[0], 'S1b'), 'hw_cluster']:.3f}, T2 {_hwt.loc[('advancement', 'S1b'), 'hw_cluster']:.3f} "
                    f"(cluster/bill {_hwt.loc[('advancement', 'S1b'), 'cluster/bill']:.1f}x); pooled 117+118 (S1pool) T2 hw "
                    f"{_sp2.hw_cluster:.3f}. This is about what score to *expect* on a new Congress, not about choosing between "
                    "models (D9): report per-Congress scores and their spread.")

# %% [markdown]
# ### D10. Do older Congresses help (expanding) or hurt (sliding window)?
#
# If the process drifts, old Congresses can teach stale patterns and a sliding window (only the last 3
# Congresses) can match or beat the expanding window. Both windows are scored on the same test bills,
# so the per-fold difference has a **paired** cluster-bootstrap CI. We run it for plain count
# baselines, for their recency-weighted versions (half-life 1 Congress), and for the graph-structural
# `rw2hop` (with and without recency weighting), and for the learned SVD embeddings (val-picked member,
# same hyperparameters in both windows). If only the unweighted models prefer the sliding
# window, the right conclusion is "weight recent edges", not "drop old Congresses". Folds share most
# training Congresses, so the sign test is descriptive. Positive values = expanding window is better.

# %%
# D10: expanding (S3) minus sliding k=3 (S2) test scores per fold, paired cluster bootstrap, several models.
_ex = []
_D10_SERIES = [("T1 sponsor_hist", *_D_PRIMARY_T1, "sponsor_hist"), ("T1 sh_hl1 (recency)", *_D_PRIMARY_T1, "sh_hl1"),
               ("T1 rw2hop", *_D_PRIMARY_T1, "rw2hop"), ("T1 rw_b0.3_hl1 (recency)", *_D_PRIMARY_T1, "rw_b0.3_hl1"),
               ("T1 popularity", *_D_PRIMARY_T1, "popularity"), ("T1 sponsor_hist t0=7", "cosponsor_t7", "Recall@20", "sponsor_hist"),
               (f"T2 {_best2} AUROC", *_D_PRIMARY_T2, _best2), (f"T2 {_best2} AP_lift", "advancement", "AP_lift", _best2)]
for c in FOLD_C:
    ue, us_ = _fold_unit("S3", c), _fold_unit("S2", c)
    for lab, task, metric, b in _D10_SERIES:
        d, lo, hi, _ = _pdiff(ue, b, us_, b, task, metric, ("D10", c, lab))
        _ex.append(dict(series=lab, congress=c, diff=d, lo=lo, hi=hi))
    _pk = _SVD_PICK[c]                                    # learned model: same member refit on the sliding window
    d, lo, hi, _ = _pdiff(_LV[c], _pk, _learn_unit(_TR(c - SLIDE_K, c), c, svd_names=(_pk,)), _pk, *_D_PRIMARY_T1, ("D10svd", c))
    _ex.append(dict(series="T1 svd (val pick)", congress=c, diff=d, lo=lo, hi=hi))
_ex = pd.DataFrame(_ex)
_ex["sig_neg"], _ex["sig_pos"] = _ex.hi < 0, _ex.lo > 0
_ext = _ex.groupby("series", sort=False).agg(mean_diff=("diff", "mean"), fold_std=("diff", "std"),
                                             folds_expanding_better=("diff", lambda x: (x > 0).mean()),
                                             folds_sliding_signif=("sig_neg", "mean"), folds_expanding_signif=("sig_pos", "mean"))
print("Expanding minus sliding (k=3) on the same test bills, S3/S2 folds:")
print(_ext.round(4).to_string())
_exc = _ex[_ex.series.isin(["T1 sponsor_hist", "T1 sh_hl1 (recency)", "T1 svd (val pick)", f"T2 {_best2} AUROC"])]
line_chart(_exc, x="congress", y="diff", series="series", yfmt="+.3f", ylabel="expanding − sliding", xlabel="test Congress",
           title="Expanding window minus sliding window (k=3) test score per fold",
           subtitle="above 0 = using all earlier Congresses beats using only the last 3 (paired on the same test bills)",
           height=320, name="D_expanding_vs_sliding")
_hurt = {k: bool((_ext.loc[k, "mean_diff"] < 0) and (_ext.loc[k, "folds_sliding_signif"] >= 0.5)) for k in _ext.index}
_helps = {k: bool((_ext.loc[k, "mean_diff"] > 0) and (_ext.loc[k, "folds_expanding_signif"] >= 0.5)) for k in _ext.index}
_dec = ["T1 sh_hl1 (recency)", "T1 rw_b0.3_hl1 (recency)", "T1 svd (val pick)", f"T2 {_best2} AUROC"]
_d10 = not any(_hurt[k] for k in _dec)
_sign = {k: stats.binomtest(int((_ex[_ex.series == k]["diff"] < 0).sum()), len(FOLD_C), 0.5).pvalue for k in _ext.index}
_conc = ("recency weighting is needed for count/graph features (unweighted models prefer the sliding window, weighted ones do not)"
         if (_d10 and any(_hurt[k] for k in ("T1 sponsor_hist", "T1 rw2hop"))) else
         "old Congresses hurt even with recency weighting: use a sliding window" if not _d10 else "older Congresses do not hurt")
record_check("D", "D10", "Older Congresses do not hurt a recency-weighted model: expanding >= sliding window (k=3) on test",
             _d10, metric=f"T1 sh_hl1 diff={_ext.loc['T1 sh_hl1 (recency)', 'mean_diff']:+.4f}; T1 svd "
                          f"{_ext.loc['T1 svd (val pick)', 'mean_diff']:+.4f}; T2 {_ext.loc[f'T2 {_best2} AUROC', 'mean_diff']:+.4f}",
             threshold="FAIL if, for a recency-weighted or val-tuned model (sh_hl1, rw_b0.3_hl1, val-picked svd, T2), mean diff < 0 "
                       "and sliding is significantly better (paired cluster CI) in >= 50% of folds",
             by_split=_bs({**{s: _d10 for s in ("S1a", "S1b", "S3", "S1a+stream", "S1b+stream", "S3+stream", "S1pool")},
                           "S2": not any(_helps[k] for k in _dec)}), kind="validity",
             na_reason="S4 / S5 train on all earlier Congresses plus part of the eval Congress: the window choice does not apply",
             detail="Mean diff (folds sliding signif. better / expanding signif. better): " + "; ".join(
                 f"{k} {r.mean_diff:+.4f} ({r.folds_sliding_signif:.0%}/{r.folds_expanding_signif:.0%})" for k, r in _ext.iterrows())
             + f". Descriptive sign tests (dependent folds): " + ", ".join(f"{k} p={v:.2f}" for k, v in _sign.items() if k in _dec)
             + f". Conclusion: {_conc}. by_split: expanding splits pass only if a recency-weighted model is used; S2 fails "
               "if expanding is significantly better.")

# %% [markdown]
# ### D26. Regime-matched vs recency training weights
#
# Section A finds the deployment Congress (119, unified Republican control) closest to the last
# Congresses with the same House / Senate majorities, and the majority-sponsor effect depends on which
# party controls the chamber. If that matters for prediction, "train on the most recent Congresses"
# (recency weighting, sliding windows) is the wrong remedy and training should upweight Congresses with
# the same regime as the target. The regime of a Congress is known when it starts (election results), so
# this is leakage-free. On every S3 test fold we refit on 108..c-1 with four weightings and score the same
# test bills: unweighted, recency (half-life 1 Congress), regime-matched (x2 for training Congresses with
# Congress c's House and Senate majorities, from `majority_party()`), and regime x recency. Task 1 uses
# `sponsor_hist`; Task 2 uses `eb_rate_t0safe` (prediction at introduction) and `eb_rate`. Positive
# paired differences = regime weighting helps. Folds with no regime-matched training Congress give 0.

# %%
# D26: paired differences between training weightings on S3 test folds (same bills), both tasks.
_rg = []
_W_PAIRS = [("T1 sponsor_hist", *_D_PRIMARY_T1, "regime x2 - unweighted", "sh_regime_x2", "sponsor_hist"),
            ("T1 sponsor_hist", *_D_PRIMARY_T1, "regime x2 - recency", "sh_regime_x2", "sh_recency_hl1"),
            ("T1 sponsor_hist", *_D_PRIMARY_T1, "regime x2 + recency - recency", "sh_regime_x2_hl1", "sh_recency_hl1"),
            ("T1 sponsor_hist", *_D_PRIMARY_T1, "recency - unweighted", "sh_recency_hl1", "sponsor_hist")]
for b in ("eb_rate_t0safe", "eb_rate"):
    _W_PAIRS += [(f"T2 {b}", *_D_PRIMARY_T2, "regime x2 - unweighted", f"{b}|regime_x2", f"{b}|unweighted"),
                 (f"T2 {b}", *_D_PRIMARY_T2, "regime x2 - recency", f"{b}|regime_x2", f"{b}|recency_hl1"),
                 (f"T2 {b}", *_D_PRIMARY_T2, "regime x2 + recency - recency", f"{b}|regime_x2_hl1", f"{b}|recency_hl1"),
                 (f"T2 {b}", *_D_PRIMARY_T2, "recency - unweighted", f"{b}|recency_hl1", f"{b}|unweighted")]
for c in FOLD_C:
    nmatch = sum(_REG.get(x) == _REG.get(c) for x in range(CONGRESSES[0], c))
    for model, task, metric, lab, a, b in _W_PAIRS:
        d, lo, hi, hw = _pdiff(_LV[c], a, _LV[c], b, task, metric, ("D26", c, model, lab)) if (nmatch or "regime" not in lab) \
            else (0.0, 0.0, 0.0, 0.0)
        _rg.append(dict(model=model, comparison=lab, congress=c, regime=_REG.get(c), n_match=nmatch, diff=d, lo=lo, hi=hi))
_rg = pd.DataFrame(_rg)
_rgt = _rg.groupby(["model", "comparison"], sort=False).agg(mean_diff=("diff", "mean"), folds_better=("lo", lambda x: int((x > 0).sum())),
                                                          folds_worse=("hi", lambda x: int((x < 0).sum())))
print("Regime per Congress (House/Senate majority): " + ", ".join(f"{c}: {r}" for c, r in _REG.items() if c >= CONGRESSES[0]))
print(f"S3 folds with >= 1 regime-matched training Congress: {int((_rg.drop_duplicates('congress').n_match > 0).sum())}/{len(FOLD_C)}. "
      "Paired test differences (folds better / worse = paired cluster CI excludes 0):")
print(_rgt.drop(index="T2 eb_rate", level="model").round(4).to_string())
bar_chart(_rg[_rg.comparison == "regime x2 + recency - recency"].assign(fold="c=" + _rg.congress.astype(str) + " " + _rg.regime)[
              ["fold", "diff", "model"]], x="fold", y="diff", series="model", yfmt="+.4f", ylabel="regime+recency − recency",
          title="Does upweighting same-regime Congresses beat recency weighting alone? (S3 test folds)",
          subtitle="paired difference on the same test bills; Task 1 Recall@20, Task 2 AUROC; 0 = no regime-matched training Congress",
          height=300, name="D_regime_weighting")
_key = _rgt.xs("regime x2 + recency - recency", level="comparison")
_reg_helps = {m: bool((r.mean_diff > 0) and (r.folds_better >= 2) and (r.folds_worse == 0)) for m, r in _key.iterrows()}
_d26 = not any(_reg_helps[m] for m in ("T1 sponsor_hist", "T2 eb_rate_t0safe"))
_r119 = _REG.get(IN_PROGRESS)
record_check("D", "D26", "Recency-only training is not beaten by regime-matched weighting (regime x2 + recency vs recency, S3 test folds)",
             bool(_d26), metric="; ".join(f"{m} {r.mean_diff:+.4f}" for m, r in _key.iterrows()),
             threshold="FAIL if, for T1 sponsor_hist or T2 eb_rate_t0safe, regime + recency beats recency alone on average AND is "
                       "significantly better (paired cluster CI) in >= 2 folds and worse in none",
             by_split=_bs({"S2": bool(_d26)}), quantity="regime_weighting", kind="drift", t0_task2=0,
             na_reason="only a sliding window drops older same-regime Congresses; expanding splits keep them (weighting is a remedy, not a split)",
             detail="Mean paired diff (folds better/worse): " + "; ".join(f"{m} {cmp}: {r.mean_diff:+.4f} ({int(r.folds_better)}/{int(r.folds_worse)})"
                                                                          for (m, cmp), r in _rgt.iterrows() if "regime" in cmp)
             + f". Deployment Congress {IN_PROGRESS} regime {_r119}; same-regime Congresses in {CONGRESSES[0]}-{CONGRESSES[-1]}: "
               f"{[c for c in CONGRESSES if _REG.get(c) == _r119]}. A FAIL means training should upweight same-regime Congresses "
               "(which a sliding window drops, hence by_split S2) rather than only recent ones; a PASS means recency weighting "
               "(or the expanding window) is enough for these baselines. Regime weights are known at the start of a Congress.")

# %% [markdown]
# ### D16. Task 2 sensitivity to information not available at prediction time
#
# CRS assigns the policy area some days after introduction, so a Task 2 model scored at introduction
# often does not have it. `eb_rate_t0safe` hides the policy area for the share of bills that lack it
# when fresh (measured on the in-progress 119th Congress). If the score drops noticeably, the Task 2
# feature set must be built "as of" the prediction time, otherwise every split overstates performance.

# %%
# D16: eb_rate vs eb_rate_t0safe on S3 test folds (policy area hidden for fresh bills).
_ts = _sel(*_D_PRIMARY_T2, split="S3", eval_role="test").pivot_table(index="eval_congress", columns="baseline", values="value")
_ts["drop"] = _ts.eb_rate - _ts.eb_rate_t0safe
print(_ts[["eb_rate", "eb_rate_t0safe", "drop"]].round(4).T.to_string())
record_check("D", "D16", "Task 2: AUROC drop when policy area is unavailable at introduction (eb_rate - eb_rate_t0safe, S3 test)",
             None, metric=float(_ts["drop"].mean()), threshold="informational",
             detail=f"Policy area hidden for {D_PA_MISS_T0:.0%} of evaluation bills (share missing among the {_D_PA_N} 119th bills "
                    f"<= {D_PA_FRESH_DAYS} days old at the snapshot, a small-sample proxy; {_D_PA_MISS_30:.0%} at <= 30 days). Mean drop {_ts['drop'].mean():.3f} AUROC "
                    f"(max {_ts['drop'].max():.3f}). The drop is roughly uniform across folds, so it shifts all splits alike "
                    "rather than favouring one; build Task 2 features as of the prediction time.")

# %% [markdown]
# ### D18. Does the candidate pool definition matter?
#
# Task 1 ranks the point-in-time pool (members in office in the month of introduction + t0). The
# legacy pool (everyone who served in the chamber at any time that Congress) also contains departed and
# not-yet-seated members, which inflates the negatives. We rerun the S3 test folds with the legacy pool
# to see whether scores or the ordering of baselines depend on that choice.

# %%
# D18: point-in-time vs legacy candidate pool on S3 test folds (t0=0), Task 1.
_rs = []
for c in FOLD_C:
    (pp, cp_), (pl, cl) = _fold_unit("S3", c)["cosponsor_t0"], _fold_unit("S3", c, pit=False)["cosponsor_t0"]
    for b in ("sponsor_hist", "rw2hop", "popularity"):
        _rs.append(dict(congress=c, baseline=b, point_in_time=pp[b][:, 1].mean(), legacy=pl[b][:, 1].mean(),
                        n_pit=len(cp_), n_legacy=len(cl)))
_rs = pd.DataFrame(_rs)
_rs["diff"] = _rs.point_in_time - _rs.legacy
_pl = np.concatenate(_D_LOG["pool_legacy"])
print("Point-in-time minus legacy pool, mean Recall@20 over S3 test folds: " + ", ".join(
    f"{b} {g['diff'].mean():+.4f}" for b, g in _rs.groupby("baseline")))
_ord_same = (_rs.pivot_table(index="congress", columns="baseline", values="point_in_time").rank(axis=1)
             == _rs.pivot_table(index="congress", columns="baseline", values="legacy").rank(axis=1)).all(axis=1).mean()
record_check("D", "D18", "Task 1: scores and baseline ordering are insensitive to the candidate pool (point-in-time vs legacy roster, S3)",
             None, metric=float(_rs[_rs.baseline == "sponsor_hist"]["diff"].mean()), threshold="informational",
             detail=f"Mean Recall@20 point-in-time minus legacy: " + ", ".join(
                 f"{b} {g['diff'].mean():+.4f}" for b, g in _rs.groupby("baseline")) + f"; baseline ordering identical in "
                    f"{_ord_same:.0%} of folds (any swap is between the tied sponsor_hist and rw2hop). Median eligible pool {np.median(_pp):.0f} (point-in-time) vs {np.median(_pl):.0f} "
                    f"(legacy); {_D_LOG['pos_not_in_office'] / max(_D_LOG['pos_pit_total'], 1):.1%} of positives fall outside the "
                    "point-in-time pool (members joining mid-window or roster approximation) and are dropped. The pool choice "
                    "shifts all splits alike; the notebook uses the point-in-time pool everywhere.")

# %% [markdown]
# ### D19. Refit on train + validation, or keep the validation-selected model?
#
# Test units "select on val, refit on train + val". A GNN chooses its epoch count by early stopping
# on val, and the refit has no val left to stop on, so the refit must reuse a fixed budget (the epoch
# or hyperparameter value chosen on val). For each S3 fold we compare the test score of the
# val-selected family member **without refit** (fit on 108..c-2) and **with refit** (same member, fit
# on 108..c-1). If refitting does not help, keeping the val-selected checkpoint (and its early-stopping
# signal) loses nothing; if it helps, refit with the val-chosen budget.

# %%
# D19: refit (train+val) vs no refit (train only) for the val-selected member of each family, S3 test folds.
_rf = []
for c in FOLD_C:
    ur, un = _fold_unit("S3", c), _fold_unit("S3norefit", c)
    for fam, task, metric in (("T1 sponsor_hist family", *_D_PRIMARY_T1), ("T1 rw2hop family", *_D_PRIMARY_T1),
                              ("T2 eb_rate family", "advancement", "AUROC")):
        pick = _ORD[(fam, metric, "S3")].loc[f"c={c}", "val_pick"]
        d, lo, hi, hw = _pdiff(ur, pick, un, pick, task, metric, ("refit", c, fam))
        _rf.append(dict(family=fam, congress=c, pick=pick, gain=d, hw=hw, sig_pos=lo > 0, sig_neg=hi < 0))
    _pk = _SVD_PICK[c]                                    # learned model: val-picked member fit on 108..c-2 vs refit on 108..c-1
    d, lo, hi, hw = _pdiff(_LV[c], _pk, _learn_unit(_TR(CONGRESSES[0], c - 1), c, svd_names=(_pk,)), _pk, *_D_PRIMARY_T1, ("refit_svd", c))
    _rf.append(dict(family="T1 svd (val pick)", congress=c, pick=_pk, gain=d, hw=hw, sig_pos=lo > 0, sig_neg=hi < 0))
    urs, uns = _fold_unit("S3", c, stream=True), _fold_unit("S3norefit", c, stream=True)
    for b in ("sponsor_hist", "rw2hop"):                      # same comparison when eval-Congress history is streamed
        d, lo, hi, hw = _pdiff(urs, b, uns, b, *_D_PRIMARY_T1, ("refit_stream", c, b))
        _rf.append(dict(family=f"T1 {b} (+stream)", congress=c, pick=b, gain=d, hw=hw, sig_pos=lo > 0, sig_neg=hi < 0))
_rf = pd.DataFrame(_rf)
_rft = _rf.groupby("family").agg(mean_gain=("gain", "mean"), min_gain=("gain", "min"), max_gain=("gain", "max"),
                                 mean_hw=("hw", "mean"), folds_refit_better=("sig_pos", "sum"), folds_refit_worse=("sig_neg", "sum"))
print("Refit (fit on 108..c-1) minus no refit (fit on 108..c-2), same val-selected member, S3 test Congress c:")
print(_rft.round(4).to_string())
bar_chart(_rf[~_rf.family.str.contains("stream")].assign(fold="c=" + _rf.congress.astype(str))[["fold", "gain", "family"]],
          x="fold", y="gain", series="family",
          yfmt="+.4f", ylabel="refit − no refit", title="Test-score gain from refitting on train + validation (S3 folds)",
          subtitle="same hyperparameters (chosen on val); above 0 = refitting with the val Congress helps", height=300,
          name="D_refit_gain")
_worse = _rft[_rft.folds_refit_worse >= 2].index.tolist()
_better = _rft[_rft.folds_refit_better >= len(FOLD_C) / 2].index.tolist()
record_check("D", "D19", "Refitting on train+val with hyperparameters fixed from val does not hurt the test score (S3 folds)",
             not _worse, metric=float(_rft.mean_gain.min()), kind="validity",
             threshold="FAIL if refit is significantly worse (paired cluster CI) in >= 2 of 7 folds for any family",
             detail="Mean gain (folds refit signif. better / worse): " + "; ".join(
                 f"{k} {r.mean_gain:+.4f} ({int(r.folds_refit_better)}/{int(r.folds_refit_worse)})" for k, r in _rft.iterrows())
             + f". Refit clearly helps for: {', '.join(_better) or 'none'}. With streaming, refit still adds "
               f"{_rft.loc['T1 sponsor_hist (+stream)', 'mean_gain']:+.4f} (sponsor_hist) / {_rft.loc['T1 rw2hop (+stream)', 'mean_gain']:+.4f} "
               "(rw2hop), so the eval Congress's own history does not replace the val Congress. For count baselines 'refit' = "
               "adding the val Congress's edges to the graph; for a GNN that is two separate things: (1) the val Congress's edges "
               "must always be in the message-passing graph at test time (no retraining needed), (2) retraining weights on "
               "train+val should reuse the val-chosen epoch count; whether (2) adds beyond (1) can only be measured with the GNN "
               "(compare on the S3 folds as here).")

# %% [markdown]
# ### D27. A locked final test: develop on folds 112-117, report once on 118
#
# In S3 / S2 every test Congress 112-117 is also the validation Congress of the next fold, and D9
# recommends comparing models on the mean over rolling test folds. If the team compares GNN
# architectures or feature sets on that mean and then reports it, the reported number was chosen on
# test data. The fix is a **locked test**: develop, tune and compare on the rolling folds 112-117 only
# (hyperparameters on each fold's val Congress, architectures on the mean over the 6 development folds
# or on the mean of their val Congresses, D17), then freeze the pipeline (features, model, val-chosen
# epochs), refit on 108-117 and score Congress 118 **once**. Congress 119 is the prospective test once
# its windows close. This cell measures what reserving 118 costs: whether the selection checks still
# pass on 6 folds, how much the miss risk (D9) and the level CI (D15) grow from k = 7 to k = 6, and
# whether the development folds predict the locked fold (118 inside the development prediction interval).
# In the chart each row is a relevant model pair: grey dots = development folds, band = 95% prediction
# interval for a new Congress from those folds, orange dot = the locked fold 118 with its paired CI.

# %%
# D27: final-test isolation; selection, sizing and level precision on dev folds (k=6) vs all folds (k=7); 118 as locked test.
_kd, _ka = len(_DEV), len(FOLD_C)
_dev_f = [f"c={c}" for c in _DEV]
_lk = []
for fam, metric in (("T1 sponsor_hist family", "Recall@20"), ("T1 rw2hop family", "Recall@20"),
                    ("T2 eb_rate family", "AUROC"), ("T2 eb_rate_t0safe family", "AUROC")):
    o = _ORD[(fam, metric, "S3")]
    od = o[o.index.isin(_dev_f)]
    _lk.append(dict(item=fam.replace(" family", ""), regret_k7=o.regret.mean(), sig_k7=int(o.regret_sig.sum()),
                    ok_k7=_sel_verdict(fam, metric, "S3"), regret_k6=od.regret.mean(), sig_k6=int(od.regret_sig.sum()),
                    ok_k6=_sel_verdict(fam, metric, "S3", folds=_DEV),
                    ok_k6_3val=_sel_verdict(fam, metric, "S3", "regret_3val", "sig3", folds=_DEV),
                    ok_k6_S2=_sel_verdict(fam, metric, "S2", folds=_DEV)))
_ld = _LSEL[_LSEL.index.isin(_dev_f)]
_lk.append(dict(item="T1 svd (learned)", regret_k7=_LSEL.regret.mean(), sig_k7=int(_LSEL.regret_sig.sum()), ok_k7=_l_ok,
                regret_k6=_ld.regret.mean(), sig_k6=int(_ld.regret_sig.sum()),
                ok_k6=bool(not ((_ld.regret.mean() > D_TOL["cosponsor_t0"]) and (_ld.regret_sig.sum() >= 2))),
                ok_k6_3val=True, ok_k6_S2=True))
_lk = pd.DataFrame(_lk).set_index("item")
print(f"Selection on development folds {_DEV[0]}-{_DEV[-1]} (k={_kd}) vs all folds (k={_ka}): mean regret, significant folds, verdict")
print(_lk[["regret_k7", "sig_k7", "ok_k7", "regret_k6", "sig_k6", "ok_k6"]].round(4).to_string())

# Level precision (D15) and locked-fold coverage: per model, mean over dev folds, CI of the mean, and 118 vs the dev PI.
_lvl_src = {f"T1 {_best1}": _sel(*_D_PRIMARY_T1, _best1, split="S3", eval_role="test").set_index("eval_congress").value,
            "T1 svd (val pick)": pd.Series(_LSEL.test.to_numpy(), index=[int(f[2:]) for f in _LSEL.index]),
            "T2 eb_rate_t0safe": _sel(*_D_PRIMARY_T2, "eb_rate_t0safe", split="S3", eval_role="test").set_index("eval_congress").value,
            "T2 eb_rate (policy area)": _sel(*_D_PRIMARY_T2, "eb_rate", split="S3", eval_role="test").set_index("eval_congress").value}
_lv = []
for k, v in _lvl_src.items():
    v.index = v.index.astype(int)
    vd = v.reindex(_DEV)
    _lv.append(dict(model=k, mean_k7=v.mean(), hw_mean_k7=1.96 * v.std(ddof=1) / np.sqrt(_ka), mean_k6=vd.mean(),
                    hw_mean_k6=1.96 * vd.std(ddof=1) / np.sqrt(_kd), pi_dev=_tq6 * vd.std(ddof=1), locked_118=v[_LOCK],
                    in_pi=bool(abs(v[_LOCK] - vd.mean()) <= _tq6 * vd.std(ddof=1))))
_lv = pd.DataFrame(_lv).set_index("model")
print(f"Reported level: mean over test folds and its 95% CI half-width (k={_ka} vs k={_kd}); {_LOCK} vs the dev prediction interval:")
print(_lv.round(4).to_string())

# Paired sizing (D9) without the locked fold.
_rel6 = _prs[_prs.kind.isin(["model", "near-tie"]) & _prs.relevant_dev]
_m6, _m7 = float(_rel6.miss_k6.max()) if len(_rel6) else 0.0, float(_dec_pairs.miss_k7.max()) if len(_dec_pairs) else 0.0
_cov = _prs[_prs.kind.isin(["model", "near-tie", "protocol"]) & (_prs.relevant | _prs.relevant_dev)]
print(f"Paired sizing, relevant pairs: max miss risk k={_ka} {_m7:.3f} -> k={_kd} {_m6:.3f}; median CI half-width of the mean "
      f"difference {_dec_pairs.hw_mean_k7.median():.4f} -> {_rel6.hw_mean_k6.median():.4f}; {_LOCK} paired difference inside the "
      f"dev prediction interval for {int(_cov.lock_in_pi.sum())}/{len(_cov)} relevant model / protocol pairs")

_LOCK_JS = r"""
const {width, height, title, subtitle, xlabel} = opts;
const m = {top: 48, right: 24, bottom: 56, left: 250};
const svg = d3.select(el).append('svg').attr('width', width).attr('height', height);
viz.title(svg, title, subtitle);
const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
const W = width - m.left - m.right, H = height - m.top - m.bottom;
const rows = Array.from(new Set(data.map(d => d.pair)));
const ys = d3.scaleBand(rows, [0, H]).padding(0.3);
const xs = d3.scaleLinear([Math.min(0, d3.min(data, d => d.lo)), d3.max(data, d => d.hi)], [0, W]).nice();
g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xs).ticks(6).tickSize(-H).tickFormat(d3.format('+.2f')))
  .call(a => a.selectAll('line').attr('stroke', viz.grid)).call(a => a.select('.domain').remove())
  .call(a => a.selectAll('text').attr('fill', viz.ink2).style('font', '11px sans-serif'));
g.append('g').call(d3.axisLeft(ys).tickSize(0)).call(viz.axis).call(a => a.select('.domain').remove());
g.append('line').attr('x1', xs(0)).attr('x2', xs(0)).attr('y1', 0).attr('y2', H).attr('stroke', viz.ink).attr('stroke-width', 1);
g.append('text').attr('x', W).attr('y', H + 34).attr('text-anchor', 'end').attr('fill', viz.ink2).style('font', '11px sans-serif').text(xlabel);
const cy = d => ys(d.pair) + ys.bandwidth() / 2;
const tip = viz.tooltip(el);
g.selectAll(null).data(data.filter(d => d.kind === 'band')).join('rect').attr('x', d => xs(d.lo)).attr('width', d => Math.max(xs(d.hi) - xs(d.lo), 1))
  .attr('y', d => ys(d.pair)).attr('height', ys.bandwidth()).attr('fill', viz.palette[0]).attr('opacity', 0.18).attr('rx', 2)
  .on('mousemove', (e, d) => tip.show(`<b>${d.pair}</b><br>dev mean ${d3.format('+.4f')(d.value)}<br>95% PI [${d3.format('+.4f')(d.lo)}, ${d3.format('+.4f')(d.hi)}]`, e))
  .on('mouseleave', () => tip.hide());
g.selectAll(null).data(data.filter(d => d.kind === 'dev')).join('circle').attr('cx', d => xs(d.value)).attr('cy', cy).attr('r', 3.5)
  .attr('fill', '#8a8984').attr('opacity', 0.8)
  .on('mousemove', (e, d) => tip.show(`<b>${d.pair}</b><br>dev fold ${d.congress}: ${d3.format('+.4f')(d.value)}`, e))
  .on('mouseleave', () => tip.hide());
const lk = data.filter(d => d.kind === 'lock');
g.selectAll(null).data(lk).join('line').attr('x1', d => xs(d.lo)).attr('x2', d => xs(d.hi)).attr('y1', d => cy(d) + 6).attr('y2', d => cy(d) + 6)
  .attr('stroke', viz.palette[1]).attr('stroke-width', 2);
g.selectAll(null).data(lk).join('circle').attr('cx', d => xs(d.value)).attr('cy', d => cy(d) + 6).attr('r', 5)
  .attr('fill', viz.palette[1]).attr('stroke', '#fff').attr('stroke-width', 1.5)
  .on('mousemove', (e, d) => tip.show(`<b>${d.pair}</b><br>locked fold ${d.congress}: ${d3.format('+.4f')(d.value)} [${d3.format('+.4f')(d.lo)}, ${d3.format('+.4f')(d.hi)}]`, e))
  .on('mouseleave', () => tip.hide());
viz.legend(svg, [['development folds', '#8a8984'], ['95% PI for a new Congress (dev folds)', viz.palette[0]], ['locked fold (paired CI)', viz.palette[1]]], m.left, height - 10);
"""
_pdl = pd.concat(_pd_rows, ignore_index=True)
_ch = []
for r in _cov.itertuples():
    lab = f"{r.task} {r.pair}"
    dd = _pdl[(_pdl.pair == r.pair) & _pdl.congress.isin(_DEV)]
    _ch.append(dict(pair=lab, kind="band", value=r.mean_dev, lo=r.mean_dev - r.pi_dev, hi=r.mean_dev + r.pi_dev, congress=0))
    _ch += [dict(pair=lab, kind="dev", value=x, lo=x, hi=x, congress=int(c)) for x, c in zip(dd["diff"], dd.congress)]
    _ch.append(dict(pair=lab, kind="lock", value=r.lock_diff, lo=r.lock_lo, hi=r.lock_hi, congress=_LOCK))
show_d3(_LOCK_JS, pd.DataFrame(_ch), height=110 + 40 * max(len(_cov), 1), width=760,
        opts=dict(title=f"Do the development folds {_DEV[0]}-{_DEV[-1]} predict the locked test Congress {_LOCK}?",
                  subtitle="paired test difference per relevant model / protocol pair (Task 1 Recall@20, Task 2 AUROC at introduction)",
                  xlabel=f"paired difference (MDE = {D_MDE['cosponsor_t0']})"), name="D_locked_test")

_sel_ok6 = bool(_lk.ok_k6.all())
_d27 = _sel_ok6 and (_m6 <= 0.2)
_d27_s2 = bool(_lk.drop(index="T1 svd (learned)").ok_k6_S2.all()) and (_m6 <= 0.2)
_dhw = _lv.hw_mean_k6 - _lv.hw_mean_k7
record_check("D", "D27", f"Final-test isolation: developing on rolling folds {_DEV[0]}-{_DEV[-1]} and reserving {_LOCK} as a locked test "
             "keeps selection and paired sizing adequate", bool(_d27), metric=round(_m6, 3), kind="validity",
             threshold=f"PASS if every selection check (D11/D12/D12b/D21/D23 rule, regret tol MDE/2) still passes on the {_kd} dev "
                       f"folds AND the max miss risk over relevant pairs at k={_kd} is <= 0.20",
             by_split=_bs({"S3": bool(_d27), "S3+stream": bool(_d27), "S2": bool(_d27_s2)}),
             na_reason={**{s_: "one test Congress, reported once: isolated by construction, nothing to reserve (its size is judged in D9)"
                           for s_ in ("S1a", "S1b", "S1a+stream", "S1b+stream", "S1pool")},
                        "S4": "within-Congress split: no rolling reuse of test Congresses", "S5": "random split: leaky reference only"},
             detail=f"Selection with k={_kd} (k={_ka}): " + "; ".join(f"{i} regret {r.regret_k6:.4f} ({r.sig_k6}/{_kd} signif., "
                    f"{'pass' if r.ok_k6 else 'FAIL'}) [{r.regret_k7:.4f}, {r.sig_k7}/{_ka}]" for i, r in _lk.iterrows())
                    + f". 3-val-fold selection (D17) on dev folds passes for all families: {bool(_lk.ok_k6_3val.all())}. Paired sizing: max "
                    f"miss risk {_m7:.3f} (k={_ka}) -> {_m6:.3f} (k={_kd}); median CI half-width of a mean difference "
                    f"{_dec_pairs.hw_mean_k7.median():.4f} -> {_rel6.hw_mean_k6.median():.4f}. Reported-level CI half-width (D15) changes by (k={_kd} minus k={_ka}) "
                    + ", ".join(f"{k} {v:+.4f}" for k, v in _dhw.items())
                    + f". Locked fold {_LOCK}: level inside the dev prediction interval for "
                    + ", ".join(f"{k} {'yes' if r.in_pi else 'NO'} ({r.locked_118:.3f} vs dev {r.mean_k6:.3f} +/- {r.pi_dev:.3f})" for k, r in _lv.iterrows())
                    + f"; paired differences inside the dev PI for {int(_cov.lock_in_pi.sum())}/{len(_cov)} relevant pairs (outside: "
                    + (", ".join(f"{r.task} {r.pair} {r.lock_diff:+.4f} vs dev {r.mean_dev:+.4f} +/- {r.pi_dev:.4f}"
                                 for r in _cov[~_cov.lock_in_pi].itertuples()) or "none")
                    + f"). A locked level outside the dev interval is drift in {_LOCK} (sections A / C), not a protocol error: report "
                      "it as measured and do not re-tune on it. Protocol: "
                      f"select on folds {_DEV[0]}-{_DEV[-1]} (val c-1, test c; architectures on the dev-fold mean or the D17 val mean), "
                      f"freeze, refit on {CONGRESSES[0]}-{_LOCK - 1}, report {_LOCK} once, then {IN_PROGRESS} prospectively. A FAIL "
                      "would mean the reserved Congress is needed for reliable selection (then report the 7-fold mean and say it was "
                      "used for selection).")
D_FINAL_PROTOCOL = dict(
    develop=f"S3(+stream) rolling folds {_DEV[0]}-{_DEV[-1]}: train {CONGRESSES[0]}..c-2 -> val c-1 -> refit {CONGRESSES[0]}..c-1 -> test c",
    select_hyperparameters="per fold on the val Congress c-1 (or the mean of the last 3 val Congresses, D17)",
    compare_architectures=f"mean over the {_kd} development test folds (paired, same bills); never on {_LOCK}",
    locked_test=_LOCK, locked_train=list(range(CONGRESSES[0], _LOCK)),
    prospective_test=IN_PROGRESS, mde=D_MDE, reserve_cost=dict(max_miss_k7=_m7, max_miss_k6=_m6, selection_ok_k6=_sel_ok6))
print("D_FINAL_PROTOCOL: " + " | ".join(f"{k}: {v}" for k, v in D_FINAL_PROTOCOL.items() if k in ("develop", "locked_test", "prospective_test")))

# %% [markdown]
# ### Section D summary
#
# One line per check. FAIL does not automatically rule out a split: read the detail for the
# implication (e.g. report the mean over rolling folds instead of a single test Congress).

# %%
# Summary of section D checks (with the splits each check fails for) and the objects exposed to the decision section.
_dc = pd.DataFrame([c for c in CHECKS if c["section"] == "D"])
_dc["result"] = _dc.passed.map({True: "PASS", False: "FAIL"}).fillna("INFO")
_dc["metric"] = _dc.metric.map(lambda v: f"{v:.4g}" if isinstance(v, (int, float, np.floating)) else str(v))
_dc["fails_for"] = _dc.by_split.map(lambda d: ",".join(k for k, v in (d or {}).items() if v is False))
for _res in ("FAIL", "PASS", "INFO"):                     # compact: the full condition lines are printed above
    _x = _dc[_dc.result == _res]
    print(f"{_res} ({len(_x)}): " + ", ".join(r.id + (f" [fails for {r.fails_for}]" if r.fails_for else "") for r in _x.itertuples()))
print(f"\nSPLIT_DEFS + SPLIT_VARIANTS (setup): {ALL_SPLITS} | SPLIT_RESULTS: {len(SPLIT_RESULTS):,} rows "
      f"({SPLIT_RESULTS.task.nunique()} tasks x {SPLIT_RESULTS.baseline.nunique()} baselines; CIs = cluster bootstrap by sponsor, "
      f"ci_*_bill = bill bootstrap) | D_FAMILY_RESULTS: {len(D_FAMILY_RESULTS):,} rows | D_PAIRED: paired-difference sizing | "
      "D_LEARNED: SVD selection per fold | D_STALENESS: recall by legislator status | D_REGIME: weighting comparison | "
      "D_LOCKED: selection / level on dev folds vs all folds | D_FINAL_PROTOCOL: locked-test protocol")
D_PAIRED, D_LEARNED, D_STALENESS, D_REGIME, D_LOCKED = _prs, _LSEL, _strt, _rgt, dict(selection=_lk, level=_lv)
