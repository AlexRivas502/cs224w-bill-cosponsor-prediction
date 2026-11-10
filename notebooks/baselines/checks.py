"""Sanity checks (DESIGN section 9). Each function raises AssertionError on failure and returns a summary.

- check_eligibility: sponsor and already-seen (lag <= t0) cosponsors are excluded; positives are exactly the
  in-pool cosponsors with lag in (t0, t0+30].
- truncation_test: features of bills whose cutoff is day D are identical when every Congress-k event (and
  referral) dated after D is deleted (pair features, x_b, the bills' target edges).
- perturbation_test: a trained model's scores of those bills are bit-identical after shifting every
  Congress-k event dated after D by +1000 days (threads = 1).
- check_t2_plain: no same-Congress path for Task 2 (no tbill sources, hbills from W_k only, leg / comm
  features read W_k only).
- check_caucus_pit: caucus_pit(k) is unchanged when all Congress-k cosponsorships are dropped.
- check_censored_eb: censored history bills never enter an EB-rate denominator.
- check_depth, check_allowlist, check_reference_bars.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import torch

from .data import INTMAX, WINDOW_DAYS, Corpus, T1Block, caucus_pit, hist_window, t1_blocks
from .features import (FORBIDDEN_PATTERNS, PHI_ALL, _per_congress, assert_allowed, leg_static, pair_features,
                       tbill_features, windowed_counts)
from .graphs import TARGET_RELS, build_graph, check_graph, standardized

# Fold-117 reference bars (digest_D.md); tolerance 5e-4 (verbatim ports), +-1e-3 where the digest rounds.
REFERENCE_BARS = [
    # (task, protocol, role, model, value, tol)
    ("t1", "plain", "test", "popularity", 0.1374843983, 5e-4),
    ("t1", "plain", "test", "party+pop", 0.1905368825, 5e-4),
    ("t1", "plain", "test", "sponsor_hist", 0.2615268317, 5e-4),
    ("t1", "plain", "test", "rw2hop", 0.2628509525, 5e-4),
    ("t1", "plain", "test", "svd_r512_hl1", 0.2832, 5e-4),
    ("t1", "stream", "test", "sponsor_hist", 0.3070785531, 5e-4),
    ("t1", "stream", "test", "rw2hop", 0.308, 1e-3),
    ("t1", "plain", "val", "sponsor_hist", 0.2542, 5e-4),
    ("t1", "stream", "val", "sponsor_hist", 0.2884, 5e-4),
    ("t1", "plain", "val", "svd_r512_hl1", 0.2757, 5e-4),
    ("t2", "plain", "test", "chamber_maj", 0.5448527364, 5e-4),
    ("t2", "plain", "test", "eb_rate_t0safe", 0.5598481668, 5e-4),
    ("t2", "plain", "test", "eb_rate", 0.6923597393, 5e-4),
]
N_T1_TEST_117, N_T2_TEST_117 = 5101, 12953


def check_eligibility(corpus: Corpus, blocks: list[T1Block]) -> dict:
    """Assert Excl contains the sponsor and every cosponsor with lag <= t0, Pos and Excl are disjoint, every
    kept bill has a positive, and positives are exactly the in-pool, non-excluded cosponsors with lag in
    (t0, t0 + 30]."""
    CD = corpus.CD
    n_sp, n_seen, n_pos = 0, 0, 0
    for b in blocks:
        assert not (b.Pos & b.Excl).any(), "a positive is also excluded"
        assert b.Pos.any(1).all(), "kept bill without positives"
        rows = np.arange(b.nb)
        hs = b.sp_col >= 0
        assert b.Excl[rows[hs], b.sp_col[hs]].all(), "sponsor not excluded"
        n_sp += int(hs.sum())
        colpos = pd.Series(np.arange(b.M), index=b.cand)
        rp = pd.Index(b.codes).get_indexer(CD["bill"])
        m = rp >= 0
        lag, leg, rp = CD["lag"][m], CD["leg"][m], rp[m]
        cp = colpos.reindex(leg).to_numpy()
        inpool = ~np.isnan(cp)
        cp = np.nan_to_num(cp, nan=0).astype(int)
        seen = (lag <= b.t0) & inpool
        assert b.Excl[rp[seen], cp[seen]].all(), "already-seen cosponsor not excluded"
        n_seen += int(seen.sum())
        win = (lag > b.t0) & (lag <= b.t0 + WINDOW_DAYS) & inpool
        expect = np.zeros_like(b.Pos)
        expect[rp[win], cp[win]] = True
        expect &= ~b.Excl
        assert np.array_equal(expect, b.Pos), "Pos differs from the (t0, t0+30] in-pool cosponsors"
        n_pos += int(b.Pos.sum())
    return dict(blocks=len(blocks), bills=int(sum(b.nb for b in blocks)), sponsors_excluded=n_sp,
                seen_excluded=n_seen, positives=n_pos)


def check_allowlist(names_by_type: dict) -> dict:
    """Assert no feature name of any type matches FORBIDDEN_PATTERNS."""
    for t, names in names_by_type.items():
        assert_allowed(names)
    return {t: len(n) for t, n in names_by_type.items()} | {"patterns": len(FORBIDDEN_PATTERNS)}


def check_depth(congresses, H: int) -> dict:
    """Assert hist_depth in {1, 2, 3} for every Congress used and that windowed counts read W_k only."""
    out = {}
    for k in congresses:
        W = hist_window(k, H)
        assert 1 <= len(W) <= 3 or H > 3, f"hist_depth {len(W)} for Congress {k}"
        out[k] = len(W)
    return out


def check_windowed_reads(corpus: Corpus, k: int, H: int) -> None:
    """Assert every windowed count table of Congress k reads only W_k."""
    W = set(hist_window(k, H))
    for kind in ("pop", "sc", "spons", "co", "rep_n"):
        assert set(windowed_counts(corpus, k, H, kind).congresses) <= W


def check_t2_plain(graph, corpus: Corpus, k: int, H: int) -> dict:
    """No same-Congress path for Task 2 under 'plain': target bills are never message sources, history bills
    come from W_k only, and legislator / committee features read W_k only."""
    for et in graph.edge_types:
        assert et[0] != "tbill", f"{et} has a tbill source"
    for et in TARGET_RELS:
        assert et[2] == "tbill"
    hc = set(np.unique(graph["hbill"].congress.numpy()).tolist())
    assert k not in hc and hc <= set(hist_window(k, H))
    check_windowed_reads(corpus, k, H)
    return dict(hbill_congresses=sorted(hc), target_rels=[e[1] for e in TARGET_RELS])


def check_caucus_pit(corpus: Corpus, k: int, H: int = 3) -> dict:
    """caucus_pit(k) must not change when every Congress-k (and later) cosponsorship is dropped."""
    a = caucus_pit(corpus, k, H)
    trunc = dataclasses.replace(corpus, cosp=corpus.cosp[corpus.cosp.congress < k])
    b = caucus_pit(trunc, k, H)
    assert np.array_equal(a, b), "caucus_pit reads Congress >= k events"
    return dict(k=k, n_other=int((a == "other").sum()))


def check_censored_eb(corpus: Corpus, kk: int) -> dict:
    """Censored law-capable bills of Congress kk never enter EB-rate denominators."""
    pc = _per_congress(corpus, kk)
    adv = corpus.adv[corpus.adv.congress == kk]
    n_unc = int((~adv.censored).sum())
    assert int(pc["rep_n"].sum()) <= n_unc, "EB denominator includes censored bills"
    assert set(pc["adv_codes"]).isdisjoint(set(corpus.billix.get_indexer(adv[adv.censored].bill_id)))
    return dict(congress=kk, denominator=int(pc["rep_n"].sum()), uncensored=n_unc, censored=int(adv.censored.sum()))


# --- Stream truncation and perturbation ----------------------------------------------------------------
def modified_corpus(corpus: Corpus, k: int, D: int, mode: str = "truncate", shift: int = 1000) -> Corpus:
    """Copy of the corpus where Congress-k cosponsorships and referrals dated after day D (and undated ones)
    are deleted (mode 'truncate') or moved +shift days later (mode 'shift')."""
    CD = dict(corpus.CD)
    late = (CD["cong"] == k) & ((CD["day"] > D) | (CD["day"] == INTMAX))
    ref = corpus.referrals.copy()
    rlate = (ref.congress == k) & (ref.day > D)
    if mode == "truncate":
        keep = ~late
        CD = {n: v[keep] for n, v in CD.items()}
        ref = ref[~rlate].reset_index(drop=True)
    else:
        movable = late & (CD["day"] != INTMAX)
        CD["day"] = np.where(movable, CD["day"] + shift, CD["day"])
        CD["lag"] = np.where(movable, CD["lag"] + shift, CD["lag"])
        ref.loc[rlate, "day"] = ref.loc[rlate, "day"] + shift
    return dataclasses.replace(corpus, CD=CD, referrals=ref)


def pick_cut_days(blocks: list[T1Block], n: int = 3) -> list[int]:
    """Early, middle and late cutoff days that have bills in the blocks."""
    cuts = np.unique(np.concatenate([b.cut for b in blocks]))
    return [int(cuts[int(q * (len(cuts) - 1))]) for q in np.linspace(0.1, 0.9, n)]


def truncation_test(corpus: Corpus, k: int, t0: int, H: int = 3, days=None, title=None) -> list[dict]:
    """For each cutoff day D: phi (all columns), x_b and the target edges of bills with cut == D are identical
    when computed from events truncated to day <= D. Also asserts max(event day used) <= cut."""
    from .features import fit_title_svd, tbill_codes
    blocks = t1_blocks(corpus, k, t0)
    days = days or pick_cut_days(blocks)
    title = title or fit_title_svd(corpus, k - 1)
    Xf, names, codes = tbill_features(corpus, k, t0, H, title)
    out = []
    for D in days:
        tc = modified_corpus(corpus, k, D, "truncate")
        Xt, _, codes_t = tbill_features(tc, k, t0, H, title)
        assert np.array_equal(codes, codes_t)
        tb_day = corpus.BL.day.reindex(codes).to_numpy()
        sel = tb_day + t0 == D
        assert np.array_equal(Xf[sel], Xt[sel]), f"x_b changes under truncation at day {D}"
        n_phi = 0
        for b in blocks:
            r = np.flatnonzero(b.cut == D)
            if not len(r):
                continue
            sb = b.subset(r)
            dbg = {}
            pf, _ = pair_features(corpus, sb, "stream", H, debug=dbg)
            pt, _ = pair_features(tc, sb, "stream", H)
            assert np.array_equal(pf, pt), f"phi changes under truncation at day {D} ({b.chamber})"
            assert dbg["max_stream_day_minus_cut"] <= 0
            n_phi += len(r)
        gf = build_graph(corpus, k, t0, H, title)
        gt = build_graph(tc, k, t0, H, title)
        tpos = np.flatnonzero(sel)
        for et in TARGET_RELS:
            ef, et_ = gf[et].edge_index.numpy(), gt[et].edge_index.numpy()
            a = sorted(map(tuple, ef[:, np.isin(ef[1], tpos)].T))
            bb = sorted(map(tuple, et_[:, np.isin(et_[1], tpos)].T))
            assert a == bb, f"target edges {et} change under truncation at day {D}"
        out.append(dict(day=D, n_tbill=int(sel.sum()), n_phi_bills=n_phi, ok=True))
    return out


def perturbation_test(model, data, corpus: Corpus, k: int, task: str = "t1", protocol: str = "stream",
                      days=None) -> list[dict]:
    """Shift every Congress-k event dated after D by +1000 days and rescore the bills with cut == D using the
    same trained model and the same standardisation: scores must be bit-identical (run with threads = 1)."""
    from .features import phi_columns
    from .train import Part
    cols = phi_columns(protocol)
    model.eval()
    blocks = data.blocks[k]
    days = days or pick_cut_days(blocks, 2)
    g0 = data.graphs[k]
    out = []
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for D in days:
            pc = modified_corpus(corpus, k, D, "shift")
            gp = standardized(build_graph(pc, k, data.t0, data.H, data.title), data.stds, data.comm_seen)
            with torch.no_grad():
                e0, e1 = model(g0), model(gp)
                n = 0
                for bi, b in enumerate(blocks):
                    r = np.flatnonzero(b.cut == D)
                    if not len(r):
                        continue
                    if task == "t1":
                        p0 = data.t1_part(k, bi, r, cols)
                        sb = b.subset(r)
                        phi1, _ = pair_features(pc, sb, "stream", data.H)
                        mu, sd = data.phi_stats(k)
                        p1 = dataclasses.replace(p0, phi=torch.from_numpy(
                            (phi1[..., cols].astype(np.float32) - mu[cols]) / sd[cols]))
                        s0, s1 = model.score_t1(e0, p0), model.score_t1(e1, p1)
                    else:
                        tb = torch.as_tensor(data.tpos[k].get_indexer(b.codes[r]))
                        sp = torch.as_tensor(b.spon[r])
                        s0 = model.score_t2(e0, Part(k, tb, sp, xb=g0["tbill"].x[tb]))
                        s1 = model.score_t2(e1, Part(k, tb, sp, xb=gp["tbill"].x[tb]))
                    assert torch.equal(s0, s1), f"scores change when post-cutoff events move (day {D})"
                    n += len(r)
            out.append(dict(day=D, bills=n, ok=True))
    finally:
        torch.set_num_threads(old_threads)
    return out


# --- Reference bars ---------------------------------------------------------------------------------------
def check_reference_bars(results, strict: bool = True) -> pd.DataFrame:
    """Compare fold-117 heuristic EvalResults with the split study's digest values (tolerance per bar).
    Returns a table; raises if strict and any bar is outside its tolerance or n differs."""
    from .evaluate import auroc
    rows = []
    for task, prot, role, model, ref, tol in REFERENCE_BARS:
        hit = [r for r in results if r.fold == 117 and r.task == task and r.protocol == prot and r.role == role
               and r.model == model and r.t0 == 0]
        if not hit:
            rows.append(dict(task=task, protocol=prot, role=role, model=model, reference=ref, value=np.nan,
                             diff=np.nan, tol=tol, n=0, ok=None))
            continue
        r = hit[0]
        v = float(r.per_bill[:, 1].mean()) if task == "t1" else auroc(r.scores, r.rows.y.to_numpy())
        n = len(r.codes)
        n_ok = role != "test" or n == (N_T1_TEST_117 if task == "t1" else N_T2_TEST_117)
        rows.append(dict(task=task, protocol=prot, role=role, model=model, reference=ref, value=v, diff=v - ref,
                         tol=tol, n=n, ok=bool(abs(v - ref) <= tol and n_ok)))
    df = pd.DataFrame(rows)
    if strict:
        bad = df[df.ok == False]   # noqa: E712
        assert bad.empty, "reference bars not reproduced:\n" + bad.to_string()
    return df


def check_population(results) -> dict:
    """Check 7: every Task-1 result of one evaluation unit (fold, role, t0) scores the same bills in the same order
    (and every Task-2 result the same rows). Units that mix quick-mode (Senate-only) learned models are skipped."""
    units: dict = {}
    for r in results:
        units.setdefault((r.task, r.fold, r.role, r.t0), []).append(r)
    n_ok = 0
    for key, rs in units.items():
        full = max(rs, key=lambda r: len(r.codes))
        for r in rs:
            if len(r.codes) == len(full.codes):
                assert np.array_equal(r.codes, full.codes), f"population differs for {r.model} in {key}"
                n_ok += 1
    return dict(units=len(units), results_checked=n_ok)


def direction_warnings(results, plain_models=("hetero_sage",)) -> list[str]:
    """Checks 9 and 11 as warnings: (9) plain-protocol models in plain_models must hit some new-legislator
    positives; (11) every GNN under +stream should beat party+pop on validation, and G1 +stream should be
    >= sponsor_hist +stream - 0.01 on validation."""
    out = []
    val = {(r.fold, r.t0, r.protocol, r.model, r.seed): r for r in results if r.task == "t1" and r.role == "val"}

    def score(r):
        return float(r.per_bill[:, 1].mean())

    for r in results:
        if r.task != "t1" or r.kind != "gnn":
            continue
        if r.protocol == "plain" and r.model in plain_models and r.role == "test":
            nh = (r.extra or {}).get("new_hits", {}).get("new", {})
            if nh and not nh.get("hits", 0) > 0:
                out.append(f"check 9: {r.model} plain fold {r.fold} has hit@20 = 0 on new-legislator positives")
        if r.protocol == "stream" and r.role == "val":
            pp = val.get((r.fold, r.t0, "stream", "party+pop", None))
            sh = val.get((r.fold, r.t0, "stream", "sponsor_hist", None))
            if pp is not None and len(pp.codes) == len(r.codes) and score(r) <= score(pp):
                out.append(f"check 11: {r.model} +stream val {score(r):.4f} <= party+pop {score(pp):.4f} (fold {r.fold})")
            if r.model == "hetero_sage" and sh is not None and len(sh.codes) == len(r.codes) and score(r) < score(sh) - 0.01:
                out.append(f"check 11: G1 +stream val {score(r):.4f} < sponsor_hist {score(sh):.4f} - 0.01 (fold {r.fold})")
    return out
