# %% [markdown]
# ## B. Graph and node conditions
#
# A GNN does not only see a table of bills: it sees **nodes** (legislators, bills, committees,
# subjects, policy areas) and **edges** (cosponsorship, sponsorship, committee referral, subject tags,
# related-bill links). A by-Congress split is only trustworthy for a graph model if
# (1) the nodes the model must score at test time are known, either from training or from history
# inside the test Congress that is available before the prediction time t0 (otherwise learned node
# embeddings are useless and the model must be *inductive*), (2) the graph's degree structure and
# density look the same across Congresses (message passing aggregates over neighbours, so a shift in
# degree changes every representation), (3) the relation schema and vocabularies (committee codes,
# subject terms, policy areas) are stable, so a "committee" or "subject" node means the same thing in
# train and test, and (4) no edge carries information from after t0 (related-bill links to later
# bills, committee edges that record the label, cosponsor edges dated after t0).
#
# **Evaluation pools.** Every pool here comes from the shared `SPLIT_DEFS` / `iter_eval_units()` (the
# same units section D backtests): S1a (train 108-115, val 116, test 117), S1b (train 108-116, val 117,
# test 118), S2 (sliding, k = `SLIDE_K`) and S3 (rolling, expanding) folds, plus the `S1pool` variant
# (val 116, test 117 + 118). Many units share the same (train Congresses, evaluated Congress) pair, so
# metrics are computed once per distinct **pool** (ids P01, P02, ...; the mapping to units is printed
# once below, later tables are keyed by pool id). Tables show the pools that evaluate 116-118; the
# per-split lines summarise all folds. The `+stream` variants (eval-Congress events dated <= intro + t0
# allowed) and S4 (within-Congress: prior Congresses + events before the cut date) are evaluated where
# the protocol changes the answer (cold start, sponsor history). S5 (random) ignores time and is not
# evaluated. Congress 108 is the first in the data, so "never seen before" is **left-censored** there.
#
# **Noise rule (no pool-size bias).** "Differs from train" verdicts use ONE fixed reference: the
# leave-one-out spread of the S1a train Congresses 108-115 (`REL_REF`, shared `rel_to_train`), whatever
# pool is evaluated. A short S2 pool is not judged against its own 3 Congresses (the max of 3 values is
# exceeded by chance about 1/(3+1) = 25% of the time). Because a split has many evaluation units, a split
# is judged on the **excess** of failures over the number expected by chance (binomial test: observed
# failures vs alpha x tests), not on "any unit fails".

# %%
# Shared helpers: evaluation pools from SPLIT_DEFS (+ S1pool), per-split summaries, noise-excess test, as-of edge counts.
B_ALL = CONGRESSES  # complete Congresses; 119 shown only where noted
B_EVAL = [116, 117, 118]
B_SPLITS = [s for s in ("S1a", "S1b", "S2", "S3") if s in SPLIT_DEFS]
_bu = [dict(split=u["split"], fold=u["fold"], role=u["role"], c=int(u["eval_congress"]),
            train=tuple(u["train"]), train_label=u["train_label"])
       for s in B_SPLITS for u in iter_eval_units(s)]
if "S1pool" in SPLIT_VARIANTS:   # protocol variant with its own pools: val 116 on 108-115, test 117 + 118 on 108-116
    _v = SPLIT_VARIANTS["S1pool"]
    _bu.append(dict(split="S1pool", fold="pool", role="val", c=_v["val"][0], train=tuple(_v["train"]),
                    train_label=_rng_label(_v["train"])))
    _bu += [dict(split="S1pool", fold="pool", role="test", c=c, train=tuple(_v["train"] + _v["val"]),
                 train_label=_rng_label(_v["train"] + _v["val"])) for c in _v["test"]]
B_UNITS = pd.DataFrame(_bu)
B_UNIT_SPLITS = list(dict.fromkeys(B_UNITS.split))
B_STREAM = {k: v["base"] for k, v in SPLIT_VARIANTS.items() if v.get("stream")}   # e.g. 'S1a+stream' -> 'S1a'
B_UNITS["tag"] = B_UNITS.split + " " + B_UNITS.role
B_POOLS = (B_UNITS.groupby(["train_label", "c"], sort=False)
           .agg(train=("train", "first"), used_by=("tag", lambda t: ", ".join(dict.fromkeys(t)))).reset_index())
B_POOLS["k"] = B_POOLS.train.map(len)
B_POOLS = B_POOLS.sort_values(["c", "k"], ascending=[True, False]).reset_index(drop=True)
B_POOLS["pool"] = [f"P{i + 1:02d}" for i in range(len(B_POOLS))]
B_UNITS = B_UNITS.merge(B_POOLS[["train_label", "c", "pool"]], on=["train_label", "c"])


def b_eval_pools(fn):
    """Apply fn(train_congress_list, eval_congress) -> dict to every distinct pool."""
    return pd.DataFrame([dict(pool=p.pool, train=p.train_label, eval=p.c, **fn(list(p.train), p.c))
                         for p in B_POOLS.itertuples()])


def b_show(df, cols, extra_key=()):
    """Print the pools that evaluate 116-118 (compact; pool ids -> units mapping printed once at the top)."""
    d = df[df["eval"].isin(B_EVAL)]
    print(d[["pool", "train", "eval", *extra_key, *cols]].round(3).to_string(index=False))


def b_by_split(df, col, agg="max", splits=None):
    """Per split: aggregate of `col` over all of the split's evaluation units."""
    m = B_UNITS.merge(df[["pool", col]], on="pool")
    return m.groupby("split", sort=False)[col].agg(agg).reindex(splits or B_UNIT_SPLITS)


def b_fmt_split(s, fmt=".1%"):
    return ", ".join(f"{k} {v:{fmt}}" for k, v in s.items() if pd.notna(v))


def b_split_dict(base, stream=None, extra=None):
    """by_split dict: base splits from `base`; '+stream' variants from `stream` (keyed by base split) or a
    copy of the base value when the protocol does not change the answer; `extra` adds e.g. S4."""
    out = {k: v for k, v in dict(base).items() if pd.notna(v)}
    src = dict(base) if stream is None else dict(stream)
    out.update({v: src[b] for v, b in B_STREAM.items() if b in src and pd.notna(src[b])})
    out.update(extra or {})
    return out


B_NA_DEFAULT = {
    "S5": "random bill split: ignores time and Congress boundaries (leaky reference); cross-Congress graph conditions do not apply",
    "S4": "within-Congress split: train and eval share one Congress's nodes / vocabulary; not evaluated by this check",
}


def b_bs(by_split, reasons=None):
    """record_check kwargs: by_split with a key for every split in ALL_SPLITS (None = not judged) and the
    na_reason for each None (defaults in B_NA_DEFAULT, overridden per split by `reasons`)."""
    bs = {k: (None if by_split.get(k) is None or (isinstance(by_split.get(k), float) and np.isnan(by_split.get(k)))
              else by_split.get(k)) for k in ALL_SPLITS}
    r = {**B_NA_DEFAULT, **(reasons or {})}
    return dict(by_split=bs, na_reason={k: r.get(k, "not evaluated by this check") for k, v in bs.items() if v is None})


def b_record(*args, by_split=None, na_reason=None, **kw):
    """record_check with every split keyed: splits missing from by_split get None plus a default na_reason."""
    if by_split is not None and na_reason is None:
        kw.update(b_bs(by_split))
    else:
        kw.update(by_split=by_split, na_reason=na_reason)
    record_check(*args, **kw)


def b_excess(df, fail_cols, alpha=REL_ALPHA, splits=None):
    """Noise-calibrated split verdict. df: one row per pool with boolean fail columns (each test has false-alarm
    probability alpha under 'no drift'). Per split: distinct pools, tests, observed failures, expected
    failures, binomial P(X >= observed); the split passes unless that probability < 0.05. Tests in one pool
    are positively correlated, so the binomial p is optimistic about excess (a conservative-to-fail proxy)."""
    rows = []
    for s in (splits or B_UNIT_SPLITS):
        d = df[df.pool.isin(B_UNITS.loc[B_UNITS.split == s, "pool"])]
        n = len(d) * len(fail_cols)
        obs = int(d[fail_cols].to_numpy().sum())
        p = float(stats.binom.sf(obs - 1, n, alpha)) if n else np.nan
        rows.append(dict(split=s, pools=len(d), tests=n, fails=obs, expected=alpha * n, p_excess=p, ok=bool(p >= 0.05)))
    return pd.DataFrame(rows).set_index("split")


def b_s4_pool(df, col):
    """S4 upper bound for vocabulary checks: value of the (108..c-1 -> c) pool, max over S4 folds (S4 also sees
    the early part of c, so its true unseen share can only be lower)."""
    if "S4" not in SPLIT_DEFS:
        return None
    lab = {c: _rng_label(range(CONGRESSES[0], c)) for c in SPLIT_DEFS["S4"]["folds"]}
    v = df[[lab.get(e) == t for t, e in zip(df.train, df["eval"])]][col]
    return float(v.max()) if len(v) else None


def b_num(v):
    """Compact number format for transposed tables: integers without decimals, shares with 3 decimals."""
    return "" if pd.isna(v) else f"{v:,.0f}" if float(v).is_integer() and abs(v) >= 2 else f"{v:.3f}"


def b_gini(x):
    """Gini coefficient of a non-negative vector (0 = equal, 1 = all mass on one node)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def b_top_share(x, q=0.10):
    """Share of total degree held by the top q fraction of nodes."""
    x = np.sort(np.asarray(x, float))[::-1]
    k = max(1, int(np.ceil(q * len(x))))
    return float(x[:k].sum() / x.sum()) if x.sum() > 0 else np.nan


mem = members[members.congress.isin(B_ALL)].dropna(subset=["chamber"])[["bioguide_id", "congress", "chamber", "party_code_asof"]]
cosp_c = cosp[cosp.congress.isin(B_ALL)]

# Dated edges per legislator and Congress (cosponsorships + sponsored bills of any type) for as-of counts.
_bact = pd.concat([cosp_c[["congress", "bioguide_id", "sponsorship_date"]].set_axis(["congress", "bioguide_id", "d"], axis=1),
                   bills.loc[bills.congress.isin(B_ALL), ["congress", "sponsor_bioguide_id", "introduced_date"]]
                   .set_axis(["congress", "bioguide_id", "d"], axis=1)]).dropna()
_bact["d"] = _bact["d"].astype("datetime64[ns]")
B_DAILY = _bact.groupby(["congress", "bioguide_id", "d"]).size().rename("n").reset_index()
B_DAILY["n_cum"] = B_DAILY.groupby(["congress", "bioguide_id"])["n"].cumsum()
B_DAILY = B_DAILY.sort_values("d", kind="stable")
B_DEG_FINAL = B_DAILY.groupby(["congress", "bioguide_id"])["n_cum"].max()
B_ACTIVE = _bact.groupby("congress")["bioguide_id"].agg(lambda s: set(s))   # legislators with >= 1 edge


def b_active_in(pool):
    """Legislators with at least one edge (sponsor or cosponsor) in any Congress of `pool`."""
    return set().union(*[B_ACTIVE[c] for c in pool if c in B_ACTIVE.index])


def b_count_asof(congress, ids, when):
    """Edges (sponsorships + cosponsorships) of legislator ids[i] in congress[i] dated <= when[i]."""
    left = pd.DataFrame({"congress": np.asarray(congress, int), "bioguide_id": pd.Series(ids).fillna("").to_numpy(),
                         "d": pd.to_datetime(pd.Series(when)).to_numpy().astype("datetime64[ns]"),
                         "_i": np.arange(len(ids))}).sort_values("d", kind="stable")
    m = pd.merge_asof(left, B_DAILY[["congress", "bioguide_id", "d", "n_cum"]], on="d",
                      by=["congress", "bioguide_id"], direction="backward")
    return m.sort_values("_i")["n_cum"].fillna(0).to_numpy()


# Task-1 positive edges per t0: cosponsorships with lag in (t0, t0+30] on uncensored law-capable bills.
B_WT = {t0: window_targets(t0) for t0 in (0, 7)}
B_POS = {}
for t0, wt in B_WT.items():
    pe = cosp_c[cosp_c.law_capable & (cosp_c.lag_days > t0) & (cosp_c.lag_days <= t0 + WINDOW_DAYS)
                & cosp_c.bill_id.isin(wt.loc[~wt.censored, "bill_id"])].copy()
    pe["n_hist"] = b_count_asof(pe.congress, pe.bioguide_id, pe.introduced_date + pd.Timedelta(days=t0))
    B_POS[t0] = pe
pos_edges = B_POS[0]

# S4 units (within-Congress): prior Congresses + events of c dated before the unit's lower cut date.
B_S4 = [dict(role=u["role"], c=u["within"][0], prior=list(u["train"]),
             lo=S4_CUT_DATES[u["within"][0]][u["within"][2]], hi=S4_CUT_DATES[u["within"][0]][u["within"][3]])
        for u in iter_eval_units("S4")] if "S4" in SPLIT_DEFS else []

print(f"members (complete Congresses): {len(mem):,} rows, {mem.bioguide_id.nunique():,} legislators; "
      f"active cosponsorships: {len(cosp_c):,}; Task-1 positive edges t0=0: {len(B_POS[0]):,}, t0=7: {len(B_POS[7]):,}")
print(f"{len(B_POOLS)} distinct evaluation pools ({len(B_UNITS)} units of {', '.join(B_UNIT_SPLITS)}); "
      "pool ids used in every table below:")
print(B_POOLS[["pool", "train_label", "c", "used_by"]].rename(columns={"train_label": "train", "c": "eval"})
      .to_string(index=False))

# %% [markdown]
# ### B1. Legislator node overlap with train (transductive view)
#
# **Why it matters.** Task 1 ranks legislators for a bill. A transductive model (a learned embedding
# per legislator, e.g. an embedding table or a GNN with ID features) can only score legislators it saw
# in training. A brand-new member falls back to an untrained vector. **This is a model and protocol
# requirement, not a split-ranking criterion**: every election replaces a sizeable share of members
# (first table), so every by-Congress split has a similar cold-start share (see the per-split line). Whether it hurts
# depends on whether the model may use history inside the evaluated Congress, which B1b measures.
# **How to read it.** The first table is per Congress against *all* earlier Congresses. The second
# table evaluates the candidate pools. "New to chamber" counts legislators who served before, but only
# in the other chamber.

# %%
# Cold-start rates per Congress (vs all earlier) and for the candidate pools; member retention heatmap.
first_seen = mem.groupby("bioguide_id")["congress"].min()
first_seen_ch = mem.groupby(["bioguide_id", "chamber"])["congress"].min()
rows = []
for c in B_ALL[1:]:
    m = mem[mem.congress == c].drop_duplicates(["bioguide_id", "chamber"])
    new = m.bioguide_id.map(first_seen) == c
    new_ch = np.array([first_seen_ch.get((b, ch), c) == c for b, ch in zip(m.bioguide_id, m.chamber)]) & ~new.values
    act = cosp_c[cosp_c.congress == c].bioguide_id.drop_duplicates()
    pe = pos_edges[pos_edges.congress == c]
    rows.append(dict(congress=c, members=m.bioguide_id.nunique(), new_members=float(new.mean()),
                     new_to_chamber=float(new_ch.mean()),
                     new_active_cosponsors=float((act.map(first_seen) == c).mean()),
                     new_pos_edges=float((pe.bioguide_id.map(first_seen) == c).mean())))
b1 = pd.DataFrame(rows)
print("One-step-ahead cold start (legislator never a member in any earlier Congress in the data):")
print(b1.set_index("congress").T.to_string(float_format=b_num))

mem_u = mem.drop_duplicates(["bioguide_id", "congress"])
act_u = cosp_c.drop_duplicates(["bioguide_id", "congress"])


def _b1(pool, c):
    seen = set(mem.loc[mem.congress.isin(pool), "bioguide_id"])
    return dict(members=float((~mem_u.loc[mem_u.congress == c, "bioguide_id"].isin(seen)).mean()),
                active_cosponsors=float((~act_u.loc[act_u.congress == c, "bioguide_id"].isin(seen)).mean()),
                pos_edges=float((~pos_edges.loc[pos_edges.congress == c, "bioguide_id"].isin(seen)).mean()))


b1p = b_eval_pools(_b1)
print("\nShare UNSEEN as a member in the train Congresses, by pool:")
b_show(b1p, ["members", "active_cosponsors", "pos_edges"])
b1_split = b_by_split(b1p, "pos_edges")
print("max share of positive edges from unseen legislators over all units: " + b_fmt_split(b1_split))

line_chart(b1.melt(id_vars="congress", value_vars=["new_members", "new_active_cosponsors", "new_pos_edges", "new_to_chamber"],
                   var_name="measure", value_name="share"),
           "congress", "share", series="measure", yfmt=".0%",
           title="Legislator cold start per Congress (never seen in any earlier Congress)",
           subtitle="Share of members / active cosponsors / Task-1 positive edges involving a new legislator; 108 omitted (left-censored)",
           ylabel="share new", name="B_cold_start_line")

# Retention heatmap: share of Congress j's members who were members in Congress i (i <= j).
sets = {c: set(mem.loc[mem.congress == c, "bioguide_id"]) for c in B_ALL}
heat = pd.DataFrame([dict(eval_congress=str(j), seen_in=str(i), share=len(sets[j] & sets[i]) / len(sets[j]) if i <= j else None)
                     for i in B_ALL for j in B_ALL])
heatmap(heat, "eval_congress", "seen_in", "share", fmt=".2f", domain=[0, 1], height=420,
        title="Member overlap: share of column Congress's members who also served in row Congress",
        subtitle="Overlap decays with distance: a train pool far in the past covers fewer test legislators",
        name="B_member_overlap_heatmap")

w1 = float(b1_split.max())
b_record("B", "B1", "Share of val/test Task-1 positive edges from legislators unseen in train (transductive-only model)",
             None, metric=round(w1, 4), threshold="info: model/protocol requirement",
             quantity="node_overlap", **b_bs({k: round(float(v), 4) for k, v in b1_split.items()},
                                           {v: "same train-cold share as the base split; the +stream warm-up is B1b"
                                            for v in B_STREAM}),
             detail=(f"Max over by-Congress units of positive edges whose cosponsor was not a member in the train Congresses: "
                     f"{b_fmt_split(b1_split)}. One-step-ahead new-edge share is {b1.new_pos_edges.min():.1%}-"
                     f"{b1.new_pos_edges.max():.1%} in every Congress (train-to-train included): elections, not the split, drive "
                     "it, so it does not rank the by-Congress splits. A purely transductive model with no within-Congress history "
                     "loses this slice. Requirement: inductive legislator features (party, state, chamber, committees) OR "
                     "message passing over eval-Congress edges dated <= t0 (B1b shows this warms up almost all of them)."))

# %% [markdown]
# ### B1b / B1c. Within-Congress warm-up before t0 (streaming view of cold start) and neighbourhood depth
#
# **Why it matters.** In the realistic streaming setup the model may use every edge of the evaluated
# Congress dated **on or before** the prediction time (bill introduction + t0): earlier cosponsorships
# and sponsorships by the same legislator. A new member who already cosponsored other bills before t0
# is no longer an empty node. But one or two edges are not a neighbourhood: message passing over a node
# with 1-2 neighbours gives a representation dominated by those few bills. B1b counts a positive as
# warm after >= 1 in-Congress edge; **B1c asks for >= 5** (the low end of typical GraphSAGE-style
# neighbour-sampling fan-outs) and also shows >= 10. The verdicts are per protocol: base splits forbid
# eval-Congress history (their cold share is B1's train-cold share), `+stream` variants allow it up to
# intro + t0, S4 allows events of the Congress dated before its cut date.
# **How to read it.** One row per (pool, t0). `cold_train` = positives whose cosponsor has no edge in
# the train Congresses; `cold_lt1/5/10` = of all positives, cold in train AND fewer than 1 / 5 / 10
# in-Congress edges by intro + t0 (`cold_lt1` is B1b's streaming-cold share); `new_ge1/5/10` = of the
# train-cold positives, share with >= 1 / 5 / 10 in-Congress edges by t0; `all_ge5` = of ALL positives,
# share whose cosponsor has >= 5 in-Congress edges by t0; `spon_cold` = evaluated bills whose sponsor
# has no train edge and no other in-Congress edge by t0.

# %%
# Streaming cold start and in-Congress degree at intro + t0 for every pool; S4 units use events before the cut date.
B_SPON = {}
for t0, wt in B_WT.items():
    eb = wt[~wt.censored & wt.congress.isin(B_ALL)][["bill_id", "congress", "introduced_date"]].merge(
        bills[["bill_id", "sponsor_bioguide_id"]], on="bill_id")
    eb["n_hist"] = b_count_asof(eb.congress, eb.sponsor_bioguide_id, eb.introduced_date + pd.Timedelta(days=t0)) - 1
    B_SPON[t0] = eb
B_DEG_K = (1, 5, 10)


def _b1b_stats(pe, h, ct, spon_cold):
    """pe: positives; h: in-Congress degree available to the model; ct: cold-in-train mask; spon_cold: bill mask."""
    out = dict(cold_train=float(ct.mean()))
    out.update({f"cold_lt{k}": float((ct & (h < k)).mean()) for k in B_DEG_K})
    out.update({f"new_ge{k}": float((h[ct] >= k).mean()) if ct.any() else np.nan for k in B_DEG_K})
    out.update(all_ge5=float((h >= 5).mean()), spon_cold=float(spon_cold.mean()))
    return out


def _b1b(pool, c):
    seen = b_active_in(pool)
    out = {}
    for t0 in (0, 7):
        pe, eb = B_POS[t0], B_SPON[t0]
        pe, eb = pe[pe.congress == c], eb[eb.congress == c]
        st = ~eb.sponsor_bioguide_id.isin(seen) & (eb.n_hist <= 0)
        out.update({f"{k}_t{t0}": v for k, v in _b1b_stats(pe, pe.n_hist, ~pe.bioguide_id.isin(seen), st).items()})
    return out


b1b = b_eval_pools(_b1b)
_b1b_cols = ["cold_train", *[f"cold_lt{k}" for k in B_DEG_K], *[f"new_ge{k}" for k in B_DEG_K], "all_ge5", "spon_cold"]
_long = pd.concat([b1b[["pool", "train", "eval"]].assign(t0=t0, **{k: b1b[f"{k}_t{t0}"] for k in _b1b_cols})
                   for t0 in (0, 7)]).sort_values(["eval", "pool", "t0"], kind="stable")
b_show(_long[_long.t0 == 0], _b1b_cols)
print("t0 = 7, same pools: cold_lt1 <= {:.2%}, cold_lt5 <= {:.2%}, train-cold with >= 5 edges >= {:.1%}".format(
    *(lambda d: (d.cold_lt1.max(), d.cold_lt5.max(), d.new_ge5.min()))(_long[(_long.t0 == 7) & _long["eval"].isin(B_EVAL)])))

# S4: history = prior Congresses + events of c dated before the unit's cut date (no per-bill streaming).
_s4 = []
for u in B_S4:
    seen = b_active_in(u["prior"])
    pe = B_POS[0][(B_POS[0].congress == u["c"]) & (B_POS[0].introduced_date >= u["lo"]) & (B_POS[0].introduced_date < u["hi"])]
    eb = B_SPON[0][(B_SPON[0].congress == u["c"]) & (B_SPON[0].introduced_date >= u["lo"]) & (B_SPON[0].introduced_date < u["hi"])]
    cut = u["lo"] - pd.Timedelta(days=1)
    h = pd.Series(b_count_asof(pe.congress, pe.bioguide_id, [cut] * len(pe)), index=pe.index)
    hs = b_count_asof(eb.congress, eb.sponsor_bioguide_id, [cut] * len(eb))
    _s4.append(dict(role=u["role"], c=u["c"], **_b1b_stats(pe, h, ~pe.bioguide_id.isin(seen),
                                                           ~eb.sponsor_bioguide_id.isin(seen) & (hs <= 0))))
b1b_s4 = pd.DataFrame(_s4)
s4_lt1, s4_lt5 = (float(b1b_s4.cold_lt1.max()), float(b1b_s4.cold_lt5.max())) if len(b1b_s4) else (np.nan, np.nan)

b1b_lt1 = b_by_split(b1b, "cold_lt1_t0")
b1b_lt5 = b_by_split(b1b, "cold_lt5_t0")
b1b_lt10 = b_by_split(b1b, "cold_lt10_t0")
b1b_warm = b_by_split(b1b, "new_ge1_t0", "min")
b1b_ge5 = b_by_split(b1b, "new_ge5_t0", "min")
b1b_sp = b_by_split(b1b, "spon_cold_t0")
b1b_ct = b_by_split(b1b, "cold_train_t0")
print("per split, max over units at t0=0 (base split | +stream protocol): cold_train " + b_fmt_split(b1b_ct)
      + " | cold_lt1 " + b_fmt_split(b1b_lt1, ".2%") + " | cold_lt5 " + b_fmt_split(b1b_lt5, ".2%"))
print(f"S4 units (history before the cut date): max cold_lt1 {s4_lt1:.2%}, cold_lt5 {s4_lt5:.2%}; "
      f"min share of train-cold positives with >= 5 in-Congress edges by t0=0: {b_fmt_split(b1b_ge5)}")

ch1b = b1b[b1b["eval"].isin(B_EVAL)].assign(pool=lambda d: d.pool + " " + d.train + "->" + d["eval"].astype(str))
ch1b = ch1b.melt(id_vars="pool", value_vars=["cold_train_t0", "cold_lt5_t0", "cold_lt1_t0"],
                 var_name="measure", value_name="share")
ch1b["measure"] = ch1b.measure.map({"cold_train_t0": "no train edge (base split)",
                                    "cold_lt5_t0": "no train edge, < 5 in-Congress edges by t0",
                                    "cold_lt1_t0": "no train edge, no in-Congress edge by t0"})
bar_chart(ch1b, "pool", "share", series="measure", yfmt=".0%", width=800,
          title="Share of Task-1 positives with a cold cosponsor, by how much history is allowed (t0 = 0)",
          subtitle="Pools evaluating 116-118; streaming history up to intro + t0 removes most cold start, but a 5-edge "
                   "neighbourhood is a stricter bar", ylabel="share of positives", name="B_streaming_cold_start")

B_NA_BASE_STREAM = {b: f"base split = protocol without eval-Congress history; its cold share is B1 (INFO); the "
                       f"protocol question is judged on {v}" for v, b in B_STREAM.items()}
B_NA_BASE_STREAM.update({s: "protocol without eval-Congress history; cold share is B1 (INFO)" for s in B_UNIT_SPLITS
                         if s not in B_NA_BASE_STREAM})
w1b = float(b1b_lt1.max())
w1b7 = float(b_by_split(b1b, "cold_lt1_t7").max())
_ok = lambda v, thr=0.02: None if pd.isna(v) else bool(v < thr)
b_record("B", "B1b", "With eval-Congress history up to intro + t0 (+stream), few positives involve a legislator with no edge at all",
             bool(w1b < 0.02), metric=round(w1b, 4), threshold="< 0.02 of positives (t0=0, every unit)",
             quantity="cold_start",
             **b_bs({**{v: _ok(b1b_lt1.get(b)) for v, b in B_STREAM.items() if b in b1b_lt1.index},
                     **({"S4": _ok(s4_lt1)} if len(b1b_s4) else {})}, B_NA_BASE_STREAM),
             detail=(f"+stream protocol: max share of positives whose cosponsor has no train edge and no in-Congress edge by intro + t0: "
                     f"t0=0 {b_fmt_split(b1b_lt1, '.2%')}; t0=7 max {w1b7:.2%}; S4 (events before the cut) {s4_lt1:.2%}. Base "
                     f"splits (no eval-Congress history): {b_fmt_split(b1b_ct)}; these are B1's train-cold shares and are left None in "
                     "by_split: '+stream' is a model-input protocol open to every by-Congress split, so a base split without it is "
                     "a transductive-model choice, not a failure of the split itself. Of the train-cold "
                     f"positives, >= {b1b_warm.min():.1%} have >= 1 in-Congress edge by t0=0. Bills whose sponsor is cold by t0: "
                     f"max {b1b_sp.max():.2%}. 2% threshold: at most a 2% Recall loss if cold nodes were scored at random. "
                     "'Warm' here means one edge; B1c applies a 5-edge bar."))
b_record("B", "B1c", "Positives from train-cold legislators have a usable in-Congress neighbourhood (>= 5 edges) by intro + t0",
             bool(b1b_lt5.max() < 0.02), metric=round(float(b1b_lt5.max()), 4),
             threshold="cold in train AND < 5 in-Congress edges at intro + t0: < 0.02 of positives (t0=0, every unit)",
             **b_bs({**{v: _ok(b1b_lt5.get(b)) for v, b in B_STREAM.items() if b in b1b_lt5.index},
                     **({"S4": _ok(s4_lt5)} if len(b1b_s4) else {})}, B_NA_BASE_STREAM),
             detail=(f"+stream protocol, t0=0: positives cold in train with < 5 in-Congress edges: {b_fmt_split(b1b_lt5, '.2%')} "
                     f"(< 10 edges: {b_fmt_split(b1b_lt10, '.2%')}); S4 {s4_lt5:.2%}. Among train-cold positives only, the share with "
                     f">= 5 in-Congress edges by t0 is >= {b1b_ge5.min():.1%} (>= 1 edge: {b1b_warm.min():.1%}). All positives with "
                     f">= 5 in-Congress edges by t0: >= {b1b.loc[b1b['eval'].isin(B_EVAL), 'all_ge5_t0'].min():.1%}. Degree = "
                     "sponsorships + cosponsorships in the eval Congress dated <= intro + t0. 5 edges = low end of usual "
                     "neighbour-sampling fan-outs; below it the node's representation rests on its input features. Base "
                     "splits forbid in-Congress history (their cold share is B1, INFO), so by_split covers +stream variants and S4 only. If this fails for +stream, keep inductive legislator "
                     "features (party, state, committees, prior-Congress embedding) next to message passing."))

# %% [markdown]
# ### B2. Sponsor overlap
#
# **Why it matters.** The sponsor is the strongest single signal for who cosponsors (sponsor-cosponsor
# ties repeat across bills). Bills whose sponsor is new to the train period have no sponsor history,
# so the model sees a cold-start *bill neighbourhood*. New sponsors are mostly freshmen, who
# sponsor fewer bills, so the bill-level share is usually smaller than the member-level share. Unlike
# B1, this depends on the train window: a short sliding window (S2) forgets sponsors who were absent
# in its few Congresses. **How to read it.** Share of law-capable bills (and of Task-1 positives) in the
# evaluated Congress whose sponsor was not a member during the train Congresses, and the stricter
# version, "never sponsored a law-capable bill in the train Congresses" (`fail` = above the noise bound).
# The bound is not a fixed band: it is the 95% upper t prediction interval of the **one-step-ahead**
# share (train 108..c-1, evaluate c) over the reference Congresses 109-115, i.e. how many new sponsors an
# ordinary election brings. As elsewhere, only units evaluating 116-118 decide a split. The streaming
# version is in B1b.

# %%
# Share of evaluated-Congress bills whose sponsor is new to the train Congresses (as member / as sponsor).
lb = bills[bills.law_capable & bills.congress.isin(B_ALL)][["bill_id", "congress", "sponsor_bioguide_id"]]


def _b2(pool, c):
    seen_m = set(mem.loc[mem.congress.isin(pool), "bioguide_id"])
    seen_s = set(lb.loc[lb.congress.isin(pool), "sponsor_bioguide_id"])
    d = lb[lb.congress == c]
    pe = pos_edges[pos_edges.congress == c]
    return dict(bills=len(d), sponsor_not_member=float((~d.sponsor_bioguide_id.isin(seen_m)).mean()),
                sponsor_never_sponsored=float((~d.sponsor_bioguide_id.isin(seen_s)).mean()),
                pos_edges_new_sponsor=float((~pe.sponsor_bioguide_id.isin(seen_m)).mean()))


b2 = b_eval_pools(_b2)
# Noise calibration: one-step-ahead (expanding 108..c-1 -> c) new-sponsor share, judged against the fixed 108-115 reference.
b2_exp = pd.Series({c: _b2(list(range(B_ALL[0], c)), c)["sponsor_never_sponsored"] for c in B_ALL[1:]})
b2_tab, b2_info = rel_to_train(b2_exp, side="upper")
b2_hi = float(b2_tab.loc[b2_tab.role == "cand", "pi_hi"].iloc[0])
b2["fail"] = b2.sponsor_never_sponsored > b2_hi
b_show(b2, ["bills", "sponsor_not_member", "sponsor_never_sponsored", "pos_edges_new_sponsor", "fail"])
print("one-step-ahead new-sponsor share (expanding pools): " + ", ".join(f"{c}={v:.3f}" for c, v in b2_exp.items())
      + f"; 95% upper t-PI of ref 109-115 = {b2_hi:.3f} (nested-LOO false alarms {b2_info['fa_count']}/{b2_info['n']})")
# Per split: 'cands' rule (only units evaluating 116-118 are judged, like rel_by_split's default).
_u2 = B_UNITS.merge(b2[["pool", "eval", "sponsor_never_sponsored", "fail"]], on="pool")
_u2c = _u2[_u2["eval"].isin(B_EVAL)]
b2_split = _u2c.groupby("split", sort=False)["sponsor_never_sponsored"].max().reindex(B_UNIT_SPLITS)
b2_ok = (~_u2c.groupby("split", sort=False)["fail"].any()).reindex(B_UNIT_SPLITS)
print("max share over units evaluating 116-118 (bills whose sponsor never sponsored in train): " + b_fmt_split(b2_split))
w2 = float(b2_split.max())
_b2s4 = float(b1b_s4.spon_cold.max()) if len(b1b_s4) else np.nan
b_record("B", "B2", "Few val/test bills have a sponsor with no sponsorship history in train, relative to train-to-train noise",
             bool(b2_ok.dropna().all()), metric=round(w2, 4), threshold=f"<= {b2_hi:.3f} ({b2_info['rule']}; one-step-ahead share)",
             quantity="sponsor_overlap",
             by_split=b_split_dict({k: None if pd.isna(v) else bool(v) for k, v in b2_ok.items()},
                                   stream={k: bool(v <= b2_hi) for k, v in b1b_sp.items()},
                                   extra={"S4": bool(_b2s4 <= b2_hi)} if len(b1b_s4) else None),
             detail=(f"Max share of law-capable bills whose sponsor never sponsored in the train Congresses (units evaluating 116-118): "
                     f"{b_fmt_split(b2_split)}. Noise bound: one-step-ahead (expanding) share over 109-115 = "
                     f"{b2_exp.loc[109:115].min():.3f}-{b2_exp.loc[109:115].max():.3f}, 95% upper t-PI {b2_hi:.3f} (nested-LOO false "
                     f"alarms {b2_info['fa_count']}/{b2_info['n']}); candidates one-step-ahead: "
                     + ", ".join(f"{c} {b2_exp[c]:.3f}" for c in B_EVAL) + ". Failing units: "
                     + ("; ".join(f"{r.split} {r.role} {r.train_label}->{r['eval']} {r.sponsor_never_sponsored:.3f}" for _, r in _u2c[_u2c.fail].iterrows()) or "none")
                     + ". S1pool fails only through its protocol: its 118 test is scored by a model refit on 108-116 (two elections "
                     "back); refitting on 108-117 (as S1b / S3 do) or streaming eval-Congress history removes it. With within-Congress "
                     f"history up to t0, bills whose sponsor is cold drop to <= {b1b_sp.max():.2%} (B1b; +stream verdicts use this), "
                     f"S4 (events before the cut date) {_b2s4:.2%}. Share of positive edges on bills with a non-member sponsor <= "
                     f"{b2.pos_edges_new_sponsor.max():.1%}: report metrics separately for new-sponsor bills."))

# %% [markdown]
# ### B3. Degree distribution stationarity
#
# **Why it matters.** GNN layers aggregate over neighbours. If bills get many more cosponsors, or a
# few legislators dominate cosponsoring in later Congresses, the scale of aggregated messages and
# the base rate of positives per bill both move. A model trained on old degree regimes is then
# miscalibrated at test time. **How to read it.** First table: per Congress, per-bill cosponsor counts
# (law-capable bills) and per-legislator cosponsorship counts, KS / PSI against the previous Congress
# (KS p-values are tiny at this sample size, so read the *statistic*), Gini and top-10% share (tail
# concentration). Second table: PSI of each pool's evaluated Congress against that pool's train
# Congresses. The noise reference is **fixed** for every pool: the one-vs-rest PSI of each Congress in
# 108-115 against the other seven, with a 95% upper t prediction interval on the log scale
# (`rel_to_train`). `*` marks a PSI above that bound. The split table compares observed failures with
# the number expected by chance (5% per test). Degrees count all active cosponsorships (final
# snapshot), a proxy for what is seen at prediction time.

# %%
# Per-Congress degree distributions; each pool's eval-vs-train PSI judged against ONE fixed reference (108-115 one-vs-rest PSI).
bill_deg = (bills[bills.law_capable & bills.congress.isin(B_ALL)][["bill_id", "congress"]]
            .assign(deg=lambda d: d.bill_id.map(cosp_c.groupby("bill_id").size()).fillna(0)))
leg_deg = (mem_u.merge(cosp_c.groupby(["congress", "bioguide_id"]).size().rename("deg").reset_index(),
                       on=["congress", "bioguide_id"], how="left").fillna({"deg": 0}))
leg_deg["deg_rel"] = leg_deg.deg / leg_deg.groupby("congress")["deg"].transform("mean")
rows = []
for c in B_ALL:
    bd, ld = bill_deg.loc[bill_deg.congress == c, "deg"], leg_deg.loc[leg_deg.congress == c, "deg"]
    r = dict(congress=c, bill_mean=bd.mean(), bill_zero=float((bd == 0).mean()), bill_gini=b_gini(bd),
             leg_mean=ld.mean(), leg_gini=b_gini(ld), leg_top10=b_top_share(ld))
    if c > B_ALL[0]:
        bp, lp = bill_deg.loc[bill_deg.congress == c - 1, "deg"], leg_deg.loc[leg_deg.congress == c - 1, "deg"]
        r.update(bill_ks=ks(bp, bd)[0], bill_psi=psi(bp, bd), leg_ks=ks(lp, ld)[0], leg_psi=psi(lp, ld))
    rows.append(r)
b3 = pd.DataFrame(rows)
print(b3.round(3).to_string(index=False))

_B3_SER = {"bill": (bill_deg, "deg"), "leg": (leg_deg, "deg"), "leg_rel": (leg_deg, "deg_rel")}
_psi_vs = lambda df, col, pool, c: psi(df.loc[df.congress.isin(pool), col], df.loc[df.congress == c, col])
B3_REF = {}
for k, (df, col) in _B3_SER.items():
    vals = {t: _psi_vs(df, col, [p for p in REL_REF if p != t], t) for t in REL_REF}
    vals.update({c: _psi_vs(df, col, REL_REF, c) for c in B_ALL if c not in REL_REF})
    tab, info = rel_to_train(pd.Series(vals), side="upper", log=True)
    B3_REF[k] = dict(tab=tab, info=info, hi=float(tab.loc[tab.role == "cand", "pi_hi"].iloc[0]))
print("\nFixed noise reference (one-vs-rest PSI inside 108-115; 95% upper t-PI, log scale): "
      + "; ".join(f"{k} {v['tab'].loc[v['tab'].role == 'ref', 'value'].min():.3f}-{v['tab'].loc[v['tab'].role == 'ref', 'value'].max():.3f}"
                  f" -> bound {v['hi']:.3f} (nested-LOO false alarms {v['info']['fa_count']}/{v['info']['n']})"
                  for k, v in B3_REF.items()))


def _b3(pool, c):
    out = {}
    for k, (df, col) in _B3_SER.items():
        out[f"{k}_psi"] = _psi_vs(df, col, pool, c)
        out[f"{k}_fail"] = bool(out[f"{k}_psi"] > B3_REF[k]["hi"])
    return out


b3p = b_eval_pools(_b3)
_b3show = b3p.copy()
for k in _B3_SER:
    _b3show[k] = _b3show[f"{k}_psi"].map("{:.3f}".format) + np.where(_b3show[f"{k}_fail"], "*", " ")
print("\nPSI of the evaluated Congress vs its pool's train Congresses (* = above the fixed noise bound):")
b_show(_b3show, list(_B3_SER))
b3x_raw = b_excess(b3p, ["bill_fail", "leg_fail"])
b3x_rel = b_excess(b3p, ["bill_fail", "leg_rel_fail"])
print("\nPer split, all pools: failures vs chance (raw = bill + raw legislator degree; scale-free = bill + legislator "
      "degree / Congress mean):")
print(pd.concat([b3x_raw[["pools", "tests", "fails", "expected", "p_excess"]],
                 b3x_rel[["fails", "p_excess"]].add_suffix("_scalefree")], axis=1).round(3).to_string())

line_chart(b3.melt(id_vars="congress", value_vars=["bill_gini", "leg_gini", "leg_top10", "bill_zero"],
                   var_name="measure", value_name="value"),
           "congress", "value", series="measure", yfmt=".2f",
           title="Degree concentration per Congress",
           subtitle="Gini of bill / legislator degree, top-10% legislators' share, share of bills with no cosponsor",
           ylabel="value", name="B_degree_concentration")
_bad = b3p[b3p[["bill_fail", "leg_fail"]].any(axis=1)]
b_record("B", "B3", "Val/test bill and legislator degree distributions are no further from the train pool than train Congresses are from each other",
             bool(b3x_raw.ok.all()), metric=int(b3x_raw.fails.max()),
             threshold="per split: failures (PSI > fixed 108-115 95% upper t-PI) not in excess of chance (binomial p >= 0.05)",
             quantity="degree_dist", by_split=b_split_dict(b3x_raw.ok.to_dict()),
             detail=(f"Fixed reference (pool-size independent): bounds bill {B3_REF['bill']['hi']:.3f}, legislator {B3_REF['leg']['hi']:.3f}, "
                     f"scale-free {B3_REF['leg_rel']['hi']:.3f}. Failures / tests (p excess) raw: "
                     + ", ".join(f"{s} {r.fails}/{r.tests} (p={r.p_excess:.2f})" for s, r in b3x_raw.iterrows())
                     + "; scale-free: " + ", ".join(f"{s} {r.fails}/{r.tests}" for s, r in b3x_rel.iterrows())
                     + ". Pools above the bound: " + ("; ".join(f"{r.pool} {r.train}->{r['eval']}" for _, r in _bad.iterrows()) or "none")
                     + ". Raw legislator degree jumps when bill volume jumps (116); if only the raw version fails, normalise "
                     "degree features per Congress / chamber. Final-snapshot degrees (proxy); binomial treats tests as "
                     "independent. +stream variants copy the base verdict (same graph)."))

# %% [markdown]
# ### B4. Committee schema stability
#
# **Why it matters.** Bill-committee referral edges are a natural relation for both tasks (the committee
# is the gatekeeper for advancement). Committees are renamed, merged, and subcommittees are reshuffled
# across Congresses. A committee node that exists only in test has no learned embedding, and a
# renamed committee splits one concept into two nodes. **How to read it.** Per Congress: share of
# committee codes and of *referral* edges (activity "Referred to") whose code never appeared in an
# earlier Congress, split into full committees vs subcommittees. Codes in this data appear to be
# normalised to current codes (e.g. 108 rows use today's subcommittee names). That helps stability
# but means historical renames are hidden, not absent.

# %%
# Committee code novelty per Congress (vs all earlier) and per pool; full committees vs subcommittees.
bc = load("bill_committees", ("bill_id", "committee_code", "committee_type", "activity", "activity_ts", "congress"))
bc = bc[bc.congress.isin(B_ALL)].copy()
bc["is_sub"] = bc.committee_type.eq("Subcommittee")
bc["referral"] = bc.activity.str.lower().str.startswith("referred")
ref = bc[bc.referral]
first_code = ref.groupby("committee_code")["congress"].min()
rows = []
for c in B_ALL[1:]:
    d = ref[ref.congress == c]
    new = d.committee_code.map(first_code) == c
    codes = d.drop_duplicates("committee_code")
    rows.append(dict(congress=c, codes=len(codes), new_codes=float((codes.committee_code.map(first_code) == c).mean()),
                     new_edges_full=float(new[~d.is_sub].mean()), new_edges_sub=float(new[d.is_sub].mean()) if d.is_sub.any() else np.nan,
                     sub_edge_share=float(d.is_sub.mean())))
b4 = pd.DataFrame(rows)
print("Referral edges whose committee code is new vs all earlier Congresses:")
print(b4.set_index("congress").T.to_string(float_format=b_num))


def _b4(pool, c):
    seen = set(ref.loc[ref.congress.isin(pool), "committee_code"])
    d = ref[ref.congress == c]
    return dict(unseen_edges=float((~d.committee_code.isin(seen)).mean()),
                unseen_edges_full=float((~d.loc[~d.is_sub, "committee_code"].isin(seen)).mean()),
                jaccard=len(seen & set(d.committee_code)) / len(seen | set(d.committee_code)))


b4p = b_eval_pools(_b4)
print("By pool (116-118): " + "; ".join(f"{r.pool} unseen {r.unseen_edges:.2%} (full {r.unseen_edges_full:.2%}), Jaccard {r.jaccard:.2f}"
                                      for r in b4p[b4p["eval"].isin(B_EVAL)].itertuples()))
b4_split = b_by_split(b4p, "unseen_edges")
w4 = float(b4_split.max())
b_record("B", "B4", "Val/test committee-referral edges point to committees seen in train (every by-Congress unit)",
             bool(w4 < 0.05), metric=round(w4, 4), threshold="< 0.05",
             by_split=b_split_dict({k: bool(v < 0.05) for k, v in b4_split.items()},
                                   extra={"S4": bool(b_s4_pool(b4p, "unseen_edges") < 0.05)}),
             detail=(f"Max share of referral edges with a committee code unseen in the train Congresses: {b_fmt_split(b4_split, '.2%')} "
                     f"(full committees only: <= {b4p.unseen_edges_full.max():.2%}). Proxy caveat: codes look normalised "
                     "to current codes, so true renames are partly hidden. 5% threshold: unseen relation targets beyond "
                     "this need a fallback (parent-committee node or chamber-level feature)."))

# %% [markdown]
# ### B5. Subject and policy-area vocabulary stability and coverage
#
# **Why it matters.** Legislative subjects (CRS index terms) and policy areas are candidate bill
# features or bill-subject edges. Two failure modes break a temporal split: (a) the **vocabulary
# changes** (a thesaurus revision makes old and new subject nodes disjoint), and (b) **coverage lag**:
# CRS assigns subjects after introduction, sometimes much later, so recent Congresses have fewer
# tagged bills. Coverage lag means test bills look different from train bills purely because of data
# latency. It is also a leakage risk (tags are not timestamped, so they may encode later events).
# **How to read it.** Per Congress: vocabulary size, share of subject terms / bill-subject links not
# seen in any earlier Congress, Jaccard vs previous Congress, and share of law-capable bills with at
# least one subject and with a policy area (final snapshot). The last line estimates policy-area
# availability **at t0** from the youngest bills of the in-progress Congress 119. Because subjects are
# essentially absent at t0, B5b is recorded as a feature-exclusion **requirement** (every split), not as
# a split condition.

# %%
# Subject vocabulary novelty, coverage per Congress, unseen-subject share per pool, policy area at t0 (119 proxy).
bs = load("bill_subjects", ("bill_id", "subject", "congress"))
bs = bs[bs.congress.isin(B_ALL + [IN_PROGRESS])]
lb_all = bills[bills.law_capable & bills.congress.isin(B_ALL + [IN_PROGRESS])][["bill_id", "congress", "policy_area", "introduced_date"]]
first_subj = bs.groupby("subject")["congress"].min()
vocab = {c: set(bs.loc[bs.congress == c, "subject"]) for c in B_ALL + [IN_PROGRESS]}
rows = []
for c in B_ALL + [IN_PROGRESS]:
    d = bs[bs.congress == c]
    lbc = lb_all[lb_all.congress == c]
    r = dict(congress=c, vocab=len(vocab[c]), subj_coverage=float(lbc.bill_id.isin(d.bill_id).mean()),
             policy_coverage=float(lbc.policy_area.notna().mean()), subj_per_tagged_bill=len(d) / max(1, d.bill_id.nunique()))
    if c > B_ALL[0]:
        r.update(new_terms=float((pd.Series(list(vocab[c])).map(first_subj) == c).mean()),
                 new_links=float((d.subject.map(first_subj) == c).mean()),
                 jaccard_prev=len(vocab[c] & vocab[c - 1]) / len(vocab[c] | vocab[c - 1]))
    rows.append(r)
b5 = pd.DataFrame(rows)
print("Subject vocabulary and coverage per Congress (119 in progress, context only):")
print(b5.set_index("congress").T.to_string(float_format=b_num))


def _b5(pool, c):
    seen = set(bs.loc[bs.congress.isin(pool), "subject"])
    d = bs[bs.congress == c]
    pa_seen = set(lb_all.loc[lb_all.congress.isin(pool), "policy_area"].dropna())
    pa = lb_all.loc[lb_all.congress == c, "policy_area"].dropna()
    return dict(unseen_subject_links=float((~d.subject.isin(seen)).mean()),
                jaccard_vocab=len(seen & vocab[c]) / len(seen | vocab[c]),
                unseen_policy_area_bills=float((~pa.isin(pa_seen)).mean()))


b5p = b_eval_pools(_b5)
print("By pool (116-118): " + "; ".join(f"{r.pool} unseen links {r.unseen_subject_links:.2%}, Jaccard {r.jaccard_vocab:.2f}, "
                                      f"unseen PA bills {r.unseen_policy_area_bills:.2%}" for r in b5p[b5p["eval"].isin(B_EVAL)].itertuples()))
# Annotation at t0: policy area from the shared PA_MISSING_AT_T0 (119 bills aged 0-7 d at the snapshot); subjects locally.
_l119 = lb_all[lb_all.congress == IN_PROGRESS]
_young = _l119[_l119.introduced_date > _l119.introduced_date.max() - pd.Timedelta(days=WINDOW_DAYS)]
pa_t0 = float(1 - PA_MISS_T0)
subj_t0 = float(_young.bill_id.isin(bs.bill_id).mean())
print(f"\nAt-t0 proxy (in-progress 119): policy area present for {pa_t0:.1%} of bills aged 0-7 d (n={PA_MISS_T0_N}, shared "
      f"PA_MISSING_AT_T0); >=1 subject for {subj_t0:.1%} of bills aged 0-{WINDOW_DAYS} d (n={len(_young):,})")

line_chart(b5.melt(id_vars="congress", value_vars=["subj_coverage", "policy_coverage", "jaccard_prev"],
                   var_name="measure", value_name="share").dropna(),
           "congress", "share", series="measure", yfmt=".0%",
           title="Subject coverage and vocabulary continuity per Congress",
           subtitle="Share of law-capable bills with >=1 subject / a policy area; Jaccard of subject vocab vs previous Congress",
           ylabel="share", name="B_subject_coverage")

regime = b5.loc[b5.congress.isin(B_ALL[1:])].sort_values("jaccard_prev").iloc[0]
cov_train = b5.loc[b5.congress.isin(range(108, 116)), "subj_coverage"].min()
cov_test = b5.loc[b5.congress.isin(B_EVAL)].set_index("congress")["subj_coverage"]
b5_split = b_by_split(b5p, "unseen_subject_links")
w5 = float(b5_split.max())
b_record("B", "B5a", "Val/test bill-subject links use subject terms seen in train (every by-Congress unit)",
             bool(w5 < 0.05), metric=round(w5, 4), threshold="< 0.05",
             by_split=b_split_dict({k: bool(v < 0.05) for k, v in b5_split.items()},
                                   extra={"S4": bool(b_s4_pool(b5p, "unseen_subject_links") < 0.05)}),
             detail=(f"Max unseen-subject link share: {b_fmt_split(b5_split, '.2%')}. But the vocabulary has a regime break at "
                     f"Congress {int(regime.congress)} (Jaccard vs previous = {regime.jaccard_prev:.2f}, vocab "
                     f"{int(b5.set_index('congress').vocab[int(regime.congress) - 1])} -> {int(regime.vocab)} terms): pre-"
                     f"{int(regime.congress)} subject nodes are a different thesaurus. Use subjects only from "
                     f"{int(regime.congress)} on, or fall back to policy areas (unseen policy-area bills <= "
                     f"{b5p.unseen_policy_area_bills.max():.2%})."))
b5_tab, b5_info = rel_to_train(b5.set_index("congress").loc[B_ALL, "subj_coverage"], side="lower")
b5b_fail = b5_tab.loc[b5_tab.role == "cand", "fail"]
b_record("B", "B5b", "Requirement: do not use legislative subjects as node / edge features at t0 (not available at prediction time; coverage lags in 117-118)",
             None, metric=round(subj_t0, 4), threshold="info: feature-exclusion requirement (all splits)",
             quantity="subject_coverage", requirement=True,
             detail=(f"At t0 only {subj_t0:.1%} of bills aged 0-{WINDOW_DAYS} d (in-progress 119, n={len(_young):,}; proxy) have any subject "
                     f"(metric), so subject edges cannot exist in an as-of-t0 graph under ANY split; a split condition on them "
                     f"does not discriminate. Final-snapshot coverage also lags: train 108-115 min {cov_train:.1%}; "
                     + ", ".join(f"{c}: {v:.1%}" for c, v in cov_test.items())
                     + f" (lower 95% t-PI {b5_tab.loc[b5_tab.role == 'cand', 'pi_lo'].iloc[0]:.1%}; outside: "
                     + (", ".join(str(c) for c in b5b_fail.index[b5b_fail == True]) or "none") + ")"   # noqa: E712
                     + ", which is CRS indexing lag (missing-not-at-random), so even a 'final snapshot' subject feature would "
                     "differ between train and 117/118. If subjects are ever used (e.g. for an offline analysis), use them from "
                     f"{int(regime.congress)} on with a has_subjects indicator. Policy area: final-snapshot coverage >= "
                     f"{b5.loc[b5.congress.isin(B_ALL), 'policy_coverage'].min():.1%}, but at t0 only about {pa_t0:.0%} of bills have one "
                     f"(shared PA_MISSING_AT_T0, n={PA_MISS_T0_N}, small-sample proxy): mask it when absent, or use title text at t0. "
                     "Section C owns the coverage quantity (subject_coverage)."))

# %% [markdown]
# ### B6. Cross-split edges: related-bill links and within-Congress future links
#
# **Why it matters.** `bill_related` connects bills. In a temporal split, an edge from a train bill to
# a bill in a *later* Congress (or a later-introduced bill in the same Congress) lets the model see
# the future when building train representations. Edges that cross the val/test boundary leak
# test structure into training. Links to *earlier* Congresses are legitimate context. **How to read
# it.** For each source Congress: number of links to earlier / same / later Congresses. Second
# table: among same-Congress links, the share pointing to a bill introduced *after* the source bill
# (these must be dropped or time-filtered when building a snapshot at t0). Caveat: the source stores
# almost only same-Congress links, so the cross-Congress count says little about cross-Congress leakage.

# %%
# Direction of bill_related links across Congresses, and within-Congress links to later-introduced bills.
rel = load("bill_related", ("bill_id", "related_bill_id", "relationship", "congress"))
rel = rel[rel.congress.isin(B_ALL + [IN_PROGRESS])].copy()
rel["rel_congress"] = pd.to_numeric(rel.related_bill_id.str.split("-").str[0], errors="coerce")
rel["direction"] = np.select([rel.rel_congress < rel.congress, rel.rel_congress > rel.congress], ["earlier", "later"], "same")
b6 = rel.groupby(["congress", "direction"]).size().unstack(fill_value=0).reindex(columns=["earlier", "same", "later"], fill_value=0)
b6["later_share"] = b6["later"] / b6.sum(axis=1)

intro = bills.set_index("bill_id")["introduced_date"]
same = rel[rel.direction == "same"].copy()
same["src_date"], same["dst_date"] = same.bill_id.map(intro), same.related_bill_id.map(intro)
same = same.dropna(subset=["src_date", "dst_date"])
same["gap"] = (same.dst_date - same.src_date).dt.days
fut = same.groupby("congress").agg(to_later_bill=("gap", lambda g: float((g > 0).mean())),
                                   later_beyond_7d=("gap", lambda g: float((g > 7).mean())),
                                   later_beyond_30d=("gap", lambda g: float((g > 30).mean())))
print("bill_related links by direction of the related bill's Congress, and same-Congress links whose target bill "
      "was introduced after the source (119 = in progress):")
print(b6.join(fut).T.to_string(float_format=b_num))

b6_all = b6.loc[b6.index.isin(B_ALL)]
n_cross = int(b6_all[["earlier", "later"]].sum().sum())
later_eval = b6_all["later"]
share_later = float(later_eval.sum() / b6_all[["earlier", "same", "later"]].sum().sum())
b_record("B", "B6a", "Related-bill links from complete Congresses to later Congresses (cross-split leak)",
             None, metric=round(share_later, 5), threshold="info: uninformative (source stores same-Congress links)",
             detail=(f"{int(later_eval.sum())} links from 108-118 point to a later Congress and only {n_cross} cross Congresses at all "
                     f"(all in {', '.join(str(c) for c in b6_all.index[b6_all[['earlier', 'later']].sum(axis=1) > 0]) or 'none'}), "
                     "against tens of thousands of same-Congress links. bill_related evidently stores almost only "
                     "same-Congress links (a source limitation), so this is NOT evidence that cross-Congress leakage is "
                     "absent. If cross-Congress links are added from another source, drop those whose target is later "
                     "than the source's split."))
w6 = float(fut.loc[fut.index.isin(B_ALL), "to_later_bill"].mean())
b_record("B", "B6b", "Within-Congress related links are time-safe without filtering",
             bool(w6 < 0.05), metric=round(w6, 4), threshold="< 0.05 (requirement: filter if it fails)",
             quantity="leakage_related_links", requirement=True,
             detail=(f"On average {w6:.1%} of same-Congress related links point to a bill introduced later than the "
                     "source (no edge timestamps exist, so target introduction date is the earliest the edge could "
                     "exist). These edges must be filtered at snapshot time t0 (keep only targets introduced <= source "
                     "intro + t0) for every split. A graph-construction requirement, not a split differentiator."))

# %% [markdown]
# ### B7. Committee edges that encode the advancement label
#
# **Why it matters.** `bill_committees` mixes the referral edge (known on day 0-7) with later activity
# edges such as "Reported by", "Markup by", "Discharged from", "Hearings by". "Reported by" *is* the
# Task 2 label, and markup / hearings are strong post-introduction signals. If the graph builder keeps
# all rows as bill-committee edges, every split (temporal or random) leaks the label. **How to read
# it.** Activity mix per Congress, the share of referral edges that occur more than 7 days after
# introduction (late referrals are not in a t0=7 snapshot), and how strongly having a non-referral edge
# predicts the advancement label.

# %%
# Activity mix in bill_committees, late referrals, and label leakage of non-referral committee edges.
bc["act"] = np.select([bc.referral, bc.activity.str.lower().str.startswith("reported"),
                       bc.activity.str.lower().str.startswith(("markup", "hearings", "discharged"))],
                      ["referral", "reported", "markup/hearing/discharge"], "other")
mix = bc.groupby(["congress", "act"]).size().unstack(fill_value=0)
mix = mix.div(mix.sum(axis=1), axis=0)
bcd = bc.merge(bills[["bill_id", "introduced_date"]], on="bill_id")
bcd["days"] = (bcd.activity_ts.dt.normalize() - bcd.introduced_date).dt.days
late = bcd[bcd.referral].groupby("congress")["days"].apply(lambda d: float((d > 7).mean())).rename("referral_after_day7")
mix = mix.join(late)
print("Committee-edge activity mix per Congress (row shares) and share of referral edges after day 7:")
print(mix.T.to_string(float_format=b_num))

nonref = set(bc.loc[~bc.referral, "bill_id"])
a = adv[~adv.censored & adv.congress.isin(B_ALL)].copy()
a["has_nonref_edge"] = a.bill_id.isin(nonref)
leak = a.groupby("has_nonref_edge")["label"].agg(["size", "mean"])
print("Advancement label rate by whether the bill has any non-referral committee edge: " + "; ".join(
    f"{k}: n={int(r['size']):,}, rate {r['mean']:.3f}" for k, r in leak.iterrows()))
recall = float(a.loc[a.label == 1, "has_nonref_edge"].mean())
b_record("B", "B7", "Graph builder must keep only referral committee edges known by t0 (label leakage)",
             None, metric=round(recall, 4), threshold="info: must filter",
             quantity="leakage_committee_edges", requirement=True,
             detail=(f"{recall:.1%} of advancement positives have a non-referral committee edge (metric). Bills with such an "
                     f"edge advance at {leak.loc[True, 'mean']:.1%} vs {leak.loc[False, 'mean']:.1%} without: keeping them "
                     "hands the model the label. Non-referral rows are "
                     f"{1 - mix['referral'].mean():.1%} of committee rows. Keep only 'Referred to' edges with activity "
                     f"date <= intro + t0; {mix.loc[mix.index.isin(B_ALL), 'referral_after_day7'].mean():.1%} of referral "
                     "edges arrive after day 7 on average. This is a feature-construction requirement for every split, "
                     "not a reason to prefer one split."))

# %% [markdown]
# ### B8. Network structure stability per Congress
#
# **Why it matters.** Structural summaries of the bill-legislator cosponsor graph (density, mean
# degree) and of homophily (share of cosponsorships within the sponsor's party) describe what a GNN
# learns to exploit. Rising polarisation changes how much "same party as sponsor" explains, so a model
# trained in a bipartisan era over-predicts cross-party cosponsors in a polarised one. **How to read
# it.** Per Congress and chamber: density = edges / (bills x members), mean cosponsors per bill,
# same-party share (cosponsor party == sponsor party, D/R only). Pool test: the evaluated value is
# compared with the **mean of that pool's train Congresses**, scaled by ONE fixed noise SD (the spread of
# the same metric over 108-115): t = (value - pool mean) / (SD_ref x sqrt(1 + 1/k)), fail if |t| exceeds
# the 95% t quantile (df = 7). The pool sets the centre (a recent sliding pool is centred near the
# present), the fixed reference sets the width, so S2 is not penalised for having 3 Congresses. Last
# table: for 116-118, the fixed-reference verdict and the trend residual (`trend_z`), which separates a
# steady trend (large t, small trend_z) from a break (large trend_z).

# %%
# Density, mean degree, homophily per Congress x chamber; pool-centred t vs a fixed 108-115 noise SD; failures vs chance per split.
lc = cosp_c[cosp_c.law_capable & cosp_c.chamber.notna()]
nb = bills[bills.law_capable & bills.congress.isin(B_ALL)].groupby(["congress", "chamber"]).size().rename("bills")
nm = mem.groupby(["congress", "chamber"])["bioguide_id"].nunique().rename("members_n")
ne = lc.groupby(["congress", "chamber"]).size().rename("edges")
dr = lc[lc.party.isin(["D", "R"]) & lc.sponsor_party.isin(["D", "R"])]
sp = (dr.party == dr.sponsor_party).groupby([dr.congress, dr.chamber]).mean().rename("same_party")
b8 = pd.concat([nb, nm, ne, sp], axis=1).reset_index()
b8["density"] = b8.edges / (b8.bills * b8.members_n)
b8["mean_cosp_per_bill"] = b8.edges / b8.bills
b8["chamber"] = b8.chamber.map({"House of Representatives": "House", "Senate": "Senate"})
print(b8.pivot(index="congress", columns="chamber", values=["density", "mean_cosp_per_bill", "same_party"]).T
      .round(4).to_string())
line_chart(b8, "congress", "same_party", series="chamber", yfmt=".0%", yzero=False,
           title="Party homophily of cosponsorship per Congress",
           subtitle="Share of cosponsorships (law-capable bills, D/R only) where the cosponsor is in the sponsor's party",
           ylabel="same-party share", name="B_same_party_share")
line_chart(b8, "congress", "density", series="chamber", yfmt=".4f",
           title="Density of the bill-legislator cosponsor graph per Congress",
           subtitle="Active cosponsorships / (law-capable bills x members), by chamber of origin",
           ylabel="density", name="B_density")

B8_METRICS = ["density", "mean_cosp_per_bill", "same_party"]
B8_KEYS = [(mt, ch) for mt in B8_METRICS for ch in ("House", "Senate")]
_b8v = b8.set_index(["chamber", "congress"])
B8_SD = {(mt, ch): float(_b8v.loc[ch, mt].reindex(REL_REF).std(ddof=1)) for mt, ch in B8_KEYS}
B8_TC = float(stats.t.ppf(1 - REL_ALPHA / 2, len(REL_REF) - 1))


def _b8(pool, c):
    out = {}
    for mt, ch in B8_KEYS:
        tr = _b8v.loc[ch, mt].reindex(pool)
        t = (float(_b8v.loc[(ch, c), mt]) - tr.mean()) / (B8_SD[(mt, ch)] * np.sqrt(1 + 1 / len(tr)))
        out[f"{mt}|{ch}"] = t
        out[f"{mt}|{ch}_fail"] = bool(abs(t) > B8_TC)
    return out


b8p = b_eval_pools(_b8)
_fcols = [f"{mt}|{ch}_fail" for mt, ch in B8_KEYS]
b8p["fails"] = b8p[_fcols].sum(axis=1)
_b8show = b8p[["pool", "train", "eval", "fails"]].copy()
for mt, ch in B8_KEYS:
    _b8show[f"{mt[:4]}_{ch[0]}"] = b8p[f"{mt}|{ch}"].map("{:+.1f}".format) + np.where(b8p[f"{mt}|{ch}_fail"], "*", " ")
print(f"\nPool-centred t (fixed 108-115 SD; * = |t| > {B8_TC:.2f}); columns dens / mean / same x House (H) / Senate (S):")
b_show(_b8show, ["fails"] + [c for c in _b8show.columns if c[-2:] in ("_H", "_S")])
b8x = b_excess(b8p, _fcols)
print("Per split, all pools, failures vs chance (5% per test, 6 tests per pool) fails/tests (p_excess): " + "; ".join(
    f"{k} {int(r.fails)}/{int(r.tests)} ({r.p_excess:.2f})" for k, r in b8x.iterrows()))

# Fixed-reference verdict per Congress (pool-independent) and trend-vs-break for 116-118.
_rt = []
for mt, ch in B8_KEYS:
    tab, info = rel_to_train(_b8v.loc[ch, mt].reindex(B_ALL), side="two")
    for c in B_EVAL:
        r = tab.loc[c]
        _rt.append(dict(metric=mt, chamber=ch, congress=c, t=r.t_pi, trend_z=r.trend_z, fail=r.fail, trend_fail=r.trend_fail))
b8rt = pd.DataFrame(_rt)
_piv = b8rt.assign(cell=lambda d: d.t.map("{:+.1f}".format) + "/" + d.trend_z.map("{:+.1f}".format)
                   + np.where(d.trend_fail, " B", np.where(d.fail, " T", "")))
print("\nFixed reference 108-115 per Congress: t vs mean / trend_z (T = outside PI but on trend; B = break from trend):")
print(_piv.pivot(index=["metric", "chamber"], columns="congress", values="cell").to_string())
w8 = int(b8x.fails.max())
_brk = b8rt[b8rt.trend_fail == True]   # noqa: E712
b_record("B", "B8", "Val/test graph structure (density, mean degree, homophily per chamber) close to the pool's train mean, relative to fixed train-to-train noise",
             bool(b8x.ok.all()), metric=w8,
             threshold=f"per split: |t| > {B8_TC:.2f} failures (fixed 108-115 SD, pool-centred) not in excess of chance (binomial p >= 0.05)",
             quantity="graph_structure", by_split=b_split_dict(b8x.ok.to_dict()),
             detail=("Failures / tests (p excess): " + ", ".join(f"{s} {r.fails}/{r.tests} (p={r.p_excess:.2f})" for s, r in b8x.iterrows())
                     + ". Pools with failures (116-118): "
                     + ("; ".join(f"{r.pool} {r.train}->{r['eval']}: " + ", ".join(k.replace('_fail', '') for k in _fcols if r[k])
                                  for _, r in b8p[(b8p.fails > 0) & b8p['eval'].isin(B_EVAL)].iterrows()) or "none")
                     + ". Breaks from the 108-115 linear trend: "
                     + ("; ".join(f"{r.metric}/{r.chamber} {r.congress} (trend_z {r.trend_z:+.1f})" for r in _brk.itertuples()) or "none")
                     + ". Reference SD and centre are separated so pool size does not set the width. Remedy for failures: "
                     "per-Congress normalisation of degree features, per-chamber metrics, a same-party-share feature per "
                     "Congress. Binomial treats the 6 correlated tests per pool as independent (proxy)."))

# %% [markdown]
# ### B9. Dyadic (sponsor-cosponsor pair) novelty: val vs its own test
#
# **Why it matters.** Link-prediction models often succeed by memorising recurring pairs: "X always
# cosponsors Y's bills". If most test positives are pairs already seen in train, a pair-memorising
# baseline will look strong; if most are new pairs, the model must generalise. Each split's **val**
# unit should have a share similar to its **own test** unit, or model selection on val may not
# transfer. Consecutive Congresses differ through elections, so some val-test gap is unavoidable. The
# gap is therefore judged against the **year-to-year swing among train Congresses**: |new_pair(c) -
# new_pair(c-1)| for c in 110-115 with expanding pools (fixed reference, 95% upper t-PI). **How to read
# it.** First table: per pool, share of positives whose (cosponsor, sponsor) pair never occurred in the
# train Congresses, and the share where both legislators were seen (warm nodes, new pair). Second: per
# split, val-test gaps vs the noise bound and failures vs chance. Per-fold gaps are kept in `B_PAIR_GAPS`
# so the decision section can relate them to D's Task-1 val-test gap.

# %%
# New-pair share per pool; val-vs-own-test gap per split/fold judged against the 110-115 year-to-year swing.
# (cosponsor, sponsor) pair codes and a pair x Congress presence matrix (fast "seen in pool" lookups).
_pair_code = pd.Series(pd.MultiIndex.from_arrays([cosp_c.bioguide_id, cosp_c.sponsor_bioguide_id.fillna("")]).factorize()[0],
                       index=cosp_c.index)
_pair_pres = np.zeros((int(_pair_code.max()) + 1, len(B_ALL)), bool)
_pair_pres[_pair_code.to_numpy(), cosp_c.congress.to_numpy() - B_ALL[0]] = True


def _b9(pool, c):
    seen_m = set(mem.loc[mem.congress.isin(pool), "bioguide_id"])
    pe = pos_edges[pos_edges.congress == c]
    new = ~_pair_pres[_pair_code.loc[pe.index].to_numpy()][:, np.asarray(pool) - B_ALL[0]].any(axis=1)
    warm = (pe.bioguide_id.isin(seen_m) & pe.sponsor_bioguide_id.isin(seen_m)).to_numpy()
    return dict(pos_edges=len(pe), new_pair=float(new.mean()), new_pair_warm_nodes=float((new & warm).mean()))


b9 = b_eval_pools(_b9)
b_show(b9, ["pos_edges", "new_pair", "new_pair_warm_nodes"])
# Reference swing: expanding pools 108..c-1 -> c for every c (one-step-ahead, like S3).
np_exp = pd.Series({c: _b9(list(range(B_ALL[0], c)), c)["new_pair"] for c in B_ALL[1:]})
b9_swing = (np_exp - np_exp.shift(1)).abs().dropna()
b9_tab, b9_info = rel_to_train(b9_swing, side="upper")
b9_hi = float(b9_tab.loc[b9_tab.role == "cand", "pi_hi"].iloc[0])
# Per split / fold gap; S1pool test = positive-weighted pooled share of 117 + 118.
u9 = B_UNITS.merge(b9[["pool", "new_pair", "pos_edges"]], on="pool")
u9["w"] = u9.new_pair * u9.pos_edges
g9 = u9.groupby(["split", "fold", "role"], sort=False)[["w", "pos_edges"]].sum()
g9 = (g9.w / g9.pos_edges).unstack("role").dropna()
g9["gap"] = (g9["val"] - g9["test"]).abs()
g9["fail"] = g9.gap > b9_hi
B_PAIR_GAPS = g9.reset_index()
gap_split = g9.groupby(level="split", sort=False).agg(folds=("gap", "size"), max=("gap", "max"), mean=("gap", "mean"),
                                                       fails=("fail", "sum")).reindex(B_UNIT_SPLITS)
gap_split["expected"] = REL_ALPHA * gap_split.folds
gap_split["p_excess"] = [float(stats.binom.sf(f - 1, n, REL_ALPHA)) for f, n in zip(gap_split.fails, gap_split.folds)]
gap_split["ok"] = gap_split.p_excess >= 0.05
print(f"\nYear-to-year swing |new_pair(c) - new_pair(c-1)| (expanding pools), ref 110-115: "
      + ", ".join(f"{c}={v:.3f}" for c, v in b9_swing.items()) + f"; 95% upper t-PI {b9_hi:.3f} "
      f"(nested-LOO false alarms {b9_info['fa_count']}/{b9_info['n']})")
print("|val - test| new-pair share per split vs that bound, max / mean, fails/folds (p_excess): " + "; ".join(
    f"{k} {r['max']:.3f} / {r['mean']:.3f}, {int(r.fails)}/{int(r.folds)} ({r.p_excess:.2f})" for k, r in gap_split.dropna(subset=["folds"]).iterrows()))
line_chart(pd.concat([np_exp.rename("share").reset_index().rename(columns={"index": "congress"}).assign(pool="expanding 108..c-1"),
                      b9[b9.pool.isin(B_UNITS.loc[B_UNITS.split == "S2", "pool"])][["eval", "new_pair"]].rename(columns={"eval": "congress", "new_pair": "share"})
                      .assign(pool=f"sliding {SLIDE_K} Congresses")]),
           "congress", "share", series="pool", yfmt=".0%", yzero=False,
           title="Share of Task-1 positives that are a new (cosponsor, sponsor) pair, by evaluated Congress",
           subtitle="Pair unseen in the train Congresses; swings track elections, and every by-Congress split inherits them",
           ylabel="new-pair share", name="B_new_pair_share")
b_record("B", "B9", "Each split's val unit has a new-pair share similar to its own test unit (relative to train year-to-year swing)",
             bool(gap_split.ok.all()), metric=round(float(gap_split["max"].max()), 4),
             threshold=f"per split: gaps above the 110-115 swing bound ({b9_hi:.3f}, 95% upper t-PI) not in excess of chance",
             by_split=b_split_dict(gap_split.ok.to_dict()),
             detail=("|val - test| new-pair share, max per split (fails/folds): "
                     + ", ".join(f"{s} {r['max']:.3f} ({int(r.fails)}/{int(r.folds)})" for s, r in gap_split.iterrows())
                     + f". Train year-to-year swing 110-115: {b9_swing.loc[110:115].min():.3f}-{b9_swing.loc[110:115].max():.3f} "
                     f"(bound {b9_hi:.3f}). Under S3, val(c) and test(c-1) are the same Congress, so this gap is the election "
                     "swing every by-Congress split inherits; it does not rank splits unless one exceeds the swing more "
                     "often than chance. Use: report a pair-memorisation baseline (score = past pair count) next to the GNN; "
                     "B_PAIR_GAPS (per split / fold) can be correlated with D's Task-1 val-test gap in the decision section. "
                     "+stream variants copy the base verdict (pairs are counted against train Congresses only)."))

# %% [markdown]
# ### B10. As-of-t0 graph completeness
#
# **Why it matters.** A full-graph GNN trained and evaluated on the final snapshot of a Congress sees
# every cosponsor edge of that Congress, including the evaluated bill's own future cosponsors (the
# Task 1 labels) and the later activity of its neighbours. At prediction time (introduction + t0), only
# part of that graph exists. This is the graph-level version of the leakage in B6b / B7: it applies to
# every split, so it is a construction requirement (snapshot or temporal-edge masking), not a split
# preference. Its size also shows how different the as-of-t0 graph is from the final one. **How to
# read it.** For each eval bill (law-capable, uncensored): `visible` = share of its Congress's final
# cosponsor edges dated <= introduction + t0. `nbr_deg_after` = share of the final degree of the bill's
# legislator neighbours known at t0 (sponsor + cosponsors by t0) that accrues after t0. `hop1_vis` =
# (sponsor + cosponsors by t0) / (sponsor + all cosponsors): share of the bill's final 1-hop neighbourhood
# that exists at t0; `hop2_vis` = edges of the as-of 1-hop legislators dated <= t0 / final edges of the final
# 1-hop legislators (2-hop edge mass). Means over bills. The train Congresses (108-115) matter as much as the
# eval ones: training examples must be built the same way (B12).

# %%
# Share of the eval Congress's final cosponsor graph visible at intro + t0, and neighbour degree accrued after t0.
b10 = []
for t0, wt in B_WT.items():
    eb = wt[~wt.censored & wt.congress.isin(B_ALL)][["bill_id", "congress", "introduced_date"]].reset_index(drop=True)
    when = (eb.introduced_date + pd.Timedelta(days=t0)).to_numpy().astype("datetime64[ns]")
    vis = np.empty(len(eb))
    for c, idx in eb.groupby("congress").indices.items():
        dts = np.sort(cosp_c.loc[cosp_c.congress == c, "sponsorship_date"].dropna().to_numpy().astype("datetime64[ns]"))
        vis[idx] = np.searchsorted(dts, when[idx], side="right") / len(dts)
    eb["visible"] = vis
    seen_c = cosp_c[cosp_c.bill_id.isin(eb.bill_id) & (cosp_c.lag_days <= t0)][["bill_id", "bioguide_id"]]
    nbr = pd.concat([eb[["bill_id"]].assign(bioguide_id=eb.bill_id.map(bills.set_index("bill_id")["sponsor_bioguide_id"])),
                     seen_c]).merge(eb[["bill_id", "congress", "introduced_date"]], on="bill_id").dropna(subset=["bioguide_id"])
    nbr["asof"] = b_count_asof(nbr.congress, nbr.bioguide_id, nbr.introduced_date + pd.Timedelta(days=t0))
    nbr["final"] = pd.Series(list(zip(nbr.congress, nbr.bioguide_id))).map(B_DEG_FINAL).fillna(0).to_numpy()
    agg = nbr.groupby("bill_id")[["asof", "final"]].sum()
    eb["nbr_deg_after"] = eb.bill_id.map(1 - agg["asof"] / agg["final"].where(agg["final"] > 0))
    # 1-hop / 2-hop neighbourhood of the bill: as-of (sponsor + cosponsors by t0, their edges by t0) vs final
    # (sponsor + all cosponsors, their final edges).
    fin = pd.concat([eb[["bill_id"]].assign(bioguide_id=eb.bill_id.map(bills.set_index("bill_id")["sponsor_bioguide_id"])),
                     cosp_c.loc[cosp_c.bill_id.isin(eb.bill_id), ["bill_id", "bioguide_id"]]]
                    ).merge(eb[["bill_id", "congress"]], on="bill_id").dropna(subset=["bioguide_id"])
    fin["final"] = B_DEG_FINAL.reindex(pd.MultiIndex.from_arrays([fin.congress, fin.bioguide_id])).fillna(0).to_numpy()
    fg = fin.groupby("bill_id")["final"].agg(["size", "sum"])
    eb["hop1_vis"] = eb.bill_id.map(nbr.groupby("bill_id").size() / fg["size"])
    eb["hop2_vis"] = eb.bill_id.map(agg["asof"] / fg["sum"].where(fg["sum"] > 0))
    b10.append(eb.groupby("congress").agg(visible=("visible", "mean"), visible_p90=("visible", lambda v: v.quantile(0.9)),
                                          nbr_deg_after=("nbr_deg_after", "mean"), hop1_vis=("hop1_vis", "mean"),
                                          hop2_vis=("hop2_vis", "mean")).assign(t0=t0).reset_index())
b10 = pd.concat(b10, ignore_index=True)
print(b10.pivot(index="congress", columns="t0", values=["visible", "nbr_deg_after", "hop1_vis", "hop2_vis"]).T
      .to_string(float_format=b_num))
print("90th percentile of visible share over bills (t0=0): " + ", ".join(
    f"{int(r.congress)} {r.visible_p90:.2f}" for r in b10[b10.t0 == 0].itertuples()))
line_chart(b10.melt(id_vars=["congress", "t0"], value_vars=["visible", "nbr_deg_after"], var_name="m", value_name="share")
           .assign(measure=lambda d: d.m.map({"visible": "final edges visible", "nbr_deg_after": "neighbour degree after t0"})
                   + " (t0=" + d.t0.astype(str) + ")"),
           "congress", "share", series="measure", yfmt=".0%",
           title="How much of a Congress's final cosponsor graph exists at prediction time",
           subtitle="Mean over eval bills: share of the Congress's final edges dated <= intro + t0; share of neighbours' degree accrued after t0",
           ylabel="share", name="B_asof_completeness")
ev10 = b10[(b10.t0 == 0) & b10.congress.isin(B_EVAL)]
miss = float(1 - ev10.visible.mean())
nbr_after = float(ev10.nbr_deg_after.mean())
tr10 = b10[(b10.t0 == 0) & b10.congress.isin(range(108, 116))]
B10_TRAIN_VIS = dict(visible=float(tr10.visible.mean()), hop1=float(tr10.hop1_vis.mean()), hop2=float(tr10.hop2_vis.mean()))
B10_EVAL_VIS = dict(visible=float(ev10.visible.mean()), hop1=float(ev10.hop1_vis.mean()), hop2=float(ev10.hop2_vis.mean()))
b_record("B", "B10", "Share of the eval Congress's final cosponsor graph that is NOT available at t0 (full-graph GNN leakage)",
             None, metric=round(miss, 4), threshold="info: must time-mask edges",
             quantity="leakage_full_graph", requirement=True,
             detail=(f"For eval bills of 116-118 at t0=0, on average {miss:.1%} of their Congress's final cosponsor edges are dated after "
                     f"intro + t0 (train 108-115: {1 - tr10.visible.mean():.1%}), and {nbr_after:.1%} of the final degree of their "
                     "known legislator neighbours accrues after t0. 100% of the bill's own Task-1 positives are in the final "
                     "graph. A GNN fed the final snapshot therefore sees mostly future structure; build the eval graph from "
                     "edges dated <= intro + t0 (temporal masking / event-based snapshots) under every split. The same rule applies "
                     "to TRAINING examples: a GNN trained on Congresses c-k..c-1 with their final graphs (and the train targets "
                     "left in the message-passing graph) sees complete neighbourhoods in training but partial ones at test, a "
                     f"self-inflicted covariate shift. On train Congresses 108-115 at t0=0 the visible share is "
                     f"{B10_TRAIN_VIS['visible']:.1%} of the Congress's edges, {B10_TRAIN_VIS['hop1']:.1%} of the bill's final 1-hop "
                     f"neighbourhood and {B10_TRAIN_VIS['hop2']:.1%} of its final 2-hop edge mass (eval 116-118: "
                     f"{B10_EVAL_VIS['visible']:.1%} / {B10_EVAL_VIS['hop1']:.1%} / {B10_EVAL_VIS['hop2']:.1%}). Requirement: generate each "
                     "training example from the as-of-(intro + t0) snapshot of its own Congress and remove its target edges "
                     "from message passing; B12 measures what ignoring this costs a count baseline. Because the missing "
                     "share is similar across Congresses, this does not favour one split."))

# %% [markdown]
# ### B12. Train / eval input consistency for a temporal GNN
#
# **Why it matters.** B10 shows the eval graph must be the as-of-(intro + t0) snapshot. If the model is
# nevertheless *trained* on final graphs of the train Congresses (with each training bill's own target
# edges still in the message-passing graph), its inputs in training are complete and partly label-bearing,
# while at val/test time they are partial and label-free. The model learns weights for features that look
# different at test time (a covariate shift created by the pipeline, not by the data), and the val score
# understates what a consistent pipeline would reach. This affects every split equally, so it is a
# construction **requirement**; its size tells how strictly the training pipeline must replay time.
# **How it is measured.** A count-feature logistic ranker (log1p of: the candidate's pair count with the
# sponsor in the 3 previous Congresses, its in-Congress pair count with the sponsor, its in-Congress
# degree) is trained on sampled bills of the train Congresses of the S2 pools evaluating 116 and 117, with
# in-Congress features taken (a) from the final graph including the bill's own edges, (b) from the final
# graph minus the bill's own edges, (c) from the as-of-intro snapshot (t0 = 0). All are evaluated on val
# bills with snapshot features (the only inputs available at prediction time). `R20_*` = mean per-bill
# share of positives in the top 20 of the point-in-time same-chamber pool (sponsor and cosponsors at t0
# excluded); `d_*` = R@20 minus the snapshot-trained model's, with a paired bootstrap 95% CI over val
# bills. `hist_only` drops the in-Congress features (reference: what in-Congress history adds at all).
# Proxy: a 3-feature linear model, not a GNN; a GNN aggregates the same in-Congress structure.

# %%
# Count-feature ranker trained on final-graph vs as-of-t0 in-Congress features, evaluated on as-of-t0 features.
from scipy import optimize as _b12opt

B12_N, B12_HIST_K, B12_SEED = 1000, 3, 12
_f12 = B_POOLS[B_POOLS.used_by.str.contains("S2 ") & B_POOLS.c.isin([116, 117])]
B12_FOLDS = [(list(p.train), int(p.c), p.pool) for p in _f12.itertuples()]
_c12 = sorted(set().union(*[set(t) | {c} for t, c, _ in B12_FOLDS]))
_wt12 = B_WT[0]
_pos12 = B_POS[0][["bill_id", "bioguide_id"]]
_eb12 = _wt12[~_wt12.censored & _wt12.congress.isin(_c12) & _wt12.bill_id.isin(_pos12.bill_id)][["bill_id", "congress", "introduced_date"]]
_eb12 = _eb12.sample(frac=1, random_state=B12_SEED).groupby("congress").head(B12_N)
_eb12 = _eb12.merge(bills[["bill_id", "sponsor_bioguide_id", "chamber"]], on="bill_id")
_eb12["month"] = _eb12.introduced_date.dt.to_period("M").dt.to_timestamp()
_cd = _eb12.merge(ROSTER_ASOF_ALL[["month", "chamber", "bioguide_id"]], on=["month", "chamber"])
_cd = _cd[_cd.bioguide_id != _cd.sponsor_bioguide_id]
_own = cosp_c.loc[cosp_c.bill_id.isin(_eb12.bill_id), ["bill_id", "bioguide_id", "lag_days"]]
_cd = _cd.merge(_own, on=["bill_id", "bioguide_id"], how="left")
_cd = _cd[~(_cd.lag_days <= 0)]                         # cosponsors present at t0 are not candidates
_cd["own"] = _cd.lag_days.notna().astype(int)           # candidate cosponsors this bill at some point (final graph)
_cd = _cd.drop(columns="lag_days").merge(_pos12.assign(y=1), on=["bill_id", "bioguide_id"], how="left")
_cd["y"] = _cd.y.fillna(0).astype(int)
_cd = _cd.reset_index(drop=True)

# Pair counts (sponsor -> cosponsor): final per Congress, as of the bill's intro, and over the previous B12_HIST_K Congresses.
_pc = cosp_c[["congress", "sponsor_bioguide_id", "bioguide_id", "sponsorship_date"]].dropna()
_pf = _pc.groupby(["congress", "sponsor_bioguide_id", "bioguide_id"]).size()
_key = pd.MultiIndex.from_arrays([_cd.congress, _cd.sponsor_bioguide_id, _cd.bioguide_id])
_cd["pair_final"] = _pf.reindex(_key).fillna(0).to_numpy()
_hist = pd.concat([_pf.rename("n").reset_index().assign(congress=lambda d, k=k: d.congress + k) for k in range(1, B12_HIST_K + 1)])
_cd["pair_hist"] = _hist.groupby(["congress", "sponsor_bioguide_id", "bioguide_id"])["n"].sum().reindex(_key).fillna(0).to_numpy()
_ev = _pc[_pc.congress.isin(_c12)].copy()
_ev["d"] = _ev.sponsorship_date.astype("datetime64[ns]")
_ev = _ev.sort_values("d", kind="stable")
_ev["n_cum"] = _ev.groupby(["congress", "sponsor_bioguide_id", "bioguide_id"]).cumcount() + 1
_q = _cd[["congress", "sponsor_bioguide_id", "bioguide_id"]].assign(d=_cd.introduced_date.astype("datetime64[ns]"), _i=np.arange(len(_cd)))
_q = pd.merge_asof(_q.sort_values("d", kind="stable"), _ev[["congress", "sponsor_bioguide_id", "bioguide_id", "d", "n_cum"]],
                   on="d", by=["congress", "sponsor_bioguide_id", "bioguide_id"], direction="backward")
_cd["pair_snap"] = _q.sort_values("_i")["n_cum"].fillna(0).to_numpy()
_cd["deg_final"] = B_DEG_FINAL.reindex(pd.MultiIndex.from_arrays([_cd.congress, _cd.bioguide_id])).fillna(0).to_numpy()
_cd["deg_snap"] = b_count_asof(_cd.congress, _cd.bioguide_id, _cd.introduced_date)
B12_VARIANTS = {"final_with_target": ("pair_final", "deg_final", 0), "final_minus_target": ("pair_final", "deg_final", 1),
                "snapshot": ("pair_snap", "deg_snap", 0), "hist_only": (None, None, 0)}


def _b12_X(d, var):
    pc, dc, minus = B12_VARIANTS[var]
    cols = [np.log1p(d.pair_hist.to_numpy())]
    if pc:
        own = minus * d.own.to_numpy()
        cols += [np.log1p(np.maximum(d[pc].to_numpy() - own, 0)), np.log1p(np.maximum(d[dc].to_numpy() - own, 0))]
    return np.column_stack(cols)


def _b12_fit(X, y, l2=1e-4):
    """L2-regularised logistic regression (intercept + weights) by L-BFGS."""
    Xb = np.column_stack([np.ones(len(X)), X])

    def f(w):
        z = Xb @ w
        p = 1 / (1 + np.exp(-z))
        g = Xb.T @ (p - y) / len(y)
        g[1:] += 2 * l2 * w[1:]
        return float((np.logaddexp(0, z) - y * z).mean() + l2 * (w[1:] @ w[1:])), g
    return _b12opt.minimize(f, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B").x


def _b12_recall(d, score, k=20):
    """Per-bill Recall@k (ties broken by a fixed random jitter)."""
    jit = np.random.default_rng(B12_SEED).random(len(d)) * 1e-9
    r = pd.Series(score + jit, index=d.index).groupby(d.bill_id).rank(ascending=False, method="first")
    g = pd.DataFrame({"bill_id": d.bill_id.to_numpy(), "hit": d.y.to_numpy() * (r.to_numpy() <= k), "y": d.y.to_numpy()})
    g = g.groupby("bill_id")[["hit", "y"]].sum()
    return (g.hit / g.y).where(g.y > 0).dropna()


_rng12 = np.random.default_rng(B12_SEED)
_r12, _w12, _d12 = [], [], {v: [] for v in ("final_with_target", "final_minus_target", "hist_only")}
for train, c, pid in B12_FOLDS:
    tr, ev = _cd[_cd.congress.isin(train)], _cd[_cd.congress == c]
    rec = {}
    for v in B12_VARIANTS:
        w = _b12_fit(_b12_X(tr, v), tr.y.to_numpy().astype(float))
        rec[v] = _b12_recall(ev, _b12_X(ev, "hist_only" if v == "hist_only" else "snapshot") @ w[1:])
        _w12.append(dict(pool=pid, variant=v, **{f"w_{n}": x for n, x in zip(["hist", "pair", "deg"], w[1:])}))
    base = rec["snapshot"]
    row = dict(pool=pid, train=_rng_label(train), eval=c, train_bills=int(tr.bill_id.nunique()), val_bills=len(base),
               **{f"R20_{v}": float(r.mean()) for v, r in rec.items()})
    for v in ("final_with_target", "final_minus_target", "hist_only"):
        dlt = (rec[v].reindex(base.index) - base).to_numpy()
        _d12[v].append(dlt)
        bs = dlt[_rng12.integers(0, len(dlt), (1000, len(dlt)))].mean(axis=1)
        row.update({f"d_{v}": float(dlt.mean()), f"d_{v}_lo": float(np.quantile(bs, 0.025)), f"d_{v}_hi": float(np.quantile(bs, 0.975))})
    _r12.append(row)
b12 = pd.DataFrame(_r12)
# Pooled over folds (bill-level paired deltas concatenated; bootstrap over bills).
b12_pool = {}
for v, ds in _d12.items():
    d = np.concatenate(ds)
    bs = d[_rng12.integers(0, len(d), (1000, len(d)))].mean(axis=1)
    b12_pool[v] = (float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))
b12_w = pd.DataFrame(_w12)
# Training-input shift: in-Congress pair count of train positives and mean candidate degree, final graph vs snapshot.
_trc = _cd[_cd.congress.isin(set().union(*[set(t) for t, _, _ in B12_FOLDS]))]
b12_shift = dict(pair_pos_final=float(_trc.loc[_trc.y == 1, "pair_final"].mean()), pair_pos_snap=float(_trc.loc[_trc.y == 1, "pair_snap"].mean()),
                 deg_final=float(_trc.deg_final.mean()), deg_snap=float(_trc.deg_snap.mean()))
print(b12[["pool", "train", "eval", "train_bills", "val_bills", "R20_snapshot", "R20_final_minus_target", "R20_final_with_target",
           "R20_hist_only"]].round(3).to_string(index=False))
for v in ("final_minus_target", "final_with_target", "hist_only"):
    print(f"d_{v} (vs snapshot-trained, 95% CI): " + "; ".join(
        f"{r['pool']} {r['d_' + v]:+.3f} [{r['d_' + v + '_lo']:+.3f}, {r['d_' + v + '_hi']:+.3f}]" for _, r in b12.iterrows())
          + "; pooled {:+.3f} [{:+.3f}, {:+.3f}]".format(*b12_pool[v]))
print("weights hist / in-Congress pair / degree (mean over folds): " + "; ".join(
    f"{v} {g.w_hist.mean():+.2f}/{g.w_pair.mean():+.2f}/{g.w_deg.mean():+.2f}" for v, g in b12_w.groupby("variant", sort=False))
      + f". Train positives' in-Congress pair count: final {b12_shift['pair_pos_final']:.2f} vs snapshot {b12_shift['pair_pos_snap']:.2f}")
_ch12 = pd.concat([b12[["pool", "eval"]].assign(variant=v, delta=b12[f"d_{v}"]) for v in ("final_minus_target", "final_with_target", "hist_only")])
_ch12["label"] = _ch12.pool + " val " + _ch12["eval"].astype(str)
_ch12["variant"] = _ch12.variant.map({"final_minus_target": "trained on final graph, own edges removed",
                                      "final_with_target": "trained on final graph incl. own edges",
                                      "hist_only": "no in-Congress features"})
bar_chart(_ch12, "label", "delta", series="variant", yfmt="+.3f",
          title="Val Recall@20 change vs a ranker trained on as-of-t0 snapshots (eval inputs are always snapshots)",
          subtitle="Count-feature logistic ranker, S2 pools evaluating 116 / 117, t0 = 0; below 0 = inconsistent training inputs cost recall",
          ylabel="Recall@20 minus snapshot-trained", name="B_train_eval_consistency")

b12_sig = [bool(r[f"d_{v}_hi"] < 0) for _, r in b12.iterrows() for v in ("final_minus_target", "final_with_target")] + [
    b12_pool[v][2] < 0 for v in ("final_minus_target", "final_with_target")]
b12_worst = float(b12[["d_final_minus_target", "d_final_with_target"]].min().min())
_b12p = b12.set_index("pool")
b_record("B", "B12", "Training inputs match eval inputs: train examples from as-of-(intro + t0) snapshots, target edges excluded from message passing",
         not any(b12_sig), metric=round(b12_worst, 4),
         threshold="FAIL (= the rule is binding) if training on final-graph features lowers val Recall@20 vs snapshot training (paired 95% CI < 0) in any fold or pooled",
         quantity="train_eval_input_shift", requirement=True,
         detail=(f"Val R@20 (snapshot inputs at eval): snapshot-trained {b_fmt_split(_b12p.R20_snapshot, '.3f')}; final graph minus own "
                 f"edges {b_fmt_split(_b12p.R20_final_minus_target, '.3f')}; final incl. own edges {b_fmt_split(_b12p.R20_final_with_target, '.3f')}; "
                 f"history only {b_fmt_split(_b12p.R20_hist_only, '.3f')}. Paired deltas vs snapshot-trained: " + "; ".join(
                     f"{p}: minus-target {r.d_final_minus_target:+.3f} [{r.d_final_minus_target_lo:+.3f}, {r.d_final_minus_target_hi:+.3f}], "
                     f"with-target {r.d_final_with_target:+.3f} [{r.d_final_with_target_lo:+.3f}, {r.d_final_with_target_hi:+.3f}]"
                     for p, r in _b12p.iterrows())
                 + "; pooled minus-target {:+.3f} [{:+.3f}, {:+.3f}]".format(*b12_pool["final_minus_target"])
                 + ", with-target {:+.3f} [{:+.3f}, {:+.3f}]".format(*b12_pool["final_with_target"])
                 + "; in-Congress features at all (snapshot vs history only) {:+.3f} [{:+.3f}, {:+.3f}]".format(
                     -b12_pool["hist_only"][0], -b12_pool["hist_only"][2], -b12_pool["hist_only"][1])
                 + ". Reading: " + ("target edges left in the training graph cost recall (pooled CI < 0)" if b12_pool["final_with_target"][2] < 0
                                    else "target edges in the training graph: no significant cost for this baseline")
                 + "; " + ("final-graph timing alone (own edges removed) also costs recall" if b12_pool["final_minus_target"][2] < 0
                           else "final-graph timing alone (own edges removed) is not significantly worse for this count baseline, so the "
                                "binding part here is target-edge exclusion; snapshot replay is still required because a GNN's "
                                "aggregations see the full 1-/2-hop shift below, which 3 counts do not")
                 + ". Learned weights (hist / pair / degree) move from "
                 + "{:+.2f}/{:+.2f}/{:+.2f} (snapshot) to {:+.2f}/{:+.2f}/{:+.2f} (final incl. own edges)".format(
                     *b12_w[b12_w.variant == "snapshot"][["w_hist", "w_pair", "w_deg"]].mean(),
                     *b12_w[b12_w.variant == "final_with_target"][["w_hist", "w_pair", "w_deg"]].mean())
                 + f". Training-input shift: train positives' in-Congress pair count with the sponsor is {b12_shift['pair_pos_final']:.2f} in "
                 f"the final graph vs {b12_shift['pair_pos_snap']:.2f} at intro; mean candidate degree {b12_shift['deg_final']:.0f} vs "
                 f"{b12_shift['deg_snap']:.0f}. Train Congresses expose only {B10_TRAIN_VIS['hop1']:.1%} of a bill's final 1-hop and "
                 f"{B10_TRAIN_VIS['hop2']:.1%} of its 2-hop edge mass at t0 (B10). Proxy: {B12_N} sampled bills per Congress, 3-feature "
                 "logistic ranker; a GNN aggregates the same in-Congress structure, so the shift carries over. Rule for every split: "
                 "replay time for training examples (as-of snapshot per example, or temporal edge masking) and drop target edges from "
                 "the message-passing graph (standard link-prediction edge split), exactly as at val / test."))

# %% [markdown]
# ### B11. Node identity across Congresses: chamber and party switchers
#
# **Why it matters.** A legislator keeps one node id across Congresses. When a House member moves to
# the Senate, or a member changes party, that node's training neighbourhood comes from the *other*
# chamber (whose bills they can no longer cosponsor in the same way) or from the other party's
# coalition, while Task 1 ranks only same-chamber legislators and party homophily is strong (B8). A
# learned embedding (or sponsor-history feature) then describes the wrong role. If such nodes carry a
# visible share of test positives and are ranked much worse, the model needs chamber-specific node
# features (or a chamber-switch reset) and the split's test score depends on how many switchers that
# Congress happens to have. **How to read it.** Per pool, Task-1 positives (t0 = 0) are classed by
# their cosponsor's status in their **most recent train Congress**: `cold` (never a member in train),
# `chamber_switch` (served only in the other chamber then), `party_switch` (same chamber, different
# `party_code_asof`), `stable`. `other_ch_edges` = for chamber switchers, share of their train
# cosponsorships on bills of the other chamber. `R@20` = share of positives of each class that a
# `sponsor_hist` ranker (train counts of cosponsoring this sponsor, popularity tie-break, point-in-time
# same-chamber pool incl. delegates, sponsor and t0 cosponsors excluded; section D's baseline,
# re-implemented per positive) places in the top 20. Computed on the S3 folds and all pools evaluating
# 116-118. Impact = switcher share x (R@20 stable - R@20 switcher) = the Recall@20 lost overall.

# %%
# Classify Task-1 positives by the cosponsor's chamber / party in their latest train Congress; sponsor_hist Recall@20 per class.
_mem1 = mem.sort_values("congress").drop_duplicates(["bioguide_id", "congress", "chamber"])
_party_c = mem.drop_duplicates(["bioguide_id", "congress"], keep="last").set_index(["bioguide_id", "congress"])["party_code_asof"]
_lb11 = bills.set_index("bill_id")[["chamber"]]
_pe11 = B_POS[0][["bill_id", "bioguide_id", "sponsor_bioguide_id", "congress", "introduced_date", "chamber"]].copy()
_pe11["month"] = _pe11.introduced_date.dt.to_period("M").dt.to_timestamp()
_ros11 = ROSTER_ASOF_ALL[["month", "chamber", "bioguide_id"]]
_seen11 = cosp_c.loc[cosp_c.law_capable & (cosp_c.lag_days <= 0), ["bill_id", "bioguide_id", "congress"]]
B11_CLASSES = ["stable", "party_switch", "chamber_switch", "cold"]


def _b11_classify(pool, c):
    """Positives of Congress c with their class relative to the latest train Congress in `pool`."""
    pe = _pe11[_pe11.congress == c].copy()
    mt = _mem1[_mem1.congress.isin(pool)]
    last = mt.groupby("bioguide_id")["congress"].max()
    pe["last"] = pe.bioguide_id.map(last)
    lastch = mt[mt.congress == mt.bioguide_id.map(last)].groupby("bioguide_id")["chamber"].agg(set)
    same_ch = np.array([isinstance(st, set) and ch in st for st, ch in zip(pe.bioguide_id.map(lastch), pe.chamber)])
    p_now = pd.Series(list(zip(pe.bioguide_id, pe.congress))).map(_party_c).to_numpy()
    p_then = pd.Series(list(zip(pe.bioguide_id, pe["last"]))).map(_party_c).to_numpy()
    pe["cls"] = np.select([pe["last"].isna(), ~same_ch, (p_now != p_then) & pd.notna(p_now) & pd.notna(p_then)],
                          ["cold", "chamber_switch", "party_switch"], "stable")
    return pe


def _b11_hits(pool, pe):
    """sponsor_hist rank of every positive in its point-in-time same-chamber pool; hit = rank <= 20 after excluding
    the sponsor and cosponsors already present at t0 (ties: average rank)."""
    tr = cosp_c[cosp_c.congress.isin(pool)]
    pop = tr.groupby("bioguide_id").size().rename("pop")
    hist = tr.groupby(["sponsor_bioguide_id", "bioguide_id"]).size().rename("h").reset_index()
    grp = pe[["sponsor_bioguide_id", "chamber", "month"]].drop_duplicates()
    cand = grp.merge(_ros11, on=["month", "chamber"])
    cand = cand[cand.bioguide_id != cand.sponsor_bioguide_id]
    cand = cand.merge(hist, on=["sponsor_bioguide_id", "bioguide_id"], how="left")
    cand["score"] = cand.h.fillna(0) * 1e6 + cand.bioguide_id.map(pop).fillna(0)
    key = ["sponsor_bioguide_id", "chamber", "month"]
    cand["rank"] = cand.groupby(key)["score"].rank(ascending=False, method="average")
    p = pe.merge(cand[key + ["bioguide_id", "score", "rank"]], on=key + ["bioguide_id"], how="left")
    in_pool = p["rank"].notna()
    # cosponsors present at t0 are excluded from the ranking: subtract those scored above the positive
    sn = _seen11[_seen11.bill_id.isin(p.bill_id)].merge(pe[["bill_id"] + key].drop_duplicates("bill_id"), on="bill_id")
    sn = sn.merge(cand[key + ["bioguide_id", "score"]], on=key + ["bioguide_id"]).rename(columns={"score": "s_seen"})
    m = p.loc[in_pool, ["bill_id", "score"]].reset_index().merge(sn[["bill_id", "s_seen"]], on="bill_id")
    above = m[m.s_seen > m.score].groupby("index").size()
    p["adj_rank"] = p["rank"] - p.index.map(above).fillna(0).to_numpy()
    p["hit"] = (p.adj_rank <= 20).where(in_pool)
    return p


_b11_pools = B_POOLS[B_POOLS.pool.isin(B_UNITS.loc[B_UNITS.split == "S3", "pool"]) | B_POOLS.c.isin(B_EVAL)]
_r11, _hits11 = [], []
for pr in _b11_pools.itertuples():
    pool, c = list(pr.train), pr.c
    pe = _b11_hits(pool, _b11_classify(pool, c))
    tr = cosp_c[cosp_c.congress.isin(pool)]
    sw = pe.loc[pe.cls == "chamber_switch", ["bioguide_id", "chamber"]].drop_duplicates()
    t_sw = tr[tr.bioguide_id.isin(sw.bioguide_id)].merge(sw.rename(columns={"chamber": "eval_ch"}), on="bioguide_id")
    r = dict(pool=pr.pool, train=pr.train_label, eval=c, n_pos=len(pe), in_pool=float(pe.hit.notna().mean()),
             other_ch_edges=float((t_sw.chamber != t_sw.eval_ch).mean()) if len(t_sw) else np.nan,
             n_switch_leg=int(pe.loc[pe.cls.isin(["chamber_switch", "party_switch"]), "bioguide_id"].nunique()))
    for k in B11_CLASSES:
        sel = pe.cls == k
        r[f"share_{k}"] = float(sel.mean())
        r[f"R20_{k}"] = float(pe.loc[sel, "hit"].mean()) if pe.loc[sel, "hit"].notna().any() else np.nan
    r["share_switch"] = r["share_chamber_switch"] + r["share_party_switch"]
    hs = pe.loc[pe.cls.isin(["chamber_switch", "party_switch"]), "hit"].dropna()
    r["R20_switch"] = float(hs.mean()) if len(hs) else np.nan
    r["n_switch_pos"] = int(len(hs))
    r["impact"] = r["share_switch"] * max(0.0, r["R20_stable"] - r["R20_switch"]) if len(hs) else 0.0
    _r11.append(r)
    _hits11.append(pe[["congress", "cls", "hit"]].assign(pool=pr.pool))
b11 = pd.DataFrame(_r11)
b_show(b11, ["share_chamber_switch", "share_party_switch", "share_cold", "other_ch_edges", "R20_stable",
             "R20_switch", "R20_cold", "n_switch_pos", "impact"])
_s3p = b11[b11.pool.isin(B_UNITS.loc[B_UNITS.split == "S3", "pool"])]
print("S3 folds, all eval Congresses: switcher share of positives " + ", ".join(f"{r['eval']} {r.share_switch:.2%}" for _, r in _s3p.iterrows())
      + f"; positives in the point-in-time pool >= {b11.in_pool.min():.1%}")

# Pooled over S3 folds: hit rate per class, reference folds (<= 115) vs candidate folds (116-118).
_h = pd.concat(_hits11)
_h = _h[_h.pool.isin(_s3p.pool)].dropna(subset=["hit"])
_h["period"] = np.where(_h.congress.isin(B_EVAL), "eval 116-118", f"eval {_s3p['eval'].min()}-115")
ch11 = _h.groupby(["cls", "period"]).agg(recall20=("hit", "mean"), n=("hit", "size")).reset_index()
print("Pooled S3 R@20 by class: " + "; ".join(f"{r.cls} {r.period} {r.recall20:.3f} (n={r.n:,})" for r in ch11.itertuples()))
bar_chart(ch11.assign(cls=pd.Categorical(ch11.cls, B11_CLASSES, ordered=True)).sort_values(["cls", "period"])
          .assign(cls=lambda d: d.cls.astype(str)),
          "cls", "recall20", series="period", yfmt=".0%",
          title="sponsor_hist Recall@20 of Task-1 positives by cosponsor's node status vs the latest train Congress",
          subtitle="S3 folds, t0 = 0; chamber / party switchers keep one node id whose train history describes another role",
          ylabel="share of positives in top 20", name="B_switcher_recall")

_c11 = b11[b11["eval"].isin(B_EVAL)]
_u11 = B_UNITS.merge(_c11[["pool", "share_switch", "impact"]], on="pool")
b11_share = _u11.groupby("split", sort=False)["share_switch"].max().reindex(B_UNIT_SPLITS)
b11_imp = _u11.groupby("split", sort=False)["impact"].max().reindex(B_UNIT_SPLITS)
_ok11 = lambda sh, im: None if pd.isna(sh) else bool(sh < 0.02 or im < 0.005)
b11_ok = {k: _ok11(b11_share[k], b11_imp[k]) for k in B_UNIT_SPLITS}
_pool_sw = _h[_h.cls.isin(["chamber_switch", "party_switch"])].hit
_pool_st = _h[_h.cls == "stable"].hit
b_record("B", "B11", "Chamber / party switchers (one node id, train neighbourhood from another chamber or party) are a negligible or harmless slice of val/test positives",
             bool(all(v for v in b11_ok.values() if v is not None)), metric=round(float(b11_share.max()), 4),
             threshold="per unit evaluating 116-118: switcher share of positives < 0.02, OR Recall@20 lost = share x (R@20 stable - switcher) < 0.005",
             quantity="node_identity_switch", by_split=b_split_dict(b11_ok),
             detail=("Max switcher share of Task-1 positives (t0=0) per split (units evaluating 116-118): " + b_fmt_split(b11_share, ".2%")
                     + "; max Recall@20 lost: " + b_fmt_split(b11_imp, ".4f")
                     + f". Pooled S3 folds: sponsor_hist R@20 switchers {_pool_sw.mean():.3f} (n={len(_pool_sw):,}) vs stable "
                     f"{_pool_st.mean():.3f} (n={len(_pool_st):,}); chamber switchers' train cosponsorships are on other-chamber bills "
                     f"in {b11.other_ch_edges.min():.0%}-{b11.other_ch_edges.max():.0%} of cases. 2% share bound as in B1b (at most a 2% "
                     "Recall loss even if every switcher positive were missed); 0.005 = half a Recall@20 point. Per-positive hit "
                     "rate (D reports per-bill Recall@20; same ranker, re-implemented here). Remedy if it fails: chamber-specific "
                     "node features (chamber and party as of the eval Congress, not of the train Congress) or reset the "
                     "learned embedding on a chamber change; also report metrics with switchers excluded. +stream variants copy "
                     "the base verdict (in-Congress history would partly fix it). Cross-reference: D25 measures the same slice "
                     "(quantity node_identity_switch) with a learned SVD embedding and a difference-in-differences against this "
                     "count ranker (D runs later in the notebook, so its numbers are not quoted here). If D25 FAILs while B11 passes, "
                     "both hold: the slice is small (bounded Recall@20 impact, judged here per split), but a transductive "
                     "embedding goes stale on it. The decision section sees one node_identity_switch row with a conflict flag; "
                     "read it as 'split-neutral, model-relevant': for a GNN with learned IDs apply the remedy above."))

# %% [markdown]
# ### B summary
#
# All graph and node checks recorded in this section. FAIL does not always mean "do not split by
# Congress": read the detail column for the remedy (allow within-Congress history up to t0, refit on
# val, inductive features, filter edges by time, restrict subjects to a stable vocabulary, normalise
# degree features). Checks tagged with a shared `quantity` are deduplicated in the decision section.
# Every split-dependent check carries a `by_split` verdict (base splits, `+stream` variants, S1pool and,
# where meaningful, S4). Distribution checks (B3, B8, B9) judge each split on failures in excess of chance
# against one fixed 108-115 noise reference, so short sliding pools are not penalised for their size.
# Every check keys every split in `ALL_SPLITS`; splits a check does not judge get None with a stated reason
# (e.g. S5 random, or base splits in B1b / B1c, whose cold share is B1's INFO and is a protocol choice, not a
# failure of the split). B11 and section D's D25 share the `node_identity_switch` key, so the decision
# section sees one row and a conflict flag if they disagree.
# B5b, B6b, B7, B10 and B12 are recorded as **requirements** (`requirement=True`): graph-construction rules every
# split must follow (no subject features at t0, time-filter related links, referral-only committee edges,
# time-masked cosponsor graph for eval AND training examples, target edges out of message passing). Each leakage path has its own quantity key, so none is hidden behind
# another section's check; count split failures from `checks_summary(requirements=False)`.

# %%
# Summary of section B checks.
bsum = pd.DataFrame([c for c in CHECKS if c["section"] == "B"])
bsum["result"] = bsum.passed.map({True: "PASS", False: "FAIL"}).fillna("INFO")
with pd.option_context("display.max_colwidth", 80):
    bsum["req"] = bsum.get("requirement", pd.Series(False, index=bsum.index)).fillna(False).map({True: "req", False: ""})
    print(bsum[["id", "result", "req", "metric", "quantity", "condition"]].to_string(index=False))
_sc = bsum[bsum.req == ""]
print(f"\nB split conditions: {int((_sc.result == 'PASS').sum())} pass, {int((_sc.result == 'FAIL').sum())} fail, "
      f"{int((_sc.result == 'INFO').sum())} info; requirements (all splits): {int((bsum.req == 'req').sum())} "
      f"({int(((bsum.req == 'req') & (bsum.result == 'FAIL')).sum())} need filtering)")
