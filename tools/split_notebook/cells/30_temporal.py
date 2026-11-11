# %% [markdown]
# ## C. Temporal structure, regime changes, censoring, leakage
#
# A by-Congress split assumes that each Congress is a comparable "draw" of the same process, and that
# whatever we feed the model at prediction time (day 0 or day 7 after introduction) was really known at
# that time. This section tests the time-related conditions behind that assumption:
#
# 1. **Within-Congress dynamics**: are bills introduced on the same calendar rhythm in every Congress?
# 2. **Cosponsorship timing**: is the 30-day prediction window equally hard in every Congress?
# 3. **Procedural breakpoints**: does the original-cosponsor share jump on a date inside a Congress, and does
#    each split's fit set contain the post-break regime it is evaluated on?
# 4. **Censoring**: how many bills lose their label because the window runs past the end of the Congress?
# 5. **Regime changes**: is each chamber's controlling party in every validation/test Congress represented in
#    training, and does the joint House x Senate regime matter at all?
# 6. **Snapshot leakage**: which bill fields describe the bill *after* introduction and must never be features?
# 7. **Data latency and label completeness**: is the most recent Congress (118) less complete than older ones, and
#    are its labels (cosponsor events, committee reports) fully loaded, or only its annotations?
# 8. **Member turnover / point-in-time pool**: how much does the full-Congress roster distort task-1 densities
#    and Recall@20 compared with members in office at t0?
# 9. **Feature availability and companion bills**: is the policy area known at t0, and do companion /
#    reintroduced bills connect evaluation bills to the training portion of each candidate split?
# 10. **Task-2 prediction time**: which t0 makes which features legitimate, and how much label is spent by waiting?
#
# "Differs from train" verdicts use the shared `rel_to_train()` rule (95% t prediction interval of the 108-115
# reference, with nested leave-one-out false-alarm rates). Pool-dependent checks iterate over the shared
# evaluation units `iter_eval_units()` (plus the S1pool variant) and record a `by_split` verdict per candidate
# split, so the decision section can score splits without parsing text. Quantities owned by another section
# (label rates -> A) are shown here only as cross-references (INFO).
# Congress 119 is still in session and is used only as context.

# %%
# Shared helpers for this section: train pool, evaluation Congresses, law-capable bill frame.
C_TRAIN = [c for c in CONGRESSES if c <= 115]
C_EVAL = [c for c in CONGRESSES if c >= 116]
C_FMT = lambda v: f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3f}"   # compact float format for transposed tables
C_SECTION = "C"   # section letter: must match QUANTITY_OWNERS so checks_summary() deduplicates correctly
lb = bills[bills.law_capable & bills.congress.isin(CONGRESSES + [IN_PROGRESS])].copy()
lb["c_start"] = lb["congress"].map(congress_start)
lb["c_end"] = lb["congress"].map(congress_end)
wt = {t0: window_targets(t0) for t0 in (0, 7)}   # task-1 windows at t0 = 0 and 7 (shared by C2, C4, C5, C11)
BY_CONGRESS_SPLITS = [k for k, v in SPLIT_DEFS.items() if v["kind"] in ("expanding", "sliding", "rolling")]
EVAL_UNITS = list(iter_eval_units())
C_BY_CONGRESS_ALL = [k for k in ALL_SPLITS if split_kind(k) in BY_CONGRESS_KINDS]   # incl. +stream / S1pool variants


def unit_masks(unit, frame):
    """(train_mask, eval_mask) over `frame` (needs bill_id, congress, introduced_date) for one eval unit.
    S5 uses the shared draw S5_ROLE, S4 the shared cut dates S4_CUT_DATES (same bills as sections A, B, D)."""
    if unit["split"] == "S5":
        role = frame.bill_id.map(S5_ROLE)
        held = ["val", "test"] if unit["role"] == "val" else ["test"]
        return (frame.congress.isin(CONGRESSES) & ~role.isin(held)).to_numpy(), (role == unit["role"]).to_numpy()
    tr = frame.congress.isin(unit["train"]).to_numpy()
    if unit["within"] is None:
        return tr, (frame.congress == unit["eval_congress"]).to_numpy()
    c, q_tr, q_lo, q_hi = unit["within"]
    qd = S4_CUT_DATES[c]
    inc = frame.congress == c
    tr = tr | (inc & (frame.introduced_date < qd[q_tr])).to_numpy()
    ev = inc & (frame.introduced_date >= qd[q_lo]) & (frame.introduced_date < qd[q_hi])
    return tr, ev.to_numpy()


# S1pool (variant, not in iter_eval_units): val 116 fit on 108-115; test 117 and 118 fit once on 108-116.
_S1P = SPLIT_VARIANTS["S1pool"]
S1POOL_UNITS = ([dict(split="S1pool", fold="pool", role="val", train=list(_S1P["train"]), eval_congress=v,
                      train_label=f"{_S1P['train'][0]}-{_S1P['train'][-1]}", within=None) for v in _S1P["val"]]
                + [dict(split="S1pool", fold="pool", role="test", train=list(_S1P["train"]) + list(_S1P["val"]),
                        eval_congress=t, train_label=f"{_S1P['train'][0]}-{_S1P['val'][-1]}", within=None)
                   for t in _S1P["test"]])


def with_variants(d):
    """Copy each base split's per-split verdict to its '+stream' variants (streaming only adds eval-Congress events
    dated before t0, which does not change censoring, regime or twin-leak verdicts). Keys must be in ALL_SPLITS."""
    out = dict(d)
    for k, v in SPLIT_VARIANTS.items():
        if v.get("stream") and v["base"] in d and k not in out:
            out[k] = d[v["base"]]
    return {k: out[k] for k in ALL_SPLITS if k in out}


def combine_by_split(*ds):
    """AND of several by_split dicts (None = not judged; a split fails if any input fails, None if all are None)."""
    out = {}
    for k in ALL_SPLITS:
        vals = [d.get(k) for d in ds if d is not None and k in d]
        if vals:
            judged = [v for v in vals if v is not None]
            out[k] = all(judged) if judged else None
    return out


S4_EVAL = split_eval_congresses("S4")   # candidate Congresses whose within-Congress slices S4 evaluates
S5_NA = ("S5 draws train and eval bills at random from every Congress 108-118, so a per-Congress drift verdict does not "
         "separate its train from its eval (S5's leakage is judged in C20)")
NO_CAND_NA = "evaluates no candidate Congress 116-118 (reference-Congress verdicts are nested-LOO and not held against splits)"


def cand_ok(*tabs):
    """Per candidate Congress: True if it passes every given rel_to_train tab (rows with role 'cand')."""
    oks = [~t.loc[t.role == "cand", "fail"].astype(bool) for t in tabs]
    return pd.concat(oks, axis=1).all(axis=1)


def full_split(bs, ok=None, s4_na=None, s5_na=S5_NA, extra=None, extra_na=None):
    """Complete a by_split dict over ALL_SPLITS and build its na_reason. ok: per-Congress pass Series (cand_ok) used to
    judge S4 on the candidate Congresses its slices evaluate (S4_EVAL); if None, S4 gets None with reason s4_na.
    extra: explicit verdicts for other splits (with reasons extra_na for None values). Returns (by_split, na_reason)."""
    out, na = dict(bs or {}), {}
    if ok is not None:
        out["S4"] = bool(all(ok[c] for c in S4_EVAL if c in ok.index))
    else:
        out["S4"], na["S4"] = None, s4_na
    out["S5"] = None
    na["S5"] = s5_na
    out.update(extra or {})
    na.update(extra_na or {})
    out = {k: out.get(k) for k in ALL_SPLITS}
    for k, v in out.items():
        if v is None and k not in na:
            na[k] = NO_CAND_NA
    return out, {k: v for k, v in na.items() if out[k] is None}


print(f"law-capable bills 108-119: {len(lb):,}; train pool {C_TRAIN[0]}-{C_TRAIN[-1]}, "
      f"evaluation candidates {C_EVAL}")

# %% [markdown]
# ### C1-C2. When are bills introduced inside a Congress?
#
# **Why it matters.** Bills introduced in the first weeks of a Congress (reintroductions, leadership
# priorities) and bills dropped in the final months (messaging bills, the end-of-Congress rush) behave
# differently: early bills have two years to gather cosponsors, late bills are mostly censored. If the
# introduction-time profile differs between train and test Congresses, a model learns a mix that the test
# set does not have. It also tells us whether a test set must be a **full Congress**: a partial window
# (for example session 1 only, as in S4) has a very different bill mix from a full Congress.
#
# **How to read it.** The heatmap shows the share of each Congress's bills introduced in each quarter of
# the Congress (Q1 = first three months). The table gives the session-1 share, the share introduced in the
# last 90 days, and the KS distance of `day_of_congress` versus the 108-115 pool (KS is an effect size here:
# with ~10k bills every p-value is tiny, so we threshold the statistic, not the p-value).

# %%
# C1/C2: distribution of introduction day_of_congress per Congress; session-1 share; last-90-day rush; KS vs train pool.
lb["quarter"] = (lb["day_of_congress"] // 91.3125).clip(0, 7).astype(int) + 1
lb["session1"] = lb["introduced_date"] < lb["c_start"] + pd.DateOffset(years=1)
lb["rush90"] = lb["introduced_date"] >= lb["c_end"] - pd.Timedelta(days=90)
train_days = lb.loc[lb.congress.isin(C_TRAIN), "day_of_congress"].values
rows = []
for c, g in lb[lb.congress.isin(CONGRESSES)].groupby("congress"):
    pool_days = lb.loc[lb.congress.isin(C_TRAIN) & (lb.congress != c), "day_of_congress"].values
    rows.append(dict(congress=c, n_bills=len(g), session1=g.session1.mean(), last90=g.rush90.mean(),
                     median_day=g.day_of_congress.median(), ks_vs_train=ks(pool_days, g.day_of_congress)[0]))
intro = pd.DataFrame(rows).set_index("congress")
print("Introduction timing per Congress (law-capable bills; KS vs 108-115 pool excluding the row's own Congress)")
print(intro.T.to_string(float_format=C_FMT))

qshare = (lb[lb.congress.isin(CONGRESSES)].groupby(["congress", "quarter"]).size()
          / lb[lb.congress.isin(CONGRESSES)].groupby("congress").size()).rename("share").reset_index()
qshare["quarter"] = "Q" + qshare["quarter"].astype(str)
qshare["congress"] = qshare["congress"].astype(str)
heatmap(qshare, "quarter", "congress", "share", fmt=".0%",
        title="Share of each Congress's law-capable bills introduced per quarter of the Congress",
        subtitle="Q1 = first 3 months, Q8 = last 3 months; rows are Congresses", height=420,
        name="C_intro_quarter_heatmap")

# Verdict: shared rel_to_train rule (train rows = LOO KS to the other 108-115 Congresses; upper tail only).
ks_tab, ks_info = rel_to_train(intro["ks_vs_train"], side="upper")
C12_S4_NA = ("S4 evaluates a fixed slice of each candidate Congress (its last 40% of introductions: val 60-80%, test >= 80%), so its introduction-time mix "
             "differs from the full Congress by construction; the whole-Congress timing profile does not describe it (C2's "
             "session-2 contrast and C6b cover S4)")
ks_eval = intro.loc[C_EVAL, "ks_vs_train"]
record_check(C_SECTION, "C1", "Introduction-day distribution (KS vs 108-115) of every eval Congress within train-to-train noise",
             not bool(ks_tab.loc[ks_tab.role == "cand", "fail"].any()), metric=float(ks_eval.max()), threshold=ks_info["rule"],
             detail=rel_detail(ks_tab, ks_info) + ". by_split judges only the candidate Congresses 116-118 each split evaluates "
                    f"(shared 'cands' mode; reference outliers such as 112, KS {intro.loc[112, 'ks_vs_train']:.3f} with a short session 1, are not "
                    "held against S2/S3). Secondary: max eval KS {ks_eval.max():.3f} vs conventional small-effect "
                    "0.1. p-values of KS itself are meaningless at this n, so the statistic is judged against its own LOO spread.",
             **dict(zip(("by_split", "na_reason"), full_split(rel_by_split(ks_tab), s4_na=C12_S4_NA))))
s1_train, r_train = intro.loc[C_TRAIN, "session1"].mean(), intro.loc[C_TRAIN, "last90"].mean()
# Session-1 vs session-2 bills (train pool, uncensored, t0=7): how different is the bill mix a within-Congress split tests on?
w7 = wt[7][~wt[7].censored & wt[7].congress.isin(C_TRAIN) & (wt[7].eligible > 0)].merge(lb[["bill_id", "session1"]], on="bill_id")
w7["rate"] = w7.n_pos / w7.eligible
sess = w7.groupby("session1").agg(bills=("bill_id", "size"), pos_rate=("rate", "mean"), any_pos=("n_pos", lambda v: (v > 0).mean()))
sess.index = sess.index.map({True: "session 1", False: "session 2"})
print("Train pool 108-115, task 1 at t0=7: session-1 vs session-2 bills")
print(sess.round(4).to_string())
sess_rel = float(sess.loc["session 2", "pos_rate"] / sess.loc["session 1", "pos_rate"] - 1)
# Verdict: shared rel_to_train rule on both shares (two-sided: a rush that shrinks matters as much as one that grows).
s1_tab, s1_info = rel_to_train(intro["session1"], side="two")
r9_tab, r9_info = rel_to_train(intro["last90"], side="two")
c2_fail = bool(s1_tab.loc[s1_tab.role == "cand", "fail"].any() or r9_tab.loc[r9_tab.role == "cand", "fail"].any())
c2_t = float(max(s1_tab.loc[s1_tab.role == "cand", "t_pi"].abs().max(), r9_tab.loc[r9_tab.role == "cand", "t_pi"].abs().max()))
record_check(C_SECTION, "C2", "Session-1 share and end-of-Congress (last 90 d) share of eval Congresses within train-to-train noise",
             not c2_fail, metric=c2_t, threshold=f"{s1_info['rule']}, for either share (metric = max |t| over candidates and both shares)",
             detail=f"session-1 share: {rel_detail(s1_tab, s1_info)}. Last-90-day share: {rel_detail(r9_tab, r9_info)}. Train mean session-1 "
                    f"share {s1_train:.3f}, last-90-day share {r_train:.3f}. Session-2 bills have a task-1 positive rate {abs(sess_rel):.0%} "
                    f"{'lower' if sess_rel < 0 else 'higher'} than session-1 bills (t0=7, train pool), so a within-Congress split (S4) that trains "
                    "on session 1 and tests on session 2 tests a structurally different bill mix; use full Congresses as test units.",
             **dict(zip(("by_split", "na_reason"), full_split(combine_by_split(rel_by_split(s1_tab), rel_by_split(r9_tab)),
                                                              s4_na=C12_S4_NA))))

# %% [markdown]
# ### C3-C4. Cosponsorship timing and task difficulty
#
# **Why it matters.** Task 1 predicts cosponsors who join in *(t0, t0+30]*. Beyond *how many* cosponsors a
# bill gets (label rate, owned by section A: A3/A4), what matters for the window is *when* the later
# cosponsors arrive: if in some Congresses non-original cosponsors trickle in over months while in others
# they join in the first weeks, the same 30-day window captures a different slice of the cosponsor list.
#
# **How to read it.** The line chart shows, per Congress, the cumulative share of cosponsorships that are
# original (lag 0) and that arrive within 7, 30 and 37 days (37 = end of the t0=7 window). C3 tests the
# **timing of non-original cosponsorships** (lag > 0) with a KS distance against the 108-115 pool, and judges
# it against the leave-one-out spread among the train Congresses themselves (a FAIL means "more different than
# train-to-train noise", not just "different"). C4 (INFO) is a cross-reference to section A's label rate.

# %%
# C3: lag distribution per Congress (law-capable, active cosponsorships); KS of non-original lags vs train pool (LOO for train rows).
lc = cosp[cosp.law_capable & cosp.congress.isin(CONGRESSES)][["congress", "lag_days", "is_original"]]
lag = lc.groupby("congress")["lag_days"].agg(
    n="size", original=lambda s: (s == 0).mean(), neg=lambda s: (s < 0).mean(),
    within7=lambda s: (s <= 7).mean(), within30=lambda s: (s <= 30).mean(), within37=lambda s: (s <= 37).mean(),
    median_lag=lambda s: s[s > 0].median())
late = lc[lc.lag_days > 0]
# share of non-original cosponsorships arriving inside the t0=7 window (8..37 d): the part of the timing the task sees
lag["late_in_t7_window"] = late.groupby("congress")["lag_days"].agg(lambda s: ((s > 7) & (s <= 37)).mean())
lag["ks_late_vs_train"] = [ks(late.loc[late.congress.isin(C_TRAIN) & (late.congress != c), "lag_days"].values,
                              late.loc[late.congress == c, "lag_days"].values)[0] for c in lag.index]
print("Cosponsorship lag per Congress (median_lag and KS over non-original cosponsorships, lag > 0; "
      "neg = date before introduction; train rows use leave-one-out pools)")
print(lag.drop(columns=["n", "neg"]).round(3).to_string())
ll = lag[["original", "within7", "within30", "within37"]].reset_index().melt("congress", var_name="share_of", value_name="share")
ll["share_of"] = ll["share_of"].map({"original": "original (lag 0)", "within7": "<= 7 d", "within30": "<= 30 d", "within37": "<= 37 d"})
line_chart(ll, "congress", "share", series="share_of", yfmt=".0%", yzero=False,
           title="Cumulative share of cosponsorships by lag after introduction, per Congress",
           subtitle="Law-capable bills, active (non-withdrawn) cosponsorships", xlabel="Congress",
           name="C_lag_shares")
ks_e, ks_t = lag.loc[C_EVAL, "ks_late_vs_train"], lag.loc[C_TRAIN, "ks_late_vs_train"]
lag_tab, lag_info = rel_to_train(lag["ks_late_vs_train"], side="upper")
band = float(ks_t.max())
record_check(C_SECTION, "C3", "Timing of non-original cosponsorships (lag > 0) in eval Congresses within train-to-train variation",
             not bool(lag_tab.loc[lag_tab.role == "cand", "fail"].any()), metric=float(ks_e.max()),
             threshold=lag_info["rule"], **dict(zip(("by_split", "na_reason"), full_split(rel_by_split(lag_tab), ok=cand_ok(lag_tab)))),
             detail=rel_detail(lag_tab, lag_info) + f". Secondary (old rule): max train LOO KS {band:.3f}; train leave-one-out range "
                    f"{ks_t.min():.3f}-{ks_t.max():.3f} (the noise band; 0.1 is the conventional 'small effect' KS, so a value between "
                    f"the band and 0.1 is a real but modest shift). Worst eval Congress: {int(ks_e.idxmax())} (median non-original lag "
                    f"{lag.loc[int(ks_e.idxmax()), 'median_lag']:.0f} d vs train {lag.loc[C_TRAIN, 'median_lag'].median():.0f} d). Share of late cosponsorships landing in the t0=7 window (8-37 d): train "
                    f"mean {lag.loc[C_TRAIN, 'late_in_t7_window'].mean():.3f}, eval {lag.loc[C_EVAL, 'late_in_t7_window'].round(3).to_dict()}. "
                    f"Original share (lag 0) is owned by A4 and not part of this verdict (train mean {lag.loc[C_TRAIN, 'original'].mean():.3f}, "
                    f"eval {lag.loc[C_EVAL, 'original'].round(3).to_dict()}). Negative-lag rows {lag['neg'].mean():.2%} (data errors, "
                    "excluded from windows). If FAIL: the window catches a different part of the cosponsor arrival curve in eval "
                    "Congresses (118 only; C23 tests whether its lag shift is a within-Congress break); use recency weighting or a post-break indicator and "
                    "report per-Congress results rather than relying on one test Congress. S4 is judged on the same candidate "
                    f"Congresses its slices test ({S4_EVAL}): the lag shift belongs to the Congress's bills, so S4's test slice of a failing "
                    "Congress inherits it (its fit set holds only the earlier part of that Congress, whose late cosponsorships are "
                    "partly censored at the cut).",
             quantity="lag_dist")

# %%
# C4 (INFO, cross-reference to A3): task-1 positive rate per eligible legislator per Congress (uncensored law-capable bills).
diff_rows = []
for t0, w in wt.items():
    u = w[~w.censored & w.congress.isin(CONGRESSES) & (w.eligible > 0)].copy()
    u["rate"] = u["n_pos"] / u["eligible"]
    g = u.groupby("congress").agg(bills=("bill_id", "size"), pos_rate=("rate", "mean"),
                                  any_pos=("n_pos", lambda s: (s > 0).mean()), mean_pos=("n_pos", "mean"))
    g["t0"] = t0
    diff_rows.append(g.reset_index())
difficulty = pd.concat(diff_rows)
tab = difficulty.pivot(index="congress", columns="t0", values=["pos_rate", "any_pos", "mean_pos"])
tab.columns = [f"{a}_t{b}" for a, b in tab.columns]
print("Task-1 difficulty per Congress (pos_rate = mean positives / eligible; any_pos = share of bills with >= 1 positive)")
print(tab[["pos_rate_t0", "pos_rate_t7", "any_pos_t7"]].round(4).T.to_string())
rel = {}
for t0 in (0, 7):
    d = difficulty[difficulty.t0 == t0].set_index("congress")["pos_rate"]
    rel[t0] = ((d.loc[C_EVAL] - d.loc[C_TRAIN].mean()) / d.loc[C_TRAIN].mean())
worst = max(rel[0].abs().max(), rel[7].abs().max())
record_check(C_SECTION, "C4", "Task-1 positive rate per eligible legislator, eval vs train (cross-reference; verdict owned by A3)", None,
             metric=float(worst), threshold="INFO (A3 owns the verdict)",
             detail=f"max relative diff vs 108-115 mean; t0=0: {rel[0].round(3).to_dict()}; t0=7: {rel[7].round(3).to_dict()}. "
                    "Shown for the timing context only; not counted separately in the decision.",
             quantity="cosp_density")

# %% [markdown]
# ### C23. Procedural breakpoints inside a Congress (original-cosponsor share and timing)
#
# **Why it matters.** The jump in the original-cosponsor share in 117-118 (table above) may come from a
# procedural change (for example electronic cosponsor submission and remote operations in the House in 2020-2021)
# rather than from gradual drift. A procedural change is a *break*: it happens on a date, not at a Congress
# boundary. If the break falls inside a Congress, that Congress is a mix of two regimes, a validation Congress
# can straddle it, and a fit set that ends before it has never seen the new regime. That decides whether a
# by-Congress cut is enough or whether a within-Congress cut (or dropping pre-break data) is needed.
#
# **How to read it.** Monthly series by introduction month (law-capable bills, uncensored at t0 = 0): share of
# 30-day cosponsors that are original, and the median lag of non-original cosponsors joining within 180 days (bills
# with a full 180-day window). Each series is de-seasonalised by its month-of-Congress mean (to remove the
# start/end-of-Congress cycle). A constant-mean CUSUM ("raw") would call any steady trend a break, because block
# permutation destroys trends; the verdict therefore uses the largest level shift BEYOND a linear trend (trend and
# step fitted jointly, segments >= 6 months; null = linear trend + 2,000 permutations of 6-month residual blocks,
# approximate). C23b then asks whether a split has more test folds on a never-seen post-break regime than
# deployment itself would. The chart shows the raw monthly series with Congress boundaries and the trend-adjusted break.

# %%
# C23: monthly original share / median lag per chamber, de-seasonalised; largest mean shift with block-permutation p-value.
mc = cosp[cosp.law_capable & cosp.congress.isin(CONGRESSES) & (cosp.lag_days >= 0)][
    ["bill_id", "congress", "chamber", "introduced_date", "lag_days"]].copy()
mc["end"] = mc.congress.map(congress_end)
mc["month"] = mc.introduced_date.dt.to_period("M").dt.to_timestamp()
mc["moc"] = ((mc.month.dt.year - mc.congress.map(congress_start).dt.year) * 12 + mc.month.dt.month - 1).clip(0, 23)
w30 = mc[(mc.lag_days <= 30) & (mc.introduced_date + pd.Timedelta(days=30) <= mc.end)]
l180 = mc[(mc.lag_days > 0) & (mc.lag_days <= 180) & (mc.introduced_date + pd.Timedelta(days=180) <= mc.end)]
MIN_N = 200   # months with fewer cosponsorships are dropped (noisy)


def cusum_break(x, minseg=6, n_perm=2000, block=6, seed=23):
    """Largest standardised mean shift and its block-permutation p-value."""
    def stat(v):
        n = v.shape[-1]
        k = np.arange(minseg, n - minseg + 1)
        cs = np.cumsum(v, axis=-1)
        m1 = cs[..., k - 1] / k
        m2 = (cs[..., -1:] - cs[..., k - 1]) / (n - k)
        z = np.abs(m1 - m2) * np.sqrt(k * (n - k) / n) / v.std(axis=-1, keepdims=True)
        return z.max(-1), k[z.argmax(-1)]
    s_obs, k_obs = stat(x)
    nb = int(np.ceil(len(x) / block))
    blocks = np.array_split(np.arange(len(x)), nb)
    rng = np.random.default_rng(seed)
    perms = np.stack([x[np.concatenate([blocks[i] for i in rng.permutation(nb)])] for _ in range(n_perm)])
    s_null, _ = stat(perms)
    return float(s_obs), int(k_obs), float((s_null >= s_obs).mean())


def trend_break(x, t, minseg=6, n_perm=2000, block=6, seed=23):
    """Largest level shift BEYOND a linear trend: max |t| of the step coefficient in y ~ 1 + t + 1(i >= k) over k
    (trend and step estimated jointly, so a steady rise is not a break and a late break does not bend the trend).
    Null: linear trend over the whole series + block-permuted residuals. Returns (stat, k, p, slope_per_unit)."""
    n = len(x)
    Z = np.column_stack([np.ones(n), t])
    M = np.eye(n) - Z @ np.linalg.pinv(Z)                          # residual maker for [1, t]
    ks_ = np.arange(minseg, n - minseg + 1)
    S = (np.arange(n)[None, :] >= ks_[:, None]).astype(float) @ M  # residualised step regressors (K x n)
    ss = (S ** 2).sum(1)

    def stat(Y):                                                   # Y: (P x n) already residualised on [1, t]
        b = Y @ S.T / ss
        s2 = ((Y ** 2).sum(1, keepdims=True) - b ** 2 * ss) / (n - 3)
        tt = np.abs(b) * np.sqrt(ss) / np.sqrt(np.maximum(s2, 1e-12))
        return tt.max(1), ks_[tt.argmax(1)]
    e = M @ x
    s_obs, k_obs = stat(e[None, :])
    nb = int(np.ceil(n / block))
    blocks = np.array_split(np.arange(n), nb)
    rng = np.random.default_rng(seed)
    perms = np.stack([e[np.concatenate([blocks[i] for i in rng.permutation(nb)])] for _ in range(n_perm)]) @ M.T
    s_null, _ = stat(perms)
    slope = float(np.linalg.lstsq(Z, x, rcond=None)[0][1])
    return float(s_obs[0]), int(k_obs[0]), float((s_null >= s_obs[0]).mean()), slope


series_defs = [("orig_share30", w30, lambda s: (s == 0).mean()), ("median_lag180", l180, "median")]
bp_rows, plot_rows = [], []
for name, df, f in series_defs:
    for ch in ("House of Representatives", "Senate"):
        d = df[df.chamber == ch]
        g = d.groupby(["month", "congress", "moc"]).lag_days.agg(n="size", v=f).reset_index()
        g = g[g.n >= MIN_N].sort_values("month").reset_index(drop=True)
        g["v_ds"] = g.v - g.groupby("moc").v.transform("mean")
        # Detrended: level shift beyond a linear trend (trend and step fitted jointly), so a steady rise is not a break.
        tt = (g.month - g.month.min()).dt.days.to_numpy() / 365.25
        stt, k, pval = cusum_break(g.v_ds.to_numpy())
        stt2, k2, pval2, b1 = trend_break(g.v_ds.to_numpy(), tt)
        Z2 = np.column_stack([np.ones(len(g)), tt, (np.arange(len(g)) >= k2).astype(float)])
        g["v_dt"] = g.v_ds - Z2[:, :2] @ np.linalg.lstsq(Z2, g.v_ds.to_numpy(), rcond=None)[0][:2]
        into2 = int(g.moc[k2])
        bp_rows.append(dict(series=name, chamber=ch[:5], months=len(g), raw_break=g.month[k].strftime("%Y-%m"), raw_p=pval,
                            trend_per_yr=b1, break_month=g.month[k2].strftime("%Y-%m"), congress=int(g.congress[k2]),
                            month_of_congress=into2 + 1, before=g.v[:k2].mean(), after=g.v[k2:].mean(),
                            shift_sd=(g.v_dt[k2:].mean() - g.v_dt[:k2].mean()) / g.v_dt[:k2].std(), stat=stt2, p=pval2,
                            inside=bool(3 <= into2 <= 20)))
        if name == "orig_share30":
            plot_rows += [dict(t=round(m.year + (m.month - 0.5) / 12, 3), v=float(v), chamber=ch[:5], congress=int(c))
                          for m, v, c in zip(g.month, g.v, g.congress)]
bp = pd.DataFrame(bp_rows)
print("Largest level shift per monthly series: raw = constant-mean CUSUM on the de-seasonalised series; trend-adjusted (break_month "
      ".. p) = step beyond a jointly fitted linear trend (verdict); before/after = raw means; trend_per_yr = slope without a step")
print(bp.drop(columns=["months", "stat"]).round(3).to_string(index=False))
alpha23 = 0.05 / len(bp)
breaks = [dict(year=round(pd.Timestamp(r.break_month).year + (pd.Timestamp(r.break_month).month - 1) / 12, 3),
               label=f"{r.chamber} break {r.break_month} (p={r.p:.3f}{'' if abs((pd.Period(r.raw_break, 'M') - pd.Period(r.break_month, 'M')).n) <= 6 else ', location not identified'})") for r in bp[bp.series == "orig_share30"].itertuples()]
bounds = [round(congress_start(c).year + 0.0, 3) for c in CONGRESSES[1:]]
show_d3(r"""
const {width, height, breaks, bounds} = opts;
const m = {top: 48, right: 20, bottom: 56, left: 56};
const svg = d3.select(el).append('svg').attr('width', width).attr('height', height);
viz.title(svg, 'Monthly share of 30-day cosponsors that are original, by chamber (law-capable bills)',
          'Thin grey lines = Congress boundaries; dashed = largest level shift beyond a linear trend (de-seasonalised)');
const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
const W = width - m.left - m.right, H = height - m.top - m.bottom;
const xs = d3.scaleLinear(d3.extent(data, d => d.t), [0, W]);
const ys = d3.scaleLinear(d3.extent(data, d => d.v), [H, 0]).nice();
g.append('g').call(d3.axisLeft(ys).ticks(5).tickSize(-W).tickFormat(d3.format('.0%')))
  .call(a => a.selectAll('line').attr('stroke', viz.grid)).call(a => a.select('.domain').remove())
  .call(a => a.selectAll('text').attr('fill', viz.ink2).style('font', '11px sans-serif'));
g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xs).ticks(12).tickFormat(d3.format('d'))).call(viz.axis);
g.selectAll(null).data(bounds).join('line').attr('x1', d => xs(d)).attr('x2', d => xs(d)).attr('y1', 0).attr('y2', H)
  .attr('stroke', viz.grid).attr('stroke-width', 1);
const keys = ['House', 'Senat'];
const color = d3.scaleOrdinal(keys, viz.palette);
const tip = viz.tooltip(el);
keys.forEach(k => {
  const rows = data.filter(d => d.chamber === k).sort((a, b) => a.t - b.t);
  g.append('path').datum(rows).attr('fill', 'none').attr('stroke', color(k)).attr('stroke-width', 1.5)
    .attr('d', d3.line().x(d => xs(d.t)).y(d => ys(d.v)));
  g.selectAll(null).data(rows).join('circle').attr('cx', d => xs(d.t)).attr('cy', d => ys(d.v)).attr('r', 2.5)
    .attr('fill', color(k)).on('mousemove', (e, d) => tip.show(`<b>${k} ${d.t.toFixed(2)}</b> (Congress ${d.congress})<br>${d3.format('.1%')(d.v)}`, e))
    .on('mouseleave', () => tip.hide());
});
breaks.forEach((b, i) => {
  g.append('line').attr('x1', xs(b.year)).attr('x2', xs(b.year)).attr('y1', 0).attr('y2', H)
    .attr('stroke', viz.ink).attr('stroke-width', 1.5).attr('stroke-dasharray', '5,4');
  g.append('text').attr('x', xs(b.year) - 4).attr('y', 12 + 14 * i).attr('text-anchor', 'end').attr('fill', viz.ink)
    .style('font', '11px sans-serif').text(b.label);
});
viz.legend(svg, [['House', color('House')], ['Senate', color('Senat')]], m.left, height - 10);
""", plot_rows, height=360, width=760, opts=dict(breaks=breaks, bounds=bounds), name="C_orig_share_breaks")


def _post_break_units(bdates):
    """Per eval unit: does it contain bills introduced after a break date while its fit set contains none (the eval
    regime was never seen in training)? bdates = first day of each post-break segment."""
    rows = []
    for un in EVAL_UNITS + S1POOL_UNITS:
        if un["split"] == "S5":
            continue
        fit_end = max(congress_end(c) for c in un["train"])
        ev_end = congress_end(un["eval_congress"])
        if un["within"] is not None:
            c, q_tr, q_lo, q_hi = un["within"]
            fit_end = max(fit_end, S4_CUT_DATES[c][q_tr])
            ev_end = min(S4_CUT_DATES[c][q_hi], congress_end(c))
        rows.append(dict(split=un["split"], fold=un["fold"], role=un["role"], eval_congress=int(un["eval_congress"]),
                         unseen=any((ev_end > d) and (fit_end <= d) for d in bdates)))
    return pd.DataFrame(rows)


bp["break_date"] = [max(pd.Timestamp(m), congress_start(c)) for m, c in zip(bp.break_month, bp.congress)]
# A break location is "identified" when the constant-mean and trend-adjusted fits put it within 6 months of each other;
# otherwise the single-break model is misspecified (e.g. a V-shaped series) and no Congress can be said to straddle it.
bp["identified"] = [abs((pd.Period(a, "M") - pd.Period(b_, "M")).n) <= 6 for a, b_ in zip(bp.raw_break, bp.break_month)]
sig_all = bp[bp.p < alpha23]
sig = sig_all[sig_all.identified]
unid = sig_all[~sig_all.identified]
sig_in = sig[sig.inside]
raw_sig = bp[bp.raw_p < alpha23]
brk_txt = "; ".join(f"{r.series}/{r.chamber}: raw {r.raw_break} (p={r.raw_p:.3f}) -> detrended {r.break_month} (Congress {r.congress}, "
                    f"month {r.month_of_congress}; {r.before:.3f} -> {r.after:.3f}, {r.shift_sd:+.1f} SD, p={r.p:.3f}; linear trend "
                    f"{r.trend_per_yr:+.4f}/yr)" for r in bp.itertuples())
lost_after_dt = [(r.series, r.chamber) for r in bp.itertuples() if r.raw_p < alpha23 and r.p >= alpha23]
record_check(C_SECTION, "C23", "No significant structural break (beyond a linear trend) in original-cosponsor share / cosponsor timing inside a Congress",
             len(sig_in) == 0, metric=int(len(sig_in)),
             threshold=f"metric = number of breaks failing the rule; FAIL if any: trend-adjusted break with block-permutation p < {alpha23:.4f} (0.05 Bonferroni over {len(bp)} series), location "
                       "identified (raw and trend-adjusted fits within 6 months), >= 3 months inside a Congress",
             detail=f"breaks: {brk_txt}. Significant constant-mean breaks: {len(raw_sig)}; significant beyond a linear trend: {len(sig_all)}, "
                    f"of which location-identified: {len(sig)}. Breaks that do not survive the trend adjustment (a steady trend, not a "
                    f"break): {lost_after_dt or 'none'}. Significant but location not identified (the two fits disagree by > 6 months, "
                    f"i.e. the series bends rather than steps; reported, not used): "
                    f"{[(r.series, r.chamber, r.raw_break, r.break_month) for r in unid.itertuples()] or 'none'}. "
                    + (f"Significant inside-Congress breaks: {[(r.series, r.chamber, r.break_month) for r in sig_in.itertuples()]}: that "
                       "Congress mixes two regimes, so a validation unit on it straddles the break and a within-Congress cut (or a "
                       "post-break indicator) matters for it. " if len(sig_in) else "No inside-Congress break survives detrending: Congress "
                       "boundaries are adequate cut points. ")
                    + f"Reconciliation with A16 (A4 original share 117/118 = break vs linear trend at Congress level): identified breaks at a "
                      f"Congress boundary (month 1-3): {[(r.series, r.chamber, r.break_month, int(r.congress)) for r in sig[~sig.inside].itertuples()] or 'none'}"
                      + (" -> agrees with A16 (a level shift at the start of that Congress, not a steady trend). " if len(sig[~sig.inside]) else
                         " -> no trend-adjusted boundary break, which does not support A16's break reading. ")
                    + "A constant-mean CUSUM with block permutation would call any trend a break, which is why the verdict uses the "
                      "trend-adjusted test. Cause (procedural rules vs "
                      "behaviour) is not identifiable from the data; the original share itself is owned by A (A4).")


def _first_post_break_congresses(bdates):
    """Congresses that are the first (whole or partial) Congress after each break date."""
    return sorted({int(next(c for c in CONGRESSES if congress_end(c) > d)) for d in bdates})


pbu = _post_break_units(list(sig.break_date)) if len(sig) else None
if pbu is not None:
    first_pb = _first_post_break_congresses(list(sig.break_date))
    # Chance rate: share of one-step Congress transitions in 108-118 that land on a first post-break Congress, i.e. the
    # probability that a deployment Congress (fit on everything before it) starts a regime its fit set never saw.
    p_dep = len(first_pb) / (len(CONGRESSES) - 1)
    rows23 = []
    for k, g in pbu[pbu.role == "test"].groupby("split"):
        n_t, n_u = len(g), int(g.unseen.sum())
        p_b = float(stats.binom.sf(n_u - 1, n_t, p_dep)) if n_u else 1.0
        rows23.append(dict(split=k, test_units=n_t, post_break_unseen=n_u, share=n_u / n_t, expected=p_dep, binom_p=p_b))
    t23 = pd.DataFrame(rows23).set_index("split")
    print(f"Test units scored on a post-break regime unseen in their fit set vs the deployment rate {p_dep:.2f} "
          f"(first post-break Congresses {first_pb} out of {len(CONGRESSES) - 1} transitions)")
    print(t23.rename_axis(None).round(3).to_string())
    bs23 = with_variants({k: bool(r.binom_p >= 0.05) for k, r in t23.iterrows()} | {"S5": None})
    aff = pbu[pbu.unseen].groupby(["split", "role"]).fold.agg(list)
    val_aff = sorted({k for (k, r) in aff.index if r == "val"})
else:
    t23, bs23, aff, val_aff, p_dep, first_pb = None, None, pd.Series(dtype=object), [], 0.0, []

# Sensitivity of C23b to the definition of "break". C23 uses a narrow one (a within-series level shift in the monthly
# original-share and lag series). Section A's A16 labels a failing candidate part a break when it is off both the
# stationary and the linear-trend interval; under that definition a candidate Congress is post-break if it has any
# Holm-surviving break part. A16 judges only candidates (116-118), so two expected rates bracket the deployment rate:
# reference Congresses assumed unbroken (n_broken / 10 transitions) and candidate transitions only (n_broken / 3).
_az = globals().get("A_ZTAB")
if _az:
    _zt = pd.DataFrame(_az).drop_duplicates(["part", "congress"], keep="last")
    _need = {"level", "fail", "kind", "holm"}
    a16_broken = (sorted({int(c) for c in _zt[_zt.level & _zt.fail & (_zt.kind == "break") & _zt.holm].congress})
                  if _need <= set(_zt.columns) else None)
    a16_src = "section A (A_ZTAB: Holm-surviving parts labelled 'break', as in A16)"
else:
    a16_broken, a16_src = None, None
if a16_broken is None:
    # Section A not in scope (section run alone): use the upper bound, every candidate Congress post-break.
    a16_broken, a16_src = list(C_EVAL), "upper-bound assumption (section A not in scope): every candidate Congress post-break"
rate_lo, rate_c = len(a16_broken) / (len(CONGRESSES) - 1), len(a16_broken) / len(C_EVAL)
pbx = pd.DataFrame([dict(split=un["split"], role=un["role"], post=int(un["eval_congress"]) in a16_broken)
                    for un in EVAL_UNITS + S1POOL_UNITS if un["split"] != "S5" and un["role"] == "test"])
sens = pbx.groupby("split").post.agg(test_units="size", post_break="sum")
sens["share"] = sens.post_break / sens.test_units
for nm, r_ in (("p_rate_lo", rate_lo), ("p_rate_cand", rate_c)):
    sens[nm] = [float(stats.binom.sf(k - 1, n, r_)) if k else 1.0 for n, k in zip(sens.test_units, sens.post_break)]
sens["c23b_narrow"] = [None if t23 is None or k not in t23.index else bool(t23.loc[k, "binom_p"] >= 0.05) for k in sens.index]
sens["a16_lo"], sens["a16_cand"] = sens.p_rate_lo >= 0.05, sens.p_rate_cand >= 0.05
flip = [k for k, r in sens.iterrows() if r.c23b_narrow is not None and r.c23b_narrow != r.a16_cand]
print(f"C23b sensitivity: A16-style break list {a16_broken} ({a16_src}); expected rate {rate_lo:.2f} (reference Congresses "
      f"unbroken) or {rate_c:.2f} (candidate transitions only)")
print(sens.rename_axis(None).round(3).to_string())
record_check(C_SECTION, "C23b", "Share of test units scored on a post-break regime unseen in training is not above the deployment rate",
             None if bs23 is None else all(v for k, v in bs23.items() if v is not None), metric=None if t23 is None else float(t23.share.max()),
             threshold=f"by_split False if binomial P(>= observed post-break test units | n test units, rate {p_dep:.2f}) < 0.05",
             detail=f"significant detrended breaks (p < {alpha23:.4f}): {[(r.series, r.chamber, str(r.break_date.date())) for r in sig.itertuples()]}; "
                    f"first post-break Congresses {first_pb}. Affected units (split, role) -> folds: {aff.to_dict()}. "
                    + ("" if t23 is None else f"Per split (affected/test units, p): "
                       + ", ".join(f"{k} {int(r.post_break_unseen)}/{int(r.test_units)} (p={r.binom_p:.2f})" for k, r in t23.iterrows()) + ". ")
                    + f"Splits whose val unit is affected: {val_aff}. A test unit on the first post-break Congress is the realistic "
                      "deployment case (deploying on 119 faces the same risk if 119 starts a new regime), not a defect: it is only a "
                      "problem if a split has MORE such units than deployment would (judged here), or if all of a split's test evidence "
                      "comes from such units (S1a with one test Congress: read its score as a pessimistic, regime-change estimate). "
                    + f"Definition: 'break' here means a within-series level shift in the monthly original-share and cosponsor-lag series "
                      f"only (C23), so the deployment rate {p_dep:.2f} counts one post-break Congress in {len(CONGRESSES) - 1} transitions. "
                      f"Sensitivity with A16's broader definition (any Holm-surviving failing level part off the trend interval): "
                      f"post-break Congresses {a16_broken} ({a16_src}); expected rate {rate_lo:.2f} (reference Congresses unbroken) to "
                      f"{rate_c:.2f} (candidate transitions only). Per split, post-break test units / test units and p at the two rates: "
                    + ", ".join(f"{k} {int(r.post_break)}/{int(r.test_units)} (p={r.p_rate_lo:.2f} / {r.p_rate_cand:.2f})" for k, r in sens.iterrows())
                    + f". Splits whose verdict flips between the narrow definition and A16's at the candidate rate: {flip or 'none'}"
                    + (" -> their C23b failure depends on the narrow break definition; read it as weak evidence. " if flip else ". ")
                    + "Under A16's definition nearly every candidate test unit is post-break, which is the deployment case, not a split "
                      "defect: no split can avoid it, so the remedy is recalibration on val and recency weighting, not a different split.",
             by_split=bs23, na_reason=("S5 has no time boundary between fit and eval bills; no regime can be unseen" if bs23 else None))

# %% [markdown]
# ### C5-C6. Label censoring at the end of each Congress
#
# **Why it matters.** Unpassed bills die when a Congress adjourns, so a bill introduced 20 days before the
# end has no full 30-day window (task 1) and one introduced 100 days before the end has no full 180-day
# window (task 2). These bills are excluded. If the excluded share differs by Congress, or if the excluded
# bills are systematically different (the end-of-Congress rush), the evaluation population differs from
# the training population. Note that 119 is censored *much* more because it is in progress.
#
# **How to read it.** The chart shows the censored share per Congress for task 1 (t0=0, t0=7) and task 2.
# The table adds the share of evaluation bills each candidate split loses (from `iter_eval_units()`); S4's
# test slice is the end of the Congress, so it is hit hardest.

# %%
# C5/C6: censored share per Congress for task 1 (t0=0/7, 30 d) and task 2 (180 d); bills lost per split part.
cen = pd.DataFrame({
    "t1_t0=0": wt[0].groupby("congress")["censored"].mean(),
    "t1_t0=7": wt[7].groupby("congress")["censored"].mean(),
    "t2_180d": adv.groupby("congress")["censored"].mean(),
    "t2_bills_lost": adv.groupby("congress")["censored"].sum(),
}).loc[CONGRESSES + [IN_PROGRESS]]
print("Censored share per Congress (119 in progress, context only)")
print(cen.T.to_string(float_format=C_FMT))
cl = cen.loc[CONGRESSES, ["t1_t0=0", "t1_t0=7", "t2_180d"]].reset_index().melt("congress", var_name="task", value_name="share")
line_chart(cl, "congress", "share", series="task", yfmt=".0%",
           title="Share of law-capable bills whose label window runs past the end of the Congress",
           subtitle="Task 1: 30-day window after t0; task 2: 180-day reporting window", xlabel="Congress",
           name="C_censoring")
# Censoring per evaluation unit of every candidate split: share of eval bills lost (task 1 t0=7, task 2).
cfr = lb[lb.congress.isin(CONGRESSES)][["bill_id", "congress", "introduced_date"]].merge(
    wt[7][["bill_id", "censored"]].rename(columns={"censored": "cen_t1"}), on="bill_id").merge(
    adv[["bill_id", "censored"]].rename(columns={"censored": "cen_t2"}), on="bill_id")
lost_rows = []
for un in EVAL_UNITS + S1POOL_UNITS:
    _, ev = unit_masks(un, cfr)
    e = cfr[ev]
    lost_rows.append(dict(split=un["split"], fold=un["fold"], role=un["role"], eval_bills=len(e),
                          t1_lost=e.cen_t1.mean(), t2_lost=e.cen_t2.mean()))
lost_u = pd.DataFrame(lost_rows)
lost = lost_u.groupby(["split", "role"]).agg(units=("fold", "size"), eval_bills=("eval_bills", "mean"),
                                             t1_lost=("t1_lost", "mean"), t2_lost=("t2_lost", "max")).round(3)
print("\nShare of eval bills lost to censoring per split and role (mean over folds; t2_lost = worst fold)")
print(lost.unstack("role").to_string())
lost_split = lost_u.groupby("split")[["t1_lost", "t2_lost"]].max()   # worst eval unit per split
mx1 = float(cen.loc[CONGRESSES, "t1_t0=7"].max())
gap1 = float(cen.loc[C_EVAL, "t1_t0=7"].mean() - cen.loc[C_TRAIN, "t1_t0=7"].mean())
record_check(C_SECTION, "C5", "Task-1 censoring (t0=7) drops < 5% of law-capable bills in every Congress 108-118",
             bool(mx1 < 0.05), metric=mx1, threshold="max censored share < 0.05",
             detail=f"censored share t0=7 ranges {cen.loc[CONGRESSES, 't1_t0=7'].min():.3f}-{mx1:.3f}; eval Congresses lose "
                    f"{gap1:+.3f} more than the train pool on average (end-of-Congress introductions have grown). 5% is the "
                    "level below which excluding bills cannot materially change the evaluation population; censoring is "
                    "structural (last ~37 days): by-Congress splits lose the same calendar slice, while the S4 test slice "
                    f"(last 20% of introductions) loses {lost_u.loc[(lost_u.split == 'S4') & (lost_u.role == 'test'), 't1_lost'].mean():.3f} "
                    f"of its bills (see table). by_split = worst eval unit's task-1 lost share < 0.05. Why 5% is a hard limit: losing a share f "
                    f"widens a metric's CI half-width by 1/sqrt(1-f)-1 (at f=0.05: {1 / np.sqrt(0.95) - 1:.1%}; worst by-Congress unit f="
                    f"{lost_split.loc[[k for k in BY_CONGRESS_SPLITS if k in lost_split.index], 't1_lost'].max():.3f}: "
                    f"{1 / np.sqrt(1 - lost_split.loc[[k for k in BY_CONGRESS_SPLITS if k in lost_split.index], 't1_lost'].max()) - 1:.1%}), "
                    "negligible; the larger risk is selection, because the dropped bills are all end-of-Congress introductions, and "
                    "below 5% they cannot move a population average by more than 5% of the gap between them and the rest.",
             quantity="censoring", by_split=with_variants({k: bool(v < 0.05) for k, v in lost_split.t1_lost.items()}))
# Are the task-2 censored bills different? Compare cosponsor count by day 30 of censored vs uncensored bills.
a2 = adv[adv.congress.isin(CONGRESSES)].merge(wt[0][["bill_id", "n_pos", "n_seen"]], on="bill_id", how="left")
a2["early_cosp"] = a2["n_pos"].fillna(0) + a2["n_seen"].fillna(0)
r2 = cen.loc[CONGRESSES, "t2_180d"].max() - cen.loc[CONGRESSES, "t2_180d"].min()
s2 = smd(a2.loc[~a2.censored, "early_cosp"], a2.loc[a2.censored, "early_cosp"])
s4_t2 = float(lost_u.loc[(lost_u.split == "S4") & (lost_u.role == "test"), "t2_lost"].mean())
oth_t2 = float(lost_u.loc[lost_u.split.isin(BY_CONGRESS_SPLITS), "t2_lost"].mean())
c6_tab, c6_info = rel_to_train(cen.loc[CONGRESSES, "t2_180d"], side="two")
record_check(C_SECTION, "C6", "Task-2 censored share (180 d) of eval Congresses within train-to-train noise",
             not bool(c6_tab.loc[c6_tab.role == "cand", "fail"].any()), metric=float(cen.loc[C_EVAL, "t2_180d"].max()),
             threshold=c6_info["rule"], **dict(zip(("by_split", "na_reason"), full_split(
                 rel_by_split(c6_tab), s4_na="S4's test slice is the end of each Congress, so its censored share is set by position, "
                                             "not by the Congress; judged separately in C6b"))),
             detail=rel_detail(c6_tab, c6_info) + f". Range 108-118 {r2:.3f}. SMD of cosponsors-by-day-30 (censored vs uncensored) = "
                    f"{s2:.3f}: the dropped end-of-Congress bills differ, so task-2 results describe bills introduced in the first ~18 "
                    f"months only. By-Congress eval sets lose {oth_t2:.3f} on average, but the S4 test slice (last 20% of introductions) "
                    f"loses {s4_t2:.3f} (C6b). Task-2 t0 assumed: window starts at introduction (any t0 <= 30 d leaves this unchanged, "
                    "see C25). If FAIL: the eval Congress loses more bills than train did; evaluate on the same uncensored slice in every split.")
record_check(C_SECTION, "C6b", "Within-Congress split S4: its test slice (last 20% of introductions) keeps >= 50% of bills for task 2",
             bool(s4_t2 < 0.5), metric=s4_t2, threshold="task-2 censored share of S4 test < 0.50",
             detail=f"S4 test slices lose {s4_t2:.3f} of their bills to task-2 censoring (task 1 t0=7: "
                    f"{lost_u.loc[(lost_u.split == 'S4') & (lost_u.role == 'test'), 't1_lost'].mean():.3f}); by-Congress eval sets lose "
                    f"{oth_t2:.3f}. The surviving S4 task-2 test bills are the few introduced early in the slice, so S4 cannot "
                    "be used to select or report a task-2 model. by_split = worst eval unit keeps >= 50% of its bills. Why 50%: "
                    f"beyond it the survivors are a minority drawn from the start of the slice (not the slice's population), and the CI "
                    f"half-width grows by 1/sqrt(1-f)-1 = {1 / np.sqrt(max(1 - s4_t2, 1e-9)) - 1:.0%} at the observed loss "
                    f"({lost_u.loc[(lost_u.split == 'S4') & (lost_u.role == 'test'), 'eval_bills'].mean() * (1 - s4_t2):,.0f} task-2 test bills left per fold).",
             by_split=with_variants({k: bool(v < 0.5) for k, v in lost_split.t2_lost.items()}))

# %% [markdown]
# ### C7-C10. Party-control regimes and party labels
#
# **Why it matters.** Cosponsorship and committee action depend on who holds the gavel. A model trained only
# on Republican-House Congresses may not transfer to a Democratic House. A by-Congress split works best if
# each chamber's controlling party in the validation and test Congresses also appears in training (and,
# only if it measurably matters, the joint House x Senate regime). If not, the test measures extrapolation
# to a new regime, and we should at least report per-regime results.
#
# **How to read it.** The strip shows the majority party per chamber per Congress from the shared
# `majority_party()` (point-in-time at mid-Congress, party as of the Congress, independents counted with their
# caucus, Senate ties broken by the Vice President's party and cross-checked against the data). Diamonds mark
# a control flip versus the previous Congress. C8 checks, for every evaluation unit of the candidate splits, that
# each chamber's controlling party appears in training (task 2 depends on the bill's own chamber). C8b asks whether
# the *joint* regime (divided vs unified government) changes anything measurable beyond the chamber majority; only
# then does joint-regime coverage become a requirement. C9 shows how narrow the margins are and how C8 would change
# if a tie went the other way;
# C10 checks that per-Congress party labels are used (a time leak otherwise).

# %%
# C7: majority per chamber per Congress from majority_party(); unified vs split; control flips.
mp = majority_party()
mp = mp[mp.congress.isin(CONGRESSES + [IN_PROGRESS])]
reg = mp.pivot(index="congress", columns="chamber", values="majority").rename(
    columns={"House of Representatives": "house", "Senate": "senate"})
reg["regime"] = reg["house"] + "H/" + reg["senate"] + "S"
reg["unified_congress"] = reg["house"] == reg["senate"]
reg["house_flip"] = reg["house"] != reg["house"].shift()
reg["senate_flip"] = reg["senate"] != reg["senate"].shift()
reg.loc[reg.index.min(), ["house_flip", "senate_flip"]] = False
print("Majority party per chamber (from majority_party(); point-in-time, caucus-adjusted)")
print(reg.to_string())
flips = [f"{c} {ch}" for c in reg.index for ch in ("house", "senate") if reg.loc[c, f"{ch}_flip"]]
rd = mp.copy()
rd["chamber"] = rd["chamber"].map({"House of Representatives": "House", "Senate": "Senate"})
rd["flip"] = [bool(reg.loc[c, ("house_flip" if ch == "House" else "senate_flip")]) for c, ch in zip(rd.congress, rd.chamber)]
rd["congress"] = rd["congress"].astype(int)
show_d3(r"""
const {width, height} = opts;
const m = {top: 48, right: 20, bottom: 50, left: 70};
const svg = d3.select(el).append('svg').attr('width', width).attr('height', height);
viz.title(svg, 'Majority party per chamber and Congress (diamond = control flip vs previous Congress)',
          'From majority_party(): mid-Congress, caucus-adjusted, VP breaks Senate ties; 119 in progress');
const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
const W = width - m.left - m.right, H = height - m.top - m.bottom;
const xs = d3.scaleBand(Array.from(new Set(data.map(d => String(d.congress)))), [0, W]).padding(0.08);
const ys = d3.scaleBand(['House', 'Senate'], [0, H]).padding(0.12);
const col = {D: '#2a78d6', R: '#e34948'};
const tip = viz.tooltip(el);
g.selectAll(null).data(data).join('rect').attr('x', d => xs(String(d.congress))).attr('y', d => ys(d.chamber))
  .attr('width', xs.bandwidth()).attr('height', ys.bandwidth()).attr('rx', 3).attr('fill', d => col[d.majority] || '#7a7a7a')
  .on('mousemove', (e, d) => tip.show(`<b>${d.congress} ${d.chamber}</b><br>majority ${d.majority}${d.flip ? ' (flip)' : ''}`, e))
  .on('mouseleave', () => tip.hide());
g.selectAll(null).data(data).join('text').attr('x', d => xs(String(d.congress)) + xs.bandwidth() / 2)
  .attr('y', d => ys(d.chamber) + ys.bandwidth() / 2 + 4).attr('text-anchor', 'middle').attr('fill', '#fff')
  .style('font', '600 12px sans-serif').text(d => d.majority);
g.selectAll(null).data(data.filter(d => d.flip)).join('path')
  .attr('transform', d => `translate(${xs(String(d.congress)) + 8},${ys(d.chamber) + 8})`)
  .attr('d', d3.symbol(d3.symbolDiamond, 40)).attr('fill', viz.ink);
g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xs)).call(viz.axis);
g.append('g').call(d3.axisLeft(ys)).call(viz.axis);
viz.legend(svg, [['Democratic majority', col.D], ['Republican majority', col.R]], m.left, height - 10);
""", rd, height=230, width=720, name="C_regime_strip")
record_check(C_SECTION, "C7", "Party-control regimes and flips across 108-118 (informational)", None,
             metric=f"{reg.loc[CONGRESSES, 'regime'].nunique()} regimes, {len([f for f in flips if int(f.split()[0]) in CONGRESSES])} flips",
             detail=f"regimes: {reg.loc[CONGRESSES, 'regime'].to_dict()}; flips at: {flips}. Unified Congresses: "
                    f"{list(reg.index[reg.unified_congress & reg.index.isin(CONGRESSES)])}.")

# %%
# C8: for every evaluation unit (incl. S1pool), does each chamber's controlling party (and the joint regime) appear in training?
cov = []
for un in [u for u in EVAL_UNITS if u["split"] in BY_CONGRESS_SPLITS] + S1POOL_UNITS:
    c, pr = un["eval_congress"], reg.loc[un["train"]]
    cov.append(dict(split=un["split"], fold=un["fold"], role=un["role"], eval_congress=c, train=un["train_label"],
                    eval_regime=reg.loc[c, "regime"], regime_in_train=int((pr.regime == reg.loc[c, "regime"]).sum()),
                    house_in_train=int((pr.house == reg.loc[c, "house"]).sum()),
                    senate_in_train=int((pr.senate == reg.loc[c, "senate"]).sum())))
cov = pd.DataFrame(cov)
cov["chambers_covered"] = (cov.house_in_train > 0) & (cov.senate_in_train > 0)
covs = cov.groupby("split", sort=False).agg(units=("fold", "size"), chambers_covered=("chambers_covered", "mean"),
                                            house_covered=("house_in_train", lambda v: (v > 0).mean()),
                                            senate_covered=("senate_in_train", lambda v: (v > 0).mean()),
                                            joint_regime_covered=("regime_in_train", lambda v: (v > 0).mean()))
print("Share of evaluation units whose chamber majorities (primary) / joint House x Senate regime (INFO) appear in training")
print(covs.round(3).to_string())
ch_split = with_variants(covs.chambers_covered.eq(1.0).to_dict())
uncov = sorted({(r.split, int(r.eval_congress)) for r in cov.itertuples() if not r.chambers_covered})
record_check(C_SECTION, "C8", "Each eval Congress's House majority and Senate majority both appear in its training Congresses",
             all(v for k, v in ch_split.items() if k in ("S1a", "S1b", "S3")), metric=float(covs.loc[["S1a", "S1b", "S3"], "chambers_covered"].min()),
             threshold="per-chamber coverage = 1.0 for every eval unit (judged on S1a, S1b, S3; by_split for all)",
             detail=f"per-chamber coverage per split {covs.chambers_covered.round(2).to_dict()}; uncovered units {uncov}. Task 2 depends on "
                    "the bill's own chamber and A5b shows chamber x controlling party is enough to match the majority-sponsor OR, "
                    "so per-chamber coverage is the requirement. Sliding windows (S2) drop chamber regimes that expanding windows keep. "
                    "S4 and S5 hold by construction: their fit sets contain bills of the eval Congress itself.",
             quantity="party_regime", **dict(zip(("by_split", "na_reason"), full_split(
                 ch_split, extra={"S4": True, "S5": True}))))

# C8b: does the JOINT regime (divided vs unified government) matter beyond the bill chamber's majority?
# Per (Congress, chamber), uncensored task-2 bills: reported rate of majority-sponsored bills and majority-sponsor log OR.
sp_cau = members.drop_duplicates(["bioguide_id", "congress"]).set_index(["bioguide_id", "congress"])["caucus"]
a8 = adv[adv.congress.isin(CONGRESSES) & ~adv.censored].merge(bills[["bill_id", "sponsor_bioguide_id"]], on="bill_id")
a8["cau"] = pd.MultiIndex.from_arrays([a8.sponsor_bioguide_id, a8.congress]).map(sp_cau)
a8["cau"] = a8.cau.where(a8.cau.isin(["D", "R"]), a8.sponsor_party)
a8 = a8.merge(MAJORITY[["congress", "chamber", "majority"]], on=["congress", "chamber"])
a8["maj"] = a8.cau == a8.majority
g8 = a8.groupby(["congress", "chamber", "maj"])["label"].agg(["sum", "size"]).unstack("maj")
lo_ = lambda k, n: np.log((k + 0.5) / (n - k + 0.5))
q8 = pd.DataFrame({"rate_maj": g8[("sum", True)] / g8[("size", True)],
                   "log_or": lo_(g8[("sum", True)], g8[("size", True)]) - lo_(g8[("sum", False)], g8[("size", False)])}).reset_index()
q8["divided"] = q8.congress.map(~reg["unified_congress"])
perm_rows = []
for ch, g in q8.groupby("chamber"):
    for col in ("rate_maj", "log_or"):
        for label, cs_ in (("108-118", CONGRESSES), ("108-115", C_TRAIN)):
            gg = g[g.congress.isin(cs_)]
            x, dv = gg[col].to_numpy(), gg.divided.to_numpy()
            obs = x[dv].mean() - x[~dv].mean()
            combos = np.array(list(__import__("itertools").combinations(range(len(x)), int(dv.sum()))))
            msk = np.zeros((len(combos), len(x)), bool)
            msk[np.arange(len(combos))[:, None], combos] = True
            null = (msk * x).sum(1) / dv.sum() - (~msk * x).sum(1) / (~dv).sum()
            perm_rows.append(dict(chamber=ch[:5], quantity=col, congresses=label, n_divided=int(dv.sum()), diff=obs,
                                  p_exact=float((np.abs(null) >= abs(obs) - 1e-12).mean()), n_perm=len(combos)))
perm = pd.DataFrame(perm_rows)
print("Divided minus unified government, per chamber (exact permutation over Congress labels; 108-115 has only 2 divided)")
print(perm.round(3).to_string(index=False))
p_main = perm[perm.congresses == "108-118"]
alpha8 = 0.05 / len(p_main)
joint_matters = bool((p_main.p_exact < alpha8).any())
jr_split = with_variants(covs.joint_regime_covered.eq(1.0).to_dict())
record_check(C_SECTION, "C8b", "Joint House x Senate regime (divided vs unified) changes task-2 rate or majority OR beyond the chamber majority",
             False if joint_matters else None, metric=float(p_main.p_exact.min()),
             threshold=f"FAIL only if any exact permutation p < {alpha8:.4f} (0.05 Bonferroni over {len(p_main)} tests, 108-118); else INFO",
             detail=f"divided-unified differences (108-118): " + "; ".join(f"{r.chamber} {r.quantity} {r.diff:+.3f} (p={r.p_exact:.3f})" for r in p_main.itertuples())
                    + f". Joint-regime coverage per split (INFO unless the effect is significant): {covs.joint_regime_covered.round(2).to_dict()}; "
                    f"116 ({reg.loc[116, 'regime']}) never occurs in 108-115. Using 108-118 to estimate a process property, not to pick a model; "
                    "with 11 Congresses the test has low power, so a non-significant result means 'no evidence', not 'no effect'.",
             quantity="party_regime_joint", by_split=jr_split if joint_matters else None)

# Val regime = test regime per fold (does tuning transfer?), reported for context.
vt = cov.pivot_table(index=["split", "fold"], columns="role", values="eval_regime", aggfunc="first")
vt_same = (vt["val"] == vt["test"]).groupby("split").mean()
never = [c for c in CONGRESSES if (reg.loc[CONGRESSES, "regime"] == reg.loc[c, "regime"]).sum() == 1]
print(f"val regime = test regime share of folds: {vt_same.round(2).to_dict()}; regimes occurring once in 108-118: "
      f"{[(c, reg.loc[c, 'regime']) for c in never]}")

# %%
# C9: how robust are the regime labels? Seat margins per chamber and ties (from MAJORITY); C8 sensitivity to ties.
mg = MAJORITY[MAJORITY.congress.isin(CONGRESSES)].copy()
mg["margin"] = (mg.seats_D - mg.seats_R).abs()
mg["margin_share"] = mg.margin / mg.n_voting
print("Seat margin (|D - R| caucus-adjusted, voting members at mid-Congress) per chamber")
print(mg.assign(chamber=mg.chamber.str[:5]).pivot(index="chamber", columns="congress", values="margin").to_string())
ties = mg[mg.tie_broken_by != "none (seat count)"]
alt_cov = []
for r in ties.itertuples():
    key = "house" if r.chamber.startswith("House") else "senate"
    other = "R" if r.majority == "D" else "D"
    rr = reg.loc[r.congress].copy()
    rr[key] = other
    regime = rr["house"] + "H/" + rr["senate"] + "S"
    n_alt = int((reg.loc[C_TRAIN, "regime"] == regime).sum())
    alt_cov.append(f"{r.congress} {key}: label {r.majority} -> regime {reg.loc[r.congress, 'regime']} "
                   f"({int((reg.loc[C_TRAIN, 'regime'] == reg.loc[r.congress, 'regime']).sum())}x in 108-115); "
                   f"if {other}: {regime} ({n_alt}x in 108-115)")
print("ties: " + ("; ".join(f"{r.congress} {r.chamber}: {r.seats_D}-{r.seats_R}, {r.tie_broken_by}" for r in ties.itertuples()) or "none"))
print("C8 sensitivity: " + ("; ".join(alt_cov) or "no ties"))
confirmed = all("agrees" in t for t in ties.tie_broken_by)
narrow = mg[mg.margin_share < 0.03]
record_check(C_SECTION, "C9", "Regime labels are robust: every tie-broken chamber is confirmed by the data (committee-reported sponsors)",
             bool(confirmed), metric=int(len(ties)), threshold="all ties confirmed by data",
             detail=f"ties: {[(int(r.congress), r.chamber[:5], int(r.seats_D), int(r.seats_R), r.majority) for r in ties.itertuples()]}; "
                    f"tie-break evidence: {list(ties.tie_broken_by)}. Sensitivity: {alt_cov}. Chambers with margin < 3% of seats: "
                    f"{[(int(r.congress), r.chamber[:5], int(r.margin)) for r in narrow.itertuples()]}: narrow majorities make a "
                    "regime label a weak proxy for agenda control, so treat regime as a stratification variable for "
                    "reporting, not as a hard requirement.")

# %%
# Member roster per Congress (used by C10).
mt = members[members.congress.isin(CONGRESSES + [IN_PROGRESS])].dropna(subset=["chamber"]).copy()

# C10: per-Congress party labels. Does setup's members.party_code_asof match the party recorded on bills that Congress?
mt_c = mt[mt.congress.isin(CONGRESSES)]
mism = mt_c[mt_c.party_code != mt_c.party_code_asof]
print(f"members.party_code (latest) differs from party_code_asof in {len(mism)} member-Congress rows "
      f"({mism.bioguide_id.nunique()} members), e.g.:")
print("; ".join(f"{r.name.split(',')[0]} {r.cmin}-{r.cmax} latest {r.latest} / as-of {r.asof}" for r in
                mism.groupby("name").agg(cmin=("congress", "min"), cmax=("congress", "max"), latest=("party_code", "first"),
                                         asof=("party_code_asof", "first")).reset_index().itertuples()) or "none")
cp = cosp[cosp.congress.isin(CONGRESSES)][["congress", "bioguide_id", "party"]].drop_duplicates()
cp = cp.merge(mt_c[["congress", "bioguide_id", "party_code_asof", "party_code"]], on=["congress", "bioguide_id"], how="inner")
agree_asof, agree_latest = float((cp.party == cp.party_code_asof).mean()), float((cp.party == cp.party_code).mean())
print(f"bill_cosponsors.party agrees with party_code_asof for {agree_asof:.2%} of member-Congress pairs, "
      f"with the latest party_code for {agree_latest:.2%}")
record_check(C_SECTION, "C10", "Per-Congress party labels (members.party_code_asof) match the party recorded on that Congress's bills",
             bool(agree_asof >= 0.995 and agree_asof >= agree_latest), metric=agree_asof,
             threshold=">= 0.995 agreement and >= latest-party agreement",
             detail=f"party_code_asof agrees {agree_asof:.2%} vs latest party_code {agree_latest:.2%}; {len(mism)} member-Congress rows "
                    f"({mism.bioguide_id.nunique()} members, e.g. {', '.join(mism.name.drop_duplicates().head(4))}) would get a "
                    "party adopted later if members.party_code were used. Requirement for every split: node party features "
                    "= party_code_asof (or caucus), never party_code; otherwise future party switches leak into past Congresses.")

# %% [markdown]
# ### C11-C14. Snapshot leakage: which fields are known at t0?
#
# **Why it matters.** The parquet tables are a **snapshot taken years after each Congress ended**. Many
# bill-level fields summarize the bill's whole life (latest action, became law, number of actions,
# cosponsors, text versions, summaries). Using any of them as a feature at t0 leaks the future, and a
# random split (S5) or even a temporal split would look far better than reality. Fields with a timestamp
# can be rebuilt "as of t0"; fields without one (policy area, subjects) are only safe if we can show they
# are normally assigned before t0.
#
# **How to read it.** C11 is a requirement, not a pass/fail test. The leakage table compares each snapshot field with its value as of t0 = 7 days
# (where a timestamped table lets us rebuild it): `share_changed` is the share of law-capable bills whose
# snapshot value differs from the t0 value. Then: when are committee referrals and CRS summaries recorded
# relative to introduction, and when was each bill last updated relative to the end of its Congress?

# %%
# C11: snapshot fields vs their value at t0=7 (rebuilt from timestamped tables), law-capable bills 108-118.
T0 = 7
b = lb[lb.congress.isin(CONGRESSES)][["bill_id", "congress", "introduced_date", "c_end", "n_cosponsors", "n_actions",
                                       "n_text_versions", "n_summaries", "latest_action_date", "law_type", "update_date"]].copy()
cut = b.set_index("bill_id")["introduced_date"] + pd.Timedelta(days=T0)
acts = load("bill_actions", ("bill_id", "action_date"))
acts = acts[acts.bill_id.isin(cut.index)]
n_act_t0 = acts[acts.action_date <= acts.bill_id.map(cut)].groupby("bill_id").size()
tv = load("bill_text_versions", ("bill_id", "version_ts"))
tv = tv[tv.bill_id.isin(cut.index)]
n_tv_t0 = tv[tv.version_ts.dt.normalize() <= tv.bill_id.map(cut)].groupby("bill_id").size()
n_cosp_t0 = wt[7].set_index("bill_id")["n_seen"]
bi = b.set_index("bill_id")
leak = pd.DataFrame([
    ("n_cosponsors", "count incl. later cosponsors", (bi.n_cosponsors != n_cosp_t0.reindex(bi.index).fillna(0)).mean()),
    ("n_actions", "all actions to date", (bi.n_actions != n_act_t0.reindex(bi.index).fillna(0)).mean()),
    ("n_text_versions", "reported/engrossed/enrolled texts", (bi.n_text_versions != n_tv_t0.reindex(bi.index).fillna(0)).mean()),
    ("latest_action_*", "last action of the bill's life", (bi.latest_action_date > cut).mean()),
    ("law_type/law_number", "became law (outcome)", bi.law_type.notna().mean()),
    ("n_summaries", "CRS summaries (no publish date)", (bi.n_summaries > 1).mean()),
    ("update_date", "record last modified", (bi.update_date > bi.c_end).mean()),
], columns=["field", "encodes", "share_changed_after_t0"])
print(f"Snapshot fields vs value as of t0={T0} days (law-capable bills 108-118)")
print(leak.round(3).to_string(index=False))
print("For law_type: share of bills that became law. For n_summaries: share with > 1 summary (later versions). "
      "For update_date: share modified after the Congress ended.")
ncs = float(leak.set_index("field").loc["n_cosponsors", "share_changed_after_t0"])
record_check(C_SECTION, "C11", "Requirement: rebuild bill-level count/outcome fields as of t0 (snapshot values are post-t0)", None,
             metric=ncs, threshold="INFO (requirement for every split)",
             detail=f"headline: the snapshot n_cosponsors differs from the t0={T0} count for {ncs:.1%} of bills. Share differing per field: "
                    + ", ".join(f"{r.field} ({r.share_changed_after_t0:.1%})" for r in leak.itertuples() if r.field != "update_date")
                    + ". Rebuild as of t0 from timestamped tables: n_cosponsors (bill_cosponsors.sponsorship_date), n_actions / "
                      "latest_action_* (bill_actions.action_date), n_text_versions (bill_text_versions.version_ts). Never use: "
                      "law_type/law_number, n_summaries, update_date (record metadata, not a feature). Applies to every split; "
                      "S5 (random) additionally leaks through same-Congress neighbours (C20).",
             quantity="leakage_snapshot_fields", requirement=True)

# %%
# C12: committee referral timing relative to introduction (bill_committees.activity_ts).
bc = load("bill_committees", ("bill_id", "activity", "activity_ts", "congress"))
ref = bc[bc.activity.str.lower().str.startswith("referred to", na=False) & bc.bill_id.isin(lb.bill_id)]
# activity_ts is UTC; convert to US Eastern so an evening referral is not counted as the next day
first_ref = (ref.groupby("bill_id")["activity_ts"].min().dt.tz_localize("UTC")
             .dt.tz_convert("America/New_York").dt.tz_localize(None))
r = lb[lb.congress.isin(CONGRESSES)][["bill_id", "congress", "introduced_date"]].copy()
r["ref_lag"] = (r.bill_id.map(first_ref).dt.normalize() - r.introduced_date).dt.days
reft = r.groupby("congress")["ref_lag"].agg(
    has_referral=lambda s: s.notna().mean(), by_day0=lambda s: (s <= 0).mean(), by_day7=lambda s: (s <= 7).mean(),
    median_lag=lambda s: s.median(), p99_lag=lambda s: s.quantile(0.99))
print("Committee referral timing (first 'Referred to' activity, days after introduction)")
print(reft.round(3).T.to_string())
b0 = float(reft.by_day0.min())
record_check(C_SECTION, "C12", "Committee referral is known at t0=0 for >= 95% of law-capable bills in every Congress",
             bool(b0 >= 0.95), metric=b0, threshold="min share referred by day 0 >= 0.95",
             detail=f"min share by day 0 {b0:.3f}, by day 7 {reft.by_day7.min():.3f}; bills without a referral "
                    f"{1 - reft.has_referral.min():.3f} (max). Referral committees are usable as t0 features if referred "
                    "by t0; other bill_committees activities (markup, hearings, reported) are post-introduction and "
                    "reported-by is the task-2 label itself.")

# %%
# C13: CRS summary timing. Only update_date exists (last modification), so update_date <= intro+7 is a lower bound
# on availability at t0. Bulk re-loads of old Congresses make this uninformative before ~115.
sm = load("bill_summaries", ("bill_id", "version_code", "update_date", "congress"))
sm = sm[sm.bill_id.isin(lb.bill_id) & (sm.version_code == "00")].groupby("bill_id")["update_date"].min()
r["sum_lag"] = (r.bill_id.map(sm).dt.normalize() - r.introduced_date).dt.days
sumt = r.groupby("congress")["sum_lag"].agg(
    has_summary=lambda s: s.notna().mean(), within7=lambda s: (s <= 7).mean(), within30=lambda s: (s <= 30).mean(),
    median_lag_days=lambda s: s.median())
print("Introduced-version CRS summary: coverage and last-update lag after introduction (days)")
print(sumt.round(3).T.to_string())
w7 = float(sumt.loc[C_EVAL, "within7"].max())
w30 = float(sumt.loc[C_EVAL, "within30"].max())
record_check(C_SECTION, "C13", "Requirement: do not use CRS summary text as a t0 feature (available by t0=7 for too few bills)", None,
             metric=w7, threshold="requirement (a t0 feature must exist for >= 90% of eval bills at t0; observed max share shown)",
             detail=f"share with a summary last updated within 7 days: {sumt.loc[C_EVAL, 'within7'].round(3).to_dict()}; within 30 days max "
                    f"{w30:.3f}; median lag {sumt.loc[C_EVAL, 'median_lag_days'].to_dict()} days. Why 90%: a feature missing for more than "
                    f"10% of eval bills at t0 is mostly a missing-indicator; here {1 - w7:.0%} of eval bills (the best eval Congress) would "
                    "have no summary at t0=7, while every training bill has one in the snapshot (train/test feature mismatch). Proxy: "
                    "update_date is the LAST edit, so availability may be earlier, but nothing shows summaries exist at t0; use the "
                    "introduced title / text instead. Same for every split, so it does not separate splits.",
             requirement=True)

# %% [markdown]
# ### C24, C14. Are the labels of the most recent Congresses fully loaded?
#
# **Why it matters.** 118 is the natural test Congress for task 1, and it looks different: lower positive density (A3),
# later non-original cosponsors (C3), and many records still being edited (C14). If cosponsor events or committee-report
# actions of 117/118 were still missing from the snapshot, those differences would be data latency, not behaviour, and
# 118 would understate task-1 and task-2 performance. Three tests: (1) internal consistency: the snapshot counts
# (bills.n_cosponsors, n_actions, latest_action_date) against the event tables; (2) staleness: events dated after the
# record's own update_date; (3) the end-of-Congress tail: if late events were missing, the share of a Congress's
# cosponsorships dated in its last 90 days (and of committee reports in its last 180 days) would be too low.
#
# **How to read it.** Mismatch and stale shares should be 0; the tail shares of 117/118 are judged with `rel_to_train`
# (lower tail: only a deficit signals missing events). The last line compares, inside 118, how often bills with and
# without cosponsor rows were modified after the Congress ended (post-end edits driven by cosponsor loading would differ).

# %%
# C24: label completeness per Congress: snapshot counts vs event tables, events after update_date, end-of-Congress tails.
bcs = load("bill_cosponsors", ("bill_id", "sponsorship_date", "withdrawn_date"))
bcs = bcs[bcs.bill_id.isin(cut.index)]
lc24 = lb[lb.congress.isin(CONGRESSES)][["bill_id", "congress", "c_end", "n_cosponsors", "n_actions", "latest_action_date",
                                          "update_date"]].set_index("bill_id")
lc24["ud"] = lc24.update_date.dt.normalize()
rows_all = bcs.groupby("bill_id").size().reindex(lc24.index).fillna(0)
rows_act = bcs[bcs.withdrawn_date.isna()].groupby("bill_id").size().reindex(lc24.index).fillna(0)
act_rows = acts.groupby("bill_id").size().reindex(lc24.index).fillna(0)
act_max = acts.groupby("bill_id").action_date.max().reindex(lc24.index)
ev_max = pd.concat([act_max, bcs.groupby("bill_id").sponsorship_date.max().reindex(lc24.index),
                    bcs.groupby("bill_id").withdrawn_date.max().reindex(lc24.index)], axis=1).max(axis=1)
lc24["cosp_count_mismatch"] = lc24.n_cosponsors != rows_all
lc24["action_count_mismatch"] = lc24.n_actions != act_rows
lc24["latest_action_mismatch"] = lc24.latest_action_date.dt.normalize() != act_max.dt.normalize()
lc24["event_after_update"] = ev_max > lc24.ud
comp = lc24.groupby("congress")[["cosp_count_mismatch", "action_count_mismatch", "latest_action_mismatch", "event_after_update"]].mean()
bcs_c = bcs.merge(lc24[["congress", "c_end"]], left_on="bill_id", right_index=True)
act_c = bcs_c[bcs_c.withdrawn_date.isna()]
comp["withdrawn_row_share"] = bcs_c.groupby("congress").withdrawn_date.agg(lambda v: v.notna().mean())
comp["cosp_last90"] = (act_c.sponsorship_date > act_c.c_end - pd.Timedelta(days=90)).groupby(act_c.congress).mean()
rp = adv[adv.congress.isin(CONGRESSES) & adv.reported_date.notna()].copy()
rp["c_end"] = rp.congress.map(congress_end)
comp["report_last180"] = (rp.reported_date > rp.c_end - pd.Timedelta(days=180)).groupby(rp.congress).mean()
print("Label completeness per Congress (law-capable bills; mismatch / stale shares should be 0)")
print(comp.T.to_string(float_format=lambda v: f"{v:.4f}"))
tail_c_tab, tail_c_info = rel_to_train(comp["cosp_last90"], side="lower")
tail_r_tab, tail_r_info = rel_to_train(comp["report_last180"], side="lower")
cons_bad = float(comp[["cosp_count_mismatch", "action_count_mismatch", "latest_action_mismatch", "event_after_update"]].max().max())
tail_fail = bool(tail_c_tab.loc[tail_c_tab.role == "cand", "fail"].any() or tail_r_tab.loc[tail_r_tab.role == "cand", "fail"].any())
x118 = lc24[lc24.congress == CONGRESSES[-1]].copy()
x118["has_cosp"] = rows_all.reindex(x118.index) > 0
x118["post_end"] = x118.ud > x118.c_end
x118["recent"] = x118.update_date >= SNAPSHOT_DATE - pd.Timedelta(days=365)
pe = x118.groupby("has_cosp")[["post_end", "recent"]].mean()
print(f"{CONGRESSES[-1]}: modified after Congress end / in last 365 d of snapshot, bills with cosponsor rows "
      f"{pe.loc[True, 'post_end']:.3f} / {pe.loc[True, 'recent']:.3f} vs without {pe.loc[False, 'post_end']:.3f} / {pe.loc[False, 'recent']:.3f}")
c24_fail = bool(cons_bad > 0 or tail_fail)
record_check(C_SECTION, "C24", "Labels of the most recent Congresses (cosponsor events, committee reports) are fully loaded",
             not c24_fail, metric=float(min(tail_c_tab.loc[tail_c_tab.role == "cand", "t_pi"].min(),
                                            tail_r_tab.loc[tail_r_tab.role == "cand", "t_pi"].min())),
             threshold=f"consistency/stale shares = 0 in every Congress AND end-of-Congress tail shares not below the PI ({tail_c_info['rule']}); "
                       "metric = min t of the two tails over candidates",
             detail=f"max consistency/stale share {cons_bad:.4f} " + ("(n_cosponsors equals bill_cosponsors rows INCLUDING withdrawn; n_actions "
                    "and latest_action_date match bill_actions; no event is dated after its record's update_date, so no record is older than "
                    "its own events). " if cons_bad == 0 else f"(per-Congress maxima: {comp.iloc[:, :4].max().round(4).to_dict()}). ")
                    + f"Cosponsorships in the last 90 days: {rel_detail(tail_c_tab, tail_c_info)}. Committee reports in the last 180 "
                    f"days: {rel_detail(tail_r_tab, tail_r_info)}. A share ABOVE the reference is not latency but a timing shift (more committee reports late in the "
                    f"Congress in {[int(c) for c in tail_r_tab.index[(tail_r_tab.role == 'cand') & (tail_r_tab.t_pi > tail_r_info['tcrit'])]]}): it "
                    "shortens the effective 180-day window for late bills and belongs to A's task-2 drift checks. Inside {CONGRESSES[-1]}, post-end / last-year edits for bills with vs without "
                    f"cosponsor rows: {pe.loc[True, 'post_end']:.3f}/{pe.loc[True, 'recent']:.3f} vs {pe.loc[False, 'post_end']:.3f}/"
                    f"{pe.loc[False, 'recent']:.3f}. Withdrawn rows: {CONGRESSES[-1]} {comp.loc[CONGRESSES[-1], 'withdrawn_row_share']:.4f} vs 108-117 max "
                    f"{comp.loc[CONGRESSES[:-1], 'withdrawn_row_share'].max():.4f} (more withdrawals recorded; cosp drops them, so labels are "
                    "unaffected). Limits: the counts are parsed from the same BILLSTATUS records as the event tables, so (1) checks parsing, "
                    "not an external source; (3) is the real latency test. If PASS: the 118 density / lag differences (A3, C3) are "
                    "behaviour, and 118 is a valid test Congress. If FAIL: drop or down-weight 118 as a test Congress until reloaded. "
                    f"S4 is judged on the candidate Congresses its slices test ({S4_EVAL}): incomplete labels would hit its end-of-Congress "
                    "test slice hardest.",
             **dict(zip(("by_split", "na_reason"), full_split(combine_by_split(rel_by_split(tail_c_tab), rel_by_split(tail_r_tab)),
                                                              ok=cand_ok(tail_c_tab, tail_r_tab)))))

# %%
# C14: bill record update_date relative to Congress end and to the snapshot date (how "final" each Congress is).
snap = bills["update_date"].max()
u = lb[lb.congress.isin(CONGRESSES + [IN_PROGRESS])].copy()
u["days_after_end"] = (u.update_date - u.c_end).dt.days
u["recent"] = u.update_date >= snap - pd.Timedelta(days=365)
updt = u.groupby("congress").agg(after_end=("days_after_end", lambda s: (s > 0).mean()),
                                 median_days_after_end=("days_after_end", "median"),
                                 updated_last_365d=("recent", "mean"))
print(f"Bill record update_date vs Congress end (snapshot date = max update_date = {snap.date()})")
print(updt.round(3).T.to_string())
rec118, rec_old = float(updt.loc[118, "updated_last_365d"]), float(updt.loc[C_TRAIN, "updated_last_365d"].mean())
record_check(C_SECTION, "C14", f"Recent-edit share of {CONGRESSES[-1]} records (INFO; becomes a split verdict only if C24 finds incomplete labels)",
             (False if c24_fail else None), metric=rec118,
             threshold="INFO unless C24 fails (then: 118 records unsettled = labels may still change)",
             detail=f"share of bills updated within 365 days of the snapshot: 118={rec118:.3f}, 117={updt.loc[117, 'updated_last_365d']:.3f}, "
                    f"108-115 mean={rec_old:.3f}. Every Congress's records post-date its end, confirming bill-level fields reflect "
                    "end-of-life state (C11). Recent edits alone do not mean missing labels: C24 finds "
                    + ("incomplete labels, so 118 records are unsettled and splits testing on 118 are flagged. " if c24_fail else
                       "no missing cosponsor events or committee reports, so the edits concern annotations (summaries, subjects: C15) "
                       "that no t0 model may use; no split verdict. ")
                    + "The old fixed band (train mean + 0.10) compared against a reference that is zero by construction (old records "
                      "are never edited) and was dropped.",
             by_split=(rel_by_split(updt.loc[CONGRESSES, "updated_last_365d"] > rec_old + 0.10) if c24_fail else None))

# %% [markdown]
# ### C15-C16. Data completeness and annotation latency
#
# **Why it matters.** Policy areas, subjects, summaries and committee referrals are added by CRS and
# clerks after introduction. If the newest Congress (118, the natural test set) is less annotated than
# older ones, a model that uses those features is tested on degraded inputs, and the test score
# understates (or misstates) real performance. Congress 119, which is still in progress, lets us measure
# how fast annotations arrive: compare coverage for young versus old bills.
#
# **How to read it.** The line chart shows per-Congress coverage of each annotation for law-capable bills.
# The second table shows, for Congress 119 only, coverage by bill age at the snapshot date.

# %%
# C15: coverage of policy_area, subjects, summaries, committee referral per Congress (law-capable bills).
subj_ids = set(load("bill_subjects", ("bill_id",))["bill_id"].unique())
sum_ids = set(load("bill_summaries", ("bill_id",))["bill_id"].unique())
ref_ids = set(first_ref.index)
cv = lb[["bill_id", "congress", "policy_area", "introduced_date"]].copy()
cv["policy_area"] = cv.policy_area.notna()
cv["subjects"] = cv.bill_id.isin(subj_ids)
cv["summary"] = cv.bill_id.isin(sum_ids)
cv["referral"] = cv.bill_id.isin(ref_ids)
FIELDS = ["policy_area", "subjects", "summary", "referral"]
covt = cv.groupby("congress")[FIELDS].mean()
print("Annotation coverage per Congress (law-capable bills; 119 in progress)")
print(covt.round(3).T.to_string())
cl2 = covt.loc[CONGRESSES].reset_index().melt("congress", var_name="annotation", value_name="coverage")
line_chart(cl2, "congress", "coverage", series="annotation", yfmt=".0%", yzero=False,
           title="Share of law-capable bills with each annotation, per Congress",
           subtitle="Coverage in the snapshot; 119 excluded (in progress)", xlabel="Congress", name="C_coverage")
# Verdicts per field with the shared rel_to_train rule (lower tail: only missing annotations hurt). Only fields that can be
# t0 features count for splits: referral (known at day 0, C12) and policy area IF task 2 predicts after ~30 days (C25).
cov_tabs = {f: rel_to_train(covt.loc[CONGRESSES, f], side="lower") for f in FIELDS}
cov_fail = {f: bool(t.loc[t.role == "cand", "fail"].any()) for f, (t, _) in cov_tabs.items()}
cov_txt = {f: rel_detail(*cov_tabs[f]) for f in FIELDS}
record_check(C_SECTION, "C15a", "Requirement: legislative subjects are not a t0 feature (coverage of recent Congresses still filling in)", None,
             metric=float(covt.loc[C_EVAL, "subjects"].min()), threshold="requirement (INFO; subjects are not available at any t0 <= 90 d, C16/C25)",
             detail=f"subjects coverage: {cov_txt['subjects']}. CRS adds subjects over months ({IN_PROGRESS} bills aged 31-90 d: "
                    f"{cv[(cv.congress == IN_PROGRESS)].pipe(lambda d: d[(SNAPSHOT_DATE - d.introduced_date).dt.days.between(31, 90)].subjects.mean()):.3f}), "
                    "so no t0 model may use them and the 117/118 deficit cannot bias a correctly built model. If subjects were used as "
                    "static bill attributes, the recent test Congresses would be scored on degraded inputs (and train would leak post-t0 "
                    "information): do not.", quantity="subject_coverage", requirement=True)
c15b_fields = ["referral", "policy_area"]
record_check(C_SECTION, "C15b", "Coverage of t0-usable annotations (referral; policy area for task 2 at t0 >= 30 d) in eval Congresses within train noise",
             not any(cov_fail[f] for f in c15b_fields), metric=float(covt.loc[C_EVAL, c15b_fields].min().min()),
             threshold=f"{cov_tabs['referral'][1]['rule']}, per field",
             detail=f"referral: {cov_txt['referral']}. Policy area (counts only for task 2 with t0 >= 30 d, C25; for task 1 and task 2 at t0 <= 7 it "
                    f"must be masked, C18): {cov_txt['policy_area']}. CRS summaries are excluded from this verdict (never a t0 feature, C13; "
                    f"coverage shown in C15c). Task-2 t0 assumed: >= 30 d for the policy-area part, any t0 for referral. S4 judged on "
                    f"the candidate Congresses its slices test ({S4_EVAL}).",
             quantity="annotation_coverage", **dict(zip(("by_split", "na_reason"), full_split(
                 combine_by_split(*(rel_by_split(cov_tabs[f][0]) for f in c15b_fields)),
                 ok=cand_ok(*(cov_tabs[f][0] for f in c15b_fields))))))
record_check(C_SECTION, "C15c", "CRS-summary coverage of eval Congresses (INFO: summaries are not a t0 feature)", None,
             metric=float(covt.loc[C_EVAL, "summary"].min()), threshold="INFO",
             detail=f"summary: {cov_txt['summary']}. The {CONGRESSES[-1]} deficit is annotation latency (C16: CRS writes summaries months "
                    "after introduction) in a field no t0 model may use (C13), so it does not penalise splits that test on 118.")

# %%
# C16: annotation latency measured on in-progress Congress 119: coverage by bill age at the snapshot date.
y = cv[cv.congress == IN_PROGRESS].copy()
y["age_days"] = (snap.normalize() - y.introduced_date).dt.days
y["age"] = pd.cut(y.age_days, [-1, 7, 30, 90, 180, 365, 10_000], labels=["0-7", "8-30", "31-90", "91-180", "181-365", ">365"])
lat = y.groupby("age", observed=True).agg(bills=("bill_id", "size"), **{f: (f, "mean") for f in FIELDS})
print(f"Congress {IN_PROGRESS}: coverage by bill age at snapshot ({snap.date()})")
print(lat.round(3).to_string())
young = (y[y.age_days <= 30][FIELDS]).mean()
gap = (lat.iloc[-1][FIELDS] - young)
record_check(C_SECTION, "C16", "Annotations exist within 30 days of introduction (119 bills aged 0-30 d vs > 365 d)", None,
             metric=f"max coverage gap {gap.max():.3f} ({gap.idxmax()})",
             detail=f"coverage for bills aged 0-30 d: {young.round(3).to_dict()}; gap vs oldest bucket {gap.round(3).to_dict()}. "
                    "Informational proxy (snapshot of 119 only, not the historical arrival time): a large gap means the "
                    "annotation is typically assigned after t0 and must not be a t0 feature without a timestamp.")

# %% [markdown]
# ### C18. Is the policy area known at prediction time?
#
# **Why it matters.** Sections B and D use the CRS policy area as a t0 feature (D's empirical-Bayes rate per
# policy area, B's bill attributes). In the snapshot almost every bill of 108-118 has one, but CRS assigns it
# *after* introduction, and the historical assignment date is not in the data. The in-progress Congress 119 is
# the only window onto that lag: if bills a few days old often lack a policy area, then a model trained on
# complete snapshots sees a feature at t0 that it would not have in deployment.
#
# **How to read it.** Policy-area missingness of 119 bills by age at the snapshot, from the shared
# `PA_MISSING_AT_T0` (so sections B, C and D quote the same numbers; 0-7 and 8-30 days matter for t0 = 0 and 7),
# and whether the youngest bills were introduced during a recess (floor actions per day from `bill_actions`).
# Proxy: CRS's current speed, not the speed in earlier Congresses.

# %%
# C18: policy-area availability at t0 from the shared proxy PA_MISSING_AT_T0 (setup; 119 bills by age at SNAPSHOT_DATE),
# plus how representative the youngest bucket's introduction window is (recess check from bill_actions dates).
pm_t = PA_MISSING_AT_T0
print(f"policy area missing by age at snapshot (shared PA_MISSING_AT_T0, {IN_PROGRESS} law-capable bills): "
      + ", ".join(f"{i} {r.share_missing:.1%} (n={int(r.n)})" for i, r in pm_t.iterrows()))
young_lo = SNAPSHOT_DATE - pd.Timedelta(days=int(pm_t.loc["0-7d", "age_hi"]))
fa = load("bill_actions", ("action_date", "action_type", "congress"))
fa = fa[(fa.congress == IN_PROGRESS) & (fa.action_type == "Floor")]
fl_day = fa.groupby("action_date").size().reindex(pd.date_range(congress_start(IN_PROGRESS), SNAPSHOT_DATE), fill_value=0)
intro_day = lb[lb.congress == IN_PROGRESS].groupby("introduced_date").size().reindex(fl_day.index, fill_value=0)
win = slice(young_lo, SNAPSHOT_DATE)
roll_fl, roll_in = fl_day.rolling(8).mean(), intro_day.rolling(8).mean()
rec = dict(floor_per_day=float(fl_day[win].mean()), floor_median_8d=float(roll_fl.median()),
           intro_per_day=float(intro_day[win].mean()), intro_median_8d=float(roll_in.median()))
print(f"youngest bucket introduced {young_lo.date()}..{SNAPSHOT_DATE.date()}: floor actions/day {rec['floor_per_day']:.1f} "
      f"(119 median 8-day window {rec['floor_median_8d']:.1f}), law-capable introductions/day {rec['intro_per_day']:.1f} "
      f"(median {rec['intro_median_8d']:.1f})")
pa07, pa830 = 1 - PA_MISS_T0, 1 - float(pm_t.loc["8-30d", "share_missing"])
hist_missing = float((~lb.loc[lb.congress.isin(CONGRESSES), "policy_area"].notna()).mean())
recess = rec["floor_per_day"] < 0.5 * rec["floor_median_8d"]
pa_ok = bool(min(pa07, pa830) >= 0.95)
record_check(C_SECTION, "C18", "Requirement: mask policy area at t0 <= 7 d (present for < 95% of 119 bills aged 0-7 and 8-30 days)", None,
             metric=float(min(pa07, pa830)),
             threshold=f"requirement (INFO; the rule binds while availability < 0.95 in either age bucket: {'binds' if not pa_ok else 'does not bind'})",
             detail=f"shared proxy PA_MISSING_AT_T0: present for {pa07:.1%} of 0-7 d bills (n={PA_MISS_T0_N}) and {pa830:.1%} of "
                    f"8-30 d bills (n={int(pm_t.loc['8-30d', 'n'])}); 108-118 snapshot missing share {hist_missing:.3f}. Small sample, "
                    f"and the youngest bills were introduced {young_lo.date()}..{SNAPSHOT_DATE.date()}, when the chambers logged "
                    f"{rec['floor_per_day']:.1f} floor actions/day vs a 119 median of {rec['floor_median_8d']:.1f}"
                    f"{' (a recess / pro-forma period)' if recess else ''}: CRS latency then may not reflect normal weeks. Coverage "
                    "at the same age for non-recess months cannot be measured (policy_area has no timestamp; every older 119 bill "
                    "is already older than 30 d). Even the 8-30 d bucket is below 95%, so the conclusion holds: the snapshot policy "
                    "area is partly a post-t0 field. Train with policy area masked at PA_MISS_T0 plus a missing-indicator, or derive "
                    "the topic from the introduced title/text; applies to D's eb_rate baseline and GNN bill features, in every split. "
                    "Recorded as a requirement: availability at t0 does not depend on the split, so it is not a split failure (same "
                    "rule as C25, B5b and the shared PA_MISS_T0).", requirement=True)

# %% [markdown]
# ### C25. Which prediction time t0 should task 2 use?
#
# **Why it matters.** Task 2 (reported within 180 days) has no fixed prediction time. Whether the policy area, referral
# details and early cosponsors are legitimate features depends on it, and that decides whether the latency checks (C15b's
# policy-area part, C18, A13 with vs without policy area) constrain the split at all. A later t0 makes more features
# available but "spends" part of the label: bills already reported by t0 are no longer a prediction.
#
# **How to read it.** For candidate t0 values: availability at that bill age of each feature. Timestamped features
# (referral, cosponsors, actions) use the worst Congress in 108-118; untimestamped ones (policy area, subjects) use 119
# bills of about that age at the snapshot (small-sample proxy, n shown); summaries use the update_date lower bound. The
# last rows give the share of task-2 positives already reported by t0 (worst Congress), i.e. label lost by waiting.

# %%
# C25: feature availability by task-2 prediction time t0 (historical where timestamped, 119 age proxy otherwise) + label spent.
T0S = (0, 7, 30, 60)
AGE_WIN = {0: (0, 2), 7: (5, 9), 30: (25, 35), 60: (50, 70)}   # 119 bill ages used as proxy for "age = t0"
pos2 = adv[adv.congress.isin(CONGRESSES) & ~adv.censored & (adv.label == 1)].copy()
pos2["rep_lag"] = (pos2.reported_date - pos2.introduced_date).dt.days
spent = pd.DataFrame({t: (pos2.rep_lag <= t).groupby(pos2.congress).mean() for t in T0S})
ref_by = pd.DataFrame({t: (r.ref_lag <= t).groupby(r.congress).mean() for t in T0S})
sum_by = pd.DataFrame({t: (r.sum_lag <= t).groupby(r.congress).mean() for t in T0S})
av_rows = {"cosponsors / actions so far (timestamped)": {t: 1.0 for t in T0S},
           "referral committee (min 108-118)": ref_by.min().to_dict(),
           "CRS summary (lower bound, min 116-118)": sum_by.loc[C_EVAL].min().to_dict()}
n_win = {}
for f in ("policy_area", "subjects"):
    av_rows[f"{f} (119 proxy)"] = {}
    for t in T0S:
        m = y.age_days.between(*AGE_WIN[t])
        av_rows[f"{f} (119 proxy)"][t] = float(y.loc[m, f].mean()) if m.any() else np.nan
        n_win[t] = int(m.sum())
av_rows["task-2 positives already reported (max 108-118)"] = spent.max().to_dict()
av = pd.DataFrame(av_rows).T
av.columns = [f"t0={t} (n119={n_win[t]})" for t in T0S]
print("Feature availability at bill age t0 (share of law-capable bills) and task-2 label spent by t0")
print(av.round(3).to_string())
print("Share of task-2 positives reported by t0, per Congress")
print(spent.T.rename(index=lambda t: f"t0={t}").round(3).to_string())
pa_ok = [t for t in T0S if av_rows["policy_area (119 proxy)"][t] >= 0.95]
t0_pa = pa_ok[0] if pa_ok else None
spent_pa = float(spent[t0_pa].max()) if t0_pa is not None else np.nan
record_check(C_SECTION, "C25", "Requirement: fix task 2's prediction time t0 and use only features available at that age", None,
             metric=t0_pa, threshold="INFO (design decision; metric = smallest candidate t0 at which policy area is present for >= 95% of 119 bills)",
             detail=f"availability by t0 {list(T0S)}: " + "; ".join(f"{k}: " + "/".join(f"{v:.2f}" for v in row.values) for k, row in av.iterrows())
                    + f". Policy area reaches 95% at t0={t0_pa} (proxy, n={n_win.get(t0_pa)}), where {spent_pa:.1%} of task-2 positives (worst "
                    f"Congress) are already reported. Options: (a) t0=0 or 7 without policy area (C18 masking; A13 'noPA' and D's noPA "
                    f"variants are the relevant numbers; C15b's policy-area part and C18 then do not constrain the split beyond masking); "
                    f"(b) t0={t0_pa} with policy area, excluding bills reported by t0 and redefining the window as (t0, 180] (C6's censoring "
                    "unchanged if the window stays anchored at introduction). Tags: C6 assumes the window starts at introduction (any t0 "
                    f"<= 30); C15b policy-area part assumes t0 >= 30; C18 assumes t0 in (0, 7) for both tasks; C12 referral is usable at "
                    "any t0. A/D task-2 checks should state which t0 they assume (requested via setup).",
             requirement=True)

# %% [markdown]
# ### C17, C22. Member turnover and the point-in-time candidate pool
#
# **Why it matters.** The legacy pool per bill is "everyone who served in the chamber that Congress". Members
# who died, resigned or were seated mid-Congress are then candidates for bills introduced while they were out of
# office: guaranteed negatives that change the positive density and can take top-k slots from real candidates
# (a member who resigned in 118 still has a strong 117 history). This is a construction requirement for every
# split, not something that separates splits, but its size tells us whether D's backtest and the labels must
# switch to the shared point-in-time pool (`window_targets(point_in_time=True)` / `ROSTER_ASOF`).
#
# **How to read it.** C17: share of the legacy pool not in office in the month of `introduced_date + 7` (per
# Congress and chamber), and the share of task-1 positives who are outside the point-in-time pool (delegates and
# roster approximation errors: unrankable positives). C22: for S3 test folds, Recall@20 of a simple history
# ranker (sponsor-cosponsor affinity in the fit Congresses + recent activity) and the positive density, with the
# full-Congress roster vs the point-in-time pool.

# %%
# C17: legacy full-Congress pool vs point-in-time pool (ROSTER_ASOF month of introduced_date + 7), t0 = 7, uncensored bills.
wp7 = window_targets(7, point_in_time=True)
pp = wt[7][["bill_id", "congress", "chamber", "censored", "n_pos", "pool", "eligible"]].merge(
    wp7[["bill_id", "pool", "eligible", "pool_month"]], on="bill_id", suffixes=("_full", "_pit"))
pp = pp[~pp.censored & pp.congress.isin(CONGRESSES) & (pp.eligible_full > 0) & (pp.eligible_pit > 0)]
pp["out_share"] = 1 - pp.pool_pit / pp.pool_full
pp["dens_full"], pp["dens_pit"] = pp.n_pos / pp.eligible_full, pp.n_pos / pp.eligible_pit
pp["ch"] = pp.chamber.str[:5]
# Positives outside the point-in-time pool (unrankable there): delegates vs other.
pos7 = cosp[(cosp.lag_days > 7) & (cosp.lag_days <= 37)][["bill_id", "bioguide_id"]].merge(
    pp[["bill_id", "congress", "chamber", "pool_month"]], on="bill_id")
rk = pd.MultiIndex.from_frame(ROSTER_ASOF[["month", "chamber", "bioguide_id"]])
pos7["in_pit"] = pd.MultiIndex.from_arrays([pos7.pool_month, pos7.chamber, pos7.bioguide_id]).isin(rk)
deleg = set(load("member_terms", ("bioguide_id", "member_type")).query("member_type in @NONVOTING_TYPES").bioguide_id)
pos7["delegate"] = pos7.bioguide_id.isin(deleg)
turn = pp.groupby(["congress", "ch"]).out_share.mean().unstack("ch")
turn.columns = [f"out_of_office_{c}" for c in turn.columns]
turn["pos_outside_pit"] = pos7.groupby("congress").in_pit.agg(lambda v: 1 - v.mean())
turn["of_which_delegates"] = pos7[~pos7.in_pit].groupby("congress").delegate.mean()
print("Share of the legacy pool not in office at introduced_date + 7 (mean over bills), and task-1 positives outside the pit pool")
print(turn.round(3).T.to_string())
mx = float(turn[[c for c in turn.columns if c.startswith("out_of_office")]].max().max())
record_check(C_SECTION, "C17", "Requirement: eligible pool = members in office at introduced_date + t0 (not the full-Congress roster)", None,
             metric=mx, threshold="INFO (construction requirement for every split; effect quantified in C22)",
             detail=f"legacy pool share out of office: max {mx:.3f}, House mean {turn.filter(like='House').mean().iloc[0]:.3f}, Senate mean "
                    f"{turn.filter(like='Senat').mean().iloc[0]:.3f}. Positives outside the point-in-time pool: "
                    f"{1 - pos7.in_pit.mean():.2%} overall ({pos7[~pos7.in_pit].delegate.mean():.0%} of them delegates, who are excluded from "
                    "ROSTER_ASOF as non-voting): count them as misses or add delegates to the pool explicitly. Same requirement for "
                    "all splits, so it does not separate them.", requirement=True)

# %%
# C22: Recall@20 (t0 = 7) of a history ranker with the full-Congress roster vs the point-in-time pool, S3 test folds.
FULL_ROSTER = members.dropna(subset=["chamber"]).groupby(["congress", "chamber"]).bioguide_id.agg(set)
C22_FOLDS = [u for u in iter_eval_units("S3") if u["role"] == "test" and u["eval_congress"] in (115, CONGRESSES[-1])]


def recall20(unit, ch, k=20):
    """Mean Recall@k over uncensored eval bills with >= 1 positive, for the full roster and the point-in-time pool."""
    c, fit = unit["eval_congress"], unit["train"]
    E = pp[(pp.congress == c) & (pp.chamber == ch) & (pp.n_pos > 0)].merge(bills[["bill_id", "sponsor_bioguide_id"]], on="bill_id")
    months = ROSTER_ASOF[(ROSTER_ASOF.congress == c) & (ROSTER_ASOF.chamber == ch)]
    U = pd.Index(sorted(FULL_ROSTER.get((c, ch), set()) | set(months.bioguide_id)))
    hist = cosp[cosp.congress.isin(fit) & cosp.law_capable]
    aff = hist[hist.sponsor_bioguide_id.isin(E.sponsor_bioguide_id)].groupby(["sponsor_bioguide_id", "bioguide_id"]).size()
    spon = pd.Index(E.sponsor_bioguide_id.dropna().unique())
    A = np.zeros((len(spon), len(U)))
    ai, aj = spon.get_indexer(aff.index.get_level_values(0)), U.get_indexer(aff.index.get_level_values(1))
    ok = (ai >= 0) & (aj >= 0)
    A[ai[ok], aj[ok]] = aff.to_numpy()[ok]
    pop = hist[hist.congress == max(fit)].groupby("bioguide_id").size().reindex(U).fillna(0).to_numpy()
    si = spon.get_indexer(E.sponsor_bioguide_id)
    S = np.where(si[:, None] >= 0, A[np.clip(si, 0, None)], 0.0) + 1e-3 * pop[None, :]
    rows = pd.Index(E.bill_id)
    ev = cosp[cosp.bill_id.isin(rows)]
    seen, posv = ev[ev.lag_days <= 7], ev[(ev.lag_days > 7) & (ev.lag_days <= 37)]
    S[np.arange(len(E))[U.get_indexer(E.sponsor_bioguide_id) >= 0], U.get_indexer(E.sponsor_bioguide_id)[U.get_indexer(E.sponsor_bioguide_id) >= 0]] = -np.inf
    r_, c_ = rows.get_indexer(seen.bill_id), U.get_indexer(seen.bioguide_id)
    S[r_[c_ >= 0], c_[c_ >= 0]] = -np.inf
    P = np.zeros(S.shape, bool)
    r_, c_ = rows.get_indexer(posv.bill_id), U.get_indexer(posv.bioguide_id)
    P[r_[c_ >= 0], c_[c_ >= 0]] = True
    npos = E.n_pos.to_numpy()
    full_mask = U.isin(list(FULL_ROSTER.get((c, ch), set())))
    mm = months.groupby("month").bioguide_id.agg(set)
    pit_mask = np.stack([U.isin(list(mm.get(m, set()))) for m in E.pool_month])
    dmask = full_mask & U.isin(list(deleg))   # delegates in the legacy roster (non-voting, excluded from ROSTER_ASOF)
    out = {}
    for name, msk in (("full", np.broadcast_to(full_mask, S.shape)), ("pit", pit_mask), ("pit_deleg", pit_mask | dmask)):
        Sm = np.where(msk, S, -np.inf)
        top = np.argpartition(-Sm, k, axis=1)[:, :k]
        valid = np.take_along_axis(Sm, top, 1) > -np.inf
        out[f"recall_{name}"] = float(((np.take_along_axis(P, top, 1) & valid).sum(1) / npos).mean())
        if name == "full":   # out-of-office (not in pit pool, not a delegate) members occupying full-roster top-k slots
            out[f"oop_in_top{k}"] = float((valid & ~np.take_along_axis(pit_mask | dmask, top, 1)).sum(1).mean())
    return dict(fold=unit["fold"], chamber=ch[:5], bills=len(E), **out)


r22 = pd.DataFrame([recall20(u, ch) for u in C22_FOLDS for ch in ("House of Representatives", "Senate")])
dens = pp[pp.congress.isin([u["eval_congress"] for u in C22_FOLDS])].groupby(["congress", "ch"])[["dens_full", "dens_pit"]].mean()
r22["dens_full"] = [dens.loc[(int(f[2:]), ch), "dens_full"] for f, ch in zip(r22.fold, r22.chamber)]
r22["dens_pit"] = [dens.loc[(int(f[2:]), ch), "dens_pit"] for f, ch in zip(r22.fold, r22.chamber)]
r22["gain_pit"] = r22.recall_pit - r22.recall_full
r22["gain_pit_deleg"] = r22.recall_pit_deleg - r22.recall_full
print("S3 test folds, t0=7: Recall@20 and positive density, full-Congress roster vs point-in-time pool "
      "(oop_in_top20 = out-of-office members in the full-roster top 20, per bill)")
print(r22.round(4).to_string(index=False))
g22 = float(r22.gain_pit_deleg.abs().max())
record_check(C_SECTION, "C22", "Point-in-time pool vs full-Congress roster: effect on task-1 Recall@20 and positive density (S3 test folds)", None,
             metric=g22, threshold="INFO (requirement for every split; metric = max |Recall@20 gain| of pit+delegates vs full roster)",
             detail="Recall@20 gain vs full roster, pit+delegates / pit only (density full -> pit): " + ", ".join(
                    f"{r.fold} {r.chamber} {r.gain_pit_deleg:+.4f} / {r.gain_pit:+.4f} ({r.dens_full:.4f} -> {r.dens_pit:.4f})"
                    for r in r22.itertuples())
                    + f". Out-of-office members take {r22.oop_in_top20.min():.2f}-{r22.oop_in_top20.max():.2f} of the 20 full-roster slots "
                      "per bill. The voting-only ROSTER_ASOF drops delegates, who cast most positives outside it (C17), so a pit pool "
                      "WITHOUT delegates loses House recall; with delegates added back it matches or beats the full roster. Ranker = sponsor-cosponsor counts "
                      "in the fit Congresses + 0.001 x activity in the last fit Congress (proxy for any history-based model). "
                      "Labels and D's ranking pool (_ROSTER) should use the same point-in-time pool including delegates.")

# %% [markdown]
# ### C19-C21. Companion and reintroduced bills across the split boundary
#
# **Why it matters.** Many bills have a twin: an identical House/Senate companion in the same Congress, or
# a reintroduction of the same bill from an earlier Congress. Twins share sponsors, cosponsors and fate.
# A reintroduction in a *training* Congress is legitimate signal (it was known when the new bill was
# introduced). A same-Congress companion in the training portion is legitimate only if it was introduced
# earlier; if it was introduced *after* the evaluation bill (possible only under the random split S5), the
# model sees the future. Same-Congress companions also show what a by-Congress split forgoes: they are known
# at t0 only if the graph includes the evaluation Congress's own history up to t0.
#
# **How to read it.** Twins = same-Congress `bill_related` links (any relationship) plus identical official
# titles as introduced (normalized; titles shared by > 30 bills are dropped as generic). For every evaluation
# unit: the share of eval bills with a same-Congress twin in the training portion (split into introduced
# before vs after the eval bill) and with an identical-title bill in a training Congress (earlier = reintroduction,
# later = future leak). Twins are themselves a CRS/clerk annotation assigned after introduction (proxy).

# %%
# C19-C21: twins (same-Congress related / identical-title bills) and reintroductions vs each split's train portion.
F = lb[lb.congress.isin(CONGRESSES)][["bill_id", "congress", "introduced_date"]].reset_index(drop=True)
pos_of = pd.Series(F.index, index=F.bill_id)
ti = load("bill_titles", ("bill_id", "title_type", "title"))
ti = ti[(ti.title_type == "Official Title as Introduced") & ti.bill_id.isin(pos_of.index)].drop_duplicates("bill_id")
ti = ti.assign(norm=ti.title.str.lower().str.replace(r"\s+", " ", regex=True).str.strip().str.rstrip("."))
ti = ti[ti.groupby("norm").bill_id.transform("size").between(2, 30)]
tp = ti[["bill_id", "norm"]].merge(ti[["bill_id", "norm"]], on="norm", suffixes=("_a", "_b"))
tp = tp[tp.bill_id_a != tp.bill_id_b]
rl = load("bill_related", ("bill_id", "related_bill_id"))
rl = rl[rl.bill_id.isin(pos_of.index) & rl.related_bill_id.isin(pos_of.index)]
pairs = pd.concat([rl.set_axis(["bill_id_a", "bill_id_b"], axis=1), rl[["related_bill_id", "bill_id"]].set_axis(["bill_id_a", "bill_id_b"], axis=1),
                   tp[["bill_id_a", "bill_id_b"]]]).drop_duplicates()
pa_, pb_ = pos_of[pairs.bill_id_a].to_numpy(), pos_of[pairs.bill_id_b].to_numpy()
ca, cb = F.congress.to_numpy()[pa_], F.congress.to_numpy()[pb_]
da, db = F.introduced_date.to_numpy()[pa_], F.introduced_date.to_numpy()[pb_]
same = ca == cb
title_pair = set(zip(tp.bill_id_a, tp.bill_id_b))
is_title = np.array([(a, b) in title_pair for a, b in zip(pairs.bill_id_a, pairs.bill_id_b)])
n_bills = len(F)


def _any_per_bill(sel):
    """Boolean per bill of F: has at least one pair (as side a) satisfying sel."""
    out = np.zeros(n_bills, bool)
    out[pa_[sel]] = True
    return out


# Signal available within the eval Congress at t0 (needs within-Congress history up to t0): twin introduced on/before day 0.
earlier_twin = _any_per_bill(same & (db <= da))
tw_rows = []
for un in EVAL_UNITS:
    tr, ev = unit_masks(un, F)
    in_tr = tr[pb_]
    sel_ev = ev[pa_]
    def share(cond):
        return float(_any_per_bill(sel_ev & in_tr & cond)[ev].mean()) if ev.any() else np.nan
    tw_rows.append(dict(split=un["split"], fold=un["fold"], role=un["role"], eval_bills=int(ev.sum()),
                        twin_in_train_before=share(same & (db <= da)), twin_in_train_after=share(same & (db > da)),
                        reintro_in_train=share(~same & (cb < ca) & is_title), future_congress_twin=share(~same & (cb > ca)),
                        twin_known_t0_in_eval_congress=float(earlier_twin[ev].mean())))
tw = pd.DataFrame(tw_rows)
tws = tw.groupby("split")[["twin_in_train_before", "twin_in_train_after", "reintro_in_train", "future_congress_twin",
                           "twin_known_t0_in_eval_congress"]].mean()
print(f"pairs: {len(pairs):,} directed twin links ({int(same.sum()):,} same-Congress, {int((~same).sum()):,} cross-Congress)")
print("Share of eval bills linked to the training portion, mean over each split's evaluation units")
print(tws.round(3).to_string())
tw_plot = tws.reset_index().melt("split", value_vars=["twin_in_train_before", "twin_in_train_after", "reintro_in_train", "future_congress_twin"],
                                 var_name="link", value_name="share")
tw_plot["link"] = tw_plot.link.map({"twin_in_train_before": "same-Congress twin, earlier", "twin_in_train_after": "same-Congress twin, LATER (leak)",
                                    "reintro_in_train": "reintroduction (earlier Congress)", "future_congress_twin": "twin in LATER Congress (leak)"})
bar_chart(tw_plot, "split", "share", series="link", yfmt=".0%",
          title="Share of evaluation bills with a twin bill in the training portion, by candidate split",
          subtitle="Twins = same-Congress related / identical-title bills; mean over each split's val and test units",
          ylabel="share of eval bills", height=340, name="C_twins_by_split")
leak_by = tws.twin_in_train_after + tws.future_congress_twin
clean = [k for k in tws.index if leak_by[k] < 0.01]
record_check(C_SECTION, "C19", "Temporal splits (S1-S4) put no twin introduced after the eval bill into training (no future twin leak)",
             bool(leak_by.drop("S5", errors="ignore").max() < 0.01), metric=float(leak_by.drop("S5", errors="ignore").max()),
             threshold="< 1% of eval bills with a later twin in train",
             detail=f"share of eval bills with a later twin in train (same-Congress later + later-Congress): {leak_by.round(3).to_dict()}; "
                    f"clean splits: {clean}. By construction, by-Congress training never contains same-Congress or later bills; "
                    "S4 only contains earlier same-Congress bills. S1pool (train 108-115/116, eval 116-118) is True by the same "
                    "construction (no fit Congress is later than or equal to an eval Congress).",
             by_split=with_variants({k: bool(v < 0.01) for k, v in leak_by.items()}) | {"S1pool": True})
s5_leak = float(leak_by.get("S5", np.nan))
record_check(C_SECTION, "C20", "Random split S5: eval bills with a later twin (same or later Congress) in training < 1%",
             bool(s5_leak < 0.01), metric=s5_leak, threshold="< 0.01",
             detail=f"S5: same-Congress twin introduced later in train {tws.loc['S5', 'twin_in_train_after']:.3f}, twin in a later Congress "
                    f"in train {tws.loc['S5', 'future_congress_twin']:.3f}. These links carry the twin's future cosponsors and "
                    "committee outcome into training, so S5 scores are optimistic by more than the generic random-split bias; "
                    "use S5 only as a leaky upper reference.", by_split={k: (bool(s5_leak < 0.01) if k == "S5" else None) for k in ALL_SPLITS},
             na_reason="C20 judges only the random split S5; temporal splits are judged in C19")
bc_t0 = float(tw.loc[tw.split.isin(BY_CONGRESS_SPLITS), "twin_known_t0_in_eval_congress"].mean())
s4_before = float(tws.loc["S4", "twin_in_train_before"])
reint = tws.reintro_in_train.loc[BY_CONGRESS_SPLITS].round(3).to_dict()
record_check(C_SECTION, "C21", "Same-Congress twin signal a by-Congress split forgoes unless the eval Congress's history up to t0 is in the graph",
             None, metric=bc_t0, threshold="INFO",
             detail=f"{bc_t0:.1%} of by-Congress eval bills have a same-Congress twin introduced on or before them (known at t0 only if "
                    f"within-Congress history is allowed); S4 train contains such a twin for {s4_before:.1%} of its eval bills. "
                    f"Reintroductions from earlier training Congresses (legitimate signal for S1-S3): {reint}. Implication: the "
                    "by-Congress protocol should state whether eval-Congress edges dated before t0 are visible (a streaming / "
                    "inductive setting); if they are, twins become usable features and the S1-vs-S4 gap narrows.")

# %% [markdown]
# ### Section C summary
#
# The table below lists every check recorded in this section. FAIL does not automatically rule out a
# by-Congress split; read the `detail` column for what each failure implies. Three kinds of result appear:
# **split-discriminating** checks with a `by_split` verdict (C1, C2, C3, C5, C6, C6b, C8, C15b, C19, C20, C23b, C24;
# C14 only if C24 finds incomplete labels; every split in ALL_SPLITS gets a verdict or a stated reason, and S4 is judged on
# the candidate Congresses its slices test wherever the verdict is about that Congress's bills: C3, C15b, C24), **drift / timing** checks (C1-C3, C23: how different the recent Congresses
# are and whether changes happen inside a Congress), and **requirements** that hold for every split (C11, C13, C15a,
# C17, C18, C25 recorded with `requirement=True`; C10, C12, C22 state the same kind of rule: build features as of t0,
# use as-of party, mask policy area at t0 <= 7, fix task 2's t0, use the point-in-time pool incl. delegates).
# C24 decides whether 118's differences are data latency (it finds no missing cosponsor events or committee reports),
# C8b whether the joint House x Senate regime needs to be covered. C4 is a cross-reference to section A.

# %%
# Summary of section C checks grouped by result (requirements = construction rules for every split, not split failures).
cs = pd.DataFrame([c for c in CHECKS if c["section"] == C_SECTION])
cs["result"] = cs["passed"].map({True: "PASS", False: "FAIL"}).fillna("INFO")
cs["metric"] = cs["metric"].map(lambda v: f"{v:.4g}" if isinstance(v, (int, float, np.floating)) else str(v))
cs["req"] = cs.get("requirement", pd.Series(False, index=cs.index)).fillna(False).astype(bool)
cs.loc[cs.req, "result"] = "REQUIREMENT"
for res in ("FAIL", "PASS", "INFO", "REQUIREMENT"):
    sub = cs[cs.result == res]
    print(f"{res} ({len(sub)}): " + ", ".join(f"{r.id}={r.metric}" for r in sub.itertuples()))
print(f"Section C: {(cs.result == 'PASS').sum()} pass, {(cs.result == 'FAIL').sum()} fail, {(cs.result == 'INFO').sum()} info, "
      f"{cs.req.sum()} requirements (one line per check with its condition is printed where it is recorded above)")
