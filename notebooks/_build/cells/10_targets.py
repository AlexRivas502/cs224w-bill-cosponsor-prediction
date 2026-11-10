# %% [markdown]
# ## A. Label / target and covariate stationarity
#
# Every forward split (train on earlier data, evaluate on later data: S1a/S1b/S2/S3 by Congress, and S4
# within the last Congresses) evaluates Congresses that come **after** its training data. The model will
# also be deployed on a later Congress (119). So any shift found here is **a difference between history and
# the deployment target**, not a defect of a forward split: the forward split is the protocol that measures
# it honestly, and the random split S5 is the one that hides it. A drift FAIL below therefore selects a
# **remedy** (recency-weighted training, recalibration on the val Congress, density-normalised metrics,
# per-Congress reporting), not a different split. All A checks are recorded as `kind='drift'`; every
# forward split, including S4 (whose test units are the late parts of 116-118), gets the same verdict from
# the candidate Congresses it evaluates, and S5 is left unjudged (None), which is not the same as passing.
# Four kinds of shift are tested:
#
# - **Label shift**: the positive rate P(y) changes (e.g. fewer bills get reported). Threshold-based
#   metrics, calibration and AUPRC all move with the base rate even when the model is unchanged.
# - **Label-definition drift**: the label stops measuring the same thing (e.g. bills advance without a
#   committee report more often), so a base-rate change is partly an artifact.
# - **Concept shift**: the relationship P(y | x) changes. This is the shift that makes a trained model
#   wrong. It can be *regime-conditional* (it depends on which party controls the chamber), in which case
#   a regime feature fixes it.
# - **Covariate shift**: the inputs P(x) change, marginally or jointly.
#
# **Train pool and candidates.** The reference pool is split S1a's train set (108-115, `REL_REF`). The
# candidate val/test Congresses are 116, 117 and 118 (S1a: val 116, test 117; S1b: val 117, test 118).
#
# **Verdict rule (shared `rel_to_train`).** A candidate **fails** a check part when it falls outside the
# 95% **t prediction interval** for one new Congress built from the 8 reference Congresses
# (mean +/- t(7, 0.975) * sd * sqrt(1 + 1/8), about 2.5 SD; log scale for rates and counts; one-sided for
# distances such as PSI or AUROC, where a reference Congress's value is its leave-one-out distance to the
# other 7). Three extra readings guard against over-calling shift:
#
# - **Multiplicity.** Each check has several parts (features, chambers, t0 values) times 3 candidates. A
#   check is recorded as **FAIL only if at least one test survives a Holm correction** within that check.
#   The family holds one test per (part x candidate) cell plus one **joint test per part** that asks
#   whether the 3 candidates as a group sit off the reference mean (two-sample t of the candidate mean vs
#   the 8 reference Congresses; it catches a drop that recurs in every candidate but is too small to
#   survive in any single cell). The raw per-part t-PI failures are still printed and kept in `detail`,
#   next to the family-wise nested-LOO false-alarm rate. `by_split`: a split fails if a candidate Congress
#   it evaluates has a Holm-surviving cell, or a raw cell failure confirmed by a Holm-surviving joint test
#   (same direction).
# - **Trend vs break.** Each failing level quantity gets one of three labels: **break** (also outside the
#   prediction interval of a linear trend fitted on 108-115), **consistent with trend** (inside the trend
#   interval AND the 108-115 slope is itself significant), or **undetermined** (inside the trend interval
#   but no significant slope: with n = 8 and 1-3 Congresses of extrapolation, the trend test simply has
#   too little power to separate trend from level shift). Only the first two map to a remedy.
# - **Deployment proxy (A15).** The in-progress 119th Congress is where the model would be used; A15 checks
#   which Congresses look most like it, and A15b whether similarity follows **recency** or **party regime**
#   (same House / Senate majorities), with and without the sponsor-party features that encode party
#   control by construction.
#
# Fixed bands (e.g. 0.8-1.25x) are kept as a secondary column. Congress 119 is excluded from every split.

# %%
# Section A shared settings: reference pool, candidates, verdict helpers (wrapping rel_to_train), a bill feature
# table (complete Congresses + 119 for A15) and a small numpy logistic regression (domain classifier, Task-2 model).
A_TRAIN = list(REL_REF)                                # 108-115 (= SPLIT_DEFS['S1a']['train'])
A_CANDS = [c for c in CONGRESSES if c > A_TRAIN[-1]]   # 116-118
A_ALPHA = REL_ALPHA                                    # per-part level of the t prediction interval
A_RATIO_BAND = (0.8, 1.25)                             # secondary fixed band (+/-20%, symmetric in log space)
A_PP_BAND = 0.05                                       # secondary fixed band for shares (+/-5 pp)
A_CH_SHORT = {"House of Representatives": "House", "Senate": "Senate"}
A_ZTAB, A_REFTAB, A_FAM = [], [], []                   # candidate parts, reference nested-LOO parts, per-check families
A_IMPL = {"trend": "predictable drift: add Congress-index/time features, sliding window (S2) or rolling refit (S3)",
          "break": ("level change beyond the train trend: recency-weighted training (refit on train+val) and recalibrate on "
                    "the val Congress; section D's D26 backtests regime weighting on top of recency (keep regime as a "
                    "feature, not a weighting scheme, unless D26 shows a gain)"),
          "undetermined": "no remedy implied: the trend test has too little power to call it trend or break"}
A_IMPL_SHORT = {"trend": "time features / S2 / S3 refit", "break": "recency-weighted training + recalibrate on val",
                "undetermined": "(none: low power)"}
# How the A drift verdicts map onto splits: every forward split is judged from the candidate Congresses it evaluates.
A_S4_NOTE = ("S4 is judged from its folds 116-118 (its test units are the late part of each, scored by a model fitted on "
             "earlier Congresses plus the early part of the same Congress, so the Congress-level shift applies, partly "
             "absorbed by the in-Congress training bills)")
A_NA = {"S5": ("random bill split: train and test mix every Congress, so it hides the history->deployment shift; "
               "not judged, which is NOT passing")}


def a_d(series, nd=3):
    """Series -> plain {int key: rounded float} dict for readable detail text."""
    return {(int(k) if isinstance(k, (int, np.integer)) else k): round(float(v), nd) for k, v in series.items()}


def a_cohen_h(p1, p2):
    """Cohen's h effect size for two proportions (0.2 small, 0.5 medium)."""
    return float(2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1)))


def a_two_prop(pos1, n1, pos2, n2):
    """Chi-square test (2x2, no continuity correction) for equal proportions -> p-value."""
    tab = np.array([[pos1, n1 - pos1], [pos2, n2 - pos2]], float)
    return float(stats.chi2_contingency(tab, correction=False)[1])


def a_p(t, df, side):
    """p-value of a t statistic for the given tail."""
    if t is None or not np.isfinite(t) or df < 1:
        return np.nan
    return float({"two": 2 * stats.t.sf(abs(t), df), "upper": stats.t.sf(t, df), "lower": stats.t.cdf(t, df)}[side])


def a_rel(values, log=False, side="two", band=None, train=None):
    """rel_to_train for the candidates 116-118, plus p-values, a trend/break label and the secondary fixed band.
    Returns (tab with ref + cand rows, info)."""
    tab, info = rel_to_train(values, train=train, side=side, log=log, cands=A_CANDS)
    n, ref = info["n"], tab[tab.role == "ref"]
    cand = tab.role == "cand"
    tab["p"] = [a_p(r.t_pi, n - 1 if r.role == "cand" else n - 2, side) for r in tab.itertuples()]
    tab["trend_p"] = [a_p(r.trend_z, n - 2 if r.role == "cand" else n - 3, side) for r in tab.itertuples()]
    # Significance of the 108-115 linear slope itself (log scale when log=True): needed to call a failure a "trend".
    fr = (np.log(ref.value.where(ref.value > 0)) if log else ref.value).astype(float).dropna()
    slope_p = float(stats.linregress(np.asarray(fr.index, float), fr.values).pvalue) if len(fr) > 2 else np.nan
    trend_ok = bool(slope_p < A_ALPHA)
    tab["kind"] = np.where(tab.fail.eq(True),
                           np.where(tab.trend_fail.eq(True), "break", "trend" if trend_ok else "undetermined"),
                           np.where(tab.trend_fail.eq(True), "off-trend", ""))
    info.update(slope_p=slope_p,lo=ref.value.min(), hi=ref.value.max(), band=band,
                band_train_fail=None if band is None else int((~ref.value.between(*band)).sum()),
                band_cand_fail=None if band is None else [int(c) for c, v in tab.value[cand].items()
                                                          if not band[0] <= v <= band[1]])
    return tab, info


def a_holm(p, alpha=None):
    """Holm step-down survivors (bool array) for a vector of p-values (NaN -> 1)."""
    alpha = A_ALPHA if alpha is None else alpha
    p = np.nan_to_num(np.asarray(p, float), nan=1.0)
    order = np.argsort(p)
    ok = p[order] <= alpha / (len(p) - np.arange(len(p)))
    k = len(p) if ok.all() else int(np.argmin(ok))
    out = np.zeros(len(p), bool)
    out[order[:k]] = True
    return out


def a_split_from(cand_fail):
    """by_split from per-candidate verdicts {congress: fails?}: a forward split fails if a candidate it evaluates fails
    (reference Congresses are not judged; same as rel_by_split's default 'cands' mode). S4 is judged from its folds
    (116-118, the same candidates); S5 gets None (see A_NA)."""
    t = pd.DataFrame({"fail": pd.Series(cand_fail, dtype=object), "role": "cand"})
    bs = rel_by_split(t, col="fail")
    s4 = [c for c in SPLIT_DEFS["S4"]["folds"] if c in cand_fail and cand_fail[c] is not None]
    bs["S4"] = (not any(bool(cand_fail[c]) for c in s4)) if s4 else None
    bs["S5"] = None
    return bs


def a_split_metric(per_congress, roles=("test",)):
    """by_split of a per-Congress metric: mean over the Congresses each split evaluates in `roles` (S4: its folds;
    S5: None)."""
    out = {}
    for k in ALL_SPLITS:
        if split_kind(k) == "random":
            out[k] = None
            continue
        cs = SPLIT_DEFS["S4"]["folds"] if split_kind(k) == "within" else split_eval_congresses(k, roles=roles)
        v = [per_congress[c] for c in cs if c in per_congress]
        out[k] = round(float(np.mean(v)), 4) if v else None
    return out


def a_joint(cand_t, n, side):
    """Joint test that the k candidates as a group are off the reference mean. With t_i = (x_i - m) / (s sqrt(1 + 1/n)),
    the two-sample t of the candidate mean vs the n reference values is mean(t_i) * sqrt(1 + 1/n) / sqrt(1/k + 1/n),
    t-distributed with n - 1 df under H0 (it accounts for the reference mean the candidates share) -> (t, p)."""
    ct = np.asarray(cand_t, float)
    ct = ct[np.isfinite(ct)]
    if len(ct) < 2 or n < 3:
        return np.nan, np.nan
    tj = float(ct.mean() * np.sqrt(1 + 1 / n) / np.sqrt(1 / len(ct) + 1 / n))
    return tj, a_p(tj, n - 1, side)


def a_record(cid, cond, parts, why, quantity=None, fmt=".3f", band_label="", level=True):
    """Print one line per part, the family-wise false-alarm rate, and record a check that FAILs if any test survives a
    Holm correction within the check. Family = (part x candidate) cells + one joint test per part (candidates as a
    group vs the reference mean, a_joint). Raw per-part t-PI failures stay in `detail`. parts: list of (label, tab,
    info) from a_rel. level=False for distances (LOO PSI / JSD / domain AUROC): no trend/break label."""
    lines, worst, wscore, refrows, crow, jrow = [], 0.0, -np.inf, [], [], []
    for lab, t, i in parts:
        name = f"{cid} {lab}".strip()
        cand, ref = t[t.role == "cand"], t[t.role == "ref"]
        for c, r in cand.iterrows():
            crow.append(dict(check=cid, part=name, lab=lab, congress=int(c), t=float(r.t_pi), p=r.p,
                             fail=bool(r.fail), trend_z=float(r.trend_z), trend_p=r.trend_p,
                             trend_fail=bool(r.trend_fail), kind=r.kind, level=level, fa_rate=i["fa_rate"], n=i["n"]))
            sc = {"two": abs(r.t_pi), "upper": r.t_pi, "lower": -r.t_pi}[i["side"]]   # worst in the failing direction
            if np.isfinite(r.t_pi) and sc > wscore:
                worst, wscore = float(r.t_pi), sc
        tj, pj = a_joint(cand.t_pi.values, i["n"], i["side"])
        jrow.append(dict(part=name, lab=lab, t=tj, p=pj, side=i["side"]))
        for c, r in ref.iterrows():
            refrows.append(dict(check=cid, part=name, congress=int(c), fail=bool(r.fail),
                                trend_fail=bool(r.trend_fail), kind=r.kind, level=level))
        lo, hi = cand.pi_lo.iloc[0], cand.pi_hi.iloc[0]
        print(f"  {name}: ref {i['lo']:{fmt}}..{i['hi']:{fmt}}, 95% PI [{lo:{fmt}}, {hi:{fmt}}] (LOO false alarms "
              f"{i['fa_count']}/{i['n']}) | " + " | ".join(
                  f"{c}: {r.value:{fmt}} t={r.t_pi:+.1f} tr={r.trend_z:+.1f}{(' FAIL-' + r.kind if level else ' FAIL') if r.fail else ''}"
                  for c, r in cand.iterrows()) + f" | joint t={tj:+.1f} p={pj:.2g}")
        band = (f"; secondary fixed band {band_label} fails {i['band_train_fail']}/{i['n']} reference Congresses and "
                f"candidates {i['band_cand_fail']}") if i["band"] is not None else ""
        lines.append(f"[{lab or 'all'}] " + rel_detail(t, i, fmt) + f"; joint candidate t={tj:+.2f} (p={pj:.2g})" + band)
    cr, jr = pd.DataFrame(crow), pd.DataFrame(jrow)
    hs = a_holm(np.r_[cr.p.values, jr.p.values])   # one Holm family: cells + joint tests
    cr["holm"] = hs[:len(cr)] & cr.fail.values      # Holm survivor (and outside the PI)
    jr["holm"] = hs[len(cr):]
    # A Holm-surviving joint test confirms the recurring shift: candidates whose own cell fails the raw t-PI in the same
    # direction then count as failing too (the joint test supplies the multiplicity control their cells lacked).
    cr["jholm"] = False
    for r in jr[jr.holm].itertuples():
        cr.loc[(cr.part == r.part) & cr.fail & (np.sign(cr.t) == np.sign(r.t)), "jholm"] = True
    cr["sig"] = cr.holm | cr.jholm
    A_ZTAB.extend(cr.drop(columns="lab").to_dict("records"))
    A_REFTAB.extend(refrows)
    rf = pd.DataFrame(refrows)
    anyf = rf.groupby("congress").fail.any()
    fam = dict(check=cid, parts=len(parts), n_ref=len(anyf), fwer_loo=float(anyf.mean()),
               fwer_nominal=float(1 - (1 - A_ALPHA) ** len(parts)), fwer_ids=[int(c) for c in anyf.index[anyf]],
               joint_survivors=[r.part for r in jr[jr.holm].itertuples()])
    A_FAM.append(fam)
    lab_of = lambda r: f"{r.lab + ' ' if r.lab else ''}{r.congress}" + (f" ({r.kind})" if level else "")
    raw = [lab_of(r) for r in cr[cr.fail].itertuples()]
    surv = [lab_of(r) for r in cr[cr.holm].itertuples()]
    jsurv = [f"{r.lab or 'all'} joint (t={r.t:+.1f})" for r in jr[jr.holm].itertuples()]
    jraw = [f"{r.lab or 'all'} joint (t={r.t:+.1f})" for r in jr[jr.p < A_ALPHA].itertuples()]
    print(f"  {cid}: raw t-PI failures {raw or 'none'}; Holm survivors (decide the verdict): cells {surv or 'none'}, "
          f"joint {jsurv or 'none'}" + (f"; family-wise nested-LOO false-alarm rate {fam['fwer_loo']:.0%} over "
                                        f"{len(parts)} parts" if len(parts) > 1 else ""))
    record_check("A", cid, cond, not (surv or jsurv), metric=float(worst),
                 threshold=parts[0][2]["rule"] + f"; FAIL only if a test survives Holm over {len(cr)} cells + {len(jr)} joint tests",
                 detail=f"VERDICT RULE: FAIL iff >= 1 test survives a Holm correction over the {len(cr)} (part x candidate) "
                        f"cells and {len(jr)} joint-candidate tests (one per part: candidate mean vs reference mean, "
                        f"two-sample t, df n-1) of this check at {A_ALPHA:.0%}. Holm survivors: cells {surv or 'none'}; "
                        f"joint {jsurv or 'none'}. Raw per-part t-PI failures (secondary, not multiplicity-corrected): "
                        f"{raw or 'none'}; raw joint p < {A_ALPHA}: {jraw or 'none'}. "
                        + "; ".join(lines) + f". Family-wise nested-LOO false-alarm rate over {len(parts)} part(s): "
                        f"{fam['fwer_loo']:.0%} (nominal {fam['fwer_nominal']:.0%}). Labels: break = outside the 108-115 "
                        f"linear-trend PI; trend = inside it and the 108-115 slope is significant; undetermined = "
                        f"neither. KIND drift: a FAIL means the later Congresses (and so the deployment target) differ "
                        f"from 108-115; the forward split measures this; apply the remedy, do not switch splits. by_split: a "
                        f"split fails if a candidate it evaluates has a Holm-surviving cell, or a raw cell failure in the "
                        f"direction of a Holm-surviving joint test; {A_S4_NOTE}; S5 not judged. " + why,
                 quantity=quantity, by_split=a_split_from(cr.groupby("congress").sig.any().to_dict()),
                 kind="drift", na_reason=A_NA)


# Bill feature table (law-capable bills, complete Congresses + the in-progress one for A15). Row i of A_X = row i
# of fb_all; fb = complete Congresses only (index kept, so fb.index indexes A_X).
_cau = members.drop_duplicates(["bioguide_id", "congress"]).set_index(["bioguide_id", "congress"])["caucus"]
A_ALLC = CONGRESSES + [IN_PROGRESS]
fb_all = bills[bills.law_capable & bills.congress.isin(A_ALLC)][
    ["bill_id", "congress", "chamber", "bill_type", "policy_area", "policy_area_harm", "sponsor_party",
     "sponsor_bioguide_id", "day_of_congress"]].reset_index(drop=True)
fb_all["ch"] = fb_all.chamber.map(A_CH_SHORT)
fb_all = fb_all.merge(MAJORITY.assign(ch=MAJORITY.chamber.map(A_CH_SHORT))[["congress", "ch", "majority"]],
                      on=["congress", "ch"], how="left")
# Sponsor bloc = caucus (independents with the party they caucus with), falling back to the party code.
fb_all["sponsor_caucus"] = pd.Series(pd.MultiIndex.from_arrays([fb_all.sponsor_bioguide_id, fb_all.congress]).map(_cau),
                                     index=fb_all.index).fillna(fb_all.sponsor_party)
fb_all["sponsor_status"] = np.where(fb_all.sponsor_caucus == fb_all.majority, "majority", "minority")
cl_all = cosp[cosp.law_capable & cosp.congress.isin(A_ALLC)]
fb_all["n_original"] = fb_all.bill_id.map(cl_all[cl_all.is_original].groupby("bill_id").size()).fillna(0)
fb_all["day_bin"] = np.minimum(fb_all.day_of_congress.clip(lower=0) // 91, 7).astype(int).astype(str)
A_CAT = ["ch", "bill_type", "sponsor_caucus", "sponsor_status", "policy_area_harm", "day_bin"]
for f in A_CAT:   # categorical design columns as plain strings (missing -> its own level)
    fb_all[f] = fb_all[f].fillna("(missing)").astype(str)
fb = fb_all[fb_all.congress.isin(CONGRESSES)]
cl = cl_all[cl_all.congress.isin(CONGRESSES)]
A_NOCAU = fb[fb.sponsor_caucus.eq("(missing)")].groupby("congress").size()
A_LEVELS = {f: fb[f].value_counts().index.tolist() for f in A_CAT}   # first (most frequent) = reference level
A_XNAMES = ["(intercept)"] + [f"{f}={l}" for f in A_CAT for l in A_LEVELS[f][1:]] + ["log1p(n_original)"]
A_X = np.column_stack([np.ones(len(fb_all))] + [(fb_all[f].values == l).astype(float)
                                                 for f in A_CAT for l in A_LEVELS[f][1:]] + [np.log1p(fb_all.n_original.values)])
A_ROW = pd.Series(fb_all.index.values, index=fb_all.bill_id)
# Policy area is assigned by CRS weeks after introduction (PA_MISS_T0 of fresh bills lack it), so it is NOT
# available at t0. A_NOPA = design columns without policy area (the t0-safe feature set).
A_NOPA = np.array([not n.startswith("policy_area_harm=") for n in A_XNAMES])
# Sponsor-party columns encode party control by construction (sponsor_status = caucus == majority; the share of bills
# sponsored by each caucus moves with the majority). A15/A15b also use t0-safe sets without them.
A_FSETS = {
    "no party": A_NOPA & np.array([not n.startswith(("sponsor_caucus=", "sponsor_status=")) for n in A_XNAMES]),
    "majority only": A_NOPA & np.array([not n.startswith("sponsor_caucus=") for n in A_XNAMES]),
    "with party": A_NOPA,
}



def a_logit(X, y, w=None, l2=1.0, iters=40):
    """L2-penalised logistic regression by Newton/IRLS (intercept unpenalised) -> coefficients."""
    w = np.ones(len(y)) if w is None else w
    pen = np.full(X.shape[1], float(l2))
    pen[0] = 0.0
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        H = (X * (w * p * (1 - p))[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(H, X.T @ (w * (y - p)) - pen * b)
        b += step
        if np.abs(step).max() < 1e-7:
            break
    return b


def a_auc(y, s):
    """AUROC from ranks (Mann-Whitney)."""
    y = np.asarray(y).astype(bool)
    r = stats.rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


adv_u = adv[~adv.censored & adv.congress.isin(CONGRESSES)].copy()
adv_u["ch"] = adv_u.chamber.map(A_CH_SHORT)
adv_u = adv_u.merge(fb[["bill_id", "sponsor_caucus", "sponsor_status", "majority"]], on="bill_id", how="left")
print(f"train pool {A_TRAIN[0]}-{A_TRAIN[-1]} | candidates {A_CANDS} | uncensored advancement labels: {len(adv_u):,} | "
      f"bill features: {A_X.shape[1]} columns over {len(fb):,} law-capable bills | verdict: outside the {1 - A_ALPHA:.0%} t prediction interval of {A_TRAIN[0]}-{A_TRAIN[-1]}")
print(f"majority party of {IN_PROGRESS} (for A15): " + "; ".join(f"{r.chamber}: {r.majority}" for r in MAJORITY[MAJORITY.congress == IN_PROGRESS].itertuples())
      + f" | bills whose sponsor has no caucus/party (treated as minority; own level in the classifiers): {A_NOCAU.to_dict()}")

# %% [markdown]
# ### A1. Advancement base rate (Task 2) per Congress, and what drives its change
# **Why it matters:** if the share of bills reported within 180 days differs a lot between train and the
# held-out Congress, a model trained on 108-115 is mis-calibrated there, and AUPRC / F1 on the test
# Congress cannot be compared with validation scores. **How to read it:** `rate` per Congress; `ratio` =
# rate / pooled-train rate, `h` = Cohen's h and `p` = chi-square p (train rows: vs the other train
# Congresses). With about 10k bills almost every gap is "significant", so the verdict uses the t prediction
# interval of the train Congresses (`t` = standardised distance, `tr` = residual vs the 108-115 trend). The decomposition splits each candidate's log rate change into a **volume part**
# (more bills introduced) and a **count part** (fewer bills reported), both relative to the mean train
# Congress.

# %%
# A1: uncensored advancement rate per Congress; t prediction interval vs train Congresses (log scale); volume/count decomposition.
a1 = adv_u.groupby("congress").agg(n=("label", "size"), n_reported=("label", "sum"), rate=("label", "mean"))
a1["censored_share"] = adv[adv.congress.isin(CONGRESSES)].groupby("congress")["censored"].mean()
tr = adv_u[adv_u.congress.isin(A_TRAIN)]
p_tr = tr.label.mean()
_ref = {c: (tr[tr.congress != c] if c in A_TRAIN else tr) for c in CONGRESSES}
a1["ratio"] = [a1.loc[c, "rate"] / _ref[c].label.mean() for c in a1.index]
a1["h"] = [a_cohen_h(_ref[c].label.mean(), a1.loc[c, "rate"]) for c in a1.index]
a1["p"] = [a_two_prop(_ref[c].label.sum(), len(_ref[c]), a1.loc[c, "n_reported"], a1.loc[c, "n"]) for c in a1.index]
print(f"pooled train rate = {p_tr:.4f} (n={len(tr):,})")
print(a1.round(4).to_string())
t1, i1 = a_rel(a1.rate, log=True, band=(p_tr * A_RATIO_BAND[0], p_tr * A_RATIO_BAND[1]))
a_record("A1", "Advancement base rate of each candidate within the spread of train Congresses", [("", t1, i1)],
         "Rates compared on the log scale. FAIL implies: report rank metrics (AUROC) next to AUPRC, recalibrate or "
         "re-threshold on the validation Congress, and compare val/test scores only after base-rate normalisation "
         "(e.g. AUPRC lift = AUPRC / base rate). See A1b/A1c for how much of the drop is a label artifact.",
         quantity="label_rate_t2", fmt=".4f", band_label="0.8-1.25x pooled train")

# Decomposition: log(rate_c / rate_ref) = log(n_reported_c / mean_train) - log(n_c / mean_train).
ref_n, ref_k = a1.loc[A_TRAIN, "n"].mean(), a1.loc[A_TRAIN, "n_reported"].mean()
dec = pd.DataFrame({"n_ratio": a1.n / ref_n, "n_reported_ratio": a1.n_reported / ref_k})
dec["rate_ratio"] = dec.n_reported_ratio / dec.n_ratio
dec["log_change"] = np.log(dec.rate_ratio)
dec["volume_part"] = -np.log(dec.n_ratio)
dec["count_part"] = np.log(dec.n_reported_ratio)
dec["volume_share"] = (dec.volume_part / dec.log_change).where(dec.log_change < 0)
print("decomposition vs the mean train Congress (log_change = volume_part + count_part):")
print(dec.loc[A_CANDS].round(3).to_string())
tk, ik = a_rel(a1.n_reported, log=True)
a_record("A1d", "Number of reported bills (numerator of the Task-2 rate) within the spread of train Congresses",
         [("n_reported", tk, ik)],
         f"Volume share of the log rate drop: {a_d(dec.loc[A_CANDS, 'volume_share'], 2)} (1 = drop entirely from "
         f"more bills; < 1 = committees also reported fewer bills). FAIL here means the rate drop is not only a "
         f"volume effect: the reporting process itself changed (or the label lost coverage, see A1b).",
         quantity="label_count_t2", fmt=".0f")

# %% [markdown]
# ### A1b-A1c. Does the Task-2 label measure the same thing in every Congress?
# **Why it matters:** "reported out of committee within 180 days" misses bills that advance without a
# committee report: suspension of the rules in the House, unanimous consent in the Senate. If that route
# grows in later Congresses, part of the base-rate drop is a **label-definition artifact**, and a model
# would be penalised for correctly ranking bills that did advance. **How to read it:** among uncensored
# bills that passed their chamber of origin within 180 days, the share *not* reported within 180 days
# (`unrep_share`; `never_rep` = no Reported action at all). A1c recomputes the base rate with an
# alternative label (reported **or** passed the origin chamber within 180 days) and checks whether the
# candidates' drop shrinks under it.

# %%
# A1b/A1c: share of chamber-passed bills without a report; alternative label (reported OR passed origin chamber).
_act = load("bill_actions", ("bill_id", "action_date", "text"))
_pass = _act[_act.text.str.match(r"Passed/agreed to in (House|Senate)", na=False)]
_pass = (_pass.assign(ch=_pass.text.str.extract(r"in (House|Senate)", expand=False))
         .groupby(["bill_id", "ch"])["action_date"].min().rename("passed_date").reset_index())
lv = adv_u.merge(_pass, on=["bill_id", "ch"], how="left")
lv["passed180"] = (lv.passed_date - lv.introduced_date).dt.days.between(0, ADV_WINDOW_DAYS)
lv["alt"] = ((lv.label == 1) | lv.passed180).astype(int)
ps = lv[lv.passed180]
vt = ps.assign(unrep=ps.label == 0, never=ps.reported_date.isna())
val_ch = vt.groupby(["congress", "ch"]).agg(n_passed=("unrep", "size"), unrep_share=("unrep", "mean")).unstack()
val_ch.columns = [f"{a}_{b}" for a, b in val_ch.columns]
val = vt.groupby("congress").agg(n_passed=("unrep", "size"), unrep_share=("unrep", "mean"), never_rep=("never", "mean"))
val = val.join(val_ch.drop(columns=[c for c in val_ch.columns if c.startswith("n_passed")]))
val["alt_rate"] = lv.groupby("congress")["alt"].mean()
val["label_rate"] = a1.rate
print(val.round(3).to_string())
tb, ib = a_rel(val.unrep_share, side="upper")
a_record("A1b", "Share of chamber-passed bills that the Task-2 label misses (no report within 180 d) not above train spread",
         [("", tb, ib)],
         "Pass routes without a committee report (suspension, unanimous consent) are invisible to the label. FAIL "
         "means the label covers less real advancement in that Congress; use the alternative label (A1c) as a "
         "robustness target or add 'passed chamber' to the positive class.", quantity="label_validity_t2")
ta, ia = a_rel(val.alt_rate, log=True)
p_alt = lv[lv.congress.isin(A_TRAIN)].alt.mean()
shr = pd.DataFrame({"log_ratio_label": np.log(val.label_rate / p_tr), "log_ratio_alt": np.log(val.alt_rate / p_alt)}).loc[A_CANDS]
shr["drop_shrinks"] = shr.log_ratio_alt.abs() < shr.log_ratio_label.abs()
print("candidate log ratio vs pooled train, Task-2 label vs alternative label:")
print(shr.round(3).to_string())
a_record("A1c", "Alternative advancement label (reported OR passed origin chamber in 180 d): base rate within train spread",
         [("", ta, ia)],
         f"The candidates' gap shrinks under the alternative label for {int(shr.drop_shrinks.sum())}/{len(shr)} candidates "
         f"(log ratios {a_d(shr.log_ratio_label, 2)} -> {a_d(shr.log_ratio_alt, 2)}). If the alternative "
         f"label passes where A1 fails, the A1 drop is partly a label-definition artifact rather than a population change.",
         quantity="label_rate_t2_alt", fmt=".4f")

# %% [markdown]
# ### A2. Advancement base rate by chamber
# **Why it matters:** House and Senate bills have different reporting rates, so a pooled rate can hide
# opposite movements. A by-Congress split needs each chamber to be roughly stable, or the model needs
# chamber as a feature (and per-chamber evaluation). **How to read it:** the chart shows the rate per
# chamber, the pooled rate and the pooled rate under A1c's alternative label; the check applies the t
# prediction-interval rule within each chamber.

# %%
# A2: advancement rate per Congress x chamber; t PI per chamber vs the same chamber's train Congresses.
a2 = adv_u.groupby(["congress", "ch"])["label"].mean().unstack()
print(a2.round(4).T.to_string())
parts2 = []
for ch in a2.columns:
    pc = adv_u[adv_u.congress.isin(A_TRAIN) & (adv_u.ch == ch)].label.mean()
    parts2.append((ch,) + a_rel(a2[ch], log=True, band=(pc * A_RATIO_BAND[0], pc * A_RATIO_BAND[1])))
chart = pd.concat([a2.stack().rename("rate").reset_index().rename(columns={"ch": "series"}),
                   a1.rate.reset_index().assign(series="Both (Task-2 label)"),
                   val.alt_rate.rename("rate").reset_index().assign(series="Both (reported or passed)")])
line_chart(chart, "congress", "rate", series="series", title="Task-2 positive rate per Congress",
           subtitle=f"Uncensored law-capable bills; train {A_TRAIN[0]}-{A_TRAIN[-1]}, candidates {A_CANDS[0]}-{A_CANDS[-1]}; "
                    "last line = alternative label", yfmt=".1%", ylabel="positive share", xlabel="Congress",
           name="A_adv_rate_by_chamber")
a_record("A2", "Per-chamber advancement rate of each candidate within the same chamber's train spread", parts2,
         "If one chamber drifts more than the other, evaluate per chamber and include chamber as a feature.",
         quantity="label_rate_t2_chamber", fmt=".4f", band_label="0.8-1.25x same-chamber train")

# %% [markdown]
# ### A3. Cosponsor-task labels (Task 1) at t0 = 0 and t0 = 7
# **Why it matters:** ranking metrics (Hits@k, MRR, AUPRC) depend on how many positives each bill has
# and on the positive density among eligible legislators. If later Congresses have fewer
# post-introduction cosponsors, the test task is harder and scores drop for reasons unrelated to the
# model. **How to read it:** per Congress and t0: share of bills with at least one positive, mean
# positives per bill, positives / eligible (density) and censored share. The eligible pool is the
# **point-in-time** roster (members in office, delegates included, in the month of introduction + t0), the
# same pool section D ranks; `density_legacy` uses the full-Congress roster for comparison. A3 checks
# density, A3b the share of bills with any positive.

# %%
# A3: window_targets(t0, point_in_time=True) per Congress (uncensored bills): >=1 positive share, mean positives,
# density over the point-in-time pool (ROSTER_ASOF_ALL, delegates included; same pool as section D); legacy pool shown.
a3_rows = []
for t0 in (0, 7):
    wt = window_targets(t0, point_in_time=True)
    wt = wt[wt.congress.isin(CONGRESSES)]
    u = wt[~wt.censored & wt.eligible.gt(0)]
    g = u.groupby("congress").agg(bills=("n_pos", "size"), any_pos=("n_pos", lambda s: (s > 0).mean()),
                                  mean_pos=("n_pos", "mean"), pos=("n_pos", "sum"), elig=("eligible", "sum"))
    g["density"] = g.pos / g.elig
    wl = window_targets(t0)   # legacy full-Congress pool (secondary)
    wl = wl[wl.congress.isin(CONGRESSES) & ~wl.censored & wl.eligible.gt(0)].groupby("congress")[["n_pos", "eligible"]].sum()
    g["density_legacy"] = wl.n_pos / wl.eligible
    g["censored_share"] = wt.groupby("congress")["censored"].mean()
    g["t0"] = t0
    a3_rows.append(g.reset_index())
a3 = pd.concat(a3_rows, ignore_index=True)
print(a3.pivot(index="congress", columns="t0", values=["any_pos", "mean_pos", "density", "density_legacy", "censored_share"]).round(4).to_string())
line_chart(a3.assign(t0=lambda d: "t0=" + d.t0.astype(str)), "congress", "density", series="t0",
           title="Cosponsor-task positive density (new cosponsors / eligible legislators)",
           subtitle="Positives = cosponsors joining in (t0, t0+30]; point-in-time pool (incl. delegates); uncensored law-capable bills",
           yfmt=".2%", ylabel="density", xlabel="Congress", name="A_cosp_density")
p3, p3b = [], []
for t0, g in a3.groupby("t0"):
    s = g.set_index("congress")
    m = s.loc[A_TRAIN, "pos"].sum() / s.loc[A_TRAIN, "elig"].sum()
    p3.append((f"t0={t0}",) + a_rel(s.density, log=True, band=(m * A_RATIO_BAND[0], m * A_RATIO_BAND[1])))
    p3b.append((f"t0={t0}",) + a_rel(s.any_pos, log=True))
a_record("A3", "Cosponsor positive density (t0=0 and t0=7) of each candidate within train spread", p3,
         "Hits@k/MRR/AUPRC move with positive density: when this fails, val/test scores are not comparable to "
         "train-era scores. Use density-normalised metrics (lift over random) and compare models only within the "
         "same held-out Congress. Section C should reference this check rather than re-test it. Denominator = "
         "point-in-time eligible pool (ROSTER_ASOF_ALL in the month of introduction + t0, delegates included), the pool "
         "section D ranks; legacy full-Congress-pool density is printed as density_legacy (secondary).",
         quantity="cosp_density", fmt=".4f", band_label="0.8-1.25x pooled train")
a_record("A3b", "Share of bills with at least one new cosponsor (t0=0 and t0=7) within train spread", p3b,
         "Bills with no positive contribute nothing to ranking metrics; a falling share shrinks the effective test set.",
         quantity="label_rate_t1", fmt=".3f")

# %% [markdown]
# ### A4. Original-cosponsor share (what is left to predict after day 0)
# **Why it matters:** Task 1 only counts cosponsors who join *after* t0. Cosponsors listed at
# introduction are inputs (t0 = 7) or simply absent from the target. If the share of cosponsorships that
# are original rises, the target shrinks in later Congresses and the day-0 graph is denser: both the label
# and the input distribution move. **How to read it:** share of active cosponsorships (law-capable
# bills) that are original, and the shares joining in days 1-30 and after day 30.

# %%
# A4: share of cosponsorships that are original (day-0) per Congress; t PI vs train Congresses.
a4 = cl.groupby("congress")["is_original"].mean().rename("original_share").to_frame()
a4["post_intro_le30"] = cl.assign(w=cl.lag_days.between(1, WINDOW_DAYS)).groupby("congress")["w"].mean()
a4["post_intro_gt30"] = cl.assign(w=cl.lag_days > WINDOW_DAYS).groupby("congress")["w"].mean()
tr4 = cl[cl.congress.isin(A_TRAIN)].is_original.mean()
print(a4.round(3).T.to_string())
t4, i4 = a_rel(a4.original_share, band=(tr4 - A_PP_BAND, tr4 + A_PP_BAND))
a_record("A4", "Original-cosponsor share of each candidate within train spread", [("", t4, i4)],
         "A rising original share means fewer post-introduction positives and a denser day-0 graph; implies a "
         "sliding window (recent Congresses) or explicit Congress-level time features. Section C should reference "
         "this check rather than re-test it.", quantity="orig_share", band_label="+/-5 pp of pooled train")

# %% [markdown]
# ### A5. Concept stability: majority vs minority sponsor advancement, overall and by regime
# **Why it matters:** majority-party sponsorship is one of the strongest Task-2 signals. If its effect
# (odds ratio, OR) changes between train and the held-out Congress, the learned relationship is wrong
# there. The effect may depend on **which party controls the chamber**. If candidates match train
# Congresses with the same controlling party, the remedy is a regime feature or interaction, not a
# different split. **How to read it:** A5 compares the pooled-chamber OR per Congress with train
# Congresses. A5b compares each candidate chamber-Congress with the train chamber-Congresses that have
# the same chamber and controlling party (random-effects z: sampling variance plus the between-Congress
# variance seen in train). A5c tests in train whether D-control and R-control ORs differ. Majority from
# `majority_party()`; sponsors are assigned to their caucus.

# %%
# A5: majority-sponsor OR per Congress (pooled) and per chamber x controlling party; regime-matched z-scores.
def a_cnt(df):
    """2x2 counts (majority pos, majority neg, minority pos, minority neg) + 0.5 continuity."""
    m, n = df[df.sponsor_status == "majority"].label, df[df.sponsor_status == "minority"].label
    return np.array([m.sum(), (1 - m).sum(), n.sum(), (1 - n).sum()], float) + 0.5


def a_lor(k):
    return float(np.log(k[0] * k[3] / (k[1] * k[2]))), float((1 / k).sum())


r5 = adv_u.groupby(["congress", "sponsor_status"])["label"].mean().unstack().add_prefix("rate_")
r5["OR"] = [np.exp(a_lor(a_cnt(g))[0]) for _, g in adv_u.groupby("congress")]
print(r5.round(3).T.to_string())
t5, i5 = a_rel(r5.OR, log=True)
a_record("A5", "Majority-sponsor odds ratio (both chambers pooled) of each candidate within train spread", [("", t5, i5)],
         "Pooled over chambers and regimes; see A5b for the regime-matched comparison.", quantity="majority_or", fmt=".2f")

cells = pd.DataFrame([dict(congress=c, ch=ch, majority=g.majority.iloc[0], k=a_cnt(g)) for (c, ch), g in
                      adv_u.groupby(["congress", "ch"])])
cells[["l", "v"]] = [a_lor(k) for k in cells.k]
cells["regime"] = cells.ch + "-" + cells.majority
ctr = cells[cells.congress.isin(A_TRAIN)]


def a_pool(df):
    return a_lor(np.sum(np.stack(df.k.values), axis=0) - 0.5 * (len(df) - 1))


def a_tau2(key):
    """Between-Congress excess variance of train cell log ORs around their `key` group's pooled log OR."""
    res = [(r.l - a_pool(ctr[ctr[key] == getattr(r, key)])[0]) ** 2 - r.v for r in ctr.itertuples()]
    return max(0.0, float(np.mean(res)))


tau_reg, tau_ch = a_tau2("regime"), a_tau2("ch")
rows = []
for r in cells.itertuples():
    same = ctr[(ctr.regime == r.regime) & (ctr.congress != r.congress)]
    chm = ctr[(ctr.ch == r.ch) & (ctr.congress != r.congress)]
    lr, vr = a_pool(same) if len(same) else (np.nan, np.nan)
    lc, vc = a_pool(chm)
    rows.append(dict(congress=r.congress, ch=r.ch, regime=r.regime, OR=np.exp(r.l), n_ref=len(same),
                     OR_regime_ref=np.exp(lr), z_regime=(r.l - lr) / np.sqrt(r.v + vr + tau_reg),
                     OR_chamber_ref=np.exp(lc), z_chamber=(r.l - lc) / np.sqrt(r.v + vc + tau_ch)))
a5b = pd.DataFrame(rows)
print(f"between-Congress tau^2 of log OR: within regime {tau_reg:.3f}, within chamber (regime ignored) {tau_ch:.3f}; "
      "candidate cells below (train cells are in the chart)")
print(a5b[a5b.congress.isin(A_CANDS)].round(2).to_string(index=False))
A_ZC5 = float(stats.norm.ppf(1 - A_ALPHA / 2))   # random-effects z is approximately normal
a5b["p"] = 2 * stats.norm.sf(a5b.z_regime.abs())
a5b["fail"] = a5b.z_regime.abs() > A_ZC5
a5b["holm"] = False
_ic = a5b.congress.isin(A_CANDS)
a5b.loc[_ic, "holm"] = a_holm(a5b.loc[_ic, "p"].values) & a5b.loc[_ic, "fail"].values   # Holm over the 6 candidate cells
cz = a5b[_ic]
rz = a5b[a5b.congress.isin(A_TRAIN)]
for r in a5b.itertuples():   # one part per chamber, so per-Congress failure counts line up with the other checks
    if r.congress in A_CANDS:
        A_ZTAB.append(dict(check="A5b", part=f"A5b {r.ch}", congress=int(r.congress), t=float(r.z_regime), p=float(r.p),
                           fail=bool(r.fail), trend_z=np.nan, trend_p=np.nan, trend_fail=False, kind="", level=False,
                           fa_rate=float(rz.fail.mean()), n=len(rz), holm=bool(r.holm), jholm=False,
                           sig=bool(r.holm)))
    elif r.congress in A_TRAIN:
        A_REFTAB.append(dict(check="A5b", part=f"A5b {r.ch}", congress=int(r.congress), fail=bool(r.fail),
                             trend_fail=False, kind="", level=False))
any5 = rz.groupby("congress").fail.any()
A_FAM.append(dict(check="A5b", parts=2, n_ref=len(any5), fwer_loo=float(any5.mean()),
                  fwer_nominal=float(1 - (1 - A_ALPHA) ** 2), fwer_ids=[int(c) for c in any5.index[any5]]))
fails5 = [f"{r.regime} {r.congress}" for r in cz.itertuples() if r.fail]
surv5 = [f"{r.regime} {r.congress}" for r in cz.itertuples() if r.holm]
record_check("A", "A5b", "Each candidate chamber-Congress's majority-sponsor OR matches train Congresses of the same chamber x controlling party",
             not surv5, metric=float(cz.z_regime.abs().max()),
             threshold=f"|z_regime| > {A_ZC5:.2f} (random-effects, 95% two-sided) AND survives Holm over {len(cz)} candidate cells",
             detail=(f"VERDICT RULE: FAIL iff a candidate cell survives Holm. Holm survivors: {surv5 or 'none'}; raw failures "
                     f"{fails5 or 'none'}. z_regime per candidate cell {dict(zip(cz.regime + ' ' + cz.congress.astype(str), cz.z_regime.round(2)))}; "
                     f"mean |z| regime-matched {cz.z_regime.abs().mean():.2f} vs chamber-only {cz.z_chamber.abs().mean():.2f}. "
                     f"Nested-LOO false alarms among {len(rz)} train cells: {int(rz.fail.sum())}; train Congresses with ANY "
                     f"failing cell {int(any5.sum())}/{len(any5)}. If regime matching passes while "
                     f"A5 (pooled) fails, the shift is regime-conditional and already present in train: add a chamber x "
                     f"controlling-party feature / majority x party interaction rather than calling it concept drift. "
                     f"House-D has only {len(ctr[ctr.regime == 'House-D'])} train cells, so that reference is thin. No joint "
                     f"candidate test here: each cell has its own regime-matched reference. {A_S4_NOTE}; S5 not judged."),
             quantity="majority_or_regime", by_split=a_split_from(cz.groupby("congress").holm.any().to_dict()),
             kind="drift", na_reason=A_NA)

# A5c: in train, does the OR differ by controlling party within a chamber? (cell means, random-effects SE)
rows = []
for ch, g in ctr.groupby("ch"):
    d, r = g[g.majority == "D"], g[g.majority == "R"]
    se = np.sqrt(((d.v + tau_reg).sum() / len(d) ** 2) + ((r.v + tau_reg).sum() / len(r) ** 2))
    rows.append(dict(ch=ch, n_D=len(d), OR_Dcontrol=np.exp(d.l.mean()), n_R=len(r), OR_Rcontrol=np.exp(r.l.mean()),
                     diff_logOR=d.l.mean() - r.l.mean(), z=(d.l.mean() - r.l.mean()) / se))
a5c = pd.DataFrame(rows)
print(a5c.round(2).to_string(index=False))
record_check("A", "A5c", "Majority-sponsor effect differs by controlling party within train (regime interaction needed?)", None,
             metric=float(a5c.diff_logOR.abs().max()), threshold=f"|z| > {A_ZC5:.2f} => model needs a regime interaction",
             detail=(f"log OR difference D-control minus R-control per chamber {dict(zip(a5c.ch, a5c.diff_logOR.round(2)))}, "
                     f"z {dict(zip(a5c.ch, a5c.z.round(2)))}. Every regime of the candidates occurs in train "
                     f"({sorted(set(cz.regime))} vs train {sorted(set(ctr.regime))}), so a by-Congress split can learn it "
                     f"if the regime is a feature."), quantity="majority_or_regime_effect")
show_d3(r"""
const {width, height, title, subtitle, split, colors} = opts;
const m = {top: 48, right: 20, bottom: 48, left: 56};
const svg = d3.select(el).append('svg').attr('width', width).attr('height', height);
viz.title(svg, title, subtitle);
const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
const W = width - m.left - m.right, H = height - m.top - m.bottom;
const xs = d3.scaleLinear([d3.min(data, d => d.congress) - 0.6, d3.max(data, d => d.congress) + 0.6], [0, W]);
const ys = d3.scaleLog([d3.min(data, d => d.lo) * 0.9, d3.max(data, d => d.hi) * 1.1], [H, 0]);
g.append('g').call(d3.axisLeft(ys).ticks(6, '~g').tickSize(-W)).call(a => a.selectAll('line').attr('stroke', viz.grid))
  .call(a => a.select('.domain').remove()).call(a => a.selectAll('text').attr('fill', viz.ink2).style('font', '11px sans-serif'));
g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xs).ticks(12).tickFormat(d3.format('d'))).call(viz.axis);
g.append('line').attr('x1', xs(split)).attr('x2', xs(split)).attr('y1', 0).attr('y2', H).attr('stroke', viz.ink2).attr('stroke-dasharray', '4,3');
g.append('text').attr('x', xs(split) + 4).attr('y', 10).attr('fill', viz.ink2).style('font', '11px sans-serif').text('candidates');
g.append('text').attr('x', -m.left + 4).attr('y', -8).attr('fill', viz.ink2).style('font', '11px sans-serif').text('odds ratio (log scale)');
const off = d => d.ch === 'House' ? -0.15 : 0.15;
const tip = viz.tooltip(el);
g.selectAll(null).data(data).join('line').attr('x1', d => xs(d.congress + off(d))).attr('x2', d => xs(d.congress + off(d)))
  .attr('y1', d => ys(d.lo)).attr('y2', d => ys(d.hi)).attr('stroke', d => colors[d.majority]).attr('stroke-width', 1.5).attr('opacity', 0.6);
g.selectAll(null).data(data).join('path')
  .attr('d', d => d3.symbol(d.ch === 'House' ? d3.symbolCircle : d3.symbolSquare, 64)())
  .attr('transform', d => `translate(${xs(d.congress + off(d))},${ys(d.OR)})`)
  .attr('fill', d => colors[d.majority]).attr('stroke', '#fff').attr('stroke-width', 1)
  .on('mousemove', (e, d) => tip.show(`<b>${d.ch} ${d.congress}</b> (${d.majority} control)<br>OR ${d.OR.toFixed(2)} [${d.lo.toFixed(2)}, ${d.hi.toFixed(2)}]`, e))
  .on('mouseleave', () => tip.hide());
viz.legend(svg, [['D controls chamber', colors.D], ['R controls chamber', colors.R]], m.left, height - 10);
svg.append('text').attr('x', width - 20).attr('y', height - 1).attr('text-anchor', 'end').attr('fill', viz.ink2)
  .style('font', '11px sans-serif').text('circle = House, square = Senate; bars = 95% CI');
""", cells.assign(OR=np.exp(cells.l), lo=np.exp(cells.l - 1.96 * np.sqrt(cells.v)), hi=np.exp(cells.l + 1.96 * np.sqrt(cells.v)))
    [["congress", "ch", "majority", "OR", "lo", "hi"]], height=340, width=720,
    opts=dict(title="Majority-sponsor odds ratio of being reported, per chamber and Congress",
              subtitle="Colour = party controlling the chamber (majority_party()); compare candidates with same-colour train points",
              split=A_TRAIN[-1] + 0.5, colors=PARTY_COLORS), name="A_majority_or_regime")

# %% [markdown]
# ### A6. Concept stability: partisanship of new cosponsors (P(y | x) for Task 1)
# **Why it matters:** a link-prediction model will learn "same party as the sponsor" as a strong prior.
# If the share of new cosponsors from the sponsor's party drifts (polarisation), the learned prior is off
# in the held-out Congress. **How to read it:** among cosponsors joining in days 1-30 of law-capable bills
# with a D/R sponsor, the share from the same party and from the other major party.

# %%
# A6: same-party and cross-party share of new (lag 1..30 days) cosponsors per Congress; t PI vs train.
nc = cl[cl.lag_days.between(1, WINDOW_DAYS) & cl.sponsor_party.isin(["D", "R"])]
nc = nc.assign(same=nc.party == nc.sponsor_party, cross=nc.party.isin(["D", "R"]) & (nc.party != nc.sponsor_party))
a6 = nc.groupby("congress")[["same", "cross"]].mean()
a6["n_new_cosp"] = nc.groupby("congress").size()
tr6 = nc[nc.congress.isin(A_TRAIN)].same.mean()
print(a6.round(3).T.to_string())
t6, i6 = a_rel(a6.same, band=(tr6 - A_PP_BAND, tr6 + A_PP_BAND))
a_record("A6", "Same-party share of new cosponsors of each candidate within train spread", [("", t6, i6)],
         "Party homophily is the main prior a cosponsor model learns; a swing beyond train spread changes which "
         "candidates rank on top (recalibrate the party prior on recent Congresses).", quantity="same_party_share",
         band_label="+/-5 pp of pooled train")

# %% [markdown]
# ### A7-A8. Marginal covariate drift (PSI) and bill volume
# **Why it matters:** even with stable labels, a held-out Congress with a different mix of policy areas,
# bill types, chambers, sponsor status, original-cosponsor counts or introduction timing tests the model
# on a differently weighted population. **How to read it:** PSI of each candidate vs pooled 108-115
# bills; a train Congress's value is its PSI vs the other 7 train Congresses. The t prediction-interval rule
# decides, on the **log scale** (PSI is bounded at 0 and right-skewed, so a raw-scale interval has a
# negative lower bound and over-calls large values; section B judges PSI the same way) (the
# usual 0.1 / 0.25 PSI bands are shown in the chart subtitle only). The chart shows PSI divided by the
# largest train LOO PSI, so bars above 1 mean "more drift than any train Congress". `policy_area_harm`
# folds categories absent from some Congress (taxonomy changes) into one bucket (`PA_UNSTABLE`).

# %%
# A7: PSI per feature (candidates vs train pool; train Congresses LOO), t PI vs train; A8: bill volume.
A7_FEATS = {"policy_area_harm": "cat", "bill_type": "cat", "ch": "cat", "sponsor_status": "cat",
            "n_original": "num", "day_of_congress": "num"}


def a_psi(feat, ref, cur):
    return psi(ref[feat], cur[feat]) if A7_FEATS[feat] == "num" else psi_categorical(ref[feat], cur[feat])


trb = fb[fb.congress.isin(A_TRAIN)]
p7, bars = [], []
for f in A7_FEATS:
    s = pd.Series({c: a_psi(f, trb[trb.congress != c], trb[trb.congress == c]) for c in A_TRAIN}
                  | {c: a_psi(f, trb, fb[fb.congress == c]) for c in A_CANDS})
    p7.append((f,) + a_rel(s, side="upper", log=True))   # PSI >= 0 and right-skewed: judge on the log scale (as B3)
    bars += [dict(feature=f, congress=str(c), ratio=s[c] / s[A_TRAIN].max()) for c in A_CANDS]
raw = {c: round(psi_categorical(trb.policy_area.fillna("(missing)"), fb[fb.congress == c].policy_area.fillna("(missing)")), 3)
       for c in A_CANDS}
print(f"raw (unharmonised) policy_area PSI vs train, inflated by taxonomy changes: {raw}; folded: {PA_UNSTABLE}")
a_record("A7", "Marginal covariate drift (PSI of 6 bill features) of each candidate within train LOO spread", p7,
         "One part per feature, log-scale PI. FAIL on a feature means the candidate's mix differs more than any typical train "
         "Congress differs from the rest: reweight training bills toward recent Congresses, or use a sliding window.",
         quantity="covariate_psi", level=False)
bar_chart(pd.DataFrame(bars), "feature", "ratio", series="congress",
          title="Covariate drift of each candidate relative to train noise (PSI / max train LOO PSI)",
          subtitle="> 1 = the candidate drifts more than any train Congress does from the other train Congresses",
          yfmt=".2f", ylabel="PSI ratio", name="A_psi_features")
vol = fb.groupby("congress").size()
t8, i8 = a_rel(vol, log=True, band=(vol[A_TRAIN].mean() * A_RATIO_BAND[0], vol[A_TRAIN].mean() * A_RATIO_BAND[1]))
a_record("A8", "Law-capable bill volume of each candidate within train spread", [("", t8, i8)],
         "More bills per Congress means more bill nodes per legislator (higher degree) and lower per-bill base "
         "rates; normalise degree/count features per Congress. Section B's degree checks cover the graph side.",
         quantity="bill_volume", fmt=".0f", band_label="0.8-1.25x mean train Congress")

# %% [markdown]
# ### A9-A10. Congress x Congress drift matrix (policy-area mix)
# **Why it matters:** the heatmap shows which Congresses look alike. If distance grows with the time
# gap, drift is gradual: the most recent Congresses are the best training data and an expanding window
# dilutes them. **How to read it:** JSD (base 2, 0 = identical) of the harmonised policy-area
# distribution. A9 compares each candidate's mean JSD to the train Congresses with each train Congress's
# mean JSD to the other train Congresses (same rule as PSI, so A7 and A9 agree in method).

# %%
# A9: pairwise JSD of policy-area mix; candidate-to-train vs within-train (LOO) mean distances. A10: gradualness.
pa = fb.groupby(["congress", "policy_area_harm"]).size().unstack(fill_value=0)
J = pd.DataFrame([[jsd(pa.loc[i], pa.loc[j]) for j in pa.index] for i in pa.index], index=pa.index, columns=pa.index)
hm = J.stack().rename("jsd").reset_index(level=1).rename(columns={"congress": "vs"}).reset_index()
hm.columns = ["Congress", "vs", "jsd"]
heatmap(hm, "vs", "Congress", "jsd", title="Policy-area mix divergence between Congresses (JSD)",
        subtitle="Law-capable bills, harmonised policy areas; 0 = identical mix. Darker = more different", fmt=".3f",
        height=420, width=640, name="A_jsd_heatmap")
mj = pd.Series({c: J.loc[c, [t for t in A_TRAIN if t != c]].mean() for c in A_TRAIN}
               | {c: J.loc[c, A_TRAIN].mean() for c in A_CANDS})
t9, i9 = a_rel(mj, side="upper")
gap = np.array([abs(i - j) for i in J.index for j in J.columns])
rho = stats.spearmanr(gap[gap > 0], J.values.ravel()[gap > 0]).statistic
nearest = {c: int(J.loc[c].drop(c).idxmin()) for c in A_CANDS}
a_record("A9", "Each candidate's mean policy-area JSD to train within the train LOO spread", [("", t9, i9)],
         f"Nearest Congress of each candidate: {nearest}.", quantity="topic_mix_jsd", fmt=".4f", level=False)
record_check("A", "A10", "Drift is gradual in time (Spearman of Congress gap vs policy-area JSD)", None,
             metric=float(rho), threshold="rho > 0.5 => gradual drift",
             detail=(f"rho={rho:.2f}. Strongly positive rho means distant Congresses differ most, so recent Congresses "
                     f"are better training data (favours sliding window S2 / rolling origin S3)."), quantity="drift_gradual")

# %% [markdown]
# ### A11. Joint covariate shift: can a classifier tell a candidate's bills from train bills?
# **Why it matters:** each marginal PSI can look fine while the joint distribution shifts (e.g. the same
# policy areas, but sponsored by a different party). A **domain classifier** (logistic regression on the
# bill features: chamber, type, sponsor caucus and majority status, harmonised policy area,
# introduction-day bin, log original cosponsors) is trained to separate train-pool bills from one
# Congress's bills; its 2-fold cross-validated AUROC is 0.5 when the two are indistinguishable. Policy area
# is **not** known at t0 for most fresh bills (`PA_MISS_T0`), so a second part drops it (the t0-safe
# feature set). **How to read it:** candidates are compared with train Congresses classified against the
# other 7 (LOO). The top coefficients say which features drive the separation.

# %%
# A11: domain-classifier AUROC (pool vs one Congress), with and without policy area; candidates vs train LOO.
_dom_cache = {}


def a_dom_rows(ip, ic, cols=None, seed=0, ratio=2):
    """2-fold CV AUROC (and full-data coefficients) of a balanced logistic classifier: A_X rows ip (label 0,
    subsampled to ratio x len(ic)) vs rows ic (label 1). cols: boolean column mask of A_X (None = all)."""
    rng = np.random.default_rng(seed)
    ip = rng.choice(ip, min(len(ip), ratio * len(ic)), replace=False)
    idx = np.concatenate([ip, ic])
    X = A_X[idx] if cols is None else A_X[idx][:, cols]
    y = np.r_[np.zeros(len(ip)), np.ones(len(ic))]
    w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    fold = rng.integers(0, 2, len(idx))
    s = np.empty(len(idx))
    for k in (0, 1):
        b = a_logit(X[fold != k], y[fold != k], w[fold != k])
        s[fold == k] = X[fold == k] @ b
    return a_auc(y, s), a_logit(X, y, w)


def a_domain(pool, c, cols=None, max_day=None):
    """Cached domain AUROC: bills of `pool` Congresses vs Congress c (optionally only bills introduced by day
    max_day of their Congress)."""
    key = (tuple(sorted(pool)), c, None if cols is None else tuple(cols), max_day)
    if key not in _dom_cache:
        ok = np.ones(len(fb_all), bool) if max_day is None else (fb_all.day_of_congress <= max_day).values
        cg = fb_all.congress.values
        _dom_cache[key] = a_dom_rows(np.flatnonzero(np.isin(cg, pool) & ok), np.flatnonzero((cg == c) & ok), cols, seed=c)
    return _dom_cache[key]


dom = pd.DataFrame({tag: {c: a_domain([t for t in A_TRAIN if t != c] if c in A_TRAIN else A_TRAIN, c, cols)[0]
                          for c in CONGRESSES} for tag, cols in [("all", None), ("noPA", A_NOPA)]})
print("domain AUROC (train: vs other train Congresses; candidates: vs 108-115):")
print(dom.T.round(3).to_string())
_tops = []
for c in A_CANDS:
    b = pd.Series(a_domain(A_TRAIN, c)[1][1:], index=A_XNAMES[1:])
    top = b.reindex(b.abs().sort_values(ascending=False).index[:2])
    _tops.append(f"{c}: " + ", ".join(f"{k} {v:+.2f}" for k, v in top.items()))
print("top separating features (log-odds of being the candidate): " + " | ".join(_tops))
a_record("A11", "Joint covariate shift (domain-classifier AUROC vs train pool) of each candidate within train LOO spread",
         [("all",) + a_rel(dom["all"], side="upper"), ("noPA",) + a_rel(dom["noPA"], side="upper")],
         f"Proxy for multivariate P(x) shift on bill features (graph features not included). Part 'all' includes policy "
         f"area, which is missing for {PA_MISS_T0:.0%} of bills <= 7 days old (PA_MISS_T0, n={PA_MISS_T0_N}; small-sample "
         f"proxy), so only 'noPA' is the t0-safe view. FAIL: the candidate occupies regions of feature space that train "
         f"covers thinly (the deployment Congress will too): prefer recency-weighted training (A14) or importance "
         f"weighting by the domain classifier's odds, and recalibrate on val. The features include sponsor caucus and "
         f"majority status, whose mix moves with party control (A15b), so part of this separation is the regime "
         f"itself; regime is better used as a feature than as a weighting scheme (section D's D26 backtests regime weighting).",
         quantity="joint_covariate_shift", level=False)

# %% [markdown]
# ### A12-A13. Calibration and concept stability of a fitted Task-2 model
# **Why it matters:** marginal ORs are a partial view. A logistic model using the same bill features as A11
# is fitted on earlier Congresses and scored on a later one. The main comparison is **forward-chained**
# (one step ahead, like split S3): Congress c is scored by a model fitted on 108..c-1, for c = 110-115
# (train reference) and 116-118 (candidates). A leave-one-out variant (train Congress scored by a model
# fitted on the other 7, which is interpolation and so easier) is shown for context. **How to read it:**
# `OE` = observed / mean predicted rate (calibration-in-the-large; this mostly tracks A1's base rate).
# `slope` = calibration slope: 1 means relative risks transfer, < 1 means the model is over-confident (the
# feature effects weakened). `coef_shift` = majority-status log-odds effect refitted on Congress c alone,
# minus the effect in the fitted model. A12 checks slope and coefficient shift; A13 checks AUROC with and
# without policy area (`auroc_noPA` = t0-safe features only, since policy area is mostly missing at t0).

# %%
# A12/A13: logistic Task-2 model, forward-chained (fit 108..c-1) and LOO; O/E, calibration slope, AUROC, coef shift.
ia_rows = adv_u.bill_id.map(A_ROW).values
Xa, ya, ca = A_X[ia_rows], adv_u.label.values.astype(float), adv_u.congress.values
j_min = A_XNAMES.index("sponsor_status=minority")
_bc = {c: a_logit(Xa[ca == c], ya[ca == c]) for c in CONGRESSES}   # per-Congress refit (for coef_shift)


def a_transfer(fit_on, c, cols=None):
    X = Xa if cols is None else Xa[:, cols]
    b = a_logit(X[np.isin(ca, fit_on)], ya[np.isin(ca, fit_on)])
    m = ca == c
    eta = X[m] @ b
    if cols is not None:
        return dict(auroc=a_auc(ya[m], eta))
    cal = a_logit(np.column_stack([np.ones(m.sum()), eta]), ya[m], l2=0.0)
    return dict(OE=ya[m].mean() / (1 / (1 + np.exp(-eta))).mean(), slope=cal[1], auroc=a_auc(ya[m], eta),
                coef_shift=-_bc[c][j_min] + b[j_min])


A_FWD = [c for c in A_TRAIN if c >= CONGRESSES[0] + 2]   # need >= 2 earlier Congresses to fit on
fwd = pd.DataFrame({c: a_transfer(list(range(CONGRESSES[0], c)), c) for c in A_FWD + A_CANDS}).T
fwd["auroc_noPA"] = pd.Series({c: a_transfer(list(range(CONGRESSES[0], c)), c, A_NOPA)["auroc"] for c in A_FWD + A_CANDS})
loo = pd.DataFrame({c: a_transfer([t for t in A_TRAIN if t != c] if c in A_TRAIN else A_TRAIN, c) for c in CONGRESSES}).T
print(fwd.join(loo[["slope", "auroc"]].add_suffix("_loo"), how="outer").dropna(subset=["auroc"]).round(3).to_string())
a_record("A12", "Fitted Task-2 model transfers forward: calibration slope and majority-coefficient shift within train spread",
         [("slope",) + a_rel(fwd.slope, train=A_FWD), ("coef_shift",) + a_rel(fwd.coef_shift, train=A_FWD)],
         f"Forward-chained reference {A_FWD[0]}-{A_FWD[-1]} (fit 108..c-1; n={len(A_FWD)}, so the PI is wide). Model "
         f"includes policy area (not known at t0 for most bills). O/E {a_d(fwd.loc[A_CANDS, 'OE'], 2)} vs "
         f"train {fwd.loc[A_FWD, 'OE'].min():.2f}-{fwd.loc[A_FWD, 'OE'].max():.2f}: an O/E shift alone is fixed by "
         f"recalibrating the intercept on val; a slope below train spread means feature effects weakened (concept "
         f"drift): train on recent Congresses, add regime interactions (A5b/A5c), and recalibrate slope on val. LOO "
         f"slopes (interpolation) for context: {a_d(loo.slope, 2)}.", quantity="model_calibration_t2")
a_record("A13", "Fitted Task-2 model AUROC (forward-chained) on each candidate not below the train spread",
         [("auroc",) + a_rel(fwd.auroc, side="lower", train=A_FWD),
          ("auroc_noPA",) + a_rel(fwd.auroc_noPA, side="lower", train=A_FWD)],
         f"Ranking quality of a simple feature model one Congress ahead (baseline backtests are in section D). Part "
         f"'auroc' uses policy area (optimistic at t0), 'auroc_noPA' only t0-safe features. LOO AUROC for context: "
         f"{a_d(loo.auroc, 3)}. FAIL means the held-out Congress is harder to rank than past one-step-ahead "
         f"Congresses were: expect lower test scores than validation-era scores, and do not read a val-to-test drop as "
         f"overfitting.", quantity="model_auroc_transfer_t2")

# %% [markdown]
# ### A18. Concept stability for Task 1: do pair-feature effects on cosponsoring stay the same?
# **Why it matters:** A6 looks at one marginal share; A12/A13 test P(y | x) only for Task 2. A link
# predictor for Task 1 learns how much each (bill, legislator) pair feature raises the chance of
# cosponsoring: past cosponsoring of this sponsor's bills, reciprocity (the sponsor cosponsored the
# legislator's bills), the legislator's general activity, same caucus, same state. If those effects moved,
# model selection on the val Congress can still work (ranking survives) while calibrated scores and the
# learned weights drift. **How to read it:** for each Congress, a logistic model on sampled (bill,
# eligible legislator) pairs at t0 = 7 (point-in-time pool incl. delegates; history counted in the 730
# days before introduction + 7 d). A18 judges each coefficient of 116-118 against 109-115 (108 is dropped
# from the reference because its look-back window starts before the data). A18b scores each Congress
# with a model fitted on all earlier Congresses (forward-chained, like A13): AUROC over pairs and the
# calibration slope.

# %%
# A18: per-Congress logistic model on (bill, eligible legislator) pairs at t0=7 with 5 time-aware pair features.
A18_T0, A18_LOOK, A18_NB, A18_NEG = 7, 730, 1500, 0.1   # t0, look-back days, bills sampled per Congress, negative keep rate
A18_REF = [c for c in A_TRAIN if c > CONGRESSES[0]]    # 109-115 (108's look-back is truncated at the data start)
A18_F = ["log1p(hist s->l)", "log1p(recip l->s)", "log1p(activity l)", "same_caucus", "same_state"]
_ids = pd.Index(pd.unique(pd.concat([cosp.bioguide_id, cosp.sponsor_bioguide_id, ROSTER_ASOF_ALL.bioguide_id]).dropna()))
_NL, _BIG = len(_ids), 100_000
_ev = cosp.dropna(subset=["sponsor_bioguide_id"])
_eday = ((_ev.sponsorship_date.values.astype("datetime64[D]") - _EPOCH).astype(np.int64))
_ls, _ss = _ids.get_indexer(_ev.bioguide_id), _ids.get_indexer(_ev.sponsor_bioguide_id)
_K_SL = np.sort((_ss.astype(np.int64) * _NL + _ls) * _BIG + _eday)   # (sponsor, cosponsor) events by day
_K_L = np.sort(_ls.astype(np.int64) * _BIG + _eday)                  # cosponsor activity by day


def a_count(sorted_keys, key, day):
    """Number of events with this key dated in [day - A18_LOOK, day)."""
    return (np.searchsorted(sorted_keys, key * _BIG + day, "left")
            - np.searchsorted(sorted_keys, key * _BIG + day - A18_LOOK, "left")).astype(float)


_wt7 = window_targets(A18_T0, point_in_time=True)
_wt7 = _wt7[_wt7.congress.isin(CONGRESSES) & ~_wt7.censored & _wt7.eligible.gt(0)].merge(
    bills[["bill_id", "sponsor_bioguide_id", "sponsor_state"]], on="bill_id").merge(fb[["bill_id", "sponsor_caucus"]], on="bill_id")
_pos = cosp[(cosp.lag_days > A18_T0) & (cosp.lag_days <= A18_T0 + WINDOW_DAYS)][["bill_id", "bioguide_id"]].assign(y=1.0)
_seen = cosp[cosp.lag_days <= A18_T0][["bill_id", "bioguide_id"]].assign(seen=True)
_rost = ROSTER_ASOF_ALL[["month", "congress", "chamber", "bioguide_id", "caucus", "state_code"]].rename(columns={"month": "pool_month"})
_rng18 = np.random.default_rng(18)
P18 = []
for c in CONGRESSES:
    b = _wt7[_wt7.congress == c]
    b = b.iloc[_rng18.choice(len(b), min(A18_NB, len(b)), replace=False)]
    pr = b.merge(_rost, on=["pool_month", "congress", "chamber"])
    pr = pr[pr.bioguide_id != pr.sponsor_bioguide_id].merge(_seen, on=["bill_id", "bioguide_id"], how="left")
    pr = pr[pr.seen.isna()].merge(_pos, on=["bill_id", "bioguide_id"], how="left")
    pr["y"] = pr.y.fillna(0.0)
    pr = pr[(pr.y == 1) | (_rng18.random(len(pr)) < A18_NEG)].copy()   # all positives, A18_NEG of negatives (weighted back)
    pr["w"] = np.where(pr.y == 1, 1.0, 1 / A18_NEG)
    day = ((pr.introduced_date.values.astype("datetime64[D]") - _EPOCH).astype(np.int64)) + A18_T0
    sc, lc = _ids.get_indexer(pr.sponsor_bioguide_id).astype(np.int64), _ids.get_indexer(pr.bioguide_id).astype(np.int64)
    pr["f0"] = np.log1p(a_count(_K_SL, sc * _NL + lc, day))
    pr["f1"] = np.log1p(a_count(_K_SL, lc * _NL + sc, day))
    pr["f2"] = np.log1p(a_count(_K_L, lc, day))
    pr["f3"] = (pr.caucus == pr.sponsor_caucus).astype(float)
    pr["f4"] = (pr.state_code == pr.sponsor_state).astype(float)
    P18.append(pr[["congress", "y", "w", "f0", "f1", "f2", "f3", "f4"]])
P18 = pd.concat(P18, ignore_index=True)
X18 = np.column_stack([np.ones(len(P18))] + [P18[f"f{k}"].values for k in range(len(A18_F))])
y18, w18, c18 = P18.y.values, P18.w.values, P18.congress.values
coef18 = pd.DataFrame({c: a_logit(X18[c18 == c], y18[c18 == c], w18[c18 == c])[1:] for c in CONGRESSES}, index=A18_F).T
print(f"pairs per Congress after negative sampling (positives all, negatives x{A18_NEG}): "
      f"{P18.groupby('congress').size().min():,}-{P18.groupby('congress').size().max():,}; positives "
      f"{int(P18.y.sum()):,}. Coefficients (log-odds per unit; per Congress fit):")
print(coef18.T.round(3).to_string())
a_record("A18", "Task-1 pair-feature effects (logistic coefficients, t0=7) of each candidate within the 109-115 spread",
         [(f,) + a_rel(coef18[f], train=A18_REF) for f in A18_F],
         f"Model: logistic on {A18_NB} sampled uncensored bills per Congress x all eligible point-in-time legislators "
         f"(ROSTER_ASOF_ALL, delegates included; sponsor and day-0..7 cosponsors removed), label = cosponsors in "
         f"({A18_T0}, {A18_T0 + WINDOW_DAYS}] days; history features counted over the {A18_LOOK} days before "
         f"introduction + {A18_T0} d (point in time). Reference 109-115 (n={len(A18_REF)}). FAIL on a coefficient = "
         f"the learned weight of that feature is stale for the candidate: refit / fine-tune on recent Congresses "
         f"(S2/S3 refit, or streaming) and recalibrate scores on val. A 1-hop/2-hop GNN learns similar signals, so "
         f"this is a proxy for the learned model's concept drift.", quantity="concept_drift_t1")

# A18b: forward-chained transfer (fit 108..c-1, score c): pair AUROC and calibration slope.
A18_FWD = [c for c in A_TRAIN if c >= CONGRESSES[0] + 2]
rows = {}
for c in A18_FWD + A_CANDS:
    fit = np.isin(c18, list(range(CONGRESSES[0], c)))
    bf = a_logit(X18[fit], y18[fit], w18[fit])
    m = c18 == c
    eta = X18[m] @ bf
    cal = a_logit(np.column_stack([np.ones(m.sum()), eta]), y18[m], w18[m], l2=0.0)
    rows[c] = dict(auroc=a_auc(y18[m], eta), slope=cal[1])
f18 = pd.DataFrame(rows).T
print("forward-chained (fit 108..c-1): " + " | ".join(f"{c}: AUROC {r.auroc:.3f}, slope {r.slope:.2f}" for c, r in f18.iterrows()))
a_record("A18b", "Task-1 pair model fitted on earlier Congresses: AUROC and calibration slope on each candidate within forward-chained spread",
         [("auroc",) + a_rel(f18.auroc, side="lower", train=A18_FWD), ("slope",) + a_rel(f18.slope, train=A18_FWD)],
         f"Forward-chained reference {A18_FWD[0]}-{A18_FWD[-1]} (n={len(A18_FWD)}). AUROC over sampled pairs (negative "
         f"sampling leaves AUROC unbiased). A pass on AUROC with a fail on A18 coefficients = ranking transfers, weights / "
         f"calibration do not: selection on val is fine, but recalibrate. A fail on AUROC = the held-out Congress is "
         f"harder to rank than past one-step-ahead Congresses.", quantity="model_auroc_transfer_t1")
line_chart(coef18.rename_axis("congress").reset_index().melt("congress", var_name="series", value_name="coef"),
           "congress", "coef", series="series", title="Task-1 pair-feature effects per Congress (logistic coefficients, t0=7)",
           subtitle=f"Fit per Congress on sampled (bill, eligible legislator) pairs; reference 109-115, candidates 116-118",
           yfmt=".2f", ylabel="log-odds coefficient", xlabel="Congress", yzero=False, name="A_task1_coefs")

# %% [markdown]
# ### A14. Which candidate split's training pool is closest to the Congress it evaluates?
# **Why it matters:** this compares the shared candidate splits (`SPLIT_DEFS`: S1a/S1b expanding, S2
# sliding k=`SLIDE_K`, S3 rolling-origin expanding) on the distribution distance between each evaluation
# unit's training Congresses and its evaluated Congress. A smaller distance means the evaluated Congress
# is better represented by that split's training data. **How to read it:** mean distance per split and
# role over its evaluation units (`iter_eval_units`). The chart shows domain-classifier AUROC per test
# fold for S2 and S3 (S1a/S1b units are a subset of S3's units). This is a proxy: closer distributions
# do not guarantee better scores (section D's backtest decides), and sliding pools have fewer examples.

# %%
# A14: distance between each eval unit's train Congresses and its eval Congress, for S1a/S1b/S2/S3.
d3t = window_targets(7, point_in_time=True)   # same point-in-time pool as A3 and section D
d3t = d3t[~d3t.censored & d3t.eligible.gt(0) & d3t.congress.isin(CONGRESSES)]
dens = d3t.groupby("congress")[["n_pos", "eligible"]].sum()
advk = adv_u.groupby("congress")["label"].agg(["sum", "size"])
rows = []
for s in ["S1a", "S1b", "S2", "S3"]:
    for u in iter_eval_units(s):
        pool, c = u["train"], u["eval_congress"]
        rows.append(dict(split=s, fold=u["fold"], role=u["role"], eval_congress=c, train=u["train_label"],
                         domain_auc=a_domain(pool, c)[0],
                         adv_logratio=abs(np.log((advk.loc[c, "sum"] / advk.loc[c, "size"]) /
                                                 (advk.loc[pool, "sum"].sum() / advk.loc[pool, "size"].sum()))),
                         dens_logratio=abs(np.log((dens.loc[c, "n_pos"] / dens.loc[c, "eligible"]) /
                                                  (dens.loc[pool, "n_pos"].sum() / dens.loc[pool, "eligible"].sum()))),
                         topic_jsd=jsd(pa.loc[pool].sum(), pa.loc[c])))
a14 = pd.DataFrame(rows)
A14_M = ["domain_auc", "adv_logratio", "dens_logratio", "topic_jsd"]
print(a14.groupby(["split", "role"])[A14_M].mean().round(4).to_string())
cmp = a14[a14.split.isin(["S2", "S3"]) & (a14.role == "test")].pivot(index="eval_congress", columns="split", values=A14_M)
wins = {m: int((cmp[m]["S2"] < cmp[m]["S3"]).sum()) for m in A14_M}
line_chart(a14[a14.split.isin(["S2", "S3"]) & (a14.role == "test")], "eval_congress", "domain_auc", series="split",
           title="How distinguishable is each test Congress from its training pool? (domain AUROC)",
           subtitle=f"S2 = sliding k={SLIDE_K} (train c-{SLIDE_K}..c-1), S3 = expanding (108..c-1); 0.5 = indistinguishable",
           yfmt=".3f", ylabel="domain AUROC", xlabel="test Congress", yzero=False, name="A_domain_auc_by_split")
share14 = float(sum(wins.values()) / (len(cmp) * len(wins)))
_d14 = a14[a14.role == "test"].groupby("split").domain_auc.mean()
by14 = {k: round(float(_d14[SPLIT_VARIANTS[k]["base"] if k in SPLIT_VARIANTS else k]), 4) for k in ALL_SPLITS
        if k in _d14.index or (k in SPLIT_VARIANTS and SPLIT_VARIANTS[k]["base"] in _d14.index and "test" not in SPLIT_VARIANTS[k])}
_pv = SPLIT_VARIANTS["S1pool"]   # S1pool: test refit on train + val, scored on each pooled test Congress
by14["S1pool"] = round(float(np.mean([a_domain(_pv["train"] + _pv["val"], c)[0] for c in _pv["test"]])), 4)
by14.update(S4=None, S5=None)
record_check("A", "A14", f"Sliding (S2, k={SLIDE_K}) training pools are closer to their test Congress than expanding (S3) pools",
             None, metric=share14, threshold="> 0.5 favours sliding window",
             detail=(f"S2 closer than S3 in {wins} of {len(cmp)} test folds per metric (domain AUROC, |log| advancement-rate "
                     f"ratio, |log| cosponsor-density ratio t0=7, topic JSD). by_split = mean test-unit domain AUROC (lower = "
                     f"closer; '+stream' variants share their base split's pools). Proxy only: distribution distance, not model performance; a {SLIDE_K}-Congress pool has "
                     f"fewer examples (section D's backtest decides)."),
             quantity="pool_distance", by_split=by14, kind="drift",
             na_reason={"S4": "within-Congress pool (prior Congresses + early part of c): a whole-Congress domain AUROC "
                              "does not apply", **A_NA})

# %% [markdown]
# ### A15. Does the test Congress look like the deployment target (the in-progress 119th)?
# **Why it matters:** a test score is only a good forecast of deployed performance if the test Congress
# resembles the Congress the model will be used on. 119 is in progress (excluded from every split), but its
# bills introduced so far can be compared with earlier Congresses. **How to read it:** to make the
# comparison fair, every Congress is restricted to bills introduced by the same day of its Congress as the
# latest 119th bill. The domain classifier uses t0-safe features only (no policy area, which young 119th
# bills mostly lack). The **primary** feature set also drops sponsor caucus and majority status: the share
# of bills sponsored by each party moves with the majority (table below), so those features make
# same-majority Congresses look alike by construction. The `with party` column keeps them for comparison.
# `119 vs c` = AUROC separating 119's bills from Congress c's (lower = more alike); `c vs c-1` shows how far
# consecutive Congresses usually are. PSIs per feature compare 119 with 116-118 and with 108-117. INFO only:
# it tells you which candidate test Congress (117 for S1a, 118 for S1b) is the better deployment proxy.

# %%
# A15: 119 (same day-of-Congress window) vs each complete Congress: domain AUROC (t0-safe, with / without sponsor-party
# features), sponsor-party mix per Congress, and A7-feature PSIs.
A15_DAY = int(fb_all.loc[fb_all.congress == IN_PROGRESS, "day_of_congress"].max())
fw = fb_all[fb_all.day_of_congress.between(0, A15_DAY)]
print(f"window: bills introduced in days 0-{A15_DAY} of each Congress; {IN_PROGRESS} has {int((fw.congress == IN_PROGRESS).sum()):,} "
      f"law-capable bills; policy area present for {fw[fw.congress == IN_PROGRESS].policy_area.notna().mean():.1%} of them "
      f"(118 same window: {fw[fw.congress == 118].policy_area.notna().mean():.1%})")
mix15 = fw.groupby("congress").agg(D_sponsored=("sponsor_caucus", lambda s: (s == "D").mean()),
                                   majority_sponsored=("sponsor_status", lambda s: (s == "majority").mean()))
mix15["regime"] = pd.Series(REGIME)
print("sponsor-party mix (same window): the D-sponsored share follows the majority, so caucus / majority-status features "
      "encode the regime")
print(mix15.T.to_string())
A15_P = "no party"   # primary feature set (see markdown)
d15 = pd.DataFrame({f"119 vs c ({fs})": {c: a_domain([c], IN_PROGRESS, A_FSETS[fs], A15_DAY)[0] for c in CONGRESSES}
                    for fs in (A15_P, "with party")})
d15[f"c vs c-1 ({A15_P})"] = pd.Series({c: a_domain([c - 1], c, A_FSETS[A15_P], A15_DAY)[0] for c in CONGRESSES[1:]})
_pool15 = [c for c in CONGRESSES if c < CONGRESSES[-1]]
d15_pool = {fs: a_domain(_pool15, IN_PROGRESS, A_FSETS[fs], A15_DAY)[0] for fs in (A15_P, "with party")}
_cons = d15[f"c vs c-1 ({A15_P})"].dropna()
print(d15.T.round(3).to_string())
print(f"119 vs pooled {CONGRESSES[0]}-{CONGRESSES[-2]}: " + ", ".join(f"{k} {v:.3f}" for k, v in d15_pool.items())
      + f"; consecutive-pair reference ({A15_P}): median {_cons.median():.3f}, max {_cons.max():.3f}")
r15 = []
for f in A7_FEATS:
    cons = [a_psi(f, fw[fw.congress == c - 1], fw[fw.congress == c]) for c in CONGRESSES[1:]]
    r15.append(dict(feature=f, consecutive_median=np.median(cons), consecutive_max=np.max(cons),
                    **{f"119_vs_{c}": a_psi(f, fw[fw.congress == c], fw[fw.congress == IN_PROGRESS]) for c in A_CANDS},
                    **{f"119_vs_{CONGRESSES[0]}-{CONGRESSES[-2]}": a_psi(f, fw[fw.congress.isin(CONGRESSES[:-1])],
                                                                         fw[fw.congress == IN_PROGRESS])}))
r15 = pd.DataFrame(r15).set_index("feature")
print("PSI of each feature, 119 vs ... (same day window; policy_area_harm inflated by young bills without a policy area):")
print(r15.round(3).to_string())
line_chart(d15.rename_axis("congress").reset_index().melt("congress", var_name="series", value_name="auroc").dropna(),
           "congress", "auroc", series="series",
           title=f"How distinguishable is the in-progress {IN_PROGRESS}th Congress from each earlier Congress?",
           subtitle=f"Domain AUROC on t0-safe bill features, bills from days 0-{A15_DAY}; 0.5 = indistinguishable; "
                    "'no party' drops sponsor caucus / majority status", yfmt=".3f", ylabel="domain AUROC",
           xlabel="Congress", yzero=False, name="A_deploy_119")
_p15 = d15[f"119 vs c ({A15_P})"].to_dict()
_p15w = d15["119 vs c (with party)"].to_dict()
by15 = a_split_metric(_p15)
best15 = min(A_CANDS, key=lambda c: _p15[c])
n_over = int((r15[f"119_vs_{CONGRESSES[-1]}"] > r15.consecutive_max).sum())
_rk = lambda d: ", ".join(f"{c}: {d[c]:.3f}" for c in sorted(d, key=d.get))
record_check("A", "A15", f"Candidate test Congresses resemble the deployment target ({IN_PROGRESS}, in progress)", None,
             metric=float(_p15[CONGRESSES[-1]]), threshold="lower domain AUROC vs 119 = better deployment proxy (INFO)",
             detail=(f"Domain AUROC of 119 vs each complete Congress (t0-safe features WITHOUT sponsor caucus / majority "
                     f"status, days 0-{A15_DAY}), most similar first: {_rk(_p15)}. With the sponsor-party features (partly "
                     f"measures party control by construction: the D-sponsored share is "
                     f"{a_d(mix15.D_sponsored, 3)}): {_rk(_p15w)}. Candidates ({A15_P}): "
                     + ", ".join(f"{c}: {_p15[c]:.3f}" for c in A_CANDS) + f"; vs pooled {CONGRESSES[0]}-{CONGRESSES[-2]} "
                     f"{d15_pool[A15_P]:.3f}; consecutive-Congress reference median {_cons.median():.3f} (max "
                     f"{_cons.max():.3f}). Closest candidate: {best15}; closest overall: {min(_p15, key=_p15.get)} "
                     f"({A15_P}) vs {min(_p15w, key=_p15w.get)} (with party). Features whose 119-vs-118 PSI exceeds the "
                     f"largest consecutive-pair PSI: {n_over}/{len(r15)}. by_split = mean {A15_P} domain AUROC of the split's "
                     f"test Congresses vs 119 (lower = test set closer to deployment; S4 = its folds 116-118; S5 not judged). "
                     f"Proxy: 119 is partial (bills introduced by day {A15_DAY}; later-session bills and outcomes unseen)."),
             quantity="deployment_proxy", by_split=by15, kind="drift", na_reason=A_NA)

# %% [markdown]
# ### A15b. Does similarity to 119 follow recency or party regime, once party-encoding features are removed?
# **Why it matters:** if Congresses with the **same House / Senate majorities** look alike in substance,
# regime is a useful feature; if similarity follows **recency**, recent data matter most. A classifier that
# sees sponsor caucus and majority status will find same-regime Congresses alike **by construction** (the
# sponsor-party mix moves with the majority), so the test is run three times: `no party` (chamber, bill
# type, day bin, original cosponsors), `majority only` (adds the caucus-agnostic 'sponsor in majority'
# flag) and `with party` (adds sponsor caucus; the earlier version). **How to read it:** every pair of
# Congresses (108-119, same day window) gets a domain AUROC (heatmap = `no party`). A regression over pairs,
# AUROC ~ gap + [gap = 1] + same_regime, separates time distance (the dummy absorbs the consecutive pairs,
# several of which share a regime; the count is printed) from sharing the regime; its one-sided p-value comes from permuting the
# regime labels across Congresses. Regimes = shared `REGIME` from `majority_party()`. Whatever the result,
# section D's D26 is the outcome-based test of regime weighting; this is a bill-feature proxy.

# %%
# A15b: pairwise domain AUROC (days 0..A15_DAY) for 108-119 under 3 t0-safe feature sets;
# AUROC ~ 1 + gap + [gap=1] + same_regime, permutation p of the same_regime coefficient.
A15_C = CONGRESSES + [IN_PROGRESS]
A_REGIME = pd.Series({c: REGIME[c] for c in A15_C})
pairs15 = pd.DataFrame([dict(a=i, b=j, gap=j - i) for k, i in enumerate(A15_C) for j in A15_C[k + 1:]])
for fs, cols in A_FSETS.items():
    pairs15[fs] = [a_domain([r.a], r.b, cols, A15_DAY)[0] for r in pairs15.itertuples()]
A15_NPERM = 2000
_X15 = np.column_stack([np.ones(len(pairs15)), pairs15.gap.values, (pairs15.gap.values == 1).astype(float)])


def a_regime_fit(reg, y):
    """OLS of pair AUROC on [1, gap, gap==1, same_regime] -> coefficients (same_regime last)."""
    same = (reg.reindex(pairs15.a).values == reg.reindex(pairs15.b).values).astype(float)
    return np.linalg.lstsq(np.column_stack([_X15, same]), y, rcond=None)[0]


_rng15, _rng15g = np.random.default_rng(15), np.random.default_rng(151)
_perms = [pd.Series(_rng15.permutation(A_REGIME.values), index=A_REGIME.index) for _ in range(A15_NPERM)]
_ia, _ib = (np.searchsorted(A15_C, pairs15[k].values) for k in ("a", "b"))
_same15 = (A_REGIME.values[_ia] == A_REGIME.values[_ib]).astype(float)
_Xgap = []
for _ in range(A15_NPERM):   # design matrices with permuted time positions (regimes stay with their Congress)
    _pos = _rng15g.permutation(len(A15_C)).astype(float)
    _g = np.abs(_pos[_ib] - _pos[_ia])
    _Xgap.append(np.column_stack([np.ones(len(_g)), _g, (_g == 1).astype(float), _same15]))
same119 = [c for c in CONGRESSES if A_REGIME[c] == A_REGIME[IN_PROGRESS]]
oth119 = [c for c in CONGRESSES if c not in same119]
reg15 = {}
for fs in A_FSETS:
    y = pairs15[fs].values
    b = a_regime_fit(A_REGIME, y)
    perm = np.array([a_regime_fit(pr, y)[-1] for pr in _perms])
    p119 = {c: a_domain([c], IN_PROGRESS, A_FSETS[fs], A15_DAY)[0] for c in CONGRESSES}
    # Recency test: permute the Congresses' time positions (regimes stay attached to their Congress) -> gap coefficient.
    gperm = [np.linalg.lstsq(Xg, y, rcond=None)[0][1] for Xg in _Xgap]
    reg15[fs] = dict(b_same=b[-1], p_perm=float((np.sum(perm <= b[-1]) + 1) / (A15_NPERM + 1)), b_gap=b[1],
                     p_gap=float((np.sum(np.array(gperm) >= b[1]) + 1) / (A15_NPERM + 1)), b_gap1=b[2],
                     auroc_119_same=float(np.mean([p119[c] for c in same119])),
                     auroc_119_other=float(np.mean([p119[c] for c in oth119])),
                     closest_to_119=int(min(p119, key=p119.get)))
reg15 = pd.DataFrame(reg15).T.astype({"closest_to_119": int})
print(f"pairs regression AUROC ~ 1 + gap + [gap=1] + same_regime (n={len(pairs15)} pairs; {A15_NPERM} permutations): "
      f"p_perm = one-sided p that sharing the regime LOWERS the AUROC (regime labels permuted); p_gap = one-sided p that "
      f"distance in time RAISES it (Congress time positions permuted); consecutive pairs sharing a regime: "
      f"{int(_same15[pairs15.gap.values == 1].sum())}/{int((pairs15.gap == 1).sum())}")
print(reg15.to_string(float_format=lambda v: f"{v:.4f}"))
hm15 = pd.concat([pairs15.rename(columns={"a": "x", "b": "y"}), pairs15.rename(columns={"a": "y", "b": "x"})])
hm15 = hm15.assign(x=hm15.x.astype(str) + " " + hm15.x.map(A_REGIME), y=hm15.y.astype(str) + " " + hm15.y.map(A_REGIME))
heatmap(hm15, "x", "y", A15_P, title="How distinguishable are two Congresses' bills? (pairwise domain AUROC)",
        subtitle=f"t0-safe features without sponsor caucus / majority status, bills from days 0-{A15_DAY}; "
                 "labels = Congress and House/Senate majority; 0.5 = alike", fmt=".2f", height=520, width=680,
        name="A_pairwise_regime_heatmap")
_r = reg15.loc[A15_P]
_rw = reg15.loc["with party"]
reg_np = bool(_r.b_same < 0 and _r.p_perm < A_ALPHA)
cand_match = [c for c in A_CANDS if A_REGIME[c] == A_REGIME[IN_PROGRESS]]
record_check("A", "A15b", f"Similarity between Congresses follows party regime after removing sponsor-party features "
                          f"(deployment target {IN_PROGRESS})", None,
             metric=float(_r.b_same),
             threshold=f"no-party same-regime coefficient < 0 with permutation p < {A_ALPHA} => regime is a real similarity axis (INFO)",
             detail=(f"Pairs regression AUROC ~ 1 + gap + [gap=1] + same_regime over {len(pairs15)} pairs of 108-119 "
                     f"(days 0-{A15_DAY}, t0-safe). same_regime coefficient (one-sided permutation p, {A15_NPERM} regime "
                     f"permutations): " + "; ".join(f"{fs}: {r.b_same:+.4f} (p={r.p_perm:.3f}; gap {r.b_gap:+.4f}/Congress p={r.p_gap:.3f}, "
                                                   f"[gap=1] {r.b_gap1:+.4f}; 119 vs same-regime {r.auroc_119_same:.3f} vs "
                                                   f"other {r.auroc_119_other:.3f}; closest {int(r.closest_to_119)})"
                                                   for fs, r in reg15.iterrows())
                     + f". The 'with party' version partly measures party control BY CONSTRUCTION: sponsor_status = "
                       f"(sponsor caucus == majority) and the D-sponsored share moves with the majority "
                       f"({a_d(mix15.D_sponsored, 3)}), so a classifier separating sponsor-party mixes finds "
                       f"same-regime Congresses alike. The no-party coefficient is the substantive test: "
                       + ("regime remains a similarity axis without party features"
                          if reg_np else "the regime effect does NOT survive without party features, so 'bills of 119 "
                                         "resemble same-regime Congresses' is not supported beyond the sponsor-party mix")
                     + f" (with party: {_rw.b_same:+.4f}, p={_rw.p_perm:.3f}). Candidate test Congresses sharing 119's "
                       f"regime: {cand_match or 'none'}. Remedy: use regime / majority status as a FEATURE (it is known at "
                       f"t0 and the model can learn regime-conditional effects, A5b/A5c); do not weight training Congresses "
                       f"by regime unless section D's D26 shows a gain over recency weighting. Proxy: bill-feature "
                       f"similarity only (no outcomes), 12 Congresses, regimes confounded with era."),
             quantity="regime_similarity", kind="drift")

# %% [markdown]
# ### Section A summary: multiplicity, trend vs break, and what survives
# **Why it matters:** section A runs dozens of check parts per candidate. At a 5% level, about one part in
# twenty fails by chance even if nothing changed, and the parts are correlated. Every check above is
# already decided by its **Holm survivors** (cells plus joint candidate tests). A17 then asks whether a
# candidate fails more than chance predicts. Because several checks track the same underlying quantity
# (A1 / A1c / A1d / A2 all see the Task-2 rate drop), A17 counts **clusters of checks** (one per underlying
# quantity family), each with a false-alarm rate of 1 - 0.95^m for m member checks, and compares the count
# with a Poisson-binomial null. A drift FAIL here means "the later Congresses, and so the deployment
# target, differ from history; the forward split measures it; apply the remedy", **not** "this split is
# invalid". The trend-vs-break table labels every failing level part and the remedy each implies. The
# heatmap shows the standardised distance `t` of every check part.

# %%
# Section A summary: per-candidate failing clusters vs chance (A17), raw part counts, trend vs break (A16); t heatmap;
# judged drift checks per split (S4 and the by-Congress splits must be judged on the same checks).
sa = pd.DataFrame([c for c in CHECKS if c["section"] == "A"])
zt = pd.DataFrame(A_ZTAB).drop_duplicates(["part", "congress"], keep="last").reset_index(drop=True)
rt = pd.DataFrame(A_REFTAB).drop_duplicates(["part", "congress"], keep="last")
ref_share = rt.groupby("congress").fail.mean()
A_CLUSTERS = {"Task-2 base rate": ["A1", "A1c", "A1d", "A2"], "Task-2 label validity": ["A1b"],
              "Task-2 concept / model": ["A5", "A5b", "A12", "A13"], "Task-1 label": ["A3", "A3b", "A4"],
              "Task-1 concept / model": ["A6", "A18", "A18b"], "Covariates": ["A7", "A8", "A9", "A11"]}
_cl_of = {k: cl for cl, ks in A_CLUSTERS.items() for k in ks}
_unc = sorted(set(zt.check) - set(_cl_of))
for k in _unc:   # any check not listed forms its own cluster
    A_CLUSTERS[k] = [k]
    _cl_of[k] = k
zt["cluster"] = zt.check.map(_cl_of)
cl_rate = {cl: 1 - (1 - A_ALPHA) ** sum(k in set(zt.check) for k in ks) for cl, ks in A_CLUSTERS.items()}
cl_rate = {cl: r for cl, r in cl_rate.items() if r > 0}


def a_pbinom_sf(k, rates):
    """P(X >= k) for X = sum of independent Bernoulli(rates) (Poisson-binomial, exact by convolution)."""
    pmf = np.array([1.0])
    for r in rates:
        pmf = np.convolve(pmf, [1 - r, r])
    return float(pmf[k:].sum())


n_checks = zt.check.nunique()
mult = []
for c in A_CANDS:
    g = zt[zt.congress == c]
    kc = g.groupby("check").sig.any()
    kcl = g.groupby("cluster").sig.any()
    k, n = int(g.fail.sum()), len(g)
    mult.append(dict(congress=c, parts=n, raw_failing=k, ref_share_max=float(ref_share.max()),
                     checks_failing=f"{int(kc.sum())}/{n_checks}",
                     check_binom_p=float(stats.binom.sf(int(kc.sum()) - 1, n_checks, A_ALPHA)),
                     clusters_failing=f"{int(kcl.sum())}/{len(cl_rate)}",
                     expected=round(sum(cl_rate.values()), 2), cluster_p=a_pbinom_sf(int(kcl.sum()), list(cl_rate.values())),
                     failing_clusters=", ".join(kcl.index[kcl]) or "none"))
mult = pd.DataFrame(mult).set_index("congress")
print(f"Section A: {(sa.passed == True).sum()} pass, {(sa.passed == False).sum()} fail, {sa.passed.isna().sum()} info "   # noqa: E712
      f"(verdicts = Holm survivors within each check, cells + joint tests). Per candidate: raw failing parts, failing checks "
      f"(binomial, correlated: optimistic) and failing clusters (Poisson-binomial; decides A17):")
print(mult.round(3).to_string())
A_CAND_FAILS = {c: zt[(zt.congress == c) & zt.fail].part.tolist() for c in A_CANDS}
A_SURVIVE = {c: zt[(zt.congress == c) & zt.sig].part.tolist() for c in A_CANDS}
print("Holm survivors (cell or joint): " + " | ".join(f"{c}: {', '.join(A_SURVIVE[c]) or 'none'}" for c in A_CANDS))
excess = {c: bool(mult.loc[c, "cluster_p"] < A_ALPHA) for c in A_CANDS}
record_check("A", "A17", "Each candidate's number of failing A check clusters (Holm survivors) is consistent with chance",
             not any(excess.values()), metric=float(max(int(v.split("/")[0]) for v in mult.clusters_failing)),
             threshold=f"FAIL if Poisson-binomial p < {A_ALPHA} for the number of failing clusters ({len(cl_rate)} "
                       f"clusters, false-alarm rate 1 - 0.95^m each)",
             detail=("; ".join(f"{c}: {r.clusters_failing} clusters failing ({r.failing_clusters}; expected {r.expected}, "
                               f"p={r.cluster_p:.2g}); {r.checks_failing} checks with a survivor (binomial p="
                               f"{r.check_binom_p:.2g}, treats correlated checks as independent); raw failing parts "
                               f"{int(r.raw_failing)}/{int(r.parts)}" for c, r in mult.iterrows())
                     + f". Clusters (one per underlying quantity family): "
                     + "; ".join(f"{cl}: {', '.join(ks)}" for cl, ks in A_CLUSTERS.items())
                     + f". Per-cluster null rate 1 - 0.95^m assumes independent members; positive correlation inside a "
                       f"cluster makes the true rate lower, so the cluster p is conservative. Reference Congresses' raw "
                       f"nested-LOO failing share: mean {ref_share.mean():.0%}, max {ref_share.max():.0%}. KIND drift: a "
                       f"flagged candidate means the post-115 Congresses (and the deployment target) differ from 108-115 "
                       f"beyond chance; every forward split measures that honestly. Remedy: recency-weighted training "
                       f"(refit on train+val), recalibration / re-thresholding on the val Congress, density-normalised "
                       f"metrics, per-Congress reporting; regime as a feature (A15b), regime weighting only if D26 shows a "
                       f"gain. {A_S4_NOTE}; S5 not judged (hides the shift)."),
             quantity="cand_fail_count", by_split=a_split_from(excess), kind="drift", na_reason=A_NA)

# Trend vs break for failing level quantities (distances are excluded: their LOO values have no time trend).
tb = zt[zt.level & zt.fail][["part", "congress", "t", "trend_z", "kind", "sig"]].copy()
tb["implies"] = tb.kind.map(A_IMPL_SHORT)
print("\nTrend vs break (failing level parts; t / tr = distance from the stationary / linear-trend PI; * = Holm survivor "
      "(cell or joint); 'trend' needs a significant 108-115 slope):")
for part, g in tb.groupby("part", sort=False):
    print(f"  {part}: " + "; ".join(f"{r.congress} {r.kind}{'*' if r.sig else ''} (t={r.t:+.1f}, tr={r.trend_z:+.1f})"
                                   for r in g.itertuples()) + " -> " + " / ".join(f"{k}: {A_IMPL_SHORT[k]}" if g.kind.nunique() > 1 else A_IMPL_SHORT[k] for k in g.kind.unique()))
if not len(tb):
    print("  none")
off = zt[zt.level & ~zt.fail & zt.trend_fail]
if len(off):
    print("Off-trend but inside the stationary PI (the train trend predicted a change that did not happen): "
          + ", ".join(f"{r.part} {r.congress} (tr={r.trend_z:+.1f})" for r in off.itertuples()))
cnt = tb.kind.value_counts()
brk_c = tb[(tb.kind == "break") & tb.sig].groupby("congress").size()
_lst = lambda k: ", ".join(f"{r.part} {r.congress}" for r in tb[tb.kind == k].itertuples()) or "none"
by16 = {k: int(sum(brk_c.get(c, 0) for c in (SPLIT_DEFS["S4"]["folds"] if split_kind(k) == "within"
                                              else split_eval_congresses(k)) if c in A_CANDS))
        for k in ALL_SPLITS if split_kind(k) != "random"}
by16["S5"] = None
record_check("A", "A16", "Candidate drift is a predictable trend rather than a structural break (failing level quantities)", None,
             metric=float(cnt.get("break", 0) / max(len(tb), 1)), threshold="share of failing level parts that are breaks (INFO)",
             detail=(f"{cnt.get('break', 0)} break, {cnt.get('trend', 0)} consistent with trend, {cnt.get('undetermined', 0)} "
                     f"undetermined among {len(tb)} failing level parts (raw t-PI failures). Breaks: {_lst('break')}. "
                     f"Consistent with trend: {_lst('trend')}. Undetermined: {_lst('undetermined')}. Break => {A_IMPL['break']}. "
                     f"Trend => {A_IMPL['trend']}. Undetermined => {A_IMPL['undetermined']}. A15b: once sponsor-party "
                     f"features are removed, the regime similarity effect is {reg15.loc[A15_P, 'b_same']:+.4f} (p="
                     f"{reg15.loc[A15_P, 'p_perm']:.3f}); section D's D26 is the outcome-based test of regime weighting. "
                     f"by_split = number of Holm-surviving (cell or joint) break parts among the split's evaluated CANDIDATE "
                     f"Congresses (S4: its folds 116-118; S5 not judged; reference Congresses not counted, so 7-fold splits are "
                     f"not penalised by construction). These breaks are differences between history and the deployment "
                     f"target, not defects of the split that reveals them. The trend PI widens with extrapolation distance, "
                     f"so breaks are hardest to detect for 118."),
             quantity="drift_type", by_split=by16, kind="drift", na_reason=A_NA)

# Judged drift checks per split (section A): S4 and the by-Congress splits must be judged on the same checks.
_lb = checks_by_split(dedup=False, include_omitted=True)
_lb = _lb[_lb.section == "A"]
jcount = _lb.groupby("split").apply(lambda g: pd.Series(dict(
    judged=int(g.judged.sum()), failing=int((g.split_passed == False).sum()),   # noqa: E712
    metric_only=int((~g.judged & g.value.notna()).sum())))).reindex(ALL_SPLITS)
print("\nSection A split-dependent checks per split (judged = True/False verdicts; A14/A15/A16 carry numeric by_split metrics instead):")
print(jcount[["judged", "failing"]].T.to_string())
_fwd = [k for k in ALL_SPLITS if split_kind(k) in BY_CONGRESS_KINDS + ("within",)]
print(f"forward splits judged on the same number of A drift checks: {jcount.loc[_fwd, 'judged'].nunique() == 1} "
      f"({ {k: int(v) for k, v in jcount.loc[_fwd, 'judged'].items()} }); S5 judged on {int(jcount.loc['S5', 'judged'])} (random split hides the shift)")

tmax = float(np.ceil(np.nanmax(np.abs(zt.t.values))))
heatmap(zt.assign(congress=zt.congress.astype(str), t=zt.t.round(2)), "congress", "part", "t",
        title="Section A: standardised distance t of each candidate from the 8 train Congresses",
        subtitle=f"t = (x - mean) / (sd * sqrt(1 + 1/n)); fails beyond the t({len(A_TRAIN) - 1}) 95% point "
                 f"(one-sided for distances); red = above train, blue = below", fmt="+.1f",
        domain=[-tmax, tmax], diverging=True, height=40 + 18 * zt.part.nunique() + 60, width=560, name="A_t_summary")
