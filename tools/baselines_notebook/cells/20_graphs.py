# %% [markdown]
# ## 2. Leak-free graphs and features
#
# **One graph per target Congress k** (L05: independent graphs per split unit; L12: seed-time semantics). Node
# types: `leg` (every legislator, point-in-time attributes as of k), `hbill` (every bill of the history window
# W_k = k-3..k-1), `comm` (40 full committees + `unk`), `tbill` (the law-capable bills of k, the prediction
# targets). History edges (sponsors, original / later cosponsors, referrals, all finished before k starts) come
# with reverse edges. Target edges point **into** `tbill` only (sponsor, cosponsors with lag <= t0, referrals dated
# <= the cutoff), so a target bill can never see another target bill and never changes a legislator's state:
# that is exactly the Task-2 "plain" rule, and Task-1 label-window cosponsors are never edges.
#
# **Features.** Every learned history feature uses the fixed window W_k with half-life-1 decay and is reported as a
# per-Congress average, plus a `hist_depth` one-hot (the data start at 108, so Congress 109 sees one earlier
# Congress while 117 would otherwise see nine). Target-bill features `x_b` come from an allowlist: bill type,
# chamber, day of Congress, sponsor's point-in-time attributes, day-0 cosponsors and referral committees (dated
# <= cutoff), and a TF-IDF/SVD embedding of the *introduced* title. Policy area, subjects, summaries, later text and
# final counts are forbidden and asserted against every feature list.
#
# **Pair features phi(b, l)** for Task 1 (a dense [bills x candidates x 22] array per block): sponsor -> candidate
# history, reciprocity, popularity, 2-hop walk mass, co-support with the bill's day-0 cosponsors, same caucus /
# state, and under +stream the same quantities from the bill's own Congress up to its cutoff (exact to the day).
#
# This section builds and caches everything the models need for each fold (validation fit: supervision 109..c-2;
# refit: 109..c-1; standardisers and the title model are fitted on supervision Congresses only), then runs the
# leak checks: graph asserts, a **stream truncation test** (deleting every event after a cutoff day changes
# nothing for the bills with that cutoff), the feature allowlist, history depth and a drift table.

# %%
FOLD_DATA = {}
with timer("graphs + features"):
    for c in FOLDS:
        for t0 in cfg.t0_list:
            for fit in ("val", "refit"):
                FOLD_DATA[(c, t0, fit)] = train.fold_data(corpus, c, fit, t0, cfg)
                d = FOLD_DATA[(c, t0, fit)]
                print(f"fold {c} t0={t0} {fit:5s}: Congresses {d.congresses[0]}-{d.congresses[-1]}, fit on "
                      f"{d.fit_congresses[0]}-{d.fit_congresses[-1]}, {d.seconds:.0f}s")
_d = FOLD_DATA[(FOLDS[0], cfg.t0_list[0], "refit")]
print()
print(graphs.describe_graph(_d.graphs[FOLDS[0]]))

# %%
# Leak checks: graph asserts on every graph (already run when FoldData was built), eligibility, truncation, allowlist,
# depth, caucus and censoring rules.
with timer("checks"):
    for (c, t0, fit), d in FOLD_DATA.items():
        assert set(d.checks) == set(d.congresses), "check_graph must run on every graph"
        for k in d.congresses:
            checks.check_t2_plain(d.graphs[k], corpus, k, d.H)
    _c, _t0 = FOLDS[0], cfg.t0_list[0]
    print("eligibility:", checks.check_eligibility(corpus, data.t1_blocks(corpus, _c, _t0)))
    print("allowlist:", checks.check_allowlist(_d.graphs[_c].feature_names | {"phi": list(features.PHI_ALL)}))
    print("hist_depth per Congress:", checks.check_depth(range(config.SUP_START, max(FOLDS) + 1), cfg.H))
    print("caucus_pit uses Congresses < k only:", checks.check_caucus_pit(corpus, _c, cfg.H))
    print("censored bills never in EB denominators:", checks.check_censored_eb(corpus, _c - 1))
    _tr = checks.truncation_test(corpus, _c, _t0, cfg.H, title=_d.title)
    print("stream truncation test (cutoff day, bills with that cutoff):",
          [(r["day"], r["n_tbill"], r["n_phi_bills"]) for r in _tr], "-> identical features")

# %%
# Drift: per-Congress means of standardised columns (supervision fit), shift of the validation Congress in sd units.
_dv = FOLD_DATA[(FOLDS[0], cfg.t0_list[0], "val")]
DRIFT = report.drift_tables(_dv, test_congress=FOLDS[0] - 1)
for _name, _tab in DRIFT.items():
    _flag = _tab[_tab.flag != ""]
    print(f"{_name}: {len(_flag)} of {len(_tab)} columns shift > 0.5 sd on validation Congress {FOLDS[0] - 1}"
          + (" (WARNING: > 1 sd: " + ", ".join(_flag.index[_flag.flag == '>1sd']) + ")" if (_flag.flag == ">1sd").any() else ""))
    if len(_flag):
        report.show(_flag[["shift_sd", "flag"]].T)
