# %% [markdown]
# ## 2. Leak-free graphs
# One graph per target Congress: legislators, committees and the bills of the 3 previous Congresses (edges both
# ways), plus the target bills, which only receive edges dated on or before their cutoff (sponsor, cosponsors so
# far, committee referrals). Then the leak checks: if any check fails, the cell raises an error.

# %%
FOLD_DATA = {}
with timer("graphs + features"):
    for c in FOLDS:
        for t0 in cfg.t0_list:
            for fit in ("val", "refit"):
                FOLD_DATA[(c, t0, fit)] = train.fold_data(corpus, c, fit, t0, cfg)
_c, _t0 = FOLDS[0], cfg.t0_list[0]
_d = FOLD_DATA[(_c, _t0, "refit")]
print(graphs.describe_graph(_d.graphs[_c]))

with timer("checks"):
    for (c, t0, fit), d in FOLD_DATA.items():
        assert set(d.checks) == set(d.congresses)          # graph asserts ran on every graph
        for k in d.congresses:
            checks.check_t2_plain(d.graphs[k], corpus, k, d.H)
    checks.check_eligibility(corpus, data.t1_blocks(corpus, _c, _t0))
    checks.check_allowlist(_d.graphs[_c].feature_names | {"phi": list(features.PHI_ALL)})
    checks.check_depth(range(config.SUP_START, max(FOLDS) + 1), cfg.H)
    checks.check_caucus_pit(corpus, _c, cfg.H)
    checks.check_censored_eb(corpus, _c - 1)
    checks.truncation_test(corpus, _c, _t0, cfg.H, title=_d.title)
print("leak checks passed: graph structure, eligibility, feature allowlist, history depth, truncation test")
