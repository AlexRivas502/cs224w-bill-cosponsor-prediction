# %% [markdown]
# ## 8. Results
# One table per task: validation score (Congress c-1) and test score (Congress c) with a 95% bootstrap CI.
# With one seed, differences between learned models below about 0.015 are noise; the cached 3-seed results are
# printed at the end for comparison.

# %%
DF = report.results_frame(RESULTS, corpus)
for c in FOLDS:
    for t0 in cfg.t0_list:
        if "t1" in cfg.tasks:
            report.show(report.comparison_table(DF, "t1", c, t0), f"\nCosponsor prediction, fold {c}, t0 = {t0}: Recall@20")
    if "t2" in cfg.tasks:
        report.show(report.comparison_table(DF, "t2", c), f"\nBill advancement, fold {c}: AUROC")

# %%
# Paired test differences (real = difference >= 0.01 and every seed's CI excludes 0), and new legislators.
checks.check_population(RESULTS)
for _w in checks.direction_warnings(RESULTS, cfg.plain_models):
    print("WARNING", _w)
PAIRED = report.show_paired(RESULTS, corpus)
if len(PAIRED):
    _ps = report.paired_seed_summary(PAIRED)
    report.show(_ps[["task", "model", "vs", "metric", "diff_mean", "real"]], "\nPaired differences (test):")
NEWLEG = report.new_legislator_table(RESULTS)
if len(NEWLEG):
    _nl = NEWLEG[NEWLEG.fold.isin(cfg.folds()[-1:]) & (NEWLEG.protocol == "stream")]
    report.show(_nl[["model", "new_hit20", "returning_hit20"]].sort_values("new_hit20"),
                "\nHit@20 for new vs returning legislators (cosponsor prediction):")
print(f"\ntotal runtime: {timer.table().loc['total (sum of stages)', 'minutes']:.1f} min")

# %%
OUT_DIR = Path(os.environ.get("BASELINES_OUT", corpus.root / "notebooks" / "baselines_results" / cfg.mode))
report.save_results(DF, OUT_DIR, timer, extra=dict(config=config.config_dict(cfg), folds=FOLDS))
if len(PAIRED):
    PAIRED.to_csv(OUT_DIR / "paired.csv", index=False)
_ref = corpus.root / "notebooks" / "baselines_results" / "reference_fold117_3seeds"
if (_ref / "results.csv").exists() and len(cfg.seeds) < 2:
    _rdf = pd.read_csv(_ref / "results.csv")
    for _task, _name in (("t1", "Cosponsor prediction"), ("t2", "Bill advancement")):
        if _task in cfg.tasks:
            report.show(report.comparison_table(_rdf, _task, 117, 0), f"\nCached 3-seed results, {_name}, fold 117:")
