# %% [markdown]
# ## 8. Results: one comparison table per task
#
# **Main tables.** One table per task and fold: every heuristic, reference and GNN with its validation score
# (Congress c-1, fit on 109..c-2) and its test score (Congress c, refit on 109..c-1) with a 95% cluster-bootstrap
# CI (resampling sponsors, B = 500), plus the secondary test metrics. With several `SEEDS`, learned models show
# the seed mean and `+- sd`.
#
# **Paired differences.** Every GNN against the best same-protocol heuristic, co-support and PairMLP (Task 1) or
# TabGBDT (Task 2), and every GNN against every other GNN. All models of one evaluation share the same bootstrap
# resamples, so the CI is paired. The rule for calling a difference **real** is: the seed-mean difference is at
# least the minimum detectable effect `config.D_MDE` (0.01 Recall@20, 0.01 AUROC, from the split study), **and** the
# paired CI excludes 0 for every seed. The bootstrap resamples bills only, not training runs. Seed-to-seed
# training noise is about 0.003-0.005 Recall@20 (sd over 3 seeds, test), the thread count alone changes results
# by similar amounts, and one test score moved by 0.012 after a minor change to the loss normaliser. With one seed,
# a difference between two *learned* models below about 0.015 is therefore within noise, even when its bills-only
# CI excludes 0. With the cached 3 seeds (below): on Task 1 the three GNNs cannot be ranked against each other
# (pairwise differences <= 0.006), only G3 clears the MDE against PairMLP (+0.013), and G1 / G2 do not; on Task 2
# G3 beats G1 (+0.020 AUROC) and no GNN beats TabGBDT by the MDE. Validation ranks G3 last among the GNNs on
# Task 1 and ties it with G1 on Task 2, so a validation-based model choice would not have picked it (see
# `BASELINES.md`, results).
#
# **Cold start.** Hit@20 on positives from **new legislators**, defined here as no cosponsorship in 108..k-1
# (about 13% of Congress-117 positives). The split study's 18-20% counts legislators *unseen in the training
# Congresses of a plain split*, which is a different definition, so the shares differ.

# %%
DF = report.results_frame(RESULTS, corpus)
for c in FOLDS:
    for t0 in cfg.t0_list:
        if "t1" in cfg.tasks:
            report.show(report.comparison_table(DF, "t1", c, t0),
                        f"\nTask 1, fold {c} (validation {c - 1}, test {c}), t0 = {t0}: Recall@20 [95% CI]")
    if "t2" in cfg.tasks:
        report.show(report.comparison_table(DF, "t2", c), f"\nTask 2, fold {c} (validation {c - 1}, test {c}): AUROC [95% CI]")

# %%
print("population identity (check 7):", checks.check_population(RESULTS))
for _w in checks.direction_warnings(RESULTS, cfg.plain_models):
    print("WARNING", _w)
PAIRED = report.show_paired(RESULTS, corpus)
PAIRED_SEEDS = report.paired_seed_summary(PAIRED)
if len(PAIRED_SEEDS):
    report.show(PAIRED_SEEDS, f"\nPaired test differences a - b over {len(cfg.seeds)} seed(s) "
                "(real = |seed-mean diff| >= MDE and every seed's CI excludes 0):")
    if len(cfg.seeds) < 2:
        print("NOTE: one seed. Differences between learned models (GNN vs GNN, GNN vs PairMLP / TabGBDT) below about "
              "0.015 are within training noise; run SEEDS = (0, 1, 2) or see the cached 3-seed table below.")
NEWLEG = report.new_legislator_table(RESULTS)
if len(NEWLEG):
    report.show(NEWLEG[NEWLEG.fold.isin(cfg.folds()[-1:])].sort_values(["protocol", "new_hit20"]),
                "\nHit@20 on positives from new legislators (no cosponsorship in 108..k-1) vs returning ones:")
_learned = [r for r in RESULTS if r.kind in ("gnn", "learned_ref")]
if _learned:
    report.show(report.fit_table(_learned), "\nLearned models: best epoch, validation and test metric, runtime:")
report.show(timer.table(), "\nTiming (seconds / minutes per stage):", "{:.1f}")

# %%
import os
OUT_DIR = Path(os.environ.get("BASELINES_OUT", corpus.root / "notebooks" / "baselines_results" / cfg.mode))
_p = report.save_results(DF, OUT_DIR, timer, extra=dict(config=config.config_dict(cfg), folds=FOLDS))
if len(PAIRED):
    PAIRED.to_csv(OUT_DIR / "paired.csv", index=False)
print(f"results written to {_p}")

# %% [markdown]
# **Cached 3-seed reference (fold 117).** `baselines_results/reference_fold117_3seeds/` holds the results of
# this notebook run with `SEEDS = (0, 1, 2)` (simple mode, default settings). The next cell prints its tables, so
# a single-seed run can be read against the seed spread.

# %%
_ref = corpus.root / "notebooks" / "baselines_results" / "reference_fold117_3seeds"
if (_ref / "results.csv").exists() and len(cfg.seeds) < 2:
    _rdf = pd.read_csv(_ref / "results.csv")
    for _task in [t for t in ("t1", "t2") if t in cfg.tasks]:
        report.show(report.comparison_table(_rdf, _task, 117, 0), f"\nCACHED 3 seeds, {_task}, fold 117:")
    if (_ref / "paired.csv").exists():
        report.show(report.paired_seed_summary(pd.read_csv(_ref / "paired.csv")), "\nCACHED 3 seeds, paired differences:")
else:
    print("cached 3-seed reference not shown (missing, or this run already has several seeds).")
