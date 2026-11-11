# %% [markdown]
# ## 7. Learned non-GNN references
#
# - **PairMLP** (Task 1 and 2) is G1 with **zero** message-passing layers: the same type encoders, heads, pair
#   features, loss and optimiser. The gap G1 - PairMLP is the value of message passing beyond the hand-built
#   pair features.
# - **TabGBDT** (HistGradientBoosting, 300 iterations, lr 0.05, 31 leaves, early stopping) and **TabLogReg**
#   (standardised, C = 1) on the target-bill features x_b (Task 2): strong tabular references fitted on the same
#   supervision Congresses as the GNNs (109..c-2 / 109..c-1).

# %%
PAIR = report.run_model_section("pair_mlp", cfg, corpus, timer)
RESULTS += PAIR
TAB = []
if "t2" in cfg.tasks and cfg.tabular_refs:
    with timer("TabGBDT / TabLogReg"):
        for c in FOLDS:
            for seed in cfg.seeds:
                TAB += heuristics.t2_tabular_results(corpus, c, cfg.tab_sup_start, cfg.H, cfg.use_title, seed)
    RESULTS += TAB
    report.show(report.fit_table(TAB), "\nTabular references (Task 2):")
