# %% [markdown]
# ## 7. Learned references without a graph
# - **PairMLP:** the same model with message passing turned off. The gap to the GNNs is what the graph adds.
# - **Gradient-boosted trees / logistic regression** (advancement only): tabular models on the bill's own features.

# %%
PAIR = report.run_model_section("pair_mlp", cfg, corpus, timer)
RESULTS += PAIR
TAB = []
if "t2" in cfg.tasks and cfg.tabular_refs:
    with timer("tabular models"):
        for c in FOLDS:
            for seed in cfg.seeds:
                TAB += heuristics.t2_tabular_results(corpus, c, cfg.tab_sup_start, cfg.H, cfg.use_title, seed)
    RESULTS += TAB
