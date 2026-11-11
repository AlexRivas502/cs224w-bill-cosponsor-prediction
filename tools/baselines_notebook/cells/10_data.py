# %% [markdown]
# ## 1. Data and labels
# Loads the Parquet tables and builds both labels, censoring, point-in-time parties and the eligible pools
# (identical to the split notebook; a test checks this). One row per Congress below.

# %%
with timer("data"):
    corpus = data.load_corpus(cfg)
    LABELS = data.label_summary(corpus, sorted(set(range(config.SUP_START - 1, max(FOLDS) + 1))), t0=0)
report.show(LABELS)
