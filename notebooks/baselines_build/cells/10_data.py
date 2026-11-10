# %% [markdown]
# ## 1. Data and labels
#
# `load_corpus` reads the parsed Parquet tables and rebuilds every task definition of the split notebook
# (`_build/cells/00_setup.py`): cosponsorship events (withdrawn rows dropped), the 30-day Task-1 window and its
# censoring flag, the 180-day advancement label and its censoring flag, point-in-time party, the caucus of
# independents, the monthly point-in-time roster (eligible pools, including delegates) and the majority party
# per chamber. `baselines_build/tests/test_parity.py` checks that these are identical to the split notebook.
#
# Below: one row per Congress with the number of evaluable Task-1 bills (uncensored, at least one new cosponsor
# in the window), their positives, the median eligible pool, and the Task-2 rows and positive rate.

# %%
with timer("data"):
    corpus = data.load_corpus(cfg)
print(f"repo: {corpus.root}\nlegislators coded: {corpus.NL:,}; law-capable bills 108-118: {len(corpus.BL):,}; "
      f"Task-2 rows (uncensored): {len(corpus.ADV):,}; cache: {corpus.cache_dir}")
print("load timings (s):", {k: round(v, 1) for k, v in corpus.timings.items()})
_cong = sorted(set(range(config.SUP_START - 1, max(FOLDS) + 1)))
with timer("data"):
    LABELS = data.label_summary(corpus, _cong, t0=0)
report.show(LABELS)
