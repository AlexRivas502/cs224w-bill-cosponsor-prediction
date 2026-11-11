# %% [markdown]
# ## 3. Heuristic baselines
# The bars a learned model has to clear, ported from the split study.
# - **Cosponsor prediction:** popularity; same party, then popularity; **sponsor history** (how often the
#   candidate backed this sponsor before); 2-hop random walk; SVD embeddings; co-support with the day-0 cosponsors.
# - **Bill advancement:** random prior; committee report rate (and two simpler rate variants).

# %%
with timer("heuristics"):
    for c in FOLDS:
        for t0 in cfg.t0_list:
            if "t1" in cfg.tasks:
                RESULTS += heuristics.t1_reference_results(corpus, c, t0, protocols=("stream", "plain"), H=cfg.H)
        if "t2" in cfg.tasks:
            RESULTS += heuristics.t2_reference_results(corpus, c)
if 117 in FOLDS and 0 in cfg.t0_list and not cfg.quick:       # sanity check against the split study
    checks.check_reference_bars([r for r in RESULTS if r.kind != "gnn"], strict=True)
    print("heuristic scores match the split study")
