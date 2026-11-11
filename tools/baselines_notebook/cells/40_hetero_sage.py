# %% [markdown]
# ## 4. HeteroSAGE
# GraphSAGE applied per edge type: each node averages its neighbours separately for each kind of edge
# (sponsor, cosponsor, referral, ...) and combines them with its own features. Works from features, not IDs, so it
# handles new legislators. 2 layers, hidden size 128.

# %%
G1 = report.run_model_section("hetero_sage", cfg, corpus, timer)
RESULTS += G1
