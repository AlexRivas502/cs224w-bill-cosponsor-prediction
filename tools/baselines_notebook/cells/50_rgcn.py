# %% [markdown]
# ## 5. R-GCN
# Each edge type gets its own learned transformation; the 11 edge types share 4 basis matrices to keep the model
# small. The encoder family used by prior work on cosponsorship. 2 layers, hidden size 128.

# %%
G2 = report.run_model_section("rgcn", cfg, corpus, timer)
RESULTS += G2
