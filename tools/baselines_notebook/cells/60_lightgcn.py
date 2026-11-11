# %% [markdown]
# ## 6. LightGCN-feat
# No learning inside the message passing: node features are spread over the graph with a fixed normalized
# average, and an MLP learns from the result. Tests whether learned message passing is needed at all.

# %%
G3 = report.run_model_section("lightgcn", cfg, corpus, timer)
RESULTS += G3
