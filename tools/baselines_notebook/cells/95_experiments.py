# %% [markdown]
# ## 10. What to experiment with
# Change one thing at a time in `baselines/config.py` and judge it on the **validation** score only.
#
# | Knob | Default | Try |
# |---|---|---|
# | `SEEDS` | (0,) | (0, 1, 2) before ranking learned models |
# | dropout (HeteroSAGE, advancement) | 0.2 | 0.3-0.5 with weight decay 1e-3 |
# | `RGCNConfig.num_bases` | 4 | 2-8 |
# | history window `RunConfig.H` | 3 | 2-4 |
# | layers | 2 | 0-3 (0 = PairMLP) |
# | `TrainConfig.lr` | 3e-3 (1e-3 LightGCN) | 1e-3 to 6e-3 |
# | cosponsor loss `TrainConfig.loss` | "softmax" | "bpr" |
# | `T0_LIST` / `T1_PROTOCOLS` | (0,) / ("stream",) | (0, 7) / ("stream", "plain") |
#
# Set `RUN_SEARCH = True` to re-run the validation search. Never choose settings from test scores.

# %%
if cfg.run_search:
    for _m in [m for m in cfg.models if m in available_models()]:
        for _task in cfg.tasks:
            _log = train.run_search(_m, _task, cfg.folds()[-1], cfg, corpus)
            print(f"{_m} {_task}: chosen {_log['chosen']} (val {_log['best_val']:.4f})")
