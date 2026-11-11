# %% [markdown]
# ## 10. What to experiment with
#
# Every knob is a field of a config dataclass in `baselines/config.py` (`APRIORI` / `FROZEN` per model and task,
# `SEARCH_SPACE` for the validation grid). Change one thing at a time, judge it on the **validation** Congress only
# (the `val` column of the fit tables), and keep the test Congress for the final comparison.
#
# Start with seeds: with one seed, two learned models closer than about 0.015 Recall@20 cannot be told apart.
# The knobs below are ordered by how much we expect them to matter (round-1 search evidence in
# `baselines_build/tuning/*.json` and `BASELINES.md`).
#
# | knob (config name) | default | suggested range | what it tests / evidence so far |
# |---|---|---|---|
# | `SEEDS` | (0,) | (0, 1, 2) | seed noise; needed before ranking learned models (sd about 0.003 Recall@20 on validation) |
# | cold-start fixes in `models/hetero_sage.py` (not implemented) | none | drop all history in-edges of 10-15% of legislators per step; or gate the residual by `degree > 0` | a freshman has no history in-edges; new-legislator hit@20 is 0.22-0.24 for every learned model, with a seed spread of +-0.03 |
# | `HeteroSAGEConfig.dropout` (G1 Task 2) | 0.2 | 0.3-0.5, plus `TrainConfig.weight_decay` 1e-3 | G1 Task 2 overfits after 5-10 epochs; dropout 0.5 helped G3 on Task 2 (+0.004 AUROC in round 1, +0.001 after the normaliser fix); never tried for G1 |
# | `RGCNConfig.num_bases` (G2) | 4 | 2-8 | 8 bases on Task 2: +0.005 AUROC over 2 seeds, re-confirmed after the fixes (below the 0.01 bar) |
# | `RunConfig.H` (history window) | 3 | 2-4 | H = 2 was worse for G1 (-0.003); H = 4 is about equal |
# | `HeteroSAGEConfig.layers` / `RGCNConfig.layers` | 2 | 0-3 | 0 = PairMLP; 1 layer loses about 0.002 on Task 1 |
# | `*.d` (hidden size) | 128 | 64-256 | flat; 64 is faster and as good |
# | `TrainConfig.lr` | 3e-3 (G1, G2), 1e-3 (G3) | 1e-3 to 6e-3 | 1e-3 is worse for G1 / G2 |
# | `TrainConfig.loss` (Task 1) | "softmax" | "softmax", "bpr" | BPR (L11) lost 0.003-0.005 for every GNN |
# | `TrainConfig.sup_half_life` | None (G1, G2, PairMLP), 3 (G3) | None, 2-4 | recency weight on supervision Congresses; active for every model since the normaliser fix (G2 hl 3: inside noise) |
# | `TrainConfig.batch_bills` | 1024 / 2048 (G1, G2), 256 / 1024 (G3) | 128-2048 | G3 Task 1 batch 128: +0.002, 1.5x slower |
# | `TrainConfig.val_smooth`, `patience` | 3; 4 (T1) / 6 (T2) | 1-3; 4-8 | early-stopping noise; E* moves by several epochs between seeds |
# | `TrainConfig.avg_window` | 0 (off) | 3-5 | tail weight averaging (SWA-style); G2 Task 1: inside seed noise |
# | `LightGCNConfig.K`, `combine` | 2, "concat" | 1-3, "mean" | number of diffusion hops; K = 3 gave no gain |
# | `*.phi_norm` | "global" | "per_congress" | transductive (uses the evaluated Congress's full pair distribution): exploration only |
# | `USE_TITLE` | True | True / False | value of the introduced-title TF-IDF/SVD |
# | `T0_LIST`, `T1_PROTOCOLS` | (0,), ("stream",) | (0, 7), ("stream", "plain") | secondary prediction time; how much same-Congress signal adds |
# | GAT | off | swap `SAGEConv` for `GATv2Conv` in `models/hetero_sage.py` | learned neighbour weights (L04) |
# | `RunConfig.tab_sup_start` | 109 | 108 | 108 = the split study's convention for the tabular references |
#
# To re-run the validation search, set `RUN_SEARCH = True` (or call `train.run_search(...)`); logs go to
# `baselines_build/tuning/`. Never pick settings from test scores.

# %%
if cfg.run_search:
    for _m in [m for m in cfg.models if m in available_models()]:
        for _task in cfg.tasks:
            _log = train.run_search(_m, _task, cfg.folds()[-1], cfg, corpus)
            print(f"{_m} {_task}: chosen {_log['chosen']} (val {_log['best_val']:.4f}, tuned on {_log['tuned_on']})")
else:
    print("RUN_SEARCH is False: using the FROZEN settings in baselines/config.py.")
