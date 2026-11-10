# %% [markdown]
# # Baselines for cosponsor prediction and bill advancement
#
# This notebook trains and compares the baselines for the two tasks of the project. All logic lives in the
# `baselines` package next to this notebook; each section here is a short explanation plus a few calls.
# Design rationale, citations and results: `notebooks/BASELINES.md` (and `baselines/DESIGN.md`).
#
# **Task 1, cosponsor prediction (temporal link prediction).** For each law-capable bill (HR, S, HJRES, SJRES)
# at prediction time *t0* days after introduction (t0 = 0 primary, 7 secondary), rank the eligible legislators
# (same chamber, in office at intro + t0, not the sponsor, not already a cosponsor by t0) by how likely they are
# to cosponsor in *(t0, t0 + 30]*. Metrics: Recall@10, Recall@20, NDCG@20 per bill over the **full** eligible
# pool, averaged over bills with at least one positive and an uncensored window.
#
# **Task 2, bill advancement (node classification).** Will a law-capable bill be reported out of committee within
# 180 days of introduction? Predicted at introduction (t0 = 0) with the policy area hidden; bills whose 180-day
# window passes the end of the Congress are dropped. Metrics: AUROC and average precision.
#
# **Split (from the split study).** Rolling origin by Congress. Fold *c*: train on earlier Congresses, validate
# on *c-1*, refit on everything up to *c-1* with the chosen settings, test on *c*. `MODE = "simple"` runs fold 117
# (validate 116, test 117); `MODE = "rolling"` runs folds 112-117. Congress 118 is a **locked** final test behind
# `LOCKED_118` (off by default, never used for tuning). Settings are chosen on validation Congresses only.
#
# *One documented deviation:* the GNNs and PairMLP use supervision Congresses **109**..c-2 (refit 109..c-1);
# Congress 108 is history only, because a Congress-108 graph would have no earlier Congress at all. The ported
# heuristics keep the split study's 108..c-2.
#
# **Protocols.** Task 1 uses **+stream** (primary): a bill may use its own Congress's events dated on or before
# its cutoff (introduction + t0, inclusive, day resolution), e.g. earlier cosponsorships of the same session.
# **plain** uses earlier Congresses only (plus the bill's own day-0 edges). Task 2 uses **plain**.
#
# **Models.** Heuristic bars from the split study (section 3); three GNNs: **G1 HeteroSAGE** (section 4),
# **G2 R-GCN** with basis decomposition (section 5) and **G3 LightGCN-feat**, fixed GCN diffusion of features
# (section 6); and learned non-GNN references: **PairMLP** (G1 with zero layers), **TabGBDT** / **TabLogReg** for
# Task 2 (section 7). Section 8 prints one comparison table per task (validation and test, bootstrap CIs, paired
# differences); section 10 lists the knobs worth experimenting with.
#
# **How to run.** Install `requirements-baselines.txt` into the Jupyter kernel's Python (see `BASELINES_RUN.md`),
# open this notebook from `notebooks/`, edit only the config cell below, then *Run All*. The parsed tables are found
# by searching upward for `parsed/`.
#
# **Runtime (measured, Apple Silicon CPU, 2026-11-09).** Simple mode, every baseline, one seed, `THREADS = 8`:
# 12-13 minutes (one run: G1 5.8, G2 3.5, G3 2.7 minutes; everything else under 1 minute). `SEEDS = (0, 1, 2)` triples the
# learned-model part (about 40 minutes). Rolling mode runs six folds at about 10-12 minutes each (about 1 hour);
# `FOLDS_ONLY` runs a subset. On a busy machine the run can take several times longer (the timing table at the end
# of section 8 shows where the time went). The first run builds caches under `derived/cache/baselines/`.

# %%
# --- Imports ------------------------------------------------------------------------------------------
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path.cwd().resolve()
for _p in [_here, *_here.parents]:                      # make `import baselines` work from any subfolder
    if (_p / "baselines" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        break
    if (_p / "notebooks" / "baselines" / "__init__.py").exists():
        sys.path.insert(0, str(_p / "notebooks"))
        break

import torch
from baselines import checks, config, data, features, graphs, report, train
from baselines.models import DISPLAY, available_models, heuristics

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 40)

# %% [markdown]
# ## 0. Configuration (the only cell you need to edit)
#
# | setting | meaning |
# |---|---|
# | `MODE` | `"simple"` = fold 117 (validate 116, test 117); `"rolling"` = folds 112-117 |
# | `TASKS` | `"t1"` cosponsor ranking, `"t2"` advancement |
# | `T0_LIST` | Task-1 prediction times; `(0, 7)` adds the secondary t0 = 7 (about +8-10 min) |
# | `T1_PROTOCOLS` | `("stream",)` primary; add `"plain"` to also train the plain-protocol model(s) in `PLAIN_MODELS` |
# | `MODELS` | GNNs and PairMLP to train: `hetero_sage` (G1), `rgcn` (G2), `lightgcn` (G3), `pair_mlp` |
# | `SEEDS` | `(0,)` for a quick look; `(0, 1, 2)` for mean and sd over seeds (needed before ranking the learned models) |
# | `DEVICE`, `THREADS` | `"cpu"` is the reference; `"mps"` works on Apple Silicon but is not faster for these sizes. If other heavy processes are running, set `THREADS` to cores / jobs (e.g. 4); oversubscription makes training several times slower |
# | `FOLDS_ONLY` | rolling mode only: run a subset of folds 112-117, e.g. `(116,)`; `()` = all |
# | `ROLLING_CONFIG` | rolling mode: `"apriori"` (clean defaults), `"search"` (re-tune per fold, overnight), `"frozen"` (exploration; tainted rows are excluded from means) |
# | `LOCKED_118` | score the locked final test Congress 118 once (leave False while developing) |
# | `USE_TITLE` | include the TF-IDF/SVD embedding of the introduced title |
# | `QUICK` | smoke test: 2 epochs, Senate-only Task-1 blocks |

# %%
MODE = "simple"                    # "simple" (fold 117) | "rolling" (folds 112-117)
TASKS = ("t1", "t2")
T0_LIST = (0,)                     # (0, 7) adds the secondary prediction time
T1_PROTOCOLS = ("stream",)         # add "plain" for the plain-protocol comparison
PLAIN_MODELS = ("hetero_sage",)    # models also trained under "plain" when it is in T1_PROTOCOLS
MODELS = ("hetero_sage", "rgcn", "lightgcn", "pair_mlp")
SEEDS = (0,)
DEVICE = "cpu"
THREADS = 8
RUN_SEARCH = False                 # re-run the validation search (slow); FROZEN settings are used otherwise
ROLLING_CONFIG = "apriori"
LOCKED_118 = False                 # the locked final test: leave False until every choice is final
FOLDS_ONLY = ()                    # rolling mode: e.g. (116, 117); () = all folds 112-117
USE_TITLE = True
QUICK = False

cfg = config.RunConfig(mode=MODE, tasks=TASKS, t0_list=T0_LIST, t1_protocols=T1_PROTOCOLS, plain_models=PLAIN_MODELS,
                       models=MODELS, seeds=SEEDS, device=DEVICE, threads=THREADS, run_search=RUN_SEARCH,
                       rolling_config=ROLLING_CONFIG, locked_118=LOCKED_118, folds_only=FOLDS_ONLY, use_title=USE_TITLE, quick=QUICK)
cfg = config.apply_env_overrides(cfg)    # only used by the headless test runner (environment variables)
train.set_threads(cfg.threads)
FOLDS = cfg.folds() + cfg.locked_folds()
RESULTS = []                             # every EvalResult of this run (validation and test rows)
timer = report.Timer()
print(f"mode {cfg.mode} -> folds {FOLDS} | tasks {cfg.tasks} | t0 {cfg.t0_list} | Task-1 protocols {cfg.t1_protocols}")
print(f"models {cfg.models} (implemented: {available_models()}) | seeds {cfg.seeds} | {cfg.device}, {cfg.threads} threads"
      + (" | QUICK smoke test" if cfg.quick else ""))
