# %% [markdown]
# # Baselines: bill advancement and cosponsor prediction
#
# Trains and compares every baseline on the leak-free Congress graph. All the logic lives in the `baselines`
# package next to this notebook; each section below is a few lines of explanation and a few calls.
# Full rationale, citations and results: `docs/BASELINES.md`.
#
# - **Bill advancement:** is a law-capable bill reported out of committee within 180 days? Metric: AUROC (and AP).
# - **Cosponsor prediction:** rank the eligible legislators by who cosponsors a new bill in the next 30 days.
#   Metric: Recall@20 over the full eligible pool (and Recall@10, NDCG@20).
# - **Split:** by Congress. Simple mode = fold 117: train up to 115, validate on 116, retrain up to 116, test on
#   117. Rolling mode repeats this for folds 112-117. Congress 118 is a locked final test, off by default.
#
# **Run:** install `notebooks/requirements.txt`, edit only the config cell, then *Run All*.
# About 12 minutes on a laptop CPU in simple mode.

# %%
import os
import sys
from pathlib import Path

import pandas as pd

for _p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:     # make `import baselines` work from any folder
    if (_p / "baselines" / "__init__.py").exists():
        sys.path.insert(0, str(_p)); break
    if (_p / "notebooks" / "baselines" / "__init__.py").exists():
        sys.path.insert(0, str(_p / "notebooks")); break

from baselines import checks, config, data, features, graphs, report, train
from baselines.models import available_models, heuristics

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 30)

# %% [markdown]
# ## 0. Configuration
# The only cell you need to edit.

# %%
MODE = "simple"                    # "simple" = fold 117 | "rolling" = folds 112-117
TASKS = ("t1", "t2")               # t1 = cosponsor prediction, t2 = bill advancement
MODELS = ("hetero_sage", "rgcn", "lightgcn", "pair_mlp")
SEEDS = (0,)                       # (0, 1, 2) to compare learned models fairly (about 3x slower)
T0_LIST = (0,)                     # cosponsor prediction time(s) in days after introduction; (0, 7) adds day 7
T1_PROTOCOLS = ("stream",)         # add "plain" to also train without current-Congress signals
PLAIN_MODELS = ("hetero_sage",)    # models trained under "plain" when it is listed above
DEVICE, THREADS = "cpu", 8         # lower THREADS if other heavy jobs are running
FOLDS_ONLY = ()                    # rolling mode: a subset of folds, e.g. (116,)
ROLLING_CONFIG = "apriori"         # rolling mode: "apriori" (default settings) | "search" | "frozen"
LOCKED_118 = False                 # score the locked test Congress 118; only once, at the very end
RUN_SEARCH = False                 # re-run the hyperparameter search on validation (slow)
USE_TITLE = True                   # use the introduced-title embedding as a bill feature
QUICK = False                      # 2-epoch smoke test

cfg = config.RunConfig(mode=MODE, tasks=TASKS, t0_list=T0_LIST, t1_protocols=T1_PROTOCOLS, plain_models=PLAIN_MODELS,
                       models=MODELS, seeds=SEEDS, device=DEVICE, threads=THREADS, run_search=RUN_SEARCH,
                       rolling_config=ROLLING_CONFIG, locked_118=LOCKED_118, folds_only=FOLDS_ONLY,
                       use_title=USE_TITLE, quick=QUICK)
cfg = config.apply_env_overrides(cfg)    # used only by the headless test runner
train.set_threads(cfg.threads)
FOLDS = cfg.folds() + cfg.locked_folds()
RESULTS = []                             # every result of this run (validation and test)
timer = report.Timer()
print(f"folds {FOLDS} | tasks {cfg.tasks} | models {cfg.models} | seeds {cfg.seeds}" + (" | QUICK" if cfg.quick else ""))
