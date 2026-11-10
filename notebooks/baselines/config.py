"""All run settings, model defaults (APRIORI / FROZEN), search spaces and the fold plan.

Every knob the notebook exposes lives here, as a dataclass field with a default. Nothing in this module
reads data. See DESIGN.md sections 2.1 (split), 5 (optimisation) and 6 (selection discipline).
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Fixed task constants (identical to the split notebook's 00_setup.py) -------------------------
FIRST_CONGRESS = 108            # data start (history only for the GNNs, section 2.1)
SUP_START = 109                 # first GNN supervision Congress (documented deviation)
ROLLING_FOLDS = tuple(range(112, 118))
SIMPLE_FOLD = 117
LOCKED_FOLD = 118
EXCLUDED_CONGRESS = 119         # in progress, never loaded as labels
T1_METRICS = ("Recall@10", "Recall@20", "NDCG@20")
T2_METRICS = ("AUROC", "AP", "AP_lift")
D_MDE = {"t1": 0.01, "t2": 0.01}            # minimum detectable effects of the split study (digest_D.md); the
                                            # only MDE constant: report.MDE imports it
D_TOL = {"t1": 0.005, "t2": 0.01}           # adoption thresholds of the offline search (section 6.2)


@dataclass
class RunConfig:
    """Top-level notebook settings (the single config cell builds one of these).

    mode: "simple" runs fold 117 only; "rolling" runs folds 112..117. Locked Congress 118 runs only when
    `locked_118` is True. `quick` is a smoke test (2 epochs, Senate-only Task-1 blocks)."""
    mode: str = "simple"
    tasks: tuple = ("t1", "t2")
    t0_list: tuple = (0,)
    t1_protocols: tuple = ("stream",)
    plain_models: tuple = ("hetero_sage",)
    models: tuple = ("hetero_sage", "rgcn", "lightgcn", "pair_mlp")
    heuristics: bool = True
    tabular_refs: bool = True
    tab_sup_start: int = SUP_START
    seeds: tuple = (0,)
    device: str = "cpu"
    threads: int = 8
    locked_118: bool = False
    folds_only: tuple = ()                   # rolling mode: restrict to these test folds, e.g. (116, 117); () = all
    run_search: bool = False
    rolling_config: str = "apriori"          # "apriori" | "search" | "frozen" (exploration only; tagged)
    allow_contaminated: bool = False
    use_title: bool = True
    H: int = 3                               # history window (Congresses) for graphs and learned features
    quick: bool = False
    verbose: bool = True
    cache_dir: Path | None = None

    def folds(self) -> list[int]:
        """Test Congresses to run: [117] (simple) or 112..117 (rolling); 118 is added by `locked_folds`."""
        if self.mode == "simple":
            return [SIMPLE_FOLD]
        if self.mode == "rolling":
            if self.folds_only:
                bad = [c for c in self.folds_only if c not in ROLLING_FOLDS]
                if bad:
                    raise ValueError(f"folds_only {bad} not in the rolling folds {ROLLING_FOLDS}")
                return [c for c in ROLLING_FOLDS if c in self.folds_only]
            return list(ROLLING_FOLDS)
        raise ValueError(f"unknown mode {self.mode!r} (use 'simple' or 'rolling')")

    def locked_folds(self) -> list[int]:
        """[118] if the locked final test is switched on, else []."""
        return [LOCKED_FOLD] if self.locked_118 else []

    def config_set(self, fold: int) -> str:
        """Which model config set a fold uses: FROZEN for simple mode and locked 118, else `rolling_config`."""
        if fold == LOCKED_FOLD or self.mode == "simple":
            return "frozen"
        return self.rolling_config


@dataclass
class TrainConfig:
    """Optimiser and loop settings (DESIGN section 5)."""
    lr: float = 3e-3
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    batch_bills: int = 1024
    max_epochs: int = 20
    patience: int = 4
    warmup_frac: float = 0.05
    min_lr_frac: float = 0.1
    clip: float = 1.0
    sup_half_life: float | None = None       # recency weight 0.5 ** ((k_last - k) / hl), rescaled to mean 1 over the
                                             # supervision bills; None = unweighted (G1, G2, PairMLP; G3 uses 3.0)
    loss: str = "softmax"                    # Task 1: "softmax" (listwise, full pool) | "bpr"; Task 2 is BCE
    val_smooth: int = 1                      # moving-average window of the validation metric used for early stopping
                                             # and E* (both tasks use 3 via _t1_train / _t2_train)
    avg_window: int = 0                      # > 1: tail weight averaging (SWA-style): validate / keep the uniform
                                             # average of the last N epochs' weights; the refit averages the same N.
                                             # Off by default (a round-2 knob; G2 Task 1 avg_window 3: 0.3720 on
                                             # two seeds vs APRIORI 0.3691-0.3750, i.e. inside seed noise)
    mix_congresses: bool = False             # G3: mini-batches mix Congresses (cheap forward)


@dataclass
class HeteroSAGEConfig:
    """G1 HeteroSAGE (layers = 0 gives PairMLP)."""
    d: int = 128
    layers: int = 2
    dropout: float = 0.2
    drop_edge: float = 0.1
    tower: int = 64
    pair_hidden: int = 32
    phi_norm: str = "global"                 # "global" | "per_congress" (TRANSDUCTIVE: standardises each Congress with
                                             # its own full-Congress pair distribution, incl. later bills' stream
                                             # features; exploration only, off by default)
    use_dot: bool = True
    use_pair: bool = True
    aggr: str = "mean"
    tuned_on: tuple = ()


@dataclass
class RGCNConfig:
    """G2 R-GCN with basis decomposition (Schlichtkrull et al. 2018, eq. 2-3)."""
    d: int = 128
    layers: int = 2
    dropout: float = 0.2
    drop_edge: float = 0.2
    num_bases: int = 4
    decomposition: str = "basis"             # "basis" | "block" (off by default) | "none"
    impl: str = "manual"                     # "manual" (per-relation loop, unit-tested equal) | "pyg" (RGCNConv)
    tower: int = 64
    pair_hidden: int = 32
    phi_norm: str = "global"
    use_dot: bool = True
    use_pair: bool = True
    tuned_on: tuple = ()


@dataclass
class LightGCNConfig:
    """G3 LightGCN-feat: parameter-free GCN-normalised diffusion of features + MLPs (SGC / SIGN style)."""
    K: int = 2
    combine: str = "concat"                  # "concat" (SIGN) | "mean" (LightGCN layer average)
    mlp: tuple = (256, 128)
    d: int = 128
    dropout: float = 0.2
    tower: int = 64
    pair_hidden: int = 32
    phi_norm: str = "global"
    use_dot: bool = True
    use_pair: bool = True
    tuned_on: tuple = ()


MODEL_CFG_CLASS = {"hetero_sage": HeteroSAGEConfig, "pair_mlp": HeteroSAGEConfig,
                   "rgcn": RGCNConfig, "lightgcn": LightGCNConfig}


def _t1_train(**kw) -> TrainConfig:
    # val_smooth = 3: one-epoch validation Recall@20 swings by more than D_TOL between epochs (challenger round 1),
    # so early stopping uses a 3-epoch moving average, as Task 2 always did. Added on 2026-11-08 AFTER the
    # round-1 search, which ran with val_smooth = 1 and the old weighted step normaliser. FROZEN = APRIORI for every
    # Task-1 entry, so no choice depends on it; the post-fix re-check of APRIORI under the current code is logged
    # as "post_fix_recheck" in tuning/G*_r1.json and summarised in BASELINES.md.
    return TrainConfig(**{**dict(batch_bills=1024, max_epochs=20, patience=4, val_smooth=3), **kw})


def _t2_train(**kw) -> TrainConfig:
    return TrainConfig(**{**dict(batch_bills=2048, max_epochs=30, patience=6, val_smooth=3), **kw})


# A-priori defaults: chosen from lectures, papers and the split study's per-fold picks (no Congress score of
# this project was used), so tuned_on = () and every rolling fold that uses them is clean.
APRIORI: dict[tuple[str, str], dict[str, Any]] = {
    ("hetero_sage", "t1"): dict(model=HeteroSAGEConfig(), train=_t1_train(lr=3e-3)),
    ("hetero_sage", "t2"): dict(model=HeteroSAGEConfig(), train=_t2_train(lr=3e-3)),
    ("pair_mlp", "t1"): dict(model=HeteroSAGEConfig(layers=0, drop_edge=0.0), train=_t1_train(lr=3e-3)),
    ("pair_mlp", "t2"): dict(model=HeteroSAGEConfig(layers=0, drop_edge=0.0), train=_t2_train(lr=3e-3)),
    ("rgcn", "t1"): dict(model=RGCNConfig(), train=_t1_train(lr=3e-3)),
    ("rgcn", "t2"): dict(model=RGCNConfig(), train=_t2_train(lr=3e-3)),
    ("lightgcn", "t1"): dict(model=LightGCNConfig(), train=_t1_train(lr=1e-3, batch_bills=256, mix_congresses=True, sup_half_life=3.0)),
    ("lightgcn", "t2"): dict(model=LightGCNConfig(), train=_t2_train(lr=1e-3, batch_bills=1024, mix_congresses=True, sup_half_life=3.0)),
}

# Output of the offline validation search on fold 117 (section 6.2). Until the search has been run this is a
# copy of APRIORI with tuned_on = (); the orchestrator overwrites entries and sets tuned_on to the validation
# Congresses the choice was made on ((116,) for Task 1, (114, 115, 116) for Task 2).
FROZEN: dict[tuple[str, str], dict[str, Any]] = copy.deepcopy(APRIORI)

# G3 LightGCN-feat, FROZEN after the round-0 grid (tuning/lightgcn_t*_fold117.json: K, combine, lr) and the
# round-1 search (tuning/G3_LightGCN_feat_r1.json: dropout, wd, width, batch, loss, sup hl, phi_norm, schedule;
# 2 seeds for the candidates). Validation only. Task 1: nothing beat APRIORI by more than seed noise, so the
# APRIORI settings are kept. Task 2: dropout 0.2 -> 0.5 (+0.004 AUROC, better in all 6 seed x Congress cells).
FROZEN[("lightgcn", "t1")] = dict(model=LightGCNConfig(tuned_on=(116,)),
                                  train=_t1_train(lr=1e-3, batch_bills=256, mix_congresses=True, sup_half_life=3.0))
FROZEN[("lightgcn", "t2")] = dict(model=LightGCNConfig(dropout=0.5, tuned_on=(114, 115, 116)),
                                  train=_t2_train(lr=1e-3, batch_bills=1024, mix_congresses=True, sup_half_life=3.0))

# G1 HeteroSAGE, FROZEN after the round-1 search (tuning/G1_HeteroSAGE_r1.json; validation only, 4 threads).
# Task 1 (+stream, t0 = 0, fit 109-115, val 116): 18 one-factor runs over lr {1e-3, 6e-3}, layers {1, 3},
# d {64, 256}, H {2, 4}, dropout {0.1, 0.3}, loss bpr, sup hl None, wd 1e-3, DropEdge {0, 0.2}, pair MLP 128,
# phi_norm per_congress, patience 6. None beat APRIORI (0.3755; seed-1 rerun 0.3767) by D_TOL = 0.005, the best
# gain was +0.0004 (inside seed noise), and lr 1e-3, H 2, BPR and dropout 0.3 were worse, so APRIORI is kept.
# Task 2 (mean val AUROC over 114-116): layers 1, no title, lr 1e-3, dropout 0.3, wd 1e-3, d 64, batch 512; the
# best gain was +0.0023 (< 0.01), so APRIORI is kept. Note: in round 1 sup_half_life had no effect for G1 (each
# step holds one Congress and train.fit normalised by the weighted bill count, so the weights cancelled); G1, G2 and
# PairMLP now carry sup_half_life = None (bitwise identical to the round-1 results), and train.fit normalises by the
# unweighted count, so a half-life set by the user is now active.
FROZEN[("hetero_sage", "t1")] = dict(model=HeteroSAGEConfig(tuned_on=(116,)), train=_t1_train(lr=3e-3))
FROZEN[("hetero_sage", "t2")] = dict(model=HeteroSAGEConfig(tuned_on=(114, 115, 116)), train=_t2_train(lr=3e-3))

# G2 R-GCN, FROZEN after the round-1 search (tuning/G2_R_GCN_r1.json, which also folds in the round-0 grid
# tuning/rgcn_t*_fold117.json; validation only, 4 threads, decisions on seed means).
# Task 1 (+stream, t0 = 0, fit 109-115, val 116): APRIORI seed mean 0.3734 (seeds 0/1/2: 0.3735 / 0.3704 / 0.3764,
# sd 0.003). 20 factors (bases {2, 8}, block decomposition, lr {1e-3, 1e-2}, wd 1e-2, d {64 (3 seeds), 64 + 30
# epochs}, layers {1, 3 (2 seeds)}, dropout {0.1, 0.4}, DropEdge {0, 0.4}, pair MLP 64 (3 seeds), H 2, BPR, sup hl
# None, 10-epoch cosine horizon) and one combination: best gain +0.0025 (< D_TOL 0.005); lr 1e-3 and BPR were worse.
# Task 2 (mean val AUROC over 114-116): APRIORI seed mean 0.7913. Best factor num_bases 8 (+0.0052 over 2 seeds,
# below the 0.01 tolerance; a documented near-miss); everything else within +-0.0025. APRIORI kept for both tasks.
FROZEN[("rgcn", "t1")] = dict(model=RGCNConfig(tuned_on=(116,)), train=_t1_train(lr=3e-3))
FROZEN[("rgcn", "t2")] = dict(model=RGCNConfig(tuned_on=(114, 115, 116)), train=_t2_train(lr=3e-3))

# One-factor-at-a-time search grids (section 6.2). Keys are "model.<field>" or "train.<field>".
SEARCH_SPACE: dict[str, dict[str, list]] = {
    "hetero_sage": {"train.lr": [1e-3, 3e-3], "model.d": [64, 128], "model.layers": [1, 2, 3],
                    "H": [2, 3, 4], "train.loss": ["softmax", "bpr"], "model.dropout": [0.1, 0.3],
                    "train.sup_half_life": [None, 3.0]},
    "rgcn": {"model.num_bases": [2, 4, 8], "train.lr": [3e-3, 1e-2], "model.layers": [2, 3]},
    "lightgcn": {"model.K": [1, 2, 3], "model.combine": ["concat", "mean"], "train.lr": [3e-4, 1e-3, 3e-3]},
    "hetero_sage_t2": {"model.layers": [1, 2], "use_title": [True, False]},
}


def model_config(name: str, task: str, config_set: str = "apriori") -> tuple[Any, TrainConfig]:
    """(model config, TrainConfig) deep copies for `name` ("hetero_sage", "rgcn", "lightgcn", "pair_mlp"),
    `task` ("t1" / "t2") and `config_set` ("apriori" / "frozen"; "search" starts from APRIORI)."""
    table = FROZEN if config_set == "frozen" else APRIORI
    if (name, task) not in table:
        raise KeyError(f"no config for model {name!r}, task {task!r}")
    e = table[(name, task)]
    return copy.deepcopy(e["model"]), copy.deepcopy(e["train"])


def with_overrides(mcfg: Any, tcfg: TrainConfig, overrides: dict[str, Any]) -> tuple[Any, TrainConfig, dict]:
    """Apply {"model.x": v, "train.y": v, other: v} overrides; returns (mcfg, tcfg, run-level extras)."""
    mcfg, tcfg, extra = copy.deepcopy(mcfg), copy.deepcopy(tcfg), {}
    for k, v in overrides.items():
        if k.startswith("model."):
            mcfg = dataclasses.replace(mcfg, **{k[6:]: v})
        elif k.startswith("train."):
            tcfg = dataclasses.replace(tcfg, **{k[6:]: v})
        else:
            extra[k] = v
    return mcfg, tcfg, extra


def fold_plan(c: int) -> dict[str, Any]:
    """Congress roles of fold c (rolling origin, DESIGN section 2.1).

    GNN supervision 109..c-2 -> validate c-1 -> refit 109..c-1 -> test c; heuristics fit on 108..c-2 / 108..c-1;
    Task-2 multi-validation Congresses c-3..c-1 (each fit on 109..v-1)."""
    c = int(c)
    return {"sup_val": list(range(SUP_START, c - 1)), "val": c - 1, "sup_refit": list(range(SUP_START, c)),
            "test": c, "heur_train_val": list(range(FIRST_CONGRESS, c - 1)),
            "heur_train_test": list(range(FIRST_CONGRESS, c)), "t2_val_multi": [c - 3, c - 2, c - 1]}


def selection_tag(tuned_on: tuple, test: int) -> str:
    """'contaminated' if the test Congress was used for tuning, 'future_tuned' if a later Congress was,
    else 'clean' (DESIGN section 6.3)."""
    tuned_on = tuple(int(x) for x in (tuned_on or ()))
    if int(test) in tuned_on:
        return "contaminated"
    if tuned_on and max(tuned_on) > int(test):
        return "future_tuned"
    return "clean"


def config_dict(obj: Any) -> dict:
    """JSON-friendly dict of a config dataclass (paths and tuples converted)."""
    d = dataclasses.asdict(obj)
    return {k: (str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def apply_env_overrides(cfg: RunConfig) -> RunConfig:
    """Headless-runner overrides: BASELINES_MODE, BASELINES_QUICK (1/0), BASELINES_THREADS, BASELINES_MODELS
    (comma list), BASELINES_TASKS (comma list), BASELINES_T0 (comma list), BASELINES_PROTOCOLS (comma list),
    BASELINES_SEEDS (comma list), BASELINES_FOLDS (comma list, rolling mode only).
    Unset variables leave the notebook's values unchanged."""
    import os
    env = os.environ
    kw: dict[str, Any] = {}
    if env.get("BASELINES_MODE"):
        kw["mode"] = env["BASELINES_MODE"]
    if env.get("BASELINES_QUICK"):
        kw["quick"] = env["BASELINES_QUICK"] not in ("0", "false", "False", "")
    if env.get("BASELINES_THREADS"):
        kw["threads"] = int(env["BASELINES_THREADS"])
    for var, name, conv in (("BASELINES_MODELS", "models", str), ("BASELINES_TASKS", "tasks", str),
                            ("BASELINES_T0", "t0_list", int), ("BASELINES_PROTOCOLS", "t1_protocols", str),
                            ("BASELINES_SEEDS", "seeds", int), ("BASELINES_FOLDS", "folds_only", int)):
        if env.get(var) is not None:
            kw[name] = tuple(conv(x) for x in env[var].split(",") if x.strip())
    return dataclasses.replace(cfg, **kw) if kw else cfg
