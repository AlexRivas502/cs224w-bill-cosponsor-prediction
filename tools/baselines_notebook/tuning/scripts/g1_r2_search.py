"""G1 HeteroSAGE round-2 validation search (fold 117: fit 109-115, select on 116; Task 2: mean of 114-116).

Usage: python g1_r2_search.py <task t1|t2> <out.jsonl> <spec> [<spec> ...]
spec = "name|seed|json-overrides", e.g. 'base|0|{}' or 'aw3|0|{"train.avg_window": 3}'.
Appends one JSON line per run (validation numbers only; no test Congress is ever loaded as an evaluation set).
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))          # notebooks/

import numpy as np
from baselines import config, data, train

train.set_threads(4)
task, out = sys.argv[1], Path(sys.argv[2])
specs = sys.argv[3:]
cfg = config.RunConfig(mode="simple", tasks=(task,), threads=4)
corpus = data.load_corpus(cfg)
C = 117
p = config.fold_plan(C)


def one(name, seed, ov):
    mcfg, tcfg = config.model_config("hetero_sage", task, "apriori")
    mcfg, tcfg, extra = config.with_overrides(mcfg, tcfg, ov)
    H = int(extra.get("H", cfg.H))
    use_title = bool(extra.get("use_title", cfg.use_title))
    t_start = time.time()
    rows = []
    vals = p["t2_val_multi"] if task == "t2" else [p["val"]]
    for v in vals:
        if task == "t1":
            d = train.fold_data(corpus, C, "val", 0, cfg, H, use_title, mcfg.phi_norm)
            sup = p["sup_val"]
            f = train.fit("hetero_sage", task, mcfg, tcfg, d, sup, v, C, "stream", seed=seed, log=None)
        else:
            sup = list(range(config.SUP_START, v))
            d = train.custom_fold_data(corpus, sup, v, 0, cfg, H, use_title, mcfg.phi_norm, tasks=("t2",))
            f = train.fit("hetero_sage", task, mcfg, tcfg, d, sup, v, C, "plain", seed=seed, log=None)
        cur = f.val_curve
        e = f.best_epoch
        rows.append(dict(val_congress=v, smoothed_best=f.val_metric, single_epoch_at_best=cur[e - 1], best_epoch=e,
                         epochs_run=f.epochs_run, curve=[round(x, 5) for x in cur],
                         epoch_to_epoch_sd=float(np.std(np.diff(cur))) if len(cur) > 2 else None))
    rec = dict(name=name, task=task, seed=seed, overrides=ov,
               val_metric=float(np.mean([r["smoothed_best"] for r in rows])),
               val_single_epoch=float(np.mean([r["single_epoch_at_best"] for r in rows])),
               per_congress=rows, seconds=round(time.time() - t_start, 1),
               train_cfg=config.config_dict(tcfg), model_cfg=config.config_dict(mcfg), H=H, use_title=use_title)
    with out.open("a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    print(f"{name} seed {seed}: val {rec['val_metric']:.4f} (single-epoch {rec['val_single_epoch']:.4f}) "
          f"E* {[r['best_epoch'] for r in rows]} {rec['seconds']:.0f}s", flush=True)


for s in specs:
    name, seed, ov = s.split("|", 2)
    one(name, int(seed), json.loads(ov))
