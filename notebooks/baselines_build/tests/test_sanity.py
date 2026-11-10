"""Sanity checks of DESIGN section 9 that do not need a GNN (runs with pytest or as a script).

Covers: eligibility (sponsor / already-seen excluded, positives exact), graph asserts on every Congress,
stream truncation (t0 = 0 and 7) with a NEGATIVE CONTROL (a deliberately leaky feature must fail the test),
perturbation test with a trained PairMLP (Task 1 and Task 2, threads = 1), determinism (two seed-0 runs),
allowlist, depth, caucus_pit truncation, censored EB denominators, fit discipline (fit raises on Congresses
>= test, fitted_on recorded) and selection tags. Set BASELINES_GNN=hetero_sage (etc.) to also run the
perturbation test on that model.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

NOTEBOOKS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NOTEBOOKS))

from baselines import checks, config, counts, data, features, graphs, report, train  # noqa: E402

torch.set_num_threads(4)
_C = []


def corpus():
    if not _C:
        _C.append(data.load_corpus())
    return _C[0]


def test_eligibility():
    c = corpus()
    for k in range(109, 118):
        for t0 in (0, 7):
            checks.check_eligibility(c, data.t1_blocks(c, k, t0))


def test_graphs_all_congresses():
    c = corpus()
    title = features.fit_title_svd(c, 116)
    for k in range(109, 118):
        g = graphs.build_graph(c, k, 0, 3, title)
        checks.check_graph(g, c, k, 0, data.t1_blocks(c, k, 0))
        checks.check_t2_plain(g, c, k, 3)
        checks.check_allowlist(g.feature_names | {"phi": list(features.PHI_ALL)})


def test_truncation():
    c = corpus()
    title = features.fit_title_svd(c, 116)
    for t0 in (0, 7):
        checks.truncation_test(c, 117, t0, 3, title=title)
    checks.truncation_test(c, 112, 0, 3)


def test_truncation_negative_control():
    """A leaky popularity column (window up to cut + 30) must be caught by the truncation test."""
    c = corpus()
    orig = features.stream_mats

    def leaky(corpus_, k, day, spon, cand, t0, cum_day_offset=0):
        return orig(corpus_, k, day, spon, cand, t0, cum_day_offset + 30)

    features.stream_mats = leaky
    try:
        caught = False
        try:
            checks.truncation_test(c, 117, 0, 3, title=features.fit_title_svd(c, 116))
        except AssertionError:
            caught = True
        assert caught, "truncation test missed a deliberately leaky feature"
    finally:
        features.stream_mats = orig


def test_caucus_and_censoring():
    c = corpus()
    for k in range(110, 119):
        checks.check_caucus_pit(c, k)
    for k in range(108, 118):
        checks.check_censored_eb(c, k)
    print("   depth:", checks.check_depth(range(109, 119), 3))
    for k in (109, 113, 117):
        checks.check_windowed_reads(c, k, 3)


def _quick_cfg(threads=1):
    return config.RunConfig(quick=True, threads=threads)


def test_fit_discipline_and_tags():
    c = corpus()
    cfg = _quick_cfg(4)
    d = train.fold_data(c, 117, "val", 0, cfg)
    assert max(d.title.fitted_on) <= 115 and d.stds["tbill"].fitted_on == tuple(range(109, 116))
    assert d.phi_fitted_on == tuple(range(109, 116))
    m, t = config.model_config("pair_mlp", "t1")
    for bad in (dict(sup=list(range(109, 116)), val=117, test=117), dict(sup=list(range(109, 118)), val=None, test=117)):
        try:
            train.fit("pair_mlp", "t1", m, t, d, bad["sup"], bad["val"], bad["test"], log=None)
            raise AssertionError("fit accepted a Congress >= test")
        except ValueError:
            pass
    assert config.selection_tag((116,), 116) == "contaminated"
    assert config.selection_tag((116,), 113) == "future_tuned"
    assert config.selection_tag((116,), 117) == "clean" and config.selection_tag((), 112) == "clean"
    import pandas as pd
    df = pd.DataFrame([dict(role="test", selection="future_tuned", task="t1", t0=0, protocol="stream", model="m",
                            metric="Recall@20", fold=113, value=0.3)])
    assert report.rolling_summary(df).empty, "tainted rows must not enter rolling means"
    for f in (train.TUNING_DIR.glob("*.json") if train.TUNING_DIR.exists() else []):
        import json
        j = json.loads(f.read_text())
        assert j.get("role", "val") == "val" and "test" not in json.dumps(j.get("runs", [])).lower().replace("tested", ""), f


def _fit_pair(task, seed=0, threads=1):
    torch.set_num_threads(threads)
    c = corpus()
    cfg = _quick_cfg(threads)
    d = train.fold_data(c, 117, "val", 0, cfg)
    m, t = config.model_config("pair_mlp", task)
    prot = "stream" if task == "t1" else "plain"
    return train.fit("pair_mlp", task, m, t, d, list(range(109, 116)), 116, 117, prot, seed=seed, log=None), d


def test_determinism():
    a, _ = _fit_pair("t1", 0, 1)
    b, _ = _fit_pair("t1", 0, 1)
    assert abs(a.val_metric - b.val_metric) < 1e-6, (a.val_metric, b.val_metric)
    torch.set_num_threads(4)


def test_perturbation_pair_mlp():
    c = corpus()
    for task in ("t1", "t2"):
        f, d = _fit_pair(task)
        print("  ", task, checks.perturbation_test(f.model, d, c, 116, task, "stream" if task == "t1" else "plain"))
    torch.set_num_threads(4)


def test_perturbation_gnn():
    name = os.environ.get("BASELINES_GNN")
    if not name:
        return
    torch.set_num_threads(1)
    c = corpus()
    cfg = _quick_cfg(1)
    d = train.fold_data(c, 117, "val", 0, cfg)
    for task in ("t1", "t2"):
        m, t = config.model_config(name, task)
        f = train.fit(name, task, m, t, d, list(range(109, 116)), 116, 117, "stream" if task == "t1" else "plain",
                      log=None)
        print("  ", name, task, checks.perturbation_test(f.model, d, c, 116, task, "stream" if task == "t1" else "plain"))
    torch.set_num_threads(4)


if __name__ == "__main__":
    ok = True
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t:.1f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                ok = False
                import traceback
                print(f"FAIL {name}: {type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}", flush=True)
    sys.exit(0 if ok else 1)
