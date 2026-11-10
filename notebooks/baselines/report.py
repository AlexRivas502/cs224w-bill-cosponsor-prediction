"""Result tables, paired comparisons, rolling summaries and timing (DESIGN section 7).

`results_frame` turns EvalResults into rows of the `Row` schema; `show_main` prints models x metrics with
95% cluster-bootstrap CIs; `show_paired` prints paired differences; `rolling_summary` averages clean rows
only (asserted); `Timer` records stage runtimes.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import D_MDE
from .data import Corpus
from .evaluate import EvalResult, paired_vs, summarize_results
from .models import DISPLAY

KIND_ORDER = {"heuristic": 0, "reference_leaky": 1, "learned_ref": 2, "gnn": 3}
MDE = D_MDE                      # single source of truth: config.D_MDE (the split study's MDE, 0.01 / 0.01)
GNN_ORDER = ("hetero_sage", "rgcn", "lightgcn")


@dataclass
class Row:
    """One reported number (DESIGN section 7 row schema)."""
    fold: int
    role: str
    task: str
    t0: int
    protocol: str
    model: str
    kind: str
    config_set: str
    selection: str
    sup: str
    seed: int | None
    metric: str
    value: float
    ci_low: float
    ci_high: float
    ci_low_bill: float
    ci_high_bill: float
    n_eval: int
    runtime_s: float


ROW_COLS = [f.name for f in fields(Row)]


def results_frame(results: list[EvalResult], corpus: Corpus, B: int = 500) -> pd.DataFrame:
    """DataFrame (Row schema) with point estimates and 95% cluster / bill bootstrap CIs."""
    rows = summarize_results(results, corpus, B)
    df = pd.DataFrame(rows, columns=ROW_COLS) if rows else pd.DataFrame(columns=ROW_COLS)
    return df


def _label(model: str) -> str:
    return DISPLAY.get(model, model)


def show_main(df: pd.DataFrame, task: str, role: str = "test", fold: int | None = None, t0: int = 0,
              metrics=None, ci: bool = True) -> pd.DataFrame:
    """Models x metrics table for one task / role (/ fold), grouped by protocol and kind, value [CI]."""
    d = df[(df.task == task) & (df.role == role) & (df.t0 == t0)]
    if fold is not None:
        d = d[d.fold == fold]
    if d.empty:
        return pd.DataFrame()
    metrics = list(metrics or (["Recall@10", "Recall@20", "NDCG@20"] if task == "t1" else ["AUROC", "AP", "AP_lift"]))
    d = d[d.metric.isin(metrics)].copy()
    agg = d.groupby(["protocol", "kind", "model", "metric"]).agg(value=("value", "mean"), lo=("ci_low", "mean"),
                                                                 hi=("ci_high", "mean"), n=("n_eval", "first"),
                                                                 seeds=("seed", "nunique")).reset_index()
    if ci:
        agg["cell"] = [f"{v:.4f} [{lo:.4f}, {hi:.4f}]" for v, lo, hi in zip(agg.value, agg.lo, agg.hi)]
    else:
        agg["cell"] = agg.value.round(4).astype(str)
    tab = agg.pivot_table(index=["protocol", "kind", "model"], columns="metric", values="cell", aggfunc="first")
    tab = tab[[m for m in metrics if m in tab.columns]]
    n = agg.groupby(["protocol", "kind", "model"]).n.first()
    tab["n"] = n
    tab = tab.reset_index()
    tab["_k"] = tab.kind.map(KIND_ORDER).fillna(9)
    tab["model"] = tab.model.map(_label)
    return tab.sort_values(["protocol", "_k", "model"]).drop(columns="_k").set_index(["protocol", "kind", "model"])


def best_heuristic(results: list[EvalResult], task: str, role: str, protocol: str, fold: int, t0: int = 0,
                   exclude=("eb_rate",)) -> EvalResult | None:
    """Best same-protocol heuristic of an evaluation unit (Recall@20 / AUROC), excluding leaky references."""
    from .evaluate import auroc
    cand = [r for r in results if r.task == task and r.role == role and r.fold == fold and r.t0 == t0
            and r.kind == "heuristic" and r.model not in exclude and (task == "t2" or r.protocol == protocol)]
    if not cand:
        return None
    key = (lambda r: r.per_bill[:, 1].mean()) if task == "t1" else (lambda r: auroc(r.scores, r.rows.y.to_numpy()))
    return max(cand, key=key)


def show_paired(results: list[EvalResult], corpus: Corpus, pairs: list[tuple] | None = None, role: str = "test",
                B: int = 500) -> pd.DataFrame:
    """Paired cluster-bootstrap differences a - b, one row per pair and seed. pairs = [(EvalResult a,
    EvalResult b), ...]; default: every GNN against the best same-protocol heuristic, cosup and PairMLP (Task 1)
    or TabGBDT (Task 2), plus GNN-vs-GNN rows (G1 vs G2, G1 vs G3, G2 vs G3), all on the same seed.
    `real` is the single-seed rule (|diff| >= MDE and CI excludes 0); use `paired_seed_summary` for the
    rule over seeds."""
    if pairs is None:
        pairs = []
        for a in results:
            if a.kind != "gnn" or a.role != role:
                continue
            same = [r for r in results if r.task == a.task and r.fold == a.fold and r.role == a.role and r.t0 == a.t0
                    and (a.task == "t2" or r.protocol == a.protocol) and r.seed in (None, a.seed)]
            refs = [best_heuristic(results, a.task, role, a.protocol, a.fold, a.t0)]
            names = ("cosup", "pair_mlp") if a.task == "t1" else ("TabGBDT",)
            refs += [r for r in same if r.model in names]
            if a.model in GNN_ORDER:   # GNN vs GNN, each unordered pair once (same seed)
                later = GNN_ORDER[GNN_ORDER.index(a.model) + 1:]
                refs += [r for r in same if r.kind == "gnn" and r.model in later and r.seed == a.seed]
            pairs += [(a, b) for b in refs if b is not None and b is not a and np.array_equal(a.codes, b.codes)]
    rows = []
    for a, b in pairs:
        d = paired_vs(a, b, corpus, B=B)
        mde = MDE[a.task]
        real = abs(d["diff"]) >= mde and (d["ci_low"] > 0 or d["ci_high"] < 0)
        rows.append(dict(fold=a.fold, task=a.task, t0=a.t0, protocol=a.protocol, model=_label(a.model),
                         vs=_label(b.model), seed=a.seed, metric=d["metric"], diff=d["diff"], ci_low=d["ci_low"],
                         ci_high=d["ci_high"], mde=mde, real=real))
    return pd.DataFrame(rows)


def paired_seed_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Aggregate `show_paired` rows over seeds: mean and sd of the difference, the widest per-seed CI bounds,
    and `real` = |seed-mean diff| >= MDE AND every seed's paired CI excludes 0 with the same sign. With one
    seed, `real` is the single-seed rule and the `seeds` column says 1 (treat learned-vs-learned rows as
    provisional: training noise is not in a bills-only bootstrap)."""
    if paired is None or not len(paired):
        return pd.DataFrame()
    keys = ["fold", "task", "t0", "protocol", "model", "vs", "metric"]
    rows = []
    for k, g in paired.groupby(keys, sort=False):
        pos = bool((g.ci_low > 0).all())
        neg = bool((g.ci_high < 0).all())
        mean = float(g["diff"].mean())
        rows.append(dict(zip(keys, k), diff_mean=mean, diff_sd=float(g["diff"].std()) if len(g) > 1 else np.nan,
                         ci_low_min=float(g.ci_low.min()), ci_high_max=float(g.ci_high.max()), seeds=len(g),
                         mde=float(g.mde.iloc[0]), real=abs(mean) >= float(g.mde.iloc[0]) and (pos or neg)))
    return pd.DataFrame(rows)


def comparison_table(df: pd.DataFrame, task: str, fold: int, t0: int = 0) -> pd.DataFrame:
    """Final per-task table: every model (heuristics, references, GNNs) with its validation and test score
    (Recall@20 / AUROC: mean over seeds, '+- sd' when there are several seeds, test with the 95% cluster
    bootstrap CI averaged over seeds) plus the secondary test metrics."""
    main = "Recall@20" if task == "t1" else "AUROC"
    others = ["Recall@10", "NDCG@20"] if task == "t1" else ["AP"]
    d = df[(df.task == task) & (df.fold == fold) & (df.t0 == t0)]
    if d.empty:
        return pd.DataFrame()
    out = []
    for (prot, kind, model), g in d.groupby(["protocol", "kind", "model"]):
        r = dict(protocol=prot, kind=kind, model=_label(model))
        for role in ("val", "test"):
            x = g[(g.role == role) & (g.metric == main)]
            if x.empty:
                r[f"{role} {main}"] = ""
                continue
            ns = x.seed.nunique()
            sd = f" +- {x.value.std():.4f}" if ns > 1 else ""
            ci = f" [{x.ci_low.mean():.4f}, {x.ci_high.mean():.4f}]" if role == "test" else ""
            r[f"{role} {main}"] = f"{x.value.mean():.4f}{sd}{ci}"
            if role == "test":
                r["seeds"] = int(ns) if ns else "-"
        for m in others:
            x = g[(g.role == "test") & (g.metric == m)]
            r[f"test {m}"] = f"{x.value.mean():.4f}" if len(x) else ""
        r["n test"] = int(g[g.role == "test"].n_eval.max()) if (g.role == "test").any() else 0
        out.append(r)
    tab = pd.DataFrame(out)
    tab["_k"] = tab.kind.map(KIND_ORDER).fillna(9)
    return tab.sort_values(["protocol", "_k", "model"]).drop(columns="_k").set_index(["protocol", "kind", "model"])


def rolling_summary(df: pd.DataFrame, baseline: str | None = None) -> pd.DataFrame:
    """Mean over folds of clean test rows per (task, t0, protocol, model, metric), with a paired t-test of
    per-fold differences against `baseline` if given. Asserts that only clean rows enter the means."""
    d = df[(df.role == "test") & (df.selection == "clean")]
    assert (d.selection == "clean").all()
    g = d.groupby(["task", "t0", "protocol", "model", "metric", "fold"]).value.mean().reset_index()
    out = g.groupby(["task", "t0", "protocol", "model", "metric"]).value.agg(["mean", "std", "count"]).reset_index()
    if baseline is not None:
        p = []
        for _, r in out.iterrows():
            a = g[(g.task == r.task) & (g.t0 == r.t0) & (g.protocol == r.protocol) & (g.model == r.model) & (g.metric == r.metric)]
            b = g[(g.task == r.task) & (g.t0 == r.t0) & (g.model == baseline) & (g.metric == r.metric)
                  & ((g.protocol == r.protocol) | (r.task == "t2"))]
            m = a.merge(b, on="fold", suffixes=("", "_b"))
            if len(m) >= 2 and r.model != baseline:
                t = stats.ttest_rel(m.value, m.value_b)
                p.append((float((m.value - m.value_b).mean()), float(t.pvalue)))
            else:
                p.append((np.nan, np.nan))
        out["diff_vs_" + baseline] = [x[0] for x in p]
        out["p_ttest"] = [x[1] for x in p]
    return out


def tainted_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows excluded from rolling means (selection future_tuned / contaminated), for a separate block."""
    return df[df.selection != "clean"]


def seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and sd over seeds per (fold, role, task, t0, protocol, model, metric)."""
    keys = ["fold", "role", "task", "t0", "protocol", "model", "metric"]
    return df.groupby(keys).value.agg(["mean", "std", "count"]).reset_index()


class Timer:
    """Stage timer: `with timer("graphs"): ...`; `timer.table()` returns the timings."""

    def __init__(self):
        self.stages: dict[str, float] = {}
        self._t0 = time.time()

    @contextmanager
    def __call__(self, name: str):
        t = time.time()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + time.time() - t

    def table(self) -> pd.DataFrame:
        """Seconds and minutes per stage plus the total."""
        d = pd.DataFrame({"seconds": self.stages}).rename_axis("stage")
        d.loc["total (sum of stages)"] = d.seconds.sum()
        d["minutes"] = d.seconds / 60
        return d.round(1)


def show_timing(timer: Timer) -> pd.DataFrame:
    """Timing table of a Timer."""
    return timer.table()


def save_results(df: pd.DataFrame, out_dir: Path, timer: Timer | None = None, extra: dict | None = None) -> Path:
    """Write results.csv (and timing / extra JSON) into out_dir; returns the CSV path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "results.csv"
    df.to_csv(p, index=False)
    meta = dict(extra or {})
    if timer is not None:
        meta["timing_s"] = {k: round(v, 2) for k, v in timer.stages.items()}
    (out_dir / "results_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return p


def new_legislator_table(results: list[EvalResult], role: str = "test") -> pd.DataFrame:
    """Hit@20 share on positives from new legislators (no cosponsorship in 108..k-1) vs returning ones."""
    rows = []
    for r in results:
        nh = (r.extra or {}).get("new_hits") if r.task == "t1" and r.role == role else None
        if nh:
            rows.append(dict(fold=r.fold, t0=r.t0, protocol=r.protocol, model=_label(r.model), seed=r.seed,
                             new_hit20=nh["new"]["hit_rate"], new_positives=nh["new"]["positives"],
                             returning_hit20=nh["returning"]["hit_rate"],
                             new_share_of_positives=nh["new"]["positives"] / max(nh["new"]["positives"] + nh["returning"]["positives"], 1)))
    return pd.DataFrame(rows)


def drift_tables(data, test_congress: int, threshold: float = 1.0) -> dict[str, pd.DataFrame]:
    """Per-Congress means of the standardised x_b, leg and phi columns of a FoldData (standardised with its
    supervision fit), with the test Congress's shift in supervision sd units ('shift_sd', 'flag')."""
    from .features import PHI_ALL, drift_table
    sup = list(data.fit_congresses)
    out = {}
    for t in ("tbill", "leg"):
        feats = {k: data.graphs[k][t].x.numpy() for k in data.congresses}
        names = data.graphs[data.congresses[0]].feature_names[t]
        tab = drift_table(feats, sup, test_congress, names)
        if t == "tbill":
            tab = tab[~tab.index.str.startswith("title_")]
        out[t] = tab
    if data.phi:
        feats = {}
        for k in data.congresses:
            mu, sd = data.phi_stats(k)
            parts = [((p[~b.Excl].astype(np.float32) - mu) / sd)[::7] for p, b in zip(data.phi[k], data.blocks[k])]
            feats[k] = np.concatenate(parts, 0) if parts else np.zeros((0, len(PHI_ALL)))
        out["phi"] = drift_table(feats, sup, test_congress, list(PHI_ALL))
    return out


def show(obj, title: str | None = None, floatfmt: str = "{:.4f}") -> None:
    """Display a DataFrame (rich table in Jupyter, plain text when run headlessly), with an optional title."""
    if title:
        print(title)
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is not None:
            with pd.option_context("display.float_format", floatfmt.format, "display.max_rows", 200):
                display(obj)
            return
    except ImportError:
        pass
    if isinstance(obj, pd.DataFrame):
        with pd.option_context("display.float_format", floatfmt.format, "display.width", 200,
                               "display.max_columns", 40, "display.max_rows", 200):
            print(obj.to_string())
    else:
        print(obj)


def fit_table(results: list[EvalResult]) -> pd.DataFrame:
    """One row per trained (fold, task, t0, protocol, seed): E*, validation metric, test metric, runtime,
    config set and selection tag."""
    from .evaluate import auroc
    rows = {}
    for r in results:
        key = (r.fold, r.task, r.t0, r.protocol, r.model, r.seed)
        v = float(r.per_bill[:, 1].mean()) if r.task == "t1" else auroc(r.scores, r.rows.y.to_numpy())
        d = rows.setdefault(key, dict(fold=r.fold, task=r.task, t0=r.t0, protocol=r.protocol, model=_label(r.model),
                                      seed=r.seed, metric="Recall@20" if r.task == "t1" else "AUROC",
                                      best_epoch=(r.extra or {}).get("best_epoch"), config_set=r.config_set,
                                      selection=r.selection, runtime_s=round(r.runtime_s, 1)))
        d[r.role] = v
    return pd.DataFrame(list(rows.values()))


def val_curves(results: list[EvalResult]) -> pd.DataFrame:
    """Validation metric per epoch (columns) for every trained run (rows)."""
    rows = []
    for r in results:
        if r.role == "val" and (r.extra or {}).get("val_curve"):
            rows.append({"run": f"{_label(r.model)} {r.task} t0={r.t0} {r.protocol} fold {r.fold} seed {r.seed}",
                         **{f"ep{i + 1}": v for i, v in enumerate(r.extra["val_curve"])}})
    return pd.DataFrame(rows).set_index("run") if rows else pd.DataFrame()


def run_model_section(model_name: str, cfg, corpus: Corpus, timer: Timer, log=print) -> list[EvalResult]:
    """Notebook helper: train `model_name` over the configured folds / tasks / t0 / protocols / seeds if it is
    selected and implemented, print its fit table; returns its EvalResults (empty list otherwise)."""
    from .models import available_models
    from .train import run_models
    if model_name not in cfg.models:
        print(f"{_label(model_name)}: not selected in MODELS, skipped.")
        return []
    if model_name not in available_models():
        print(f"{_label(model_name)}: not implemented yet (baselines/models/), skipped.")
        return []
    with timer(_label(model_name)):
        res = run_models(model_name, cfg, corpus, log=log)
    show(fit_table(res), f"\n{_label(model_name)}: validation (fit on 109..c-2) and test (refit on 109..c-1) scores")
    return res
