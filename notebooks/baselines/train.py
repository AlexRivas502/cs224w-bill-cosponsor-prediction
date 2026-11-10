"""Generic training loop shared by every learned model (DESIGN section 5).

Pieces: `FoldData` (cached graphs, Task-1 blocks with pair features, Task-2 rows for one fold and one feature
fit), losses (full-pool listwise softmax / BPR for Task 1, BCE for Task 2), AdamW with warmup + cosine,
`fit` (early stopping on the validation-Congress metric, best weights), `run_model` (validation fit -> E* ->
refit on 109..c-1 with the same horizon, stop at E* -> test) and `run_search` (one factor at a time,
validation only, JSON logs).
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import (D_TOL, SEARCH_SPACE, SUP_START, RunConfig, TrainConfig, config_dict, fold_plan, model_config,
                     selection_tag, with_overrides)
from .data import Corpus, T1Block, t1_blocks, t2_rows
from .evaluate import EvalResult, auroc, block_metrics, new_leg_mask, new_legislator_hits
from .features import PHI_ALL, block_phi, fit_title_svd, phi_columns, phi_moments, state_index
from .graphs import build_graph, check_graph, fit_node_standardizers, standardized
from .models import build_model

EVAL_TIEBREAK = 1e-9
TUNING_DIR = Path(__file__).resolve().parents[1] / "baselines_build" / "tuning"


# --- Reproducibility -------------------------------------------------------------------------------------
def set_seed(s: int, deterministic: bool = True) -> None:
    """Seed random, numpy and torch; deterministic algorithms with warn_only (scatter reductions are not
    bitwise reproducible across thread counts)."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def set_threads(n: int) -> None:
    """torch.set_num_threads(n) (use 4 when several jobs share the machine)."""
    torch.set_num_threads(int(n))


# --- Fold data ---------------------------------------------------------------------------------------
@dataclass
class Part:
    """One mini-batch chunk from a single Congress (and, for Task 1, a single block)."""
    k: int
    tb: torch.Tensor
    sp: torch.Tensor
    cand: torch.Tensor | None = None
    phi: torch.Tensor | None = None
    Pos: torch.Tensor | None = None
    Excl: torch.Tensor | None = None
    xb: torch.Tensor | None = None
    y: torch.Tensor | None = None
    w: float = 1.0


class FoldData:
    """Graphs, Task-1 blocks (+ raw phi) and Task-2 rows of Congresses `congresses` for one feature fit.

    `fit_congresses` are the supervision Congresses: node standardisers, phi moments and the committee 'unk'
    mapping use them only; the title TF-IDF/SVD is fitted on Congresses <= max(fit_congresses)."""

    def __init__(self, corpus: Corpus, fit_congresses, congresses, t0: int, H: int, use_title: bool = True,
                 phi_norm: str = "global", tasks=("t1", "t2"), quick: bool = False, device: str = "cpu",
                 check: bool = True):
        t_start = time.time()
        self.corpus, self.t0, self.H, self.quick, self.device = corpus, int(t0), int(H), quick, device
        self.fit_congresses = tuple(int(k) for k in fit_congresses)
        self.congresses = tuple(int(k) for k in congresses)
        self.tasks = tuple(tasks)
        self.title = fit_title_svd(corpus, max(self.fit_congresses), use_title=use_title)
        assert max(self.title.fitted_on) <= max(self.fit_congresses)
        raw = {k: build_graph(corpus, k, t0, H, self.title) for k in self.congresses}
        self.checks = {}
        self.blocks = {}
        for k in self.congresses:
            bl = t1_blocks(corpus, k, t0) if "t1" in self.tasks else []
            if quick:
                bl = [b for b in bl if b.chamber == "Senate"]
            self.blocks[k] = bl
            if check:
                self.checks[k] = check_graph(raw[k], corpus, k, t0, bl)
        self.stds = fit_node_standardizers(raw, list(self.fit_congresses))
        seen = np.zeros(len(corpus.committees) + 1, bool)
        for k in self.fit_congresses:
            g = raw[k]
            for rel in (("hbill", "referred", "comm"), ("comm", "referred_t", "tbill")):
                ei = g[rel].edge_index
                seen[(ei[1] if rel[2] == "comm" else ei[0]).numpy()] = True
        seen[-1] = True
        self.comm_seen = seen
        self.graphs = {k: _to_device(standardized(raw[k], self.stds, seen), device) for k in self.congresses}
        self.tpos = {k: pd.Index(raw[k].tbill_codes) for k in self.congresses}
        self.phi = {}
        if "t1" in self.tasks:
            for k in self.congresses:
                self.phi[k] = [block_phi(corpus, b, H) for b in self.blocks[k]]
        self.phi_norm = phi_norm
        self._phi_moments = {}
        if "t1" in self.tasks:
            sup_phi = [p for k in self.fit_congresses for p in self.phi[k]]
            sup_ex = [b.Excl for k in self.fit_congresses for b in self.blocks[k]]
            self.phi_mu, self.phi_sd = phi_moments(sup_phi, sup_ex)
            self.phi_fitted_on = self.fit_congresses
        self.t2 = {}
        if "t2" in self.tasks:
            for k in self.congresses:
                rows = t2_rows(corpus, [k])
                if quick:
                    rows = rows[rows.chamber == "Senate"]
                tb = self.tpos[k].get_indexer(rows.code.to_numpy())
                assert (tb >= 0).all()
                sp = corpus.BL.spon.reindex(rows.code).to_numpy()
                self.t2[k] = dict(rows=rows, tb=tb, sp=sp, y=rows.y.to_numpy(np.float32))
        g0 = self.graphs[self.congresses[0]]
        self.meta = dict(x_dims={t: int(g0[t].x.shape[1]) for t in ("leg", "hbill", "comm", "tbill")},
                         n_states=len(state_index(corpus)), n_comm=len(corpus.committees) + 1,
                         xb_dim=int(g0["tbill"].x.shape[1]), phi_dim=len(PHI_ALL), n_leg=corpus.NL)
        self.seconds = time.time() - t_start

    def phi_stats(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(mean, sd) used to standardise phi of Congress k: supervision moments ("global") or Congress k's own
        eligible pairs ("per_congress", an option that uses the Congress's own feature distribution)."""
        if self.phi_norm == "global":
            return self.phi_mu, self.phi_sd
        if k not in self._phi_moments:
            self._phi_moments[k] = phi_moments(self.phi[k], [b.Excl for b in self.blocks[k]])
        return self._phi_moments[k]

    def t1_part(self, k: int, bi: int, rows: np.ndarray, cols: list[int], w: float = 1.0) -> Part:
        """Task-1 Part for rows of block bi of Congress k with phi columns `cols` (standardised)."""
        b = self.blocks[k][bi]
        mu, sd = self.phi_stats(k)
        phi = (self.phi[k][bi][rows][..., cols].astype(np.float32) - mu[cols]) / sd[cols]
        dev = self.device
        return Part(k=k, tb=torch.as_tensor(np.array(self.tpos[k].get_indexer(b.codes[rows])), device=dev),
                    sp=torch.as_tensor(np.array(b.spon[rows]), device=dev), cand=torch.as_tensor(np.array(b.cand), device=dev),
                    phi=torch.from_numpy(phi).to(dev), Pos=torch.from_numpy(b.Pos[rows]).to(dev),
                    Excl=torch.from_numpy(b.Excl[rows]).to(dev), w=w)

    def t2_part(self, k: int, rows: np.ndarray, w: float = 1.0) -> Part:
        """Task-2 Part for row positions of Congress k's _ADV rows."""
        d = self.t2[k]
        dev = self.device
        tb = torch.as_tensor(np.array(d["tb"][rows]), device=dev)
        return Part(k=k, tb=tb, sp=torch.as_tensor(np.array(d["sp"][rows]), device=dev),
                    xb=self.graphs[k]["tbill"].x[tb], y=torch.as_tensor(np.array(d["y"][rows]), device=dev), w=w)


def _to_device(g, device):
    if device == "cpu":
        return g
    h = g.to(device)
    for a in ("k", "t0", "H", "window", "tbill_codes", "hbill_codes", "tbill_day", "feature_names"):
        setattr(h, a, getattr(g, a))
    h._hom = None
    return h


_FOLD_CACHE: dict = {}


def fold_data(corpus: Corpus, c: int, fit: str, t0: int, run_cfg: RunConfig, H: int | None = None,
              use_title: bool | None = None, phi_norm: str = "global") -> FoldData:
    """Cached FoldData of fold c: fit "val" (supervision 109..c-2, Congresses 109..c-1) or "refit"
    (supervision 109..c-1, Congresses 109..c)."""
    H = run_cfg.H if H is None else H
    use_title = run_cfg.use_title if use_title is None else use_title
    p = fold_plan(c)
    sup = p["sup_val"] if fit == "val" else p["sup_refit"]
    ev = [p["val"]] if fit == "val" else [p["test"]]
    key = (corpus.token, c, fit, t0, H, use_title, phi_norm, run_cfg.quick, tuple(run_cfg.tasks), run_cfg.device)
    if key not in _FOLD_CACHE:
        _FOLD_CACHE[key] = FoldData(corpus, sup, sup + ev, t0, H, use_title, phi_norm, run_cfg.tasks,
                                    run_cfg.quick, run_cfg.device)
    return _FOLD_CACHE[key]


def custom_fold_data(corpus: Corpus, sup, ev: int, t0: int, run_cfg: RunConfig, H: int | None = None,
                     use_title: bool | None = None, phi_norm: str = "global", tasks=None) -> FoldData:
    """Cached FoldData for an arbitrary (supervision, evaluation) pair (Task-2 multi-validation search)."""
    H = run_cfg.H if H is None else H
    use_title = run_cfg.use_title if use_title is None else use_title
    tasks = tuple(tasks or run_cfg.tasks)
    key = (corpus.token, tuple(sup), ev, t0, H, use_title, phi_norm, run_cfg.quick, tasks, run_cfg.device)
    if key not in _FOLD_CACHE:
        _FOLD_CACHE[key] = FoldData(corpus, sup, list(sup) + [ev], t0, H, use_title, phi_norm, tasks,
                                    run_cfg.quick, run_cfg.device)
    return _FOLD_CACHE[key]


def clear_cache() -> None:
    """Drop cached FoldData objects (frees memory between folds in rolling mode)."""
    _FOLD_CACHE.clear()


# --- Losses ------------------------------------------------------------------------------------------
def listwise_softmax_loss(S: torch.Tensor, Pos: torch.Tensor, Excl: torch.Tensor) -> torch.Tensor:
    """Per-bill multi-positive listwise softmax over the full eligible pool (DESIGN 5):
    L_b = -(1/|P_b|) sum_p [s_p - logaddexp(s_p, LSE_{n in N_b} s_n)], N_b = pool minus Excl minus P_b."""
    neg = ~(Pos | Excl)
    lse = torch.logsumexp(S.masked_fill(~neg, float("-inf")), 1, keepdim=True)
    lse = torch.where(neg.any(1, keepdim=True), lse, torch.full_like(lse, -1e4))
    Sp = torch.where(Pos, S, torch.zeros_like(S))
    term = (Sp - torch.logaddexp(Sp, lse.expand_as(S))) * Pos
    return -term.sum(1) / Pos.sum(1).clamp(min=1)


def bpr_loss(S: torch.Tensor, Pos: torch.Tensor, Excl: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Per-bill mean over (positive, negative) pairs of softplus(s_n - s_p) (L11 BPR, all negatives)."""
    out = []
    for i in range(0, S.shape[0], chunk):
        s, p, e = S[i:i + chunk], Pos[i:i + chunk], Excl[i:i + chunk]
        neg = ~(p | e)
        d = F.softplus(s.unsqueeze(1) - s.unsqueeze(2))          # [b, M(pos), M(neg)] = softplus(s_n - s_p)
        m = p.unsqueeze(2) & neg.unsqueeze(1)
        out.append((d * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1))
    return torch.cat(out)


def t1_loss(S, part: Part, loss: str) -> torch.Tensor:
    """Per-bill Task-1 loss vector for a part."""
    if loss == "softmax":
        return listwise_softmax_loss(S, part.Pos, part.Excl)
    if loss == "bpr":
        return bpr_loss(S, part.Pos, part.Excl)
    raise ValueError(f"unknown Task-1 loss {loss!r}")


# --- Optimiser ---------------------------------------------------------------------------------------
def build_optimizer(model: torch.nn.Module, tcfg: TrainConfig, total_steps: int):
    """AdamW (no decay on biases, LayerNorm and embeddings) + linear warmup then cosine to min_lr_frac x lr
    over `total_steps`. Asserts that no parameter is still lazy (uninitialised)."""
    from torch.nn.parameter import UninitializedParameter
    assert not any(isinstance(p, UninitializedParameter) for p in model.parameters()), \
        "lazy parameters left: run a dry forward before build_optimizer"
    no_decay = set()
    for m in model.modules():
        if isinstance(m, (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d)):
            no_decay |= {id(p) for p in m.parameters(recurse=False)}
    decay, nodecay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (nodecay if (id(p) in no_decay or p.ndim < 2) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tcfg.weight_decay},
                             {"params": nodecay, "weight_decay": 0.0}], lr=tcfg.lr, betas=tuple(tcfg.betas))
    T = max(int(total_steps), 1)
    warm = max(int(round(tcfg.warmup_frac * T)), 1)

    def lam(step):
        """LR multiplier at a step: linear warmup, then cosine to min_lr_frac."""
        if step < warm:
            return (step + 1) / warm
        prog = min((step - warm) / max(T - warm, 1), 1.0)
        return tcfg.min_lr_frac + (1 - tcfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lam)


# --- Steps -------------------------------------------------------------------------------------------
def sup_weights(sup, half_life, counts: dict[int, int] | None = None) -> dict[int, float]:
    """Recency weight per supervision Congress: 0.5 ** ((k_last - k) / hl), or 1 if hl is None.

    With `counts` (supervision bills per Congress) the weights are rescaled to mean 1 over the supervision
    bills, sum_k w_k n_k = sum_k n_k, so the loss scale (and the effective learning rate) matches the unweighted
    loss. hl = None always gives exactly 1.0 per Congress."""
    kl = max(sup)
    w = {k: (1.0 if half_life is None else 0.5 ** ((kl - k) / half_life)) for k in sup}
    if half_life is None or not counts:
        return w
    n = {k: float(counts.get(k, 0)) for k in sup}
    z = sum(w[k] * n[k] for k in sup)
    scale = sum(n.values()) / z if z > 0 else 1.0
    return {k: w[k] * scale for k in sup}


def sup_counts(data: "FoldData", task: str, sup) -> dict[int, int]:
    """Number of supervision bills per Congress (Task-1 kept bills over all blocks / Task-2 rows)."""
    if task == "t1":
        return {int(k): int(sum(b.nb for b in data.blocks[k])) for k in sup}
    return {int(k): int(len(data.t2[k]["y"])) for k in sup}


def make_steps(data: FoldData, task: str, sup, batch: int, mix: bool, rng: np.random.Generator) -> list[dict]:
    """One epoch of steps. Each step maps Congress k -> list of (block index or None, row positions).
    mix=False: every step holds bills of one Congress (one encoder pass); mix=True: bills across Congresses."""
    items = []
    for k in sup:
        if task == "t1":
            for bi, b in enumerate(data.blocks[k]):
                items.append(np.c_[np.full(b.nb, k), np.full(b.nb, bi), np.arange(b.nb)])
        else:
            n = len(data.t2[k]["y"])
            items.append(np.c_[np.full(n, k), np.full(n, -1), np.arange(n)])
    if not items:
        return []
    steps = []
    if mix:
        allx = np.vstack(items)
        allx = allx[rng.permutation(len(allx))]
        chunks = [allx[i:i + batch] for i in range(0, len(allx), batch)]
    else:
        chunks = []
        by_k: dict = {}
        for it in items:
            by_k.setdefault(int(it[0, 0]), []).append(it)
        for k, its in by_k.items():
            a = np.vstack(its)
            a = a[rng.permutation(len(a))]
            chunks += [a[i:i + batch] for i in range(0, len(a), batch)]
        chunks = [chunks[i] for i in rng.permutation(len(chunks))]
    for ch in chunks:
        st: dict = {}
        for k in np.unique(ch[:, 0]):
            ck = ch[ch[:, 0] == k]
            st[int(k)] = [(int(bi) if bi >= 0 else None, np.sort(ck[ck[:, 1] == bi, 2])) for bi in np.unique(ck[:, 1])]
        steps.append(st)
    return steps


def steps_per_epoch(data: FoldData, task: str, sup, batch: int, mix: bool) -> int:
    """Number of steps per epoch for a supervision set (deterministic count)."""
    return len(make_steps(data, task, sup, batch, mix, np.random.default_rng(0)))


# --- Fit -----------------------------------------------------------------------------------------------
@dataclass
class FitResult:
    """Outcome of one fit: the model (best weights if validated), best epoch E*, validation curve,
    best validation metric, wall time, schedule horizon and training-loss curve."""
    model: torch.nn.Module
    best_epoch: int
    val_curve: list
    val_metric: float
    seconds: float
    horizon_steps: int
    loss_curve: list = field(default_factory=list)
    epochs_run: int = 0


def evaluate_val(model, data: FoldData, task: str, k: int, cols: list[int]) -> float:
    """Validation metric on Congress k: Task 1 mean Recall@20 over the full pool; Task 2 AUROC."""
    if task == "t1":
        S = predict_t1(model, data, k, cols)
        per = np.vstack([block_metrics(s, b, EVAL_TIEBREAK) for s, b in zip(S, data.blocks[k])])
        return float(per[:, 1].mean())
    s = predict_t2(model, data, k)
    return auroc(s, data.t2[k]["y"])


def fit(model_name: str, task: str, mcfg, tcfg: TrainConfig, data: FoldData, sup, val: int | None,
        test_congress: int, protocol: str = "stream", stop_epoch: int | None = None, horizon_epochs: int | None = None,
        seed: int = 0, log: Callable | None = print) -> FitResult:
    """Train `model_name` on supervision Congresses `sup` of `data`.

    With `val`: early stopping on the validation metric (patience, optional moving average), best weights
    restored. Without `val` (refit): train exactly `stop_epoch` epochs on a schedule whose horizon is
    `horizon_epochs` (default max_epochs) x steps per epoch. Raises if any supervision or validation Congress
    is >= test_congress."""
    sup = [int(k) for k in sup]
    if any(k >= test_congress for k in sup) or (val is not None and val >= test_congress):
        raise ValueError(f"supervision {sup} / validation {val} must precede the test Congress {test_congress}")
    if val is not None and val in sup:
        raise ValueError("validation Congress inside the supervision set")
    assert set(sup) <= set(data.fit_congresses), "supervision Congresses must be the FoldData fit Congresses"
    t_start = time.time()
    set_seed(seed)
    cols = phi_columns(protocol)
    meta = dict(data.meta, phi_dim=len(cols), task=task, protocol=protocol)
    model = build_model(model_name, mcfg, meta).to(data.device)
    if hasattr(model, "prepare"):
        model.prepare(data, sup + ([val] if val is not None else []))
    max_epochs = 2 if data.quick else tcfg.max_epochs
    spe = steps_per_epoch(data, task, sup, tcfg.batch_bills, tcfg.mix_congresses)
    horizon = (horizon_epochs or max_epochs) * spe
    opt, sched = build_optimizer(model, tcfg, horizon)
    # recency weights, rescaled to mean 1 over the supervision bills; the loss below divides by the UNWEIGHTED
    # bill count, so the weights act both within mixed steps (G3) and across one-Congress steps (G1, G2, PairMLP)
    wk = sup_weights(sup, tcfg.sup_half_life, sup_counts(data, task, sup))
    rng = np.random.default_rng(seed)
    n_epochs = max_epochs if stop_epoch is None else min(int(stop_epoch), max_epochs)
    curve, smooth, losses = [], [], []
    aw = int(getattr(tcfg, "avg_window", 0) or 0)
    tail, eval_model = [], None
    best, best_state, best_epoch, bad = -np.inf, None, 0, 0
    patience = 10 ** 9 if data.quick else tcfg.patience
    for ep in range(1, n_epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for st in make_steps(data, task, sup, tcfg.batch_bills, tcfg.mix_congresses, rng):
            opt.zero_grad(set_to_none=True)
            num, den = 0.0, 0.0
            for k, parts in st.items():
                emb = model(data.graphs[k])
                for bi, rows in parts:
                    if task == "t1":
                        p = data.t1_part(k, bi, rows, cols, wk[k])
                        lb = t1_loss(model.score_t1(emb, p), p, tcfg.loss)
                    else:
                        p = data.t2_part(k, rows, wk[k])
                        lb = F.binary_cross_entropy_with_logits(model.score_t2(emb, p), p.y, reduction="none")
                    num = num + p.w * lb.sum()
                    den += len(rows)               # unweighted count (a weighted one would cancel w per Congress)
            loss = num / max(den, 1e-12)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.clip)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * den
            nb += den
        losses.append(tot / max(nb, 1e-12))
        if aw > 1:                           # tail weight averaging: keep the last `aw` epochs' weights
            tail = (tail + [{n: t.detach().clone() for n, t in model.state_dict().items()}])[-aw:]
        if val is None:
            if log:
                log(f"    epoch {ep:2d}  loss {losses[-1]:.4f}  (refit, stop at {n_epochs})")
            continue
        if aw > 1:
            if eval_model is None:
                eval_model = copy.deepcopy(model)
            avg_state = average_states(tail)
            eval_model.load_state_dict(avg_state)
            m = evaluate_val(eval_model, data, task, val, cols)
        else:
            m = evaluate_val(model, data, task, val, cols)
        curve.append(m)
        s = float(np.mean(curve[-tcfg.val_smooth:]))
        smooth.append(s)
        if s > best + 1e-6:
            best, best_epoch, bad = s, ep, 0
            best_state = avg_state if aw > 1 else copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if log:
            log(f"    epoch {ep:2d}  loss {losses[-1]:.4f}  val {'Recall@20' if task == 't1' else 'AUROC'} {m:.4f}"
                + (f" (avg {s:.4f})" if tcfg.val_smooth > 1 else "") + ("  *" if best_epoch == ep else ""))
        if bad >= patience:
            break
    if val is not None and best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = n_epochs
        if aw > 1 and tail:                  # refit: the same tail average the validation fit selected
            model.load_state_dict(average_states(tail))
    return FitResult(model, best_epoch, curve, float(best) if val is not None else float("nan"),
                     time.time() - t_start, horizon, losses, ep)


def average_states(states: list[dict]) -> dict:
    """Uniform average of state dicts (floating tensors; other entries are taken from the last state).
    Tail weight averaging along one SGD trajectory, as in SWA (Izmailov et al. 2018)."""
    out = {}
    for name, last in states[-1].items():
        if torch.is_floating_point(last):
            out[name] = torch.stack([s[name] for s in states]).mean(0)
        else:
            out[name] = last.clone()
    return out


@torch.no_grad()
def predict_t1(model, data: FoldData, k: int, cols: list[int], batch: int = 1024) -> list[np.ndarray]:
    """Score matrices [nb, M] (float32) for every Task-1 block of Congress k (eval mode)."""
    model.eval()
    emb = model(data.graphs[k])
    out = []
    for bi, b in enumerate(data.blocks[k]):
        S = [model.score_t1(emb, data.t1_part(k, bi, np.arange(i, min(i + batch, b.nb)), cols)).float().cpu().numpy()
             for i in range(0, b.nb, batch)]
        out.append(np.vstack(S) if S else np.zeros((0, b.M), np.float32))
    return out


@torch.no_grad()
def predict_t2(model, data: FoldData, k: int, batch: int = 8192) -> np.ndarray:
    """Task-2 logits for every _ADV row of Congress k (eval mode)."""
    model.eval()
    emb = model(data.graphs[k])
    n = len(data.t2[k]["y"])
    return np.concatenate([model.score_t2(emb, data.t2_part(k, np.arange(i, min(i + batch, n)))).float().cpu().numpy()
                           for i in range(0, n, batch)]) if n else np.zeros(0)


def _rng(sup) -> str:
    return f"{min(sup)}-{max(sup)}" if len(sup) > 1 else str(sup[0])


def t1_result(model_name: str, S: list, data: FoldData, k: int, fold: int, role: str, protocol: str, **kw) -> EvalResult:
    """EvalResult of Task-1 score matrices on Congress k's blocks (tie-break 1e-9 x tieb)."""
    bl = data.blocks[k]
    per = np.vstack([block_metrics(s, b, EVAL_TIEBREAK) for s, b in zip(S, bl)])
    r = EvalResult("t1", fold, role, data.t0, protocol, model_name, np.concatenate([b.codes for b in bl]),
                   per_bill=per, **kw)
    r.new_hits = new_legislator_hits(S, bl, new_leg_mask(data.corpus, k), EVAL_TIEBREAK)
    return r


def t2_result(model_name: str, s: np.ndarray, data: FoldData, k: int, fold: int, role: str, **kw) -> EvalResult:
    """EvalResult of Task-2 scores on Congress k's rows."""
    rows = data.t2[k]["rows"]
    return EvalResult("t2", fold, role, data.t0, "plain", model_name, rows.code.to_numpy(), scores=s, rows=rows, **kw)


def run_model(model_name: str, task: str, c: int, run_cfg: RunConfig, corpus: Corpus, t0: int = 0,
              protocol: str = "stream", config_set: str | None = None, overrides: dict | None = None,
              seed: int = 0, log: Callable | None = print, keep_scores: bool = False) -> dict:
    """Validation fit (109..c-2, early stopping on c-1) -> E* -> refit on 109..c-1 with the same horizon,
    stopped at E* -> test on c. Returns {"val": EvalResult, "test": EvalResult, "fit_val": FitResult,
    "fit_test": FitResult, "config": ..., "scores": test scores if keep_scores}."""
    config_set = config_set or run_cfg.config_set(c)
    mcfg, tcfg = model_config(model_name, task, "frozen" if config_set == "frozen" else "apriori")
    mcfg, tcfg, extra = with_overrides(mcfg, tcfg, overrides or {})
    H = int(extra.get("H", run_cfg.H))
    use_title = bool(extra.get("use_title", run_cfg.use_title))
    tag = selection_tag(getattr(mcfg, "tuned_on", ()), c)
    if tag == "contaminated" and not run_cfg.allow_contaminated:
        raise ValueError(f"config tuned on {mcfg.tuned_on} would be tested on {c}; set allow_contaminated to explore")
    if task == "t2":
        protocol = "plain"
    p = fold_plan(c)
    t_start = time.time()
    dv = fold_data(corpus, c, "val", t0, run_cfg, H, use_title, mcfg.phi_norm)
    if log:
        log(f"[{model_name} {task} t0={t0} {protocol} fold {c}] validation fit {_rng(p['sup_val'])} -> {p['val']}")
    fv = fit(model_name, task, mcfg, tcfg, dv, p["sup_val"], p["val"], c, protocol, seed=seed, log=log)
    cols = phi_columns(protocol)
    common = dict(kind="learned_ref" if model_name == "pair_mlp" else "gnn", config_set=config_set, seed=seed)
    if task == "t1":
        vres = t1_result(model_name, predict_t1(fv.model, dv, p["val"], cols), dv, p["val"], c, "val", protocol,
                         sup=_rng(p["sup_val"]), selection=tag, **common)
    else:
        vres = t2_result(model_name, predict_t2(fv.model, dv, p["val"]), dv, p["val"], c, "val",
                         sup=_rng(p["sup_val"]), selection=tag, **common)
    dr = fold_data(corpus, c, "refit", t0, run_cfg, H, use_title, mcfg.phi_norm)
    if log:
        log(f"[{model_name} {task} t0={t0} {protocol} fold {c}] refit {_rng(p['sup_refit'])}, stop at E* = {fv.best_epoch}")
    max_epochs = 2 if dr.quick else tcfg.max_epochs
    fr = fit(model_name, task, mcfg, tcfg, dr, p["sup_refit"], None, c, protocol, stop_epoch=fv.best_epoch,
             horizon_epochs=max_epochs, seed=seed, log=log)
    if task == "t1":
        S = predict_t1(fr.model, dr, c, cols)
        tres = t1_result(model_name, S, dr, c, c, "test", protocol, sup=_rng(p["sup_refit"]), selection=tag, **common)
    else:
        S = predict_t2(fr.model, dr, c)
        tres = t2_result(model_name, S, dr, c, c, "test", sup=_rng(p["sup_refit"]), selection=tag, **common)
    secs = time.time() - t_start
    vres.runtime_s = tres.runtime_s = secs
    info = dict(best_epoch=fv.best_epoch, val_metric=fv.val_metric, val_curve=fv.val_curve, H=H, use_title=use_title,
                model_cfg=config_dict(mcfg), train_cfg=config_dict(tcfg))
    vres.extra, tres.extra = dict(info), dict(info)
    for r_ in (vres, tres):
        if getattr(r_, "new_hits", None) is not None:
            r_.extra["new_hits"] = r_.new_hits
    if log:
        log(f"[{model_name} {task} fold {c}] E* = {fv.best_epoch}, val {fv.val_metric:.4f}, "
            f"test {tres.per_bill[:, 1].mean() if task == 't1' else auroc(tres.scores, tres.rows.y.to_numpy()):.4f}, {secs:.0f}s")
    out = dict(val=vres, test=tres, fit_val=fv, fit_test=fr, config=info)
    if keep_scores:
        out["scores"] = S
        out["data_refit"] = dr
    return out


def run_models(model_name: str, run_cfg: RunConfig, corpus: Corpus, log: Callable | None = print,
               folds=None, protocols=None) -> list[EvalResult]:
    """Run one model over the configured folds, tasks, t0 values, Task-1 protocols and seeds; returns the
    validation and test EvalResults (plain protocol only for models in run_cfg.plain_models)."""
    out = []
    folds = list(folds or (run_cfg.folds() + run_cfg.locked_folds()))
    for c in folds:
        for task in run_cfg.tasks:
            t0s = run_cfg.t0_list if task == "t1" else (0,)
            prots = (protocols or run_cfg.t1_protocols) if task == "t1" else ("plain",)
            for t0 in t0s:
                for prot in prots:
                    if task == "t1" and prot == "plain" and model_name not in run_cfg.plain_models:
                        continue
                    ov, cs = None, None
                    if run_cfg.config_set(c) == "search":   # clean per-fold search on c-1 (Task 2: c-3..c-1)
                        s = run_search(model_name, task, c, run_cfg, corpus, t0=t0, protocol=prot, log=log)
                        ov, cs = s["chosen"], "search"
                    for seed in run_cfg.seeds:
                        r = run_model(model_name, task, c, run_cfg, corpus, t0, prot, config_set=cs, overrides=ov,
                                      seed=seed, log=log)
                        out += [r["val"], r["test"]]
    return out


# --- Offline search (validation only) ----------------------------------------------------------------
def _val_score(model_name, task, c, run_cfg, corpus, overrides, t0=0, protocol="stream", seed=0, log=None) -> dict:
    """Validation metric of one configuration: Task 1 on c-1 (fit 109..c-2); Task 2 the mean over c-3..c-1,
    each fit on 109..v-1."""
    mcfg, tcfg = model_config(model_name, task, "apriori")
    mcfg, tcfg, extra = with_overrides(mcfg, tcfg, overrides)
    H = int(extra.get("H", run_cfg.H))
    use_title = bool(extra.get("use_title", run_cfg.use_title))
    p = fold_plan(c)
    t_start = time.time()
    if task == "t1":
        d = fold_data(corpus, c, "val", t0, run_cfg, H, use_title, mcfg.phi_norm)
        f = fit(model_name, task, mcfg, tcfg, d, p["sup_val"], p["val"], c, protocol, seed=seed, log=log)
        vals, eps = [f.val_metric], [f.best_epoch]
    else:
        vals, eps = [], []
        for v in p["t2_val_multi"]:
            sup = list(range(SUP_START, v))
            d = custom_fold_data(corpus, sup, v, 0, run_cfg, H, use_title, mcfg.phi_norm, tasks=("t2",))
            f = fit(model_name, task, mcfg, tcfg, d, sup, v, c, "plain", seed=seed, log=log)
            vals.append(f.val_metric)
            eps.append(f.best_epoch)
    return dict(val_metric=float(np.mean(vals)), val_per_congress=vals, best_epochs=eps, seconds=time.time() - t_start)


def run_search(model_name: str, task: str, c: int, run_cfg: RunConfig, corpus: Corpus, space: dict | None = None,
               t0: int = 0, protocol: str = "stream", seed: int = 0, log: Callable | None = print,
               out_dir: Path | None = None) -> dict:
    """One-factor-at-a-time search from APRIORI on validation only (DESIGN 6.2). A change is adopted when it
    gains >= D_TOL (0.005 Recall@20 / 0.01 AUROC). Writes tuning/<model>_<task>.json; returns the log."""
    key = model_name + ("_t2" if task == "t2" and (model_name + "_t2") in SEARCH_SPACE else "")
    space = space or SEARCH_SPACE.get(key, {})
    tol = D_TOL[task]
    p = fold_plan(c)
    tuned_on = (p["val"],) if task == "t1" else tuple(p["t2_val_multi"])
    chosen: dict = {}
    base = _val_score(model_name, task, c, run_cfg, corpus, chosen, t0, protocol, seed)
    logs = [dict(overrides={}, **base, adopted=True, note="apriori")]
    if log:
        log(f"[search {model_name} {task}] apriori val {base['val_metric']:.4f}")
    best = base["val_metric"]
    mcfg0, tcfg0 = model_config(model_name, task, "apriori")
    for fac, values in space.items():
        cur = chosen.get(fac, _current(fac, mcfg0, tcfg0, run_cfg))
        for v in values:
            if v == cur:
                continue
            ov = dict(chosen, **{fac: v})
            r = _val_score(model_name, task, c, run_cfg, corpus, ov, t0, protocol, seed)
            gain = r["val_metric"] - best
            adopt = gain >= tol
            logs.append(dict(overrides=_jsonable(ov), factor=fac, value=v, **r, gain=gain, adopted=adopt))
            if log:
                log(f"[search {model_name} {task}] {fac}={v}: val {r['val_metric']:.4f} (gain {gain:+.4f})"
                    + ("  ADOPTED" if adopt else ""))
            if adopt:
                chosen[fac], best = v, r["val_metric"]
    res = dict(model=model_name, task=task, fold=c, t0=t0, protocol=protocol, tuned_on=list(tuned_on),
               role="val", tol=tol, chosen=_jsonable(chosen), best_val=best, runs=logs)
    out_dir = Path(out_dir or TUNING_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{model_name}_{task}_fold{c}.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def _current(fac: str, mcfg, tcfg, run_cfg):
    if fac.startswith("model."):
        return getattr(mcfg, fac[6:])
    if fac.startswith("train."):
        return getattr(tcfg, fac[6:])
    return getattr(run_cfg, fac)


def _jsonable(d: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
