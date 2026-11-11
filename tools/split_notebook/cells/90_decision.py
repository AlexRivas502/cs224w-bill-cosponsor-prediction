# %% [markdown]
# ## Decision: which split to use
#
# Sections A-D each tested one family of conditions a split relies on and recorded every result in the
# shared check registry (`CHECKS`) and, for D, in the backtest table `SPLIT_RESULTS`. This section reads
# only those two objects: nothing below is hardcoded, so if you change a threshold or rerun on new data,
# the tables, scores and the recommendation update with it.
#
# **How checks are used.** Every check has a *kind*. **Validity** checks ask whether a split gives an
# honest, usable estimate (leakage, val predicts test, enough test Congresses, censoring): a FAIL is a
# reason to prefer a different split, and only these rank the splits. **Drift** checks ask whether later
# Congresses differ from earlier ones: every forward split fails them by design (that is the split doing
# its job), so a drift FAIL selects a *remedy* (recalibration, recency weighting, per-Congress reporting),
# not a different split. **Requirements** are construction rules every split must follow (time-filter
# edges, rebuild features as of t0); they never count as split failures. A few split-dependent checks
# have no kind ("unclassified"); they are not used for the main ranking, but the recommendation is
# re-run with them included as a robustness test.

# %%
# All recorded checks, grouped by section: status, kind, metric, threshold and a compact detail line.
pd.set_option("display.width", 250)


def _dec_status(passed, req):
    s = {True: "PASS", False: "FAIL"}.get(passed if isinstance(passed, (bool, np.bool_)) else None, "INFO")
    return s + ("/REQ" if req else "")


def _dec_fmt(v, n=28):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return f"{v:.4g}"
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "~"


def _dec_short(s, n):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "~"


DEC_ALL = checks_summary(dedup=False).reset_index(drop=True)
DEC_ALL["status"] = [_dec_status(p, r) for p, r in zip(DEC_ALL.passed, DEC_ALL.requirement)]
DEC_ALL["kind_lbl"] = DEC_ALL.kind.map(lambda k: k if isinstance(k, str) else "unclass.")
_dec_kept = checks_summary(dedup=True)
_dec_kept_ids = set(_dec_kept.id)
DEC_ALL["counted"] = DEC_ALL.id.isin(_dec_kept_ids)
DEC_SECTIONS = {"A": "A. targets & covariates", "B": "B. graph & nodes", "C": "C. temporal & leakage",
                "D": "D. backtest"}
print(f"{len(DEC_ALL)} checks recorded; {len(_dec_kept_ids)} remain after de-duplicating shared quantities "
      "('dup' = verdict owned by another section's check of the same quantity; not counted twice)")
for sec, g in DEC_ALL.groupby("section", sort=True):
    t = pd.DataFrame({
        "id": g.id, "status": g.status, "kind": g.kind_lbl, "dup": np.where(g.counted, "", "dup"),
        "metric": g.metric.map(_dec_fmt), "threshold": g.threshold.map(lambda v: _dec_short(v, 34)),
        "condition / detail": [_dec_short(c, 58) + " | " + _dec_short(d, 50) for c, d in zip(g.condition, g.detail)],
    })
    print(f"\n--- {DEC_SECTIONS.get(sec, sec)} ({len(g)} checks) ---")
    print(t.to_string(index=False, max_colwidth=115))

# Counts used for decisions: de-duplicated, construction requirements excluded.
_dec_nr = checks_summary(requirements=False)
_dec_cnt = (_dec_nr.assign(status=[_dec_status(p, False) for p in _dec_nr.passed],
                           kind=_dec_nr.kind.map(lambda k: k if isinstance(k, str) else "unclassified"))
            .pivot_table(index="section", columns=["kind", "status"], values="id", aggfunc="count", fill_value=0))
print("\nDe-duplicated, non-requirement checks by section x kind x status (these are the counts the decision uses):")
print(_dec_cnt.to_string())
print(f"Construction requirements (apply to every split, not counted as failures): "
      f"{', '.join(checks_requirements().id)}")
_dec_conf = _dec_nr[_dec_nr.conflict.fillna(False).astype(bool)]
if len(_dec_conf):
    print("Quantities where sections disagree (kept verdict | hidden FAILs):")
    for r in _dec_conf.itertuples():
        print(f"  {r.quantity}: kept {r.id} = {_dec_status(r.passed, False)} | hidden fails: {r.hidden_fail_ids or '-'}"
              f" | other checks: {r.dup_ids}")

# %%
# Chart: check status by section (stacked bars, one bar per section x kind group, all recorded checks).
_dec_c = DEC_ALL.assign(grp=lambda d: np.where(d.requirement, "requirement", d.kind_lbl.replace({"unclass.": "unclassified"})))
_dec_c["st"] = _dec_c.status.str.replace("/REQ", "", regex=False)
_dec_chart1 = (_dec_c.groupby(["section", "grp", "st"]).size().rename("n").reset_index()
               .assign(sec=lambda d: d.section.map(DEC_SECTIONS)))
show_d3(r"""
const {width, height} = opts;
const m = {top: 48, right: 20, bottom: 40, left: 210};
const svg = d3.select(el).append('svg').attr('width', width).attr('height', height);
viz.title(svg, 'Check status by section and kind (all recorded checks)',
          'Validity FAILs rank splits; drift FAILs pick remedies; requirements are build rules for every split');
const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
const W = width - m.left - m.right, H = height - m.top - m.bottom;
const grpOrder = ['validity', 'drift', 'unclassified', 'requirement'];
const rows = [];
Array.from(new Set(data.map(d => d.sec))).sort().forEach(s => grpOrder.forEach(k => {
  if (data.some(d => d.sec === s && d.grp === k)) rows.push(s + ' | ' + k); }));
const sts = ['FAIL', 'PASS', 'INFO'];
const color = {PASS: viz.palette[2], FAIL: viz.palette[7], INFO: '#b9b8b2'};
const byRow = rows.map(r => { const [s, k] = r.split(' | '); const o = {row: r};
  sts.forEach(t => { const f = data.find(d => d.sec === s && d.grp === k && d.st === t); o[t] = f ? f.n : 0; }); return o; });
const stack = d3.stack().keys(sts)(byRow);
const xs = d3.scaleLinear([0, d3.max(byRow, d => d.FAIL + d.PASS + d.INFO) || 1], [0, W]).nice();
const ys = d3.scaleBand(rows, [0, H]).padding(0.18);
g.append('g').call(d3.axisTop(xs).ticks(6).tickSize(-H)).call(a => a.selectAll('line').attr('stroke', viz.grid))
  .call(a => a.select('.domain').remove()).call(a => a.selectAll('text').attr('fill', viz.ink2).style('font', '11px sans-serif'));
g.append('g').call(d3.axisLeft(ys).tickSize(0)).call(viz.axis).call(a => a.select('.domain').remove());
const tip = viz.tooltip(el);
stack.forEach(layer => g.selectAll(null).data(layer).join('rect')
  .attr('y', d => ys(d.data.row)).attr('height', ys.bandwidth())
  .attr('x', d => xs(d[0])).attr('width', d => Math.max(0, xs(d[1]) - xs(d[0])))
  .attr('fill', color[layer.key]).attr('stroke', '#fff').attr('stroke-width', 1)
  .on('mousemove', (e, d) => tip.show(`<b>${d.data.row}</b><br>${layer.key}: ${d.data[layer.key]}`, e))
  .on('mouseleave', () => tip.hide()));
viz.legend(svg, sts.map(s => [s, color[s]]), m.left, height - 12);
svg.append('text').attr('x', m.left + W).attr('y', height - 3).attr('text-anchor', 'end').attr('fill', viz.ink2)
  .style('font', '11px sans-serif').text('number of checks');
""", _dec_chart1[["sec", "grp", "st", "n"]], height=470, width=760, name="X_check_status")

# %% [markdown]
# ### Split scorecard: what supports and what undermines each candidate
#
# For every candidate split (the five base splits plus D's protocol variants) the scorecard combines
# (1) the **validity** checks that judged it (passes support the split, fails undermine it), (2) the
# unclassified split-dependent checks (shown separately), (3) how many drift checks it exposes, and (4) the
# backtest in `SPLIT_RESULTS`: for each task the baseline that **validation** would pick (best mean val
# score, as a modeler would choose), its mean val and test score, the mean |val - test| gap per fold
# relative to the test score's own 95% CI half-width (ratio < 1 = the gap is within sampling noise),
# the fold-to-fold standard deviation of the test score, and the number of test bills. Task 2 uses only
# baselines available at introduction (no policy area, per requirement C18 and the D16 drop).
# The `SPLIT_RESULTS` baselines are all **unweighted** (every training Congress counts the same). Section D
# also scored recency-weighted family members (e.g. `sh_hl1`, half-life 1 Congress) in `D_FAMILY_RESULTS`,
# but only for S2 and S3. The `*_rw` columns repeat the val pick with those members added, so S2 and S3
# can be compared on a model that is allowed to down-weight old Congresses; `-` means "not evaluated".
# **How to read it:** a usable split has 0 validity fails with many checks judged; a split judged by few
# checks has not "passed", it was simply not tested. Splits with 1 test fold have no fold variability.

# %%
# Per-split scorecard from checks_split_scores() (validity / unclassified / drift), SPLIT_RESULTS and D_FAMILY_RESULTS.
DEC_PRIMARY = {"cosponsor_t0": globals().get("_D_PRIMARY_T1", ("cosponsor_t0", "Recall@20"))[1],
               "cosponsor_t7": globals().get("_D_PRIMARY_T1", ("cosponsor_t0", "Recall@20"))[1],
               "advancement": globals().get("_D_PRIMARY_T2", ("advancement", "AUROC"))[1]}
# Baselines that need the policy area (missing for most bills at t0 <= 7 d: C18) are not a Task-2 option.
DEC_T2_EXCLUDE = [b for b in SPLIT_RESULTS.baseline.unique() if b == "eb_rate"]
DEC_TASK_LABEL = {"cosponsor_t0": "T1 t0=0", "cosponsor_t7": "T1 t0=7", "advancement": "T2"}
# Recency-weighted / near-tie family rows (S2 and S3 only). Task 2 keeps only the t0-safe members (policy area hidden).
_dec_fam = globals().get("D_FAMILY_RESULTS", pd.DataFrame(columns=SPLIT_RESULTS.columns))
_dec_fam = _dec_fam[(_dec_fam.task != "advancement") | _dec_fam.baseline.astype(str).str.endswith("_t0s")]
DEC_FAM_SPLITS = sorted(set(_dec_fam.split))
DEC_RES_EXT = pd.concat([SPLIT_RESULTS, _dec_fam[SPLIT_RESULTS.columns.intersection(_dec_fam.columns)]],
                        ignore_index=True)


def _dec_backtest(split, task, res=None):
    """Val-selected baseline of one split/task and its val/test level, gap, fold sd and size."""
    res = SPLIT_RESULTS if res is None else res
    m = DEC_PRIMARY[task]
    d = res[(res.split == split) & (res.task == task) & (res.metric == m)]
    if task == "advancement":
        d = d[~d.baseline.isin(DEC_T2_EXCLUDE)]
    if d.empty:
        return None
    best = d[d.eval_role == "val"].groupby("baseline").value.mean().idxmax()
    b = d[d.baseline == best]
    v = b[b.eval_role == "val"].set_index("fold")
    t = b[b.eval_role == "test"].set_index("fold")
    gap = (v.value - t.value).dropna()
    hw = ((t.ci_high - t.ci_low) / 2).mean()
    return dict(split=split, task=task, metric=m, baseline=best, val=v.value.mean(), test=t.value.mean(),
                gap_abs=gap.abs().mean(), gap_signed=gap.mean(), test_hw=hw,
                gap_ratio=gap.abs().mean() / hw if hw > 0 else np.nan,
                fold_sd=t.value.std(ddof=1) if len(t) > 1 else np.nan, n_folds=len(t), n_eval=int(t.n_eval.sum()))


DEC_BT = pd.DataFrame([r for s in ALL_SPLITS for k in DEC_PRIMARY for r in [_dec_backtest(s, k)] if r])
DEC_BT_RW = pd.DataFrame([r for s in DEC_FAM_SPLITS for k in DEC_PRIMARY for r in [_dec_backtest(s, k, DEC_RES_EXT)] if r])

import contextlib as _dec_ctx
import io as _dec_io
with _dec_ctx.redirect_stdout(_dec_io.StringIO()):   # the kind=None warning is reported once, below
    DEC_SC_V = checks_split_scores(kinds=("validity",))
DEC_SC_U = checks_split_scores(kinds=(None,))
DEC_SC_D = checks_split_scores(kinds=("drift",))
DEC_UNCLASS = sorted(set(checks_by_split(kinds=(None,)).id))
print(f"Unclassified split-dependent checks (not ranked; used in the robustness re-run): {', '.join(DEC_UNCLASS) or '-'}")

_bt1 = DEC_BT[DEC_BT.task == "cosponsor_t0"].set_index("split")
_bt2 = DEC_BT[DEC_BT.task == "advancement"].set_index("split")
_bt1r = DEC_BT_RW[DEC_BT_RW.task == "cosponsor_t0"].set_index("split") if len(DEC_BT_RW) else pd.DataFrame(columns=["baseline", "test"])
_bt2r = DEC_BT_RW[DEC_BT_RW.task == "advancement"].set_index("split") if len(DEC_BT_RW) else pd.DataFrame(columns=["baseline", "test"])
DEC_SCORE = pd.DataFrame({
    "kind": [split_kind(s) for s in ALL_SPLITS],
    "val_judged": DEC_SC_V.n_judged, "val_fail": DEC_SC_V.n_fail,
    "unc_judged": DEC_SC_U.n_judged, "unc_fail": DEC_SC_U.n_fail,
    "drift_fail": DEC_SC_D.n_fail, "drift_judged": DEC_SC_D.n_judged,
    "folds": _bt1.n_folds, "T1_base": _bt1.baseline, "T1_test": _bt1.test, "T1_gap/hw": _bt1.gap_ratio,
    "T1_fold_sd": _bt1.fold_sd, "T1_n_eval": _bt1.n_eval, "T1_base_rw": _bt1r.baseline, "T1_test_rw": _bt1r.test,
    "T2_base": _bt2.baseline, "T2_test": _bt2.test, "T2_gap/hw": _bt2.gap_ratio, "T2_fold_sd": _bt2.fold_sd,
    "T2_base_rw": _bt2r.baseline, "T2_test_rw": _bt2r.test,
}, index=pd.Index(ALL_SPLITS, name="split"))
print(f"\nScorecard (T1 = cosponsor t0=0 {DEC_PRIMARY['cosponsor_t0']}, T2 = advancement {DEC_PRIMARY['advancement']}; "
      "val-selected baseline; gap/hw = mean |val-test| / test 95% CI half-width; *_rw = val pick incl. recency-weighted "
      f"members, evaluated for {', '.join(DEC_FAM_SPLITS) or 'no split'} only):")
print(DEC_SCORE.round(3).astype(object).where(DEC_SCORE.notna(), "-").to_string())
_dec_nostream = [s for s in SPLIT_DEFS if split_kind(s) in BY_CONGRESS_KINDS and s + "+stream" not in SPLIT_VARIANTS]
print(f"Splits with no +stream variant evaluated (their streaming behaviour is unknown, not a protocol mismatch): "
      f"{', '.join(_dec_nostream) or 'none'}")

print("\nWhat undermines (-) / supports (+) each split (validity checks; unclassified fails in [ ]):")
_dec_lb = checks_by_split(kinds=None)
for s in ALL_SPLITS:
    g = _dec_lb[_dec_lb.split == s]
    gv = g[g.kind == "validity"]
    gu = g[~g.kind.map(lambda k: isinstance(k, str))]
    sup = ", ".join(gv.id[gv.split_passed == True])        # noqa: E712
    und = ", ".join(gv.id[gv.split_passed == False])       # noqa: E712
    uf = ", ".join(gu.id[gu.split_passed == False])        # noqa: E712
    print(f"  {s:10} - {und or 'none':12} + {sup or '-'}" + (f"  [unclassified fails: {uf}]" if uf else ""))

# %%
# Chart: split x check matrix of per-split verdicts (validity + unclassified split-dependent checks).
_dec_m = checks_by_split(kinds=("validity", None), include_omitted=True)
_dec_m = _dec_m.assign(v=np.select([_dec_m.split_passed == True, _dec_m.split_passed == False],   # noqa: E712
                                   ["PASS", "FAIL"], "not judged"),
                       grp=_dec_m.kind.map(lambda k: k if isinstance(k, str) else "unclassified"),
                       why=_dec_m.na_reason.fillna("").map(lambda s: _dec_short(s, 90)))
_dec_m = (_dec_m.assign(_o=(_dec_m.grp != "validity").astype(int))
          .sort_values(["_o", "section", "id"], kind="stable").drop(columns="_o"))
show_d3(r"""
const {width, height, splits} = opts;
const m = {top: 52, right: 16, bottom: 34, left: 92};
const svg = d3.select(el).append('svg').attr('width', width).attr('height', height);
viz.title(svg, 'Per-split verdicts of split-dependent checks',
          'Left block: validity checks (rank splits). Right block: unclassified (robustness only). Hover for detail.');
const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
const W = width - m.left - m.right, H = height - m.top - m.bottom;
const ids = Array.from(new Set(data.map(d => d.id)));
const xs = d3.scaleBand(ids, [0, W]).padding(0.06), ys = d3.scaleBand(splits, [0, H]).padding(0.08);
const color = {PASS: viz.palette[2], FAIL: viz.palette[7], 'not judged': '#ecebe7'};
const tip = viz.tooltip(el);
g.selectAll(null).data(data).join('rect').attr('x', d => xs(d.id)).attr('y', d => ys(d.split))
  .attr('width', xs.bandwidth()).attr('height', ys.bandwidth()).attr('rx', 2).attr('fill', d => color[d.v])
  .on('mousemove', (e, d) => tip.show(`<b>${d.id} · ${d.split}</b> (${d.grp})<br>${d.v}${d.why ? '<br>' + d.why : ''}`, e))
  .on('mouseleave', () => tip.hide());
const nVal = new Set(data.filter(d => d.grp === 'validity').map(d => d.id)).size;
if (nVal > 0 && nVal < ids.length) g.append('line').attr('x1', xs(ids[nVal]) - 1).attr('x2', xs(ids[nVal]) - 1)
  .attr('y1', -4).attr('y2', H).attr('stroke', viz.ink2).attr('stroke-dasharray', '3,2');
g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xs).tickSize(0)).call(viz.axis)
  .call(a => a.select('.domain').remove()).selectAll('text').attr('transform', 'rotate(-45)').attr('text-anchor', 'end')
  .attr('dx', '-0.3em').attr('dy', '0.4em');
g.append('g').call(d3.axisLeft(ys).tickSize(0)).call(viz.axis).call(a => a.select('.domain').remove());
viz.legend(svg, Object.keys(color).map(k => [k, color[k]]), m.left, 38);
""", _dec_m[["id", "split", "v", "grp", "why"]], height=400, width=860,
     opts=dict(splits=list(ALL_SPLITS)), name="X_split_check_matrix")

# %%
# Chart: val (hollow) and test (filled) scores of the val-selected baseline per fold, by split and task.
_dec_pts = []
for r in DEC_BT[DEC_BT.task.isin(["cosponsor_t0", "advancement"])].itertuples():
    d = SPLIT_RESULTS[(SPLIT_RESULTS.split == r.split) & (SPLIT_RESULTS.task == r.task)
                      & (SPLIT_RESULTS.metric == r.metric) & (SPLIT_RESULTS.baseline == r.baseline)]
    for q in d.itertuples():
        _dec_pts.append(dict(task=f"{DEC_TASK_LABEL[r.task]} {r.metric}",
                             split=r.split, role=q.eval_role, fold=q.fold, value=float(q.value), base=r.baseline,
                             ev="" if pd.isna(q.eval_congress) else str(int(q.eval_congress))))
_dec_pts = pd.DataFrame(_dec_pts)
show_d3(r"""
const {width, height, splits} = opts;
const tasks = Array.from(new Set(data.map(d => d.task)));
const m = {top: 52, right: 14, bottom: 58, left: 50}, gapX = 46;
const svg = d3.select(el).append('svg').attr('width', width).attr('height', height);
viz.title(svg, 'Validation vs test score of the val-selected baseline, every fold of every split',
          'Hollow = validation, filled = test; vertical spread = fold-to-fold variability (one panel per task)');
const PW = (width - m.left - m.right - gapX * (tasks.length - 1)) / tasks.length, H = height - m.top - m.bottom;
const color = {val: viz.palette[0], test: viz.palette[1]};
const tip = viz.tooltip(el);
tasks.forEach((t, i) => {
  const dd = data.filter(d => d.task === t);
  const g = svg.append('g').attr('transform', `translate(${m.left + i * (PW + gapX)},${m.top})`);
  const xs = d3.scaleBand(splits, [0, PW]).padding(0.2);
  const ys = d3.scaleLinear(d3.extent(dd, d => d.value), [H, 0]).nice();
  g.append('g').call(d3.axisLeft(ys).ticks(5).tickSize(-PW).tickFormat(d3.format('.2f')))
    .call(a => a.selectAll('line').attr('stroke', viz.grid)).call(a => a.select('.domain').remove())
    .call(a => a.selectAll('text').attr('fill', viz.ink2).style('font', '11px sans-serif'));
  g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xs).tickSize(0)).call(viz.axis)
    .call(a => a.select('.domain').remove()).selectAll('text').attr('transform', 'rotate(-40)').attr('text-anchor', 'end');
  g.append('text').attr('x', 0).attr('y', -8).attr('fill', viz.ink).style('font', '600 11px sans-serif').text(t);
  g.selectAll(null).data(dd).join('circle')
    .attr('cx', d => xs(d.split) + xs.bandwidth() * (d.role === 'val' ? 0.3 : 0.7)).attr('cy', d => ys(d.value)).attr('r', 3.5)
    .attr('fill', d => d.role === 'test' ? color.test : '#fff').attr('stroke', d => color[d.role]).attr('stroke-width', 1.5)
    .on('mousemove', (e, d) => tip.show(`<b>${d.split} · ${d.fold}</b> (${d.role}${d.ev ? ', Congress ' + d.ev : ''})<br>${d.base}: ${d3.format('.3f')(d.value)}`, e))
    .on('mouseleave', () => tip.hide());
});
viz.legend(svg, [['validation', color.val], ['test', color.test]], m.left, height - 8);
""", _dec_pts, height=380, width=880, opts=dict(splits=list(ALL_SPLITS)), name="X_val_test_by_split")

# %% [markdown]
# ### Recommendation: explicit rules applied to the recorded checks
#
# The recommendation is a short chain of rules. Each rule names the checks it reads, prints the values
# it used and its verdict, and says what it implies. A conclusion is labelled **robust** when at least two
# independent checks support it and none contradicts it (and no section disagrees on that quantity);
# otherwise it is **borderline** and the printout says what to watch. The split ranking is computed per
# task (checks are assigned to Task 1, Task 2 or both from their wording) and is re-run with the
# unclassified checks included. A pick is labelled ROBUST only if (a) both runs pick the same split, (b) the
# criterion that separates it from the runner-up comes from rules that are all robust, and (c) its
# model-selection checks were run on that split itself rather than inherited from its base split.
#
# | Rule | Question | If it fails |
# |---|---|---|
# | R1 | Any validity FAIL for the split (leakage, val does not predict test, too few test Congresses, censoring)? | drop the split for model selection |
# | R2 | Cold start: share of eval positives from legislators unseen in train (B1) vs with in-Congress history (B1b/B1c) | inductive features and/or the `+stream` protocol |
# | R3 | Does streaming eval-Congress history up to t0 change scores (D13, SPLIT_RESULTS) and keep val honest (D14)? | evaluate a temporal GNN with `+stream` |
# | R4 | Party regimes: do they change the effects (A5b, D26, C23b)? | evaluate per regime |
# | R4b | Does the split drop the eval Congress's regime from training (C8), and does that matter given R4? | prefer a window that keeps every regime |
# | R5 | Expanding vs sliding window, with and without recency weighting (D10, SPLIT_RESULTS, D_FAMILY_RESULTS, A14) | sliding window, or expanding + recency weights |
# | R6 | Val predicts test level and model choice (D1, D20, D14, D2b, D11/D12/D12b/D21/D23/D17), per split actually tested | report per-Congress scores; flag inferred evidence |
# | R7 | Fold stability and test size (D3, D4, D9, D27) | use several rolling test folds; lock the last Congress |
# | R8 | Drift FAILs exposed by the chosen split | remedies (below), not a different split |

# %%
# Rule engine: read CHECKS / SPLIT_RESULTS, apply R1-R8, rank splits per task and print the recommendation.
_DEC_C = {c["id"]: c for c in CHECKS}
_DEC_KEPT = checks_summary(requirements=False).set_index("id")


def _dc(cid):
    return _DEC_C.get(cid)


def _dpass(cid):
    """Overall verdict of a check (True / False / None) or 'missing'."""
    c = _dc(cid)
    return "missing" if c is None else c["passed"]


def _dsplit(cid, split):
    c = _dc(cid)
    if c is None or not c.get("by_split"):
        return None
    return c["by_split"].get(split)


def _dconflict(cid):
    return bool(cid in _DEC_KEPT.index and bool(_DEC_KEPT.loc[cid, "conflict"]))


def _dtask(row):
    """Task a check is about, from the wording of its condition: 'T1', 'T2' or 'both'. (The t0_task2 tag is not
    used: checks such as D9 / D26 carry it for their Task-2 part but also cover Task 1.)"""
    t = str(row["condition"]).lower()
    t1 = any(k in t for k in ("task 1", "task-1", "cosponsor"))
    t2 = any(k in t for k in ("task 2", "task-2", "advancement"))
    return "T1" if t1 and not t2 else "T2" if t2 and not t1 else "both"


DEC_RULES = []


def _drule(rid, question, evidence, verdict, implication, support, against=()):
    """Record one rule outcome. support / against = check ids whose result backs / contradicts the verdict."""
    support = [s for s in support if _dc(s) is not None]
    against = [a for a in against if _dc(a) is not None]
    conf = [s for s in list(support) + list(against) if _dconflict(s)]
    robust = len(support) >= 2 and not against and not conf
    DEC_RULES.append(dict(rule=rid, question=question, evidence=evidence, verdict=verdict, implication=implication,
                          support=", ".join(support), against=", ".join(against),
                          strength="robust" if robust else "borderline"))


# --- R1: validity filter (per task) ---------------------------------------------------------------
_dec_lbv = checks_by_split(kinds=("validity",), include_omitted=True)
_dec_lbu = checks_by_split(kinds=(None,), include_omitted=True)
_dec_task_of = {r["id"]: _dtask(r) for r in CHECKS}


def _dec_task_scores(lb, task):
    g = lb[lb.id.map(lambda i: _dec_task_of.get(i, "both") in (task, "both"))]
    out = {}
    for s in ALL_SPLITS:
        h = g[g.split == s]
        out[s] = dict(judged=int(h.judged.sum()), fail=int((h.split_passed == False).sum()),           # noqa: E712
                      fail_ids=", ".join(h.id[h.split_passed == False]))                                # noqa: E712
    return pd.DataFrame(out).T


DEC_TASKS = {"T1": "cosponsor_t0", "T2": "advancement"}
DEC_TS = {t: (_dec_task_scores(_dec_lbv, t), _dec_task_scores(_dec_lbu, t)) for t in DEC_TASKS}
for t, (sv, su) in DEC_TS.items():
    bad = sv[sv.fail > 0]
    _drule(f"R1-{t}", f"{t}: which splits have a validity FAIL?",
           "; ".join(f"{s}: {r.fail_ids}" for s, r in bad.iterrows()) or "none",
           f"drop {', '.join(bad.index) or 'none'}",
           "not usable for model selection / honest reporting of this task",
           support=sorted({i for r in bad.itertuples() for i in r.fail_ids.split(", ") if i}))

# --- R2: cold start -------------------------------------------------------------------------------
DEC_COLD_MAX = 0.02   # same bound as B1b / B1c (share of positives); B1 itself has no threshold (INFO)
_b1 = _dc("B1")
_b1_by = {k: v for k, v in ((_b1 or {}).get("by_split") or {}).items() if isinstance(v, (int, float))}
_b1_max = max(_b1_by.values()) if _b1_by else np.nan
_cold_big = bool(_b1_by) and _b1_max > DEC_COLD_MAX
_stream_ok = _dpass("B1b") is True and _dpass("B1c") in (True, "missing")
_drule("R2", "Cold start: are eval positives from legislators unseen in training?",
       f"B1 train-cold share of positives by split {', '.join(f'{k} {v:.1%}' for k, v in _b1_by.items())} "
       f"(bound {DEC_COLD_MAX:.0%}); B1b +stream = {_dc('B1b')['metric'] if _dc('B1b') else 'n/a'}, "
       f"B1c = {_dc('B1c')['metric'] if _dc('B1c') else 'n/a'}",
       ("cold start is LARGE for a transductive model without in-Congress history" if _cold_big else
        "cold start is small") + ("; resolved once eval-Congress history up to t0 is in the graph" if _stream_ok else ""),
       ("use inductive node features (party, chamber, state, committee, tenure) AND time-filtered in-Congress "
        "history (+stream); never rely on per-legislator embeddings learned only from training Congresses")
       if _cold_big else "a transductive model is acceptable",
       support=["B1", "B1b", "B1c"] if _stream_ok else ["B1"],
       against=[] if _stream_ok or not _cold_big else ["B1b"])

# --- R3: streaming protocol -----------------------------------------------------------------------
DEC_MDE = dict(globals().get("D_MDE", {"cosponsor_t0": 0.01, "cosponsor_t7": 0.01, "advancement": 0.01}))
_dec_bt = DEC_BT.set_index(["split", "task"])


def _dec_gain(task, base="S3"):
    try:
        return float(_dec_bt.loc[(base + "+stream", task), "test"] - _dec_bt.loc[(base, task), "test"])
    except KeyError:
        return np.nan


DEC_STREAM_GAIN = {t: _dec_gain(k) for t, k in DEC_TASKS.items()}
DEC_PREFER_STREAM = {t: bool(DEC_STREAM_GAIN[t] >= DEC_MDE[k]) and _dpass("D14") is not False
                     for t, k in DEC_TASKS.items()}
for t, k in DEC_TASKS.items():
    # Built outside the f-string: nested same-quote f-strings need Python 3.12+.
    _d13_note = f"; D13 = {_dec_fmt(_dc('D13')['metric'])}" if t == "T1" and _dc('D13') else ""
    _drule(f"R3-{t}", f"{t}: does eval-Congress history up to t0 (+stream) change the test score?",
           f"S3+stream - S3 test {DEC_PRIMARY[k]} (val-selected baseline) = {DEC_STREAM_GAIN[t]:+.4f} vs MDE "
           f"{DEC_MDE[k]}{_d13_note}; D14 (val honest under stream) = "
           f"{_dec_status(_dpass('D14'), False) if _dc('D14') else 'n/a'}",
           "yes: evaluate with +stream" if DEC_PREFER_STREAM[t] else "no material gain: plain or +stream are equivalent",
           "the GNN must see the eval Congress's own edges dated <= intro + t0 (time-filtered neighbor sampling)"
           if DEC_PREFER_STREAM[t] else "streaming is optional for this task; keep it for consistency with Task 1",
           support=(["D13", "D14"] + (["B1b"] if t == "T1" else [])) if DEC_PREFER_STREAM[t] else [])

# --- R4: party regimes ----------------------------------------------------------------------------
_c8_bad = [s for s in ALL_SPLITS if _dsplit("C8", s) is False]
_c23_bad = [s for s in ALL_SPLITS if _dsplit("C23b", s) is False]
_regime_stable = _dpass("A5b") is True and _dpass("D26") is True
_drule("R4", "Party control: do regime changes alter the effects a model learns?",
       f"A5b (majority effect matches same regime) = {_dec_status(_dpass('A5b'), False)}; D26 (regime weighting not "
       f"needed) = {_dec_status(_dpass('D26'), False)}; C23b (post-break test units) fails for {', '.join(_c23_bad) or 'none'}",
       "regime effects are stable given chamber-majority features" if _regime_stable else "regime changes effects: evaluate per regime",
       "include chamber-majority / sponsor-in-majority features; report test folds per regime",
       support=[i for i in ("A5b", "D26") if _dpass(i) is True],
       against=[i for i in ("A5b", "D26") if _dpass(i) is False])
# R4b: C8 says a split drops the eval Congress's regime from training. If R4 finds regime effects stable, losing a
# regime matters less, so A5b / D26 PASS count AGAINST "dropping a regime hurts" (the penalty is then borderline).
_drule("R4b", "Does the split drop the eval Congress's party regime from training, and does that matter?",
       f"C8 regime coverage fails for {', '.join(_c8_bad) or 'none'}; but R4 finds regime effects "
       f"{'stable' if _regime_stable else 'NOT stable'} (A5b {_dec_status(_dpass('A5b'), False)}, "
       f"D26 {_dec_status(_dpass('D26'), False)})",
       (f"mild penalty for {', '.join(_c8_bad)}" if _regime_stable else f"avoid {', '.join(_c8_bad)}") if _c8_bad
       else "every split keeps the eval regime",
       ("a missing regime is a coverage gap, not a demonstrated score loss; use it only as a tie-break"
        if _regime_stable else "keep every regime in training (expanding window)") if _c8_bad else "-",
       support=["C8"] if _c8_bad else [],
       against=[i for i in ("A5b", "D26") if _dpass(i) is True] if _c8_bad else [])

# --- R5: expanding vs sliding, with and without recency weighting ---------------------------------
DEC_TOL = {k: v / 2 for k, v in DEC_MDE.items()}


def _dec_win_diff(task, base, res):
    """Mean over folds of test(S3) - test(S2) for one baseline (same test bills per fold); NaN if missing."""
    m = DEC_PRIMARY[task]
    d = res[(res.task == task) & (res.metric == m) & (res.baseline == base) & (res.eval_role == "test")
            & res.split.isin(["S2", "S3"])].pivot_table(index="eval_congress", columns="split", values="value")
    return float((d["S3"] - d["S2"]).mean()) if {"S2", "S3"} <= set(d.columns) and len(d) else np.nan


_rw_base = {"cosponsor_t0": [b for b in ("sh_hl1", "rw_b0.3_hl1") if (DEC_RES_EXT.baseline == b).any()]}
_unw_base = {"cosponsor_t0": [b for b in ("sponsor_hist", "rw2hop") if (SPLIT_RESULTS.baseline == b).any()]}
DEC_WIN = {"T1 unweighted " + b: _dec_win_diff("cosponsor_t0", b, SPLIT_RESULTS) for b in _unw_base["cosponsor_t0"]}
DEC_WIN.update({"T1 recency " + b: _dec_win_diff("cosponsor_t0", b, DEC_RES_EXT) for b in _rw_base["cosponsor_t0"]})
_t2b = _bt2.baseline.get("S3") if "S3" in _bt2.index else None
if _t2b is not None:
    DEC_WIN[f"T2 {_t2b}"] = _dec_win_diff("advancement", _t2b, SPLIT_RESULTS)
for k in ("cosponsor_t0", "advancement"):
    if {"S2", "S3"} <= set(DEC_BT_RW[DEC_BT_RW.task == k].split):
        _r = DEC_BT_RW[DEC_BT_RW.task == k].set_index("split")
        DEC_WIN[f"{DEC_TASK_LABEL[k].split()[0]} val-picked incl. " + ("recency" if k == "cosponsor_t0" else "near-tie family")] = \
            float(_r.loc["S3", "test"] - _r.loc["S2", "test"])
_tol1, _tol2 = DEC_TOL["cosponsor_t0"], DEC_TOL["advancement"]
_unw = [v for k, v in DEC_WIN.items() if k.startswith("T1 unweighted") and np.isfinite(v)]
_rw = [v for k, v in DEC_WIN.items() if (k.startswith("T1 recency") or k == "T1 val-picked incl. recency") and np.isfinite(v)]
_t2w = [v for k, v in DEC_WIN.items() if k.startswith("T2") and np.isfinite(v)]
DEC_SLIDE_BETTER_UNW = bool(_unw) and max(_unw) < -_tol1          # unweighted: sliding wins by more than MDE/2
DEC_EXP_BETTER_RW = bool(_rw) and min(_rw) > _tol1                 # recency-weighted: expanding wins by more than MDE/2
DEC_SLIDE_BETTER_RW = bool(_rw) and max(_rw) < -_tol1
_t2_eq = bool(_t2w) and all(abs(v) <= DEC_MDE["advancement"] for v in _t2w)
_d10 = _dpass("D10")
_a14 = _dc("A14")
if DEC_EXP_BETTER_RW:
    _r5v = "expanding window + recency weighting beats sliding"
elif DEC_SLIDE_BETTER_RW:
    _r5v = "sliding window beats expanding even with recency weighting"
else:
    _r5v = ("window length is a wash ONLY with recency weighting; without it the sliding window is better"
            if DEC_SLIDE_BETTER_UNW else "window length does not matter")
_drule("R5", "Do older Congresses hurt (expanding S3 vs sliding S2), with and without recency weighting?",
       "test S3 - S2: " + ", ".join(f"{k} {v:+.4f}" for k, v in DEC_WIN.items() if np.isfinite(v))
       + f" (tolerance MDE/2 = {_tol1} T1, MDE = {DEC_MDE['advancement']} T2); D10 (recency-weighted not worse) = "
       f"{_dec_status(_d10, False)}; A14 distribution closeness favouring sliding = "
       f"{_dec_fmt(_a14['metric']) if _a14 else 'n/a'} (> 0.5 favours sliding, proxy only)",
       _r5v + ("; Task 2 is equivalent within the MDE" if _t2_eq else ""),
       ("CONDITIONAL: use an expanding window only together with recency weighting (decay chosen on val); with "
        "unweighted count / graph features S2 is as good or better. The window choice gives no score reason to "
        "prefer S3 over S2; only R4b (regime coverage) and fold evidence do") if DEC_SLIDE_BETTER_UNW and not DEC_EXP_BETTER_RW
       else "keep all Congresses 108..c-1 with recency weights" if DEC_EXP_BETTER_RW else "train on the last k Congresses",
       support=(["D10"] if _d10 is True else []),
       against=(["A14"] if _a14 is not None and isinstance(_a14["metric"], (int, float)) and _a14["metric"] > 0.5 else []))
DEC_R5_TEXT = _r5v

# --- R6: val predicts test ------------------------------------------------------------------------
# Selection checks (does val pick the test-best model?) and level checks, read PER SPLIT from by_split: evidence
# from a base split is not transferred to its +stream variant.
DEC_SEL_IDS = [i for i in ("D11", "D12", "D12b", "D21", "D23", "D17") if _dc(i)]
_lvl1 = [i for i in ("D1", "D20", "D14", "D22") if _dc(i)]


def _dec_judged(ids, split, task):
    """Checks among ids that judged this split (by_split True/False; numeric = metric only) and are about task."""
    out = {}
    for i in ids:
        if _dec_task_of.get(i, "both") not in (task, "both"):
            continue
        v = _dsplit(i, split)
        if isinstance(v, (bool, np.bool_)):
            out[i] = bool(v)
    return out


def _dec_sel_status(split, task):
    """(own selection checks {id: pass}, inherited-from-base {id: pass} or {}) for one split."""
    own = _dec_judged(DEC_SEL_IDS, split, task)
    base = SPLIT_VARIANTS[split]["base"] if split in SPLIT_VARIANTS else None
    inh = _dec_judged(DEC_SEL_IDS, base, task) if (base and not own) else {}
    return own, inh


_r6_splits = [s for s in ALL_SPLITS if split_kind(s) in BY_CONGRESS_KINDS and len(split_eval_congresses(s, ("test",))) > 1]
_r6_txt, _r6_inf = [], []
for _s in _r6_splits:
    own, inh = _dec_sel_status(_s, "T1")
    lv = _dec_judged(_lvl1, _s, "T1")
    _r6_txt.append(f"{_s}: level {','.join(f'{i}={_dec_status(v, False)}' for i, v in lv.items()) or 'none'}; selection "
                   + (",".join(f"{i}={_dec_status(v, False)}" for i, v in own.items()) if own else
                      f"NOT RUN (inferred from {SPLIT_VARIANTS[_s]['base']}: {','.join(inh) or 'none'})" if inh else "none"))
    if not own and inh:
        _r6_inf.append(_s)
_sel_ok = all(_dpass(i) is True for i in DEC_SEL_IDS if _dec_task_of.get(i) != "T2")
_tested = [x for x in _r6_splits if _dec_sel_status(x, "T1")[0]]
_drule("R6-T1", "T1: on the splits where selection was tested, does val predict the test level and pick the test-best model?",
       "; ".join(t for x, t in zip(_r6_splits, _r6_txt) if x in _tested),
       (f"yes for {', '.join(_tested)} (by-Congress val is a faithful proxy there)"
        if all(_dpass(i) is True for i in _lvl1) and _sel_ok else "partly"),
       "tune / early-stop on val Congress c-1, refit on 108..c-1 (D19), test on c",
       support=[i for i in _lvl1 + DEC_SEL_IDS if _dpass(i) is True and _dec_task_of.get(i) != "T2"],
       against=[i for i in _lvl1 + DEC_SEL_IDS if _dpass(i) is False and _dec_task_of.get(i) != "T2"])
if _r6_inf:
    # Variants judged on the level only: their selection evidence is inherited, so this conclusion is borderline by
    # construction (support counts only checks that judged the variant itself; the inherited ones are 'against').
    _inf_lvl = sorted({i for x in _r6_inf for i, v in _dec_judged(_lvl1, x, "T1").items() if v})
    _inf_sel = sorted({i for x in _r6_inf for i in _dec_sel_status(x, "T1")[1]})
    _drule("R6-T1s", f"T1: is val a faithful proxy under {', '.join(_r6_inf)} (streaming)?",
           "; ".join(t for x, t in zip(_r6_splits, _r6_txt) if x in _r6_inf),
           "level carries over (tested); model selection NOT tested, inferred from the base split",
           f"before trusting model choices made under {', '.join(_r6_inf)}, rerun D11 / D17 (sponsor_hist and rw2hop "
           "families) on its folds; until then report that selection reliability is inferred from plain S3",
           support=[i for i in _inf_lvl if i in ("D14",)], against=_inf_sel)
_d2b_bad = [s for s in ALL_SPLITS if _dsplit("D2b", s) is False]
_drule("R6-T2", "T2: does the val Congress predict the test AUROC level and the model choice?",
       f"level: D2b = {_dec_status(_dpass('D2b'), False)} (fails for {', '.join(_d2b_bad) or 'none'}); D4 std = "
       f"{_dec_fmt(_dc('D4')['metric']) if _dc('D4') else 'n/a'}; selection D12/D12b = "
       f"{_dec_status(_dpass('D12'), False)}/{_dec_status(_dpass('D12b'), False)}",
       ("model ranking transfers but the AUROC LEVEL does not transfer between Congresses" if _dpass("D2b") is False
        else "level and ranking transfer"),
       ("select Task-2 models on val (rankings transfer), but report per-Congress AUROC and the fold mean and never quote "
        "a val AUROC as the expected test level (D2b). AUROC is invariant to monotone recalibration, so no recalibration "
        "fixes a D2b failure; if calibrated probabilities are needed, recalibrate on the eval Congress's already-resolved "
        "bills (D2b), not on Congress c-1") if _dpass("D2b") is False else
       "select on val and report per-Congress AUROC",
       support=[i for i in ("D12", "D12b", "D6") if _dpass(i) is True] + (["D2b", "D4"] if _dpass("D2b") is False else []),
       against=[])

# --- R7: fold stability, test size, locked test ---------------------------------------------------
_d9_bad = [s for s in ALL_SPLITS if _dsplit("D9", s) is False]
_fp = globals().get("D_FINAL_PROTOCOL", {})
_drule("R7", "Is one test Congress enough, and can the last Congress be locked away?",
       f"D9 single-test-Congress miss risk fails for {', '.join(_d9_bad) or 'none'}; D3 T1 fold CV = "
       f"{_dec_fmt(_dc('D3')['metric']) if _dc('D3') else 'n/a'}; D27 locked test = {_dec_status(_dpass('D27'), False)} "
       f"(locked {_fp.get('locked_test', 'n/a')})",
       "use rolling one-step-ahead test folds" + (f" and lock {_fp.get('locked_test')} as the final test"
                                                  if _dpass("D27") is True else ""),
       "compare architectures on the mean over development folds (paired, same bills); touch the locked Congress once",
       support=[i for i in ("D9", "D27", "D17") if _dc(i) and _dpass(i) in (True, False)],
       against=[])

# --- R8: drift remedies (FAILs of kind 'drift') ----------------------------------------------------
DEC_REMEDY = {
    "label_validity_t2": "Task-2 label misses more chamber-passed bills lately: also report label_alt (reported OR passed)",
    "label_rate_t2_chamber": "per-chamber advancement rate moves: chamber-specific intercepts; if probabilities are needed, "
                             "recalibrate on the eval Congress's already-resolved bills (as D2b), not on c-1",
    "label_rate_t2": "advancement base rate moves: use AUROC and AP-lift, not raw AP or accuracy; recalibrate (if needed) on "
                     "the eval Congress's resolved bills",
    "cosp_density": "positive density moves: compare ranking metrics per Congress; do not compare raw recall across Congresses",
    "orig_share": "original-cosponsor share moves: t0=0 vs t0=7 positives differ by Congress; report both t0",
    "bill_volume": "bill volume moves: normalise counts per Congress (degree / activity features as rates)",
    "model_auroc_transfer_t2": "Task-2 model degrades forward: retrain every Congress, recency-weight, report per Congress",
    "cand_fail_count": "several drifts at once: rely on rolling folds rather than one test Congress",
    "lag_dist": "cosponsor timing moves: window labels at fixed t0 (already) and monitor lag per Congress",
    "sponsor_overlap": "new sponsors: inductive sponsor features plus in-Congress history (+stream)",
}
# Remedies keyed by check id, for drift checks without a quantity tag.
DEC_REMEDY_ID = {
    "D2b": "Task-2 AUROC level does not transfer between Congresses: report per-Congress AUROC and the fold mean, never a "
           "val AUROC as the expected test level (AUROC ignores recalibration, so recalibrating cannot fix this); for "
           "calibrated probabilities recalibrate on the eval Congress's already-resolved bills, not on c-1",
}
DEC_REMEDY_DEFAULT = ("level shift between Congresses: report per Congress; if calibrated scores are needed, recalibrate on "
                      "the eval Congress's already-resolved data, not on the previous Congress")
_dec_drift = checks_summary(requirements=False, kinds=("drift",))
_dec_drift = _dec_drift[_dec_drift.passed.map(lambda v: v is False or v is np.False_)]
DEC_DRIFT_FAILS = [(r.id, r.quantity if isinstance(r.quantity, str) else None) for r in _dec_drift.itertuples()]
_drule("R8", "Which drift checks fail (remedies, not a different split)?",
       ", ".join(f"{i}" + (f" ({q})" if q else "") for i, q in DEC_DRIFT_FAILS) or "none",
       f"{len(DEC_DRIFT_FAILS)} drift FAILs", "see the remedy list below",
       support=[i for i, _ in DEC_DRIFT_FAILS])

DEC_RULES = pd.DataFrame(DEC_RULES)
for r in DEC_RULES.itertuples():
    print(f"[{r.rule}] {r.question} -> {r.verdict}  <{r.strength.upper()}; support: {r.support or '-'}"
          f"{'; against: ' + r.against if r.against else ''}>\n    evidence: {_dec_short(r.evidence, 460)}\n"
          f"    implies : {r.implication}")

# %%
# Rank the surviving splits per task (validity only, then with unclassified checks) and print the decision.


DEC_MIN_JUDGED_FRAC = 0.5   # a split judged by < half as many checks as the best-covered split has too little evidence


def _dec_penalties(s):
    """Rule penalties of a split: {rule: 0/1}. R4b = C8 says the eval regime is missing from training. R5 = the
    window comparison favours the other window by more than MDE/2 for a recency-weighted model (else no penalty)."""
    k = split_kind(s)
    return {"R4b": int(_dsplit("C8", s) is False),
            "R5": int((k == "sliding" and DEC_EXP_BETTER_RW) or (k in ("rolling", "expanding") and DEC_SLIDE_BETTER_RW))}


def _dec_protocol(s, task):
    """'match' / 'plain' (a +stream variant exists and R3 asks for it) / 'no +stream variant evaluated' / 'stream'."""
    want = DEC_PREFER_STREAM[task]
    if s.endswith("+stream"):
        return "match" if want else "stream (not needed)"
    if not want:
        return "match"
    return "plain (stream variant exists)" if s + "+stream" in SPLIT_VARIANTS else "no +stream variant evaluated"


def _dec_rank(task, with_unclassified=False):
    sv, su = DEC_TS[task]
    k = DEC_TASKS[task]
    nj = sv.judged.astype(int) + (su.judged.astype(int) if with_unclassified else 0)
    min_j = int(np.ceil(DEC_MIN_JUDGED_FRAC * nj.max()))
    rows = []
    for s in ALL_SPLITS:
        bt = _dec_bt.loc[(s, k)] if (s, k) in _dec_bt.index else None
        fail = int(sv.loc[s, "fail"]) + (int(su.loc[s, "fail"]) if with_unclassified else 0)
        judged = int(sv.loc[s, "judged"]) + (int(su.loc[s, "judged"]) if with_unclassified else 0)
        proto = _dec_protocol(s, task)
        pen = _dec_penalties(s)
        own, inh = _dec_sel_status(s, task)
        rows.append(dict(split=s, fail=fail, judged=judged, low_evidence=int(judged < min_j),
                         protocol_mismatch=int(proto not in ("match",)), protocol=proto,
                         rule_penalty=sum(pen.values()), pen_rules=",".join(r for r, v in pen.items() if v) or "-",
                         sel_checks=len(own), sel_inferred=int(bool(inh)),
                         folds=int(bt["n_folds"]) if bt is not None else 0,
                         test=float(bt["test"]) if bt is not None else np.nan,
                         drift_fail=int(DEC_SCORE.loc[s, "drift_fail"])))
    r = pd.DataFrame(rows)
    # order: no validity fails, enough evidence, the protocol R3 asks for (a split with no +stream variant is treated
    # like a plain split: its streaming behaviour is unknown), R4b/R5 penalties, more checks judged, more test folds,
    # fewer drift fails. The test score is shown but not used: it differs by protocol and is not a validity criterion.
    r = r.sort_values(["fail", "low_evidence", "protocol_mismatch", "rule_penalty", "judged", "folds", "drift_fail"],
                      ascending=[True, True, True, True, False, False, True], kind="stable").reset_index(drop=True)
    return r


DEC_RECO = {}
_crit = ["fail", "low_evidence", "protocol_mismatch", "rule_penalty", "judged", "folds"]
for t, k in DEC_TASKS.items():
    r0, r1 = _dec_rank(t), _dec_rank(t, with_unclassified=True)
    top0, top1 = r0.split.iloc[0], r1.split.iloc[0]
    clean = r0[(r0.fail == 0) & (r0.low_evidence == 0)]
    runner = r0.split.iloc[1] if len(r0) > 1 else None
    tie = runner is not None and r0.iloc[0][_crit].tolist() == r0.iloc[1][_crit].tolist()
    DEC_RECO[t] = dict(split=top0, robust_to_unclassified=top0 == top1, alt=top1, runner_up=runner, tie=tie,
                       clean=list(clean.split), rank=r0)
    print(f"\n=== {t} ({k}) ranking (validity checks for this task + shared ones) ===")
    print(r0.head(5).round(3).to_string(index=False))
    print(f"With unclassified checks included the top split is {top1} "
          f"({'same' if top0 == top1 else 'DIFFERENT -> borderline'}); usable (0 validity fails, enough evidence): "
          f"{', '.join(clean.split)}; too little evidence: {', '.join(r0.split[(r0.fail == 0) & (r0.low_evidence == 1)]) or '-'}")

print("\n" + "=" * 100 + "\nRECOMMENDATION\n" + "=" * 100)
_lock = _fp.get("locked_test") if _dpass("D27") is True else None
_rs = DEC_RULES.set_index("rule").strength
for t, k in DEC_TASKS.items():
    rc = DEC_RECO[t]
    s = rc["split"]
    base = SPLIT_VARIANTS[s]["base"] if s in SPLIT_VARIANTS else s
    d = SPLIT_DEFS[base]
    bt = _dec_bt.loc[(s, k)]
    risks = [i for i in (DEC_SC_D.loc[s, "fail_ids"] or "").split(", ") if i and _dec_task_of.get(i, "both") in (t, "both")]
    unc = [i for i in (DEC_TS[t][1].loc[s, "fail_ids"] or "").split(", ") if i]
    # which ranking criterion separates the pick from the runner-up, and which rule(s) produced that criterion
    _r = rc["rank"].set_index("split")
    ru = rc["runner_up"]
    _why = next((c for c in _crit if _r.loc[s, c] != _r.loc[ru, c]), "drift_fail")
    if _why == "rule_penalty":
        _ps, _pr = _dec_penalties(s), _dec_penalties(ru)
        _why_rules = [x for x in _ps if _ps[x] != _pr[x]]
    else:
        _why_rules = {"fail": [f"R1-{t}"], "protocol_mismatch": [f"R3-{t}"]}.get(_why, [])
    _weak = [x for x in _why_rules if _rs.get(x) != "robust"]
    own, inh = _dec_sel_status(s, t)
    reasons = []
    if not rc["robust_to_unclassified"]:
        reasons.append(f"including unclassified checks would pick {rc['alt']}")
    if rc["tie"]:
        reasons.append(f"tied with {ru} on every ranking criterion except the drift count")
    if _weak:
        reasons.append(f"separated from {ru} only by borderline rule(s) {', '.join(_weak)}")
    if _why in ("judged", "folds", "drift_fail"):
        reasons.append(f"separated from {ru} only by '{_why}' (amount of evidence, not a rule)")
    if not own:
        reasons.append("selection reliability (does val pick the test-best model?) was not tested on this split"
                       + (f"; inferred from plain {base} ({', '.join(inh)} pass there)" if inh else ""))
    strength = "BORDERLINE" if reasons else "ROBUST"
    print(f"\n{t} ({k}): use {s} = {d['name']}{' + streaming eval-Congress history' if s.endswith('+stream') else ''}  [{strength}]")
    print(f"  definition : {SPLIT_VARIANTS[s]['desc'] if s in SPLIT_VARIANTS else d['desc']}")
    if d["kind"] in BY_CONGRESS_KINDS and "folds" in d:
        dev = [c for c in d["folds"] if c != _lock]
        print(f"  folds      : development test Congresses {dev[0]}-{dev[-1]} (val = c-1, train = "
              f"{'108..c-2' if d['kind'] == 'rolling' else f'last {SLIDE_K} before val'})"
              + (f"; LOCKED final test {_lock} (train {_rng_label(_fp.get('locked_train', []))}, touched once); "
                 f"prospective {_fp.get('prospective_test', IN_PROGRESS)}" if _lock else ""))
    print(f"  backtest   : val-picked '{bt['baseline']}' (unweighted SPLIT_RESULTS baselines) test {DEC_PRIMARY[k]} "
          f"{bt['test']:.3f} over {int(bt['n_folds'])} fold(s), fold sd {bt['fold_sd']:.3f}, mean |val-test| = "
          f"{bt['gap_ratio']:.2f}x the test CI half-width, {int(bt['n_eval']):,} test bills")
    _ok = _dec_lbv[(_dec_lbv.split == s) & (_dec_lbv.split_passed == True)].id                    # noqa: E712
    print(f"  relies on  : validity checks passed for {s} ({t} + shared): "
          f"{', '.join(i for i in _ok if _dec_task_of.get(i, 'both') in (t, 'both')) or '-'}")
    print(f"  selection  : " + (f"tested on {s}: {', '.join(f'{i}={_dec_status(v, False)}' for i, v in own.items())}" if own else
                                f"NOT tested on {s}; selection reliability under {s} is INFERRED from plain {base} "
                                f"({', '.join(f'{i}={_dec_status(v, False)}' for i, v in inh.items()) or 'nothing'}). "
                                f"Rerun D11 / D17 (sponsor_hist and rw2hop families) on {s} folds to confirm"))
    print(f"  risks      : drift FAILs on this split for {t} -> {', '.join(risks) or 'none'}"
          + (f"; unclassified fails -> {', '.join(unc)}" if unc else ""))
    _ru_note = (f"{ru} (loses on '{_why}': {_r.loc[ru, _why]} vs {_r.loc[s, _why]}"
                + (f" [{_r.loc[ru, 'protocol']}]" if _why == "protocol_mismatch" else "")
                + (f"; decided by {', '.join(f'{x} ({_rs.get(x)})' for x in _why_rules)}" if _why_rules else "")
                + f"; runner-up test {_r.loc[ru, 'test']:.3f} vs pick {_r.loc[s, 'test']:.3f})")
    print(f"  runner-up  : {_ru_note}")
    for x in reasons:
        print(f"  borderline : {x}")
    # Window condition (R5): is the pick an expanding window whose advantage depends on recency weighting?
    _wt = {kk: vv for kk, vv in DEC_WIN.items() if kk.startswith(t) and np.isfinite(vv)}
    if split_kind(s) in ("rolling", "expanding") and "S2" in rc["clean"] and _wt:
        _wtxt = ", ".join(f"{kk} {vv:+.4f}" for kk, vv in _wt.items())
        if t == "T1" and DEC_SLIDE_BETTER_UNW and not DEC_EXP_BETTER_RW:
            print(f"  condition  : expanding only WITH recency weighting; otherwise S2 is as good or better (R5, S3 minus S2: "
                  f"{_wtxt}). S2 is a legitimate alternative"
                  + (" (no +stream variant of S2 evaluated yet)" if DEC_PREFER_STREAM[t] and "S2" in _dec_nostream else ""))
        elif t != "T1":
            _mx = max(abs(v) for v in _wt.values())
            print(f"  condition  : S3 vs S2 test (S3 minus S2): {_wtxt}; "
                  + (f"|diff| <= MDE {DEC_MDE[k]}, so the windows are equivalent on score and S2 is a legitimate alternative"
                     if _mx <= DEC_MDE[k] else "the window matters on score: prefer the better one")
                  + "; no recency-weighted Task-2 baseline was evaluated")
    if t == "T2" and _dpass("D2b") is False:
        print("  reporting  : AUROC level does not transfer between Congresses (D2b): report per-Congress AUROC and the fold "
              "mean; never quote a val AUROC as the expected test level")
    _fids, _uids = DEC_TS[t][0]["fail_ids"], DEC_TS[t][1]["fail_ids"]
    print("  not for selection: " + ", ".join(
        f"{x} ({_fids[x] or 'too little evidence' + (', unclassified fail ' + _uids[x] if _uids[x] else '')})"
        for x in ALL_SPLITS if x not in rc["clean"]))
    DEC_RECO[t].update(strength=strength, reasons=reasons)

# One pipeline for both tasks? Use Task 1's protocol for Task 2 if it is usable there and costs < 1 MDE.
_s1, _s2 = DEC_RECO["T1"]["split"], DEC_RECO["T2"]["split"]
if _s1 != _s2:
    _g = float(_dec_bt.loc[(_s1, "advancement"), "test"] - _dec_bt.loc[(_s2, "advancement"), "test"])
    _share = _s1 in DEC_RECO["T2"]["clean"] and _g > -DEC_MDE["advancement"]
    print(f"\nOne pipeline for both tasks: {_s1} for Task 2 is {'usable' if _s1 in DEC_RECO['T2']['clean'] else 'NOT usable'} "
          f"and changes the Task-2 test {DEC_PRIMARY['advancement']} by {_g:+.4f} (MDE {DEC_MDE['advancement']}) -> "
          + (f"run both tasks on {_s1} (same folds; Task 2 simply gains little from streaming)" if _share
             else f"keep {_s2} for Task 2")
          + (f". Note: Task-2 selection checks were not run on {_s1} either (inferred from plain "
             f"{SPLIT_VARIANTS[_s1]['base'] if _s1 in SPLIT_VARIANTS else _s1})" if _share and not _dec_sel_status(_s1, 'T2')[0] else ""))

print("\nRemedies for the drift the chosen splits expose (R8):")
for i, q in DEC_DRIFT_FAILS:
    print(f"  {i:5} {q or '(no quantity)':24} -> {DEC_REMEDY_ID.get(i) or DEC_REMEDY.get(q, DEC_REMEDY_DEFAULT)}")
print("\nRobust vs borderline conclusions:")
print("  picks     : " + "; ".join(f"{t} {DEC_RECO[t]['split']} [{DEC_RECO[t]['strength']}]" for t in DEC_TASKS))
for lab in ("robust", "borderline"):
    rr = DEC_RULES[DEC_RULES.strength == lab]
    print(f"  {lab:10}: " + "; ".join(f"{r.rule} ({r.verdict})" for r in rr.itertuples()))
# Evidence gaps the decision depends on (computed from what was and was not evaluated).
_gaps = []
_rw_missing = [x for x in ALL_SPLITS if split_kind(x) in BY_CONGRESS_KINDS and x not in DEC_FAM_SPLITS]
if _rw_missing:
    _gaps.append(f"recency-weighted baselines (e.g. sh_hl1) exist only for {', '.join(DEC_FAM_SPLITS)}; not evaluated for "
                 f"{', '.join(_rw_missing)} (add them to SPLIT_RESULTS so a recency-weighted val pick is possible there)")
if DEC_PREFER_STREAM["T1"] and _dec_nostream:
    _gaps.append(f"no +stream variant for {', '.join(_dec_nostream)}: Task-1 standing under streaming is unknown")
for t in DEC_TASKS:
    _own, _inh = _dec_sel_status(DEC_RECO[t]["split"], t)
    if not _own:
        _gaps.append(f"{t} pick {DEC_RECO[t]['split']}: no model-selection check (D11 / D17 / D21 / D23) run on its folds")
print("\nEvidence gaps (close these before treating the borderline picks as settled):")
for x in _gaps or ["none"]:
    print(f"  - {x}")
print("\nMonitor on every new fold / Congress: " + ", ".join(sorted({q for _, q in DEC_DRIFT_FAILS if q}))
      + "; plus the validity checks the choice relies on (rerun this notebook when a Congress completes).")

# %% [markdown]
# ### Practical guidance for the GNN
#
# Read this together with the RECOMMENDATION printout above; the guidance assumes its rules came out the
# way they do in the current run (re-check if a rerun flips a verdict).
#
# - **Folds.** Develop on rolling-origin folds: for each development test Congress *c*, train on 108..c-2,
#   tune and early-stop on *c-1*, refit on 108..c-1 with the tuned hyperparameters, and score *c*. Compare
#   architectures on the mean over the development folds (paired, same bills), never on one Congress. Keep
#   the last complete Congress (`D_FINAL_PROTOCOL['locked_test']`) as a locked test you score once, and treat
#   the in-progress 119th as a prospective check only. A single by-Congress split (train 108-115 / val 116 /
#   test 117 or 118) is fine for quick debugging, but R7/D9 show one test Congress cannot settle close model
#   comparisons.
# - **Inductive, not purely transductive.** Many eval positives come from legislators absent from the
#   training Congresses (R2). Give legislator nodes inductive features (party as of the Congress, chamber,
#   state, committee seats, tenure, majority status) and do not depend on per-node ID embeddings learned only
#   from the training Congresses. GraphSAGE-style aggregation or a heterogeneous GNN with feature encoders
#   fits; a lookup-table embedding model does not.
# - **Time-filtered neighbor sampling (the `+stream` protocol).** For a bill scored at t0, the graph may
#   contain every edge dated <= introduced_date + t0: earlier Congresses plus the eval Congress's own
#   sponsorships and cosponsorships so far. Never use the eval Congress's final graph (B10), committee edges
#   recorded after t0 (B7), related-bill links not yet known (B6b), or snapshot count fields (C11). Build
#   training examples the same way, from as-of-t0 snapshots (B12), so train and eval inputs match.
# - **Window and weighting (conditional).** The expanding window is only as good as the sliding one *if*
#   the model down-weights old Congresses (R5/D10). With unweighted count or graph features, the last-3-
#   Congress sliding window (S2) scores as well or better. So: if you use an expanding window, add a recency
#   weight or edge-time decay and choose its half-life on validation. If you do not, S2 is a legitimate
#   choice. The remaining arguments for the expanding window are weak (R4b: S2 leaves the eval Congress's
#   party regime out of training, but regime effects look stable, A5b/D26) or about evidence (more checks
#   were run on S3). Add chamber-majority / sponsor-in-majority features rather than regime-specific weighting (R4/D26).
# - **Model selection under streaming.** The selection checks (D11/D17/D21/D23) were run on plain S3
#   folds, not on S3+stream. Until they are rerun under streaming, treat "val picks the test-best model" for
#   the streaming protocol as inferred, not tested. D14 only shows that the score *level* carries over.
# - **Task 2.** Pick models on validation (rankings transfer, D12/D12b). The AUROC *level* does not carry
#   over from one Congress to the next (D2b), so report per-Congress AUROC plus the fold mean, and never quote
#   a val AUROC as the expected test level. AUROC does not change under monotone recalibration, so
#   recalibrating cannot fix this. If you need calibrated probabilities, recalibrate on the eval Congress's
#   already-resolved bills (labels are known 180 days after introduction), not on Congress c-1. Hide the
#   policy area for bills younger than its availability lag (C18/C25). Task 2 can share Task 1's folds.
# - **Monitor.** Every quantity in the "Monitor" line drifts between Congresses. Rerun this notebook when a
#   Congress completes. If a validity check the choice relies on flips, the ranking changes automatically.
