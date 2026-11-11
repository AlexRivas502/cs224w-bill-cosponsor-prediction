# Split-analysis notebook: conventions for builder and challenger agents

Repo root: `/Users/alex_rivas/Documents/Stanford/CS224W/Final Project/cs224w-bill-cosponsor-prediction`
(quote paths: they contain spaces). Final deliverable: `notebooks/01_split_analysis.ipynb`, assembled
from `tools/split_notebook/cells/*.py` (jupytext percent format: `# %%` code cells, `# %% [markdown]`
markdown cells whose lines start with `# `).

## Files and ownership
| File | Owner | Topic |
|---|---|---|
| `cells/00_setup.py` | orchestrator / setup agent only | imports, loaders, shared task definitions, drift stats, check registry, D3 helpers |
| `cells/10_targets.py` | builder A | label / target and covariate drift across Congresses |
| `cells/20_graph.py` | builder B | node overlap, cold start, degree & schema stability, cross-split edges |
| `cells/30_temporal.py` | builder C | within-Congress timing, regime changes, censoring, snapshot leakage, data latency |
| `cells/40_splits.py` | builder D | candidate splits + baseline backtest (stability of scores across splits/folds) |
| `cells/90_decision.py` | synthesis agent (final round) | collects all checks, scores candidate splits, recommendation |

Edit ONLY the file you own. Never edit `00_setup.py`, other sections, the harness, or anything under
`parsed/`, `billstatus_bulk/`, `congress_api/`. If you need a shared helper changed, put a request in
your returned `setup_requests` instead.

## Test before you return (mandatory)
From the repo root:
```bash
uv run --no-project --python 3.13 --with pandas --with pyarrow --with scipy \
    python tools/split_notebook/run_cells.py 20_graph.py        # setup + your section
```
- Prints OK/FAIL per section and chart results; details in `tools/split_notebook/out/<section>/report.json`
  (e.g. `out/20_graph/report.json`), your printed output in `out/<section>/<section>.stdout.txt`
  (read it: that is what the user sees). A full run (no arguments) writes to `out/all/`.
- Charts are executed in Node against a minimal DOM (`js_check.js`): thrown errors, missing `<svg>`, or
  NaN/undefined/Infinity in SVG attributes fail the run. Real browsers can't run in this sandbox.
- A section must run in under ~90 s and print a readable amount (aim < 250 lines of stdout).
- Iterate until your section is OK and your charts pass. Do not return a failing section.
- Other builders run the harness at the same time; each run writes to its own `out/<section>/` folder.

## Shared API from 00_setup.py (already in scope; do not redefine)
- Constants: `CONGRESSES` = 108..118 (complete), `IN_PROGRESS` = 119 (exclude from splits; may be shown
  for context), `LAW_TYPES`, `WINDOW_DAYS`=30, `ADV_WINDOW_DAYS`=180, `PARTY_CODE`, `congress_start(c)`,
  `congress_end(c)`, `PALETTE`, `PARTY_COLORS`, `ROOT`, `PARSED`.
- `load(table, columns_tuple=None)` -> cached DataFrame (`congress` is int; `*_date` columns datetime64).
  Treat it as read-only (`.copy()` before mutating). Tables: bills, bill_cosponsors, bill_committees,
  bill_subjects, bill_actions, bill_recorded_votes, bill_related, bill_titles, bill_summaries,
  bill_text_versions, bill_cbo_estimates, congresses, congress_sessions, members (per congress:
  bioguide_id, party (full name), state, district, chamber in {'House of Representatives','Senate'}),
  legislators, member_terms, member_party_history, committees, committee_history, house_votes,
  house_vote_members (118th+ only). Inspect columns with `load(t).dtypes` if needed.
- `bills` (all bills + `law_capable`, `chamber` (same labels as members), `day_of_congress`,
  `policy_area_harm` = CRS policy area with categories absent from any of 108-118 and missing labels folded
  into "(unstable/missing)"; folded list in `PA_UNSTABLE`). bills.sponsor_party is a code: D, R, I, ID, L.
  Use `policy_area_harm` instead of rebuilding a harmonised taxonomy locally.
- `members` (+ `party_code` = the member's LATEST party (legacy; do not use as a per-Congress feature),
  `party_code_asof` = party as of that Congress from member_party_history (use this for rosters, node
  features, party-control logic), `caucus` = D/R; independents assigned from data (party whose bills they
  cosponsor more), 'other' on exact ties). `chamber` NaNs are filled from member_terms (no row lacks a
  chamber now). Note: the as-of column is named `party_code_asof`, not `party_asof`, so it does not clash
  with section C's local `party_asof` merge.
- `in_office_at(date, voting_only=True, slack_days=30)` -> members in office on a date (bioguide_id,
  congress, chamber, state_code, district); year-granular terms refined with activity dates; delegates
  excluded by default. `congress_midpoint(c)` = congress_start(c) + 365 days. Use for point-in-time
  eligible pools (e.g. `in_office_at(congress_start(c) + pd.Timedelta(days=t0))`).
- `ROSTER_ASOF_ALL`: shared point-in-time roster, one row per (month, member in office on day
  `ROSTER_REF_DAY`=15 of that month), INCLUDING Delegates / Resident Commissioners (column `voting` = False for
  them): month, congress, chamber, bioguide_id, caucus, party_code_asof, state_code, district, voting
  (Jan 2003 .. snapshot). This is the Task-1 eligible pool (task definition: eligible same-chamber legislators;
  delegates can cosponsor and cast ~96% of the positives outside the voting-only pool, C17). `ROSTER_ASOF` =
  its voting rows only (same columns minus `voting`, unchanged from before): use it for seat / majority counts.
  `roster_asof(date, include_delegates=True)` = rows of that month (ROSTER_ASOF_ALL by default). Use the
  delegate-inclusive pool for any legislator candidate pool (A, D) instead of a full-Congress roster, so every
  section ranks the same pool (D's `_ROSTER` / `_RA` should switch to ROSTER_ASOF_ALL).
- `SNAPSHOT_DATE` = max bills.update_date (day); `SNAPSHOT_LATEST_INTRO` = latest introduced_date.
  `PA_MISSING_AT_T0`: share of in-progress (119th) law-capable bills without a policy area by age at the
  snapshot (index age_bucket 0-7d / 8-30d / 31-90d / 91-180d / >180d; columns n, n_missing, share_missing).
  `PA_MISS_T0` / `PA_MISS_T0_N` = the 0-7d bucket. Use these instead of local estimates of
  policy-area-at-t0 missingness (small-sample proxy; say so in `detail`).
- `cosp`: one row per active (non-withdrawn) cosponsorship: bill_id, bioguide_id, party (code),
  sponsorship_date, is_original, congress, introduced_date, sponsor_bioguide_id, sponsor_party,
  chamber, bill_type, law_capable, `lag_days` (sponsorship - introduction).
- `window_targets(t0, horizon=30, law_capable_only=True, point_in_time=False, include_delegates=True)` -> per
  bill: n_pos (new cosponsors in (t0, t0+30]), n_seen (cosponsors by t0), pool, eligible, censored. Task 1 uses
  t0=0 and t0=7. Default pool = everyone who served in the chamber that Congress (legacy); `point_in_time=True`
  uses `ROSTER_ASOF_ALL` (delegates included; `include_delegates=False` -> voting-only `ROSTER_ASOF`) in the
  month of introduced_date + t0 (adds `pool_month`; eligible clipped at 0).
  Prefer point_in_time=True for new work; the pool is smaller for ~95% of bills.
- `adv` = `build_advancement_labels()` -> per law-capable bill: label (reported within 180 d),
  censored, reported_date, sponsor_party, policy_area, policy_area_harm, chamber, `passed_origin_date`
  (first 'Passed/agreed to in <origin chamber>' action) and `label_alt` (reported OR passed the origin
  chamber within 180 d). Named passed_origin_date (not passed_date) so A's local merge does not collide.
  `adv_cutoff(c)` = congress_end(c) - ADV_WINDOW_DAYS: bills introduced after it are censored for Task 2.
- `majority_party()` -> (congress, chamber, majority, seats_D, seats_R, seats_other, n_voting,
  tie_broken_by). Point-in-time count of voting members in office at `congress_midpoint(c)` (delegates
  excluded), party as of the Congress, independents counted with their caucus. Ties (117th Senate, 50-50)
  are broken by the Vice President's party (`VP_PARTY`, a documented constant: the parsed data has no
  chair/leader field) and cross-checked against the party sponsoring most committee-reported bills;
  `tie_broken_by` says which. `MAJORITY` holds the table. Use this; do not re-derive control locally.
- Candidate splits (single source of truth, moved from section D): `SPLIT_DEFS` (S1a, S1b, S2 sliding
  k=`SLIDE_K`=3, S3 rolling folds `FOLD_C`=112..118, S4 within-Congress cuts `S4_CUTS`, S5 random
  `S5_FRACS`) and `iter_eval_units(split=None)` -> dicts (split, fold, role 'val'/'test', train = whole
  Congresses to fit on (test units refit on train+val), eval_congress, train_label, within (S4 only:
  (c, train_q, eval_lo_q, eval_hi_q)), random (S5 only)). Every section evaluates exactly these units;
  do not define your own pools (B_POOLS, "sliding last 4", sliding3 should map onto these). D consumes them.
- `SPLIT_VARIANTS` (D's protocol variants, candidate splits for the decision section, NOT yielded by
  iter_eval_units): 'S1a+stream', 'S1b+stream', 'S3+stream' (base pools + eval-Congress events dated
  <= introduced_date + t0) and 'S1pool' (train 108-115 | val 116 | test 117+118). `ALL_SPLITS` = SPLIT_DEFS +
  SPLIT_VARIANTS keys. `split_kind(s)`, `split_eval_congresses(s, roles=('val','test'))`, `BY_CONGRESS_KINDS`.
- `S5_ROLE`: shared S5 draw, Series bill_id -> 'train'/'val'/'test' (law-capable bills of 108-118, seed 5,
  S5_FRACS; identical to D's original draw). C and D must use it, not a local draw.
- `S4_CUT_DATES[c][q]` for c in S4 folds, q in (0.6, 0.8, 1.0): introduction-date quantile of c's law-capable
  bills floored to the day (1.0 = day after the Congress ends). Fit on introduced < cut[0.6]; val =
  cut[0.6] <= introduced < cut[0.8]; test = introduced >= cut[0.8].
- Drift stats: `psi(a, b)`, `psi_categorical(a, b)` (<0.1 stable, 0.1-0.25 moderate, >0.25 shift,
  `psi_level(v)`), `jsd(p, q)`, `ks(a, b)` -> (stat, p), `smd(a, b)`.
- Shared "different from train noise" rule: `rel_to_train(values, train=None, side='two'|'upper'|'lower',
  log=False, alpha=0.05, cands=None)` -> (tab, info). Reference set is FIXED: `REL_REF` = 108-115 (S1a
  train), whatever pool you look at. `values` = Series by Congress (for distances: reference Congresses get
  their LOO distance to the other reference Congresses, candidates their distance to the reference pool).
  tab (one row per Congress; reference rows judged by nested LOO, role 'ref'/'cand'): value, z, t_pi,
  pi_lo/pi_hi, fail (outside the 95% t prediction interval), trend_pred, trend_z, trend_fail (break vs a
  linear trend over the reference), in_range. info: n, tcrit, fa_count / fa_rate (nested-LOO false
  alarms), fwer_loo / fwer_nominal (chance at least one of k unchanged candidates fails), rule (use as
  the record_check threshold). `rel_detail(tab, info)` -> one-line detail text.
  `rel_by_split(tab, col='fail', splits=None, roles=('val','test'), mode=None)` -> by_split dict. Modes:
  * `'cands'` (DEFAULT for rel_to_train tabs, i.e. inputs with a `role` column; `REL_SPLIT_MODE`): a split fails
    if any CANDIDATE Congress (116-118) it evaluates fails. Reference Congresses (109-115, evaluated only by S2/S3
    folds) are not judged: their nested-LOO verdict uses a reference that contains LATER Congresses, and counting
    them made 7-fold splits (S2/S3/S3+stream) fail more by construction (e.g. A1c/A9/A11/A12 passed every candidate
    but failed S2/S3). A split evaluating no candidate gets None.
  * `'excess'`: fails if the number of failing evaluated Congresses (reference + candidates) exceeds chance:
    Poisson-binomial p < 0.05 with per-Congress rate max(nested-LOO fa_rate, alpha). Low power for S1a / S1b
    (2 eval Congresses: both must fail), so it is a cross-check, not the default.
  * `'any'` (legacy; DEFAULT for plain Series / frames without `role`, e.g. deterministic threshold verdicts like
    coverage < 95%): fails if any evaluated Congress fails.
  `rel_split_table(tab, col='fail', ...)` -> one row per split: n_eval, n_fail, n_cand, n_cand_fail, fa_rate,
  expected, binom_p, pass_any, pass_cands, pass_excess (print it in `detail` or for the decision section).
  Remaining asymmetry under 'cands': S1a / S1b evaluate 2 candidates, S2 / S3 / S1pool evaluate 3.
  ALL sections should use this for "differs from train" verdicts (keep any old rule only as a secondary column).
- `record_check(section, check_id, condition, passed, metric=None, threshold=None, detail="", quantity=None,
  by_split=None, requirement=False)`: `requirement=True` marks a construction rule every split must follow
  (B6b, B7, B10, C11, C13, C17 style: "filter X to t0", "rebuild Y as of t0"), listed by `checks_requirements()`,
  deduplicated apart from non-requirement checks, excluded from `checks_by_split()`, and NOT counted as split
  failures (count from `checks_summary(requirements=False)`). `by_split` = dict split -> True (holds) / False (fails) / None (n/a) or a per-split
  metric, keys from `ALL_SPLITS`. FILL IT for every check whose outcome depends on the split (omit splits
  not covered); leave None for split-independent checks.
  `quantity` names the measured quantity when more than one section may test it. Owners are in
  `QUANTITY_OWNERS` (e.g. cosp_density, orig_share, label_rate_t1/t2, majority_or -> A; node_overlap,
  cold_start, degree_dist, graph_structure, sponsor_overlap, node_identity_switch, train_eval_input_shift,
  leakage_related_links, leakage_committee_edges, leakage_full_graph -> B; subject_coverage, annotation_coverage, lag_dist, party_regime, censoring,
  leakage_snapshot_fields -> C; score_stability, val_test_gap -> D; A also owns deployment_proxy, drift_type,
  cand_fail_count,
  drift_gradual, label_validity_t2, label_rate_t2_alt, label_count_t2, label_rate_t2_chamber,
  majority_or_regime(_effect), same_party_share, bill_volume, topic_mix_jsd, joint_covariate_shift,
  model_calibration_t2, model_auroc_transfer_t2, pool_distance). Non-owners should only test a quantity if they add a
  different angle, and still tag it. Leakage is split per path (the broad `leakage` key is kept for backward
  compatibility only; do not use it for new checks: it hid B6b's FAIL behind C11's INFO).
  `checks_summary(dedup=True, requirements=True)` returns one row per quantity (owner's verdict, with
  `dup_ids`, `hidden_fail_ids`, a `conflict` flag that is also True when a kept INFO hides a FAIL, a
  `requirement` column, plus the `by_split` dict); the decision section must count failures from
  `checks_summary(requirements=False)` and read `hidden_fail_ids` for rows with conflict=True. `checks_by_split(dedup=True)` = long table (id, quantity, passed, split, value,
  split_passed) for scoring candidate splits.
  `kind=` ('validity' | 'drift' | 'requirement'): the decision section RANKS splits on 'validity' checks only
  (`checks_split_scores()`, default kinds=('validity',)); 'drift' FAILs choose remedies. If kind is omitted, the
  default comes from `QUANTITY_KIND[quantity]` (e.g. cold_start -> validity; sponsor_overlap, node_identity_switch
  -> drift; train_eval_input_shift -> requirement, which also sets requirement=True), else kind=None. Pass kind=
  explicitly on every split-dependent check: `checks_split_scores()` prints a WARNING listing split-dependent
  checks with kind=None, because they are not scored.
  Other fields: passed True/False, or None for informational. Use ids prefixed by your section letter (A1, A2, ...,
  B1, ...). EVERY condition you test must end in a record_check with a concrete, data-derived metric
  and an explicit threshold rationale in `detail`. The decision section only sees CHECKS (via checks_summary()).
- Charts (D3 v7): `line_chart(df, x, y, series=None, title, subtitle, yfmt, ylabel, xlabel, yzero, height,
  width, name)`, `bar_chart(df, x, y, series=None, ...)` (y domain spans 0 and negative values), `heatmap(df, x, y, value, fmt, domain,
  diverging, ...)`, or custom `show_d3(js_body, data, height, width, opts, name)` where js_body runs
  with `el, data, d3, opts, viz` in scope (`viz.tooltip(el)`, `viz.axis(g)`, `viz.title(svg, t, sub)`,
  `viz.legend(svg, [[label, color]], x, y)`, `viz.palette`, `viz.ink`, `viz.ink2`, `viz.grid`).
  Always pass a unique `name=` (prefix with your section letter, e.g. "B_overlap_heatmap").

## Chart rules (from the dataviz guidelines)
- Pick the form by the job: change over Congresses -> line; comparison across categories -> bars;
  Congress x Congress matrices -> heatmap; distributions -> histogram/CCDF/box.
- Categorical colors in fixed palette order, never cycled; > 8 series -> fold into "Other" or facet.
  Party colors via PARTY_COLORS. Sequential = one hue; diverging = two hues + neutral midpoint.
- One y-axis per chart (no dual axes). Thin marks, 2px lines, hover tooltip on marks, legend whenever
  >= 2 series, text in ink colors (never series color), title says what the chart shows.
- Prefer 3-6 strong charts per section over many weak ones. Every chart must support a check.

## Content standards
- Python: pandas/numpy/scipy only. Vectorize; `cosp` has 2.1M rows, bill_actions 0.9M.
- Every code cell starts with a 1-2 line comment saying what it computes. Markdown cells explain WHY a
  condition matters for the split decision (1 short paragraph each), and how to read the output.
- Print compact tables (`.round(3)`, `to_string()`), not raw frames of thousands of rows.
- Never hardcode a number that comes from the data; compute it.
- Be honest: if a check fails, say what it implies for the split (e.g. "use sliding window",
  "reweight", "exclude 119", "evaluate per regime"). If a test is only a proxy, say so in `detail`.
- Ground facts: party control of the chambers must be derived from `majority_party()`, not memory.

## Task definitions (fixed; do not change)
- Task 1, cosponsor prediction: for each law-capable bill at t0 in {0, 7} days, rank eligible
  same-chamber legislators; positives = cosponsors with lag in (t0, t0+30]; censored windows excluded.
- Task 2, advancement: law-capable bill reported out of committee within 180 days; censored excluded.
- Candidate splits to evaluate (builder D owns the definitions; others may reference them):
  S1 expanding by Congress: train 108-115, val 116, test 117/118 variants; S2 sliding window (last k
  Congresses for train); S3 rolling-origin folds (expanding, one-step-ahead); S4 within-Congress
  temporal (by introduction date inside each Congress); S5 random bill split (leaky reference only).

## Python version
The user's Jupyter kernel is Python 3.11 (pandas 3.1). Code must be 3.11-compatible: no nested f-strings that reuse the outer quote character (PEP 701, 3.12+). Test with the user's interpreter:
`~/.pyenv/shims/python tools/split_notebook/run_cells.py`.
