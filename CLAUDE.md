# CLAUDE.md

Instructions for working in this repo (CS224W final project: predicting U.S. Congress bill advancement and
cosponsorship with graph ML). Read `README.md` for how to set up and run, the "Where things live" table below for the repo map, `PROJECT_MILESTONE.md` for the current state of
the project, and `docs/BASELINES.md` for why every model, loss and hyperparameter was chosen.

## The project in one screen

- **Data:** Congresses 108-118 (2003-2024), parsed into 21 Parquet tables in `data/parsed/` (in git). 119 is in
  session and never used as labels. Raw downloads and caches in `data/` are local only.
- **Cosponsor prediction (Task 1, `t1`):** temporal link prediction. For a bill at day t0 (0, optionally 7), rank
  every eligible same-chamber legislator (not the sponsor, not already a cosponsor) by who cosponsors in
  (t0, t0+30]. Metrics: **Recall@20** (headline), Recall@10, NDCG@20, over the full pool (no sampled negatives).
- **Bill advancement (Task 2, `t2`):** node classification. Is a law-capable bill (HR, S, HJRES, SJRES) reported
  out of committee within 180 days? Information at introduction only (t0 = 0), policy area hidden. Metrics:
  **AUROC** (headline) and AP. About 2.8% positive in Congress 117.
- **Split (rolling origin by Congress):** fold c trains GNNs on 109..c-2, validates on c-1, refits on 109..c-1 and
  tests on c. Simple mode = fold 117; rolling mode = folds 112-117. 108 is history only. **118 is the locked
  final test**: scored once, at the very end, behind `LOCKED_118`.
- **Graph:** one leak-free graph per target Congress k. History = bills of k-3..k-1 with edges in both directions;
  target bills (Congress k) only *receive* edges dated on or before their cutoff (sponsor, cosponsors so far,
  committee referrals). Same-Congress signal for Task 1 ("+stream") enters only through the 22 pair features `phi`.
- **Current models:** heuristics, TabGBDT / TabLogReg, PairMLP (0 message-passing layers), G1 HeteroSAGE,
  G2 R-GCN, G3 LightGCN-feat. Fold-117 numbers are in `PROJECT_MILESTONE.md` sections 5.3 and 6.3.

## Rules that must not be broken

1. **No leakage.** A target bill never sends messages (`graphs.EDGE_TYPES` has no `tbill` source). Nothing
   dated after a bill's cutoff may reach its features, its edges or its pair features. Never add a feature
   whose name matches `features.FORBIDDEN_PATTERNS` (policy area, subjects, summaries, text versions, related
   bills, latest action, final counts, reported date): these are written as a bill advances and encode the label.
   If a new signal needs one of these, stop and ask.
2. **Never select anything on test scores.** Choose hyperparameters, epochs and model variants on the
   validation Congress (c-1) only. A tuned config records the Congresses it was tuned on in `tuned_on`.
   Never run with `LOCKED_118 = True` unless the whole team agrees that every choice is final.
3. **Compare like with like.** Same fold, same t0, same protocol, same evaluation population, same seeds. Learned
   models run on 3 seeds `(0, 1, 2)` before you claim a difference. With one seed, gaps below about 0.015 are
   noise. A difference is "real" only if it is at least the MDE (0.01) and every seed's paired CI excludes 0
   (`report.paired_seed_summary`).
4. **Every new model is compared against PairMLP** (the same model without message passing) and the best
   heuristic. "Beats the heuristics" is not enough: the question of this project is whether message passing adds
   anything over PairMLP.
5. **Don't touch** `data_processing/secrets.ini` (API key, gitignored), the raw data folders, or the cached
   reference results in `notebooks/baselines_results/reference_fold117_3seeds/` (they back the milestone tables).

## Where things live

| What | Where |
|---|---|
| All modelling logic | `notebooks/baselines/` (package): `config.py` (run, model and train configs; APRIORI / FROZEN tables), `data.py` (load corpus, labels, pools), `features.py` (node and pair features, forbidden patterns), `graphs.py` (leak-free graphs), `train.py` (fit, early stop, refit, seeds, search), `evaluate.py` (metrics, bootstrap CIs), `report.py` (tables, paired tests), `checks.py` (leak checks) |
| Models | `notebooks/baselines/models/`: `heads.py` (shared `TypeEncoder`, `LinkHead`, `NodeHead`, and `GraphModel` = PairMLP), one file per GNN, `heuristics.py`, `__init__.py` (`REGISTRY`, `DISPLAY`) |
| Design document | `notebooks/baselines/DESIGN.md` (binding spec the package was built from) |
| Notebook source | `tools/baselines_notebook/cells/*.py` (jupytext percent format), assembled into `notebooks/02_baselines.ipynb` |
| Tests | `tools/baselines_notebook/tests/` (`test_sanity.py`: leak, determinism, perturbation; `test_parity.py`: labels match the split study) |
| Tuning logs | `tools/baselines_notebook/tuning/*.json` (evidence for every FROZEN entry) |
| Split study | `notebooks/01_split_analysis.ipynb` from `tools/split_notebook/cells/`; `docs/split_report.html` |

## Adding a new model

Follow the pattern of `models/hetero_sage.py`. A new GNN is a subclass of `GraphModel` that overrides only
`message_passing`, so it reuses the shared encoder, both heads, both losses, the training loop and evaluation.

1. **Model file** `notebooks/baselines/models/<name>.py`: `class MyGNN(GraphModel)` with `__init__(self, cfg, meta)`
   and `message_passing(self, h, graph) -> dict`. It takes `{node_type: [n, d]}` states and must return at least
   `"leg"` and `"tbill"`. Use `graphs.EDGE_TYPES` and never add an edge with a `tbill` source. Apply DropEdge only
   to history edges in training mode (see `HeteroSAGE.edge_dict`). Start the file with a docstring that says what
   the model is, which paper it comes from, and why it stays leak-free, as the existing model files do.
2. **Register it** in `models/__init__.py`: add `REGISTRY["<name>"] = ("<module>", "<Class>")` and a display name
   in `DISPLAY` (GNNs are labelled `G4 ...`, `G5 ...`).
3. **Config** in `config.py`: a `@dataclass` config (keep the shared fields `d`, `dropout`, `tower`, `pair_hidden`,
   `phi_norm`, `use_dot`, `use_pair` and `tuned_on: tuple = ()`), an entry in `MODEL_CFG_CLASS`, and `APRIORI`
   entries for `("<name>", "t1")` and `("<name>", "t2")`. A-priori defaults come from the paper or lectures, not from
   our scores, so they carry `tuned_on = ()`. `FROZEN` copies APRIORI automatically. Only change a FROZEN entry
   after a validation search, with a comment that names the tuning log and the gain, like the existing entries.
   Optionally add a `SEARCH_SPACE["<name>"]` grid.
4. **Report:** add the name to `report.GNN_ORDER` so it is included in the GNN-vs-GNN paired comparisons.
5. **Notebook section:** a new cell file `tools/baselines_notebook/cells/65_<name>.py` (numbered so it sorts before
   `70_references.py`) with 2-3 plain-language markdown sentences and exactly this code:
   ```python
   G4 = report.run_model_section("<name>", cfg, corpus, timer)
   RESULTS += G4
   ```
   Add `"<name>"` to the default `MODELS` tuple in `cells/00_config.py`, then rebuild (commands below).
6. **Verify:** run the quick headless run, the test suite, and the perturbation test for your model
   (`BASELINES_GNN=<name>`), which fails if scores change when post-cutoff events move.
7. **Document:** add the model to `docs/BASELINES.md` (why this model, which question it answers, its
   hyperparameters and their source) and to the results tables once you have 3-seed numbers.

New heuristics go in `models/heuristics.py` and are returned by `t1_reference_results` / `t2_reference_results`;
they need no training. New features go in `features.py`; check that they pass `assert_allowed` and the
truncation test.

## Trying out an idea (an experiment)

- Change **one thing at a time** and judge it on the **validation** score (the `val` column), never on test.
- For a hyperparameter, use an override instead of editing defaults: `config.with_overrides(mcfg, tcfg,
  {"model.dropout": 0.5, "train.lr": 1e-3})`, or add it to `SEARCH_SPACE` and run with `RUN_SEARCH = True`.
- Adopt a change only if it beats the current setting by more than the tolerance (`config.D_TOL`: 0.005 Recall@20,
  0.01 AUROC) on the seed mean. Otherwise keep the current setting and write the near-miss down.
- Save the evidence as a JSON log in `tools/baselines_notebook/tuning/`, update the FROZEN entry with a comment
  that cites it, and set `tuned_on` to the validation Congresses used.
- Planned directions (milestone section 7, "how can we make message passing matter?") are a good place to start.
  Each one comes with the ablation that would show it worked.

## Commands

Run from the repo root. The code must run on **Python 3.11** (the notebook kernel): no nested same-quote f-strings
or other 3.12-only syntax. Dependencies are in `notebooks/requirements.txt` (torch and torch_geometric are needed
for the GNNs; pyg-lib / torch-sparse are not available, so training is full-batch, not NeighborLoader).

```bash
python tools/baselines_notebook/assemble.py                       # cells -> notebooks/02_baselines.ipynb
python tools/baselines_notebook/run_cells.py --quick --threads 8 --run <name>   # headless 2-epoch run, ~2 min
python -m pytest -q tools/baselines_notebook/tests               # 13 tests
BASELINES_GNN=<model> python -m pytest -q tools/baselines_notebook/tests/test_sanity.py -k perturbation_gnn
python tools/split_notebook/assemble.py && python tools/split_notebook/run_cells.py   # split study
```

Headless runs write to `tools/baselines_notebook/out/<name>/` (gitignored). Override the config cell with runner
flags instead of editing it: `--models rgcn,pair_mlp --seeds 0,1,2 --tasks t1`, plus `--mode`, `--t0`,
`--protocols`, `--folds`, and section names to run only some sections (they map to the `BASELINES_*` environment
variables in `config.apply_env_overrides`). A full simple-mode run with one seed
takes about 12 minutes on a laptop CPU; three seeds take about 3x that.

## Conventions

- **Edit notebook source cells, not the `.ipynb`.** Notebooks are generated from `tools/*/cells/*.py`.
  Reassembling replaces a notebook's outputs. `01_split_analysis.ipynb` holds executed outputs, so if you change its
  cells, copy the changed source into the existing notebook instead of reassembling, or re-run it afterwards.
- **Keep the notebook thin.** Logic lives in the `baselines` package. Each notebook section is a heading, a few
  plain sentences, and a few calls. Long explanations belong in `docs/BASELINES.md`. Notebook output should be
  the results tables, not logs: a check that passes prints one line, and a check that fails raises.
- **Every check is an assert.** New leak or consistency checks go in `checks.py` (or `graphs.check_graph`) and
  raise on failure, and get a test in `tests/` when they are cheap enough.
- **Results must be reproducible.** Fixed seeds, deterministic data order, results saved with their config
  (`report.save_results`). Report means over seeds with bootstrap CIs clustered by sponsor; state the number of
  seeds next to every learned-model number.
- **Writing style** (docs, milestone, notebook markdown): plain, direct sentences; explain why a choice was made,
  not only what it is; every number in prose must come from a saved results file. No lecture-slide references in
  the milestone.
- **Git:** don't commit or push unless asked. Never commit `secrets.ini`, `data/` outside `data/parsed/`, or
  `tools/*/out/`.
