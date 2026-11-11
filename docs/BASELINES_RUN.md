# Running the baselines notebook

## 1. Install (once)

Use the Python that your JupyterLab kernel runs (Python 3.11). From the `notebooks/` folder:

```bash
cd "/Users/alex_rivas/Documents/Stanford/CS224W/Final Project/cs224w-bill-cosponsor-prediction/notebooks"
python -m pip install -r notebooks/requirements.txt
```

To check you are installing into the kernel's Python, run `import sys; print(sys.executable)` in a notebook
cell and use that path in place of `python` above. pyg-lib, torch-sparse and torch-scatter are **not** needed:
every model uses full-batch message passing.

Tested with torch 2.15, torch_geometric 2.8.1, pandas 3.1.0, numpy 2.4.6, scipy 1.17.1, pyarrow 26.0.1 and
scikit-learn 1.9.

## 2. Run

1. Open `notebooks/02_baselines.ipynb` in JupyterLab (the notebook finds `data/parsed/` by searching upward).
2. Edit only the config cell (section 0):
   - `MODE = "simple"` runs fold 117 (validate 116, test 117). This is the default.
   - `MODE = "rolling"` runs folds 112-117. `FOLDS_ONLY = (116,)` (for example) restricts it to some folds.
   - `SEEDS = (0, 1, 2)` gives mean and sd over seeds. Use this before ranking the learned models against each other.
   - `TASKS`, `MODELS`, `T0_LIST`, `T1_PROTOCOLS` select what is trained.
   - `THREADS = 8` by default. If other heavy jobs are running, use cores / jobs (for example 4).
   - `LOCKED_118 = False`. Switch it on only once, at the very end, for the locked final test.
3. Run All. Results go to `notebooks/baselines_results/<mode>/results.csv` (plus `paired.csv` and
   `results_meta.json`).

## 3. Expected runtime (Apple Silicon CPU, measured)

| run | settings | wall time |
|---|---|---|
| simple mode, default | fold 117, both tasks, all baselines, 1 seed, 8 threads | 12-13 minutes (measured twice on 2026-11-09: 12.8 and 11.7 minutes, with other jobs running; identical results) |
| simple mode, 3 seeds | as above with `SEEDS = (0, 1, 2)` | 41 minutes (measured; partly alongside another job) |
| rolling mode, one fold | `MODE = "rolling"`, `FOLDS_ONLY = (116,)`, 1 seed | 12.1 minutes (measured; 4 threads, alongside another job) |
| rolling mode, all folds | folds 112-117, 1 seed | about 1 hour (estimate: about 10-12 minutes per fold) |

The first run builds caches under `data/cache/baselines/` (about 1 extra minute). On a busy machine
everything can take several times longer. The timing table at the end of section 8 shows the time per stage.

## 4. Headless run (what was used for testing)

```bash
cd "/Users/alex_rivas/Documents/Stanford/CS224W/Final Project/cs224w-bill-cosponsor-prediction/notebooks"
uv run --no-project --python 3.11 --with pandas==3.1.0 --with numpy==2.4.6 --with scipy==1.17.1 \
  --with pyarrow==26.0.1 --with torch --with torch_geometric --with scikit-learn \
  python tools/baselines_notebook/run_cells.py --threads 8 --run my_run          # simple mode
#   add: --mode rolling --folds 116      (one rolling fold)
#        --seeds 0,1,2                    (three seeds)
#        --quick                          (2-epoch smoke test, about 2 minutes)
python tools/baselines_notebook/assemble.py       # rebuild 02_baselines.ipynb from tools/baselines_notebook/cells/*.py
```

The outputs go to `tools/baselines_notebook/out/<run>/` (one stdout file per section, `results.csv`, `report.json`).
Leakage and parity tests: `python -m pytest tools/baselines_notebook/tests -q` (add `--with pytest` to the `uv run`
line). They take about 1 minute.
