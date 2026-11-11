# Predicting Bill Advancement and Cosponsorship with Graph Deep Learning

How to run the project. For now this covers the baselines notebook (`notebooks/02_baselines.ipynb`): heuristics,
tabular models and three GNN baselines for bill advancement and cosponsor prediction.

## 1. Requirements

- **Python 3.11** (3.12+ is not tested).
- About 2 GB of disk for the Python packages (mostly PyTorch). The data comes with the repo (~176 MB).
- A CPU is enough. No GPU, and no CUDA-specific install, is needed.

## 2. Get the code and data

```bash
git clone https://github.com/AlexRivas502/cs224w-bill-cosponsor-prediction.git
cd cs224w-bill-cosponsor-prediction
```

The parsed Parquet tables the notebook reads are already in `data/parsed/`. Nothing else needs to be downloaded.

## 3. Set up the environment

Create a virtual environment, install the dependencies, and register it as a Jupyter kernel:

```bash
python3.11 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r notebooks/requirements.txt ipykernel
python -m ipykernel install --user --name cs224w-bills --display-name "Python 3.11 (cs224w-bills)"
```

Check the install:

```bash
python -c "import torch, torch_geometric, pandas, sklearn; print(torch.__version__, torch_geometric.__version__, pandas.__version__)"
```

Tested with torch 2.15, torch_geometric 2.8.1, pandas 3.1.0, numpy 2.4.6, scipy 1.17.1, pyarrow 26.0.1 and
scikit-learn 1.9. You do **not** need pyg-lib, torch-sparse or torch-scatter: every model uses full-batch
message passing.

<details>
<summary>Using uv instead of venv + pip</summary>

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r notebooks/requirements.txt ipykernel
python -m ipykernel install --user --name cs224w-bills --display-name "Python 3.11 (cs224w-bills)"
```
</details>

## 4. Run the baselines notebook

```bash
jupyter lab notebooks/02_baselines.ipynb
```

1. Select the kernel **Python 3.11 (cs224w-bills)** (Kernel > Change Kernel).
2. Optionally edit the config cell (section 0). It is the only cell you need to change:

   | Setting | Default | Meaning |
   |---|---|---|
   | `MODE` | `"simple"` | `"simple"`: fold 117 (train up to 115, validate on 116, retrain up to 116, test on 117). `"rolling"`: folds 112-117 |
   | `TASKS` | `("t1", "t2")` | `t1` = cosponsor prediction, `t2` = bill advancement |
   | `MODELS` | all four | `"hetero_sage"`, `"rgcn"`, `"lightgcn"`, `"pair_mlp"` |
   | `SEEDS` | `(0,)` | `(0, 1, 2)` to compare learned models fairly (about 3x slower) |
   | `THREADS` | `8` | lower it if the machine has fewer cores or is busy |
   | `QUICK` | `False` | `True` = 2-epoch smoke test (about 2 minutes) |
   | `LOCKED_118` | `False` | the locked final test; leave it off |

3. **Run All** (Run > Run All Cells).

The notebook finds `data/parsed/` on its own. On the first run it also builds a cache in `data/cache/baselines/`,
which adds about a minute.

**Outputs:** tables in the notebook (one per task, validation and test scores with 95% CIs, then paired
differences and a 3-seed reference), and files in `notebooks/baselines_results/<mode>/`: `results.csv`,
`paired.csv` and `results_meta.json`.

**Runtime** (laptop CPU, 8 threads):

| Run | Time |
|---|---|
| `QUICK = True` | ~2 minutes |
| simple mode, 1 seed (default) | ~12 minutes |
| simple mode, `SEEDS = (0, 1, 2)` | ~40 minutes |
| rolling mode, all 6 folds, 1 seed | ~1 hour |

## 5. Run without Jupyter (optional)

The same notebook can run headlessly from the terminal, with the virtual environment active. This is also the
quickest way to check that the setup works:

```bash
python tools/baselines_notebook/run_cells.py --quick --threads 8 --run smoke     # ~2 minutes
python tools/baselines_notebook/run_cells.py --threads 8 --run full              # default simple-mode run
#   options: --seeds 0,1,2   --models rgcn,pair_mlp   --tasks t1   --mode rolling --folds 116
```

Each section's output goes to `tools/baselines_notebook/out/<run>/`, together with `results.csv`.

Tests (leakage checks and label parity, about 1 minute):

```bash
python -m pip install pytest
python -m pytest -q tools/baselines_notebook/tests
```

## 6. Troubleshooting

- **`ModuleNotFoundError: torch` (or `baselines`) in the notebook:** the notebook is not using the
  `cs224w-bills` kernel. Change the kernel, or run `import sys; print(sys.executable)` in a cell to see which
  Python it is using.
- **`SyntaxError` on import:** the kernel is older than Python 3.11.
- **Very slow:** lower `THREADS` if other heavy jobs are running, or set `QUICK = True` to check the setup first.
- **Editing the notebook:** `02_baselines.ipynb` is generated from `tools/baselines_notebook/cells/*.py`. Edit the
  cells, then rebuild with `python tools/baselines_notebook/assemble.py` (this clears the notebook's outputs).
