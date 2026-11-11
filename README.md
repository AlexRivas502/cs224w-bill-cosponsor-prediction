# Predicting Bill Advancement and Cosponsorship with Graph Deep Learning

CS224W (Fall 2026) project by Andrew Lawlor and Alex Rivas. We turn 20 years of U.S. Congress records (Congresses 108–118) into a leak-free temporal graph and predict two things:
- **Bill advancement:** will a bill be reported out of committee within 180 days?
- **Cosponsor prediction:** who will cosponsor a bill in the next 30 days?

**Start here:** [`PROJECT_MILESTONE.md`](PROJECT_MILESTONE.md), the current report draft (data, tasks, protocol, baselines and results). The original plan is in [`PROJECT_PROPOSAL.md`](PROJECT_PROPOSAL.md).

## Repository map

```
PROJECT_MILESTONE.md      report draft (becomes the Medium post)
PROJECT_PROPOSAL.md       submitted proposal
data_processing/          download + ETL: raw Congress data -> Parquet tables
notebooks/
  01_split_analysis.ipynb   which train/validation/test split is valid (and why)
  02_baselines.ipynb        heuristics, tabular models and three GNN baselines
  baselines/                Python package used by 02 (data, graphs, models, training, metrics)
  baselines_results/        cached 3-seed reference results shown in 02
  requirements.txt          everything both notebooks need
  vendor/                   D3 (charts in 01)
docs/
  BASELINES.md              why each model, loss, optimizer and hyperparameter was chosen; full results
  BASELINES_RUN.md          how to run 02, modes and runtimes
  split_report.html         interactive report of the split analysis, with the recommended split
  figures/                  figures used in the report (draw.io sources + PNGs)
tools/                    for developers: build and test harnesses (notebooks are assembled from source cells)
  split_notebook/           source cells + headless runner for 01
  baselines_notebook/       source cells, runner, leakage/parity tests, tuning logs for 02
  report/                   builds docs/split_report.html from 01's outputs
data/
  parsed/                   the 21 Parquet tables both notebooks read (~176 MB, in git)
  billstatus_bulk/, congress_api/, cache/   raw downloads and caches (~6 GB, not in git)
```

## Quickstart

1. **Install** into the Python your Jupyter kernel uses (Python 3.11):
   ```bash
   python -m pip install -r notebooks/requirements.txt
   ```
2. **Data.** The parsed tables in `data/parsed/` come with the repo, so you can skip this step. To rebuild them from the raw sources (several hours, and an API key):
   ```bash
   bash data_processing/bulk_bill_download.sh                 # GovInfo BILLSTATUS XML -> data/billstatus_bulk/
   python3 data_processing/fetch_congress.py                  # Congress.gov API -> data/congress_api/ (needs an API key, see the script)
   uv run data_processing/parsers.py                          # Spark ETL -> data/parsed/*.parquet (21 tables)
   ```
3. **Run the notebooks** from JupyterLab, in order: `notebooks/01_split_analysis.ipynb` (about 3 min), then `notebooks/02_baselines.ipynb` (about 12 min on CPU in the default mode). Both find `data/` automatically.

## Editing the notebooks

The notebooks are generated from source cells, so edit the cells and rebuild:

| Notebook | Edit | Rebuild | Test headlessly |
|---|---|---|---|
| 01 | `tools/split_notebook/cells/*.py` | `python tools/split_notebook/assemble.py` | `python tools/split_notebook/run_cells.py` |
| 02 | `tools/baselines_notebook/cells/*.py` (+ `notebooks/baselines/`) | `python tools/baselines_notebook/assemble.py` | `python tools/baselines_notebook/run_cells.py --quick` and `python -m pytest tools/baselines_notebook/tests` |

Rebuilding a notebook replaces its outputs. Re-run it afterwards.
