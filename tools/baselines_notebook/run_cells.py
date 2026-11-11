"""Headless test harness for notebooks/02_baselines.ipynb (no Jupyter kernel needed).

Runs tools/baselines_notebook/cells/NN_*.py in one shared namespace, the way the notebook would, from the
notebooks/ folder, and writes tools/baselines_notebook/out/<run>/{<section>.stdout.txt, stdout.txt, results.csv,
report.json}.

Usage (from anywhere):
  uv run --no-project --python 3.11 --with pandas==3.1.0 --with numpy==2.4.6 --with scipy==1.17.1 \\
      --with pyarrow==26.0.1 --with torch --with torch_geometric --with scikit-learn \\
      python tools/baselines_notebook/run_cells.py [--quick] [--mode simple|rolling] [--threads N]
      [--models a,b] [--tasks t1,t2] [--t0 0,7] [--protocols stream,plain] [--seeds 0,1,2]
      [--folds 116,117] [--run NAME] [sections ...]

Options map to the BASELINES_* environment overrides read by config.apply_env_overrides. Exit code 1 on failure.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

BUILD = Path(__file__).resolve().parent
CELLS = BUILD / "cells"
OUT = BUILD / "out"
NOTEBOOKS = BUILD.parent.parent / "notebooks"   # the notebook runs from <repo>/notebooks


def parse_cells(path):
    """Split a jupytext percent-format file into [(kind, source)]."""
    cells, kind, buf = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    if kind is not None:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return [(k, s) for k, s in cells if s.strip()]


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--mode")
    ap.add_argument("--threads")
    ap.add_argument("--models")
    ap.add_argument("--tasks")
    ap.add_argument("--t0")
    ap.add_argument("--protocols")
    ap.add_argument("--seeds", help="comma list, e.g. 0,1,2")
    ap.add_argument("--folds", help="rolling mode: comma list of test folds, e.g. 116,117")
    ap.add_argument("--run")
    ap.add_argument("--echo", action="store_true", help="also print section output live")
    a = ap.parse_args()
    env = {"BASELINES_QUICK": "1" if a.quick else None, "BASELINES_MODE": a.mode, "BASELINES_THREADS": a.threads,
           "BASELINES_MODELS": a.models, "BASELINES_TASKS": a.tasks, "BASELINES_T0": a.t0,
           "BASELINES_PROTOCOLS": a.protocols, "BASELINES_SEEDS": a.seeds, "BASELINES_FOLDS": a.folds}
    for k, v in env.items():
        if v is not None:
            os.environ[k] = v
    names = a.sections or sorted(p.name for p in CELLS.glob("[0-9]*.py"))
    files = [CELLS / n for n in names]
    if CELLS / "00_config.py" not in files:
        files = [CELLS / "00_config.py"] + files
    run = a.run or (("quick" if a.quick else (a.mode or "simple")) + ("" if not a.sections else "+" + "+".join(Path(n).stem for n in a.sections)))
    out = OUT / run
    out.mkdir(parents=True, exist_ok=True)
    os.environ["BASELINES_OUT"] = str(out)
    os.chdir(NOTEBOOKS)
    ns = {"__name__": "__main__"}
    report = {"run": run, "sections": [], "ok": True, "env": {k: v for k, v in env.items() if v is not None}}
    allbuf = io.StringIO()
    t_all = time.time()
    for f in files:
        sec = {"file": f.name, "ok": True, "cells": 0, "seconds": 0.0, "error": None}
        buf = io.StringIO()
        sink = _Tee(buf, allbuf, sys.__stdout__) if a.echo else _Tee(buf, allbuf)
        t0 = time.time()
        for i, (kind, src) in enumerate(parse_cells(f)):
            if kind != "code":
                continue
            sec["cells"] += 1
            try:
                with contextlib.redirect_stdout(sink):
                    exec(compile(src, f"{f.name}[cell {i}]", "exec"), ns)
            except Exception:
                sec["ok"] = False
                sec["error"] = traceback.format_exc(limit=8)
                buf.write("\n!!! " + sec["error"])
                break
        sec["seconds"] = round(time.time() - t0, 1)
        (out / f"{f.stem}.stdout.txt").write_text(buf.getvalue())
        report["sections"].append(sec)
        report["ok"] &= sec["ok"]
        print(f"{'OK  ' if sec['ok'] else 'FAIL'} {f.name}: {sec['cells']} code cells, {sec['seconds']}s"
              + ("" if sec["ok"] else f"\n{sec['error']}"), flush=True)
        if not sec["ok"]:
            break
    report["seconds_total"] = round(time.time() - t_all, 1)
    timer = ns.get("timer")
    if timer is not None:
        report["timing_s"] = {k: round(v, 1) for k, v in timer.stages.items()}
    (out / "stdout.txt").write_text(allbuf.getvalue())
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"total {report['seconds_total']}s; outputs in {out}")
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
