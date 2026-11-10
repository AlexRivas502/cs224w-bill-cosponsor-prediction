"""Offline test harness for the split-analysis notebook (no Jupyter kernel needed).

Runs notebooks/_build/cells/00_setup.py plus the given section files in one shared namespace, the way
the notebook would, then runs every D3 chart through js_check.js (Node + a minimal DOM).

Usage (from the repo root):
  uv run --no-project --python 3.13 --with pandas --with pyarrow --with scipy \
      python notebooks/_build/run_cells.py                    # all sections
  ... python notebooks/_build/run_cells.py 20_overlap.py       # setup + one section

Writes notebooks/_build/out/<run>/report.json (errors, timings, checks, chart results) and
out/<run>/<section>.stdout.txt (what the notebook would print), where <run> is the section stem
(e.g. out/20_graph/) or "all". Exit code 1 if anything failed.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

BUILD = Path(__file__).resolve().parent
CELLS = BUILD / "cells"
OUT = BUILD / "out"
REPO = BUILD.parent.parent


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


def main():
    names = sys.argv[1:] or sorted(p.name for p in CELLS.glob("[0-9]*.py") if p.name != "00_setup.py")
    # Separate output folder per run so builders testing in parallel don't overwrite each other.
    out = OUT / ("all" if not sys.argv[1:] else "+".join(Path(n).stem for n in sys.argv[1:]))
    out.mkdir(parents=True, exist_ok=True)
    files = [CELLS / "00_setup.py"] + [CELLS / n for n in names if n != "00_setup.py"]
    os.environ["NB_HARNESS_OUT"] = str(out)
    os.chdir(REPO / "notebooks")
    ns = {"__name__": "__main__"}
    report = {"sections": [], "ok": True}
    for f in files:
        sec = {"file": f.name, "ok": True, "cells": 0, "seconds": 0.0, "error": None, "charts_before": 0}
        sec["charts_before"] = len(ns.get("HARNESS_CHARTS", []))
        buf = io.StringIO()
        t0 = time.time()
        for i, (kind, src) in enumerate(parse_cells(f)):
            if kind != "code":
                continue
            sec["cells"] += 1
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(src, f"{f.name}[cell {i}]", "exec"), ns)
            except Exception:
                sec["ok"] = False
                sec["error"] = traceback.format_exc(limit=6)
                buf.write("\n!!! " + sec["error"])
                break
        sec["seconds"] = round(time.time() - t0, 1)
        sec["stdout_chars"] = len(buf.getvalue())
        (out / f"{f.stem}.stdout.txt").write_text(buf.getvalue())
        report["sections"].append(sec)
        report["ok"] &= sec["ok"]
        print(f"{'OK  ' if sec['ok'] else 'FAIL'} {f.name}: {sec['cells']} code cells, {sec['seconds']}s"
              + ("" if sec["ok"] else f"\n{sec['error']}"))

    charts = ns.get("HARNESS_CHARTS", [])
    (out / "charts.json").write_text(json.dumps(charts))
    js = subprocess.run(["node", str(BUILD / "js_check.js"), str(out / "charts.json"), str(out / "charts_report.json")],
                        capture_output=True, text=True)
    print(js.stdout.strip() or js.stderr.strip())
    chart_results = json.loads((out / "charts_report.json").read_text()) if (out / "charts_report.json").exists() else []
    report["charts"] = chart_results
    report["charts_ok"] = js.returncode == 0 and all(c["ok"] for c in chart_results)
    report["checks"] = ns.get("CHECKS", [])
    report["ok"] &= report["charts_ok"]
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"checks recorded: {len(report['checks'])}; report: {out / 'report.json'}")
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
