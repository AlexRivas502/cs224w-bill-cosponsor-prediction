"""Assemble notebooks/02_baselines.ipynb from tools/baselines_notebook/cells/NN_*.py (jupytext percent format).

Usage: python tools/baselines_notebook/assemble.py
"""
import json
from pathlib import Path

from run_cells import CELLS, parse_cells

NOTEBOOK = CELLS.parent.parent.parent / "notebooks" / "02_baselines.ipynb"


def to_lines(text):
    """Notebook JSON source lines (each but the last ends with a newline)."""
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def main():
    cells = []
    for f in sorted(CELLS.glob("[0-9]*.py")):
        for kind, src in parse_cells(f):
            if kind == "markdown":
                src = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in src.splitlines())
                cells.append({"cell_type": "markdown", "metadata": {}, "source": to_lines(src)})
            else:
                cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                              "outputs": [], "source": to_lines(src)})
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}}}
    NOTEBOOK.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {NOTEBOOK} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
