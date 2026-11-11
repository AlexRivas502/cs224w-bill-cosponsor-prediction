"""Label parity (DESIGN check 2): the package's task definitions equal the split notebook's.

Execs tools/split_notebook/cells/00_setup.py once (output suppressed) and compares window_targets(0 / 7,
point_in_time=True), build_advancement_labels(), ROSTER_ASOF_ALL, MAJORITY and the members caucus / as-of
party with baselines.data. Runs with pytest or as a script (exit code 1 on failure).
"""
import contextlib
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
NOTEBOOKS = REPO / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

from baselines import data  # noqa: E402

_NS = {}


def _setup_ns():
    if not _NS:
        cwd = os.getcwd()
        os.chdir(NOTEBOOKS)
        try:
            src = (REPO / "tools" / "split_notebook" / "cells" / "00_setup.py").read_text()
            ns = {"__name__": "__main__"}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, "00_setup.py", "exec"), ns)
            _NS.update(ns)
        finally:
            os.chdir(cwd)
    return _NS


_CORPUS = []


def _corpus():
    if not _CORPUS:
        _CORPUS.append(data.load_corpus())
    return _CORPUS[0]


def _norm(df):
    df = df.reset_index(drop=True).copy()
    for c in df.columns:
        if str(df[c].dtype) in ("object", "str", "string") or str(df[c].dtype).startswith("string"):
            df[c] = df[c].astype(object)
    return df


def test_window_targets():
    ns, c = _setup_ns(), _corpus()
    for t0 in (0, 7):
        a = ns["window_targets"](t0, point_in_time=True)
        b = data.window_targets(c, t0, point_in_time=True)
        pd.testing.assert_frame_equal(_norm(a), _norm(b))
        a = ns["window_targets"](t0)
        b = data.window_targets(c, t0)
        pd.testing.assert_frame_equal(_norm(a), _norm(b))


def test_advancement_labels():
    ns, c = _setup_ns(), _corpus()
    pd.testing.assert_frame_equal(_norm(ns["build_advancement_labels"]()), _norm(c.adv))


def test_roster_and_majority():
    ns, c = _setup_ns(), _corpus()
    a, b = _norm(ns["ROSTER_ASOF_ALL"]), _norm(c.roster_all)
    b["month"] = b["month"].astype(a["month"].dtype)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    pd.testing.assert_frame_equal(_norm(ns["MAJORITY"]), _norm(c.majority))


def test_members_caucus_party():
    ns, c = _setup_ns(), _corpus()
    m = ns["members"]
    for col in ("caucus", "party_code_asof", "chamber"):
        pd.testing.assert_series_equal(m[col].reset_index(drop=True).astype(object),
                                       c.members[col].reset_index(drop=True).astype(object), check_names=False)
    assert abs(ns["PA_MISS_T0"] - c.info["pa_miss_t0"]) < 1e-12


if __name__ == "__main__":
    ok = True
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t:.1f}s)")
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"FAIL {name}: {type(e).__name__}: {str(e)[:2000]}")
    sys.exit(0 if ok else 1)
