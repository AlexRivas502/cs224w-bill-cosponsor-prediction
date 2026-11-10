#!/bin/bash
# Usage: g2_run.sh <task> <out.jsonl> <spec> ...   (runs g2_r2_search.py in the pinned uv environment)
HERE="$(cd "$(dirname "$0")" && pwd)"
exec uv run --no-project --python 3.11 --with pandas==3.1.0 --with numpy==2.4.6 --with scipy==1.17.1 \
  --with pyarrow==26.0.1 --with torch --with torch_geometric --with scikit-learn \
  python "$HERE/g2_r2_search.py" "$@"
