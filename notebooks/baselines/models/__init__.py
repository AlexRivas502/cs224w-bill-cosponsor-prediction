"""Model zoo: heuristics, shared heads and the GNN baselines.

`build_model(name, cfg, meta)` maps a model name to its class. GNN modules are imported lazily, so the
package works (heuristics, PairMLP) before every GNN file exists; `available_models()` lists what can run.
"""
from __future__ import annotations

import importlib

# name -> (module, class). pair_mlp is the shared GraphModel base without message passing (= G1 with L = 0).
REGISTRY = {
    "pair_mlp": ("heads", "GraphModel"),
    "hetero_sage": ("hetero_sage", "HeteroSAGE"),
    "rgcn": ("rgcn", "RGCN"),
    "lightgcn": ("lightgcn", "LightGCNFeat"),
}
DISPLAY = {"pair_mlp": "PairMLP", "hetero_sage": "G1 HeteroSAGE", "rgcn": "G2 R-GCN", "lightgcn": "G3 LightGCN-feat"}


def model_class(name: str):
    """The model class registered under `name` (ImportError / AttributeError if not implemented yet)."""
    mod, cls = REGISTRY[name]
    return getattr(importlib.import_module(f"{__name__}.{mod}"), cls)


def available_models() -> list[str]:
    """Registered model names whose module and class import cleanly."""
    out = []
    for n in REGISTRY:
        try:
            model_class(n)
            out.append(n)
        except (ImportError, AttributeError):
            pass
    return out


def build_model(name: str, cfg, meta: dict):
    """Instantiate model `name` with its config dataclass and the data meta dict (dims, sizes)."""
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {list(REGISTRY)}")
    if name == "pair_mlp" and getattr(cfg, "layers", 0) != 0:
        raise ValueError("pair_mlp must have layers = 0")
    return model_class(name)(cfg, meta)
