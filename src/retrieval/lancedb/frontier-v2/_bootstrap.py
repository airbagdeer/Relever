"""sys.path + minimax-module bootstrap for the frontier-v2 indexers.

frontier-v2 is a *sibling* of the proven ``minimax`` package and deliberately
reuses its stable library functions (checkpoint store, corpus dataclass +
cache helpers, the dense encode/bucketing/OOM machinery, and the two
multi-vector index functions) instead of forking ~1000 lines. This module
wires the import paths so that:

  - ``from config import ...``      resolves to ``src/retrieval/config.py``
  - ``from checkpoint import ...``  resolves to ``minimax/checkpoint.py``
  - ``from corpus_loader import ...`` resolves to ``minimax/corpus_loader.py``
  - ``import paths`` / ``model_registry`` / ``corpus_loader_v2`` resolve to the
    frontier-v2 siblings (the running script's own directory is already on
    ``sys.path[0]``).

The three ``index_*`` filenames collide by name between ``minimax/`` and
``frontier-v2/``; to import the minimax versions unambiguously we load them by
explicit file path via :func:`load_minimax` under a distinct module name.

Importing this module has the side effect of mutating ``sys.path`` (idempotent).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_HERE = Path(__file__).resolve().parent           # .../src/retrieval/lancedb/frontier-v2
_LANCEDB_DIR = _HERE.parent                        # .../src/retrieval/lancedb
_MINIMAX = _LANCEDB_DIR / "minimax"                # .../src/retrieval/lancedb/minimax
_RETRIEVAL = _HERE.parents[1]                      # .../src/retrieval  (parents: [0]=lancedb, [1]=retrieval)

# Insert all three layers at index 0; the final visible order is
# [retrieval, minimax, frontier-v2, <script dir>]. There are no bare-import
# name clashes between the three layers (config is only in retrieval,
# checkpoint/corpus_loader only in minimax, and paths/model_registry/
# corpus_loader_v2/_dense_core only in frontier-v2), so the priority order is
# immaterial in practice. The only same-named modules are the index_* scripts,
# which we never bare-import — they are loaded by explicit path via load_minimax.
for _p in (str(_HERE), str(_MINIMAX), str(_RETRIEVAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_minimax(filename: str, modname: str) -> ModuleType:
    """Load a minimax module by explicit path under a unique ``modname``.

    Used for the three same-named ``index_*`` modules so we get the minimax
    implementation regardless of sys.path ordering. The loaded module's own
    ``from config/checkpoint/corpus_loader import ...`` lines resolve via the
    paths inserted above.
    """
    path = _MINIMAX / filename
    if not path.exists():
        raise FileNotFoundError(f"minimax module not found: {path}")
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


MINIMAX_DIR = _MINIMAX
RETRIEVAL_DIR = _RETRIEVAL
FRONTIER_V2_DIR = _HERE
