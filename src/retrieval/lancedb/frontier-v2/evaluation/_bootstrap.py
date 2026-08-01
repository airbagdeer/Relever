"""sys.path bootstrap for the frontier-v2 *evaluation* package.

The ``frontier-v2`` directory name contains a hyphen, so it cannot be imported
as a Python package. The established convention in this tree (see
``frontier-v2/_bootstrap.py``) is therefore: run each script standalone with its
own directory on ``sys.path[0]`` and use *flat sibling imports* (``import
metrics``), after a tiny shim wires up the central ``config`` module.

This evaluation bootstrap is intentionally narrower than the parent one: the
evaluation layer reads only ranked candidate JSONs + label JSONLs from disk and
needs nothing from ``minimax`` or the parent ``frontier-v2`` indexers. It only
has to make ``from config import ...`` resolve to ``src/retrieval/config.py``.

To avoid shadowing the central ``config.py`` (and the parent ``paths.py`` /
``model_registry.py``), the evaluation modules deliberately use distinct names
(``eval_config.py`` rather than ``config.py``/``paths.py``). The script's own
directory therefore stays at ``sys.path[0]`` and the retrieval dir is *appended*
so sibling imports always win.

Importing this module mutates ``sys.path`` (idempotent).
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # .../frontier-v2/evaluation
_RETRIEVAL = _HERE.parents[2]                     # .../src/retrieval (parents: [0]=frontier-v2, [1]=lancedb, [2]=retrieval)
_RERANK = _RETRIEVAL / "rerank"                   # .../src/retrieval/rerank

# Append (not insert-at-0) so the running script's own dir keeps priority for
# the flat sibling imports; we only need `config` to be importable.
if str(_RETRIEVAL) not in sys.path:
    sys.path.append(str(_RETRIEVAL))

# The rerank evaluation reads the cached reranker scores. `score_store` is the single
# source of truth for that on-disk layout, so it is imported rather than reimplemented.
# It bootstraps via `_rerank_bootstrap` (deliberately NOT named `_bootstrap`) so this
# appended path cannot shadow the `_bootstrap` module of the evaluation package itself.
if str(_RERANK) not in sys.path:
    sys.path.append(str(_RERANK))

RETRIEVAL_DIR = _RETRIEVAL
EVALUATION_DIR = _HERE
RERANK_DIR = _RERANK
