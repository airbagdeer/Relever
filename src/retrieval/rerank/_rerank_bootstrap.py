"""sys.path bootstrap for the rerank package.

Mirrors the convention established by ``frontier-v2/_bootstrap.py`` and
``frontier-v2/evaluation/_bootstrap.py``: each script in this directory is run
standalone (so its own directory is ``sys.path[0]``) and uses *flat sibling
imports* (``import reranker_registry``). This shim only has to make
``from config import ...`` resolve to ``src/retrieval/config.py``.

``src/retrieval`` is *appended* rather than inserted at 0 so the running
script's own directory keeps priority for sibling imports. The module names here
(``reranker_registry``, ``score_pool``, ``rerank_listwise``) deliberately avoid
colliding with ``config.py`` / ``paths.py`` / ``model_registry.py`` elsewhere in
the tree.

Importing this module mutates ``sys.path`` (idempotent).
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # .../src/retrieval/rerank
_RETRIEVAL = _HERE.parent                # .../src/retrieval

if str(_RETRIEVAL) not in sys.path:
    sys.path.append(str(_RETRIEVAL))

RETRIEVAL_DIR = _RETRIEVAL
RERANK_DIR = _HERE
