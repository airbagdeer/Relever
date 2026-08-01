"""pytest configuration for the rerank tests.

The rerank package uses flat sibling imports (``import reranker_registry``), following
the convention in frontier-v2. pytest's default ``prepend`` import mode puts the *test*
file's directory on sys.path, not the package's, so we add both the rerank dir and the
frontier-v2 evaluation dir here.

Markers mirror ``frontier-v2/evaluation/conftest.py``: ``unit`` tests are pure and fast;
``integration`` tests touch real data on disk; ``gpu`` tests download and run a model and
are deselected by default (``-m "not gpu"``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent           # .../rerank/tests
_RERANK = _HERE.parent                            # .../rerank
_RETRIEVAL = _RERANK.parent                       # .../src/retrieval
_EVALUATION = _RETRIEVAL / "lancedb" / "frontier-v2" / "evaluation"

for _p in (str(_RERANK), str(_RETRIEVAL), str(_EVALUATION)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, isolated unit test")
    config.addinivalue_line("markers", "integration: touches real data on disk")
    config.addinivalue_line("markers", "gpu: downloads/runs a real model; slow")
