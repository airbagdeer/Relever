"""pytest configuration for the frontier-v2 evaluation tests.

Registers the custom markers used by the test suite so a bare ``uv run pytest``
in this directory runs warning-free. The evaluation modules are flat siblings
(the ``frontier-v2`` dir name has a hyphen and is not an importable package), so
pytest's default ``prepend`` import mode — which puts the test file's own
directory on ``sys.path`` — is what makes ``import metrics`` resolve here.
"""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, isolated unit test")
    config.addinivalue_line("markers", "integration: touches real data on disk")
