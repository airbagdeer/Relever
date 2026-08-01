"""Shared pytest fixtures: make sibling minimax modules importable from tests."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))                     # minimax/
sys.path.insert(0, str(_HERE.parents[2]))                 # src/retrieval/
