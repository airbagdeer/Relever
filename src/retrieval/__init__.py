# Intentionally empty.
#
# The previous `from .config import *` was unused (no caller imports symbols
# via `from retrieval import ...`; everything goes through sys.path tricks
# and direct module imports). Keeping this empty avoids side-effecting
# imports (langchain_huggingface) when pytest collects tests from sibling
# venvs that don't have the full project deps installed.
