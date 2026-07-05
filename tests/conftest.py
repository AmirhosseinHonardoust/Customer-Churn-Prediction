"""Make ``src`` and ``data`` importable in tests (repo uses script-style imports)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "data"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)
