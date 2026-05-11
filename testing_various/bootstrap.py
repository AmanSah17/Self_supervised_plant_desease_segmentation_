from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root(levels_up: int = 2) -> Path:
    """Add the repository root to sys.path for relocated helper scripts."""
    root = Path(__file__).resolve().parents[levels_up - 1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
