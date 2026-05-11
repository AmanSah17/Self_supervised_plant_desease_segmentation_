from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.root_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = asdict(payload) if is_dataclass(payload) else payload
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2, default=str)
        return path

    def write_text(self, name: str, content: str) -> Path:
        path = self.root_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path
