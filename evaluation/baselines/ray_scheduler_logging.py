from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


class RaySchedulerLogger:
    def __init__(self, log_dir: str):
        self._file = None
        self.log_path: Optional[str] = None
        try:
            scheduler_log_dir = os.path.join(log_dir, "ray_scheduler")
            os.makedirs(scheduler_log_dir, exist_ok=True)
            self.log_path = os.path.join(
                scheduler_log_dir,
                f"ray_scheduler_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log",
            )
            self._file = open(self.log_path, "a", encoding="utf-8")
            self.info(f"Ray scheduler logging to: {self.log_path}")
        except Exception as exc:
            self._file = None
            self.log_path = None
            self.info(f"Warning: unable to open Ray scheduler log file: {exc}")

    def _write(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line)
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()

    def info(self, message: str) -> None:
        self._write(message)

    def event(self, name: str, payload: Dict[str, Any]) -> None:
        self._write(f"{name} {json.dumps(payload, sort_keys=True, default=str)}")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
