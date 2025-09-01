from __future__ import annotations

import json
import os
import threading
import time
from statistics import mean
from typing import Dict, List


from .storage import LOGS_DIR


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.total_requests = 0
        self.total_errors = 0
        self.latencies_ms: List[float] = []
        self.per_model_requests: Dict[str, int] = {}
        self.per_model_errors: Dict[str, int] = {}
        os.makedirs(LOGS_DIR, exist_ok=True)
        self.log_path = os.path.join(LOGS_DIR, "requests.jsonl")

    def record(self, model_id: str, ok: bool, latency_ms: float, extra: Dict) -> None:
        with self._lock:
            self.total_requests += 1
            if not ok:
                self.total_errors += 1
                self.per_model_errors[model_id] = self.per_model_errors.get(model_id, 0) + 1
            self.latencies_ms.append(latency_ms)
            self.per_model_requests[model_id] = self.per_model_requests.get(model_id, 0) + 1

            entry = {
                "ts": time.time(),
                "model_id": model_id,
                "ok": ok,
                "latency_ms": latency_ms,
                **extra,
            }
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception:
                # Logging failure should not crash the server
                pass

    def snapshot(self) -> Dict:
        with self._lock:
            lat_sorted = sorted(self.latencies_ms)
            p95 = 0.0
            if lat_sorted:
                idx = int(0.95 * (len(lat_sorted) - 1))
                p95 = lat_sorted[idx]
            return {
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "avg_latency_ms": mean(self.latencies_ms) if self.latencies_ms else 0.0,
                "p95_latency_ms": p95,
                "per_model_requests": dict(self.per_model_requests),
                "per_model_errors": dict(self.per_model_errors),
                "started_at": self.started_at,
            }

    def _reset_counters_unlocked(self) -> None:
        self.total_requests = 0
        self.total_errors = 0
        self.latencies_ms = []
        self.per_model_requests = {}
        self.per_model_errors = {}

    def rebuild_from_log(self) -> None:
        """Recompute all aggregates by reading the requests log file."""
        with self._lock:
            self._reset_counters_unlocked()
            if not os.path.exists(self.log_path):
                return
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                        except Exception:
                            continue
                        self.total_requests += 1
                        ok = bool(entry.get("ok", False))
                        if not ok:
                            self.total_errors += 1
                        mid = str(entry.get("model_id", "-"))
                        self.per_model_requests[mid] = self.per_model_requests.get(mid, 0) + 1
                        if not ok:
                            self.per_model_errors[mid] = self.per_model_errors.get(mid, 0) + 1
                        lat = float(entry.get("latency_ms", 0.0))
                        self.latencies_ms.append(lat)
            except Exception:
                # Ignore rebuild errors; metrics will just reset
                pass

    def purge_model_data(self, model_id: str) -> None:
        """Remove all log lines for a model and rebuild counters."""
        with self._lock:
            if not os.path.exists(self.log_path):
                # Nothing to purge
                self.per_model_requests.pop(model_id, None)
                self.per_model_errors.pop(model_id, None)
                return
            tmp_path = self.log_path + ".tmp"
            try:
                with open(self.log_path, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
                    for line in src:
                        try:
                            entry = json.loads(line)
                            if str(entry.get("model_id")) == model_id:
                                continue  # skip
                        except Exception:
                            # Keep malformed lines
                            pass
                        dst.write(line)
                # Replace atomically
                os.replace(tmp_path, self.log_path)
            except Exception:
                # Best effort; clean tmp if exists
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
            # Clear per-model quick views and rebuild everything from file
            self._reset_counters_unlocked()
        # Rebuild outside the file write block
        self.rebuild_from_log()


metrics = Metrics()
