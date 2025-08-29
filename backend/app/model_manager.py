from __future__ import annotations

import importlib
import json
import os
from typing import Any, Dict, List, Optional

import pickle

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover - optional
    joblib = None  # type: ignore

from .storage import get_active_model, list_models

try:  # numpy is available via requirements
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


class LoadedModel:
    def __init__(self, model_id: str, model_obj: Any, features: List[str]):
        self.model_id = model_id
        self.model = model_obj
        self.features = features

    def predict(self, rows: List[Dict[str, Any]]):
        # Convert list[dict] -> 2D list in feature order
        X = [[row.get(f) for f in self.features] for row in rows]
        # Try scikit-learn style predict
        if hasattr(self.model, "predict"):
            preds = self.model.predict(X)
            preds_list = list(preds)
            # Convert numpy types to Python builtins for JSON serialization
            def _to_py(v: Any) -> Any:
                try:
                    if np is not None and isinstance(v, np.generic):
                        return v.item()
                except Exception:
                    pass
                try:
                    return v.tolist()  # type: ignore[attr-defined]
                except Exception:
                    return v
            return [_to_py(p) for p in preds_list]
        # If model is a callable
        if callable(self.model):
            return [self.model(x) for x in X]
        raise ValueError("Model does not support prediction")


class ModelManager:
    def __init__(self) -> None:
        self.loaded: Optional[LoadedModel] = None

    def _load_file(self, path: str):
        # Prefer joblib if available; fallback to pickle
        if joblib is not None:
            return joblib.load(path)
        with open(path, "rb") as f:
            return pickle.load(f)

    def ensure_active_loaded(self) -> Optional[LoadedModel]:
        active = get_active_model()
        if active is None:
            self.loaded = None
            return None
        if self.loaded and self.loaded.model_id == active.id:
            return self.loaded
        # Load from disk
        model_obj = self._load_file(active.path)
        self.loaded = LoadedModel(active.id, model_obj, active.features)
        return self.loaded

    def predict(self, rows: List[Dict[str, Any]]):
        lm = self.ensure_active_loaded()
        if not lm:
            raise RuntimeError("No active model deployed")
        return lm.model_id, lm.predict(rows)


manager = ModelManager()
