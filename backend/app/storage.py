from __future__ import annotations

import json
import os
import time
import uuid
from typing import Dict, List, Optional
import shutil

from .schemas import ModelMeta


DATA_DIR = os.environ.get("APP_DATA_DIR", os.path.join(os.getcwd(), "data"))
STORE_DIR = os.path.join(DATA_DIR, "model_store")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
META_FILE = os.path.join(STORE_DIR, "models.json")


def ensure_dirs() -> None:
    os.makedirs(STORE_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


def _read_meta() -> Dict[str, ModelMeta]:
    if not os.path.exists(META_FILE):
        return {}
    with open(META_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: ModelMeta(**v) for k, v in raw.items()}


def _write_meta(models: Dict[str, ModelMeta]) -> None:
    serial = {k: v.model_dump() for k, v in models.items()}
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(serial, f, indent=2)


def list_models() -> List[ModelMeta]:
    return list(_read_meta().values())


def get_active_model() -> Optional[ModelMeta]:
    for m in _read_meta().values():
        if m.active:
            return m
    return None


def set_active_model(model_id: str) -> Optional[ModelMeta]:
    models = _read_meta()
    if model_id not in models:
        return None
    for m in models.values():
        m.active = False
    models[model_id].active = True
    _write_meta(models)
    return models[model_id]


def save_uploaded_model(name: str, features: List[str], file_bytes: bytes) -> ModelMeta:
    ensure_dirs()
    model_id = str(uuid.uuid4())
    model_dir = os.path.join(STORE_DIR, model_id)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "wb") as f:
        f.write(file_bytes)
    meta = ModelMeta(
        id=model_id,
        name=name,
        features=features,
        created_at=time.time(),
        path=model_path,
        active=False,
    )
    models = _read_meta()
    models[model_id] = meta
    _write_meta(models)
    return meta


def get_model(model_id: str) -> Optional[ModelMeta]:
    models = _read_meta()
    return models.get(model_id)


def delete_model(model_id: str) -> bool:
    """Delete a model's metadata and files. If it was active, no model remains active."""
    models = _read_meta()
    meta = models.pop(model_id, None)
    if meta is None:
        return False
    # Remove files on disk
    try:
        model_dir = os.path.dirname(meta.path)
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
    except Exception:
        # Best-effort deletion of files; continue to persist metadata removal
        pass
    # Ensure no other model accidentally left active if the deleted was active
    if meta.active:
        # No model is set active now
        for m in models.values():
            m.active = False
    _write_meta(models)
    return True

