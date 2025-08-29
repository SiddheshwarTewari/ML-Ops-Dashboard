from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(default_factory=list, description="List of feature dicts")


class PredictResponse(BaseModel):
    model_id: str
    predictions: List[Any]


class ModelMeta(BaseModel):
    id: str
    name: str
    features: List[str]
    created_at: float
    path: str
    active: bool = False


class ModelsList(BaseModel):
    models: List[ModelMeta]


class MetricsSnapshot(BaseModel):
    total_requests: int
    total_errors: int
    avg_latency_ms: float
    p95_latency_ms: float
    per_model_requests: Dict[str, int]
    per_model_errors: Dict[str, int]
    started_at: float

