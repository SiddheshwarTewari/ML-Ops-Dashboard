from __future__ import annotations

import os
import time
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .metrics import metrics
from .model_manager import manager
from .schemas import PredictRequest, PredictResponse, ModelsList, ModelMeta
from .storage import ensure_dirs, list_models, save_uploaded_model, set_active_model, get_active_model


app = FastAPI(title="MLOps Dashboard (MVP)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static dashboard
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
def on_startup() -> None:
    ensure_dirs()


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/models", response_model=ModelsList)
def get_models():
    return ModelsList(models=list_models())


@app.post("/models/upload", response_model=ModelMeta)
async def upload_model(
    name: str = Form(...),
    features_csv: str = Form(...),
    model: UploadFile = File(...),
):
    if not model.filename.endswith((".pkl", ".pickle")):
        raise HTTPException(status_code=400, detail="Only .pkl/.pickle files supported in MVP")
    features = [f.strip() for f in features_csv.split(",") if f.strip()]
    content = await model.read()
    meta = save_uploaded_model(name=name, features=features, file_bytes=content)
    return meta


@app.post("/deploy/{model_id}", response_model=ModelMeta)
def deploy_model(model_id: str):
    meta = set_active_model(model_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Model not found")
    # Refresh loaded model
    manager.ensure_active_loaded()
    return meta


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()
    model_id = "-"
    try:
        model_id, preds = manager.predict(req.rows)
        ok = True
        return PredictResponse(model_id=model_id, predictions=preds)
    except Exception as e:
        ok = False
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        elapsed_ms = (time.time() - start) * 1000.0
        metrics.record(model_id=model_id, ok=ok, latency_ms=elapsed_ms, extra={"n_rows": len(req.rows)})


@app.get("/metrics")
def get_metrics():
    return JSONResponse(metrics.snapshot())


