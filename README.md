MLOps Dashboard for Small Teams (MVP)

Overview
- Upload a trained model, deploy as an API, and monitor basic metrics.
- Built with Python (FastAPI) and a simple web UI.
- Local file-based model registry and deployment state; no external DB required.

Quick Start
1) Install Python 3.10+.
2) Create a virtualenv and install dependencies:
   - `python -m venv .venv && . .venv/Scripts/activate` (Windows)
   - `pip install -r requirements.txt`
3) Run the server:
   - `uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000`
4) Open the dashboard:
   - Visit `http://localhost:8000` in your browser.

Example Usage
- Train any scikit-learn model and save it as `.pkl`.
- From the dashboard:
  - Upload the `.pkl`, provide a model name and features in the order expected by the model (e.g., `sepal_length,sepal_width,petal_length,petal_width`).
      - For example, input for feature for provided logreg.pkl model:  
         `f1, f2, f3, f4`
  - Deploy the model from the dropdown.
  - Test predictions via the Predict panel using JSON.  
      - For example, input for Predict panel for provided logreg.pkl model:    
         `{`    
            &emsp;`"rows": [`     
               &emsp;&emsp;`{ "f1": 5.1, "f2": 3.5, "f3": 1.4, "f4": 0.2 }`      
            &emsp;`]`      
         `}`     
 

Docker
- Build: `docker build -t mlops-dashboard .`
- Run: `docker run -it --rm -p 8000:8000 -v %cd%/data:/app/data mlops-dashboard`
- Or with compose: `docker compose up --build`

Features
- Model upload: accepts `.pkl` scikit-learn models and metadata (name, feature list).
- Deploy: choose one active model to serve predictions.
- Predict: `POST /predict` with JSON rows for quick testing.
- Monitor: `GET /metrics` (JSON) and a dashboard UI (polling every 5s).

API Summary
- `POST /models/upload` (multipart): fields `name`, `features_csv`, file `model`.
- `POST /deploy/{model_id}`: mark model as active.
- `GET /models`: list all models and deployment status.
- `POST /predict`: body `{ "rows": [ {feature: value, ...}, ... ] }`.
- `GET /metrics`: returns counts and latencies.

Project Layout
- `backend/app/main.py` — FastAPI app and routes
- `backend/app/model_manager.py` — model registry, loading, prediction
- `backend/app/metrics.py` — in-memory metrics and request logging
- `backend/app/storage.py` — file-based storage for models and metadata
- `backend/app/schemas.py` — pydantic request/response schemas
- `backend/app/static/index.html` — dashboard UI
- `requirements.txt` — Python dependencies
- `Dockerfile`, `docker-compose.yml` — containerization

Notes
- For scikit-learn pickles, ensure `scikit-learn` is installed in the runtime.
- Uploaded models are stored under `data/model_store/<model_id>/model.pkl` with metadata.
- Logs and metrics are persisted under `data/logs/`.

Limitations (MVP)
- Only `.pkl` models are supported; no auto-schema inference.
- Basic in-memory metrics; restarts reset counters (logs persist to file).
- No authentication; do not expose publicly as-is.
