# Anime Motion Flow Backend

FastAPI service for MP4 upload, frame-by-frame OpenCV preprocessing, and future RAFT inference integration.

## Setup

```powershell
uv sync
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

If your machine needs CPU-only PyTorch or a different CUDA runtime, edit the PyTorch index in `pyproject.toml` before running `uv sync`.
