from __future__ import annotations

import base64
import shutil
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

CACHE_DIR = Path("video_cache")
UPLOAD_DIR = CACHE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Anime Motion Flow Backend",
    version="0.1.0",
    description="Self-supervised cel-anime motion estimation service.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode_jpeg_data_url(frame: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError("Failed to encode preview frame")

    payload = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{payload}"


def run_raft_stub(prev_gray: np.ndarray, gray: np.ndarray) -> dict[str, Any]:
    """
    Placeholder for RAFT or another PyTorch optical-flow model.

    Replace this with frame normalization, tensorization, model inference,
    flow post-processing, and structural line-art alignment metrics.
    """
    with torch.no_grad():
        _device = "cuda" if torch.cuda.is_available() else "cpu"

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude, _angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    edges = cv2.Canny(gray, 80, 160)

    return {
        "mean_flow": float(np.mean(magnitude)),
        "max_flow": float(np.max(magnitude)),
        "line_density": float(np.count_nonzero(edges) / edges.size),
        "edges": edges,
    }


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type not in {"video/mp4", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Only MP4 video uploads are supported")

    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="File must have a .mp4 extension")

    upload_id = uuid.uuid4().hex
    video_path = UPLOAD_DIR / f"{upload_id}.mp4"

    with video_path.open("wb") as target:
        shutil.copyfileobj(file.file, target)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise HTTPException(status_code=422, detail="Uploaded video could not be decoded")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_stride = max(total_frames // 12, 1)

    frame_index = 0
    prev_gray: np.ndarray | None = None
    previews: list[str] = []
    metrics: list[dict[str, float | int]] = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame_index += 1
        resized = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            result = run_raft_stub(prev_gray, gray)
            edges_bgr = cv2.cvtColor(result["edges"], cv2.COLOR_GRAY2BGR)
            blended = cv2.addWeighted(resized, 0.72, edges_bgr, 0.28, 0)

            metrics.append(
                {
                    "frame": frame_index,
                    "mean_flow": round(float(result["mean_flow"]), 4),
                    "max_flow": round(float(result["max_flow"]), 4),
                    "line_density": round(float(result["line_density"]), 4),
                }
            )

            if len(previews) < 12 and frame_index % sample_stride == 0:
                previews.append(encode_jpeg_data_url(blended))

        prev_gray = gray

    capture.release()

    return {
        "upload_id": upload_id,
        "filename": file.filename,
        "frames_read": frame_index,
        "fps": round(fps, 3),
        "torch_device": "cuda" if torch.cuda.is_available() else "cpu",
        "metrics": metrics[:120],
        "previews": previews,
        "status": "processed_with_raft_stub",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
