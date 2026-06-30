from __future__ import annotations

from contextlib import asynccontextmanager
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .model import load_model
from .processor import (
    BOUNDARY,
    encode_jpeg,
    render_preview_frame,
    save_upload_to_temp,
    stream_as_multipart,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Anime Motion Flow API",
    version="1.0.0",
    description="FastAPI backend for RAFT-powered anime motion estimation.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict[str, Path] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/process-video")
async def process_video(file: UploadFile = File(...)):
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    if "mp4" not in content_type and not filename.endswith(".mp4"):
        raise HTTPException(status_code=415, detail="Only MP4 uploads are supported")

    job_id = uuid.uuid4().hex
    temp_path = save_upload_to_temp(file)
    jobs[job_id] = temp_path

    stream_url = f"/api/process-video/{job_id}/stream"
    return JSONResponse(
        {
            "job_id": job_id,
            "filename": file.filename,
            "stream_url": stream_url,
            "status": "queued",
        }
    )


@app.get("/api/process-video/{job_id}/stream")
def stream_video(job_id: str):
    video_path = jobs.get(job_id)
    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail="Processing job not found")

    return StreamingResponse(
        stream_as_multipart(video_path)(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY.decode('ascii')}",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/process-video/{job_id}/preview")
def preview_video(job_id: str):
    video_path = jobs.get(job_id)
    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail="Processing job not found")

    frame = render_preview_frame(video_path)
    return Response(
        content=encode_jpeg(frame),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.delete("/api/process-video/{job_id}")
def delete_job(job_id: str):
    video_path = jobs.pop(job_id, None)
    if video_path and video_path.exists():
        try:
            video_path.unlink()
        except OSError:
            pass
    return {"status": "deleted"}
