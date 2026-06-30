from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .model import load_model
from .processor import (
    BOUNDARY,
    MotionMetadata,
    encode_jpeg,
    render_frame_at_index,
    render_preview_frame,
    save_upload_to_temp,
    stream_as_multipart,
)

MOTION_INDEX_PATH = Path("motion_index.json")


@dataclass
class VideoJob:
    video_path: Path
    motion_registry: list[MotionMetadata] = field(default_factory=list)


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

jobs: dict[str, VideoJob] = {}


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
    motion_registry: list[MotionMetadata] = []
    temp_path = save_upload_to_temp(file)
    jobs[job_id] = VideoJob(video_path=temp_path, motion_registry=motion_registry)

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
    job = jobs.get(job_id)
    if not job or not job.video_path.exists():
        raise HTTPException(status_code=404, detail="Processing job not found")

    job.motion_registry.clear()
    return StreamingResponse(
        stream_as_multipart(
            job.video_path,
            motion_registry=job.motion_registry,
            motion_index_path=MOTION_INDEX_PATH,
        )(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY.decode('ascii')}",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/process-video/{job_id}/preview")
def preview_video(job_id: str):
    job = jobs.get(job_id)
    if not job or not job.video_path.exists():
        raise HTTPException(status_code=404, detail="Processing job not found")

    frame = render_preview_frame(job.video_path)
    return Response(
        content=encode_jpeg(frame),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/process-video/{job_id}/frame/{frame_index}")
def frame_thumbnail(job_id: str, frame_index: int):
    job = jobs.get(job_id)
    if not job or not job.video_path.exists():
        raise HTTPException(status_code=404, detail="Processing job not found")

    try:
        frame = render_frame_at_index(job.video_path, frame_index)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return Response(
        content=encode_jpeg(frame),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.delete("/api/process-video/{job_id}")
def delete_job(job_id: str):
    job = jobs.pop(job_id, None)
    if job and job.video_path.exists():
        try:
            job.video_path.unlink()
        except OSError:
            pass
    return {"status": "deleted"}


def _load_motion_index() -> list[MotionMetadata]:
    if not MOTION_INDEX_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="motion_index.json was not found. Process a video to completion first.",
        )

    try:
        data = json.loads(MOTION_INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="motion_index.json is invalid") from exc

    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail="motion_index.json must contain a list")

    return data


def _group_motion_segments(
    matches: list[MotionMetadata],
    job_id: str | None = None,
) -> list[dict[str, float | int | str]]:
    if not matches:
        return []

    sorted_matches = sorted(matches, key=lambda item: int(item["frame"]))
    segments: list[list[MotionMetadata]] = [[sorted_matches[0]]]

    for item in sorted_matches[1:]:
        current_frame = int(item["frame"])
        previous_frame = int(segments[-1][-1]["frame"])
        if current_frame == previous_frame + 1:
            segments[-1].append(item)
        else:
            segments.append([item])

    results: list[dict[str, float | int | str]] = []
    for segment in segments:
        velocities = [float(item["avg_velocity"]) for item in segment]
        representative_frame = int(segment[len(segment) // 2]["frame"])
        thumbnail_url = (
            f"/api/process-video/{job_id}/frame/{representative_frame}"
            if job_id
            else ""
        )
        results.append(
            {
                "direction": str(segment[0]["direction"]),
                "start_frame": int(segment[0]["frame"]),
                "end_frame": int(segment[-1]["frame"]),
                "representative_frame": representative_frame,
                "thumbnail_url": thumbnail_url,
                "start_timestamp": float(segment[0]["timestamp"]),
                "end_timestamp": float(segment[-1]["timestamp"]),
                "frame_count": len(segment),
                "mean_velocity": float(sum(velocities) / len(velocities)),
                "peak_velocity": float(max(velocities)),
            }
        )

    return results


@app.get("/api/search-actions")
def search_actions(
    direction: str = Query(..., pattern="^(left|right|up|down|static)$"),
    min_velocity: float = Query(0.0, ge=0.0),
    job_id: str | None = Query(None),
):
    normalized_direction = direction.lower()
    motion_index = _load_motion_index()
    matches = [
        item
        for item in motion_index
        if str(item.get("direction", "")).lower() == normalized_direction
        and float(item.get("avg_velocity", 0.0)) >= min_velocity
    ]
    active_job_id = job_id or next(reversed(jobs), None)
    segments = _group_motion_segments(matches, active_job_id)
    return {
        "query": {
            "direction": normalized_direction,
            "min_velocity": min_velocity,
            "job_id": active_job_id,
        },
        "match_count": len(matches),
        "segments": segments,
    }
