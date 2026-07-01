from __future__ import annotations

"""Video decoding, preprocessing, streaming, and motion-index utilities.

This module connects raw uploaded MP4 files to the model/visualization layers.
It intentionally uses generators so FastAPI can begin streaming frames while the
rest of the clip is still being processed.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Generator, TypeAlias

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np

from .model import infer_flow
from .visualization import render_academic_flow_visualization

# Boundary token used by multipart/x-mixed-replace. Each encoded JPEG frame is
# written as one part in the stream.
BOUNDARY = b"frame"

# Pixel/frame threshold used for semantic metadata extraction. Values below this
# are treated as background noise for direction and average velocity summaries.
MOTION_THRESHOLD = 2.0
MotionMetadata: TypeAlias = dict[str, int | float | str]


def _flow_tensor_to_numpy(flow_tensor: Any) -> np.ndarray:
    """Normalize Torch or NumPy flow tensors to H x W x 2 float32 arrays."""

    if hasattr(flow_tensor, "detach"):
        flow_tensor = flow_tensor.detach()
    if hasattr(flow_tensor, "cpu"):
        flow_tensor = flow_tensor.cpu()
    if hasattr(flow_tensor, "numpy"):
        flow_tensor = flow_tensor.numpy()

    flow = np.asarray(flow_tensor, dtype=np.float32)

    if flow.ndim == 4:
        if flow.shape[0] != 1:
            raise ValueError("Batched flow tensors must contain exactly one sample")
        flow = flow[0]

    if flow.ndim != 3:
        raise ValueError("Flow must have shape [2, H, W] or [H, W, 2]")

    if flow.shape[0] == 2:
        flow = np.moveaxis(flow, 0, -1)
    elif flow.shape[-1] != 2:
        raise ValueError("Flow must contain exactly two displacement channels [u, v]")

    if not np.isfinite(flow).all():
        flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)

    return flow.astype(np.float32, copy=False)


def _dominant_direction(angles: np.ndarray, mask: np.ndarray) -> str:
    """Convert dense per-pixel flow angles into one coarse direction label."""

    if not np.any(mask):
        return "static"

    active_angles = angles[mask]
    # OpenCV cartToPolar uses image coordinates: 0 degrees is right, 90 degrees
    # is down, 180 degrees is left, and 270 degrees is up.
    bins = {
        "right": int(np.count_nonzero((active_angles < 45.0) | (active_angles >= 315.0))),
        "down": int(np.count_nonzero((active_angles >= 45.0) & (active_angles < 135.0))),
        "left": int(np.count_nonzero((active_angles >= 135.0) & (active_angles < 225.0))),
        "up": int(np.count_nonzero((active_angles >= 225.0) & (active_angles < 315.0))),
    }
    return max(bins, key=bins.get)


def extract_motion_metadata(flow_tensor: Any, frame_index: int, fps: float) -> MotionMetadata:
    """Create one searchable motion descriptor for a processed frame pair.

    The dense flow field is reduced to a timestamp, dominant direction, and
    average velocity over moving pixels. This keeps the search index compact
    while preserving the main action cue for a presentation/demo workflow.
    """

    flow = _flow_tensor_to_numpy(flow_tensor)
    magnitude, angles = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    moving_mask = magnitude > MOTION_THRESHOLD

    if np.any(moving_mask):
        mean_velocity = float(np.mean(magnitude[moving_mask]))
    else:
        mean_velocity = 0.0

    safe_fps = fps if fps > 0 else 1.0
    return {
        "frame": int(frame_index),
        "timestamp": float(frame_index / safe_fps),
        "direction": _dominant_direction(angles, moving_mask),
        "avg_velocity": mean_velocity,
    }


def save_upload_to_temp(upload_file) -> Path:
    """Persist an UploadFile to a temporary MP4 path.

    FastAPI's UploadFile is stream-backed; saving it once gives OpenCV a normal
    filesystem path for VideoCapture while keeping memory usage bounded.
    """

    suffix = Path(upload_file.filename or "upload.mp4").suffix.lower()
    if suffix != ".mp4":
        suffix = ".mp4"

    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(handle.name)
    try:
        with handle:
            while True:
                chunk = upload_file.file.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    finally:
        upload_file.file.seek(0)
    return temp_path


def compute_motion_reference_frame(frame: np.ndarray) -> np.ndarray:
    """Build the 2D image used for optical-flow estimation.

    Anime clips often contain sharp line art, flat color, and compression noise.
    Bilateral smoothing preserves major boundaries while reducing noise; the
    grayscale/blurred output gives Farneback or RAFT a stable intensity surface.
    """

    if frame is None or frame.size == 0:
        raise ValueError("Empty frame received from decoder")

    smoothed = cv2.bilateralFilter(frame, d=7, sigmaColor=42, sigmaSpace=42)
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), sigmaX=0.0)


def compute_structure_edge_map(frame: np.ndarray) -> np.ndarray:
    """Return a Sobel line/structure map for research diagnostics.

    The live flow path currently uses smoothed luminance frames because edge-only
    flow was too noisy for presentation. Keeping this function makes it easy to
    compare future structural-alignment experiments without rewriting the
    preprocessing code.
    """

    if frame is None or frame.size == 0:
        raise ValueError("Empty frame received from decoder")

    smoothed = cv2.bilateralFilter(frame, d=7, sigmaColor=48, sigmaSpace=48)
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    edge_map = normalized.astype(np.uint8)
    return cv2.GaussianBlur(edge_map, (5, 5), sigmaX=0.0)


def encode_jpeg(frame: np.ndarray) -> bytes:
    """Encode one BGR frame as JPEG bytes for HTTP image responses."""

    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("Could not encode output frame")
    return buffer.tobytes()


def multipart_frame_generator(
    video_path: Path,
    motion_registry: list[MotionMetadata] | None = None,
    motion_index_path: Path | None = None,
) -> Generator[bytes, None, None]:
    """Yield multipart JPEG chunks for the processed video stream.

    Processing happens pairwise: frame_t is compared with frame_t+1 to estimate
    flow for frame_t+1. The first raw frame is yielded immediately so the UI has
    a visible image before the first pairwise flow result is ready.
    """

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be decoded by OpenCV")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_index = 1
        ok, prev_frame = capture.read()
        if not ok:
            raise ValueError("No readable frames found in the uploaded video")

        prev_motion_frame = compute_motion_reference_frame(prev_frame)
        yield _multipart_chunk(prev_frame)

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            motion_frame = compute_motion_reference_frame(frame)
            flow = infer_flow(prev_motion_frame, motion_frame)
            if motion_registry is not None:
                # Registry entries are lightweight JSON records. They let the UI
                # search motion after or during a stream without storing full
                # flow tensors for every frame.
                motion_registry.append(extract_motion_metadata(flow, frame_index, fps))

            overlay = render_academic_flow_visualization(frame, flow)
            yield _multipart_chunk(overlay)

            prev_motion_frame = motion_frame
    finally:
        capture.release()
        if motion_registry is not None and motion_index_path is not None:
            # Persist when the generator finishes. With multipart streaming, this
            # happens after the browser has consumed the clip or the stream ends.
            motion_index_path.write_text(
                json.dumps(motion_registry, indent=2),
                encoding="utf-8",
            )


def render_preview_frame(video_path: Path) -> np.ndarray:
    """Decode and return the first raw frame of a job video."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be decoded by OpenCV")

    try:
        ok, frame = capture.read()
        if not ok:
            raise ValueError("No readable frames found in the uploaded video")
        return frame
    finally:
        capture.release()


def render_frame_at_index(video_path: Path, frame_index: int) -> np.ndarray:
    """Decode a specific 1-based frame index for search-result thumbnails."""

    if frame_index < 1:
        raise ValueError("frame_index must be 1 or greater")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be decoded by OpenCV")

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index - 1, 0))
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Frame {frame_index} could not be decoded")
        return frame
    finally:
        capture.release()


def _multipart_chunk(frame: np.ndarray) -> bytes:
    """Wrap a JPEG frame in multipart response headers."""

    jpeg = encode_jpeg(frame)
    headers = (
        b"--" + BOUNDARY + b"\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
    )
    return headers + jpeg + b"\r\n"


def stream_as_multipart(
    video_path: Path,
    motion_registry: list[MotionMetadata] | None = None,
    motion_index_path: Path | None = None,
):
    """Factory used by FastAPI so each request receives a fresh generator."""

    def generator():
        yield from multipart_frame_generator(video_path, motion_registry, motion_index_path)

    return generator
