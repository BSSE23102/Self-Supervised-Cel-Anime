from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np

from .model import infer_flow
from .visualization import render_academic_flow_visualization

BOUNDARY = b"frame"


def _draw_status_label(frame: np.ndarray, label: str) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (12, 12), (360, 54), (0, 0, 0), thickness=-1)
    cv2.putText(
        output,
        label,
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return output


def save_upload_to_temp(upload_file) -> Path:
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


def compute_structure_edge_map(frame: np.ndarray) -> np.ndarray:
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
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("Could not encode output frame")
    return buffer.tobytes()


def multipart_frame_generator(video_path: Path) -> Generator[bytes, None, None]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be decoded by OpenCV")

    try:
        ok, prev_frame = capture.read()
        if not ok:
            raise ValueError("No readable frames found in the uploaded video")

        prev_edge_map = compute_structure_edge_map(prev_frame)
        yield _multipart_chunk(_draw_status_label(prev_frame, "Stream initialized"))

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            edge_map = compute_structure_edge_map(frame)
            flow = infer_flow(prev_edge_map, edge_map)
            overlay = render_academic_flow_visualization(frame, flow)
            yield _multipart_chunk(overlay)

            prev_edge_map = edge_map
    finally:
        capture.release()


def render_preview_frame(video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be decoded by OpenCV")

    try:
        ok, frame = capture.read()
        if not ok:
            raise ValueError("No readable frames found in the uploaded video")
        return _draw_status_label(frame, "Preview ready")
    finally:
        capture.release()


def _multipart_chunk(frame: np.ndarray) -> bytes:
    jpeg = encode_jpeg(frame)
    headers = (
        b"--" + BOUNDARY + b"\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
    )
    return headers + jpeg + b"\r\n"


def stream_as_multipart(video_path: Path):
    def generator():
        yield from multipart_frame_generator(video_path)

    return generator
