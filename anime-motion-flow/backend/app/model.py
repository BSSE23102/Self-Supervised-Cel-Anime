from __future__ import annotations

"""Optical-flow model loading and inference.

The project is designed around RAFT, but RAFT/Torch can be heavy on local
Windows machines. This module therefore exposes one stable infer_flow() function
and chooses the best available backend at runtime:

1. RAFT when ANIME_FLOW_RAFT_MODE asks for it and Torch/TorchVision load cleanly.
2. OpenCV Farneback dense optical flow as an interactive fallback.

Both paths return the same shape: H x W x 2, with channels [u, v].
"""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any, Final

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np

# RAFT inference is protected by a lock because the model object and CUDA context
# are shared global resources in the FastAPI process.
MODEL_LOCK: Final[Lock] = Lock()
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RaftRuntime:
    """Everything required to execute a loaded RAFT model."""

    model: Any
    device: Any
    torch: Any
    functional: Any


def _infer_flow_with_opencv(frame_a_input: np.ndarray, frame_b_input: np.ndarray) -> np.ndarray:
    """Compute dense optical flow with OpenCV Farneback.

    Farneback is not as accurate as RAFT on complex motion, but it is fast,
    dependency-light, and good enough for local streaming demos. The function
    accepts the same preprocessed 2D motion-reference frames used by RAFT.
    """

    frame_a = np.squeeze(frame_a_input).astype("uint8")
    frame_b = np.squeeze(frame_b_input).astype("uint8")

    if frame_a.ndim != 2 or frame_b.ndim != 2:
        raise ValueError("Optical flow fallback expects 2D motion frames")
    if frame_a.shape != frame_b.shape:
        raise ValueError("Motion frame pair shape mismatch")

    frame_a = cv2.GaussianBlur(frame_a, (5, 5), sigmaX=0.0)
    frame_b = cv2.GaussianBlur(frame_b, (5, 5), sigmaX=0.0)

    # Work at reduced resolution to keep the stream responsive, then scale flow
    # vectors back into original pixel units after resizing.
    height, width = frame_a.shape
    work_width = max(width // 2, 64)
    work_height = max(height // 2, 64)
    small_a = cv2.resize(frame_a, (work_width, work_height), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(frame_b, (work_width, work_height), interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(
        small_a,
        small_b,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=27,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= width / work_width
    flow[..., 1] *= height / work_height
    return flow


def _load_torch_stack():
    """Import Torch/TorchVision lazily so OpenCV mode avoids heavy imports."""

    try:
        import torch
        import torch.nn.functional as functional
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        return torch, functional, Raft_Large_Weights, raft_large
    except (ImportError, OSError, RuntimeError) as exc:
        LOGGER.warning(
            "Torch/TorchVision could not be imported; using OpenCV optical flow: %s",
            exc,
        )
        return None


@lru_cache(maxsize=1)
def load_model() -> RaftRuntime | None:
    """Load RAFT once, or return None to signal OpenCV fallback.

    Environment control:
    - ANIME_FLOW_RAFT_MODE=opencv/off/false uses OpenCV only.
    - ANIME_FLOW_RAFT_MODE=cuda requires CUDA and falls back if unavailable.
    - ANIME_FLOW_RAFT_MODE=cpu forces CPU RAFT, useful for correctness tests but
      usually too slow for interactive streaming.
    """

    raft_mode = os.getenv("ANIME_FLOW_RAFT_MODE", "opencv").strip().lower()
    if raft_mode in {"", "off", "opencv", "false", "0", "no"}:
        LOGGER.info(
            "Using OpenCV optical flow. Set ANIME_FLOW_RAFT_MODE=cuda or cpu to enable RAFT."
        )
        return None

    stack = _load_torch_stack()
    if stack is None:
        return None

    torch, functional, raft_weights, raft_large = stack
    cuda_available = torch.cuda.is_available()
    allow_cpu_raft = raft_mode in {"cpu", "force", "1", "true", "yes"}
    if raft_mode == "cuda" and not cuda_available:
        LOGGER.warning("ANIME_FLOW_RAFT_MODE=cuda requested, but CUDA is unavailable.")
        return None
    if raft_mode == "auto" and not cuda_available:
        LOGGER.warning(
            "CUDA is unavailable; using OpenCV optical flow for interactive streaming. "
            "Set ANIME_FLOW_RAFT_MODE=cpu to force CPU RAFT."
        )
        return None

    device = torch.device("cuda" if cuda_available else "cpu")

    try:
        # TorchVision pads/normalizes internally for RAFT weights, but our
        # _prepare_input() still ensures dimensions are multiples of 8 because
        # the RAFT encoder/decoder pyramid expects divisible spatial shapes.
        model = raft_large(weights=raft_weights.DEFAULT, progress=True)
        model.eval()
        model = model.to(device)
        LOGGER.info("Loaded RAFT model on %s", device)
        return RaftRuntime(
            model=model,
            device=device,
            torch=torch,
            functional=functional,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.warning("RAFT model could not be loaded; using OpenCV optical flow: %s", exc)
        return None


def _prepare_input(frame_a: np.ndarray, frame_b: np.ndarray, runtime: RaftRuntime):
    """Convert 2D motion-reference frames into RAFT-compatible tensors."""

    torch = runtime.torch
    functional = runtime.functional

    tensor_a = torch.as_tensor(frame_a, dtype=torch.float32, device=runtime.device)
    tensor_b = torch.as_tensor(frame_b, dtype=torch.float32, device=runtime.device)

    if tensor_a.ndim != 2 or tensor_b.ndim != 2:
        raise ValueError("RAFT input expects 2D motion frames")
    if tensor_a.shape != tensor_b.shape:
        raise ValueError("Motion frame pair shape mismatch")

    tensor_a = tensor_a / 255.0 if tensor_a.max() > 1.0 else tensor_a
    tensor_b = tensor_b / 255.0 if tensor_b.max() > 1.0 else tensor_b

    # RAFT expects 3-channel images. The pipeline uses a stable luminance motion
    # reference, so the single channel is repeated into pseudo-RGB.
    tensor_a = tensor_a.unsqueeze(0).repeat(3, 1, 1)
    tensor_b = tensor_b.unsqueeze(0).repeat(3, 1, 1)

    height, width = tensor_a.shape[-2:]
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    padding = (0, pad_w, 0, pad_h)

    # Replicate padding avoids introducing black borders that could create false
    # motion at the image edges.
    tensor_a = functional.pad(tensor_a, padding, mode="replicate")
    tensor_b = functional.pad(tensor_b, padding, mode="replicate")
    return tensor_a.unsqueeze(0), tensor_b.unsqueeze(0), height, width


def infer_flow(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    """Return optical flow for a pair of preprocessed motion-reference frames."""

    runtime = load_model()
    if runtime is None:
        return _infer_flow_with_opencv(frame_a, frame_b)

    try:
        image1, image2, height, width = _prepare_input(frame_a, frame_b, runtime)
        with runtime.torch.no_grad():
            with MODEL_LOCK:
                flow_predictions = runtime.model(image1, image2)

        flow = flow_predictions[-1][0].permute(1, 2, 0).detach().cpu().numpy()
        return flow[:height, :width]
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.warning("RAFT inference failed; using OpenCV optical flow: %s", exc)
        return _infer_flow_with_opencv(frame_a, frame_b)
