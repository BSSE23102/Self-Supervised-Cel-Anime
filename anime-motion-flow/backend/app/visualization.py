from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _flow_tensor_to_numpy(flow_tensor: Any) -> np.ndarray:
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


def _resize_flow_to_frame(flow: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    frame_h, frame_w = frame_shape
    flow_h, flow_w = flow.shape[:2]

    if (flow_h, flow_w) == (frame_h, frame_w):
        return flow

    scale_x = frame_w / max(flow_w, 1)
    scale_y = frame_h / max(flow_h, 1)
    resized = cv2.resize(flow, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    resized[..., 0] *= scale_x
    resized[..., 1] *= scale_y
    return resized


def _robust_value_channel(
    magnitude: np.ndarray,
    *,
    activation_percentile: float,
    saturation_percentile: float,
) -> np.ndarray:
    moving = magnitude[magnitude > 1e-6]
    if moving.size == 0:
        return np.zeros_like(magnitude, dtype=np.uint8)

    low = float(np.percentile(moving, activation_percentile))
    high = float(np.percentile(moving, saturation_percentile))
    if high <= low:
        low = 0.0
        high = float(moving.max())

    if high <= low:
        return np.zeros_like(magnitude, dtype=np.uint8)

    clipped = np.clip(magnitude, low, high)
    normalized = (clipped - low) / (high - low)
    normalized[magnitude <= low] = 0.0
    normalized = np.power(normalized, 1.65)
    value = np.round(normalized * 220.0).astype(np.uint8)
    return cv2.GaussianBlur(value, (0, 0), sigmaX=1.2, sigmaY=1.2)


def _draw_selective_velocity_needles(
    arrows: np.ndarray,
    flow: np.ndarray,
    magnitude: np.ndarray,
    threshold: float,
    step: int,
) -> np.ndarray:
    height, width = magnitude.shape
    max_vector_length = max(float(step) * 1.35, 1.0)

    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            speed = float(magnitude[y, x])
            if speed < threshold:
                continue

            dx = float(flow[y, x, 0])
            dy = float(flow[y, x, 1])
            length = float(np.hypot(dx, dy))
            if length <= 1e-6:
                continue

            scale = min(1.0, max_vector_length / length)
            end_x = int(np.clip(round(x + dx * scale), 0, width - 1))
            end_y = int(np.clip(round(y + dy * scale), 0, height - 1))

            cv2.arrowedLine(
                arrows,
                (x, y),
                (end_x, end_y),
                color=(80, 255, 80),
                thickness=2,
                line_type=cv2.LINE_AA,
                tipLength=0.28,
            )
            cv2.circle(arrows, (x, y), 1, (30, 180, 30), thickness=-1, lineType=cv2.LINE_AA)

    return arrows


def render_academic_flow_visualization(
    frame: np.ndarray,
    flow_tensor: Any,
    *,
    sample_step: int = 32,
    activation_percentile: float = 78.0,
    saturation_percentile: float = 99.2,
    adaptive_percentile: float = 97.0,
    min_arrow_magnitude: float = 18.0,
    frame_alpha: float = 0.82,
    hsv_alpha: float = 0.38,
    arrow_alpha: float = 0.9,
) -> np.ndarray:
    """
    Render a hybrid optical-flow diagnostic layer.

    The dense layer encodes flow direction as HSV hue and robustly normalized
    speed as HSV value. The sparse layer draws green arrows only at sampled
    locations whose velocity exceeds the adaptive high-motion threshold.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Frame must be a non-empty BGR image")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must have shape [H, W, 3] in BGR channel order")
    if sample_step < 2:
        raise ValueError("sample_step must be at least 2 pixels")

    source_frame = np.ascontiguousarray(frame, dtype=np.uint8)
    flow = _flow_tensor_to_numpy(flow_tensor)
    flow = _resize_flow_to_frame(flow, source_frame.shape[:2])
    flow = cv2.GaussianBlur(flow, (0, 0), sigmaX=1.15, sigmaY=1.15)

    u = flow[..., 0]
    v = flow[..., 1]
    magnitude, angle = cv2.cartToPolar(u, v, angleInDegrees=True)

    hsv = np.zeros(source_frame.shape, dtype=np.uint8)
    hsv[..., 0] = np.mod(angle / 2.0, 180.0).astype(np.uint8)
    hsv[..., 1] = 205
    hsv[..., 2] = _robust_value_channel(
        magnitude,
        activation_percentile=activation_percentile,
        saturation_percentile=saturation_percentile,
    )
    dense_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    moving = magnitude[magnitude > 1e-6]
    if moving.size == 0:
        threshold = float("inf")
    else:
        adaptive_threshold = float(np.percentile(moving, adaptive_percentile))
        threshold = max(adaptive_threshold, float(min_arrow_magnitude))

    arrow_layer = np.zeros_like(source_frame)
    arrow_layer = _draw_selective_velocity_needles(
        arrow_layer,
        flow,
        magnitude,
        threshold,
        sample_step,
    )

    base = cv2.addWeighted(source_frame, frame_alpha, dense_bgr, hsv_alpha, 0.0)
    composite = cv2.addWeighted(base, 1.0, arrow_layer, arrow_alpha, 0.0)
    return np.clip(composite, 0, 255).astype(np.uint8)
