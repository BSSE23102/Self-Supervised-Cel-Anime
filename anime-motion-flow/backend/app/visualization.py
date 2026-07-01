from __future__ import annotations

"""Optical-flow visualization utilities.

The live UI uses a clean vector overlay rather than a full-frame heatmap. Dense
HSV flow maps are still supported for diagnostics, but the default renderer
prioritizes readable presentation output: original frame plus sparse,
high-confidence motion arrows.
"""

from typing import Any

import cv2
import numpy as np


def _flow_tensor_to_numpy(flow_tensor: Any) -> np.ndarray:
    """Accept Torch/NumPy flow tensors and return H x W x 2 float32 arrays."""

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
    """Resize flow to match a frame while preserving displacement units."""

    frame_h, frame_w = frame_shape
    flow_h, flow_w = flow.shape[:2]

    if (flow_h, flow_w) == (frame_h, frame_w):
        return flow

    scale_x = frame_w / max(flow_w, 1)
    scale_y = frame_h / max(flow_h, 1)
    resized = cv2.resize(flow, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

    # Resizing changes the spatial coordinate system. The vector components must
    # be scaled too, otherwise a 10-pixel displacement at half resolution would
    # still look like only 10 pixels after returning to full resolution.
    resized[..., 0] *= scale_x
    resized[..., 1] *= scale_y
    return resized


def _robust_value_channel(
    magnitude: np.ndarray,
    *,
    activation_percentile: float,
    saturation_percentile: float,
) -> np.ndarray:
    """Map motion magnitude to a robust HSV value/brightness channel.

    Percentile normalization prevents a single extreme vector from making every
    other moving region look black. This is only used when show_heatmap=True.
    """

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

    # Gamma-like compression keeps subtle action visible without letting noisy
    # mid-range motion dominate the diagnostic layer.
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
    """Draw sparse arrows only where local motion is strong and coherent.

    A generic grid of arrows makes anime scenes unreadable. This routine samples
    a grid, looks inside a local window, and draws an arrow only if enough pixels
    exceed the adaptive magnitude threshold.
    """

    height, width = magnitude.shape
    half_window = max(step // 3, 5)
    max_vector_length = max(float(step) * 1.05, 1.0)
    min_vector_length = max(float(step) * 0.12, 2.5)

    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            y0 = max(y - half_window, 0)
            y1 = min(y + half_window + 1, height)
            x0 = max(x - half_window, 0)
            x1 = min(x + half_window + 1, width)

            local_magnitude = magnitude[y0:y1, x0:x1]
            active_mask = local_magnitude >= threshold
            if not np.any(active_mask):
                continue

            active_ratio = float(np.count_nonzero(active_mask)) / float(active_mask.size)
            if active_ratio < 0.025:
                continue

            # Magnitude-weighted averaging gives the arrow the dominant local
            # direction instead of trusting a single noisy grid-center vector.
            weights = local_magnitude[active_mask].astype(np.float32)
            weight_sum = float(weights.sum())
            if weight_sum <= 1e-6:
                continue

            local_flow = flow[y0:y1, x0:x1][active_mask]
            dx = float(np.sum(local_flow[:, 0] * weights) / weight_sum)
            dy = float(np.sum(local_flow[:, 1] * weights) / weight_sum)
            length = float(np.hypot(dx, dy))
            if length < min_vector_length:
                continue

            yy, xx = np.mgrid[y0:y1, x0:x1]
            start_x = int(np.clip(round(float(np.sum(xx[active_mask] * weights) / weight_sum)), 0, width - 1))
            start_y = int(np.clip(round(float(np.sum(yy[active_mask] * weights) / weight_sum)), 0, height - 1))

            # Cap visual length so fast cuts do not produce arrows that stretch
            # across the whole frame and hide the source animation.
            scale = min(1.0, max_vector_length / length)
            end_x = int(np.clip(round(start_x + dx * scale), 0, width - 1))
            end_y = int(np.clip(round(start_y + dy * scale), 0, height - 1))

            cv2.arrowedLine(
                arrows,
                (start_x, start_y),
                (end_x, end_y),
                color=(36, 255, 94),
                thickness=2,
                line_type=cv2.LINE_AA,
                tipLength=0.24,
            )

    return arrows


def render_academic_flow_visualization(
    frame: np.ndarray,
    flow_tensor: Any,
    *,
    sample_step: int = 40,
    activation_percentile: float = 86.0,
    saturation_percentile: float = 99.2,
    adaptive_percentile: float = 98.0,
    min_arrow_magnitude: float = 6.0,
    frame_alpha: float = 1.0,
    hsv_alpha: float = 0.0,
    arrow_alpha: float = 0.95,
    show_heatmap: bool = False,
) -> np.ndarray:
    """
    Render a clean optical-flow diagnostic frame.

    By default the output preserves the source frame and overlays only sparse,
    locally averaged vectors from high-motion regions. The optional HSV layer is
    retained for research diagnostics but disabled for the live stream because it
    can obscure semantic image content.
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

    # Smooth the vector field before visualization. This does not change the
    # search metadata; it only makes the rendered arrows less jittery.
    flow = cv2.GaussianBlur(flow, (0, 0), sigmaX=1.8, sigmaY=1.8)

    u = flow[..., 0]
    v = flow[..., 1]
    magnitude, angle = cv2.cartToPolar(u, v, angleInDegrees=True)

    # The threshold adapts per frame: a quiet scene draws little or nothing, and
    # a fast action scene shows only the top-motion regions.
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

    if show_heatmap:
        # OpenCV HSV hue is [0, 179], so degrees [0, 360] are divided by 2.
        # Hue represents direction; value represents robustly normalized speed.
        hsv = np.zeros(source_frame.shape, dtype=np.uint8)
        hsv[..., 0] = np.mod(angle / 2.0, 180.0).astype(np.uint8)
        hsv[..., 1] = 180
        hsv[..., 2] = _robust_value_channel(
            magnitude,
            activation_percentile=activation_percentile,
            saturation_percentile=saturation_percentile,
        )
        dense_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        base = cv2.addWeighted(source_frame, frame_alpha, dense_bgr, hsv_alpha, 0.0)
    else:
        base = source_frame

    # The arrow layer is black except where arrows are drawn, so addWeighted()
    # preserves the source frame and adds green vectors on top.
    composite = cv2.addWeighted(base, 1.0, arrow_layer, arrow_alpha, 0.0)
    return np.clip(composite, 0, 255).astype(np.uint8)
