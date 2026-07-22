"""
Standalone frame-processing module.

This file is intentionally kept separate from app.py: it holds NO reference
to CUDA memory allocation or model loading. app.py loads the SAM 3.1
predictor exactly once at startup and hands it to `process_sam_frame` on
every call. That means this module can be edited and re-imported via
importlib.reload() (see the /dev-reload endpoint in app.py) without ever
tearing down or reloading the multi-GB model weights sitting in VRAM.

Keep this file free of module-level state that depends on the predictor —
reload() re-executes the whole module body, so anything stateful here would
be reset on every hot reload.
"""

import cv2
import numpy as np

# Default prompt: a single foreground point roughly in the center of frame,
# used when no explicit click/prompt has been provided by the client.
DEFAULT_PROMPT_POINT_RATIO = (0.5, 0.5)
MASK_COLOR_BGR = (60, 180, 255)  # amber overlay
MASK_ALPHA = 0.45


def process_sam_frame(frame: np.ndarray, predictor) -> np.ndarray:
    """
    Run SAM 3.1 inference on a single BGR frame using a default center-point
    prompt, overlay the resulting mask with a semi-transparent color, and
    return the annotated frame.

    Args:
        frame: BGR uint8 frame straight from the video capture source.
        predictor: A pre-loaded SAM 3.1 predictor instance (already on CUDA).

    Returns:
        The frame with a translucent mask overlay drawn on top.
    """
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    predictor.set_image(rgb_frame)

    point_x = int(width * DEFAULT_PROMPT_POINT_RATIO[0])
    point_y = int(height * DEFAULT_PROMPT_POINT_RATIO[1])
    point_coords = np.array([[point_x, point_y]])
    point_labels = np.array([1])  # 1 = foreground click

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    best_mask = masks[int(np.argmax(scores))].astype(bool)

    return _overlay_mask(frame, best_mask)


def _overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a semi-transparent color layer over the frame wherever mask is True."""
    overlay = frame.copy()
    overlay[mask] = MASK_COLOR_BGR

    blended = cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, contours, -1, MASK_COLOR_BGR, thickness=2)

    return blended
