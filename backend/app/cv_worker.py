"""
Standalone frame-processing module.

This file is intentionally kept separate from app.py: it holds NO reference
to CUDA memory allocation or model loading. app.py loads the SAM 3.1
predictor exactly once at startup and hands it to `run_video_session` on
every "Kick Off" request. That means this module can be edited and
re-imported via importlib.reload() (see the /dev-reload endpoint in app.py)
without ever tearing down or reloading the multi-GB model weights sitting
in VRAM.

Keep this file free of module-level state that depends on the predictor —
reload() re-executes the whole module body, so anything stateful here would
be reset on every hot reload.

The SAM 3.1 predictor returned by `build_sam3_predictor` is a session/video
API (start_session -> add_prompt -> propagate_in_video), not a per-frame
image predictor, so a "run" here means: open a fresh session on the video,
prompt frame 0 with a text concept, and stream annotated frames out as the
model propagates through the clip.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger("uvicorn.error")

# Text concept the predictor is prompted with on frame 0 of every session.
PROMPT_TEXT = "fish"

MASK_COLOR_BGR = (60, 180, 255)  # amber overlay
MASK_ALPHA = 0.45


def run_video_session(predictor, video_source, stop_event, on_frame):
    """
    Start a fresh SAM 3.1 session on `video_source`, prompt it with
    PROMPT_TEXT on frame 0, and propagate forward through the video.

    Calls `on_frame(annotated_bgr_frame)` for every frame the model produces
    a result for, in order, until the video ends or `stop_event` is set
    (e.g. the Pause button, or a new Kick Off superseding this run).

    Args:
        predictor: The SAM 3.1 session predictor (handle_request /
            handle_stream_request API) loaded once at process startup.
        video_source: Path to a video file (or frame directory) to run on.
        stop_event: threading.Event; checked between frames to allow
            cooperative cancellation.
        on_frame: Callback invoked with each annotated BGR frame.
    """
    session = predictor.handle_request(
        {"type": "start_session", "resource_path": video_source}
    )
    session_id = session["session_id"]

    try:
        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": PROMPT_TEXT,
            }
        )

        capture = cv2.VideoCapture(video_source)
        try:
            next_frame_idx = 0
            stream = predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": 0,
                }
            )
            for update in stream:
                if stop_event.is_set():
                    predictor.handle_request({"type": "cancel_propagation", "session_id": session_id})
                    break

                frame_idx = update["frame_index"]

                # Internal buffering (hotstart delay / confirmation flush)
                # can replay a frame index we've already advanced past;
                # skip it rather than treating it as end-of-video.
                if frame_idx < next_frame_idx:
                    continue

                # Frames are otherwise produced in increasing order, so read
                # forward in lockstep instead of reseeking every frame.
                frame = None
                while next_frame_idx <= frame_idx:
                    success, candidate = capture.read()
                    if not success:
                        break
                    frame = candidate
                    next_frame_idx += 1

                if frame is None:
                    break  # genuinely out of video frames

                annotated = _overlay_masks(frame, update["outputs"].get("out_binary_masks"))
                on_frame(annotated)
        finally:
            capture.release()
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})


def _overlay_masks(frame: np.ndarray, binary_masks) -> np.ndarray:
    """Blend a semi-transparent color layer over every detected object's mask."""
    if binary_masks is None or len(binary_masks) == 0:
        return frame

    combined_mask = np.zeros(frame.shape[:2], dtype=bool)
    for mask in binary_masks:
        combined_mask |= mask.astype(bool)

    overlay = frame.copy()
    overlay[combined_mask] = MASK_COLOR_BGR
    blended = cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0)

    contours, _ = cv2.findContours(
        combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, contours, -1, MASK_COLOR_BGR, thickness=2)

    return blended
