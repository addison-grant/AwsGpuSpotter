"""
CPU-only stand-in for the real SAM 3.1 session predictor.

The real `sam3` package unconditionally calls `.cuda()` while building the
model, so there is no way to load actual weights on a CUDA-less laptop.
This module fakes just enough of the `handle_request`/`handle_stream_request`
surface (see `sam3.model.sam3_base_predictor.Sam3BasePredictor`) for
`cv_worker.run_video_session` to run end-to-end locally: it "detects" a
moving blob instead of running real inference, so the /start-run ->
/video-stream -> mask-overlay pipeline can be exercised without a GPU.

Never used when CUDA is available -- see app.py's load_sam_predictor.
"""

import time
import uuid

import cv2
import numpy as np


class MockSam3Predictor:
    """Fakes the subset of the session API that cv_worker.run_video_session uses."""

    def __init__(self):
        self._sessions: dict[str, dict] = {}

    def handle_request(self, request):
        request_type = request["type"]

        if request_type == "start_session":
            resource_path = request["resource_path"]
            capture = cv2.VideoCapture(resource_path)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 300
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            capture.release()

            session_id = str(uuid.uuid4())
            self._sessions[session_id] = {
                "frame_count": frame_count,
                "height": height,
                "width": width,
                "cancelled": False,
            }
            return {"session_id": session_id}

        if request_type == "add_prompt":
            return {"frame_index": request["frame_index"], "outputs": {}}

        if request_type == "cancel_propagation":
            self._sessions[request["session_id"]]["cancelled"] = True
            return {"is_success": True}

        if request_type == "close_session":
            self._sessions.pop(request["session_id"], None)
            return {"is_success": True}

        raise RuntimeError(f"MockSam3Predictor: unsupported request type {request_type!r}")

    def handle_stream_request(self, request):
        if request["type"] != "propagate_in_video":
            raise RuntimeError(
                f"MockSam3Predictor: unsupported stream request type {request['type']!r}"
            )

        session = self._sessions[request["session_id"]]
        height, width = session["height"], session["width"]
        radius = min(height, width) // 8
        yy, xx = np.ogrid[:height, :width]

        for frame_idx in range(session["frame_count"]):
            if session["cancelled"]:
                return

            # A blob that drifts around the frame, standing in for a tracked fish.
            center_x = width // 2 + int((width // 4) * np.sin(frame_idx / 15))
            center_y = height // 2 + int((height // 4) * np.cos(frame_idx / 20))
            mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2

            # Real inference has real latency; without this the mock would
            # spin through the whole clip in a fraction of a second.
            time.sleep(1 / 30)

            yield {"frame_index": frame_idx, "outputs": {"out_binary_masks": mask[None, ...]}}
