"""
Core FastAPI server for the SAM 3.1 live video worker.

Design goal: the SAM 3.1 predictor is loaded onto the GPU exactly once, at
process startup. Everything that changes during development -- the actual
per-frame processing logic in cv_worker.py -- is imported as a module
reference and can be swapped out at runtime via importlib.reload(), so you
never pay the cost of re-loading multi-GB checkpoints into VRAM just to
tweak a prompt or an overlay color.

Run with: uv run uvicorn app.app:app --host 0.0.0.0 --port 8000
"""

import asyncio
import importlib
import logging
import threading
import time
from contextlib import asynccontextmanager

import cv2
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app import cv_worker

logger = logging.getLogger("uvicorn.error")

CHECKPOINT_PATH = "/workspace/checkpoints/sam3.1_hiera_large.pt"
MODEL_CONFIG = "sam3.1_hiera_l.yaml"
VIDEO_SOURCE = 0  # local capture device index, or an RTSP/HTTP URL string
JPEG_QUALITY = 80


class AppState:
    """Holds long-lived, GPU-resident objects that must never be reloaded."""

    def __init__(self):
        self.predictor = None
        self.device: str | None = None
        self.capture: cv2.VideoCapture | None = None
        self.is_running: bool = False
        self.lock = threading.Lock()


state = AppState()


def load_sam_predictor():
    """
    Load SAM 3.1 checkpoints onto CUDA. This is intentionally the only place
    in the whole app that touches model weights -- it runs once at startup
    and is never re-invoked by the hot-reload endpoint.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA not available -- falling back to CPU (inference will be slow).")

    # NOTE: import is deferred so a missing/optional SAM package doesn't
    # block the rest of the app (e.g. during local frontend-only dev).
    from sam3 import build_sam3_predictor  # provided by the SAM 3.1 package

    predictor = build_sam3_predictor(
        config_path=MODEL_CONFIG,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )

    logger.info("SAM 3.1 predictor loaded onto %s", device)
    return predictor, device


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.predictor, state.device = load_sam_predictor()
    state.capture = cv2.VideoCapture(VIDEO_SOURCE)
    yield
    if state.capture is not None:
        state.capture.release()


app = FastAPI(title="SAM 3.1 Live Inference Worker", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/dev-reload")
async def dev_reload():
    """
    Hot-reload the frame-processing module (cv_worker.py) without touching
    the GPU-resident predictor. Use this while iterating on prompt logic,
    mask coloring, or overlay drawing during local development.
    """
    try:
        importlib.reload(cv_worker)
    except Exception as exc:  # surface the reload error instead of crashing the server
        logger.exception("Failed to reload cv_worker")
        return JSONResponse(status_code=500, content={"reloaded": False, "error": str(exc)})

    return {"reloaded": True, "module": cv_worker.__name__}


@app.post("/start-run")
async def start_run():
    with state.lock:
        state.is_running = True
    return {"running": True}


@app.post("/stop-run")
async def stop_run():
    with state.lock:
        state.is_running = False
    return {"running": False}


@app.get("/health")
async def health():
    return {
        "device": state.device,
        "cuda_available": torch.cuda.is_available(),
        "running": state.is_running,
    }


def _generate_mjpeg_frames():
    """Blocking generator run in a worker thread; yields multipart JPEG chunks."""
    while True:
        if not state.is_running or state.capture is None:
            time.sleep(0.1)
            continue

        success, frame = state.capture.read()
        if not success:
            time.sleep(0.1)
            continue

        try:
            annotated = cv_worker.process_sam_frame(frame, state.predictor)
        except Exception:
            logger.exception("Frame processing failed; passing raw frame through")
            annotated = frame

        ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


async def _mjpeg_async_generator():
    """Bridges the blocking OpenCV/GPU generator into an async stream without
    blocking the event loop -- each frame is produced on a thread executor."""
    loop = asyncio.get_event_loop()
    gen = _generate_mjpeg_frames()

    while True:
        chunk = await loop.run_in_executor(None, next, gen, None)
        if chunk is None:
            break
        yield chunk


@app.get("/video-stream")
async def video_stream():
    return StreamingResponse(
        _mjpeg_async_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
