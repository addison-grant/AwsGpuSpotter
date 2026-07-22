"""
Core FastAPI server for the SAM 3.1 live video worker.

Design goal: the SAM 3.1 predictor is loaded onto the GPU exactly once, at
process startup. Everything that changes during development -- the video
session and mask overlay logic in cv_worker.py -- is imported as a module
reference and can be swapped out at runtime via importlib.reload(), so you
never pay the cost of re-loading multi-GB checkpoints into VRAM just to
tweak a prompt or an overlay color.

Run with: uv run uvicorn app.app:app --host 0.0.0.0 --port 8000
"""

import asyncio
import importlib
import logging
import os
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

CHECKPOINT_PATH = "/workspace/checkpoints/sam3.1_hiera_tiny.pt"
VIDEO_SOURCE = "/Users/addisongrant/Workspace/FishAIWorkspace/_VIDEOS_COMMON/initial_test/GOPR7094_trim_720_179s--26sec_grayworld.mov"  # local capture device index, or an RTSP/HTTP URL string
JPEG_QUALITY = 80


class AppState:
    """Holds long-lived, GPU-resident objects that must never be reloaded."""

    def __init__(self):
        self.predictor = None
        self.device: str | None = None
        self.is_running: bool = False
        self.worker_thread: threading.Thread | None = None
        self.stop_event: threading.Event = threading.Event()
        self.latest_frame = None  # last annotated BGR frame from the active session
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
        # The vendored sam3 package unconditionally calls .cuda() while
        # building the model, so there is no way to load real weights here.
        # Use a mock predictor instead, so the /start-run -> /video-stream
        # pipeline is still exercisable during local, GPU-less dev.
        logger.warning("CUDA not available -- using MockSam3Predictor (no real inference).")
        from app.mock_predictor import MockSam3Predictor

        return MockSam3Predictor(), device

    # NOTE: import is deferred so a missing/optional SAM package doesn't
    # block the rest of the app (e.g. during local frontend-only dev).
    from sam3 import build_sam3_predictor  # provided by the SAM 3.1 package

    # Falls back to HuggingFace auto-download when the GPU-instance
    # checkpoint path isn't present (e.g. local dev on a laptop).
    checkpoint_path = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None
    predictor = build_sam3_predictor(
        checkpoint_path=checkpoint_path,
        use_fa3=(device == "cuda"),
    )

    logger.info("SAM 3.1 predictor loaded onto %s", device)
    return predictor, device


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.predictor, state.device = load_sam_predictor()
    yield
    state.stop_event.set()
    if state.worker_thread is not None:
        state.worker_thread.join(timeout=5)


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
    """
    Kick off inference: always starts a brand-new SAM 3.1 session from frame
    0 of VIDEO_SOURCE, superseding any run already in progress.
    """
    if state.predictor is None:
        return JSONResponse(
            status_code=503,
            content={"error": "SAM 3.1 predictor is not loaded (no CUDA device available)."},
        )

    with state.lock:
        # Tell any in-flight session to stop; it will notice on its next
        # frame and clean itself up (cancel_propagation + close_session).
        state.stop_event.set()

        stop_event = threading.Event()
        state.stop_event = stop_event
        state.latest_frame = None
        state.is_running = True

        def _on_frame(frame):
            with state.lock:
                state.latest_frame = frame

        def _run():
            try:
                cv_worker.run_video_session(state.predictor, VIDEO_SOURCE, stop_event, _on_frame)
            except Exception:
                logger.exception("Video session failed")
            finally:
                with state.lock:
                    if state.stop_event is stop_event:
                        state.is_running = False

        thread = threading.Thread(target=_run, daemon=True)
        state.worker_thread = thread
        thread.start()

    return {"running": True}


@app.post("/stop-run")
async def stop_run():
    with state.lock:
        state.is_running = False
        state.stop_event.set()
    return {"running": False}


@app.get("/health")
async def health():
    return {
        "device": state.device,
        "cuda_available": torch.cuda.is_available(),
        "running": state.is_running,
    }


def _generate_mjpeg_frames():
    """Blocking generator run in a worker thread; yields multipart JPEG chunks.

    Frames come from state.latest_frame, which the active session's worker
    thread (see /start-run) updates as SAM 3.1 propagates through the video.
    """
    while True:
        with state.lock:
            frame = state.latest_frame

        if frame is None:
            time.sleep(0.1)
            continue

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        time.sleep(1 / 30)


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
