"""
backend/main.py  — Video Only Version
───────────────────────────────────────
Video model: checkpoints_video_v2/best_video_model_v2.pth (V2-aug, AUC 0.998)

Run:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import logging, sys, tempfile, time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.video_predictor_v2 import VideoPredictorV2

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("deepfake_api")

_VIDEO_CONFIG     = Path("C:/imagevideo/training_video/video_config_v2.yaml")
_VIDEO_CHECKPOINT = Path("C:/imagevideo/checkpoints_video_v2/best_video_model_v2.pth")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading V2-aug video model...")
    try:
        vid = VideoPredictorV2(checkpoint_path=_VIDEO_CHECKPOINT, config_path=_VIDEO_CONFIG)
        app.state.video_predictor = vid
        app.state.model_loaded    = True
        app.state.device          = str(vid.device)
        logger.info("V2-aug video model loaded on %s", vid.device)
    except Exception as e:
        logger.error("Video model failed: %s", e)
        app.state.video_predictor = None
        app.state.model_loaded    = False

    print("Deepfake Video Detector API running!")
    print("Docs: http://localhost:8000/docs")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Deepfake Video Detector API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class VideoPredictionResponse(BaseModel):
    prediction:      str
    confidence:      float
    is_fake:         bool
    frames_analyzed: int
    raw_score:       float
    filename:        str
    processing_time: float


_VIDEO_EXTS    = {".mp4", ".mov", ".avi"}
_MAX_VID_BYTES = 100 * 1024 * 1024


async def _validate(file, exts, max_bytes):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in exts:
        raise HTTPException(400, f"Invalid file '{ext}'. Allowed: {sorted(exts)}")
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(413, f"Too large ({len(data)/1e6:.1f}MB). Max {max_bytes//1048576}MB.")
    return data


@app.get("/")
async def root():
    return {"message": "Deepfake Video Detector API v2", "status": "running"}


@app.get("/health")
async def health(request: Request):
    return {
        "status":      "ok" if request.app.state.model_loaded else "error",
        "model":       "V2-aug Pure Temporal (AUC 0.998)",
        "device":      getattr(request.app.state, "device", "unknown"),
    }


@app.post("/predict/video", response_model=VideoPredictionResponse)
async def predict_video(request: Request, file: UploadFile = File(...)):
    if not request.app.state.model_loaded:
        raise HTTPException(500, "Video model not loaded.")

    data   = await _validate(file, _VIDEO_EXTS, _MAX_VID_BYTES)
    suffix = Path(file.filename or "vid.mp4").suffix.lower()
    tmp    = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data); tmp = Path(f.name)
        t0 = time.perf_counter()
        r  = request.app.state.video_predictor.predict(tmp)
        elapsed = time.perf_counter() - t0

        logger.info("Video: %s (%.1f%%, %.3fs) [%s]",
                    r["prediction"], r["confidence"], elapsed, file.filename)

        return VideoPredictionResponse(
            prediction      = r["prediction"],
            confidence      = r["confidence"],
            is_fake         = r["is_fake"],
            frames_analyzed = r["frames_analyzed"],
            raw_score       = r["raw_score"],
            filename        = file.filename or "unknown",
            processing_time = round(elapsed, 4),
        )
    except HTTPException: raise
    except Exception as e:
        logger.exception("Video prediction error: %s", e)
        raise HTTPException(500, f"Model error: {e}")
    finally:
        if tmp and tmp.exists(): tmp.unlink()