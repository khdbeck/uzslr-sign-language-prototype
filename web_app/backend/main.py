import sys
import pathlib
import numpy as np
import asyncio
import cv2
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# path resolution
# this file lives at: <repo>/web_app/backend/main.py
_BACKEND_DIR = pathlib.Path(__file__).parent.resolve()   # .../web_app/backend
_WEBAPP_DIR  = _BACKEND_DIR.parent                        # .../web_app
_REPO_ROOT   = _WEBAPP_DIR.parent                         # .../uzslr-isolated-dynamic

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from web_app.backend.config import (
    LLM_ENABLED, LLM_MODELS, LLM_DEFAULT_MODEL,
    LLM_HAND_ABSENT_FRAMES, DEFAULT_SYSTEM_PROMPT, ADMIN_PASSWORD,
)
from web_app.backend.slovo import SlovoRecognizer, preprocess_frame

# absolute paths to assets
_SIGNS_DIR    = _REPO_ROOT  / "show-50-signs" / "signs"
_FRONTEND_DIR = _WEBAPP_DIR / "frontend"


# global singletons 
model      = None
_inference_lock = asyncio.Lock()
_logger = logging.getLogger(__name__)

# Active system prompt — starts as default, admin can change at runtime.
# Resets to DEFAULT_SYSTEM_PROMPT on every server restart (by design).
_active_prompt: str = None   # set properly in lifespan after config is loaded

# Simple in-memory admin tokens (token → True). Cleared on restart.
import secrets as _secrets
_admin_tokens: set = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    # sign recognition model 
    model = await asyncio.to_thread(SlovoRecognizer)
    print(f"[startup] Slovo loaded: {model.name}, {model.no_event_id} signs + no_event ({model.provider})")

    # ollama (only in LLM image)
    if LLM_ENABLED:
        from web_app.backend.llm_client import start_ollama
        start_ollama()

    # initialise active prompt from config default
    global _active_prompt
    _active_prompt = DEFAULT_SYSTEM_PROMPT

    print("[startup] ready")
    yield

    # shutdown
    if LLM_ENABLED:
        from web_app.backend.llm_client import stop_ollama
        stop_ollama()
    print("[shutdown] done")


app = FastAPI(lifespan=lifespan)


# REST: frontend config 
@app.get("/api/config")
async def get_config():
    """
    Called by the frontend on load to know whether LLM features are available.
    Keeps all feature-flag logic server-side.
    """
    return JSONResponse({
        "llmEnabled":          LLM_ENABLED,
        "llmModels":           LLM_MODELS,
        "llmDefaultModel":     LLM_DEFAULT_MODEL,
        "llmHandAbsentFrames": LLM_HAND_ABSENT_FRAMES,
        "defaultSystemPrompt": DEFAULT_SYSTEM_PROMPT,
        "activePrompt":        _active_prompt,
        "recognitionModel": "slovo",
        "recognitionVariant": model.name,
        "recognitionLanguage": "rsl",
        "classCount": len(model.labels),
        "signCount": model.no_event_id,
        "bufferSize": model.frame_count,
        "sampleFps": model.sample_fps,
        "inputType": "jpeg",
        "executionProvider": model.provider,
        "confidenceThreshold": 0.1,
    })


@app.get("/api/recognition-signs")
async def recognition_signs():
    # Optional example clip + poster per sign: frontend/slovo-images/<variant>/<index>.{mp4,jpg}
    folder = _FRONTEND_DIR / "slovo-images" / model.name
    examples = [
        {"video": f"slovo-images/{model.name}/{i}.mp4", "poster": f"slovo-images/{model.name}/{i}.jpg"}
        if (folder / f"{i}.mp4").is_file() else None
        for i in range(model.no_event_id)
    ]
    return {"model": "slovo", "signs": model.labels[:-1], "examples": examples,
            "noEventClassId": model.no_event_id}


@app.post("/api/infer-video")
async def infer_video(file: UploadFile):
    """Test the same recognizer on a local clip; do not retain uploaded video."""
    if file.filename is None or pathlib.Path(file.filename).suffix.lower() not in {".mp4", ".webm", ".mov", ".avi"}:
        raise HTTPException(400, "Upload MP4, WebM, MOV or AVI")
    with _tempfile.TemporaryDirectory() as directory:
        target = pathlib.Path(directory) / ("clip" + pathlib.Path(file.filename).suffix.lower())
        size = 0
        try:
            with target.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 50 * 1024 * 1024:
                        raise HTTPException(413, "Video must be smaller than 50 MB")
                    output.write(chunk)
            async with _inference_lock:
                result = await asyncio.to_thread(model.predict_video, target)
            return {**result, "model": "slovo"}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            await file.close()


# REST: admin endpoints 
class AdminVerifyRequest(BaseModel):
    password: str

@app.post("/api/admin/verify")
async def admin_verify(req: AdminVerifyRequest):
    if not ADMIN_PASSWORD:
        raise HTTPException(status_code=503, detail="Admin access is disabled")
    if not _secrets.compare_digest(req.password.encode("utf-8"), ADMIN_PASSWORD.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Incorrect password")
    token = _secrets.token_hex(32)
    _admin_tokens.add(token)
    return JSONResponse({"token": token})


class AdminPromptRequest(BaseModel):
    prompt: str
    token:  str

@app.post("/api/admin/prompt")
async def admin_set_prompt(req: AdminPromptRequest):
    if req.token not in _admin_tokens:
        raise HTTPException(status_code=403, detail="Not authorised")
    global _active_prompt
    _active_prompt = req.prompt
    return JSONResponse({"ok": True})


@app.get("/api/active-prompt")
async def get_active_prompt():
    return JSONResponse({"prompt": _active_prompt})


# REST: sentence formation 
class SentenceRequest(BaseModel):
    signs:        list[str]
    systemPrompt: str
    model:        str = LLM_DEFAULT_MODEL


@app.post("/api/form-sentence")
async def form_sentence_endpoint(req: SentenceRequest):
    if not LLM_ENABLED:
        raise HTTPException(status_code=503, detail="LLM not available in this image")

    if not req.signs:
        raise HTTPException(status_code=400, detail="No signs provided")

    from web_app.backend.llm_client import form_sentence
    # admin sends their edited prompt; regular users send empty string -> use server active prompt
    effective_prompt = req.systemPrompt if req.systemPrompt.strip() else _active_prompt
    try:
        sentence = await form_sentence(
            signs=req.signs,
            system_prompt=effective_prompt,
            model=req.model,
        )
        return JSONResponse({"sentence": sentence})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket: real-time sign inference 
@app.websocket("/ws/infer")
async def websocket_infer(ws: WebSocket):
    await ws.accept()

    frame_buffer = []

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("text"):
                if message["text"] == "reset":
                    frame_buffer.clear()
                    await ws.send_json({"bufferSize": 0, "bufferFull": False})
                else:
                    await ws.send_json({"error": "Slovo expects binary JPEG frames"})
                continue
            payload = message.get("bytes")
            if not payload or len(payload) > 512 * 1024:
                await ws.send_json({"error": "Invalid frame size"})
                continue
            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None or max(frame.shape[:2]) > 2048:
                await ws.send_json({"error": "Invalid JPEG frame"})
                continue
            frame_buffer.append(preprocess_frame(frame))
            response = {"bufferSize": len(frame_buffer), "bufferFull": False,
                        "prediction": None, "model": "slovo"}
            if len(frame_buffer) == model.frame_count:
                async with _inference_lock:
                    result = await asyncio.to_thread(model.predict, list(frame_buffer))
                response.update(result)
                response["bufferFull"] = True
                # Start a fresh clip after inference so slow CPUs cannot mix
                # frames captured before and after a several-second pause.
                frame_buffer.clear()
            await ws.send_json(response)

    except WebSocketDisconnect:
        pass
    except Exception:
        _logger.exception("Slovo WebSocket inference failed")
        await ws.close(code=1011, reason="Recognition failed; reconnect to retry")


# REST: sign video production
import subprocess as _subprocess, tempfile as _tempfile, shutil as _shutil
from fastapi.responses import FileResponse

_SIGN_VIDEOS_DIR = _WEBAPP_DIR / "sign_videos"


@app.get("/api/signs-list")
async def get_signs_list():
    """Return list of all available pre-rendered sign names."""
    if not _SIGN_VIDEOS_DIR.exists():
        return JSONResponse({"signs": []})
    signs = sorted(
        p.stem for p in _SIGN_VIDEOS_DIR.glob("*.mp4")
        if p.stem != "rest"
    )
    return JSONResponse({"signs": signs})


class ProduceVideoRequest(BaseModel):
    signs: list[str]


@app.post("/api/produce-sign-video")
async def produce_sign_video(req: ProduceVideoRequest):
    """Stitch requested signs with rest.mp4 between them and return MP4."""
    if not req.signs:
        raise HTTPException(status_code=400, detail="No signs provided")

    missing = [s for s in req.signs if not (_SIGN_VIDEOS_DIR / f"{s}.mp4").exists()]
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown signs: {missing}")

    # reject adjacent duplicates (same sign twice in a row)
    for i in range(len(req.signs) - 1):
        if req.signs[i] == req.signs[i + 1]:
            raise HTTPException(status_code=400, detail=f"Adjacent duplicate: '{req.signs[i]}' appears twice in a row")

    rest = _SIGN_VIDEOS_DIR / "rest.mp4"
    clips = []
    for i, sign in enumerate(req.signs):
        clips.append(str(_SIGN_VIDEOS_DIR / f"{sign}.mp4"))
        if i < len(req.signs) - 1 and rest.exists():
            clips.append(str(rest))

    tmp_dir = pathlib.Path(_tempfile.mkdtemp())
    try:
        list_file = tmp_dir / "concat.txt"
        with open(list_file, "w") as f:
            for clip in clips:
                # ffmpeg concat: single-quote the path and escape internal single quotes as '\''
                safe = clip.replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        out_file = tmp_dir / "output.mp4"
        result = _subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(out_file)
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"ffmpeg error: {result.stderr[-300:]}")

        return FileResponse(path=str(out_file), media_type="video/mp4", filename="sign_sequence.mp4")
    except HTTPException:
        _shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception as e:
        _shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


# static files
app.mount("/signs",       StaticFiles(directory=str(_SIGNS_DIR)),       name="signs")
app.mount("/sign-videos", StaticFiles(directory=str(_SIGN_VIDEOS_DIR)), name="sign-videos")
app.mount("/",            StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
