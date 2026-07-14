"""
main.py
-------
FastAPI service exposing:

  GET  /health              -> liveness check
  POST /classify             -> classify a single uploaded image
  WS   /ws/classify          -> real-time classification stream (send base64
                                 JPEG frames, e.g. from a webcam, receive
                                 predictions back as fast as the model runs)

Run:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import binascii
import io
import time

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from app.model import classify_image, load_model, bytes_to_image

app = FastAPI(
    title="Real-Time Image Classification API",
    description="Classify images via REST or a live WebSocket stream using a pretrained ResNet18 model.",
    version="1.0.0",
)

# Loosen CORS so the bundled test client (or any frontend) can call this
# from a different origin/port during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    # Warm the model up once at startup instead of on the first request.
    load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify")
async def classify(file: UploadFile = File(...), top_k: int = 5):
    """Classify a single uploaded image (multipart/form-data)."""
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        return JSONResponse(status_code=400, content={"error": "File is not a valid image"})

    result = classify_image(image, top_k=top_k)
    return result


@app.websocket("/ws/classify")
async def websocket_classify(websocket: WebSocket):
    """
    Real-time classification stream.

    Client protocol:
      - Client sends a text message per frame: either raw base64 JPEG/PNG
        data, or a data URL like "data:image/jpeg;base64,....".
      - Server replies with a JSON object per frame:
          {"predictions": [...], "inference_ms": float, "frame_ms": float}
        or {"error": "..."} if the frame couldn't be decoded/classified.

    This keeps the connection open and processes frames as they arrive,
    which is what makes it suitable for a live webcam feed rather than
    one HTTP round-trip per frame.
    """
    await websocket.accept()
    try:
        while True:
            frame_start = time.time()
            data = await websocket.receive_text()

            if "," in data and data.strip().startswith("data:"):
                data = data.split(",", 1)[1]

            try:
                raw = base64.b64decode(data)
                image = bytes_to_image(raw)
                result = classify_image(image)
                result["frame_ms"] = round((time.time() - frame_start) * 1000, 2)
                await websocket.send_json(result)
            except (binascii.Error, UnidentifiedImageError, OSError) as e:
                await websocket.send_json({"error": f"Could not decode frame: {e}"})

    except WebSocketDisconnect:
        pass
