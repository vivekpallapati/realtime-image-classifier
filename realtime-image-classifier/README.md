# Real-Time Image Classification API

A FastAPI service that classifies images using a pretrained ResNet18 (ImageNet, 1000 classes).
Supports both single-image REST classification and a live WebSocket stream for
real-time classification of a webcam feed.

## Features

- `POST /classify` — upload one image, get top-5 predictions
- `WS /ws/classify` — persistent connection for real-time frame-by-frame classification
- `GET /health` — health check
- Bundled browser client (`client/index.html`) that streams your webcam to the WebSocket endpoint and shows live predictions with confidence bars
- Dockerfile for containerized deployment
- Pytest test suite

## Project structure

```
realtime-image-classifier/
├── app/
│   ├── main.py       # FastAPI app: REST + WebSocket endpoints
│   ├── model.py      # Model loading + inference (ResNet18)
│   └── __init__.py
├── client/
│   └── index.html    # Webcam test client
├── tests/
│   └── test_api.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first request downloads pretrained ResNet18 weights (~45MB) from
torchvision, so you'll need internet access the first time you run it.
After that they're cached locally.

## Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs: http://localhost:8000/docs

## Try it

### Single image (REST)

```bash
curl -X POST "http://localhost:8000/classify" \
  -F "file=@/path/to/your/image.jpg"
```

Response:
```json
{
  "predictions": [
    {"label": "golden_retriever", "confidence": 0.9123},
    {"label": "Labrador_retriever", "confidence": 0.0421},
    ...
  ],
  "inference_ms": 34.2
}
```

### Real-time webcam (WebSocket)

1. Start the server (above).
2. Open `client/index.html` directly in your browser (e.g. double-click it, or `python -m http.server` from the `client/` folder).
3. Click **Start**, allow camera access. Live predictions stream in below the video feed.

The client sends the WebSocket URL as `ws://localhost:8000/ws/classify` by default — change it in the input box if your server runs elsewhere.

## Run tests

```bash
pytest -v
```

## Deploy with Docker

```bash
docker build -t realtime-classifier .
docker run -p 8000:8000 realtime-classifier
```

## Swapping in a faster/different model

Everything model-specific lives in `app/model.py`. To use a lighter model for
faster CPU inference (e.g. on a Raspberry Pi or a laptop without a GPU), swap:

```python
_model = models.resnet18(weights=_weights)
```

for something like:

```python
_weights = models.MobileNet_V3_Small_Weights.DEFAULT
_model = models.mobilenet_v3_small(weights=_weights)
```

No other code changes needed — labels come from `_weights.meta["categories"]`
automatically.

## Notes on "real-time" performance

- On CPU, ResNet18 typically runs in ~30–80ms per frame — good enough for a
  responsive live feed (10–20 fps depending on hardware).
- The WebSocket client uses simple backpressure (it only sends a new frame
  once it gets a reply for the last one) so slow hardware won't queue up a
  backlog of stale frames.
- For GPU inference, install a CUDA-enabled build of `torch` and the app will
  automatically use it (see `DEVICE` in `app/model.py`).
