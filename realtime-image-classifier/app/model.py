"""
model.py
--------
Loads a pretrained ResNet18 (ImageNet) model and exposes a single
`classify_image(image: PIL.Image) -> list[dict]` function used by both
the REST endpoint and the WebSocket real-time stream.

Swapping models: replace `_load_model()` with any torchvision model
(e.g. mobilenet_v3_small for faster CPU inference) — the rest of the
pipeline (transform, softmax, top-k) stays the same.
"""

import io
import time
from typing import List, Dict

import torch
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_labels: List[str] = []
_weights = models.ResNet18_Weights.DEFAULT

_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    """Loads the model once and caches it (call at app startup)."""
    global _model, _labels
    if _model is None:
        # weights=DEFAULT downloads pretrained ImageNet weights on first run
        # (requires internet the first time; cached under ~/.cache/torch after that)
        _model = models.resnet18(weights=_weights)
        _model.eval()
        _model.to(DEVICE)
        # categories come bundled with the weights object -- always matches
        # the exact label ordering the model was trained with
        _labels = _weights.meta["categories"]
    return _model


def classify_image(image: Image.Image, top_k: int = 5) -> Dict:
    """Runs inference on a single PIL image, returns top-k predictions + timing."""
    model = load_model()

    start = time.time()
    tensor = _preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    top_probs, top_idxs = torch.topk(probs, top_k)
    elapsed_ms = round((time.time() - start) * 1000, 2)

    predictions = [
        {"label": _labels[idx.item()], "confidence": round(prob.item(), 4)}
        for prob, idx in zip(top_probs, top_idxs)
    ]

    return {"predictions": predictions, "inference_ms": elapsed_ms}


def bytes_to_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))
