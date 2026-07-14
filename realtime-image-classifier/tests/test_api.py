"""
Basic smoke tests for the API.
Run with: pytest -v
(Requires internet access on first run so torchvision can download weights.)
"""
import io
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _dummy_image_bytes():
    img = Image.new("RGB", (224, 224), color=(120, 180, 220))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_classify_returns_predictions():
    buf = _dummy_image_bytes()
    resp = client.post(
        "/classify",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 5
    assert "inference_ms" in body


def test_classify_rejects_invalid_file():
    bad_file = io.BytesIO(b"not an image")
    resp = client.post(
        "/classify",
        files={"file": ("test.txt", bad_file, "text/plain")},
    )
    assert resp.status_code == 400
