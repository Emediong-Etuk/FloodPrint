from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from PIL import Image

from ai.flood_vision import HEAD_CLASSES, FloodVision, VisionUnavailableError
from config import MODELS_DIR
from services.image_service import ImageValidationError, resolve_upload, save_image, validate_image

WEIGHTS = MODELS_DIR / "openclip" / "vit_b_32-quickgelu-laion400m_e32-46683a32.pt"


def jpeg_bytes(size=(320, 240), color=(30, 80, 140), fmt="JPEG"):
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, fmt)
    return b.getvalue()


def test_valid_image_input():
    v = validate_image(jpeg_bytes(), "photo.jpg")
    assert v.format == "JPEG" and v.width == 320


@pytest.mark.parametrize("data,name", [(b"", "a.jpg"), (b"hello", "a.txt"), (jpeg_bytes(), "a.gif"),
                                       (jpeg_bytes(fmt="PNG"), "a.jpg"), (jpeg_bytes((20, 20)), "a.jpg")])
def test_invalid_image_input(data, name):
    with pytest.raises(ImageValidationError):
        validate_image(data, name)


def test_corrupt_image():
    data = jpeg_bytes()[:200]
    with pytest.raises(ImageValidationError):
        validate_image(data, "broken.jpg")


def test_oversize_image():
    with pytest.raises(ImageValidationError):
        validate_image(jpeg_bytes(), "a.jpg", max_mb=0.0001)


def test_safe_storage(tmp_path):
    name = save_image(Image.new("RGB", (100, 100)), tmp_path)
    assert resolve_upload(tmp_path, name) is not None
    assert resolve_upload(tmp_path, "../../etc/passwd") is None


def test_model_unavailable_handling(tmp_path):
    v = FloodVision(tmp_path, allow_download=False)
    with pytest.raises(VisionUnavailableError):
        v.load()
    v2 = FloodVision(MODELS_DIR, weights_path=tmp_path / "missing.pt", allow_download=False)
    with pytest.raises(VisionUnavailableError):
        v2.load()


@pytest.mark.skipif(not WEIGHTS.exists(), reason="OpenCLIP weights not downloaded")
def test_real_inference_confidence_output():
    v = FloodVision(MODELS_DIR, allow_download=False).load()
    r = v.analyze(Image.new("RGB", (300, 300), (120, 120, 120)))
    assert 0.0 <= r.confidence <= 1.0
    assert set(r.class_probabilities) == set(HEAD_CLASSES)
    assert abs(sum(r.class_probabilities.values()) - 1) < 1e-5
    assert r.analysis_status == "completed" and len(r.heatmap) == 3
    with pytest.raises(TypeError):
        v.analyze(b"not an image")
