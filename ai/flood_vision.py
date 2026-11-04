"""Flood Vision: photo analysis for community flood reports.

Pipeline
--------
1. Image encoder: OpenCLIP ViT-B/32 (QuickGELU) pretrained on LAION-400M
   (weights from the OpenCLIP GitHub release, ~605 MB, cached on first use).
2. Classifier head: multinomial logistic regression trained on CLIP embeddings
   of human-verified Open Images photos (flooding / waterlogging / no flooding).
   See ``scripts/train_vision_head.py`` and ``data/models/vision_metrics.json``.
3. Severity split for flooding photos (moderate vs severe): zero-shot CLIP
   text-prompt comparison. This split has *not* been validated on labelled data.
4. Coarse water map: the head is applied to overlapping crops to show where in
   the frame water-like content scores highest. This is not pixel segmentation.

The model cannot measure depth or tell whether a road is passable, and it never
marks a report as verified.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

ARCH = "ViT-B-32-quickgelu"
PRETRAINED_TAG = "laion400m_e32"
WEIGHTS_URL = (
    "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
    "vit_b_32-quickgelu-laion400m_e32-46683a32.pt"
)
WEIGHTS_SHA256_PREFIX = "46683a32"
MODEL_NAME = f"OpenCLIP {ARCH} ({PRETRAINED_TAG}) + FLOODPRINT linear head"

HEAD_CLASSES = ("no_flooding", "waterlogging", "flooding")

CLASS_LABELS = {
    "no_visible_flooding": "No visible flooding",
    "waterlogging_minor": "Waterlogging / minor surface water",
    "moderate_flooding": "Moderate flooding",
    "severe_flooding": "Severe visible flooding",
    "uncertain": "Uncertain / unable to classify",
}
CLASS_TO_SEVERITY = {
    "no_visible_flooding": "none",
    "waterlogging_minor": "minor",
    "moderate_flooding": "moderate",
    "severe_flooding": "severe",
    "uncertain": None,
}

SEVERITY_PROMPTS = {
    "moderate": [
        "a photo of a street with shallow flood water",
        "a road partly covered by brown floodwater",
        "ankle-deep flood water on a city street",
        "cars driving slowly through shallow flood water",
    ],
    "severe": [
        "a photo of a street submerged under deep flood water",
        "cars submerged in deep floodwater",
        "houses flooded up to the windows",
        "people wading waist-deep through a flooded street",
    ],
}

ZERO_SHOT_PROMPTS = {
    "no_flooding": [
        "a photo of a dry street",
        "a photo of a road with no water",
        "a photo of a city neighbourhood on a dry day",
        "a photo of a river or lake under normal conditions",
    ],
    "waterlogging": [
        "a photo of puddles on a road",
        "a photo of a wet street with small puddles after rain",
        "a photo of a waterlogged road with shallow standing water",
    ],
    "flooding": [
        "a photo of a flooded street",
        "a photo of a flood",
        "a photo of floodwater covering roads and houses",
    ],
}


class VisionUnavailableError(RuntimeError):
    """Raised when the model weights or head cannot be loaded."""


@dataclass
class VisionResult:
    classification: str
    label: str
    confidence: float
    water_detected: bool
    severity: str | None
    class_probabilities: dict[str, float]
    severity_probabilities: dict[str, float] | None
    model_name: str
    head_version: str
    analysis_status: str = "completed"
    low_confidence: bool = False
    notes: list[str] = field(default_factory=list)
    heatmap: list[list[float]] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _download(url: str, dest: Path, sha_prefix: str | None, timeout: float = 60) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                h.update(chunk)
    if sha_prefix and not h.hexdigest().startswith(sha_prefix):
        tmp.unlink(missing_ok=True)
        raise VisionUnavailableError("Downloaded weights failed checksum verification")
    tmp.replace(dest)


class FloodVision:
    """Lazy-loading flood photo analyser. Thread-safe load, reusable instance."""

    def __init__(
        self,
        models_dir: Path | str,
        head_path: Path | str | None = None,
        weights_path: Path | str | None = None,
        confidence_threshold: float = 0.55,
        allow_download: bool = True,
    ):
        self.models_dir = Path(models_dir)
        self.head_path = Path(head_path) if head_path else self.models_dir / "vision_head.joblib"
        env_weights = os.environ.get("FLOODPRINT_VISION_WEIGHTS")
        self.weights_path = Path(weights_path or env_weights or self.models_dir / "openclip" / Path(WEIGHTS_URL).name)
        self.confidence_threshold = confidence_threshold
        self.allow_download = allow_download
        self._lock = threading.Lock()
        self._model = None
        self._preprocess = None
        self._head = None
        self._head_meta: dict = {}
        self._text: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------ loading
    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._head is not None

    def load(self) -> "FloodVision":
        with self._lock:
            if self.is_loaded:
                return self
            self._load_head()
            self._load_backbone()
            self._prepare_text()
        return self

    def _load_head(self) -> None:
        import joblib

        if not self.head_path.exists():
            raise VisionUnavailableError(
                f"Vision head not found at {self.head_path}. Run `python scripts/train_vision_head.py`."
            )
        bundle = joblib.load(self.head_path)
        self._head = bundle["classifier"]
        self._head_meta = bundle.get("metadata", {})
        classes = tuple(bundle.get("classes", HEAD_CLASSES))
        if classes != HEAD_CLASSES:
            raise VisionUnavailableError(f"Unexpected head classes: {classes}")

    def _load_backbone(self) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise VisionUnavailableError(f"Vision dependencies missing: {exc}") from exc
        if not self.weights_path.exists():
            if not self.allow_download:
                raise VisionUnavailableError(f"Vision weights not found at {self.weights_path}")
            log.info("Downloading OpenCLIP weights to %s", self.weights_path)
            try:
                _download(WEIGHTS_URL, self.weights_path, WEIGHTS_SHA256_PREFIX)
            except Exception as exc:
                raise VisionUnavailableError(f"Could not download vision weights: {exc}") from exc
        try:
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
            model, _, preprocess = open_clip.create_model_and_transforms(ARCH, pretrained=str(self.weights_path))
            model.eval()
        except Exception as exc:
            raise VisionUnavailableError(f"Could not initialise vision model: {exc}") from exc
        self._model, self._preprocess = model, preprocess
        self._tokenizer = open_clip.get_tokenizer(ARCH)

    def _encode_text(self, prompts: list[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            t = self._model.encode_text(self._tokenizer(prompts))
            t = t / t.norm(dim=-1, keepdim=True)
            m = t.mean(dim=0)
            return (m / m.norm()).cpu().numpy()

    def _prepare_text(self) -> None:
        self._text = {f"sev_{k}": self._encode_text(v) for k, v in SEVERITY_PROMPTS.items()}
        for k, v in ZERO_SHOT_PROMPTS.items():
            self._text[f"zs_{k}"] = self._encode_text(v)

    # ---------------------------------------------------------- inference
    def embed(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        import torch

        if not self.is_loaded:
            self.load()
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = torch.stack([self._preprocess(im.convert("RGB")) for im in images[i:i + batch_size]])
                e = self._model.encode_image(batch)
                e = e / e.norm(dim=-1, keepdim=True)
                out.append(e.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def head_proba(self, emb: np.ndarray) -> np.ndarray:
        return self._head.predict_proba(emb)

    def zero_shot_proba(self, emb: np.ndarray) -> np.ndarray:
        T = np.stack([self._text[f"zs_{c}"] for c in HEAD_CLASSES])
        logits = 100.0 * emb @ T.T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def _severity(self, emb: np.ndarray) -> dict[str, float]:
        logits = 100.0 * np.array([emb @ self._text["sev_moderate"], emb @ self._text["sev_severe"]])
        logits -= logits.max()
        p = np.exp(logits) / np.exp(logits).sum()
        return {"moderate": float(p[0]), "severe": float(p[1])}

    def _crops(self, img: Image.Image, grid: int = 3) -> list[Image.Image]:
        w, h = img.size
        cw, ch = w / 2, h / 2
        crops = []
        for r in range(grid):
            for c in range(grid):
                x0 = c * (w - cw) / (grid - 1)
                y0 = r * (h - ch) / (grid - 1)
                crops.append(img.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))))
        return crops

    def analyze(self, image: Image.Image, with_heatmap: bool = True) -> VisionResult:
        if not isinstance(image, Image.Image):
            raise TypeError("analyze() expects a PIL image")
        if not self.is_loaded:
            self.load()
        img = image.convert("RGB")
        imgs = [img] + (self._crops(img) if with_heatmap else [])
        emb = self.embed(imgs)
        probs = self.head_proba(emb)
        p = probs[0]
        class_p = {c: float(v) for c, v in zip(HEAD_CLASSES, p)}
        top = HEAD_CLASSES[int(np.argmax(p))]
        conf = float(np.max(p))
        notes = []
        sev_p = None
        if conf < self.confidence_threshold:
            classification = "uncertain"
            notes.append(
                f"Top class confidence {conf:.0%} is below the {self.confidence_threshold:.0%} threshold."
            )
        elif top == "no_flooding":
            classification = "no_visible_flooding"
        elif top == "waterlogging":
            classification = "waterlogging_minor"
        else:
            sev_p = self._severity(emb[0])
            classification = "severe_flooding" if sev_p["severe"] >= 0.5 else "moderate_flooding"
            notes.append("Moderate vs severe is a zero-shot estimate and has not been validated.")
        water_p = class_p["waterlogging"] + class_p["flooding"]
        heat = None
        if with_heatmap:
            hp = probs[1:]
            heat = (hp[:, 1] + hp[:, 2]).reshape(3, 3).round(4).tolist()
        notes.append("A photo cannot show water depth or whether a road is passable.")
        return VisionResult(
            classification=classification,
            label=CLASS_LABELS[classification],
            confidence=conf,
            water_detected=bool(water_p >= 0.5),
            severity=CLASS_TO_SEVERITY[classification],
            class_probabilities=class_p,
            severity_probabilities=sev_p,
            model_name=MODEL_NAME,
            head_version=self._head_meta.get("version", "unknown"),
            low_confidence=conf < self.confidence_threshold,
            notes=notes,
            heatmap=heat,
        )

    @property
    def metadata(self) -> dict:
        return dict(self._head_meta)


def heatmap_overlay(img: Image.Image, heat: list[list[float]], alpha: float = 0.45) -> Image.Image:
    """Blend a 3x3 patch-score grid over the photo (sky-blue = higher water score)."""
    arr = np.asarray(heat, dtype=np.float32)
    small = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
    mask = small.resize(img.size, Image.BILINEAR)
    color = Image.new("RGB", img.size, (56, 189, 248))
    a = mask.point(lambda v: int(v * alpha))
    out = img.convert("RGB").copy()
    out.paste(color, (0, 0), a)
    return out
