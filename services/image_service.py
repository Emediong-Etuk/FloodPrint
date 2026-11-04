"""Upload validation and safe storage for incident photos."""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
MAX_PIXELS = 40_000_000
MIN_SIDE = 64
STORE_MAX_SIDE = 1600


class ImageValidationError(ValueError):
    pass


@dataclass
class ValidatedImage:
    image: Image.Image
    format: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    gps: tuple[float, float] | None


def _exif_gps(img: Image.Image) -> tuple[float, float] | None:
    try:
        exif = img.getexif()
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        if not gps_ifd:
            return None

        def to_deg(v):
            d, m, s = (float(x) for x in v)
            return d + m / 60 + s / 3600

        lat = to_deg(gps_ifd[2]) * (-1 if gps_ifd.get(1) == "S" else 1)
        lon = to_deg(gps_ifd[4]) * (-1 if gps_ifd.get(3) == "W" else 1)
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except Exception:
        return None
    return None


def validate_image(data: bytes, filename: str, max_mb: float = 8.0) -> ValidatedImage:
    """Check extension, size, real format (magic bytes via Pillow) and integrity."""
    if not data:
        raise ImageValidationError("The file is empty.")
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError("Only JPG, JPEG, PNG and WEBP images are accepted.")
    if len(data) > max_mb * 1024 * 1024:
        raise ImageValidationError(f"Image is larger than {max_mb:g} MB.")
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        probe.verify()  # integrity check; object unusable afterwards
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("The file is not a readable image or is corrupted.") from exc
    if fmt not in ALLOWED_FORMATS:
        raise ImageValidationError(f"Unsupported image format: {fmt}.")
    if ext in {".jpg", ".jpeg"} and fmt != "JPEG" or ext == ".png" and fmt != "PNG" or ext == ".webp" and fmt != "WEBP":
        raise ImageValidationError("File extension does not match the image contents.")
    gps = _exif_gps(img)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if min(img.size) < MIN_SIDE:
        raise ImageValidationError(f"Image is too small (minimum {MIN_SIDE}px per side).")
    return ValidatedImage(img, fmt, ALLOWED_FORMATS[fmt], img.width, img.height, len(data), gps)


def perceptual_hash(img: Image.Image) -> str:
    import imagehash

    return str(imagehash.phash(img))


def hash_distance(a: str | None, b: str | None) -> int | None:
    if not a or not b:
        return None
    import imagehash

    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)


def save_image(img: Image.Image, uploads_dir: Path) -> str:
    """Save a re-encoded copy with a random name. Returns a path relative to uploads_dir.

    Re-encoding strips metadata (including GPS) and neutralises any payload
    hidden in the original file. User-supplied names are never used.
    """
    uploads_dir = Path(uploads_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.jpg"
    out = img.copy()
    out.thumbnail((STORE_MAX_SIDE, STORE_MAX_SIDE))
    out.save(uploads_dir / name, "JPEG", quality=88)
    return name


def resolve_upload(uploads_dir: Path, relative: str | None) -> Path | None:
    """Resolve a stored relative path, refusing anything outside uploads_dir."""
    if not relative:
        return None
    base = Path(uploads_dir).resolve()
    p = (base / relative).resolve()
    if base not in p.parents or not p.is_file():
        return None
    return p
