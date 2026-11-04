"""Build the labelled image set used to train and evaluate the Flood Vision head.

Source: Google Open Images V7 human-verified image-level labels (annotations
CC BY 4.0; images are listed by Open Images as CC BY 2.0 by their Flickr
authors). Images are downloaded from the public ``open-images-dataset`` S3
bucket, resized, and kept locally under ``data/raw/vision_images`` (gitignored).
Only the manifest of image IDs and labels is committed.

Class definitions (all from *human-verified* labels):
- ``flooding``      Flood = 1
- ``waterlogging``  Puddle = 1 and not Flood = 1   (minor surface water)
- ``no_flooding``   Flood verified absent (= 0, hard negatives: water scenes humans
                    rejected as floods), plus street / road / residential scenes
                    with no Flood or Puddle positive label.

Usage:
    python scripts/build_vision_dataset.py [--max-street 900] [--workers 24]
"""

from __future__ import annotations

import argparse
import io
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, RAW_DIR  # noqa: E402

log = logging.getLogger("build_vision_dataset")

LABEL_URLS = {
    "train": "https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv",
    "validation": "https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv",
    "test": "https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-human-imagelabels.csv",
}
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"

FLOOD = "/m/0dbtv"
PUDDLE = "/m/09x984"
STREET_LIKE = {"/m/06gfj", "/m/01c8br", "/m/0180xr", "/m/02nfxt", "/m/039jbq"}  # road, street, neighbourhood, residential, urban
WANTED = {FLOOD, PUDDLE, *STREET_LIKE}


def fetch_labels(split: str, cache: Path) -> pd.DataFrame:
    """Stream a label file and keep only the label IDs we need."""
    if cache.exists():
        return pd.read_csv(cache)
    log.info("Streaming %s labels (filtering on the fly)", split)
    pattern = "|".join(WANTED)
    # curl | grep keeps memory flat for the 2.7 GB train file.
    cmd = f"curl -sS --fail '{LABEL_URLS[split]}' | grep -E '{pattern}'"
    out = subprocess.run(cmd, shell=True, check=True, capture_output=True).stdout
    df = pd.read_csv(io.BytesIO(out), names=["ImageID", "Source", "LabelName", "Confidence"])
    df["split"] = split
    df.to_csv(cache, index=False)
    return df


def build_manifest(labels: pd.DataFrame, max_street: int, seed: int) -> pd.DataFrame:
    piv = labels.pivot_table(index=["ImageID", "split"], columns="LabelName",
                             values="Confidence", aggfunc="max").reset_index()
    for col in WANTED:
        if col not in piv:
            piv[col] = np.nan
    flood_pos = piv[FLOOD] == 1
    puddle_pos = piv[PUDDLE] == 1
    flood_neg = piv[FLOOD] == 0
    street = piv[list(STREET_LIKE)].eq(1).any(axis=1)

    flooding = piv[flood_pos].assign(label="flooding", origin="flood_verified_present")
    waterlog = piv[puddle_pos & ~flood_pos].assign(label="waterlogging", origin="puddle_verified_present")
    hard_neg = piv[flood_neg & ~puddle_pos].assign(label="no_flooding", origin="flood_verified_absent")
    street_neg = piv[street & ~flood_pos & ~puddle_pos & ~flood_neg]
    street_neg = street_neg.sample(n=min(max_street, len(street_neg)), random_state=seed)
    street_neg = street_neg.assign(label="no_flooding", origin="street_scene_no_water_label")

    man = pd.concat([flooding, waterlog, hard_neg, street_neg], ignore_index=True)
    man = man[["ImageID", "split", "label", "origin"]].drop_duplicates("ImageID")

    # Stratified 70 / 15 / 15 split with a fixed seed.
    rng = np.random.default_rng(seed)
    man["subset"] = ""
    for lab, idx in man.groupby("label").groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_tr, n_va = int(0.70 * n), int(0.15 * n)
        man.loc[idx[:n_tr], "subset"] = "train"
        man.loc[idx[n_tr:n_tr + n_va], "subset"] = "val"
        man.loc[idx[n_tr + n_va:], "subset"] = "test"
    return man.rename(columns={"ImageID": "image_id", "split": "oi_split"})


def download_image(row: pd.Series, out_dir: Path, max_side: int = 512) -> bool:
    dest = out_dir / f"{row.image_id}.jpg"
    if dest.exists():
        return True
    try:
        r = requests.get(IMAGE_URL.format(split=row.oi_split, image_id=row.image_id), timeout=60)
        if r.status_code != 200:
            return False
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img.thumbnail((max_side, max_side))
        img.save(dest, "JPEG", quality=90)
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-street", type=int, default=900)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cache_dir = RAW_DIR / "openimages_labels"
    cache_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.concat(
        [fetch_labels(s, cache_dir / f"{s}_filtered.csv") for s in LABEL_URLS], ignore_index=True
    )
    man = build_manifest(labels, args.max_street, args.seed)

    img_dir = RAW_DIR / "vision_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ok = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_image, row, img_dir): row.image_id for row in man.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            ok[futs[fut]] = fut.result()
            if i % 250 == 0:
                log.info("  %d / %d images", i, len(futs))
    man["downloaded"] = man["image_id"].map(ok)
    man = man[man["downloaded"]].drop(columns="downloaded")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "vision_dataset_manifest.csv"
    man.to_csv(out, index=False)
    log.info("Manifest saved to %s", out)
    log.info("\n%s", man.groupby(["label", "subset"]).size().unstack())


if __name__ == "__main__":
    main()
