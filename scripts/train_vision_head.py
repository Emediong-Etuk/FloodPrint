"""Train and evaluate the Flood Vision classifier head.

    python scripts/build_vision_dataset.py   # once, fetches labelled images
    python scripts/train_vision_head.py

Steps
1. Embed every manifest image with the frozen OpenCLIP image encoder.
2. Choose the logistic-regression regularisation strength by 5-fold
   cross-validation on the training subset.
3. Report metrics two ways:
   - fixed held-out test subset (15% stratified split, never used for fitting);
   - 5-fold stratified cross-validation over all images (out-of-fold
     predictions, more stable given the small number of flood photos).
4. Compare with a zero-shot CLIP prompt baseline on the same images.
5. Fit the final head on train + val and save it with its metadata.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ai.flood_vision import ARCH, HEAD_CLASSES, PRETRAINED_TAG, WEIGHTS_URL, FloodVision  # noqa: E402
from config import MODELS_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402

log = logging.getLogger("train_vision_head")
SEED = 42


def metrics_block(y_true: np.ndarray, proba: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 f1_score, roc_auc_score)

    y_pred = proba.argmax(axis=1)
    rep = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=list(HEAD_CLASSES),
                                output_dict=True, zero_division=0)
    flood_true = (y_true == 2).astype(int)
    water_true = (y_true >= 1).astype(int)
    flood_pred = (y_pred == 2).astype(int)
    tp = int(((flood_pred == 1) & (flood_true == 1)).sum())
    fn = int(((flood_pred == 0) & (flood_true == 1)).sum())
    fp = int(((flood_pred == 1) & (flood_true == 0)).sum())
    tn = int(((flood_pred == 0) & (flood_true == 0)).sum())
    return {
        "n": int(len(y_true)),
        "class_counts": {c: int((y_true == i).sum()) for i, c in enumerate(HEAD_CLASSES)},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": {c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score")} for c in HEAD_CLASSES},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "binary_flooding": {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            "false_negative_rate": fn / (tp + fn) if tp + fn else None,
            "roc_auc": float(roc_auc_score(flood_true, proba[:, 2])),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        },
        "water_vs_dry_roc_auc": float(roc_auc_score(water_true, proba[:, 1] + proba[:, 2])),
    }


def bootstrap_ci(y_true, proba, fn, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            vals.append(fn(y_true[idx], proba[idx]))
        except ValueError:
            continue
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
    import joblib

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    man = pd.read_csv(PROCESSED_DIR / "vision_dataset_manifest.csv")
    img_dir = RAW_DIR / "vision_images"
    cls_index = {c: i for i, c in enumerate(HEAD_CLASSES)}
    y = man["label"].map(cls_index).to_numpy()

    emb_path = RAW_DIR / "vision_embeddings.npy"
    vis = FloodVision(MODELS_DIR, head_path=Path("/nonexistent"))
    vis._load_backbone()
    vis._prepare_text()
    vis._head = object()  # marks loaded; head not needed for embedding
    if emb_path.exists() and len(np.load(emb_path)) == len(man):
        emb = np.load(emb_path)
    else:
        log.info("Embedding %d images", len(man))
        imgs = [Image.open(img_dir / f"{i}.jpg").convert("RGB") for i in man["image_id"]]
        emb = vis.embed(imgs, batch_size=64)
        np.save(emb_path, emb)

    tr = (man["subset"] == "train").to_numpy()
    va = (man["subset"] == "val").to_numpy()
    te = (man["subset"] == "test").to_numpy()

    base = LogisticRegression(max_iter=5000, class_weight="balanced")
    grid = GridSearchCV(base, {"C": [0.1, 0.3, 1, 3, 10, 30]}, scoring="f1_macro",
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    grid.fit(emb[tr], y[tr])
    C = float(grid.best_params_["C"])
    log.info("Best C = %s (CV macro-F1 %.3f)", C, grid.best_score_)

    # Held-out test: fit on train+val
    trva = tr | va
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=C).fit(emb[trva], y[trva])
    test_proba = clf.predict_proba(emb[te])
    test_metrics = metrics_block(y[te], test_proba)

    # 5-fold CV across all images
    oof = cross_val_predict(LogisticRegression(max_iter=5000, class_weight="balanced", C=C), emb, y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")
    cv_metrics = metrics_block(y, oof)

    def flood_recall(yt, pp):
        m = yt == 2
        return float((pp[m].argmax(1) == 2).mean())

    def macro_f1(yt, pp):
        from sklearn.metrics import f1_score
        return f1_score(yt, pp.argmax(1), average="macro")

    cv_metrics["bootstrap_95ci"] = {"flood_recall": bootstrap_ci(y, oof, flood_recall),
                                    "macro_f1": bootstrap_ci(y, oof, macro_f1)}

    # Zero-shot baseline (no training)
    zs_all = vis.zero_shot_proba(emb)
    zs_metrics = {"test": metrics_block(y[te], zs_all[te]), "all_images": metrics_block(y, zs_all)}

    version = "oi-linear-" + datetime.now(timezone.utc).strftime("%Y%m%d")
    meta = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backbone": {"arch": ARCH, "pretrained": PRETRAINED_TAG, "weights_url": WEIGHTS_URL},
        "classifier": {"type": "LogisticRegression", "C": C, "class_weight": "balanced"},
        "dataset": {
            "source": "Open Images V7 human-verified image-level labels",
            "license": "Annotations CC BY 4.0 (Google); images listed as CC BY 2.0 by their authors",
            "manifest": "data/processed/vision_dataset_manifest.csv",
            "counts": man.groupby(["label", "subset"]).size().unstack(fill_value=0).to_dict(),
        },
        "classes": list(HEAD_CLASSES),
    }
    final = LogisticRegression(max_iter=5000, class_weight="balanced", C=C).fit(emb[trva], y[trva])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": final, "classes": list(HEAD_CLASSES), "metadata": meta},
                MODELS_DIR / "vision_head.joblib", compress=3)
    report = {"metadata": meta, "held_out_test": test_metrics, "cross_validation_5fold": cv_metrics,
              "zero_shot_baseline": zs_metrics, "cv_selection_macro_f1": float(grid.best_score_)}
    (MODELS_DIR / "vision_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "test_accuracy": test_metrics["accuracy"], "test_macro_f1": test_metrics["macro_f1"],
        "test_flood": test_metrics["binary_flooding"],
        "cv_accuracy": cv_metrics["accuracy"], "cv_macro_f1": cv_metrics["macro_f1"],
        "cv_flood": cv_metrics["binary_flooding"],
        "zero_shot_test_macro_f1": zs_metrics["test"]["macro_f1"],
        "zero_shot_all_macro_f1": zs_metrics["all_images"]["macro_f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
