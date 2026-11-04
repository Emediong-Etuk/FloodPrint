"""Train and evaluate the flood-risk model.

    python scripts/download_risk_data.py   # once, fetches DFO + GSOD
    python scripts/train_model.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.model_training import build_training_table, train_and_evaluate  # noqa: E402
from config import MODELS_DIR, RAW_DIR  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    gsod = RAW_DIR / "gsod_west_africa.csv.gz"
    dfo = RAW_DIR / "dfo_flood_archive.xlsx"
    if not gsod.exists() or not dfo.exists():
        sys.exit("Training data missing. Run `python scripts/download_risk_data.py` first.")
    table, meta, events = build_training_table(gsod, dfo)
    logging.info("Training table: %d rows, %d stations, %.2f%% positive",
                 len(table), table["station"].nunique(), 100 * table["y"].mean())
    results = train_and_evaluate(table, meta, events, MODELS_DIR)
    summary = {k: {m: v["test"][m] for m in ("roc_auc", "pr_auc", "brier_skill_vs_base_rate", "recall", "precision", "f1")}
               for k, v in results["models"].items()}
    print(json.dumps(summary, indent=2))
    if "spatial_holdout_nigeria" in results:
        t = results["spatial_holdout_nigeria"]["test"]
        print("Spatial hold-out (Nigeria):", {k: t[k] for k in ("roc_auc", "pr_auc", "recall", "precision")})
    logging.info("Done in %.0f s", time.time() - t0)


if __name__ == "__main__":
    main()
