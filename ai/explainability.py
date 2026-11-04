"""Explanations for risk predictions.

Local contributions use a simple, model-agnostic occlusion method: each feature
is replaced by its *reference value* (the training-set median, i.e. a typical
day) while the others are held fixed, and the change in calibrated probability
is recorded. Positive values mean the feature's current value pushes risk up
compared with a typical day. These are one-at-a-time effects, so they do not
sum exactly to the prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai.feature_engineering import FEATURE_DESCRIPTIONS

# Season sine/cosine are reported together as one "season" factor.
_GROUPS = {"doy_sin": "season", "doy_cos": "season"}


def local_contributions(model, X: pd.DataFrame, reference: dict[str, float]) -> list[dict]:
    base = float(model.predict_proba(X)[0])
    groups: dict[str, list[str]] = {}
    for f in model.features:
        groups.setdefault(_GROUPS.get(f, f), []).append(f)
    batch, names = [], []
    for gname, cols in groups.items():
        Xr = X.copy()
        for c in cols:
            Xr[c] = reference[c]
        batch.append(Xr)
        names.append((gname, cols))
    probs = model.predict_proba(pd.concat(batch, ignore_index=True))
    out = []
    for (gname, cols), p in zip(names, probs):
        out.append(
            {
                "feature": gname,
                "label": "Season / time of year" if gname == "season" else FEATURE_DESCRIPTIONS.get(gname, gname),
                "value": None if gname == "season" else float(X.iloc[0][cols[0]]),
                "reference": None if gname == "season" else float(reference[cols[0]]),
                "effect": base - float(p),
            }
        )
    out.sort(key=lambda d: abs(d["effect"]), reverse=True)
    return out


def _fmt(v: float | None, feature: str) -> str:
    if v is None:
        return ""
    if feature in {"wet_days_7d", "prior_events_5y"}:
        return f"{v:.0f}"
    if feature == "elevation_m":
        return f"{v:.0f} m"
    return f"{v:.0f} mm"


def explain_prediction(pred, top_n: int = 3, min_effect: float = 0.005) -> str:
    """Plain-language summary built only from the computed contributions."""
    ups = [c for c in pred.contributions if c["effect"] >= min_effect][:top_n]
    downs = [c for c in pred.contributions if c["effect"] <= -min_effect][:2]
    parts = []
    if ups:
        reasons = []
        for c in ups:
            if c["feature"] == "season":
                reasons.append("the time of year")
            else:
                reasons.append(
                    f"{c['label'].lower()} is {_fmt(c['value'], c['feature'])} "
                    f"(typical: {_fmt(c['reference'], c['feature'])})"
                )
        parts.append("Risk is raised mainly because " + "; ".join(reasons) + ".")
    if downs:
        reasons = []
        for c in downs:
            if c["feature"] == "season":
                reasons.append("the time of year")
            else:
                reasons.append(f"{c['label'].lower()} is {_fmt(c['value'], c['feature'])}")
        parts.append("It is held down by " + "; ".join(reasons) + ".")
    if not parts:
        parts.append("No single factor moves the estimate much away from a typical day.")
    if pred.imputed_features:
        parts.append(
            "Some inputs were unavailable and were filled with typical values: "
            + ", ".join(FEATURE_DESCRIPTIONS.get(f, f) for f in pred.imputed_features)
            + "."
        )
    return " ".join(parts)


def permutation_importance_table(model, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 3, seed: int = 0):
    """Global importance: drop in ROC-AUC when a feature column is shuffled."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X))
    rows = []
    for f in model.features:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].values)
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)))
        rows.append({"feature": f, "label": FEATURE_DESCRIPTIONS.get(f, f),
                     "auc_drop_mean": float(np.mean(drops)), "auc_drop_std": float(np.std(drops))})
    return sorted(rows, key=lambda r: r["auc_drop_mean"], reverse=True)
