"""Training pipeline for the flood-risk model.

Data
----
- Rainfall: NOAA GSOD daily precipitation for West / Central African stations.
- Labels: DFO Global Active Archive of Large Flood Events.

Target
------
``y = 1`` when a DFO event *begins* within days ``t+1 .. t+7`` and the station
lies inside the event's influence radius (equal-area circle from the reported
affected area, clipped to 50-300 km).

Evaluation
----------
- Chronological split: train <= 2012, calibration/validation 2013-2016,
  test 2017-01-01 .. 2021-09-30 (DFO archive ends Oct 2021).
- Spatial check: model trained without any Nigerian station, tested on
  Nigerian stations in the test period.
- Baselines: (a) climatology logistic regression on season + recent flood
  record + elevation; (b) XGBoost on the full feature set.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ai.feature_engineering import (
    FEATURES,
    PREDICTION_WINDOW_DAYS,
    build_daily_features,
    haversine_km,
    load_dfo_events,
)
from ai.risk_prediction import CalibratedRiskModel

log = logging.getLogger(__name__)

TRAIN_END = "2012-12-31"
VAL_END = "2016-12-31"
TEST_END = "2021-09-30"
MIN_YEAR_COVERAGE = 0.85
MIN_ANNUAL_MM = 200.0
NIGERIA_PREFIX = ("65", "650", "651", "652")


# ---------------------------------------------------------------- dataset
def clean_gsod(gsod: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (daily rainfall in mm, station metadata) after quality filtering."""
    g = gsod.copy()
    g["date"] = pd.to_datetime(g["DATE"])
    g["rain_mm"] = g["PRCP"].where(g["PRCP"] < 99.0) * 25.4  # 99.99 = missing
    g["year"] = g["date"].dt.year
    yr = g.groupby(["STATION", "year"]).agg(valid=("rain_mm", "count"), total=("rain_mm", "sum"))
    ndays = np.where(pd.Series(yr.index.get_level_values("year")).apply(
        lambda y: pd.Timestamp(year=y, month=12, day=31).dayofyear).values == 366, 366, 365)
    yr["coverage"] = yr["valid"].values / ndays
    good = yr[(yr["coverage"] >= MIN_YEAR_COVERAGE) & (yr["total"] >= MIN_ANNUAL_MM)].index
    g = g.set_index(["STATION", "year"]).loc[lambda d: d.index.isin(good)].reset_index()
    meta = (
        g.groupby("STATION")
        .agg(lat=("LATITUDE", "first"), lon=("LONGITUDE", "first"),
             elevation_m=("ELEVATION", "first"), name=("NAME", "first"),
             station_years=("year", "nunique"))
        .reset_index()
        .rename(columns={"STATION": "station"})
    )
    daily = g[["STATION", "date", "rain_mm"]].rename(columns={"STATION": "station"})
    return daily, meta


REGIONAL_RADIUS_KM = 100.0


def regional_series(station: str, wide: pd.DataFrame, meta: pd.DataFrame) -> pd.Series:
    """Mean daily rainfall over all quality-checked stations within 100 km.

    Area-averaging reduces single-gauge noise and is closer to the gridded
    model rainfall (Open-Meteo) used at prediction time.
    """
    row = meta.loc[meta["station"] == station].iloc[0]
    d = haversine_km(row["lat"], row["lon"], meta["lat"].values, meta["lon"].values)
    cols = meta.loc[d <= REGIONAL_RADIUS_KM, "station"].tolist()
    sub = wide[cols]
    s = sub.mean(axis=1, skipna=True).where(sub.notna().sum(axis=1) > 0)
    own = wide[station]
    return s.loc[own.first_valid_index():own.last_valid_index()]


def station_dataset(
    station: str, rain: pd.Series, meta_row: pd.Series, events: pd.DataFrame
) -> pd.DataFrame:
    feats = build_daily_features(rain.asfreq("D"))
    feats["elevation_m"] = float(meta_row["elevation_m"])

    d = haversine_km(meta_row["lat"], meta_row["lon"], events["lat"].values, events["lon"].values)
    near = events[d <= events["radius_km"].values].sort_values("began")
    began = near["began"].values.astype("datetime64[D]")
    t = feats.index.values.astype("datetime64[D]")
    # Prior events in [t - 5y, t)
    lo = np.searchsorted(began, t - np.timedelta64(1826, "D"), side="left")
    hi = np.searchsorted(began, t, side="left")
    feats["prior_events_5y"] = (hi - lo).astype(float)
    # Label: any onset in (t, t + 7]
    a = np.searchsorted(began, t + np.timedelta64(1, "D"), side="left")
    b = np.searchsorted(began, t + np.timedelta64(PREDICTION_WINDOW_DAYS + 1, "D"), side="left")
    feats["y"] = (b > a).astype(int)
    feats["station"] = station
    feats["lat"] = float(meta_row["lat"])
    feats["lon"] = float(meta_row["lon"])
    feats.index.name = "date"
    return feats.reset_index()


def build_training_table(gsod_path: Path, dfo_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gsod = pd.read_csv(gsod_path, dtype={"STATION": str})
    daily, meta = clean_gsod(gsod)
    events = load_dfo_events(dfo_path)
    wide = daily.pivot_table(index="date", columns="station", values="rain_mm").asfreq("D")
    frames = [station_dataset(r["station"], regional_series(r["station"], wide, meta), r, events)
              for _, r in meta.iterrows()]
    table = pd.concat(frames, ignore_index=True)
    table = table[table["date"] <= TEST_END]
    table = table.dropna(subset=FEATURES).reset_index(drop=True)
    return table, meta, events


# ----------------------------------------------------------------- metrics
def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float, base_rate: float) -> dict:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    brier = brier_score_loss(y, p)
    brier_ref = brier_score_loss(y, np.full_like(p, base_rate, dtype=float))
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
        "pr_auc": float(average_precision_score(y, p)) if y.sum() > 0 else None,
        "brier": float(brier),
        "brier_skill_vs_base_rate": float(1 - brier / brier_ref) if brier_ref > 0 else None,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    qs = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, len(qs) - 2)
    out = []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append({"mean_predicted": float(p[m].mean()), "observed_rate": float(y[m].mean()),
                    "count": int(m.sum())})
    return out


def best_threshold(y: np.ndarray, p: np.ndarray, beta: float = 2.0) -> float:
    """Threshold maximising F-beta on validation data (beta=2 weights recall higher)."""
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(y, p)
    fb = (1 + beta**2) * prec * rec / np.maximum(beta**2 * prec + rec, 1e-12)
    i = int(np.nanargmax(fb[:-1]))
    return float(thr[i])


# ------------------------------------------------------------------ models
class PlattCalibrator:
    """Logistic calibration on the logit of the raw score."""

    def fit(self, p, y):
        from sklearn.linear_model import LogisticRegression

        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._z(p), y)
        return self

    @staticmethod
    def _z(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def predict(self, p):
        return self.lr.predict_proba(self._z(p))[:, 1]

def make_rf(seed: int = 42):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=300, max_features="sqrt",
        max_samples=0.5, class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
    )


RAIN_FEATURES = ["rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d", "max_daily_7d",
                 "wet_days_7d", "fcst_3d", "fcst_7d", "fcst_max_daily_7d", "prior_events_5y"]


def make_xgb(pos_weight: float, seed: int = 42):
    """XGBoost comparison; risk may only rise with more rain (monotone constraints)."""
    from xgboost import XGBClassifier

    mono = "(" + ",".join("1" if f in RAIN_FEATURES else "0" for f in FEATURES) + ")"
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, reg_lambda=5.0,
        colsample_bytree=0.8, min_child_weight=50, scale_pos_weight=pos_weight, monotone_constraints=mono,
        eval_metric="aucpr", n_jobs=4, random_state=seed, tree_method="hist",
    )


def make_climatology():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))


CLIMATOLOGY_FEATURES = ["doy_sin", "doy_cos", "prior_events_5y", "elevation_m"]


def fit_calibrated(estimator, Xtr, ytr, Xcal, ycal, features) -> CalibratedRiskModel:
    estimator.fit(Xtr[features].to_numpy(dtype=float), ytr)
    raw_cal = estimator.predict_proba(Xcal[features].to_numpy(dtype=float))[:, 1]
    # Platt scaling keeps the model's ranking (isotonic created ties with so few positives).
    iso = PlattCalibrator().fit(raw_cal, ycal)
    medians = {f: float(Xtr[f].median()) for f in features}
    return CalibratedRiskModel(estimator, iso, features, medians)


def evaluate(model: CalibratedRiskModel, X: pd.DataFrame, y: np.ndarray, threshold: float, base_rate: float) -> dict:
    p = model.predict_proba(X)
    m = classification_metrics(y, p, threshold, base_rate)
    m["calibration"] = calibration_bins(y, p)
    return m


def train_and_evaluate(table: pd.DataFrame, meta: pd.DataFrame, events: pd.DataFrame,
                       out_dir: Path, seed: int = 42) -> dict:
    import joblib

    from ai.explainability import permutation_importance_table

    tr = table[table["date"] <= TRAIN_END]
    va = table[(table["date"] > TRAIN_END) & (table["date"] <= VAL_END)]
    te = table[(table["date"] > VAL_END) & (table["date"] <= TEST_END)]
    ytr, yva, yte = tr["y"].to_numpy(), va["y"].to_numpy(), te["y"].to_numpy()
    base_rate = float(ytr.mean())
    log.info("Rows: train %d (%d pos), val %d (%d pos), test %d (%d pos)",
             len(tr), ytr.sum(), len(va), yva.sum(), len(te), yte.sum())

    results: dict = {"models": {}}

    # Climatology baseline
    clim = fit_calibrated(make_climatology(), tr, ytr, va, yva, CLIMATOLOGY_FEATURES)
    thr = best_threshold(yva, clim.predict_proba(va))
    results["models"]["climatology_logreg"] = {"test": evaluate(clim, te, yte, thr, base_rate),
                                               "features": CLIMATOLOGY_FEATURES}

    # XGBoost comparison
    xgb = fit_calibrated(make_xgb((1 - base_rate) / base_rate, seed), tr, ytr, va, yva, FEATURES)
    thr = best_threshold(yva, xgb.predict_proba(va))
    results["models"]["xgboost"] = {"test": evaluate(xgb, te, yte, thr, base_rate), "features": FEATURES}

    # Random Forest (deployed)
    rf = fit_calibrated(make_rf(seed), tr, ytr, va, yva, FEATURES)
    rf_thr = best_threshold(yva, rf.predict_proba(va))
    results["models"]["random_forest"] = {
        "validation": evaluate(rf, va, yva, rf_thr, base_rate),
        "test": evaluate(rf, te, yte, rf_thr, base_rate),
        "features": FEATURES,
    }

    # Spatial generalisation: no Nigerian stations in training, test on Nigeria.
    is_ng = table["station"].str.startswith(("650", "651", "652"))
    tr_s = tr[~is_ng.loc[tr.index]]
    va_s = va[~is_ng.loc[va.index]]
    te_s = te[is_ng.loc[te.index]]
    if len(te_s) and te_s["y"].sum() > 0:
        rf_s = fit_calibrated(make_rf(seed), tr_s, tr_s["y"].to_numpy(), va_s, va_s["y"].to_numpy(), FEATURES)
        thr_s = best_threshold(va_s["y"].to_numpy(), rf_s.predict_proba(va_s))
        results["spatial_holdout_nigeria"] = {
            "description": "Random Forest trained without any Nigerian station; tested on Nigerian stations 2017-2021.",
            "train_stations": int(tr_s["station"].nunique()),
            "test_stations": int(te_s["station"].nunique()),
            "test": evaluate(rf_s, te_s, te_s["y"].to_numpy(), thr_s, float(tr_s["y"].mean())),
        }

    # Port Harcourt station, test period
    ph = te[te["station"] == "65250099999"]
    if len(ph):
        results["port_harcourt_station_test"] = {
            "rows": int(len(ph)),
            "positives": int(ph["y"].sum()),
            "metrics": evaluate(rf, ph, ph["y"].to_numpy(), rf_thr, base_rate) if 0 < ph["y"].sum() < len(ph) else None,
        }

    # Rolling-origin validation used for model selection (test period untouched)
    rolling = []
    for a, b in (("2008-12-31", "2012-12-31"), ("2012-12-31", "2016-12-31")):
        ftr = table[table["date"] <= a]
        fva = table[(table["date"] > a) & (table["date"] <= b)]
        fold = {"train_end": a, "validate_until": b, "positives": int(fva["y"].sum())}
        for name, mk, feats in (("random_forest", lambda: make_rf(seed), FEATURES),
                                ("climatology_logreg", make_climatology, CLIMATOLOGY_FEATURES)):
            est = mk().fit(ftr[feats].to_numpy(dtype=float), ftr["y"].to_numpy())
            p = est.predict_proba(fva[feats].to_numpy(dtype=float))[:, 1]
            from sklearn.metrics import average_precision_score, roc_auc_score
            fold[name] = {"roc_auc": float(roc_auc_score(fva["y"], p)),
                          "pr_auc": float(average_precision_score(fva["y"], p))}
        rolling.append(fold)
    results["rolling_origin_validation"] = rolling

    # Global importance (subsample for speed)
    sub = te.sample(n=min(60000, len(te)), random_state=seed)
    results["permutation_importance"] = permutation_importance_table(rf, sub, sub["y"].to_numpy())

    version = "rf-dfo-gsod-" + datetime.now(timezone.utc).strftime("%Y%m%d")
    results["metadata"] = {
        "model_version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "DFO large-flood onset within 7 days inside event influence radius",
        "prediction_window_days": PREDICTION_WINDOW_DAYS,
        "train_period": [str(tr["date"].min().date()), TRAIN_END],
        "validation_period": [str(va["date"].min().date()), VAL_END],
        "test_period": [str(te["date"].min().date()), TEST_END],
        "train_base_rate": base_rate,
        "stations_used": int(table["station"].nunique()),
        "rows": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te))},
        "positives": {"train": int(ytr.sum()), "validation": int(yva.sum()), "test": int(yte.sum())},
        "dfo_events_in_region": int(((events["lat"].between(-2, 18)) & (events["lon"].between(-20, 18))).sum()),
        "decision_threshold": rf_thr,
        "seed": seed,
        "rf_params": make_rf(seed).get_params(),
        "rainfall_aggregation": f"mean of quality-checked GSOD stations within {REGIONAL_RADIUS_KM:.0f} km",
    }

    reference = {f: float(tr[f].median()) for f in FEATURES}
    bundle = {
        "model": rf,
        "version": version,
        "features": FEATURES,
        "medians": rf.medians,
        "reference": reference,
        "metadata": results["metadata"],
        "metrics_summary": {k: v["test"] for k, v in results["models"].items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_dir / "risk_model.joblib", compress=3)
    (out_dir / "risk_model_metrics.json").write_text(json.dumps(results, indent=2, default=str))
    # Region-level event list used at inference for the prior_events_5y feature.
    ev = events[(events["lat"].between(-2, 18)) & (events["lon"].between(-20, 18))]
    ev.to_csv(out_dir / "dfo_events_region.csv", index=False)
    meta.to_csv(out_dir / "risk_training_stations.csv", index=False)
    return results
