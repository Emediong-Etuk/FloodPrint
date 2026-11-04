"""Download the public datasets used to train the flood-risk model.

1. Dartmouth Flood Observatory (DFO) Global Active Archive of Large Flood Events
   (1985-2021, 5,130 events). Fetched from a public GitHub mirror of the
   ``FloodArchive.xlsx`` export because floodobservatory.colorado.edu is not
   always reachable. Citation: G.R. Brakenridge, "Global Active Archive of Large
   Flood Events", Dartmouth Flood Observatory, University of Colorado.
2. NOAA Global Surface Summary of the Day (GSOD) daily station precipitation
   from the NOAA Open Data Dissemination bucket on AWS (public domain).

Usage:
    python scripts/download_risk_data.py [--start 1995] [--end 2021]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, RAW_DIR  # noqa: E402

log = logging.getLogger("download_risk_data")

DFO_URL = (
    "https://raw.githubusercontent.com/Asfandyarriaz/"
    "Statistical-Analysis-of-the-Relationship-Between-Global-Warming-and-Flood-Frequency/"
    "main/FloodArchive.xlsx"
)
ISD_HISTORY_URL = "https://noaa-isd-pds.s3.amazonaws.com/isd-history.csv"
GSOD_URL = "https://noaa-gsod-pds.s3.amazonaws.com/{year}/{station}.csv"

# West / Central African study region (Gulf of Guinea and Sahel margin).
STUDY_BBOX = {"min_lat": 0.0, "max_lat": 16.0, "min_lon": -18.0, "max_lon": 16.0}
COUNTRIES = {"NI", "GH", "BN", "TO", "IV", "CM", "SG", "GV", "SL", "LI", "UV", "NG",
             "GA", "EK", "CD", "PU", "ML", "GB"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_dfo(dest: Path) -> dict:
    log.info("Downloading DFO archive")
    r = requests.get(DFO_URL, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    df = pd.read_excel(io.BytesIO(r.content))
    return {"url": DFO_URL, "sha256": sha256(r.content), "events": int(len(df)),
            "began_min": str(df["Began"].min().date()), "began_max": str(df["Began"].max().date())}


def select_stations(start: int, end: int) -> pd.DataFrame:
    r = requests.get(ISD_HISTORY_URL, timeout=60)
    r.raise_for_status()
    st = pd.read_csv(io.BytesIO(r.content), dtype=str)
    st = st.dropna(subset=["LAT", "LON"])
    st["LAT"] = st["LAT"].astype(float)
    st["LON"] = st["LON"].astype(float)
    st["BEGIN"] = st["BEGIN"].astype(int)
    st["END"] = st["END"].astype(int)
    m = (
        st.LAT.between(STUDY_BBOX["min_lat"], STUDY_BBOX["max_lat"])
        & st.LON.between(STUDY_BBOX["min_lon"], STUDY_BBOX["max_lon"])
        & st.CTRY.isin(COUNTRIES)
        & (st.BEGIN <= start * 10000 + 101)
        & (st.END >= end * 10000 + 101)
    )
    st = st[m].copy()
    st["station"] = st["USAF"] + st["WBAN"]
    return st.drop_duplicates("station")


def fetch_station_year(station: str, year: int) -> pd.DataFrame | None:
    try:
        r = requests.get(GSOD_URL.format(year=year, station=station), timeout=60)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    df = pd.read_csv(io.BytesIO(r.content), dtype={"STATION": str},
                     usecols=["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME",
                              "PRCP", "PRCP_ATTRIBUTES"])
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1995)
    ap.add_argument("--end", type=int, default=2021)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"dfo": download_dfo(RAW_DIR / "dfo_flood_archive.xlsx")}

    stations = select_stations(args.start, args.end)
    log.info("Selected %d GSOD stations", len(stations))
    jobs = [(s, y) for s in stations["station"] for y in range(args.start, args.end + 1)]
    frames = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_station_year, s, y): (s, y) for s, y in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            df = fut.result()
            if df is not None:
                frames.append(df)
            if i % 500 == 0:
                log.info("  %d / %d station-years fetched", i, len(jobs))
    gsod = pd.concat(frames, ignore_index=True)
    out = RAW_DIR / "gsod_west_africa.csv.gz"
    gsod.to_csv(out, index=False, compression="gzip")
    manifest["gsod"] = {
        "source": "NOAA GSOD via s3://noaa-gsod-pds",
        "stations": int(gsod.STATION.nunique()),
        "rows": int(len(gsod)),
        "years": [args.start, args.end],
        "bbox": STUDY_BBOX,
    }
    (PROCESSED_DIR / "risk_data_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Saved %s (%d rows, %d stations)", out, len(gsod), gsod.STATION.nunique())


if __name__ == "__main__":
    main()
