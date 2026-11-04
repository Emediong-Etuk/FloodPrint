"""Place lookup: configured neighbourhoods first, OpenStreetMap Nominatim second.

Nominatim's usage policy requires an identifying User-Agent and at most one
request per second; results are cached and restricted to the region's box.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
import requests

from ai.feature_engineering import haversine_km
from config import Region

log = logging.getLogger(__name__)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "FLOODPRINT/1.0 (community flood intelligence prototype)"


@dataclass
class Place:
    name: str
    lat: float
    lon: float
    source: str


class GeocodingService:
    def __init__(self, region: Region, timeout_s: float = 8.0, enable_remote: bool = True):
        self.region = region
        self.timeout_s = timeout_s
        self.enable_remote = enable_remote
        self._cache: dict[str, list[Place]] = {}
        self._last_call = 0.0
        self._lock = threading.Lock()

    def nearest_location(self, lat: float, lon: float) -> tuple[str, float]:
        locs = self.region.locations
        d = haversine_km(lat, lon, np.array([l.lat for l in locs]), np.array([l.lon for l in locs]))
        i = int(np.argmin(d))
        return locs[i].name, float(d[i])

    def search(self, query: str, limit: int = 5) -> list[Place]:
        q = (query or "").strip()
        if len(q) < 2:
            return []
        local = [Place(l.name, l.lat, l.lon, "FLOODPRINT neighbourhood list")
                 for l in self.region.locations if q.lower() in l.name.lower()]
        if local or not self.enable_remote:
            return local[:limit]
        key = q.lower()
        if key in self._cache:
            return self._cache[key]
        r = self.region
        with self._lock:
            wait = 1.0 - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
        try:
            resp = requests.get(
                NOMINATIM_URL,
                params={"q": f"{q}, {r.name}", "format": "jsonv2", "limit": limit, "bounded": 1,
                        "viewbox": f"{r.min_lon},{r.max_lat},{r.max_lon},{r.min_lat}"},
                headers={"User-Agent": USER_AGENT},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            places = [Place(p.get("display_name", q)[:120], float(p["lat"]), float(p["lon"]), "OpenStreetMap Nominatim")
                      for p in resp.json()]
            places = [p for p in places if r.contains(p.lat, p.lon)]
        except (requests.RequestException, ValueError, KeyError) as exc:
            log.warning("Geocoding failed: %s", exc)
            return []
        self._cache[key] = places
        return places
