#!/usr/bin/env python3
"""FloodPrint Room: street-level flood reports from a group chat.

The Switch agent runs this script for every request. The script does the
exact work (matching places, counting reports, timing, rain lookups); the
agent decides what the person asked and explains the answer.

Standard library only. Reports live in reports.jsonl next to this file.

    python3 floodroom.py report "Rumuola" knee --by Ada --note "junction by the market"
    python3 floodroom.py status "ada george"
    python3 floodroom.py brief
    python3 floodroom.py clear "Rumuola" --by Ada
    python3 floodroom.py memory "D-Line"
    python3 floodroom.py places
    python3 floodroom.py seed      # demo history, marked as demo
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGION_FILE = HERE.parent / "config" / "regions" / "port_harcourt.json"
REPORTS_FILE = Path(os.environ.get("FLOODROOM_REPORTS", HERE / "reports.jsonl"))

LAGOS = timezone(timedelta(hours=1), "WAT")  # Nigeria has no daylight saving
LIVE_HOURS = 6  # a report counts as "live" for this long

# Depth, from least to most serious, with what it means for someone travelling.
DEPTHS = {
    "dry": (0, "✅", "clear, no water reported"),
    "ankle": (1, "🟡", "passable with care"),
    "knee": (2, "🟠", "avoid in small cars, walk with care"),
    "waist": (3, "🔴", "do not drive or walk through"),
    "impassable": (4, "⛔", "road cut off, find another route"),
}
DEPTH_WORDS = {
    "dry": "dry", "clear": "dry", "gone": "dry", "receded": "dry",
    "ankle": "ankle", "shallow": "ankle", "small": "ankle", "puddle": "ankle",
    "knee": "knee", "shin": "knee", "moderate": "knee",
    "waist": "waist", "thigh": "waist", "deep": "waist", "chest": "impassable",
    "impassable": "impassable", "cut": "impassable", "blocked": "impassable",
    "severe": "impassable", "neck": "impassable",
}
ALIASES = {
    "mile1": "Mile 1 Diobu", "mileone": "Mile 1 Diobu", "diobu": "Mile 1 Diobu",
    "dline": "D-Line", "gra2": "GRA Phase 2", "gra": "GRA Phase 2",
    "oldgra": "Old GRA", "transamadi": "Trans-Amadi", "adageorge": "Ada George",
    "uniport": "Choba", "rumuokwuta": "Rumuigbo",
}

EXIT_BAD_INPUT = 2


# ---------------------------------------------------------------- places ---

def load_places() -> list[dict]:
    with open(REGION_FILE, encoding="utf-8") as f:
        return json.load(f)["locations"]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def match_place(text: str, places: list[dict]) -> dict | None:
    """Turn what someone typed ("ada-george rd", "mile one") into a known place."""
    key = _norm(text)
    if not key:
        return None
    by_norm = {_norm(p["name"]): p for p in places}
    by_name = {p["name"]: p for p in places}
    if key in by_norm:
        return by_norm[key]
    for alias, name in ALIASES.items():
        if key == alias or key.startswith(alias):
            return by_name.get(name)
    # "rumuola junction" contains "rumuola"; prefer the longest match
    hits = [n for n in by_norm if n in key]
    if hits:
        return by_norm[max(hits, key=len)]
    close = difflib.get_close_matches(key, list(by_norm), n=1, cutoff=0.75)
    return by_norm[close[0]] if close else None


def parse_depth(text: str) -> str | None:
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in DEPTH_WORDS:
            return DEPTH_WORDS[word]
    return None


# --------------------------------------------------------------- storage ---

def load_reports() -> list[dict]:
    if not REPORTS_FILE.exists():
        return []
    out = []
    with open(REPORTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_report(rec: dict) -> None:
    REPORTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def next_id(reports: list[dict]) -> str:
    return f"FP-{len(reports) + 1:04d}"


def _ts(rec: dict) -> datetime:
    return datetime.fromisoformat(rec["at"])


def _local(dt: datetime) -> str:
    return dt.astimezone(LAGOS).strftime("%a %d %b %H:%M")


def _ago(dt: datetime, now: datetime) -> str:
    mins = int((now - dt).total_seconds() // 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins} min ago"
    return f"{mins // 60}h {mins % 60:02d}m ago"


# ---------------------------------------------------------- live picture ---

def live_reports(reports, place, now):
    """Reports for a place in the live window, newest first, cut at the last 'dry'."""
    cutoff = now - timedelta(hours=LIVE_HOURS)
    mine = [r for r in reports if r["place"] == place and cutoff <= _ts(r) <= now]
    mine.sort(key=_ts, reverse=True)
    live = []
    for r in mine:
        if r["depth"] == "dry":
            break
        live.append(r)
    return live, (mine[0] if mine and mine[0]["depth"] == "dry" else None)


def area_state(reports, place, now) -> dict:
    live, cleared = live_reports(reports, place, now)
    if not live:
        return {"place": place, "depth": "dry" if cleared else None,
                "reports": 0, "people": 0, "latest": cleared, "confirmed": False}
    latest = live[0]
    people = {r["by"].lower() for r in live}
    return {
        "place": place,
        "depth": latest["depth"],
        "reports": len(live),
        "people": len(people),
        "latest": latest,
        "confirmed": len(people) >= 2,
    }


def memory(reports, place) -> dict:
    """Days this place has flooded, from every report the room has ever had."""
    days = sorted({_ts(r).astimezone(LAGOS).date() for r in reports
                   if r["place"] == place and DEPTHS[r["depth"]][0] >= 2})
    worst = max((r for r in reports if r["place"] == place),
                key=lambda r: DEPTHS[r["depth"]][0], default=None)
    return {"place": place, "flood_days": days, "worst": worst}


# ------------------------------------------------------------------ rain ---

def rain_next_hours(lat: float, lon: float, hours: int = 6) -> dict | None:
    """Open-Meteo forecast (no key). Returns None if the network is unavailable."""
    if os.environ.get("FLOODROOM_OFFLINE"):
        return None
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "timezone": "Africa/Lagos",
        "hourly": "precipitation,precipitation_probability",
        "forecast_hours": hours,
    })
    try:
        with urllib.request.urlopen(f"https://api.open-meteo.com/v1/forecast?{q}", timeout=8) as r:
            data = json.load(r)["hourly"]
    except Exception:
        return None
    mm = [x or 0 for x in data.get("precipitation", [])]
    prob = [x or 0 for x in data.get("precipitation_probability", [])]
    return {"hours": len(mm), "total_mm": round(sum(mm), 1),
            "peak_mm": round(max(mm, default=0), 1), "max_prob": max(prob, default=0)}


def rain_line(rain: dict | None) -> str:
    if rain is None:
        return "🌧 Rain forecast: unavailable right now"
    total = rain["total_mm"]
    if total < 1:
        feel = "little or no rain expected"
    elif total < 10:
        feel = "some rain expected"
    elif total < 25:
        feel = "heavy rain expected, water may rise"
    else:
        feel = "very heavy rain expected, expect flooding"
    return (f"🌧 Next {rain['hours']}h: {total} mm, up to {rain['max_prob']}% chance ({feel})")


# -------------------------------------------------------------- commands ---

def depth_line(depth: str | None) -> str:
    if depth is None:
        return "⚪ no reports in the last 6h"
    _, icon, advice = DEPTHS[depth]
    return f"{icon} {depth.upper()}: {advice}"


def cmd_report(args, places, now) -> int:
    place = match_place(args.place, places)
    depth = parse_depth(args.depth)
    if not place:
        print(f"Unknown place: {args.place!r}. Run `places` for the list.")
        return EXIT_BAD_INPUT
    if not depth:
        print(f"Unknown depth: {args.depth!r}. Use one of: {', '.join(DEPTHS)}.")
        return EXIT_BAD_INPUT
    reports = load_reports()
    rec = {"id": next_id(reports), "at": now.isoformat(timespec="seconds"),
           "place": place["name"], "depth": depth, "by": args.by or "someone",
           "note": args.note or "", "photo": bool(args.photo)}
    if args.demo:
        rec["demo"] = True
    save_report(rec)
    reports.append(rec)
    st = area_state(reports, place["name"], now)
    mem = memory(reports, place["name"])

    if depth == "dry":
        print(f"✅ {rec['id']}: {place['name']} marked clear by {rec['by']}. Thank you.")
        return 0
    print(f"📍 {rec['id']} logged: {place['name']}, {depth} deep, by {rec['by']}"
          + (" (with photo)" if rec["photo"] else ""))
    print(depth_line(depth))
    if st["confirmed"]:
        print(f"👥 CONFIRMED: {st['reports']} reports from {st['people']} people in the last {LIVE_HOURS}h")
    else:
        print("👤 1 person so far. One more report from someone else confirms it.")
    n = len(mem["flood_days"])
    if n >= 2:
        print(f"🧠 The room remembers: {place['name']} has flooded on {n} days on record.")
    return 0


def cmd_status(args, places, now) -> int:
    place = match_place(args.place, places)
    if not place:
        print(f"Unknown place: {args.place!r}. Run `places` for the list.")
        return EXIT_BAD_INPUT
    reports = load_reports()
    st = area_state(reports, place["name"], now)
    mem = memory(reports, place["name"])
    print(f"📍 {place['name']}")
    print(depth_line(st["depth"]))
    if st["latest"] is not None:
        r = st["latest"]
        note = f": \"{r['note']}\"" if r.get("note") else ""
        print(f"🕒 Latest: {r['by']}, {_ago(_ts(r), now)}{note}")
    if st["reports"]:
        tag = "CONFIRMED" if st["confirmed"] else "unconfirmed"
        print(f"👥 {st['reports']} report(s) from {st['people']} people ({tag})")
    print(rain_line(rain_next_hours(place["lat"], place["lon"])))
    n = len(mem["flood_days"])
    if n:
        last = mem["flood_days"][-1].strftime("%d %b")
        print(f"🧠 Memory: flooded on {n} day(s) on record, last on {last}")
    else:
        print("🧠 Memory: no floods on record here yet")
    return 0


def cmd_brief(args, places, now) -> int:
    reports = load_reports()
    states = [area_state(reports, p["name"], now) for p in places]
    wet = [s for s in states if s["depth"] not in (None, "dry")]
    wet.sort(key=lambda s: (DEPTHS[s["depth"]][0], s["people"]), reverse=True)
    stamp = now.astimezone(LAGOS).strftime("%H:%M")
    if not wet:
        print(f"🌤 Port Harcourt at {stamp}: no flooding reported in the last {LIVE_HOURS}h.")
    else:
        print(f"🌊 Port Harcourt at {stamp}: {len(wet)} area(s) with water")
        for s in wet:
            _, icon, advice = DEPTHS[s["depth"]]
            tag = "confirmed" if s["confirmed"] else "1 report"
            print(f"{icon} {s['place']}: {s['depth']}, {advice} ({tag}, {_ago(_ts(s['latest']), now)})")
    centre = {"lat": 4.8156, "lon": 7.0122}
    print(rain_line(rain_next_hours(centre["lat"], centre["lon"])))
    # Places that flood often but have no report yet: worth a look.
    watch = []
    for p in places:
        if any(s["place"] == p["name"] for s in wet):
            continue
        n = len(memory(reports, p["name"])["flood_days"])
        if n >= 3:
            watch.append((n, p["name"]))
    if watch:
        watch.sort(reverse=True)
        names = ", ".join(f"{name} ({n} days)" for n, name in watch[:3])
        print(f"👀 No report yet, but these flood often: {names}")
    return 0


def cmd_clear(args, places, now) -> int:
    args.depth = "dry"
    args.note = args.note or "water gone"
    args.photo = False
    return cmd_report(args, places, now)


def cmd_memory(args, places, now) -> int:
    place = match_place(args.place, places)
    if not place:
        print(f"Unknown place: {args.place!r}. Run `places` for the list.")
        return EXIT_BAD_INPUT
    reports = load_reports()
    mem = memory(reports, place["name"])
    days = mem["flood_days"]
    print(f"🧠 {place['name']}: flooded on {len(days)} day(s) on record")
    for d in days[-5:]:
        day_reports = [r for r in reports if r["place"] == place["name"]
                       and _ts(r).astimezone(LAGOS).date() == d]
        worst = max(day_reports, key=lambda r: DEPTHS[r["depth"]][0])
        print(f"- {d.strftime('%a %d %b %Y')}: {worst['depth']}, {len(day_reports)} report(s)")
    if mem["worst"] is not None and DEPTHS[mem["worst"]["depth"]][0] >= 2:
        w = mem["worst"]
        print(f"Worst seen: {w['depth']} on {_local(_ts(w))}")
    return 0


def cmd_places(args, places, now) -> int:
    print("Known places: " + ", ".join(p["name"] for p in places))
    return 0


def cmd_seed(args, places, now) -> int:
    """Demo history so the room has a memory on day one. Every record is marked demo."""
    history = [
        (40, "D-Line", "waist"), (40, "Rumuola", "knee"), (40, "Mile 1 Diobu", "waist"),
        (26, "D-Line", "knee"), (26, "Ada George", "knee"), (26, "Rumuola", "waist"),
        (12, "D-Line", "waist"), (12, "Mile 1 Diobu", "knee"), (12, "Rumuola", "knee"),
        (5, "D-Line", "knee"), (5, "Rumuokoro", "ankle"), (5, "Rumuola", "knee"),
    ]
    reports = load_reports()
    names = ["Ada", "Tonye", "Ibim", "Chidi"]
    for i, (days_ago, place, depth) in enumerate(history):
        at = (now - timedelta(days=days_ago, hours=i % 3)).isoformat(timespec="seconds")
        rec = {"id": next_id(reports), "at": at, "place": place, "depth": depth,
               "by": names[i % len(names)], "note": "", "photo": False, "demo": True}
        save_report(rec)
        reports.append(rec)
    print(f"Seeded {len(history)} demo reports across {len({h[1] for h in history})} places.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="floodroom", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="log a flood report")
    r.add_argument("place")
    r.add_argument("depth")
    r.add_argument("--by")
    r.add_argument("--note")
    r.add_argument("--photo", action="store_true", help="the report came with a photo")
    r.add_argument("--demo", action="store_true")

    s = sub.add_parser("status", help="can I pass through a place?")
    s.add_argument("place")

    sub.add_parser("brief", help="the whole city right now")

    c = sub.add_parser("clear", help="the water has gone")
    c.add_argument("place")
    c.add_argument("--by")
    c.add_argument("--note")
    c.add_argument("--demo", action="store_true")

    m = sub.add_parser("memory", help="flood history of a place")
    m.add_argument("place")

    sub.add_parser("places", help="list known places")
    sub.add_parser("seed", help="add demo history")
    return p


COMMANDS = {"report": cmd_report, "status": cmd_status, "brief": cmd_brief,
            "clear": cmd_clear, "memory": cmd_memory, "places": cmd_places, "seed": cmd_seed}


def main(argv: list[str] | None = None, now: datetime | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # emoji on Windows consoles
    args = build_parser().parse_args(argv)
    now = now or datetime.now(timezone.utc)
    return COMMANDS[args.cmd](args, load_places(), now)


if __name__ == "__main__":
    sys.exit(main())
