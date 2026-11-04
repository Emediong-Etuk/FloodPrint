"""Populate the database with clearly labelled demonstration data.

    python scripts/seed_demo_data.py            # add demo data
    python scripts/seed_demo_data.py --reset    # remove existing demo data first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings, load_region  # noqa: E402
from database import repository as repo  # noqa: E402
from database.connection import connect, initialize  # noqa: E402
from services.demo_data import has_demo_data, seed_demo_data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete existing demo records first")
    args = ap.parse_args()
    conn = connect(get_settings().db_path)
    initialize(conn)
    if args.reset:
        print("Removed", repo.delete_demo_data(conn), "demo incidents")
    if has_demo_data(conn):
        print("Demo data already present; use --reset to recreate it.")
        return
    print(seed_demo_data(conn, load_region()))


if __name__ == "__main__":
    main()
