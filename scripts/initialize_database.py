"""Create the SQLite schema (safe to run repeatedly).

    python scripts/initialize_database.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from database.connection import connect, initialize  # noqa: E402


def main() -> None:
    path = get_settings().db_path
    conn = connect(path)
    initialize(conn)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    print(f"Database ready at {path}: {', '.join(tables)}")


if __name__ == "__main__":
    main()
