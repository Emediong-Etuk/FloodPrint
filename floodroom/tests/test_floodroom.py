import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["FLOODROOM_OFFLINE"] = "1"

import floodroom as fr  # noqa: E402

NOW = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)


def run(*argv, now=NOW):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = fr.main(list(argv), now=now)
    return code, buf.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(fr, "REPORTS_FILE", Path(self.tmp.name) / "r.jsonl")
        self.patch.start()
        self.places = fr.load_places()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()


class TestMatching(Base):
    def test_places_people_type(self):
        cases = {
            "rumuola junction": "Rumuola", "ada-george rd": "Ada George",
            "Ada George": "Ada George", "mile one": "Mile 1 Diobu", "diobu": "Mile 1 Diobu",
            "dline": "D-Line", "D Line": "D-Line", "trans amadi": "Trans-Amadi",
            "old gra": "Old GRA", "rumuokorro": "Rumuokoro",
        }
        for typed, want in cases.items():
            with self.subTest(typed=typed):
                self.assertEqual(fr.match_place(typed, self.places)["name"], want)

    def test_unknown_place(self):
        self.assertIsNone(fr.match_place("lagos island", self.places))
        self.assertIsNone(fr.match_place("", self.places))

    def test_depth_words(self):
        self.assertEqual(fr.parse_depth("knee deep"), "knee")
        self.assertEqual(fr.parse_depth("up to my waist"), "waist")
        self.assertEqual(fr.parse_depth("road is blocked"), "impassable")
        self.assertEqual(fr.parse_depth("water has gone"), "dry")
        self.assertIsNone(fr.parse_depth("bad"))


class TestReports(Base):
    def test_one_person_is_unconfirmed_two_confirm(self):
        _, out = run("report", "Rumuola", "knee", "--by", "Ada")
        self.assertIn("FP-0001", out)
        self.assertIn("1 person so far", out)
        _, out = run("report", "rumuola", "knee", "--by", "Tonye")
        self.assertIn("CONFIRMED", out)

    def test_same_person_twice_does_not_confirm(self):
        run("report", "Rumuola", "knee", "--by", "Ada")
        _, out = run("report", "Rumuola", "waist", "--by", "ada")
        self.assertNotIn("CONFIRMED", out)

    def test_reports_expire_after_live_window(self):
        run("report", "Rumuola", "waist", "--by", "Ada")
        later = NOW + timedelta(hours=fr.LIVE_HOURS, minutes=1)
        _, out = run("status", "Rumuola", now=later)
        self.assertIn("no reports in the last 6h", out)
        self.assertIn("flooded on 1 day", out)

    def test_clear_resets_live_status(self):
        run("report", "Rumuola", "waist", "--by", "Ada")
        run("clear", "Rumuola", "--by", "Tonye", now=NOW + timedelta(minutes=30))
        _, out = run("status", "Rumuola", now=NOW + timedelta(minutes=31))
        self.assertIn("DRY", out)
        _, out = run("brief", now=NOW + timedelta(minutes=31))
        self.assertIn("no flooding reported", out)

    def test_brief_orders_worst_first(self):
        run("report", "Rumuola", "ankle", "--by", "Ada")
        run("report", "D-Line", "impassable", "--by", "Tonye")
        _, out = run("brief")
        self.assertLess(out.index("D-Line"), out.index("Rumuola"))

    def test_bad_input_exit_code(self):
        code, out = run("report", "Atlantis", "knee")
        self.assertEqual(code, fr.EXIT_BAD_INPUT)
        code, out = run("report", "Rumuola", "bad")
        self.assertEqual(code, fr.EXIT_BAD_INPUT)
        self.assertFalse(fr.REPORTS_FILE.exists())

    def test_seed_builds_memory_and_watch_list(self):
        run("seed")
        recs = fr.load_reports()
        self.assertTrue(all(r["demo"] for r in recs))
        _, out = run("brief")
        self.assertIn("flood often", out)
        self.assertIn("D-Line", out)

    def test_report_is_stored_as_json_lines(self):
        run("report", "Woji", "knee", "--by", "Ada", "--note", "near the bridge", "--photo")
        rec = json.loads(fr.REPORTS_FILE.read_text().splitlines()[0])
        self.assertEqual(rec["place"], "Woji")
        self.assertTrue(rec["photo"])
        self.assertEqual(rec["note"], "near the bridge")


class TestRain(unittest.TestCase):
    def test_offline_returns_none(self):
        self.assertIsNone(fr.rain_next_hours(4.8, 7.0))
        self.assertIn("unavailable", fr.rain_line(None))

    def test_parses_open_meteo(self):
        body = json.dumps({"hourly": {"precipitation": [0, 2.5, 8, None, 1, 0],
                                      "precipitation_probability": [10, 60, 90, 80, 40, 20]}})
        fake = io.BytesIO(body.encode())
        with mock.patch.dict(os.environ, {"FLOODROOM_OFFLINE": ""}), \
             mock.patch.object(fr.urllib.request, "urlopen", return_value=fake):
            rain = fr.rain_next_hours(4.8, 7.0)
        self.assertEqual(rain["total_mm"], 11.5)
        self.assertEqual(rain["max_prob"], 90)
        self.assertIn("heavy rain", fr.rain_line(rain))


if __name__ == "__main__":
    unittest.main()
