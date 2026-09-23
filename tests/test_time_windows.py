"""Tests for deterministic host-owned calendar windows."""

import unittest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from university_agent.time_windows import resolve_remaining_week_window


class ResolveRemainingWeekWindowTests(unittest.TestCase):
    def test_midweek_window_uses_local_now_and_sunday_end(self):
        now = datetime(2054, 9, 23, 10, 30, tzinfo=ZoneInfo("Europe/Madrid"))

        lower, upper = resolve_remaining_week_window(
            now=now,
            timezone_name="Europe/Madrid",
        )

        self.assertEqual(lower, datetime(2054, 9, 23, 10, 30))
        self.assertEqual(upper, datetime(2054, 9, 27, 23, 59, 59, 999999))
        self.assertIsNone(lower.tzinfo)
        self.assertIsNone(upper.tzinfo)

    def test_host_timezone_conversion_precedes_boundary_calculation(self):
        now = datetime(2054, 9, 20, 23, 30, tzinfo=timezone.utc)

        lower, upper = resolve_remaining_week_window(
            now=now,
            timezone_name="Europe/Madrid",
        )

        self.assertEqual(lower, datetime(2054, 9, 21, 1, 30))
        self.assertEqual(upper, datetime(2054, 9, 27, 23, 59, 59, 999999))

    def test_sunday_still_ends_on_same_day(self):
        now = datetime(2054, 9, 27, 12, 0, tzinfo=ZoneInfo("Europe/Madrid"))

        lower, upper = resolve_remaining_week_window(
            now=now,
            timezone_name="Europe/Madrid",
        )

        self.assertEqual(lower, datetime(2054, 9, 27, 12, 0))
        self.assertEqual(upper, datetime(2054, 9, 27, 23, 59, 59, 999999))

    def test_daylight_saving_offset_does_not_change_local_week_end(self):
        now = datetime(2026, 10, 24, 12, 0, tzinfo=ZoneInfo("Europe/Madrid"))

        lower, upper = resolve_remaining_week_window(
            now=now,
            timezone_name="Europe/Madrid",
        )

        self.assertEqual(lower, datetime(2026, 10, 24, 12, 0))
        self.assertEqual(upper, datetime(2026, 10, 25, 23, 59, 59, 999999))

    def test_timezone_naive_now_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "now must be timezone-aware"):
            resolve_remaining_week_window(
                now=datetime(2054, 9, 23, 10, 30),
                timezone_name="Europe/Madrid",
            )

    def test_blank_or_padded_timezone_name_is_rejected(self):
        now = datetime(2054, 9, 23, tzinfo=timezone.utc)

        for timezone_name in ("", " ", " Europe/Madrid"):
            with self.subTest(timezone_name=timezone_name):
                with self.assertRaises(ValueError):
                    resolve_remaining_week_window(
                        now=now,
                        timezone_name=timezone_name,
                    )

    def test_unknown_timezone_is_rejected(self):
        with self.assertRaises(ZoneInfoNotFoundError):
            resolve_remaining_week_window(
                now=datetime(2054, 9, 23, tzinfo=timezone.utc),
                timezone_name="Fictional/Nowhere",
            )

    def test_exact_week_end_has_no_remaining_window(self):
        now = datetime(
            2054,
            9,
            27,
            23,
            59,
            59,
            999999,
            tzinfo=ZoneInfo("Europe/Madrid"),
        )

        with self.assertRaisesRegex(ValueError, "no time remains"):
            resolve_remaining_week_window(
                now=now,
                timezone_name="Europe/Madrid",
            )


if __name__ == "__main__":
    unittest.main()
