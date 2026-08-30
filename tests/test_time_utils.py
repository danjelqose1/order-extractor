from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from time_utils import (  # noqa: E402
    current_platform_year,
    parse_platform_filter_datetime,
    platform_year_utc_bounds,
    utc_isoformat,
)


def test_utc_isoformat_marks_sqlite_naive_values_as_utc():
    stored = datetime(2026, 8, 29, 7, 26, 33, 547044)

    assert utc_isoformat(stored) == "2026-08-29T07:26:33.547044Z"


def test_utc_isoformat_converts_aware_values_to_utc():
    tirana_summer = timezone(timedelta(hours=2))
    local_value = datetime(2026, 8, 29, 9, 26, 33, tzinfo=tirana_summer)

    assert utc_isoformat(local_value) == "2026-08-29T07:26:33Z"


def test_platform_date_filter_uses_tirana_summer_midnight():
    start = parse_platform_filter_datetime("2026-08-29")
    end = parse_platform_filter_datetime("2026-08-29", end_exclusive=True)

    assert start == datetime(2026, 8, 28, 22, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 8, 29, 22, 0, tzinfo=timezone.utc)


def test_platform_date_filter_uses_tirana_winter_midnight():
    start = parse_platform_filter_datetime("2026-01-15")
    end = parse_platform_filter_datetime("2026-01-15", end_exclusive=True)

    assert start == datetime(2026, 1, 14, 23, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 1, 15, 23, 0, tzinfo=timezone.utc)


def test_platform_date_filter_handles_daylight_saving_transition_days():
    spring_start = parse_platform_filter_datetime("2026-03-29")
    spring_end = parse_platform_filter_datetime("2026-03-29", end_exclusive=True)
    autumn_start = parse_platform_filter_datetime("2026-10-25")
    autumn_end = parse_platform_filter_datetime("2026-10-25", end_exclusive=True)

    assert spring_start == datetime(2026, 3, 28, 23, 0, tzinfo=timezone.utc)
    assert spring_end == datetime(2026, 3, 29, 22, 0, tzinfo=timezone.utc)
    assert autumn_start == datetime(2026, 10, 24, 22, 0, tzinfo=timezone.utc)
    assert autumn_end == datetime(2026, 10, 25, 23, 0, tzinfo=timezone.utc)


def test_platform_year_boundaries_and_current_year_use_tirana():
    start, end = platform_year_utc_bounds(2026)

    assert start == datetime(2025, 12, 31, 23, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 12, 31, 23, 0, tzinfo=timezone.utc)
    assert current_platform_year(datetime(2026, 12, 31, 23, 30, tzinfo=timezone.utc)) == 2027


def test_explicit_offsets_are_respected_for_datetime_filters():
    parsed = parse_platform_filter_datetime("2026-08-29T09:26:33+02:00")

    assert parsed == datetime(2026, 8, 29, 7, 26, 33, tzinfo=timezone.utc)
