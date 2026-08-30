from __future__ import annotations

import os
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_PLATFORM_TIMEZONE = "Europe/Tirane"
PLATFORM_TIMEZONE_NAME = (os.getenv("PLATFORM_TIMEZONE") or DEFAULT_PLATFORM_TIMEZONE).strip()
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _load_platform_timezone() -> ZoneInfo:
    try:
        return ZoneInfo(PLATFORM_TIMEZONE_NAME)
    except ZoneInfoNotFoundError:
        return ZoneInfo(DEFAULT_PLATFORM_TIMEZONE)


PLATFORM_TIMEZONE = _load_platform_timezone()


def as_utc(value: datetime) -> datetime:
    """Normalize stored datetimes to aware UTC.

    SQLite drops timezone metadata from SQLAlchemy DateTime values. Datetimes
    read back without tzinfo are therefore treated as the UTC values written by
    this application, not as server-local time.
    """

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def utc_isoformat(value: Optional[datetime]) -> Optional[str]:
    """Serialize an instant as ISO 8601 with an explicit UTC designator."""

    if value is None:
        return None
    return as_utc(value).isoformat().replace("+00:00", "Z")


def current_platform_year(now: Optional[datetime] = None) -> int:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(PLATFORM_TIMEZONE).year


def platform_year_utc_bounds(year: int) -> Tuple[datetime, datetime]:
    start = datetime(int(year), 1, 1, tzinfo=PLATFORM_TIMEZONE)
    end = datetime(int(year) + 1, 1, 1, tzinfo=PLATFORM_TIMEZONE)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def parse_platform_filter_datetime(value: str, *, end_exclusive: bool = False) -> datetime:
    """Parse an API date filter using the factory's business timezone.

    Date-only inputs describe a Tirana business day. Explicit offsets are
    respected. Offset-free date-times are interpreted in the platform timezone.
    The result is returned in UTC for comparison with stored UTC timestamps.
    """

    text_value = str(value or "").strip()
    if not text_value:
        raise ValueError("Date value is required")

    if _DATE_ONLY_RE.fullmatch(text_value):
        parsed_date = date.fromisoformat(text_value)
        local_value = datetime.combine(parsed_date, time.min, tzinfo=PLATFORM_TIMEZONE)
        if end_exclusive:
            local_value = local_value + timedelta(days=1)
        return local_value.astimezone(timezone.utc)

    normalized = text_value[:-1] + "+00:00" if text_value.endswith("Z") else text_value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=PLATFORM_TIMEZONE)
    return parsed.astimezone(timezone.utc)
