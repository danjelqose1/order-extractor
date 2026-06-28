from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from analysis_signals import generate_analysis_signals


ANALYTICS_STATUSES = ("approved", "in_production", "completed")
_DIMENSION_RE = re.compile(
    r"(-?\d+(?:[.,]\d+)?)\s*[x×]\s*(-?\d+(?:[.,]\d+)?)",
    flags=re.IGNORECASE,
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return parsed


def _date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def _parse_requested_date(value: Optional[str], field_name: str) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name}. Use YYYY-MM-DD.") from exc


def _dimension(value: Any) -> Optional[Tuple[float, float]]:
    match = _DIMENSION_RE.search(str(value or ""))
    if not match:
        return None
    width = _number(match.group(1), -1)
    height = _number(match.group(2), -1)
    if width <= 0 or height <= 0:
        return None
    return width, height


def _format_dimension_value(value: float) -> str:
    rounded = round(value, 2)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def _row_total_area(row: Dict[str, Any]) -> float:
    direct = _number(row.get("area"), 0.0)
    if direct > 0:
        return direct
    parsed = _dimension(row.get("dimension"))
    if not parsed:
        return 0.0
    quantity = max(0.0, _number(row.get("quantity"), 0.0))
    return (parsed[0] * parsed[1] * quantity) / 1_000_000


def _order_label(order: Dict[str, Any]) -> str:
    numbers = order.get("order_numbers")
    if isinstance(numbers, list):
        cleaned = [str(item).strip() for item in numbers if str(item).strip()]
        if cleaned:
            return ", ".join(cleaned)
    return str(order.get("order_number") or order.get("id") or "—")


def _percent_delta(current: float, previous: float) -> Optional[float]:
    if previous == 0:
        return 0.0 if current == 0 else None
    return ((current - previous) / previous) * 100.0


def _previous_period(start: date, end: date) -> Tuple[date, date]:
    days = max(1, (end - start).days + 1)
    previous_end = start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=days - 1)
    return previous_start, previous_end


def _build_records(
    orders: Sequence[Dict[str, Any]],
    *,
    start: Optional[date],
    end: Optional[date],
    client: str,
    glass_type: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    client_filter = client.casefold()
    type_filter = glass_type.casefold()
    for order in orders:
        created = _date(order.get("created_at"))
        if start and (not created or created < start):
            continue
        if end and (not created or created > end):
            continue
        order_client = str(order.get("client") or "—").strip() or "—"
        if client_filter and order_client.casefold() != client_filter:
            continue
        rows: List[Dict[str, Any]] = []
        for row in order.get("rows") or []:
            if not isinstance(row, dict):
                continue
            row_type = str(row.get("type") or "—").strip() or "—"
            if type_filter and row_type.casefold() != type_filter:
                continue
            quantity = max(0, int(_number(row.get("quantity"), 0)))
            rows.append(
                {
                    "type": row_type,
                    "dimension": str(row.get("dimension") or "").strip(),
                    "quantity": quantity,
                    "area_m2": _row_total_area(row),
                }
            )
        if not rows:
            continue
        records.append(
            {
                "order_id": _order_label(order),
                "client": order_client,
                "created": created,
                "rows": rows,
                "units": sum(row["quantity"] for row in rows),
                "area_m2": sum(row["area_m2"] for row in rows),
            }
        )
    return records


def _dimension_buckets(
    rows: Iterable[Dict[str, Any]],
    tolerance_mm: float,
    orientation_agnostic: bool,
) -> List[Dict[str, Any]]:
    buckets: List[Dict[str, Any]] = []
    for row in rows:
        parsed = _dimension(row.get("dimension"))
        if not parsed:
            continue
        width, height = parsed
        if orientation_agnostic and width > height:
            width, height = height, width
        quantity = max(0, int(_number(row.get("quantity"), 0)))
        if quantity <= 0:
            continue
        bucket = next(
            (
                item
                for item in buckets
                if abs(item["width_ref"] - width) <= tolerance_mm
                and abs(item["height_ref"] - height) <= tolerance_mm
            ),
            None,
        )
        if bucket is None:
            bucket = {
                "width_ref": width,
                "height_ref": height,
                "width_values": [],
                "height_values": [],
                "qty": 0,
                "order_ids": set(),
                "clients": set(),
            }
            buckets.append(bucket)
        bucket["width_values"].append(width)
        bucket["height_values"].append(height)
        bucket["qty"] += quantity
        bucket["order_ids"].add(str(row.get("order_id") or "—"))
        bucket["clients"].add(str(row.get("client") or "—"))

    result: List[Dict[str, Any]] = []
    for bucket in buckets:
        width = sum(bucket["width_values"]) / len(bucket["width_values"])
        height = sum(bucket["height_values"]) / len(bucket["height_values"])
        order_count = len(bucket["order_ids"] - {"—"})
        client_count = len(bucket["clients"] - {"—"})
        quantity = int(bucket["qty"])
        result.append(
            {
                "dimension": f"{_format_dimension_value(width)} × {_format_dimension_value(height)}",
                "qty": quantity,
                "orders": order_count,
                "clients": client_count,
                "repeatScore": round(
                    quantity * (1 + (order_count / 5) + (client_count / 5)),
                    2,
                ),
                "orderSpread": order_count,
                "clientSpread": client_count,
                "batchingOpportunity": order_count >= 3 and client_count >= 2 and quantity >= 8,
                "sortQty": quantity,
            }
        )
    return sorted(result, key=lambda item: (-item["qty"], item["dimension"]))[:20]


def _compute_period(
    records: Sequence[Dict[str, Any]],
    start: Optional[date],
    end: Optional[date],
) -> Dict[str, Any]:
    totals: Dict[str, Any] = {
        "orders": len(records),
        "units": 0,
        "area_m2": 0.0,
        "avg_area_per_order": 0.0,
        "avg_units_per_order": 0.0,
        "avg_orders_per_day": 0.0,
        "distinct_clients": 0,
        "distinct_types": 0,
    }
    client_map: Dict[str, Dict[str, Any]] = {}
    type_map: Dict[str, Dict[str, Any]] = {}
    daily_map: Dict[date, Dict[str, Any]] = {}
    dimension_rows: List[Dict[str, Any]] = []
    contributions: List[Dict[str, Any]] = []

    for record in records:
        units = int(record["units"])
        area = float(record["area_m2"])
        totals["units"] += units
        totals["area_m2"] += area
        client = record["client"]
        client_entry = client_map.setdefault(
            client,
            {"client": client, "orders": 0, "units": 0, "area_m2": 0.0},
        )
        client_entry["orders"] += 1
        client_entry["units"] += units
        client_entry["area_m2"] += area
        contributions.append(
            {
                "order_id": record["order_id"],
                "client": client,
                "area_m2": area,
                "units": units,
            }
        )
        created = record.get("created")
        if created:
            day = daily_map.setdefault(
                created,
                {"date": created.isoformat(), "orders": 0, "units": 0, "area_m2": 0.0},
            )
            day["orders"] += 1
            day["units"] += units
            day["area_m2"] += area
        for row in record["rows"]:
            row_type = row["type"]
            type_entry = type_map.setdefault(
                row_type,
                {"type": row_type, "lines": 0, "units": 0, "area_m2": 0.0},
            )
            type_entry["lines"] += 1
            type_entry["units"] += row["quantity"]
            type_entry["area_m2"] += row["area_m2"]
            dimension_rows.append(
                {
                    **row,
                    "order_id": record["order_id"],
                    "client": client,
                }
            )

    series_start = start
    series_end = end
    dated_records = [record["created"] for record in records if record.get("created")]
    if not series_start and not series_end and dated_records:
        series_start = min(dated_records)
        series_end = max(dated_records)
    if series_start and series_end:
        cursor = series_start
        while cursor <= series_end:
            daily_map.setdefault(
                cursor,
                {"date": cursor.isoformat(), "orders": 0, "units": 0, "area_m2": 0.0},
            )
            cursor += timedelta(days=1)

    daily = []
    for day_key in sorted(daily_map):
        item = daily_map[day_key]
        orders_count = item["orders"]
        daily.append(
            {
                **item,
                "label": f"{day_key.strftime('%b')} {day_key.day}",
                "avgAreaPerOrder": item["area_m2"] / orders_count if orders_count else 0,
                "avgUnitsPerOrder": item["units"] / orders_count if orders_count else 0,
            }
        )

    totals["area_m2"] = round(totals["area_m2"], 6)
    totals["avg_area_per_order"] = (
        totals["area_m2"] / totals["orders"] if totals["orders"] else 0
    )
    totals["avg_units_per_order"] = (
        totals["units"] / totals["orders"] if totals["orders"] else 0
    )
    totals["avg_orders_per_day"] = totals["orders"] / len(daily) if daily else 0
    totals["distinct_clients"] = len(client_map)
    totals["distinct_types"] = len(type_map)

    top_clients = sorted(
        client_map.values(),
        key=lambda item: (-item["area_m2"], item["client"].casefold()),
    )
    top_types = sorted(
        type_map.values(),
        key=lambda item: (-item["area_m2"], item["type"].casefold()),
    )
    for item in top_clients:
        item["area_m2"] = round(item["area_m2"], 6)
        item["share_area_pct"] = (
            (item["area_m2"] / totals["area_m2"]) * 100 if totals["area_m2"] else 0
        )
    for item in top_types:
        item["area_m2"] = round(item["area_m2"], 6)
        item["share_area_pct"] = (
            (item["area_m2"] / totals["area_m2"]) * 100 if totals["area_m2"] else 0
        )
    for item in contributions:
        item["area_m2"] = round(item["area_m2"], 6)
        item["area_share_pct"] = (
            (item["area_m2"] / totals["area_m2"]) * 100 if totals["area_m2"] else 0
        )
    contributions.sort(key=lambda item: (-item["area_m2"], str(item["order_id"])))

    return {
        "period": {
            "start": start.isoformat() if start else "",
            "end": end.isoformat() if end else "",
        },
        "totals": totals,
        "topClients": top_clients,
        "topTypes": top_types,
        "topDimensions": [],
        "orderContributions": contributions,
        "daily": daily,
    }


def _attach_growth(
    rows: Sequence[Dict[str, Any]],
    previous_rows: Sequence[Dict[str, Any]],
    key: str,
) -> None:
    previous = {str(item.get(key)): item for item in previous_rows}
    for item in rows:
        baseline = previous.get(str(item.get(key)))
        item["growth_pct"] = _percent_delta(
            _number(item.get("area_m2")),
            _number(baseline.get("area_m2")) if baseline else 0,
        )


def _signals_summary(
    current: Dict[str, Any],
    previous: Optional[Dict[str, Any]],
    period: Dict[str, str],
    previous_period: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    def dimensions(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "dimension": item.get("dimension"),
                "qty": item.get("qty"),
                "orders": item.get("orders"),
                "clients": item.get("clients"),
                "repeat_score": item.get("repeatScore"),
                "order_spread": item.get("orderSpread"),
                "client_spread": item.get("clientSpread"),
                "batching_opportunity": bool(item.get("batchingOpportunity")),
            }
            for item in (payload or {}).get("topDimensions", [])
        ]

    return {
        "compare_enabled": previous is not None,
        "period": period,
        "previous_period": previous_period or {},
        "current": {
            "totals": current.get("totals") or {},
            "top_clients": current.get("topClients") or [],
            "top_types": current.get("topTypes") or [],
            "top_dimensions": dimensions(current),
            "order_contributions": current.get("orderContributions") or [],
            "daily": current.get("daily") or [],
        },
        "previous": {
            "totals": (previous or {}).get("totals") or {},
            "top_clients": (previous or {}).get("topClients") or [],
            "top_types": (previous or {}).get("topTypes") or [],
            "top_dimensions": dimensions(previous),
            "order_contributions": (previous or {}).get("orderContributions") or [],
            "daily": (previous or {}).get("daily") or [],
        },
        "meta": {
            "thresholds": {
                "growth_delta_medium_pct": 10,
                "growth_delta_high_pct": 20,
                "client_share_medium_pct": 35,
                "client_share_high_pct": 45,
                "mix_shift_medium_pct": 10,
                "mix_shift_high_pct": 16,
            }
        },
    }


def build_analysis_summary(
    orders: Sequence[Dict[str, Any]],
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    compare_previous: bool = True,
    client: str = "",
    glass_type: str = "",
    tolerance_mm: float = 1.0,
    orientation_agnostic: bool = True,
    all_time: bool = False,
) -> Dict[str, Any]:
    safe_tolerance = max(0.0, min(float(tolerance_mm), 5.0))
    effective_compare = bool(compare_previous and not all_time)
    available_dates = sorted(
        item for item in (_date(order.get("created_at")) for order in orders) if item
    )
    requested_start = _parse_requested_date(start_date, "start_date")
    requested_end = _parse_requested_date(end_date, "end_date")
    if requested_start and requested_end and requested_start > requested_end:
        requested_start, requested_end = requested_end, requested_start

    if all_time:
        current_start = None
        current_end = None
    elif requested_start or requested_end:
        current_start = requested_start
        current_end = requested_end
    else:
        current_end = available_dates[-1] if available_dates else datetime.now(timezone.utc).date()
        current_start = current_end - timedelta(days=29)

    current_records = _build_records(
        orders,
        start=current_start,
        end=current_end,
        client=client,
        glass_type=glass_type,
    )
    current = _compute_period(current_records, current_start, current_end)
    current["topDimensions"] = _dimension_buckets(
        (
            {
                **row,
                "order_id": record["order_id"],
                "client": record["client"],
            }
            for record in current_records
            for row in record["rows"]
        ),
        safe_tolerance,
        orientation_agnostic,
    )

    previous: Optional[Dict[str, Any]] = None
    previous_period: Optional[Dict[str, str]] = None
    if effective_compare and current_start and current_end:
        previous_start, previous_end = _previous_period(current_start, current_end)
        previous_records = _build_records(
            orders,
            start=previous_start,
            end=previous_end,
            client=client,
            glass_type=glass_type,
        )
        previous = _compute_period(previous_records, previous_start, previous_end)
        previous["topDimensions"] = _dimension_buckets(
            (
                {
                    **row,
                    "order_id": record["order_id"],
                    "client": record["client"],
                }
                for record in previous_records
                for row in record["rows"]
            ),
            safe_tolerance,
            orientation_agnostic,
        )
        previous_period = {
            "start": previous_start.isoformat(),
            "end": previous_end.isoformat(),
        }

    _attach_growth(current["topClients"], (previous or {}).get("topClients", []), "client")
    _attach_growth(current["topTypes"], (previous or {}).get("topTypes", []), "type")

    available_clients: Set[str] = set()
    available_types: Set[str] = set()
    for order in orders:
        available_clients.add(str(order.get("client") or "—").strip() or "—")
        for row in order.get("rows") or []:
            if isinstance(row, dict):
                available_types.add(str(row.get("type") or "—").strip() or "—")

    period = current["period"]
    insights = generate_analysis_signals(
        _signals_summary(current, previous, period, previous_period)
    )
    computed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "filters": {
            "startDate": period["start"],
            "endDate": period["end"],
            "compareToPrevious": effective_compare,
            "client": client,
            "glassType": glass_type,
            "tolerance": safe_tolerance,
            "orientationAgnostic": bool(orientation_agnostic),
            "allTime": bool(all_time),
        },
        "period": period,
        "previousPeriod": previous_period,
        "current": current,
        "previous": previous,
        "kpis": current["totals"],
        "charts": {
            "daily": current["daily"],
            "topClients": current["topClients"],
            "topGlassTypes": current["topTypes"],
        },
        "topClients": current["topClients"],
        "topGlassTypes": current["topTypes"],
        "topDimensions": current["topDimensions"],
        "insights": insights,
        "availableFilters": {
            "clients": sorted(available_clients, key=str.casefold),
            "glassTypes": sorted(available_types, key=str.casefold),
        },
        "computedAt": computed_at,
        "cacheStatus": "fresh",
        "sourceStatuses": list(ANALYTICS_STATUSES),
    }
