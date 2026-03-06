from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return number


def _to_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _pct_delta(current: float, previous: float) -> Optional[float]:
    if previous == 0:
        if current == 0:
            return 0.0
        return None
    return ((current - previous) / previous) * 100.0


def _severity_for_ratio(value: float, medium: float, high: float) -> Optional[str]:
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return None


def _severity_for_abs_delta(value: float, medium: float, high: float) -> Optional[str]:
    magnitude = abs(value)
    if magnitude >= high:
        return "high"
    if magnitude >= medium:
        return "medium"
    return None


def _format_period(period: Dict[str, Any]) -> str:
    start = str(period.get("start") or "").strip()
    end = str(period.get("end") or "").strip()
    if start and end:
        return f"{start} -> {end}"
    if start:
        return f"from {start}"
    if end:
        return f"until {end}"
    return "all available data"


def _make_signal(
    signal_id: str,
    signal_type: str,
    severity: str,
    title: str,
    explanation: str,
    metric_key: str,
    metric_value: Any,
    comparison_value: Any = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "id": signal_id,
        "type": signal_type,
        "severity": severity,
        "title": title,
        "explanation": explanation,
        "metricKey": metric_key,
        "metricValue": metric_value,
        "comparisonValue": comparison_value,
        "meta": meta or {},
    }


def _build_growth_signal(summary: Dict[str, Any], thresholds: Dict[str, float]) -> Optional[Dict[str, Any]]:
    compare_enabled = bool(summary.get("compare_enabled"))
    if not compare_enabled:
        return None

    current = summary.get("current") if isinstance(summary.get("current"), dict) else {}
    previous = summary.get("previous") if isinstance(summary.get("previous"), dict) else {}
    current_totals = current.get("totals") if isinstance(current.get("totals"), dict) else {}
    previous_totals = previous.get("totals") if isinstance(previous.get("totals"), dict) else {}

    tracked_metrics = [
        ("area_m2", "area"),
        ("units", "units"),
        ("orders", "orders"),
    ]

    strongest: Optional[Tuple[str, float, float, float]] = None
    for key, label in tracked_metrics:
        current_value = _to_float(current_totals.get(key))
        previous_value = _to_float(previous_totals.get(key))
        delta = _pct_delta(current_value, previous_value)
        if delta is None:
            continue
        if strongest is None or abs(delta) > abs(strongest[1]):
            strongest = (label, delta, current_value, previous_value)

    if strongest is None:
        return None

    metric_label, delta_pct, current_value, previous_value = strongest
    severity = _severity_for_abs_delta(
        delta_pct,
        thresholds.get("growth_delta_medium_pct", 10.0),
        thresholds.get("growth_delta_high_pct", 20.0),
    )
    if not severity:
        return None

    trend_word = "spike" if delta_pct >= 0 else "decline"
    title = f"{metric_label.capitalize()} {trend_word} ({delta_pct:+.1f}%)"
    period_text = _format_period(summary.get("period") if isinstance(summary.get("period"), dict) else {})
    previous_period_text = _format_period(summary.get("previous_period") if isinstance(summary.get("previous_period"), dict) else {})
    explanation = (
        f"{metric_label.capitalize()} changed by {delta_pct:+.1f}% in {period_text}, "
        f"compared with {previous_period_text}."
    )
    return _make_signal(
        signal_id=f"growth-{metric_label}",
        signal_type="growth_spike_decline",
        severity=severity,
        title=title,
        explanation=explanation,
        metric_key=metric_label,
        metric_value=current_value,
        comparison_value=previous_value,
        meta={
            "period": summary.get("period") or {},
            "comparisonPeriod": summary.get("previous_period") or {},
            "thresholdPct": {
                "medium": thresholds.get("growth_delta_medium_pct", 10.0),
                "high": thresholds.get("growth_delta_high_pct", 20.0),
            },
            "deltaPct": round(delta_pct, 3),
        },
    )


def _build_client_concentration_signal(summary: Dict[str, Any], thresholds: Dict[str, float]) -> Optional[Dict[str, Any]]:
    current = summary.get("current") if isinstance(summary.get("current"), dict) else {}
    top_clients = _to_list(current.get("top_clients"))
    totals = current.get("totals") if isinstance(current.get("totals"), dict) else {}
    total_area = _to_float(totals.get("area_m2"))
    if not top_clients or total_area <= 0:
        return None

    top_client = top_clients[0]
    client_name = str(top_client.get("client") or "Unknown client")
    top_area = _to_float(top_client.get("area_m2"))
    share_pct = _to_float(top_client.get("share_area_pct"))
    if share_pct <= 0:
        share_pct = (top_area / total_area) * 100.0 if total_area else 0.0

    severity = _severity_for_ratio(
        share_pct,
        thresholds.get("client_share_medium_pct", 35.0),
        thresholds.get("client_share_high_pct", 45.0),
    )
    if not severity:
        return None

    title = f"Client concentration: {client_name} at {share_pct:.1f}%"
    explanation = (
        f"{client_name} contributes {share_pct:.1f}% of total area ({top_area:.3f} m2) "
        f"in the selected period."
    )
    return _make_signal(
        signal_id="client-concentration",
        signal_type="client_concentration",
        severity=severity,
        title=title,
        explanation=explanation,
        metric_key="top_client_area_share_pct",
        metric_value=round(share_pct, 3),
        comparison_value=100.0 - round(share_pct, 3),
        meta={
            "client": client_name,
            "period": summary.get("period") or {},
            "thresholdPct": {
                "medium": thresholds.get("client_share_medium_pct", 35.0),
                "high": thresholds.get("client_share_high_pct", 45.0),
            },
        },
    )


def _build_glass_mix_shift_signal(summary: Dict[str, Any], thresholds: Dict[str, float]) -> Optional[Dict[str, Any]]:
    compare_enabled = bool(summary.get("compare_enabled"))
    if not compare_enabled:
        return None

    current = summary.get("current") if isinstance(summary.get("current"), dict) else {}
    previous = summary.get("previous") if isinstance(summary.get("previous"), dict) else {}
    current_types = _to_list(current.get("top_types"))
    previous_types = _to_list(previous.get("top_types"))
    if not current_types or not previous_types:
        return None

    current_share: Dict[str, float] = {}
    previous_share: Dict[str, float] = {}

    for entry in current_types:
        type_name = str(entry.get("type") or "").strip()
        if not type_name:
            continue
        current_share[type_name] = _to_float(entry.get("share_area_pct"))
    for entry in previous_types:
        type_name = str(entry.get("type") or "").strip()
        if not type_name:
            continue
        previous_share[type_name] = _to_float(entry.get("share_area_pct"))

    all_types = set(current_share.keys()) | set(previous_share.keys())
    if not all_types:
        return None

    max_type = None
    max_shift = 0.0
    for type_name in all_types:
        shift = current_share.get(type_name, 0.0) - previous_share.get(type_name, 0.0)
        if abs(shift) > abs(max_shift):
            max_shift = shift
            max_type = type_name

    if not max_type:
        return None

    severity = _severity_for_abs_delta(
        max_shift,
        thresholds.get("mix_shift_medium_pct", 10.0),
        thresholds.get("mix_shift_high_pct", 16.0),
    )
    if not severity:
        return None

    direction = "up" if max_shift >= 0 else "down"
    title = f"Glass mix shift: {max_type} {direction} {abs(max_shift):.1f}pp"
    explanation = (
        f"{max_type} changed by {max_shift:+.1f} percentage points in area share "
        f"versus the previous comparison period."
    )
    return _make_signal(
        signal_id="glass-mix-shift",
        signal_type="glass_type_mix_shift",
        severity=severity,
        title=title,
        explanation=explanation,
        metric_key="glass_type_share_delta_pp",
        metric_value=round(max_shift, 3),
        comparison_value=0,
        meta={
            "type": max_type,
            "period": summary.get("period") or {},
            "comparisonPeriod": summary.get("previous_period") or {},
            "thresholdPctPoints": {
                "medium": thresholds.get("mix_shift_medium_pct", 10.0),
                "high": thresholds.get("mix_shift_high_pct", 16.0),
            },
        },
    )


def _build_dimension_pattern_signal(summary: Dict[str, Any], thresholds: Dict[str, float]) -> Optional[Dict[str, Any]]:
    current = summary.get("current") if isinstance(summary.get("current"), dict) else {}
    dimensions = _to_list(current.get("top_dimensions"))
    if not dimensions:
        return None

    candidate = None
    for entry in dimensions:
        qty = _to_float(entry.get("qty"))
        orders = _to_float(entry.get("orders"))
        clients = _to_float(entry.get("clients"))
        repeat_score = _to_float(entry.get("repeat_score"))
        if orders < thresholds.get("dimension_orders_min", 3.0):
            continue
        if clients < thresholds.get("dimension_clients_min", 2.0):
            continue
        if qty < thresholds.get("dimension_qty_min", 8.0):
            continue
        if candidate is None or repeat_score > _to_float(candidate.get("repeat_score")):
            candidate = entry

    if not candidate:
        return None

    repeat_score = _to_float(candidate.get("repeat_score"))
    severity = _severity_for_ratio(
        repeat_score,
        thresholds.get("dimension_repeat_medium", 15.0),
        thresholds.get("dimension_repeat_high", 22.0),
    )
    if not severity:
        severity = "medium"

    dimension_text = str(candidate.get("dimension") or "dimension bucket")
    title = f"Repeated dimension pattern: {dimension_text}"
    explanation = (
        f"{dimension_text} repeats across {int(_to_float(candidate.get('orders')))} orders and "
        f"{int(_to_float(candidate.get('clients')))} clients."
    )
    return _make_signal(
        signal_id="dimension-pattern",
        signal_type="repeated_dimension_pattern",
        severity=severity,
        title=title,
        explanation=explanation,
        metric_key="dimension_repeat_score",
        metric_value=round(repeat_score, 3),
        comparison_value=None,
        meta={
            "dimension": dimension_text,
            "qty": _to_float(candidate.get("qty")),
            "orders": _to_float(candidate.get("orders")),
            "clients": _to_float(candidate.get("clients")),
            "batchingOpportunity": bool(candidate.get("batching_opportunity")),
            "thresholds": {
                "ordersMin": thresholds.get("dimension_orders_min", 3.0),
                "clientsMin": thresholds.get("dimension_clients_min", 2.0),
                "qtyMin": thresholds.get("dimension_qty_min", 8.0),
            },
        },
    )


def _build_large_order_signal(summary: Dict[str, Any], thresholds: Dict[str, float]) -> Optional[Dict[str, Any]]:
    current = summary.get("current") if isinstance(summary.get("current"), dict) else {}
    contributions = _to_list(current.get("order_contributions"))
    if not contributions:
        return None
    top_order = contributions[0]
    share_pct = _to_float(top_order.get("area_share_pct"))
    if share_pct <= 0:
        return None

    severity = _severity_for_ratio(
        share_pct,
        thresholds.get("order_share_medium_pct", 18.0),
        thresholds.get("order_share_high_pct", 25.0),
    )
    if not severity:
        return None

    order_label = str(top_order.get("order_id") or "Order")
    title = f"Large order contribution: {order_label} at {share_pct:.1f}%"
    explanation = (
        f"{order_label} contributes {share_pct:.1f}% of total area in the selected period, "
        f"which is above the configured concentration threshold."
    )
    return _make_signal(
        signal_id="large-order",
        signal_type="unusually_large_order_contribution",
        severity=severity,
        title=title,
        explanation=explanation,
        metric_key="order_area_share_pct",
        metric_value=round(share_pct, 3),
        comparison_value=thresholds.get("order_share_medium_pct", 18.0),
        meta={
            "orderId": order_label,
            "client": top_order.get("client"),
            "period": summary.get("period") or {},
            "thresholdPct": {
                "medium": thresholds.get("order_share_medium_pct", 18.0),
                "high": thresholds.get("order_share_high_pct", 25.0),
            },
        },
    )


def generate_analysis_signals(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = summary if isinstance(summary, dict) else {}
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    threshold_overrides = meta.get("thresholds") if isinstance(meta.get("thresholds"), dict) else {}

    thresholds: Dict[str, float] = {
        "growth_delta_medium_pct": _to_float(threshold_overrides.get("growth_delta_medium_pct"), 10.0),
        "growth_delta_high_pct": _to_float(threshold_overrides.get("growth_delta_high_pct"), 20.0),
        "client_share_medium_pct": _to_float(threshold_overrides.get("client_share_medium_pct"), 35.0),
        "client_share_high_pct": _to_float(threshold_overrides.get("client_share_high_pct"), 45.0),
        "mix_shift_medium_pct": _to_float(threshold_overrides.get("mix_shift_medium_pct"), 10.0),
        "mix_shift_high_pct": _to_float(threshold_overrides.get("mix_shift_high_pct"), 16.0),
        "dimension_orders_min": _to_float(threshold_overrides.get("dimension_orders_min"), 3.0),
        "dimension_clients_min": _to_float(threshold_overrides.get("dimension_clients_min"), 2.0),
        "dimension_qty_min": _to_float(threshold_overrides.get("dimension_qty_min"), 8.0),
        "dimension_repeat_medium": _to_float(threshold_overrides.get("dimension_repeat_medium"), 15.0),
        "dimension_repeat_high": _to_float(threshold_overrides.get("dimension_repeat_high"), 22.0),
        "order_share_medium_pct": _to_float(threshold_overrides.get("order_share_medium_pct"), 18.0),
        "order_share_high_pct": _to_float(threshold_overrides.get("order_share_high_pct"), 25.0),
    }

    builders = [
        _build_growth_signal,
        _build_client_concentration_signal,
        _build_glass_mix_shift_signal,
        _build_dimension_pattern_signal,
        _build_large_order_signal,
    ]

    signals: List[Dict[str, Any]] = []
    for builder in builders:
        signal = builder(payload, thresholds)
        if signal:
            signals.append(signal)

    severity_rank = {"high": 3, "medium": 2, "low": 1}
    signals.sort(
        key=lambda item: (
            -severity_rank.get(str(item.get("severity")), 0),
            str(item.get("type") or ""),
            str(item.get("id") or ""),
        )
    )
    return signals

