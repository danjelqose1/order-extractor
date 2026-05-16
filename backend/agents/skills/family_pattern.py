from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple


DIMENSION_RE = re.compile(r"^(\d{2,5})x(\d{2,5})$")
FAMILY_MISMATCH_CODE = "POSSIBLE_DIMENSION_FAMILY_MISMATCH"
FAMILY_MISMATCH_THRESHOLD = 0.65


def _parse_dimension(value: Any) -> Tuple[Optional[int], Optional[int], str]:
    if value is None:
        return None, None, ""
    raw = str(value)
    token = raw.strip().lower().replace("×", "x")
    token = re.sub(r"\s+", "", token)
    match = DIMENSION_RE.match(token)
    if not match:
        return None, None, raw
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        return None, None, raw
    return width, height, raw


def _dimension_key(value: Any) -> str:
    width, height, _raw = _parse_dimension(value)
    if width is None or height is None:
        return ""
    return f"{width}x{height}"


def _parse_quantity(value: Any) -> int:
    try:
        quantity = int(float(value))
    except (TypeError, ValueError):
        return 1
    return quantity if quantity > 0 else 1


def _area_value(row: Dict[str, Any]) -> Optional[float]:
    for key in ("area", "extracted_area", "area_m2"):
        if key not in row:
            continue
        try:
            area = float(str(row.get(key)).replace(",", "."))
        except (TypeError, ValueError):
            continue
        return area if area > 0 else None
    return None


def _dimension_area(dimension: str, quantity: int = 1) -> Optional[float]:
    width, height, _raw = _parse_dimension(dimension)
    if width is None or height is None:
        return None
    return (width * height * max(1, quantity)) / 1_000_000


def _area_similarity(current_dimension: str, candidate_dimension: str, quantity: int) -> Tuple[bool, Optional[float]]:
    current_area = _dimension_area(current_dimension, quantity)
    candidate_area = _dimension_area(candidate_dimension, quantity)
    if current_area is None or candidate_area is None:
        return False, None
    larger = max(current_area, candidate_area)
    if larger <= 0:
        return False, None
    ratio = abs(current_area - candidate_area) / larger
    return ratio <= 0.055, ratio


def _text_similarity(a: str, b: str) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (0 if char_a == char_b else 1),
                )
            )
        previous = current
    return previous[-1]


def _glass_type(row: Dict[str, Any]) -> str:
    return re.sub(r"\s+", " ", str(row.get("type") or row.get("glass_type") or "").strip().lower())


def _position_prefix(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.split(r"[-/]", text, maxsplit=1)[0]


def _row_identity(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("row_id") or row.get("id") or row.get("_rid") or ""),
        str(row.get("position") or ""),
        str(row.get("order_number") or ""),
    )


def _same_source_row(row: Dict[str, Any], candidate_row: Dict[str, Any]) -> bool:
    row_id, position, order = _row_identity(row)
    candidate_id, candidate_position, candidate_order = _row_identity(candidate_row)
    if row_id and candidate_id and row_id == candidate_id:
        return True
    return bool(position and position == candidate_position and (not order or not candidate_order or order == candidate_order))


def _collect_rows(
    nearby_rows: Optional[Iterable[Dict[str, Any]]],
    order_rows: Optional[Iterable[Dict[str, Any]]],
    order_context: Optional[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], str]]:
    rows: List[Tuple[Dict[str, Any], str]] = []
    seen: set[Tuple[str, str, str, str]] = set()

    def add(source_rows: Optional[Iterable[Dict[str, Any]]], source: str) -> None:
        if not source_rows:
            return
        for item in source_rows:
            if not isinstance(item, dict):
                continue
            dimension = _dimension_key(item.get("dimension"))
            key = (*_row_identity(item), dimension)
            if key in seen:
                continue
            seen.add(key)
            rows.append((deepcopy(item), source))

    add(nearby_rows, "nearby_rows")
    add(order_rows, "order_rows")
    if isinstance(order_context, dict):
        add(order_context.get("rows_before"), "nearby_rows")
        add(order_context.get("rows_after"), "nearby_rows")
        add(order_context.get("nearby_rows"), "nearby_rows")
        add(order_context.get("order_rows"), "order_rows")
        add(order_context.get("original_order_rows"), "original_order_rows")
        add(order_context.get("raw_order_rows"), "original_order_rows")
    return rows


def _candidate_record(
    row: Dict[str, Any],
    candidate_row: Dict[str, Any],
    source: str,
    current_dimension: str,
    candidate_dimension: str,
    quantity: int,
) -> Optional[Dict[str, Any]]:
    current_width, current_height, _raw_current = _parse_dimension(current_dimension)
    candidate_width, candidate_height, _raw_candidate = _parse_dimension(candidate_dimension)
    if (
        current_width is None
        or current_height is None
        or candidate_width is None
        or candidate_height is None
        or current_dimension == candidate_dimension
    ):
        return None

    width_delta = abs(current_width - candidate_width)
    height_delta = abs(current_height - candidate_height)
    both_sides_close = width_delta <= 10 and height_delta <= 10
    one_side_near = width_delta <= 5 or height_delta <= 5
    area_close, area_delta_ratio = _area_similarity(current_dimension, candidate_dimension, quantity)
    edit_distance = _text_similarity(current_dimension.replace("x", ""), candidate_dimension.replace("x", ""))
    if not ((both_sides_close and area_close) or (one_side_near and edit_distance <= 3 and area_close)):
        return None

    current_type = _glass_type(row)
    candidate_type = _glass_type(candidate_row)
    same_glass_type = bool(current_type and current_type == candidate_type)
    same_prefix = bool(_position_prefix(row.get("position")) and _position_prefix(row.get("position")) == _position_prefix(candidate_row.get("position")))
    same_source = source == "original_order_rows" and _same_source_row(row, candidate_row)

    confidence = 0.28
    if both_sides_close:
        confidence += 0.18
    if one_side_near:
        confidence += 0.08
    if edit_distance <= 2:
        confidence += 0.1
    elif edit_distance <= 3:
        confidence += 0.06
    if area_delta_ratio is not None:
        if area_delta_ratio <= 0.02:
            confidence += 0.12
        elif area_delta_ratio <= 0.04:
            confidence += 0.08
        else:
            confidence += 0.04
    if same_glass_type:
        confidence += 0.1
    if source == "nearby_rows":
        confidence += 0.08
    if same_prefix:
        confidence += 0.04
    if source == "original_order_rows":
        confidence += 0.1
    if same_source:
        confidence += 0.1
    if not same_glass_type and source != "nearby_rows" and not same_source:
        confidence -= 0.07

    return {
        "dimension": candidate_dimension,
        "source": source,
        "position": candidate_row.get("position"),
        "type": candidate_row.get("type") or candidate_row.get("glass_type"),
        "width_delta": width_delta,
        "height_delta": height_delta,
        "edit_distance": edit_distance,
        "area_delta_ratio": round(area_delta_ratio, 4) if area_delta_ratio is not None else None,
        "same_glass_type": same_glass_type,
        "same_position_prefix": same_prefix,
        "same_original_row": same_source,
        "confidence": max(0.0, min(0.85, confidence)),
    }


def analyze_dimension_family(
    row: Dict[str, Any],
    nearby_rows: Optional[List[Dict[str, Any]]] = None,
    order_rows: Optional[List[Dict[str, Any]]] = None,
    order_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    working_row = deepcopy(row or {})
    current_dimension = _dimension_key(working_row.get("dimension"))
    trace: List[str] = ["Checked dimension family context"]
    if not current_dimension:
        return {
            "success": False,
            "suspicious": False,
            "target_field": "dimension",
            "original_value": working_row.get("dimension"),
            "suggested_value": None,
            "confidence": 0.0,
            "method": "family_pattern_repair",
            "reasoning": "Current dimension is not parseable as width x height.",
            "evidence": {"candidate_count": 0},
            "trace": trace,
            "safe_to_auto_apply": False,
        }

    quantity = _parse_quantity(working_row.get("quantity"))
    rows = _collect_rows(nearby_rows, order_rows, order_context)
    grouped: Dict[str, Dict[str, Any]] = {}
    for candidate_row, source in rows:
        candidate_dimension = _dimension_key(candidate_row.get("dimension"))
        record = _candidate_record(working_row, candidate_row, source, current_dimension, candidate_dimension, quantity)
        if not record:
            continue
        entry = grouped.setdefault(
            candidate_dimension,
            {
                "dimension": candidate_dimension,
                "support_count": 0,
                "sources": [],
                "support_rows": [],
                "confidence": 0.0,
                "same_glass_type_count": 0,
                "nearby_count": 0,
            },
        )
        entry["support_count"] += 1
        if record["source"] not in entry["sources"]:
            entry["sources"].append(record["source"])
        if record["same_glass_type"]:
            entry["same_glass_type_count"] += 1
        if record["source"] == "nearby_rows":
            entry["nearby_count"] += 1
        if len(entry["support_rows"]) < 4:
            entry["support_rows"].append(record)
        entry["confidence"] = max(float(entry["confidence"]), float(record["confidence"]))

    if not grouped:
        return {
            "success": False,
            "suspicious": False,
            "target_field": "dimension",
            "original_value": working_row.get("dimension"),
            "suggested_value": None,
            "confidence": 0.0,
            "method": "family_pattern_repair",
            "reasoning": "No near dimension family candidate was found.",
            "evidence": {"candidate_count": 0},
            "trace": trace + ["No near family candidates found"],
            "safe_to_auto_apply": False,
        }

    candidates: List[Dict[str, Any]] = []
    for entry in grouped.values():
        confidence = float(entry["confidence"])
        if entry["support_count"] >= 2:
            confidence += 0.05
        if entry["same_glass_type_count"] >= 2:
            confidence += 0.03
        if "original_order_rows" in entry["sources"]:
            confidence += 0.04
        if entry["nearby_count"] == 0 and "original_order_rows" not in entry["sources"]:
            confidence -= 0.06
        if entry["nearby_count"] == 0 and "original_order_rows" not in entry["sources"] and entry["support_count"] < 2:
            confidence -= 0.18
        entry["confidence"] = round(max(0.0, min(0.85, confidence)), 2)
        candidates.append(entry)

    candidates.sort(key=lambda item: (-float(item["confidence"]), -int(item["support_count"]), item["dimension"]))
    best = candidates[0]
    if len(candidates) > 1 and abs(float(best["confidence"]) - float(candidates[1]["confidence"])) < 0.08:
        return {
            "success": False,
            "suspicious": False,
            "target_field": "dimension",
            "original_value": working_row.get("dimension"),
            "suggested_value": None,
            "confidence": 0.0,
            "method": "family_pattern_repair",
            "reasoning": "Multiple near dimension family candidates were similarly plausible.",
            "evidence": {"candidate_count": len(candidates), "candidates": candidates},
            "trace": trace + ["Multiple weak family candidates found; no warning emitted"],
            "safe_to_auto_apply": False,
        }

    confidence = float(best["confidence"])
    if confidence < FAMILY_MISMATCH_THRESHOLD:
        return {
            "success": False,
            "suspicious": False,
            "target_field": "dimension",
            "original_value": working_row.get("dimension"),
            "suggested_value": None,
            "confidence": confidence,
            "method": "family_pattern_repair",
            "reasoning": "Near candidates were present, but support was too weak for a family warning.",
            "evidence": {"candidate_count": len(candidates), "candidates": candidates},
            "trace": trace + [f"Best family candidate confidence {confidence:.2f} below threshold"],
            "safe_to_auto_apply": False,
        }

    support = best["support_rows"][0] if best.get("support_rows") else {}
    trace.extend(
        [
            f"Found near family candidate {best['dimension']}",
            f"Width delta {support.get('width_delta')}, height delta {support.get('height_delta')}",
            "Area is close between current and candidate dimensions",
        ]
    )
    if best["same_glass_type_count"]:
        trace.append("Same glass type block supports the candidate")
    if "original_order_rows" in best["sources"]:
        trace.append("Original extracted row data supports the candidate")
    trace.append(f"Family pattern confidence {confidence:.2f}")

    return {
        "success": True,
        "suspicious": True,
        "target_field": "dimension",
        "original_value": working_row.get("dimension"),
        "suggested_value": best["dimension"],
        "confidence": confidence,
        "method": "family_pattern_repair",
        "recommended_action": "PATTERN_REPAIR",
        "reasoning": (
            f"{current_dimension} is valid, but {best['dimension']} is a near dimension-family candidate "
            "supported by the order context."
        ),
        "evidence": {
            "candidate_count": len(candidates),
            "selected_candidate": best,
            "candidates": candidates,
        },
        "trace": trace,
        "safe_to_auto_apply": False,
    }


def family_pattern_diagnostic_issue() -> Dict[str, str]:
    return {
        "code": FAMILY_MISMATCH_CODE,
        "field": "dimension",
        "message": "Dimension is plausible but differs from a nearby/order family pattern.",
        "recommended_action": "PATTERN_REPAIR",
    }


def attach_family_pattern_diagnostic(
    row: Dict[str, Any],
    diagnostics: Dict[str, Any],
    nearby_rows: Optional[List[Dict[str, Any]]] = None,
    order_rows: Optional[List[Dict[str, Any]]] = None,
    order_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    updated = deepcopy(diagnostics or {})
    analysis = analyze_dimension_family(row, nearby_rows, order_rows, order_context)
    if not analysis.get("suspicious"):
        return updated
    issues = updated.get("issues") if isinstance(updated.get("issues"), list) else []
    if not any(isinstance(issue, dict) and issue.get("code") == FAMILY_MISMATCH_CODE for issue in issues):
        issues = [*issues, family_pattern_diagnostic_issue()]
    updated["issues"] = issues
    if updated.get("severity") == "ok" or not updated.get("severity"):
        updated["severity"] = "warning"
    updated["requires_human_review"] = True
    updated["family_pattern"] = {
        "suggested_value": analysis.get("suggested_value"),
        "confidence": analysis.get("confidence"),
        "reasoning": analysis.get("reasoning"),
        "trace": analysis.get("trace"),
        "evidence": analysis.get("evidence"),
    }
    return updated


__all__ = [
    "FAMILY_MISMATCH_CODE",
    "FAMILY_MISMATCH_THRESHOLD",
    "analyze_dimension_family",
    "attach_family_pattern_diagnostic",
    "family_pattern_diagnostic_issue",
]
