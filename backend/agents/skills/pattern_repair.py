from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple


DIMENSION_RE = re.compile(r"^(\d{2,5})x(\d{2,5})$")
AREA_MISMATCH_ABSOLUTE = 0.01
AREA_MISMATCH_PERCENT = 2.0
MIN_DIMENSION_MM = 200
MAX_DIMENSION_MM = 6000


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


def _parse_quantity(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and value > 0 else None
    text = str(value).strip()
    if re.fullmatch(r"\d+", text):
        quantity = int(text)
        return quantity if quantity > 0 else None
    if re.fullmatch(r"\d+\.0+", text):
        quantity = int(float(text))
        return quantity if quantity > 0 else None
    return None


def _parse_area(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        area = float(value)
        return area if area > 0 else None
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace(" ", "")
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    try:
        area = float(cleaned)
    except ValueError:
        return None
    return area if area > 0 else None


def _area_value(row: Dict[str, Any]) -> Any:
    for key in ("area", "extracted_area", "area_m2"):
        if key in row:
            return row.get(key)
    return None


def _issue_codes(diagnostics: Optional[Dict[str, Any]]) -> List[str]:
    issues = diagnostics.get("issues") if isinstance(diagnostics, dict) else None
    if not isinstance(issues, list):
        return []
    return [str(issue.get("code") or "").strip().upper() for issue in issues if isinstance(issue, dict)]


def _nearby_rows(
    nearby_rows: Optional[Iterable[Dict[str, Any]]] = None,
    order_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str]] = set()

    def add_row(item: Dict[str, Any]) -> None:
        dimension = _dimension_key(item.get("dimension"))
        fingerprint = (
            str(item.get("row_id") or item.get("id") or ""),
            str(item.get("position") or ""),
            dimension,
            str(item.get("quantity") or ""),
        )
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        rows.append(deepcopy(item))

    if isinstance(nearby_rows, list):
        for item in nearby_rows:
            if isinstance(item, dict):
                add_row(item)
    if isinstance(order_context, dict):
        for key in ("rows_before", "rows_after", "nearby_rows"):
            value = order_context.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        add_row(item)
    return rows


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _glass_type(row: Dict[str, Any]) -> str:
    return _normalize_text(row.get("type") or row.get("glass_type"))


def _position_prefix(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.split(r"[-/]", text, maxsplit=1)[0]


def _calculated_area(dimension: str, quantity: int) -> Optional[float]:
    width, height, _raw = _parse_dimension(dimension)
    if width is None or height is None or quantity <= 0:
        return None
    return (width * height * quantity) / 1_000_000


def _area_match_score(dimension: str, quantity: int, extracted_area: Optional[float]) -> Tuple[bool, Optional[float], Optional[float]]:
    calculated = _calculated_area(dimension, quantity)
    if calculated is None or extracted_area is None:
        return False, calculated, None
    delta = abs(extracted_area - calculated)
    tolerance = max(AREA_MISMATCH_ABSOLUTE, calculated * (AREA_MISMATCH_PERCENT / 100.0))
    return delta <= tolerance, calculated, delta


def _dimension_relation(original: str, candidate: str) -> str:
    original_width, original_height, _raw_original = _parse_dimension(original)
    candidate_width, candidate_height, _raw_candidate = _parse_dimension(candidate)
    if (
        original_width is None
        or original_height is None
        or candidate_width is None
        or candidate_height is None
    ):
        return "nearby_supported"

    original_sides = (str(original_width), str(original_height))
    candidate_sides = (str(candidate_width), str(candidate_height))
    if original_width == candidate_width and candidate_sides[1].startswith(original_sides[1]):
        return "missing_trailing_digit"
    if original_height == candidate_height and candidate_sides[0].startswith(original_sides[0]):
        return "missing_trailing_digit"
    if original_width == candidate_width and candidate_sides[1].endswith(original_sides[1]):
        return "missing_leading_digit"
    if original_height == candidate_height and candidate_sides[0].endswith(original_sides[0]):
        return "missing_leading_digit"
    if original_width == candidate_height and original_height == candidate_width:
        return "swapped_dimensions"
    if original_sides[0] in candidate_sides[0] or original_sides[1] in candidate_sides[1]:
        return "truncated_ocr_value"
    return "nearby_supported"


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.68:
        return "medium"
    return "low"


def _is_dimension_repair_relevant(row: Dict[str, Any], diagnostics: Optional[Dict[str, Any]]) -> bool:
    codes = set(_issue_codes(diagnostics))
    if codes & {
        "MISSING_DIMENSION",
        "INVALID_DIMENSION",
        "INVALID_DIMENSION_FORMAT",
        "DIMENSION_OUT_OF_RANGE",
        "SUSPICIOUS_DIMENSION_SIZE",
        "AREA_MISMATCH",
        "POSSIBLE_DIMENSION_OCR_ERROR",
    }:
        return True
    width, height, _raw = _parse_dimension(row.get("dimension"))
    if width is None or height is None:
        return True
    return width < MIN_DIMENSION_MM or height < MIN_DIMENSION_MM or width > MAX_DIMENSION_MM or height > MAX_DIMENSION_MM


def suggest_pattern_repair(
    row: Dict[str, Any],
    diagnostics: Optional[Dict[str, Any]] = None,
    nearby_rows: Optional[List[Dict[str, Any]]] = None,
    order_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a deterministic pattern repair suggestion without mutating inputs."""
    working_row = deepcopy(row or {})
    working_diagnostics = deepcopy(diagnostics or {})
    neighbors = _nearby_rows(nearby_rows, order_context)
    trace: List[str] = ["Pattern repair inspected nearby extraction rows"]
    original_dimension = _dimension_key(working_row.get("dimension")) or str(working_row.get("dimension") or "")

    if not _is_dimension_repair_relevant(working_row, working_diagnostics):
        return {
            "success": False,
            "target_field": "dimension",
            "original_value": working_row.get("dimension"),
            "suggested_value": None,
            "confidence": 0.0,
            "confidence_label": "low",
            "reasoning": "The row diagnostics do not point to a dimension repair.",
            "evidence": {"candidate_count": 0, "diagnostic_codes": _issue_codes(working_diagnostics)},
            "trace": trace + ["Pattern repair skipped because dimension was not suspicious"],
            "method": "pattern_repair",
            "safe_to_auto_apply": False,
        }

    quantity = _parse_quantity(working_row.get("quantity")) or 1
    extracted_area = _parse_area(_area_value(working_row))
    current_type = _glass_type(working_row)
    current_prefix = _position_prefix(working_row.get("position"))

    if extracted_area is None:
        return {
            "success": False,
            "target_field": "dimension",
            "original_value": working_row.get("dimension"),
            "suggested_value": None,
            "confidence": 0.0,
            "confidence_label": "low",
            "reasoning": "Pattern repair needs a row area to validate dimension candidates.",
            "evidence": {"candidate_count": 0, "diagnostic_codes": _issue_codes(working_diagnostics)},
            "trace": trace + ["No positive row area was available for quantity-aware area math"],
            "method": "pattern_repair",
            "safe_to_auto_apply": False,
        }

    candidate_map: Dict[str, Dict[str, Any]] = {}
    for neighbor in neighbors:
        dimension = _dimension_key(neighbor.get("dimension"))
        if not dimension or dimension == original_dimension:
            continue
        nearby_quantity = _parse_quantity(neighbor.get("quantity"))
        if nearby_quantity is not None and nearby_quantity != quantity:
            continue
        entry = candidate_map.setdefault(
            dimension,
            {
                "dimension": dimension,
                "count": 0,
                "same_glass_type_count": 0,
                "same_position_prefix_count": 0,
                "sample_rows": [],
            },
        )
        entry["count"] += 1
        if current_type and _glass_type(neighbor) == current_type:
            entry["same_glass_type_count"] += 1
        if current_prefix and _position_prefix(neighbor.get("position")) == current_prefix:
            entry["same_position_prefix_count"] += 1
        if len(entry["sample_rows"]) < 3:
            entry["sample_rows"].append(
                {
                    "position": neighbor.get("position"),
                    "dimension": dimension,
                    "quantity": nearby_quantity,
                    "area": _area_value(neighbor),
                    "type": neighbor.get("type") or neighbor.get("glass_type"),
                }
            )

    if not candidate_map:
        return {
            "success": False,
            "target_field": "dimension",
            "original_value": working_row.get("dimension"),
            "suggested_value": None,
            "confidence": 0.0,
            "confidence_label": "low",
            "reasoning": "No nearby supported dimension candidates were found.",
            "evidence": {"candidate_count": 0, "diagnostic_codes": _issue_codes(working_diagnostics)},
            "trace": trace + ["No nearby dimension candidates were available"],
            "method": "pattern_repair",
            "safe_to_auto_apply": False,
        }

    scored: List[Dict[str, Any]] = []
    for candidate, evidence in candidate_map.items():
        area_matches, calculated_area, delta = _area_match_score(candidate, quantity, extracted_area)
        relation = _dimension_relation(original_dimension, candidate)
        confidence = 0.35
        if area_matches:
            confidence += 0.32
        else:
            confidence -= 0.25
        count = int(evidence["count"])
        if count >= 3:
            confidence += 0.18
        elif count >= 2:
            confidence += 0.14
        elif count == 1:
            confidence += 0.03
        if evidence["same_glass_type_count"]:
            confidence += min(0.08, 0.04 * int(evidence["same_glass_type_count"]))
        if evidence["same_position_prefix_count"]:
            confidence += min(0.06, 0.03 * int(evidence["same_position_prefix_count"]))
        if relation in {"missing_trailing_digit", "missing_leading_digit", "truncated_ocr_value"}:
            confidence += 0.1
        elif relation == "swapped_dimensions":
            confidence += 0.08
        if "POSSIBLE_DIMENSION_OCR_ERROR" in set(_issue_codes(working_diagnostics)):
            confidence += 0.04

        scored.append(
            {
                **evidence,
                "area_matches": area_matches,
                "calculated_area": round(calculated_area, 3) if calculated_area is not None else None,
                "area_delta": round(delta, 3) if delta is not None else None,
                "relation": relation,
                "confidence": max(0.0, min(0.95, confidence)),
            }
        )

    area_supported = [item for item in scored if item["area_matches"]]
    if not area_supported:
        best_unmatched = sorted(scored, key=lambda item: (-item["confidence"], item["dimension"]))[0]
        return {
            "success": False,
            "target_field": "dimension",
            "original_value": working_row.get("dimension"),
            "suggested_value": None,
            "confidence": round(best_unmatched["confidence"], 2),
            "confidence_label": _confidence_label(float(best_unmatched["confidence"])),
            "reasoning": "Nearby dimensions exist, but none match the row's quantity-aware area.",
            "evidence": {
                "diagnostic_codes": _issue_codes(working_diagnostics),
                "candidate_count": len(scored),
                "candidates": scored,
            },
            "trace": trace + ["Checked area consistency", "Rejected pattern candidates because area mismatch remained"],
            "method": "pattern_repair",
            "safe_to_auto_apply": False,
        }

    area_supported.sort(key=lambda item: (-item["confidence"], -int(item["count"]), item["dimension"]))
    best = area_supported[0]
    ambiguous = len(area_supported) > 1 and abs(float(best["confidence"]) - float(area_supported[1]["confidence"])) < 0.08
    if ambiguous:
        best["confidence"] = max(0.0, float(best["confidence"]) - 0.12)

    confidence = round(float(best["confidence"]), 2)
    trace.append("Checked area consistency")
    trace.append(f"Found nearby repeated dimension {best['dimension']} ({best['count']} support row(s))")
    trace.append(f"Area {round(extracted_area, 3)} matches {best['dimension']}")
    if best["same_glass_type_count"]:
        trace.append("Same glass type block supports the candidate")
    if ambiguous:
        trace.append("Confidence reduced because multiple candidates were plausible")
    trace.append(f"Pattern repair confidence {confidence:.2f}")

    if confidence < 0.55:
        return {
            "success": False,
            "target_field": "dimension",
            "original_value": working_row.get("dimension"),
            "suggested_value": None,
            "confidence": confidence,
            "confidence_label": _confidence_label(confidence),
            "reasoning": "Pattern support was present but too weak for a repair suggestion.",
            "evidence": {
                "diagnostic_codes": _issue_codes(working_diagnostics),
                "candidate_count": len(scored),
                "candidates": scored,
            },
            "trace": trace,
            "method": "pattern_repair",
            "safe_to_auto_apply": False,
        }

    reasoning = (
        f"Nearby rows support {best['dimension']} and its quantity-aware area "
        f"matches the extracted area {round(extracted_area, 3)}."
    )
    return {
        "success": True,
        "target_field": "dimension",
        "original_value": working_row.get("dimension"),
        "suggested_value": best["dimension"],
        "confidence": confidence,
        "confidence_label": _confidence_label(confidence),
        "recommended_action": "ACCEPT_PATTERN_SUGGESTION_OR_REVIEW",
        "reasoning": reasoning,
        "evidence": {
            "diagnostic_codes": _issue_codes(working_diagnostics),
            "candidate_count": len(scored),
            "selected_candidate": best,
            "candidates": scored,
            "area": round(extracted_area, 3),
            "quantity": quantity,
        },
        "trace": trace,
        "method": "pattern_repair",
        "safe_to_auto_apply": False,
    }


__all__ = ["suggest_pattern_repair"]
