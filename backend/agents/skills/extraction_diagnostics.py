from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


DIMENSION_RE = re.compile(r"^(\d{2,5})x(\d{2,5})$")
MIN_DIMENSION_MM = 200
MAX_DIMENSION_MM = 6000
AREA_MISMATCH_PERCENT = 2.0
AREA_MISMATCH_ABSOLUTE = 0.01
OCR_FALLBACK_NO_COORDINATES_STEP = "STORE_ROW_COORDINATES_DURING_EXTRACTION"


def _issue(code: str, message: str, field: str, recommended_action: str) -> Dict[str, str]:
    return {
        "code": code,
        "message": message,
        "field": field,
        "recommended_action": recommended_action,
    }


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


def _parse_quantity(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and value > 0 else None
    text = str(value).strip()
    if not text:
        return None
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


def _dimension_key(value: Any) -> str:
    width_mm, height_mm, _raw = _parse_dimension(value)
    if width_mm is None or height_mm is None:
        return ""
    return f"{width_mm}x{height_mm}"


def _issue_codes(diagnostics: Optional[Dict[str, Any]]) -> List[str]:
    issues = diagnostics.get("issues") if isinstance(diagnostics, dict) else None
    if not isinstance(issues, list):
        return []
    return [str(issue.get("code") or "").strip().upper() for issue in issues if isinstance(issue, dict)]


def _nearby_rows(order_context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(order_context, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for key in ("rows_before", "rows_after"):
        value = order_context.get(key)
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, dict))
    return rows


def _same_dimension_neighbors(row: Dict[str, Any], order_context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    current_dimension = _dimension_key(row.get("dimension"))
    if not current_dimension:
        return []
    return [
        nearby
        for nearby in _nearby_rows(order_context)
        if _dimension_key(nearby.get("dimension")) == current_dimension
    ]


def _has_matching_neighbor(row: Dict[str, Any], neighbors: List[Dict[str, Any]]) -> bool:
    quantity = _parse_quantity(row.get("quantity"))
    area = _parse_area(_area_value(row))
    if quantity is None or area is None:
        return False
    for nearby in neighbors:
        nearby_quantity = _parse_quantity(nearby.get("quantity"))
        nearby_area = _parse_area(_area_value(nearby))
        if nearby_quantity == quantity and nearby_area is not None and abs(nearby_area - area) <= 0.01:
            return True
    return False


def _has_same_dimension_area_conflict(row: Dict[str, Any], neighbors: List[Dict[str, Any]]) -> bool:
    area = _parse_area(_area_value(row))
    if area is None:
        return False
    for nearby in neighbors:
        nearby_area = _parse_area(_area_value(nearby))
        if nearby_area is not None and abs(nearby_area - area) > 0.01:
            return True
    return False


def _position_prefix(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split("-", 1)[0]


def _breaks_nearby_pattern(row: Dict[str, Any], order_context: Optional[Dict[str, Any]]) -> bool:
    current_dimension = _dimension_key(row.get("dimension"))
    current_prefix = _position_prefix(row.get("position"))
    if not current_dimension or not current_prefix:
        return False
    pattern_counts: Dict[Tuple[str, str], int] = {}
    for nearby in _nearby_rows(order_context):
        dimension = _dimension_key(nearby.get("dimension"))
        prefix = _position_prefix(nearby.get("position"))
        if dimension and prefix:
            key = (prefix, dimension)
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
    same_position_patterns = [
        count
        for (prefix, dimension), count in pattern_counts.items()
        if prefix == current_prefix and dimension != current_dimension
    ]
    return bool(same_position_patterns and max(same_position_patterns) >= 2)


def _base_diagnosis(
    severity: str,
    summary: str,
    likely_cause: str,
    recommended_action: str,
    confidence: float,
) -> Dict[str, Any]:
    return {
        "severity": severity,
        "summary": summary,
        "likely_cause": likely_cause,
        "recommended_action": recommended_action,
        "confidence": max(0.0, min(1.0, float(confidence))),
        "safe_to_auto_fix": False,
        "suggested_fix": None,
    }


def _area_value(row: Dict[str, Any]) -> Any:
    for key in ("area", "extracted_area", "area_m2"):
        if key in row:
            return row.get(key)
    return None


def _row_id(row: Dict[str, Any]) -> str:
    for key in ("row_id", "id", "order_row_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    position = row.get("position")
    return str(position) if position is not None else ""


def _position_count(row: Dict[str, Any], position: str) -> Optional[int]:
    for key in ("position_count", "duplicate_position_count"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    counts = row.get("position_counts")
    if isinstance(counts, dict) and position:
        value = counts.get(position)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _has_possible_ocr_dimension_fix(
    width_mm: int,
    height_mm: int,
    quantity: int,
    extracted_total_area: float,
) -> bool:
    for side in ("width", "height"):
        base = width_mm if side == "width" else height_mm
        other = height_mm if side == "width" else width_mm
        if base >= 400 or len(str(base)) > 3:
            continue
        for digit in range(10):
            candidate = int(f"{base}{digit}")
            if candidate < MIN_DIMENSION_MM or candidate > MAX_DIMENSION_MM:
                continue
            candidate_area = (candidate * other * quantity) / 1_000_000
            tolerance = max(
                AREA_MISMATCH_ABSOLUTE,
                candidate_area * (AREA_MISMATCH_PERCENT / 100.0),
            )
            if abs(extracted_total_area - candidate_area) <= tolerance:
                return True
    return False


def _select_fallback_target_field(diagnostics: Optional[Dict[str, Any]], requested: Optional[str] = None) -> str:
    allowed = {"dimension", "type", "quantity", "area", "position"}
    if requested in allowed:
        return str(requested)
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
        return "dimension"
    if codes & {"INVALID_AREA", "MISSING_EXTRACTED_AREA", "INVALID_EXTRACTED_AREA"}:
        return "area"
    if codes & {"MISSING_QUANTITY", "INVALID_QUANTITY"}:
        return "quantity"
    if "GLASS_TYPE_UNCLEAR" in codes:
        return "type"
    if codes & {"POSITION_WARNING", "EMPTY_POSITION", "DUPLICATE_POSITION"}:
        return "position"
    return "dimension"


def _fallback_original_value(row: Dict[str, Any], target_field: str) -> Any:
    if target_field == "type":
        return row.get("type", row.get("glass_type"))
    if target_field == "area":
        return _area_value(row)
    return row.get(target_field)


def _dimension_matches_area(dimension: str, quantity: int, area: float) -> bool:
    width_mm, height_mm, _raw = _parse_dimension(dimension)
    if width_mm is None or height_mm is None or quantity <= 0:
        return False
    calculated = (width_mm * height_mm * quantity) / 1_000_000
    tolerance = max(AREA_MISMATCH_ABSOLUTE, calculated * (AREA_MISMATCH_PERCENT / 100.0))
    return abs(area - calculated) <= tolerance


def _dimension_is_suspicious(row: Dict[str, Any], diagnostics: Dict[str, Any]) -> bool:
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
    width_mm, height_mm, _raw = _parse_dimension(row.get("dimension"))
    return width_mm is None or height_mm is None or width_mm < MIN_DIMENSION_MM or height_mm < MIN_DIMENSION_MM


def _pattern_dimension_suggestion(
    row: Dict[str, Any],
    diagnostics: Dict[str, Any],
    order_context: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not _dimension_is_suspicious(row, diagnostics):
        return None
    quantity = _parse_quantity(row.get("quantity")) or 1
    area = _parse_area(_area_value(row))
    if area is None:
        return None

    counts: Dict[str, int] = {}
    for nearby in _nearby_rows(order_context):
        dimension = _dimension_key(nearby.get("dimension"))
        if not dimension:
            continue
        nearby_quantity = _parse_quantity(nearby.get("quantity"))
        if nearby_quantity is not None and nearby_quantity != quantity:
            continue
        counts[dimension] = counts.get(dimension, 0) + 1
    if not counts:
        return None

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    best_dimension, best_count = ranked[0]
    if best_count < 2:
        return None
    if not _dimension_matches_area(best_dimension, quantity, area):
        return None

    return {
        "dimension": best_dimension,
        "count": best_count,
        "nearby_pattern": f"{best_dimension} repeats {best_count} time(s) nearby and matches row area.",
    }


def ocr_fallback_row_repair(
    row: dict,
    diagnostics: Optional[dict] = None,
    target_field: Optional[str] = None,
    order_context: Optional[dict] = None,
    row_index: Optional[int] = None,
    pdf_id: Optional[str] = None,
) -> dict:
    diagnostics = diagnostics if isinstance(diagnostics, dict) else diagnose_extraction_row_issue(row)
    target = _select_fallback_target_field(diagnostics, target_field)
    codes = _issue_codes(diagnostics)
    original_value = _fallback_original_value(row, target)

    if str(diagnostics.get("severity") or "ok") not in {"warning", "error"}:
        return {
            "success": False,
            "target_field": target,
            "original_value": original_value,
            "suggested_value": None,
            "confidence": 0.0,
            "method": "diagnostics_ok_no_fallback",
            "reason": "Backend diagnostics did not report a row-level warning or error.",
            "evidence": {"diagnostic_codes": codes},
            "safe_to_auto_apply": False,
        }

    if target == "dimension":
        suggestion = _pattern_dimension_suggestion(row, diagnostics, order_context)
        if suggestion:
            return {
                "success": True,
                "target_field": "dimension",
                "original_value": original_value,
                "suggested_value": suggestion["dimension"],
                "confidence": 0.85,
                "method": "pattern_fallback_no_pdf_coordinates",
                "reason": "Nearby repeated dimensions match this row's quantity-aware area.",
                "evidence": {
                    "diagnostic_codes": codes,
                    "nearby_pattern": suggestion["nearby_pattern"],
                },
                "safe_to_auto_apply": False,
            }

    return {
        "success": False,
        "target_field": target,
        "original_value": original_value,
        "suggested_value": None,
        "confidence": 0.0,
        "method": "pattern_fallback_no_pdf_coordinates",
        "reason": "PDF row coordinates are not available yet",
        "recommended_next_step": OCR_FALLBACK_NO_COORDINATES_STEP,
        "evidence": {
            "diagnostic_codes": codes,
            "nearby_pattern": None,
            "row_index": row_index,
            "pdf_id": pdf_id,
        },
        "safe_to_auto_apply": False,
    }


def diagnose_extraction_row_issue(row: dict) -> dict:
    issues: List[Dict[str, str]] = []
    error_codes = set()

    dimension_value = row.get("dimension")
    width_mm: Optional[int] = None
    height_mm: Optional[int] = None
    if dimension_value is None or not str(dimension_value).strip():
        error_codes.add("MISSING_DIMENSION")
        issues.append(
            _issue(
                "MISSING_DIMENSION",
                "Dimension is missing from the extracted row.",
                "dimension",
                "OCR_FALLBACK_DIMENSION",
            )
        )
    else:
        width_mm, height_mm, _raw_dimension = _parse_dimension(dimension_value)
        if width_mm is None or height_mm is None:
            error_codes.add("INVALID_DIMENSION_FORMAT")
            issues.append(
                _issue(
                    "INVALID_DIMENSION_FORMAT",
                    "Dimension must contain width and height in millimeters, such as 1200x1400.",
                    "dimension",
                    "OCR_FALLBACK_DIMENSION",
                )
            )
        elif (
            width_mm < MIN_DIMENSION_MM
            or height_mm < MIN_DIMENSION_MM
            or width_mm > MAX_DIMENSION_MM
            or height_mm > MAX_DIMENSION_MM
        ):
            issues.append(
                _issue(
                    "SUSPICIOUS_DIMENSION_SIZE",
                    "Dimension is outside the expected production range for glass rows.",
                    "dimension",
                    "HUMAN_REVIEW_DIMENSION",
                )
            )

    quantity_missing = (
        "quantity" not in row
        or row.get("quantity") is None
        or not str(row.get("quantity")).strip()
    )
    quantity = _parse_quantity(row.get("quantity"))
    if quantity is None:
        code = "MISSING_QUANTITY" if quantity_missing else "INVALID_QUANTITY"
        error_codes.add(code)
        issues.append(
            _issue(
                code,
                (
                    "Quantity is missing from the extracted row."
                    if quantity_missing
                    else "Quantity must be a positive whole number."
                ),
                "quantity",
                "HUMAN_REVIEW_QUANTITY",
            )
        )

    area_raw = _area_value(row)
    area_missing = area_raw is None or not str(area_raw).strip()
    extracted_area = _parse_area(area_raw)
    if extracted_area is None:
        code = "MISSING_EXTRACTED_AREA" if area_missing else "INVALID_EXTRACTED_AREA"
        error_codes.add(code)
        issues.append(
            _issue(
                code,
                (
                    "Extracted area is missing from the row."
                    if area_missing
                    else "Extracted area must be a positive number."
                ),
                "area",
                "HUMAN_REVIEW_AREA",
            )
        )

    if ("type" in row or "glass_type" in row) and not str(
        row.get("type") or row.get("glass_type") or ""
    ).strip():
        issues.append(
            _issue(
                "GLASS_TYPE_UNCLEAR",
                "Glass type is missing or unclear for this extracted row.",
                "type",
                "OCR_FALLBACK_TYPE",
            )
        )

    calculated_area = None
    difference = None
    difference_percent = None
    if width_mm is not None and height_mm is not None and quantity is not None:
        calculated_area = round((width_mm * height_mm * quantity) / 1_000_000, 3)

    if calculated_area is not None and extracted_area is not None:
        difference = round(abs(extracted_area - calculated_area), 3)
        difference_percent = (
            round((difference / calculated_area) * 100.0, 2)
            if calculated_area
            else None
        )
        tolerance = max(
            AREA_MISMATCH_ABSOLUTE,
            calculated_area * (AREA_MISMATCH_PERCENT / 100.0),
        )
        if difference > tolerance:
            issues.append(
                _issue(
                    "AREA_MISMATCH",
                    "Extracted area does not match width x height x quantity.",
                    "area",
                    "OCR_FALLBACK_DIMENSION",
                )
            )

            if _has_possible_ocr_dimension_fix(
                width_mm,
                height_mm,
                quantity,
                extracted_area,
            ):
                issues.append(
                    _issue(
                        "POSSIBLE_DIMENSION_OCR_ERROR",
                        "A width or height value may have lost a digit during OCR.",
                        "dimension",
                        "OCR_FALLBACK_DIMENSION",
                    )
                )

    if "position" in row:
        position = str(row.get("position") or "").strip()
        if not position:
            issues.append(
                _issue(
                    "EMPTY_POSITION",
                    "Position is empty for this extracted row.",
                    "position",
                    "HUMAN_REVIEW_POSITION",
                )
            )
        elif (
            bool(row.get("duplicate_position"))
            or bool(row.get("position_duplicate"))
            or bool(row.get("is_duplicate_position"))
            or (_position_count(row, position) or 0) > 1
        ):
            issues.append(
                _issue(
                    "DUPLICATE_POSITION",
                    "Position appears more than once in the available row context.",
                    "position",
                    "HUMAN_REVIEW_POSITION",
                )
            )

    severity = "ok"
    if issues:
        severity = "error" if error_codes else "warning"

    return {
        "row_id": _row_id(row),
        "severity": severity,
        "issues": issues,
        "computed": {
            "calculated_area": calculated_area,
            "extracted_area": round(extracted_area, 3) if extracted_area is not None else None,
            "difference": difference,
            "difference_percent": difference_percent,
        },
        "requires_human_review": severity != "ok",
    }


def diagnose_extraction_row_warning(
    row: dict,
    diagnostics: Optional[dict] = None,
    order_context: Optional[dict] = None,
) -> dict:
    diagnostics = diagnostics if isinstance(diagnostics, dict) else diagnose_extraction_row_issue(row)
    severity = str(diagnostics.get("severity") or "ok")
    codes = set(_issue_codes(diagnostics))
    if severity not in {"warning", "error"} or not codes:
        return _base_diagnosis(
            "ok",
            "No extraction issue was detected for this row.",
            "Backend diagnostics did not report a row-level warning or error.",
            "MANUAL_REVIEW",
            1.0,
        )

    neighbors = _same_dimension_neighbors(row, order_context)
    context_notes: List[str] = []
    if neighbors and _has_matching_neighbor(row, neighbors):
        context_notes.append("Nearby rows repeat the same dimension, quantity, and area.")
    if neighbors and _has_same_dimension_area_conflict(row, neighbors):
        context_notes.append("Nearby rows use the same dimension but a different area.")
    if _breaks_nearby_pattern(row, order_context):
        context_notes.append("This row breaks a repeated nearby position/dimension pattern.")
    context_suffix = f" {' '.join(context_notes)}" if context_notes else ""

    if "GLASS_TYPE_UNCLEAR" in codes:
        return _base_diagnosis(
            severity,
            f"Glass type needs review.{context_suffix}",
            "The extracted glass type is unclear or inconsistent with row context.",
            "OCR_FALLBACK_TYPE",
            0.72,
        )

    if "MISSING_DIMENSION" in codes:
        return _base_diagnosis(
            severity,
            f"Dimension is missing for this row.{context_suffix}",
            "The PDF extraction did not capture a width x height value.",
            "OCR_FALLBACK_DIMENSION",
            0.9,
        )

    if "INVALID_DIMENSION" in codes or "INVALID_DIMENSION_FORMAT" in codes:
        return _base_diagnosis(
            severity,
            f"Dimension format is invalid for this row.{context_suffix}",
            "The dimension text is not a reliable width x height value.",
            "OCR_FALLBACK_DIMENSION",
            0.88,
        )

    if "MISSING_QUANTITY" in codes or "INVALID_QUANTITY" in codes:
        return _base_diagnosis(
            severity,
            f"Quantity needs review.{context_suffix}",
            "The extracted quantity is missing or is not a positive whole number.",
            "CHECK_QUANTITY",
            0.86,
        )

    if "MISSING_EXTRACTED_AREA" in codes or "INVALID_EXTRACTED_AREA" in codes or "INVALID_AREA" in codes:
        return _base_diagnosis(
            severity,
            f"Area needs review.{context_suffix}",
            "The extracted row area is missing or is not a positive number.",
            "CHECK_AREA",
            0.86,
        )

    if "AREA_MISMATCH" in codes:
        computed = diagnostics.get("computed") if isinstance(diagnostics.get("computed"), dict) else {}
        calculated_area = _parse_area(computed.get("calculated_area"))
        extracted_area = _parse_area(computed.get("extracted_area"))
        quantity = _parse_quantity(row.get("quantity")) or 1
        likely_cause = "The extracted row area does not match width x height x quantity."
        recommended_action = "CHECK_AREA"
        confidence = 0.74

        if "POSSIBLE_DIMENSION_OCR_ERROR" in codes or _breaks_nearby_pattern(row, order_context):
            likely_cause = "A dimension digit may have been missed or misread during OCR."
            recommended_action = "OCR_FALLBACK_DIMENSION"
            confidence = 0.8
        elif calculated_area and extracted_area and quantity > 1 and abs(extracted_area - (calculated_area / quantity)) <= 0.01:
            likely_cause = "The extracted area looks like a single-piece area, but row area should be total area."
            recommended_action = "CHECK_AREA"
            confidence = 0.9
        elif neighbors and _has_same_dimension_area_conflict(row, neighbors):
            likely_cause = "Rows with the same dimension nearby have a different area, so the area value may be wrong."
            recommended_action = "CHECK_AREA"
            confidence = 0.82

        return _base_diagnosis(
            severity,
            f"Area does not match the backend quantity-aware calculation.{context_suffix}",
            likely_cause,
            recommended_action,
            confidence,
        )

    return _base_diagnosis(
        severity,
        f"Backend diagnostics reported a row issue that needs review.{context_suffix}",
        "The issue does not match a specialized rule yet.",
        "MANUAL_REVIEW",
        0.6,
    )


__all__ = [
    "diagnose_extraction_row_issue",
    "diagnose_extraction_row_warning",
    "ocr_fallback_row_repair",
]
