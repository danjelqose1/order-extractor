from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


ROW_SCORE_MIN = 0.05
ROW_SCORE_MAX = 0.99

PENALTY_AREA_DIMENSION_MISMATCH = 0.30
PENALTY_SUSPICIOUS_TRUNCATION = 0.20
PENALTY_DIMENSION_REPAIR = 0.15
PENALTY_MERGED_HEADER_WARNING = 0.10
PENALTY_QUANTITY_MISSING = 0.15
PENALTY_TYPE_MISSING = 0.15
PENALTY_WEAK_POSITION = 0.10
PENALTY_WEAK_DIMENSION = 0.10

ORDER_PENALTY_DUPLICATE_WARNINGS = 0.05
ORDER_PENALTY_DECLARED_MISSING = 0.05
ORDER_PENALTY_DECLARED_MISMATCH = 0.08
ORDER_PENALTY_REPAIRED_ROWS = 0.05
ORDER_REPAIRED_THRESHOLD = 2

STRONG_POSITION_RE = re.compile(r"^\d{1,3}-\d{1,3}(?:/[A-Za-z0-9À-ÿ.\-]+)?$")
STRONG_DIMENSION_RE = re.compile(r"^\d{3,4}x\d{3,4}$")
DECLARED_TOTAL_HINT_RE = re.compile(r"(?im)^(?:m2|totale)\s+\d+\s+[\d.,]+\s*$")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_flags(row: Mapping[str, Any]) -> List[str]:
    flags = row.get("flags")
    if not isinstance(flags, list):
        return []
    output: List[str] = []
    for flag in flags:
        text = str(flag or "").strip().upper()
        if text:
            output.append(text)
    return output


def _warning_items_for_row(
    row_warnings: Optional[Mapping[Any, Any]],
    idx: int,
    row: Mapping[str, Any],
) -> List[str]:
    if not isinstance(row_warnings, Mapping):
        return []
    candidates = [idx, str(idx)]
    rid = row.get("_rid")
    if rid:
        candidates.extend([rid, str(rid)])
    for key in candidates:
        value = row_warnings.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if item]
    return []


def _is_dimension_repaired(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("correctedDimension")
        or row.get("corrected_dimension")
        or row.get("repair_reason")
        or str(row.get("dimension_source") or "").strip().lower() == "raw_repair"
    )


def _has_warning_token(items: Iterable[str], token: str) -> bool:
    marker = token.lower()
    return any(marker in str(item or "").lower() for item in items)


def has_declared_totals_hint(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(DECLARED_TOTAL_HINT_RE.search(text))


def confidence_label(score: float) -> str:
    if score >= 0.90:
        return "high"
    if score >= 0.75:
        return "medium"
    return "low"


def score_row_confidence(
    row: Mapping[str, Any],
    row_warning_items: Optional[Iterable[str]] = None,
) -> Tuple[float, List[str]]:
    warnings = [str(item) for item in (row_warning_items or []) if item]
    flags = set(_normalize_flags(row))

    score = 1.0
    reasons: List[str] = []

    if "AREA_DIMENSION_MISMATCH" in flags or bool(row.get("area_mismatch")):
        score -= PENALTY_AREA_DIMENSION_MISMATCH
        reasons.append("area_dimension_mismatch")

    if "SUSPICIOUS_TRUNCATION" in flags:
        score -= PENALTY_SUSPICIOUS_TRUNCATION
        reasons.append("suspicious_truncation")

    if _is_dimension_repaired(row):
        score -= PENALTY_DIMENSION_REPAIR
        reasons.append("dimension_repair_applied")

    if _has_warning_token(warnings, "type_changed_without_header") or _has_warning_token(warnings, "merged_header"):
        score -= PENALTY_MERGED_HEADER_WARNING
        reasons.append("merged_header_warning")

    if _has_warning_token(warnings, "quantity_defaulted_to_one") or row.get("quantity") in (None, "", 0):
        score -= PENALTY_QUANTITY_MISSING
        reasons.append("quantity_missing_or_defaulted")

    type_text = str(row.get("type") or "").strip()
    if not type_text or _has_warning_token(warnings, "type_propagated"):
        score -= PENALTY_TYPE_MISSING
        reasons.append("type_missing_or_propagated")

    position = str(row.get("position") or "").strip()
    if not STRONG_POSITION_RE.match(position):
        score -= PENALTY_WEAK_POSITION
        reasons.append("weak_position_format")

    dimension = str(row.get("dimension") or "").strip().lower().replace("×", "x").replace(" ", "")
    if not STRONG_DIMENSION_RE.match(dimension):
        score -= PENALTY_WEAK_DIMENSION
        reasons.append("weak_dimension_format")

    return round(_clamp(score, ROW_SCORE_MIN, ROW_SCORE_MAX), 2), reasons


def annotate_rows_with_confidence(
    rows: List[Dict[str, Any]],
    *,
    row_warnings: Optional[Mapping[Any, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    output: List[Dict[str, Any]] = []
    repaired_count = 0

    for idx, row in enumerate(rows or []):
        warning_items = _warning_items_for_row(row_warnings, idx, row)
        score, reasons = score_row_confidence(row, warning_items)
        working = dict(row)
        working["confidence_score"] = score
        working["confidence_reasons"] = reasons
        output.append(working)
        if _is_dimension_repaired(working):
            repaired_count += 1

    return output, {"repaired_row_count": repaired_count}


def compute_order_confidence(
    *,
    rows: List[Dict[str, Any]],
    warnings: Optional[Iterable[str]],
    declared_units: Optional[int],
    declared_area: Optional[float],
    parsed_units: Optional[int],
    parsed_area: Optional[float],
    expected_declared_totals: bool,
    repaired_row_count: Optional[int] = None,
) -> Dict[str, Any]:
    safe_rows = rows or []
    row_scores: List[float] = []
    for row in safe_rows:
        value = row.get("confidence_score")
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = None
        if score is not None and score == score:
            row_scores.append(score)

    base = sum(row_scores) / len(row_scores) if row_scores else ROW_SCORE_MAX
    score = base
    reasons: List[str] = []

    warning_items = [str(item) for item in (warnings or []) if item]
    if warning_items and len(set(warning_items)) < len(warning_items):
        score -= ORDER_PENALTY_DUPLICATE_WARNINGS
        reasons.append("duplicate_warnings_detected")

    if expected_declared_totals and declared_units is None and declared_area is None:
        score -= ORDER_PENALTY_DECLARED_MISSING
        reasons.append("declared_totals_missing")

    totals_mismatch = False
    if declared_units is not None and parsed_units is not None and int(declared_units) != int(parsed_units):
        totals_mismatch = True
    if declared_area is not None and parsed_area is not None and abs(float(declared_area) - float(parsed_area)) > 0.05:
        totals_mismatch = True
    if totals_mismatch:
        score -= ORDER_PENALTY_DECLARED_MISMATCH
        reasons.append("declared_totals_mismatch")

    repaired = repaired_row_count if repaired_row_count is not None else sum(1 for row in safe_rows if _is_dimension_repaired(row))
    if repaired > ORDER_REPAIRED_THRESHOLD:
        score -= ORDER_PENALTY_REPAIRED_ROWS
        reasons.append("high_repaired_row_count")

    score = round(_clamp(score, ROW_SCORE_MIN, ROW_SCORE_MAX), 2)
    return {
        "order_confidence_score": score,
        "order_confidence_label": confidence_label(score),
        "order_confidence_reasons": reasons,
    }


__all__ = [
    "annotate_rows_with_confidence",
    "compute_order_confidence",
    "confidence_label",
    "has_declared_totals_hint",
    "score_row_confidence",
]
