from __future__ import annotations

import base64
import os
import re
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


DIMENSION_RE = re.compile(r"^(\d{2,5})x(\d{2,5})$")
MIN_DIMENSION_MM = 200
MAX_DIMENSION_MM = 6000
AREA_MISMATCH_PERCENT = 2.0
AREA_MISMATCH_ABSOLUTE = 0.01
OCR_FALLBACK_NO_COORDINATES_STEP = "STORE_ROW_COORDINATES_DURING_EXTRACTION"
DIMENSION_IN_TEXT_RE = re.compile(r"\b(\d{2,5})\s*[x×]\s*(\d{2,5})\b", re.IGNORECASE)
OPENAI_VISION_PAGE_UNAVAILABLE_REASON = (
    "OpenAI OCR fallback could not run because the original PDF page was not available."
)
POSITION_ORDER_PREFIX_RE = re.compile(r"^R-?\d{2}-\d{4}/", re.IGNORECASE)
POSITION_LABEL_RE = re.compile(r"^(?:position|posizione|pos|nr|no)\s*[:#-]?\s*", re.IGNORECASE)


OpenAIVisionRepairFn = Callable[..., Any]


def _short_text(value: Any, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}..."


def _normalize_anchor_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower().replace("×", "x"))


def _numeric_text_variants(value: Any) -> List[str]:
    area = _parse_area(value)
    if area is None:
        return []
    variants = {
        f"{area:.3f}",
        f"{area:.2f}",
        f"{area:.1f}",
        str(area).rstrip("0").rstrip("."),
    }
    return [item for item in variants if item]


def _text_line_bbox(line: Dict[str, Any]) -> Optional[Dict[str, float]]:
    bbox = line.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _extract_pdf_text_lines(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    if not pdf_bytes:
        return []
    try:
        import fitz  # type: ignore
    except Exception:
        return []

    lines: List[Dict[str, Any]] = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_index, page in enumerate(doc, start=1):
                page_dict = page.get_text("dict") or {}
                for block in page_dict.get("blocks") or []:
                    for line in block.get("lines") or []:
                        spans = line.get("spans") or []
                        text = " ".join(
                            str(span.get("text") or "").strip()
                            for span in spans
                            if str(span.get("text") or "").strip()
                        )
                        bbox = _text_line_bbox(line)
                        if text and bbox:
                            lines.append(
                                {
                                    "page": page_index,
                                    "bbox": bbox,
                                    "text": _short_text(text),
                                }
                            )
    except Exception:
        return []
    return lines


def _score_row_location_match(row: Dict[str, Any], line: Dict[str, Any]) -> Tuple[float, bool]:
    text = str(line.get("text") or "")
    normalized = _normalize_anchor_text(text)
    score = 0.0
    has_strong_anchor = False

    dimension = _dimension_key(row.get("dimension"))
    if dimension and _normalize_anchor_text(dimension) in normalized:
        score += 3.0
        has_strong_anchor = True

    position = str(row.get("position") or "").strip()
    if position and _normalize_anchor_text(position) in normalized:
        score += 2.0
        has_strong_anchor = True

    for area_variant in _numeric_text_variants(_area_value(row)):
        if _normalize_anchor_text(area_variant) in normalized:
            score += 1.5
            break

    quantity = _parse_quantity(row.get("quantity"))
    if quantity is not None and re.search(rf"(?<!\d){quantity}(?!\d)", text):
        score += 0.75

    glass_type = str(row.get("type") or row.get("glass_type") or "").strip()
    if glass_type:
        type_words = [
            _normalize_anchor_text(word)
            for word in re.split(r"\s+", glass_type)
            if len(_normalize_anchor_text(word)) >= 4
        ]
        if type_words and any(word in normalized for word in type_words[:4]):
            score += 0.75

    return score, has_strong_anchor


def attach_pdf_row_locations(rows: Sequence[Dict[str, Any]], pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Attach best-effort PDF text-layer location metadata without changing row data."""
    text_lines = _extract_pdf_text_lines(pdf_bytes)
    if not text_lines:
        return [dict(row, row_location=None) for row in rows or []]

    output: List[Dict[str, Any]] = []
    used_line_indexes: set[int] = set()
    for row in rows or []:
        working = dict(row)
        best: Optional[Tuple[float, int, Dict[str, Any]]] = None
        for index, line in enumerate(text_lines):
            if index in used_line_indexes:
                continue
            score, has_strong_anchor = _score_row_location_match(working, line)
            if score < 3.0 or not has_strong_anchor:
                continue
            if best is None or score > best[0]:
                best = (score, index, line)

        if best is None:
            working["row_location"] = None
        else:
            score, index, line = best
            used_line_indexes.add(index)
            confidence = max(0.45, min(0.95, score / 7.0))
            working["row_location"] = {
                "page": int(line["page"]),
                "bbox": dict(line["bbox"]),
                "source": "pdf_text_layer",
                "confidence": round(confidence, 2),
                "matched_text": _short_text(line.get("text")),
            }
        output.append(working)
    return output


def extract_pdf_text_layer_text(pdf_bytes: bytes) -> str:
    lines = _extract_pdf_text_lines(pdf_bytes)
    return "\n".join(str(line.get("text") or "") for line in lines if str(line.get("text") or "").strip())


def extract_pdf_text_for_row_location(pdf_bytes: bytes, row_location: Dict[str, Any]) -> str:
    if not pdf_bytes or not isinstance(row_location, dict):
        return ""
    bbox = row_location.get("bbox")
    if not isinstance(bbox, dict):
        return ""
    try:
        page_number = int(row_location.get("page") or 0)
        x0 = float(bbox.get("x0"))
        y0 = float(bbox.get("y0"))
        x1 = float(bbox.get("x1"))
        y1 = float(bbox.get("y1"))
    except (TypeError, ValueError):
        return ""
    if page_number < 1 or x1 <= x0 or y1 <= y0:
        return ""
    try:
        import fitz  # type: ignore
    except Exception:
        return ""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            if page_number > len(doc):
                return ""
            page = doc[page_number - 1]
            rect = fitz.Rect(x0, y0, x1, y1)
            words = page.get_text("words", clip=rect) or []
    except Exception:
        return ""
    if not words:
        return ""
    words = sorted(words, key=lambda item: (round(float(item[1]), 1), float(item[0])))
    return _short_text(" ".join(str(item[4]) for item in words if len(item) >= 5))


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
        "POSSIBLE_DIMENSION_FAMILY_MISMATCH",
    }:
        return "dimension"
    if codes & {"INVALID_AREA", "MISSING_EXTRACTED_AREA", "INVALID_EXTRACTED_AREA"}:
        return "area"
    if codes & {"MISSING_QUANTITY", "INVALID_QUANTITY"}:
        return "quantity"
    if "GLASS_TYPE_UNCLEAR" in codes:
        return "type"
    if codes & {"MISSING_POSITION", "POSITION_WARNING", "EMPTY_POSITION", "DUPLICATE_POSITION"}:
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
        "POSSIBLE_DIMENSION_FAMILY_MISMATCH",
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


def _row_location(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    value = row.get("row_location")
    return value if isinstance(value, dict) else None


def _row_location_text(row: Dict[str, Any]) -> str:
    location = _row_location(row)
    if not location:
        return ""
    return _short_text(location.get("matched_text") or location.get("ocr_text") or "")


def _dimension_candidates_from_text(text: str) -> List[str]:
    candidates: List[str] = []
    seen: set[str] = set()
    for match in DIMENSION_IN_TEXT_RE.finditer(text or ""):
        candidate = f"{int(match.group(1))}x{int(match.group(2))}"
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def _text_layer_dimension_suggestion(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    region_text = _row_location_text(row)
    if not region_text:
        return None
    quantity = _parse_quantity(row.get("quantity")) or 1
    area = _parse_area(_area_value(row))
    original = _dimension_key(row.get("dimension"))

    ranked: List[Tuple[float, str, str]] = []
    for candidate in _dimension_candidates_from_text(region_text):
        confidence = 0.65
        reason = "Text-layer row region contains a dimension candidate."
        if area is not None and _dimension_matches_area(candidate, quantity, area):
            confidence = 0.9
            reason = "Text-layer row region contains a dimension that matches the row's quantity-aware area."
        if original and candidate == original:
            confidence -= 0.25
            reason = "Text-layer row region repeated the current dimension value."
        if original and candidate != original:
            original_width, original_height, _raw = _parse_dimension(original)
            candidate_width, candidate_height, _candidate_raw = _parse_dimension(candidate)
            if (
                original_width is not None
                and original_height is not None
                and candidate_width == original_width
                and str(candidate_height).startswith(str(original_height))
            ):
                confidence = max(confidence, 0.85)
                reason = "Text-layer row region suggests the extracted height may have lost a trailing digit."
        ranked.append((confidence, candidate, reason))

    if not ranked:
        return None
    confidence, candidate, reason = sorted(ranked, key=lambda item: (-item[0], item[1]))[0]
    if confidence < 0.75 or (original and candidate == original):
        return None
    return {
        "dimension": candidate,
        "confidence": round(min(0.95, confidence), 2),
        "reason": reason,
    }


def _row_location_evidence(row: Dict[str, Any], codes: List[str]) -> Dict[str, Any]:
    location = _row_location(row) or {}
    return {
        "diagnostic_codes": codes,
        "page": location.get("page"),
        "bbox": deepcopy(location.get("bbox")) if isinstance(location.get("bbox"), dict) else None,
        "source": location.get("source"),
        "matched_text": _short_text(location.get("matched_text")),
        "ocr_text": _short_text(location.get("matched_text")),
        "row_location_confidence": location.get("confidence"),
    }


def _should_attempt_openai_vision_page_ocr(codes: Sequence[str], target_field: str) -> bool:
    code_set = {str(code or "").strip().upper() for code in codes or []}
    if target_field == "dimension":
        return bool(code_set & {"MISSING_DIMENSION", "INVALID_DIMENSION", "INVALID_DIMENSION_FORMAT"})
    if target_field == "position":
        return bool(code_set & {"MISSING_POSITION", "POSITION_WARNING", "EMPTY_POSITION", "DUPLICATE_POSITION"})
    return False


def _compact_row_for_prompt(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "order_number": row.get("order_number") or "",
        "type": row.get("type") or row.get("glass_type") or "",
        "dimension": row.get("dimension") or "",
        "position": row.get("position") or "",
        "quantity": row.get("quantity") if row.get("quantity") is not None else "",
        "area": _area_value(row) if _area_value(row) is not None else row.get("area", ""),
    }


def _nearby_rows_for_ocr_context(order_context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(order_context, dict):
        return rows
    for key in ("rows_before", "rows_after", "nearby_rows"):
        value = order_context.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                compact = _compact_row_for_prompt(item)
                if compact not in rows:
                    rows.append(compact)
    return rows[:8]


def _pdf_page_count(pdf_bytes: bytes) -> int:
    if not pdf_bytes:
        return 0
    try:
        import fitz  # type: ignore
    except Exception:
        return 0
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return len(doc)
    except Exception:
        return 0


def _extract_pdf_page_texts(pdf_bytes: bytes) -> List[str]:
    if not pdf_bytes:
        return []
    try:
        import fitz  # type: ignore
    except Exception:
        return []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return [str(page.get_text("text") or "").strip() for page in doc]
    except Exception:
        return []


def _page_text_anchor_score(
    *,
    text: str,
    row: Dict[str, Any],
    order_context: Optional[Dict[str, Any]],
    target_field: str,
) -> float:
    normalized = _normalize_anchor_text(text)
    if not normalized:
        return 0.0

    score = 0.0
    position = str(row.get("position") or "").strip()
    if position and _normalize_anchor_text(position) in normalized:
        score += 5.0

    dimension = _dimension_key(row.get("dimension"))
    if dimension and _normalize_anchor_text(dimension) in normalized:
        score += 3.0

    order_number = str(row.get("order_number") or "").strip()
    if order_number and _normalize_anchor_text(order_number) in normalized:
        score += 2.0

    for area_variant in _numeric_text_variants(_area_value(row)):
        if _normalize_anchor_text(area_variant) in normalized:
            score += 1.5
            break

    quantity = _parse_quantity(row.get("quantity"))
    if quantity is not None and re.search(rf"(?<!\d){quantity}(?!\d)", text):
        score += 0.5

    glass_type = str(row.get("type") or row.get("glass_type") or "").strip()
    if glass_type:
        matched_type_words = 0
        for word in re.split(r"\s+", glass_type):
            anchor = _normalize_anchor_text(word)
            if len(anchor) >= 4 and anchor in normalized:
                matched_type_words += 1
        score += min(2.0, matched_type_words * 0.5)

    if isinstance(order_context, dict):
        for key, weight in (("rows_before", 0.9), ("rows_after", 0.9), ("nearby_rows", 0.6)):
            for nearby in order_context.get(key) or []:
                if not isinstance(nearby, dict):
                    continue
                nearby_position = str(nearby.get("position") or "").strip()
                if nearby_position and _normalize_anchor_text(nearby_position) in normalized:
                    score += weight
                nearby_dimension = _dimension_key(nearby.get("dimension"))
                if nearby_dimension and _normalize_anchor_text(nearby_dimension) in normalized:
                    score += weight * 0.6

    if target_field == "position" and not position and dimension and score:
        score += 0.75
    return score


def _estimate_page_from_row_index(
    *,
    row_index: Optional[int],
    page_count: int,
    order_context: Optional[Dict[str, Any]],
) -> int:
    if page_count <= 1:
        return 1
    total_rows = 0
    if isinstance(order_context, dict):
        order_rows = order_context.get("order_rows")
        if isinstance(order_rows, list):
            total_rows = len(order_rows)
    try:
        index = int(row_index if row_index is not None else 0)
    except (TypeError, ValueError):
        index = 0
    index = max(0, index)
    if total_rows > 0:
        estimated = int((index / max(1, total_rows)) * page_count) + 1
    else:
        estimated = 1
    return max(1, min(page_count, estimated))


def _select_pdf_pages_for_row_repair(
    *,
    row: Dict[str, Any],
    pdf_bytes: bytes,
    order_context: Optional[Dict[str, Any]],
    target_field: str,
    row_index: Optional[int],
) -> Dict[str, Any]:
    page_count = _pdf_page_count(pdf_bytes)
    if page_count <= 0:
        return {"pages": [], "confidence": 0.0, "reason": "pdf_page_unavailable", "page_texts": []}

    location = _row_location(row)
    if location:
        try:
            page = int(location.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        if 1 <= page <= page_count:
            page_texts = _extract_pdf_page_texts(pdf_bytes)
            return {
                "pages": [page],
                "confidence": float(location.get("confidence") or 0.85),
                "reason": "row_location_page",
                "page_texts": page_texts,
            }

    page_texts = _extract_pdf_page_texts(pdf_bytes)
    scored: List[Tuple[float, int]] = []
    for index, page_text in enumerate(page_texts, start=1):
        scored.append(
            (
                _page_text_anchor_score(
                    text=page_text,
                    row=row,
                    order_context=order_context,
                    target_field=target_field,
                ),
                index,
            )
        )

    if scored:
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_page = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= 2.5:
            confidence = max(0.45, min(0.82, best_score / 8.0))
            pages = [best_page]
            if best_score < 4.5 or (best_score - second_score) < 1.25:
                nearby = best_page + 1 if best_page < page_count else best_page - 1
                if 1 <= nearby <= page_count and nearby not in pages:
                    pages.append(nearby)
            return {
                "pages": pages,
                "confidence": round(confidence, 2),
                "reason": "pdf_text_anchor_search",
                "page_texts": page_texts,
                "page_scores": scored[:4],
            }

    estimated_page = _estimate_page_from_row_index(
        row_index=row_index,
        page_count=page_count,
        order_context=order_context,
    )
    pages = [estimated_page]
    if page_count > 1:
        nearby = estimated_page + 1 if estimated_page < page_count else estimated_page - 1
        if 1 <= nearby <= page_count:
            pages.append(nearby)
    return {
        "pages": pages,
        "confidence": 0.38,
        "reason": "row_index_page_estimate",
        "page_texts": page_texts,
    }


def _render_pdf_pages_to_png(pdf_bytes: bytes, pages: Sequence[int], dpi: int = 180) -> List[Dict[str, Any]]:
    if not pdf_bytes or not pages:
        return []
    try:
        import fitz  # type: ignore
    except Exception:
        return []
    rendered: List[Dict[str, Any]] = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_number in pages:
                if page_number < 1 or page_number > len(doc):
                    continue
                pix = doc[page_number - 1].get_pixmap(dpi=dpi, alpha=False)
                rendered.append({"page": int(page_number), "png_bytes": pix.tobytes("png")})
    except Exception:
        return []
    return rendered


def _page_context_snippet(page_texts: Sequence[str], pages: Sequence[int], limit: int = 1600) -> str:
    snippets: List[str] = []
    for page in pages:
        if page < 1 or page > len(page_texts):
            continue
        text = _short_text(page_texts[page - 1], limit=limit)
        if text:
            snippets.append(f"Page {page}: {text}")
    return "\n".join(snippets)


def _format_prompt_rows(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "None"
    lines: List[str] = []
    for index, row in enumerate(rows, start=1):
        compact = _compact_row_for_prompt(row)
        lines.append(
            (
                f"{index}. order={compact['order_number']} type={compact['type']} "
                f"dimension={compact['dimension']} position={compact['position']} "
                f"qty={compact['quantity']} area={compact['area']}"
            ).strip()
        )
    return "\n".join(lines)


def _build_openai_vision_repair_prompt(
    *,
    row: Dict[str, Any],
    target_field: str,
    order_context: Optional[Dict[str, Any]],
    page_context: str,
    pages: Sequence[int],
) -> str:
    rows_before = order_context.get("rows_before") if isinstance(order_context, dict) else []
    rows_after = order_context.get("rows_after") if isinstance(order_context, dict) else []
    order_number = row.get("order_number") or (order_context or {}).get("order_number") or ""
    glass_type = row.get("type") or row.get("glass_type") or ""
    position = str(row.get("position") or "").strip()
    area = _area_value(row)
    quantity = row.get("quantity")
    target_instruction = (
        "Return only dimension candidates in WIDTHxHEIGHT format, one per line. "
        "Use millimeters. If no candidate is visible, return NO_VALUE."
        if target_field == "dimension"
        else "Return only position candidates, one per line. If no candidate is visible, return NO_VALUE."
    )
    return (
        "Recover one missing or suspicious field from the attached glass-order PDF page image(s).\n"
        f"Target field: {target_field}\n"
        f"{target_instruction}\n"
        "Do not return JSON, explanation, labels, surrounding table text, or any field other than the target.\n\n"
        "Row to repair:\n"
        f"- order number: {order_number}\n"
        f"- glass type: {glass_type}\n"
        f"- current dimension: {row.get('dimension') or ''}\n"
        f"- current position: {position}\n"
        f"- area: {area if area is not None else row.get('area', '')}\n"
        f"- quantity: {quantity if quantity is not None else ''}\n\n"
        "Neighboring rows before:\n"
        f"{_format_prompt_rows(rows_before if isinstance(rows_before, list) else [])}\n\n"
        "Neighboring rows after:\n"
        f"{_format_prompt_rows(rows_after if isinstance(rows_after, list) else [])}\n\n"
        f"Most likely page(s): {', '.join(str(page) for page in pages)}\n"
        "Page/table context from the PDF text layer:\n"
        f"{page_context or 'No reliable text layer context is available; use the page image.'}"
    )


def _extract_openai_responses_output_text(response_json: Dict[str, Any]) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    parts: List[str] = []
    for output_item in response_json.get("output") or []:
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content") or []:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") in {"output_text", "text"}:
                text_value = content_item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(text_value.strip())
    return "\n".join(parts).strip()


def _call_openai_vision_page_ocr(
    *,
    target_field: str,
    prompt: str,
    page_images: Sequence[Dict[str, Any]],
    pages: Sequence[int],
    row: Dict[str, Any],
    diagnostics: Dict[str, Any],
    order_context: Dict[str, Any],
    pdf_id: Optional[str] = None,
) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    try:
        import httpx  # type: ignore
    except Exception as exc:
        raise RuntimeError("httpx is not available for OpenAI OCR fallback.") from exc

    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for image in page_images:
        encoded = base64.b64encode(image.get("png_bytes") or b"").decode("ascii")
        if not encoded:
            continue
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{encoded}",
            }
        )
    if len(content) == 1:
        raise RuntimeError("No rendered page images were available for OpenAI OCR fallback.")

    payload = {
        "model": os.getenv("OCR_MODEL", os.getenv("EXTRACTION_MODEL", "gpt-5-mini")),
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are a strict OCR verifier for glass-order tables. "
                            "Return only the requested target field candidates."
                        ),
                    }
                ],
            },
            {"role": "user", "content": content},
        ],
    }
    response = httpx.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=httpx.Timeout(timeout=90.0, connect=15.0, read=90.0, write=30.0),
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Responses API error ({response.status_code}): {response.text.strip()}")
    response_json = response.json()
    return {
        "text": _extract_openai_responses_output_text(response_json),
        "raw_response": response_json,
    }


def _position_candidates_from_text(text: str) -> List[str]:
    candidates: List[str] = []
    seen: set[str] = set()
    for raw_line in re.split(r"[\n,;]+", text or ""):
        candidate = POSITION_LABEL_RE.sub("", raw_line.strip())
        candidate = POSITION_ORDER_PREFIX_RE.sub("", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip(" .:-")
        if not candidate or candidate.upper() in {"NO_VALUE", "N/A", "NONE", "UNKNOWN"}:
            continue
        if len(candidate) > 60 or not re.search(r"\d", candidate):
            continue
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def _openai_vision_candidate_from_text(text: str, target_field: str) -> Optional[str]:
    if target_field == "dimension":
        candidates = _dimension_candidates_from_text(text)
        return candidates[0] if candidates else None
    candidates = _position_candidates_from_text(text)
    return candidates[0] if candidates else None


def _openai_vision_confidence(
    *,
    target_field: str,
    candidate: str,
    row: Dict[str, Any],
    page_selection_confidence: float,
    model_confidence: Optional[float],
) -> float:
    if model_confidence is not None:
        return max(0.0, min(1.0, float(model_confidence)))
    confidence = max(0.55, min(0.82, page_selection_confidence + 0.2))
    if target_field == "dimension":
        quantity = _parse_quantity(row.get("quantity")) or 1
        area = _parse_area(_area_value(row))
        if area is not None and _dimension_matches_area(candidate, quantity, area):
            confidence = max(confidence, 0.84)
    else:
        current = str(row.get("position") or "").strip()
        if not current:
            confidence = max(confidence, 0.78)
    return round(min(0.92, confidence), 2)


def _openai_vision_page_ocr_repair(
    *,
    row: Dict[str, Any],
    diagnostics: Dict[str, Any],
    target_field: str,
    order_context: Optional[Dict[str, Any]],
    row_index: Optional[int],
    pdf_id: Optional[str],
    pdf_bytes: Optional[bytes],
    openai_vision_repair_fn: Optional[OpenAIVisionRepairFn],
) -> Dict[str, Any]:
    codes = _issue_codes(diagnostics)
    original_value = _fallback_original_value(row, target_field)
    nearby_rows = _nearby_rows_for_ocr_context(order_context)

    if not pdf_bytes:
        return {
            "success": False,
            "target_field": target_field,
            "original_value": original_value,
            "suggested_value": None,
            "confidence": 0.0,
            "method": "openai_vision_page_ocr",
            "reason": OPENAI_VISION_PAGE_UNAVAILABLE_REASON,
            "evidence": {
                "diagnostic_codes": codes,
                "method": "openai_vision_page_ocr",
                "page": None,
                "ocr_text": "",
                "nearby_rows": nearby_rows,
                "row_index": row_index,
                "pdf_id": pdf_id,
            },
            "safe_to_auto_apply": False,
        }

    page_selection = _select_pdf_pages_for_row_repair(
        row=row,
        pdf_bytes=pdf_bytes,
        order_context=order_context,
        target_field=target_field,
        row_index=row_index,
    )
    pages: List[int] = []
    for page in page_selection.get("pages") or []:
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            continue
        if page_number > 0 and page_number not in pages:
            pages.append(page_number)
    page_images = _render_pdf_pages_to_png(pdf_bytes, pages)
    if not pages or not page_images:
        return {
            "success": False,
            "target_field": target_field,
            "original_value": original_value,
            "suggested_value": None,
            "confidence": 0.0,
            "method": "openai_vision_page_ocr",
            "reason": OPENAI_VISION_PAGE_UNAVAILABLE_REASON,
            "evidence": {
                "diagnostic_codes": codes,
                "method": "openai_vision_page_ocr",
                "page": pages[0] if pages else None,
                "pages": pages,
                "ocr_text": "",
                "nearby_rows": nearby_rows,
                "row_index": row_index,
                "pdf_id": pdf_id,
                "page_selection": page_selection.get("reason"),
            },
            "safe_to_auto_apply": False,
        }

    page_texts = page_selection.get("page_texts") if isinstance(page_selection.get("page_texts"), list) else []
    page_context = _page_context_snippet(page_texts, pages)
    prompt = _build_openai_vision_repair_prompt(
        row=row,
        target_field=target_field,
        order_context=order_context or {},
        page_context=page_context,
        pages=pages,
    )

    try:
        vision_response = (openai_vision_repair_fn or _call_openai_vision_page_ocr)(
            target_field=target_field,
            prompt=prompt,
            page_images=page_images,
            pages=pages,
            row=deepcopy(row),
            diagnostics=deepcopy(diagnostics),
            order_context=deepcopy(order_context or {}),
            pdf_id=pdf_id,
        )
    except Exception as exc:
        return {
            "success": False,
            "target_field": target_field,
            "original_value": original_value,
            "suggested_value": None,
            "confidence": 0.0,
            "method": "openai_vision_page_ocr",
            "reason": f"OpenAI OCR fallback could not run: {exc}",
            "evidence": {
                "diagnostic_codes": codes,
                "method": "openai_vision_page_ocr",
                "page": pages[0],
                "pages": pages,
                "ocr_text": "",
                "nearby_rows": nearby_rows,
                "row_index": row_index,
                "pdf_id": pdf_id,
                "page_selection": page_selection.get("reason"),
            },
            "safe_to_auto_apply": False,
        }

    model_confidence: Optional[float] = None
    if isinstance(vision_response, dict):
        raw_text = str(
            vision_response.get("text")
            or vision_response.get("output_text")
            or vision_response.get("suggested_value")
            or ""
        ).strip()
        if vision_response.get("confidence") is not None:
            try:
                model_confidence = float(vision_response.get("confidence"))
            except (TypeError, ValueError):
                model_confidence = None
    else:
        raw_text = str(vision_response or "").strip()

    candidate = _openai_vision_candidate_from_text(raw_text, target_field)
    evidence = {
        "diagnostic_codes": codes,
        "method": "openai_vision_page_ocr",
        "page": pages[0],
        "pages": pages,
        "ocr_text": _short_text(raw_text),
        "nearby_rows": nearby_rows,
        "row_index": row_index,
        "pdf_id": pdf_id,
        "page_selection": page_selection.get("reason"),
        "page_context": _short_text(page_context, 360),
    }
    if not candidate:
        return {
            "success": False,
            "target_field": target_field,
            "original_value": original_value,
            "suggested_value": None,
            "confidence": 0.0,
            "method": "openai_vision_page_ocr",
            "reason": f"OpenAI vision page OCR did not return a usable {target_field} candidate.",
            "evidence": evidence,
            "safe_to_auto_apply": False,
        }

    confidence = _openai_vision_confidence(
        target_field=target_field,
        candidate=candidate,
        row=row,
        page_selection_confidence=float(page_selection.get("confidence") or 0.0),
        model_confidence=model_confidence,
    )
    return {
        "success": True,
        "target_field": target_field,
        "original_value": original_value,
        "suggested_value": candidate,
        "confidence": confidence,
        "method": "openai_vision_page_ocr",
        "reason": f"OpenAI vision page OCR suggested a {target_field} candidate from page {pages[0]}.",
        "evidence": evidence,
        "safe_to_auto_apply": False,
    }


def ocr_fallback_row_repair(
    row: dict,
    diagnostics: Optional[dict] = None,
    target_field: Optional[str] = None,
    order_context: Optional[dict] = None,
    row_index: Optional[int] = None,
    pdf_id: Optional[str] = None,
    pdf_bytes: Optional[bytes] = None,
    openai_vision_repair_fn: Optional[OpenAIVisionRepairFn] = None,
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

    if _should_attempt_openai_vision_page_ocr(codes, target):
        return _openai_vision_page_ocr_repair(
            row=deepcopy(row or {}),
            diagnostics=deepcopy(diagnostics),
            target_field=target,
            order_context=deepcopy(order_context or {}),
            row_index=row_index,
            pdf_id=pdf_id,
            pdf_bytes=pdf_bytes,
            openai_vision_repair_fn=openai_vision_repair_fn,
        )

    location = _row_location(row)
    location_no_suggestion: Optional[Dict[str, Any]] = None
    if location:
        evidence = _row_location_evidence(row, codes)
        if target == "dimension":
            text_layer_suggestion = _text_layer_dimension_suggestion(row)
            if text_layer_suggestion:
                return {
                    "success": True,
                    "target_field": "dimension",
                    "original_value": original_value,
                    "suggested_value": text_layer_suggestion["dimension"],
                    "confidence": text_layer_suggestion["confidence"],
                    "method": "pdf_text_layer_row_region",
                    "reason": text_layer_suggestion["reason"],
                    "evidence": evidence,
                    "safe_to_auto_apply": False,
                }
        location_no_suggestion = {
            "success": False,
            "target_field": target,
            "original_value": original_value,
            "suggested_value": None,
            "confidence": float(location.get("confidence") or 0.0),
            "method": "pdf_text_layer_row_region",
            "reason": "Text-layer row region did not contain a confident correction.",
            "evidence": evidence,
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

    if location_no_suggestion is not None:
        return location_no_suggestion

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

    if "POSSIBLE_DIMENSION_FAMILY_MISMATCH" in codes:
        family = diagnostics.get("family_pattern") if isinstance(diagnostics.get("family_pattern"), dict) else {}
        suggested = family.get("suggested_value")
        likely_cause = "The dimension is valid, but it may not match a nearby or original order dimension family."
        if suggested:
            likely_cause = f"The dimension is valid, but {suggested} is a nearby/order family candidate."
        return _base_diagnosis(
            severity,
            f"Dimension may not match the nearby/order family pattern.{context_suffix}",
            likely_cause,
            "PATTERN_REPAIR",
            float(family.get("confidence") or 0.72),
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
    "attach_pdf_row_locations",
    "diagnose_extraction_row_issue",
    "diagnose_extraction_row_warning",
    "extract_pdf_text_for_row_location",
    "extract_pdf_text_layer_text",
    "ocr_fallback_row_repair",
]
