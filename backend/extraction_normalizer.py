from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple


DIMENSION_IN_TYPE_WARNING = "Dimension-like text was removed from type field"
_DIMENSION_TOKEN_RE = re.compile(r"(?<!\d)(\d{2,5})\s*[xX×]\s*(\d{2,5})(?!\d)")
_KELI_MARKER_RE = re.compile(r"\bKELI(?:\s+ALBANIA)?\b", re.IGNORECASE)
_KELI_CORRELATED_DOCUMENT_RE = re.compile(
    r"\bDOCUMENTO\s+CORRELATO\b\s*[:\-]?\s*"
    r"(?P<order>R\s*-\s*\d{2}\s*-\s*\d{4})\b",
    re.IGNORECASE,
)
_KELI_GLASS_ORDER_RE = re.compile(
    r"\bORDINE\s+DI\s+VETRO\b\s*[:#\-]?\s*"
    r"(?P<document>\d{2}\s*-\s*\d{4})\b",
    re.IGNORECASE,
)


def _is_keli_format(source_text: str, vendor_format: str = "") -> bool:
    explicit_format = re.sub(r"[^A-Z0-9]+", "", str(vendor_format or "").upper())
    return explicit_format in {"KELI", "KELIALBANIA"} or bool(
        _KELI_MARKER_RE.search(source_text or "")
    )


def _normalize_keli_order_number(value: str) -> str:
    compact = re.sub(r"\s+", "", str(value or "").upper())
    match = re.fullmatch(r"R-(\d{2})-(\d{4})", compact)
    return f"R-{match.group(1)}-{match.group(2)}" if match else ""


def _normalize_keli_supplier_document_number(value: str) -> str:
    compact = re.sub(r"\s+", "", str(value or ""))
    match = re.fullmatch(r"(\d{2})-(\d{4})", compact)
    return f"{match.group(1)}-{match.group(2)}" if match else ""


def normalize_order_metadata(
    payload: Dict[str, Any],
    source_text: str,
    *,
    vendor_format: str = "",
) -> Dict[str, Any]:
    """Apply deterministic, format-specific order metadata without mutating inputs."""
    normalized = dict(payload or {})
    rows = [dict(row) if isinstance(row, dict) else row for row in normalized.get("rows") or []]
    normalized["rows"] = rows

    if not _is_keli_format(source_text, vendor_format):
        return normalized

    correlated_match = _KELI_CORRELATED_DOCUMENT_RE.search(source_text or "")
    if not correlated_match:
        return normalized

    order_number = _normalize_keli_order_number(correlated_match.group("order"))
    if not order_number:
        return normalized

    normalized["order_number"] = order_number
    for row in rows:
        if isinstance(row, dict):
            row["order_number"] = order_number

    supplier_match = _KELI_GLASS_ORDER_RE.search(source_text or "")
    if supplier_match:
        supplier_document_number = _normalize_keli_supplier_document_number(
            supplier_match.group("document")
        )
        if supplier_document_number:
            normalized["supplier_document_number"] = supplier_document_number

    return normalized


def _dimension_key_from_parts(width: str, height: str) -> str:
    try:
        return f"{int(width)}x{int(height)}"
    except (TypeError, ValueError):
        return f"{width}x{height}".lower()


def _dimension_key(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    match = re.fullmatch(r"(\d{2,5})\s*[xX×]\s*(\d{2,5})", text)
    if not match:
        return ""
    return _dimension_key_from_parts(match.group(1), match.group(2))


def _normalize_type_whitespace(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value or "")
    cleaned = re.sub(r"\s+,", ",", cleaned)
    return cleaned.strip()


def _dimension_has_type_context(type_text: str, start: int, end: int) -> bool:
    before = type_text[:start]
    after = type_text[end:]
    has_glass_before = bool(
        re.search(
            r"\b(?:vetri?|lowe|satinat|satinato|float|strat|laminat|tempered|33\.1|\d+\s*vetri)\b",
            before,
            flags=re.IGNORECASE,
        )
    )
    has_spacer_after = bool(
        re.search(
            r"(?:\bc\.?\s*caldo\b|\bcaldo\b|\b\d+\s*mm\b)",
            after,
            flags=re.IGNORECASE,
        )
    )
    return has_glass_before and has_spacer_after


def normalizeExtractedRow(row: Dict[str, Any]) -> Dict[str, Any]:
    """Clean one extracted row without mutating the caller's row object."""
    working = dict(row or {})
    type_text = "" if working.get("type") is None else str(working.get("type"))
    dimension_key = _dimension_key(working.get("dimension"))
    removed_dimension = False

    def replace_dimension(match: re.Match[str]) -> str:
        nonlocal removed_dimension
        found_key = _dimension_key_from_parts(match.group(1), match.group(2))
        should_remove = (dimension_key and found_key == dimension_key) or _dimension_has_type_context(
            type_text,
            match.start(),
            match.end(),
        )
        if not should_remove:
            return match.group(0)
        removed_dimension = True
        return " "

    cleaned_type = _DIMENSION_TOKEN_RE.sub(replace_dimension, type_text)
    cleaned_type = _normalize_type_whitespace(cleaned_type)
    if cleaned_type != type_text:
        working["type"] = cleaned_type

    if removed_dimension:
        existing = list(working.get("_normalization_warnings") or [])
        if DIMENSION_IN_TYPE_WARNING not in existing:
            existing.append(DIMENSION_IN_TYPE_WARNING)
        working["_normalization_warnings"] = existing
    return working


def normalize_extracted_rows(rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    normalized_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for row in rows or []:
        normalized = normalizeExtractedRow(row)
        for warning in normalized.pop("_normalization_warnings", []) or []:
            if warning not in warnings:
                warnings.append(warning)
        normalized_rows.append(normalized)
    return normalized_rows, warnings


__all__ = [
    "DIMENSION_IN_TYPE_WARNING",
    "normalizeExtractedRow",
    "normalize_extracted_rows",
    "normalize_order_metadata",
]
