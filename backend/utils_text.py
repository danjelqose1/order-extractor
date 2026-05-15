from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


ORDER_NUMBER_RE = re.compile(r"R-?\s*\d{2}-\d{4}", re.IGNORECASE)
TYPE_LINE_RE = re.compile(r"\b\d+\s+VETRI\b.*", re.IGNORECASE)
DIMENSION_RE = re.compile(r"\b\d{3,4}\s*[x×]\s*\d{3,4}\b")
CLIENT_RE = re.compile(r"CLIENTE\s*[:\-]?\s*(.+)", re.IGNORECASE)
TOTAL_LINE_RE = re.compile(r"^(?:m2|totale)\s+(\d+)\s+([\d.,]+)", re.IGNORECASE)
TOTAL_KEYWORD_RE = re.compile(r"\b(?:totale|total|totali)\b", re.IGNORECASE)
UNITS_LABEL_RE = re.compile(r"\b(?:pezzi|pezzo|pezz[i1]|pz|pieces?|units?)\b", re.IGNORECASE)
AREA_LABEL_RE = re.compile(r"\b(?:m2|m²|mq)\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?(?![A-Za-z])")
ORDER_TOTAL_AREA_TOLERANCE = 0.05
AREA_QTY_FRAGMENT_RE = re.compile(r"\d+(?:[.,]\d+)?")


def normalize_order_number(value: str) -> str:
    if not value:
        return ""
    cleaned = value.strip().upper().replace(" ", "")
    match = re.search(r"R-?\d{2}-\d{4}", cleaned)
    if not match:
        return cleaned
    token = match.group(0)
    token = token.replace(" ", "").replace("R-", "R-")
    token = token.replace("R", "R-") if token.startswith("R") and not token.startswith("R-") else token
    token = token.replace("--", "-")
    prefix, rest = token.split("-", 1)
    if len(rest) == 6 and rest[2] != "-":
        rest = f"{rest[:2]}-{rest[2:]}"
    return f"{prefix}-{rest}".upper()


def normalize_position(value: str) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"\s+", "", value.strip())
    cleaned = re.sub(r"^[A-Za-z]-?\d{2,}-\d{4}/", "", cleaned, flags=re.IGNORECASE)
    segments = cleaned.split("/")
    base = segments[0] if segments else ""
    suffix = "/".join(segments[1:]) if len(segments) > 1 else ""
    if base and "-" not in base:
        digits = re.findall(r"\d+", base)
        if len(digits) >= 2:
            base = f"{digits[0]}-{digits[1]}"
        elif len(digits) == 1 and len(digits[0]) >= 3:
            base = f"{digits[0][:-1]}-{digits[0][-1]}"
    base = base.strip()
    if suffix:
        return f"{base}/{suffix.strip()}" if base else suffix.strip()
    return base


def clean_dimension(value: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    if not value:
        return "", None
    token = value.strip().lower().replace("×", "x").replace(" ", "")
    match = re.match(r"^(\d{3,4})x(\d{3,4})$", token)
    if not match:
        return "", None
    width, height = int(match.group(1)), int(match.group(2))
    return f"{width}x{height}", (width, height)


def compute_area_from_dimension(dimension: str) -> Optional[float]:
    _, dims = clean_dimension(dimension)
    if not dims:
        return None
    width, height = dims
    return round((width * height) / 1_000_000, 3)


def extract_client_hint(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = CLIENT_RE.search(stripped)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def build_signature(prepared_text: str, extra_rows: Optional[Iterable[Dict[str, str]]] = None) -> Dict[str, str]:
    text = prepared_text or ""
    order_numbers = {
        normalize_order_number(match.group(0))
        for match in ORDER_NUMBER_RE.finditer(text)
    }
    type_lines = [match.group(0).strip() for match in TYPE_LINE_RE.finditer(text)]
    dims = Counter(dim.strip().lower().replace("×", "x").replace(" ", "") for dim in DIMENSION_RE.findall(text))
    client = extract_client_hint(text) or ""

    if extra_rows:
        for row in extra_rows:
            num = normalize_order_number(str(row.get("order_number", "")))
            if num:
                order_numbers.add(num)
            dim = (row.get("dimension") or "").strip().lower().replace("×", "x").replace(" ", "")
            if dim:
                dims[dim] += 1
            typ = (row.get("type") or "").strip()
            if typ and typ not in type_lines:
                type_lines.append(typ)

    top_dims = [item[0] for item in dims.most_common(3)]
    signature_parts = [
        "orders:" + ",".join(sorted(order_numbers)),
        "types:" + " | ".join(type_lines[:4]),
        "dims:" + ",".join(top_dims),
        "client:" + client,
    ]
    signature = "\n".join(signature_parts)
    pattern_hash = hashlib.sha1(signature.encode("utf-8", "ignore")).hexdigest()
    pattern_text = signature + "\n\n" + text[:1500]
    return {
        "signature": signature,
        "pattern_hash": pattern_hash,
        "pattern_text": pattern_text,
        "order_numbers": ",".join(sorted(order_numbers)),
    }


def parse_declared_totals(text: Optional[str]) -> Tuple[Optional[int], Optional[float]]:
    if not text:
        return None, None
    declared_units = None
    declared_area = None
    for line in text.splitlines()[::-1]:
        stripped = line.strip()
        if not stripped:
            continue
        match = TOTAL_LINE_RE.match(stripped)
        if match:
            try:
                declared_units = int(match.group(1))
            except Exception:
                declared_units = None
            try:
                declared_area = float(match.group(2).replace(".", "").replace(",", "."))
            except Exception:
                declared_area = None
            if declared_units or declared_area:
                break
        if not TOTAL_KEYWORD_RE.search(stripped):
            continue
        units, area = _parse_total_line(stripped)
        if units is not None or area is not None:
            declared_units = units
            declared_area = area
            break
    return declared_units, declared_area


def _parse_decimal_number(value: str) -> Optional[float]:
    token = str(value or "").strip().replace(" ", "")
    if not token:
        return None
    if "," in token and "." in token:
        token = token.replace(".", "").replace(",", ".")
    elif "," in token:
        token = token.replace(",", ".")
    try:
        parsed = float(token)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _parse_total_line(line: str) -> Tuple[Optional[int], Optional[float]]:
    normalized = str(line or "").replace("m²", "m2")
    units = None
    area = None

    unit_patterns = [
        r"(?:pezzi|pezzo|pz|pieces?|units?)\D{0,24}(\d+)",
        r"(\d+)\D{0,24}(?:pezzi|pezzo|pz|pieces?|units?)",
    ]
    for pattern in unit_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            try:
                units = int(match.group(1))
            except ValueError:
                units = None
            break

    area_patterns = [
        r"(?:m2|mq)\s*[:=]?\s*(\d+(?:[.,]\d+)?)",
        r"(\d+(?:[.,]\d+)?)\s*(?:m2|mq)",
    ]
    for pattern in area_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            area = _parse_decimal_number(match.group(1))
            break

    numbers = NUMBER_RE.findall(normalized)
    if (units is None or area is None) and len(numbers) >= 2:
        if units is None:
            try:
                units = int(float(numbers[0].replace(",", ".")))
            except ValueError:
                units = None
        if area is None:
            area = _parse_decimal_number(numbers[-1])

    if not (TOTAL_KEYWORD_RE.search(normalized) and (UNITS_LABEL_RE.search(normalized) or AREA_LABEL_RE.search(normalized) or len(numbers) >= 2)):
        return None, None
    return units, area


def build_order_total_diagnostics(
    pdf_units: Optional[int],
    pdf_area: Optional[float],
    extracted_units: Optional[int],
    extracted_area: Optional[float],
) -> Optional[Dict[str, Any]]:
    if pdf_units is None and pdf_area is None:
        return None
    extracted_units_value = int(extracted_units or 0)
    extracted_area_value = round(float(extracted_area or 0.0), 3)
    pdf_area_value = round(float(pdf_area), 3) if pdf_area is not None else None
    unit_delta = int(pdf_units) - extracted_units_value if pdf_units is not None else None
    area_delta = round(pdf_area_value - extracted_area_value, 3) if pdf_area_value is not None else None

    unit_mismatch = unit_delta is not None and unit_delta != 0
    area_mismatch = area_delta is not None and abs(area_delta) > ORDER_TOTAL_AREA_TOLERANCE
    if not unit_mismatch and not area_mismatch:
        return None

    pdf_units_text = str(pdf_units) if pdf_units is not None else "unknown"
    pdf_area_text = f"{pdf_area_value:.3f}" if pdf_area_value is not None else "unknown"
    extracted_area_text = f"{extracted_area_value:.3f}"
    missing_parts: List[str] = []
    if unit_delta is not None and unit_delta > 0:
        missing_parts.append(f"{unit_delta} units")
    elif unit_delta is not None and unit_delta < 0:
        missing_parts.append(f"{abs(unit_delta)} extra units")
    if area_delta is not None and area_delta > ORDER_TOTAL_AREA_TOLERANCE:
        missing_parts.append(f"{area_delta:.3f} m²")
    elif area_delta is not None and area_delta < -ORDER_TOTAL_AREA_TOLERANCE:
        missing_parts.append(f"{abs(area_delta):.3f} extra m²")
    delta_text = " / ".join(missing_parts) if missing_parts else "a total mismatch"
    delta_label = "Difference" if "extra" in delta_text else "Missing"

    return {
        "pdf_units": pdf_units,
        "extracted_units": extracted_units_value,
        "unit_delta": unit_delta,
        "pdf_area": pdf_area_value,
        "extracted_area": extracted_area_value,
        "area_delta": area_delta,
        "severity": "warning",
        "message": (
            f"PDF total says {pdf_units_text} units / {pdf_area_text} m², "
            f"but extracted rows total {extracted_units_value} units / {extracted_area_text} m². "
            f"{delta_label} {delta_text}."
        ),
    }


__all__ = [
    "normalize_position",
    "normalize_order_number",
    "clean_dimension",
    "compute_area_from_dimension",
    "extract_client_hint",
    "build_signature",
    "parse_declared_totals",
    "build_order_total_diagnostics",
]
