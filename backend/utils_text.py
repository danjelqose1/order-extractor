from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple


ORDER_NUMBER_RE = re.compile(r"R-?\s*\d{2}-\d{4}", re.IGNORECASE)
TYPE_LINE_RE = re.compile(r"\b\d+\s+VETRI\b.*", re.IGNORECASE)
DIMENSION_RE = re.compile(r"\b\d{3,4}\s*[x×]\s*\d{3,4}\b")
CLIENT_RE = re.compile(r"CLIENTE\s*[:\-]?\s*(.+)", re.IGNORECASE)
TOTAL_LINE_RE = re.compile(r"^(?:m2|totale)\s+(\d+)\s+([\d.,]+)", re.IGNORECASE)
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
    return declared_units, declared_area


__all__ = [
    "normalize_position",
    "normalize_order_number",
    "clean_dimension",
    "compute_area_from_dimension",
    "extract_client_hint",
    "build_signature",
    "parse_declared_totals",
]
