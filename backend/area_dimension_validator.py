from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


DIMENSION_PARSE_RE = re.compile(r"^(\d{3,4})x(\d{2,4})$")


def _parse_area_source(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace(" ", "")
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except Exception:
        return None


def _parse_dimension(value: Any) -> Tuple[Optional[int], Optional[int], str]:
    if value is None:
        return None, None, ""
    raw = str(value)
    token = raw.strip().lower().replace("×", "x").replace(" ", "")
    match = DIMENSION_PARSE_RE.match(token)
    if not match:
        return None, None, raw
    try:
        width = int(match.group(1))
        height = int(match.group(2))
    except Exception:
        return None, None, raw
    return width, height, raw


def _ensure_flags(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


def _apply_flag(flags: List[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def apply_area_dimension_validation(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    output: List[Dict[str, Any]] = []
    for row in rows:
        working = dict(row)
        width_mm, height_mm, dimension_raw = _parse_dimension(working.get("dimension"))
        area_source = _parse_area_source(working.get("area"))
        area_computed = None
        if width_mm is not None and height_mm is not None:
            area_computed = round((width_mm * height_mm) / 1_000_000, 3)

        area_final = area_source if area_source is not None else area_computed
        flags = _ensure_flags(working.get("flags"))

        if area_source is not None and area_computed is not None:
            diff = abs(area_source - area_computed)
            tolerance = max(0.01, abs(area_source) * 0.02)
            if diff > tolerance:
                _apply_flag(flags, "AREA_DIMENSION_MISMATCH")
                if height_mm is not None and (height_mm < 400 or len(str(height_mm)) <= 3):
                    _apply_flag(flags, "SUSPICIOUS_TRUNCATION")
                if width_mm:
                    suggested_height = round((area_source * 1_000_000) / width_mm)
                    if 400 <= suggested_height <= 4000:
                        working["suggested_height"] = suggested_height
                        working["suggested_dimension"] = f"{width_mm}x{suggested_height}"
                        _apply_flag(flags, "SUGGESTED_DIMENSION_FROM_AREA")

        working["dimension_raw"] = dimension_raw
        working["width_mm"] = width_mm
        working["height_mm"] = height_mm
        working["area_source"] = area_source
        working["area_computed"] = area_computed
        working["area_final"] = area_final
        working["flags"] = flags
        if area_computed is not None:
            working["computed_area"] = area_computed
        if area_source is not None and area_computed is not None:
            working["area_mismatch"] = "AREA_DIMENSION_MISMATCH" in flags

        output.append(working)
    return output


__all__ = ["apply_area_dimension_validation"]
