from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


SUSPICIOUS_DIMENSION_RE = re.compile(r"^\d{3,4}x\d{2,3}$")
WINDOW_DIMENSION_RE = re.compile(r"(\d{3,4})\s*[x×]\s*(\d{3,4})")
ORDER_PREFIX_RE = re.compile(r"^[A-Za-z]-?\d{2,}-\d{4}/", re.IGNORECASE)


def _normalize_dimension_token(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("×", "x").replace(" ", "")


def _extract_width_height(item: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    width = item.get("width")
    height = item.get("height")
    if width is not None and height is not None:
        try:
            return int(width), int(height)
        except Exception:
            pass
    token = _normalize_dimension_token(item.get("dimension"))
    match = re.match(r"^(\d{3,4})x(\d{2,4})$", token)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _is_suspicious(item: Dict[str, Any]) -> bool:
    width, height = _extract_width_height(item)
    if width is not None and height is not None:
        if height < 400 and width >= 300:
            return True
    token = _normalize_dimension_token(item.get("dimension"))
    if token and SUSPICIOUS_DIMENSION_RE.match(token):
        return True
    return False


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _build_position_keys(position: str, order_number: str) -> List[str]:
    pos = (position or "").strip()
    order = (order_number or "").strip()
    if not pos:
        return []
    stripped = ORDER_PREFIX_RE.sub("", pos)
    base = pos.split("/")[0] if pos else ""
    base_stripped = stripped.split("/")[0] if stripped else ""
    keys: List[str] = []
    if order and pos:
        keys.append(f"{order}/{pos}")
    keys.append(pos)
    if stripped and stripped != pos:
        keys.append(stripped)
    if order and stripped and stripped != pos:
        keys.append(f"{order}/{stripped}")
    if base and base != pos:
        keys.append(base)
    if order and base:
        keys.append(f"{order}/{base}")
    if base_stripped and base_stripped not in (base, stripped):
        keys.append(base_stripped)
    if order and base_stripped and base_stripped not in (base, stripped):
        keys.append(f"{order}/{base_stripped}")
    return _dedupe_keep_order(keys)


def _compile_position_patterns(key: str) -> List[re.Pattern[str]]:
    cleaned = (key or "").strip()
    if not cleaned:
        return []
    escaped = re.escape(cleaned).replace(r"\ ", r"\s*")
    patterns = [re.compile(escaped, re.IGNORECASE)]
    compact = re.sub(r"\s+", "", cleaned)
    if compact:
        loose = r"\s*".join(re.escape(ch) for ch in compact)
        if loose != escaped:
            patterns.append(re.compile(loose, re.IGNORECASE))
    return patterns


def _find_text_window(text: str, keys: Iterable[str], window: int = 300) -> str:
    if not text:
        return ""
    for key in keys:
        for pattern in _compile_position_patterns(key):
            match = pattern.search(text)
            if match:
                start = max(0, match.start() - window)
                end = min(len(text), match.end() + window)
                return text[start:end]
    return ""


def _choose_best_match(
    candidates: Iterable[Tuple[int, int]],
    width_target: int,
) -> Optional[Tuple[int, int]]:
    best: Optional[Tuple[int, int]] = None
    best_score = 2
    for width, height in candidates:
        if len(str(height)) != 4 or height < 400:
            continue
        if abs(width - width_target) > 2:
            continue
        score = 0 if width == width_target else 1
        if score < best_score:
            best = (width, height)
            best_score = score
            if score == 0:
                break
    return best


def _apply_flag(item: Dict[str, Any], flag: str) -> None:
    flags = item.get("flags")
    if not isinstance(flags, list):
        flags = []
    if flag not in flags:
        flags.append(flag)
    item["flags"] = flags


def apply_dimension_repair(raw_text: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return []
    text = raw_text or ""
    output: List[Dict[str, Any]] = []
    for item in items:
        working = dict(item)
        if not _is_suspicious(working):
            output.append(working)
            continue

        _apply_flag(working, "SUSPICIOUS_TRUNCATION")
        if not text:
            output.append(working)
            continue

        width_current, _ = _extract_width_height(working)
        if width_current is None:
            output.append(working)
            continue

        keys = _build_position_keys(
            working.get("position", ""),
            working.get("order_number", ""),
        )
        window_text = _find_text_window(text, keys)
        if not window_text:
            output.append(working)
            continue

        candidates: List[Tuple[int, int]] = []
        for match in WINDOW_DIMENSION_RE.finditer(window_text):
            try:
                candidates.append((int(match.group(1)), int(match.group(2))))
            except Exception:
                continue
        best = _choose_best_match(candidates, width_current)
        if best and not working.get("correctedDimension"):
            width_new, height_new = best
            working["correctedWidth"] = width_new
            working["correctedHeight"] = height_new
            working["correctedDimension"] = f"{width_new}x{height_new}"
            working["dimension_source"] = "raw_repair"
            working["repair_reason"] = "column_boundary_truncation"

        output.append(working)

    return output


__all__ = ["apply_dimension_repair"]
