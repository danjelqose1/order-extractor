from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple

import fitz


TEXT_NOT_FOUND = "TEXT_NOT_FOUND"
MULTIPLE_MATCHES_NO_SAFE_BOUND_MATCH = "MULTIPLE_MATCHES_NO_SAFE_BOUND_MATCH"
SCANNED_OR_IMAGE_ONLY = "SCANNED_OR_IMAGE_ONLY"
VECTOR_OR_OUTLINED_TEXT = "VECTOR_OR_OUTLINED_TEXT"
FONT_REPLACEMENT_UNSUPPORTED = "FONT_REPLACEMENT_UNSUPPORTED"
VERIFICATION_FAILED = "VERIFICATION_FAILED"
UNKNOWN_NATIVE_EDIT_ERROR = "UNKNOWN_NATIVE_EDIT_ERROR"
NATIVE_EDIT_USED_CLEANUP_FALLBACK = "NATIVE_EDIT_USED_CLEANUP_FALLBACK"
BACKGROUND_CHANGED_AROUND_EDIT = "BACKGROUND_CHANGED_AROUND_EDIT"

CONTENT_STREAM_FIRST = "content_stream_first"
REDACT_INSERT_TIGHT = "redact_insert_tight"
SAFE_OVERLAY_FALLBACK = "safe_overlay_fallback"


@dataclass(frozen=True)
class NativeTextEditResult:
    success: bool
    edited_pdf_bytes: Optional[bytes] = None
    reason: Optional[str] = None
    warnings: Tuple[str, ...] = ()
    strategy: Optional[str] = None

    def to_api_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "success": self.success,
            "warnings": list(self.warnings),
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.strategy:
            payload["strategy"] = self.strategy
        return payload


def _normalize_rect(bounds: Optional[Dict[str, Any]], page_rect: fitz.Rect) -> Optional[fitz.Rect]:
    if not bounds:
        return None
    try:
        x = float(bounds.get("x", 0))
        y = float(bounds.get("y", 0))
        width = float(bounds.get("width", 0))
        height = float(bounds.get("height", 0))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None

    # Frontend coordinates are normalized top-left viewport coordinates.
    return fitz.Rect(
        page_rect.x0 + x * page_rect.width,
        page_rect.y0 + y * page_rect.height,
        page_rect.x0 + (x + width) * page_rect.width,
        page_rect.y0 + (y + height) * page_rect.height,
    )


def _rect_center_distance(a: fitz.Rect, b: fitz.Rect) -> float:
    ax = (a.x0 + a.x1) / 2
    ay = (a.y0 + a.y1) / 2
    bx = (b.x0 + b.x1) / 2
    by = (b.y0 + b.y1) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _choose_match(matches: List[fitz.Rect], selected_rect: Optional[fitz.Rect], page_rect: fitz.Rect) -> Optional[fitz.Rect]:
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    if selected_rect is None:
        return None

    ranked = sorted(matches, key=lambda rect: _rect_center_distance(rect, selected_rect))
    best = ranked[0]
    distance = _rect_center_distance(best, selected_rect)
    page_diagonal = (page_rect.width**2 + page_rect.height**2) ** 0.5
    if distance > max(24, page_diagonal * 0.18):
        return None
    return best


def _match_index(matches: List[fitz.Rect], target: fitz.Rect) -> int:
    return min(range(len(matches)), key=lambda idx: _rect_center_distance(matches[idx], target))


def _pdf_escape_literal(text: str) -> bytes:
    encoded = text.encode("latin-1", "ignore")
    encoded = encoded.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
    return encoded


def _replace_nth_occurrence_bytes(value: bytes, old: bytes, new: bytes, occurrence_index: int, seen: int) -> Tuple[bytes, int, bool]:
    if not old:
        return value, seen, False
    parts: List[bytes] = []
    offset = 0
    changed = False
    while True:
        idx = value.find(old, offset)
        if idx < 0:
            parts.append(value[offset:])
            break
        parts.append(value[offset:idx])
        if seen == occurrence_index and not changed:
            parts.append(new)
            changed = True
        else:
            parts.append(old)
        seen += 1
        offset = idx + len(old)
    return b"".join(parts), seen, changed


def _replace_text_in_hex_strings(stream: bytes, original_text: str, replacement_text: str, occurrence_index: int, seen: int) -> Tuple[bytes, int, bool]:
    changed = False
    original_encodings = [
        original_text.encode("utf-8"),
        original_text.encode("latin-1", "ignore"),
        original_text.encode("utf-16-be"),
        b"\xfe\xff" + original_text.encode("utf-16-be"),
    ]
    replacement_encodings = [
        replacement_text.encode("utf-8"),
        replacement_text.encode("latin-1", "ignore"),
        replacement_text.encode("utf-16-be"),
        b"\xfe\xff" + replacement_text.encode("utf-16-be"),
    ]

    def _replace(match: re.Match[bytes]) -> bytes:
        nonlocal seen, changed
        body = re.sub(rb"\s+", b"", match.group(1))
        try:
            decoded = bytes.fromhex(body.decode("ascii"))
        except Exception:
            return match.group(0)
        for old, new in zip(original_encodings, replacement_encodings):
            if old and old in decoded:
                updated, seen, did_change = _replace_nth_occurrence_bytes(decoded, old, new, occurrence_index, seen)
                if did_change:
                    changed = True
                    return b"<" + updated.hex().encode("ascii") + b">"
                return match.group(0)
        return match.group(0)

    return re.sub(rb"<([0-9A-Fa-f\s]+)>", _replace, stream), seen, changed


def _replace_text_in_literal_strings(stream: bytes, original_text: str, replacement_text: str, occurrence_index: int, seen: int) -> Tuple[bytes, int, bool]:
    changed = False
    old = _pdf_escape_literal(original_text)
    new = _pdf_escape_literal(replacement_text)

    def _replace(match: re.Match[bytes]) -> bytes:
        nonlocal seen, changed
        body = match.group(1)
        if old not in body:
            return match.group(0)
        updated, seen, did_change = _replace_nth_occurrence_bytes(body, old, new, occurrence_index, seen)
        if did_change:
            changed = True
            return b"(" + updated + b")"
        return match.group(0)

    return re.sub(rb"\(((?:\\.|[^\\)])*)\)", _replace, stream), seen, changed


def _try_content_stream_replace(doc: fitz.Document, page: fitz.Page, occurrence_index: int, original_text: str, replacement_text: str) -> bool:
    seen = 0
    changed_any = False
    for xref in page.get_contents() or []:
        stream = doc.xref_stream(xref)
        if not stream:
            continue
        updated, seen, changed = _replace_text_in_hex_strings(stream, original_text, replacement_text, occurrence_index, seen)
        if not changed:
            updated, seen, changed = _replace_text_in_literal_strings(stream, original_text, replacement_text, occurrence_index, seen)
        if changed:
            doc.update_stream(xref, updated)
            changed_any = True
            break
    return changed_any


def _text_exists_near(page: fitz.Page, text: str, target: fitz.Rect) -> bool:
    for rect in page.search_for(text):
        if _rect_center_distance(rect, target) < max(8, target.height * 1.5):
            return True
    return False


def _expanded_text_rect(rect: fitz.Rect, replacement_text: str, font_size: float, page_rect: fitz.Rect) -> fitz.Rect:
    pad_x = max(0.3, min(0.8, font_size * 0.045))
    pad_y = max(0.1, min(0.4, font_size * 0.025))
    estimated_width = max(rect.width, len(replacement_text or "") * font_size * 0.58)
    expanded = fitz.Rect(
        rect.x0 - pad_x,
        rect.y0 - pad_y,
        rect.x0 + estimated_width + pad_x,
        rect.y0 + max(rect.height, font_size * 1.25) + pad_y,
    )
    return expanded & page_rect


def _line_segments(page: fitz.Page) -> List[Tuple[fitz.Point, fitz.Point]]:
    segments: List[Tuple[fitz.Point, fitz.Point]] = []
    for drawing in page.get_drawings() or []:
        for item in drawing.get("items", []):
            if not item or item[0] != "l":
                continue
            segments.append((item[1], item[2]))
    return segments


def _avoid_nearby_lines(rect: fitz.Rect, page: fitz.Page) -> fitz.Rect:
    adjusted = fitz.Rect(rect)
    center_y = (adjusted.y0 + adjusted.y1) / 2
    center_x = (adjusted.x0 + adjusted.x1) / 2
    for start, end in _line_segments(page):
        horizontal = abs(start.y - end.y) < 0.35
        vertical = abs(start.x - end.x) < 0.35
        if horizontal:
            y = start.y
            overlaps_x = max(start.x, end.x) >= adjusted.x0 and min(start.x, end.x) <= adjusted.x1
            if overlaps_x and abs(y - adjusted.y0) < 1.4:
                adjusted.y0 = min(adjusted.y1 - 0.5, y + 0.4)
            elif overlaps_x and abs(y - adjusted.y1) < 1.4:
                adjusted.y1 = max(adjusted.y0 + 0.5, y - 0.4)
            elif overlaps_x and adjusted.y0 < y < adjusted.y1:
                if y <= center_y:
                    adjusted.y0 = min(adjusted.y1 - 0.5, y + 0.4)
                else:
                    adjusted.y1 = max(adjusted.y0 + 0.5, y - 0.4)
        if vertical:
            x = start.x
            overlaps_y = max(start.y, end.y) >= adjusted.y0 and min(start.y, end.y) <= adjusted.y1
            if overlaps_y and abs(x - adjusted.x0) < 1.4:
                adjusted.x0 = min(adjusted.x1 - 0.5, x + 0.4)
            elif overlaps_y and abs(x - adjusted.x1) < 1.4:
                adjusted.x1 = max(adjusted.x0 + 0.5, x - 0.4)
            elif overlaps_y and adjusted.x0 < x < adjusted.x1:
                if x <= center_x:
                    adjusted.x0 = min(adjusted.x1 - 0.5, x + 0.4)
                else:
                    adjusted.x1 = max(adjusted.x0 + 0.5, x - 0.4)
    return adjusted


def _insert_replacement_text(page: fitz.Page, rect: fitz.Rect, replacement_text: str, font_size: float) -> None:
    # Keep the first implementation conservative: use a standard PDF base font.
    written = page.insert_textbox(
        rect,
        replacement_text,
        fontsize=font_size,
        fontname="helv",
        color=(0, 0, 0),
        align=fitz.TEXT_ALIGN_LEFT,
    )
    if written < 0:
        # Fall back to point insertion if textbox wrapping/clipping rejects the fit.
        baseline_y = min(rect.y1 - 1, rect.y0 + font_size)
        page.insert_text(
            fitz.Point(rect.x0, baseline_y),
            replacement_text,
            fontsize=font_size,
            fontname="helv",
            color=(0, 0, 0),
        )


def native_text_replace(
    pdf_bytes: bytes,
    *,
    page_index: int,
    original_text: str,
    replacement_text: str,
    bounds: Optional[Dict[str, Any]] = None,
    font_size: Optional[float] = None,
    font_family: Optional[str] = None,
    native_edit_mode: str = CONTENT_STREAM_FIRST,
    preserve_nearby_lines: bool = True,
) -> NativeTextEditResult:
    del font_family  # Font matching can be expanded later; helv is the safe v1 base font.
    warnings: List[str] = []
    if not pdf_bytes:
        return NativeTextEditResult(False, reason=UNKNOWN_NATIVE_EDIT_ERROR)
    original_text = str(original_text or "")
    replacement_text = str(replacement_text or "")
    if not original_text.strip():
        return NativeTextEditResult(False, reason=TEXT_NOT_FOUND)

    try:
        doc = fitz.open(stream=bytes(pdf_bytes), filetype="pdf")
        if page_index < 0 or page_index >= doc.page_count:
            doc.close()
            return NativeTextEditResult(False, reason=TEXT_NOT_FOUND)

        page = doc.load_page(page_index)
        extracted_text = page.get_text("text") or ""
        if not extracted_text.strip():
            doc.close()
            return NativeTextEditResult(False, reason=SCANNED_OR_IMAGE_ONLY)

        matches = page.search_for(original_text)
        if not matches:
            doc.close()
            return NativeTextEditResult(False, reason=TEXT_NOT_FOUND)

        selected_rect = _normalize_rect(bounds, page.rect)
        target = _choose_match(matches, selected_rect, page.rect)
        if target is None:
            doc.close()
            return NativeTextEditResult(False, reason=MULTIPLE_MATCHES_NO_SAFE_BOUND_MATCH)

        safe_font_size = float(font_size or 0)
        if safe_font_size <= 0:
            safe_font_size = max(6, min(72, target.height * 0.78))

        occurrence_index = _match_index(matches, target)
        if native_edit_mode == SAFE_OVERLAY_FALLBACK:
            doc.close()
            return NativeTextEditResult(False, reason=FONT_REPLACEMENT_UNSUPPORTED)

        if native_edit_mode == CONTENT_STREAM_FIRST:
            if _try_content_stream_replace(doc, page, occurrence_index, original_text, replacement_text):
                edited_bytes = doc.tobytes(garbage=4, deflate=True)
                doc.close()
                verify_doc = fitz.open(stream=edited_bytes, filetype="pdf")
                verify_page = verify_doc.load_page(page_index)
                verify_text = verify_page.get_text("text") or ""
                original_still_nearby = _text_exists_near(verify_page, original_text, target) if original_text != replacement_text else False
                verify_doc.close()
                if replacement_text and replacement_text not in verify_text:
                    return NativeTextEditResult(False, reason=VERIFICATION_FAILED, warnings=tuple(warnings))
                if original_still_nearby:
                    return NativeTextEditResult(False, reason=VERIFICATION_FAILED, warnings=tuple(warnings))
                return NativeTextEditResult(True, edited_pdf_bytes=edited_bytes, warnings=tuple(warnings), strategy="content_stream")

        cover_rect = _expanded_text_rect(target, original_text, safe_font_size, page.rect)
        if preserve_nearby_lines:
            cover_rect = _avoid_nearby_lines(cover_rect, page)
        text_rect = _expanded_text_rect(target, replacement_text, safe_font_size, page.rect)

        warnings.append(NATIVE_EDIT_USED_CLEANUP_FALLBACK)
        warnings.append(BACKGROUND_CHANGED_AROUND_EDIT)
        page.add_redact_annot(cover_rect, fill=(1, 1, 1))
        page.apply_redactions()
        _insert_replacement_text(page, text_rect, replacement_text, safe_font_size)

        edited_bytes = doc.tobytes(garbage=4, deflate=True)
        doc.close()

        verify_doc = fitz.open(stream=edited_bytes, filetype="pdf")
        verify_page = verify_doc.load_page(page_index)
        verify_text = verify_page.get_text("text") or ""
        verify_doc.close()
        if replacement_text and replacement_text not in verify_text:
            return NativeTextEditResult(False, reason=VERIFICATION_FAILED, warnings=tuple(warnings))

        return NativeTextEditResult(True, edited_pdf_bytes=edited_bytes, warnings=tuple(warnings), strategy="redact_insert_tight")
    except fitz.FileDataError:
        return NativeTextEditResult(False, reason=UNKNOWN_NATIVE_EDIT_ERROR)
    except Exception as exc:
        return NativeTextEditResult(False, reason=UNKNOWN_NATIVE_EDIT_ERROR, warnings=(str(exc),))
