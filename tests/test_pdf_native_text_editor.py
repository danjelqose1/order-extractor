from __future__ import annotations

import sys
from pathlib import Path

import fitz


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.pdf_native_text_editor import (  # noqa: E402
    BACKGROUND_CHANGED_AROUND_EDIT,
    MULTIPLE_MATCHES_NO_SAFE_BOUND_MATCH,
    NATIVE_EDIT_USED_CLEANUP_FALLBACK,
    SCANNED_OR_IMAGE_ONLY,
    TEXT_NOT_FOUND,
    native_text_replace,
)


def _make_pdf(lines: list[tuple[float, float, str]]) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=300, height=180)
    for x, y, text in lines:
        page.insert_text(fitz.Point(x, y), text, fontsize=12, fontname="helv", color=(0, 0, 0))
    data = doc.tobytes()
    doc.close()
    return data


def _make_table_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=300, height=180)
    page.draw_rect(fitz.Rect(40, 40, 180, 90), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(110, 40), fitz.Point(110, 90), color=(0, 0, 0), width=1)
    page.insert_text(fitz.Point(62, 68), "93", fontsize=12, fontname="helv", color=(0, 0, 0))
    data = doc.tobytes()
    doc.close()
    return data


def _drawing_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = len(doc.load_page(0).get_drawings())
    doc.close()
    return count


def _extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = doc.load_page(0).get_text("text")
    doc.close()
    return text


def _normalized_bounds_for(pdf_bytes: bytes, text: str, occurrence: int = 0) -> dict[str, float]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    rect = page.search_for(text)[occurrence]
    bounds = {
        "x": rect.x0 / page.rect.width,
        "y": rect.y0 / page.rect.height,
        "width": rect.width / page.rect.width,
        "height": rect.height / page.rect.height,
    }
    doc.close()
    return bounds


def test_native_text_replace_simple_decimal_keeps_original_bytes_unchanged():
    pdf_bytes = _make_pdf([(50, 50, "Thickness 1.0")])
    original_copy = bytes(pdf_bytes)
    result = native_text_replace(
        pdf_bytes,
        page_index=0,
        original_text="1.0",
        replacement_text="1.1",
        bounds=_normalized_bounds_for(pdf_bytes, "1.0"),
        font_size=12,
    )

    assert result.success
    assert result.strategy == "content_stream"
    assert result.warnings == ()
    assert result.edited_pdf_bytes
    assert pdf_bytes == original_copy
    assert "1.1" in _extract_text(result.edited_pdf_bytes)


def test_native_text_replace_percent_value():
    pdf_bytes = _make_pdf([(50, 50, "Completion 47%")])
    result = native_text_replace(
        pdf_bytes,
        page_index=0,
        original_text="47%",
        replacement_text="48%",
        bounds=_normalized_bounds_for(pdf_bytes, "47%"),
        font_size=12,
    )

    assert result.success
    assert result.strategy == "content_stream"
    assert "48%" in _extract_text(result.edited_pdf_bytes or b"")


def test_native_text_replace_uses_selected_bounds_for_repeated_values():
    pdf_bytes = _make_pdf([(50, 50, "Value 1.0"), (50, 100, "Value 1.0")])
    result = native_text_replace(
        pdf_bytes,
        page_index=0,
        original_text="1.0",
        replacement_text="1.1",
        bounds=_normalized_bounds_for(pdf_bytes, "1.0", occurrence=1),
        font_size=12,
    )

    assert result.success
    assert result.strategy == "content_stream"
    doc = fitz.open(stream=result.edited_pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    replacement_rect = page.search_for("1.1")[0]
    second_original_rect = fitz.Rect(
        _normalized_bounds_for(pdf_bytes, "1.0", occurrence=1)["x"] * page.rect.width,
        _normalized_bounds_for(pdf_bytes, "1.0", occurrence=1)["y"] * page.rect.height,
        (_normalized_bounds_for(pdf_bytes, "1.0", occurrence=1)["x"] + _normalized_bounds_for(pdf_bytes, "1.0", occurrence=1)["width"]) * page.rect.width,
        (_normalized_bounds_for(pdf_bytes, "1.0", occurrence=1)["y"] + _normalized_bounds_for(pdf_bytes, "1.0", occurrence=1)["height"]) * page.rect.height,
    )
    doc.close()
    assert abs(replacement_rect.y0 - second_original_rect.y0) < 16


def test_native_text_replace_text_not_found_returns_clean_reason():
    pdf_bytes = _make_pdf([(50, 50, "Thickness 1.0")])
    result = native_text_replace(
        pdf_bytes,
        page_index=0,
        original_text="missing",
        replacement_text="found",
    )

    assert not result.success
    assert result.reason == TEXT_NOT_FOUND


def test_native_text_replace_requires_bounds_for_ambiguous_repeated_values():
    pdf_bytes = _make_pdf([(50, 50, "Value 1.0"), (50, 100, "Value 1.0")])
    result = native_text_replace(
        pdf_bytes,
        page_index=0,
        original_text="1.0",
        replacement_text="1.1",
    )

    assert not result.success
    assert result.reason == MULTIPLE_MATCHES_NO_SAFE_BOUND_MATCH


def test_native_text_replace_blank_or_scanned_page_returns_clean_reason():
    doc = fitz.open()
    doc.new_page(width=300, height=180)
    pdf_bytes = doc.tobytes()
    doc.close()

    result = native_text_replace(
        pdf_bytes,
        page_index=0,
        original_text="1.0",
        replacement_text="1.1",
    )

    assert not result.success
    assert result.reason == SCANNED_OR_IMAGE_ONLY


def test_content_stream_edit_preserves_table_lines_without_redaction_patch():
    pdf_bytes = _make_table_pdf()
    result = native_text_replace(
        pdf_bytes,
        page_index=0,
        original_text="93",
        replacement_text="94",
        bounds=_normalized_bounds_for(pdf_bytes, "93"),
        font_size=12,
        native_edit_mode="content_stream_first",
        preserve_nearby_lines=True,
    )

    assert result.success
    assert result.strategy == "content_stream"
    assert result.warnings == ()
    assert _drawing_count(result.edited_pdf_bytes or b"") == _drawing_count(pdf_bytes)
    assert "94" in _extract_text(result.edited_pdf_bytes or b"")


def test_content_stream_edit_near_watermark_avoids_cleanup_warning():
    doc = fitz.open()
    page = doc.new_page(width=300, height=180)
    page.insert_text(fitz.Point(35, 120), "WATERMARK", fontsize=30, fontname="helv", color=(0.75, 0.75, 0.75))
    page.insert_text(fitz.Point(62, 68), "47%", fontsize=12, fontname="helv", color=(0, 0, 0))
    pdf_bytes = doc.tobytes()
    doc.close()

    result = native_text_replace(
        pdf_bytes,
        page_index=0,
        original_text="47%",
        replacement_text="48%",
        bounds=_normalized_bounds_for(pdf_bytes, "47%"),
        font_size=12,
        native_edit_mode="content_stream_first",
        preserve_nearby_lines=True,
    )

    assert result.success
    assert result.strategy == "content_stream"
    assert result.warnings == ()
    assert "WATERMARK" in _extract_text(result.edited_pdf_bytes or b"")


def test_forced_tight_cleanup_warns_and_keeps_table_line_drawings():
    pdf_bytes = _make_table_pdf()
    result = native_text_replace(
        pdf_bytes,
        page_index=0,
        original_text="93",
        replacement_text="94",
        bounds=_normalized_bounds_for(pdf_bytes, "93"),
        font_size=12,
        native_edit_mode="redact_insert_tight",
        preserve_nearby_lines=True,
    )

    assert result.success
    assert result.strategy == "redact_insert_tight"
    assert NATIVE_EDIT_USED_CLEANUP_FALLBACK in result.warnings
    assert BACKGROUND_CHANGED_AROUND_EDIT in result.warnings
    assert _drawing_count(result.edited_pdf_bytes or b"") >= _drawing_count(pdf_bytes)
    assert "94" in _extract_text(result.edited_pdf_bytes or b"")
