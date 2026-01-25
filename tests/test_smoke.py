from __future__ import annotations

import os

import pytest

from backend.app import extract_pages_text
from backend.llm import call_llm_for_extraction_multi


PDF_PATH_CANDIDATES = [
    os.path.join("/mnt", "data", "xhama-ORGITO 16 HYRJE.pdf"),
    os.path.join(os.path.dirname(__file__), "xhama-ORGITO 16 HYRJE.pdf"),
]


def _locate_pdf() -> str | None:
    for candidate in PDF_PATH_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


@pytest.mark.skipif(_locate_pdf() is None, reason="Sample PDF not available in current environment")
def test_multi_page_extraction_smoke():
    pdf_path = _locate_pdf()
    assert pdf_path is not None

    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()

    pages_text = extract_pages_text(pdf_bytes)
    bundle = call_llm_for_extraction_multi(pages_text)
    data = bundle.get("data") or {}
    rows = data.get("rows") or []

    assert len(rows) == 120, f"expected 120 rows, got {len(rows)}"

    declared_units = bundle.get("declared_units")
    if declared_units is not None:
        assert declared_units == 120

    warnings = (data.get("warnings") or []) + (bundle.get("warnings") or [])
    split_warnings = [w for w in warnings if "split" in w.lower()]
    assert len(split_warnings) <= 2
