from __future__ import annotations

from io import BytesIO
import importlib
import json
import sys
import httpx
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import fitz
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
INDEX_HTML = ROOT_DIR / "docs" / "index.html"
APP_JS = ROOT_DIR / "docs" / "js" / "app.js"
APP_PY = ROOT_DIR / "backend" / "app.py"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

photo = importlib.import_module("manual_photo_assist")


def _text(raw=None, value=None, warning=None):
    return {"raw": raw, "value": value, "warning": warning}


def _dimension(raw=None, value=None, unit="unknown", warning=None):
    return {
        "raw": raw,
        "value": value,
        "unit": unit,
        "normalized_mm": None,
        "warning": warning,
    }


def _quantity(raw=None, value=None, warning=None):
    return {"raw": raw, "value": value, "warning": warning}


def _index(raw=None, value=None, warning=None):
    return {"raw": raw, "value": value, "warning": warning}


def _row(**overrides):
    row = {
        "source_line": "K1 1 80 x 231.5 = 2 WC",
        "section": "Vila 1",
        "client_reference": _text("K1", "K1"),
        "index_number": _index("1", 1),
        "width": _dimension("80", 80, "cm"),
        "height": _dimension("231.5", 231.5, "cm"),
        "quantity": _quantity("2", 2),
        "glass_type": _text(None, None),
        "notes": {"raw": "WC", "value": "WC"},
        "warnings": [],
    }
    row.update(overrides)
    return row


def _ai_result(rows=None, **overrides):
    payload = {
        "document": {
            "client_name": _text("Klienti", "Klienti"),
            "order_number": _text(None, None),
            "glass_type": _text("4F + 16 + LowE", "4F + 16 + LowE"),
        },
        "rows": rows if rows is not None else [_row()],
        "global_warnings": [],
        "raw_detected_text": "4F + 16 + LowE\nVila 1\nK1 1 80 x 231.5 = 2 WC",
    }
    payload.update(overrides)
    return photo.PhotoAssistAIResult.model_validate(payload)


def _normalized_image():
    return photo.NormalizedImage(
        content=b"image",
        mime_type="image/png",
        width=100,
        height=100,
        original_format="PNG",
    )


def test_photo_assist_normalizes_exif_orientation_without_storing_a_file():
    source = Image.new("RGB", (40, 20), "white")
    exif = Image.Exif()
    exif[274] = 6
    stream = BytesIO()
    source.save(stream, format="JPEG", quality=95, exif=exif)

    normalized = photo.normalize_uploaded_image(stream.getvalue())

    assert normalized.mime_type == "image/jpeg"
    assert (normalized.width, normalized.height) == (20, 40)
    assert normalized.content.startswith(b"\xff\xd8")


def test_photo_assist_flattens_mpo_detected_iphone_photo_to_jpeg(monkeypatch):
    source = Image.new("RGB", (40, 20), "white")
    stream = BytesIO()
    source.save(stream, format="JPEG", quality=95)
    original_open = photo.Image.open

    def open_as_mpo(image_stream):
        opened = original_open(image_stream)
        opened.format = "MPO"
        return opened

    monkeypatch.setattr(photo.Image, "open", open_as_mpo)

    normalized = photo.normalize_uploaded_image(stream.getvalue())

    assert normalized.original_format == "MPO"
    assert normalized.mime_type == "image/jpeg"
    assert normalized.content.startswith(b"\xff\xd8")


def test_photo_assist_downsamples_large_jpeg_before_model_upload():
    source = Image.new("RGB", (3600, 900), "white")
    stream = BytesIO()
    source.save(stream, format="JPEG", quality=90)

    normalized = photo.normalize_uploaded_image(stream.getvalue())

    assert max(normalized.width, normalized.height) <= photo.MAX_IMAGE_LONG_EDGE
    assert normalized.mime_type == "image/jpeg"
    assert len(normalized.content) < len(stream.getvalue())


def test_photo_assist_rejects_extreme_pixel_dimensions_before_decode(monkeypatch):
    class OversizedImage:
        format = "JPEG"
        width = 10000
        height = 10000

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(photo.Image, "open", lambda _stream: OversizedImage())

    with pytest.raises(photo.InvalidImageError, match="dimensions are too large"):
        photo.normalize_uploaded_image(b"not-empty")


def test_photo_assist_rejects_heic_with_a_safe_conversion_message():
    fake_heic = b"\x00\x00\x00\x18ftypheic" + (b"\x00" * 32)

    with pytest.raises(photo.UnsupportedImageError, match="Upload JPG or PNG"):
        photo.normalize_uploaded_image(fake_heic)


def test_photo_assist_mixed_order_preserves_sections_references_indexes_raw_values_and_notes():
    result = photo.normalize_extraction(
        _ai_result(
            rows=[
                _row(),
                _row(
                    source_line="K2 1 137.5 x 90 + 1",
                    section="Vila 2",
                    client_reference=_text("K2", "K2"),
                    index_number=_index("1", 1),
                    width=_dimension("137.5", 137.5, "cm", "Unclear handwritten digit"),
                    height=_dimension("90", 90, "cm"),
                    quantity=_quantity("1", 1),
                    notes={"raw": None, "value": None},
                ),
            ]
        )
    )

    assert result.rows[0].section == "Vila 1"
    assert result.rows[0].client_reference.value == "K1"
    assert result.rows[0].index_number.value == 1
    assert result.rows[0].width.raw == "80"
    assert result.rows[0].width.normalized_mm == 800
    assert result.rows[0].height.normalized_mm == 2315
    assert result.rows[0].notes.value == "WC"
    assert result.rows[0].glass_type.value == "4F + 16 + LowE"
    assert "Duplicate index number 1." in result.rows[0].warnings
    assert "Duplicate index number 1." in result.rows[1].warnings


def test_photo_assist_simple_list_does_not_invent_references_or_indexes():
    rows = []
    for number in range(6):
        rows.append(
            _row(
                source_line=f"{80 + number} x 120 = 2",
                section=None,
                client_reference=_text(None, None),
                index_number=_index(None, None),
                width=_dimension(str(80 + number), 80 + number, "unknown"),
                height=_dimension("120", 120, "unknown"),
                quantity=_quantity("2", 2),
                notes={"raw": None, "value": None},
            )
        )

    result = photo.normalize_extraction(_ai_result(rows=rows))

    assert len(result.rows) == 6
    assert all(row.index_number.value is None for row in result.rows)
    assert all(row.client_reference.value is None for row in result.rows)
    assert all(row.width.normalized_mm is None for row in result.rows)
    assert all("Dimensions may be written in centimetres" in (row.width.warning or "") for row in result.rows)


def test_photo_assist_reasons_over_simple_glass_list_with_selected_cm_unit():
    heading = _row(
        source_line="3/3 transparent tek",
        section=None,
        client_reference=_text(None, None),
        index_number=_index(None, None),
        width=_dimension(None, None),
        height=_dimension(None, None),
        quantity=_quantity(None, None),
        glass_type=_text("transparent tek", "transparent tek"),
        notes={"raw": "3/3", "value": "3/3"},
    )
    dimensions = [
        ("108 x 137.5 = 3", 108, 137.5, 3, "="),
        ("108 x 78.5 = 3", 108, 78.5, 3, "="),
        ("110 x 137.5 = 3", 110, 137.5, 3, "="),
        ("110 x 78.5 = 3", 110, 78.5, 3, "="),
        ("70 x 132 2 2", 70, 132, 2, "2"),
        ("70 x 78.5 = 2", 70, 78.5, 2, "="),
    ]
    rows = [heading]
    for source, width, height, quantity, note in dimensions:
        rows.append(
            _row(
                source_line=source,
                section=None,
                client_reference=_text(None, None),
                index_number=_index(None, None),
                width=_dimension(str(width), width, "unknown", "Unit unclear"),
                height=_dimension(str(height), height, "unknown", "Unit unclear"),
                quantity=_quantity(str(quantity), quantity),
                glass_type=_text(None, None),
                notes={"raw": note, "value": note},
                warnings=["Unit not clear from image"],
            )
        )

    result = photo.normalize_extraction(
        _ai_result(
            rows=rows,
            document={
                "client_name": _text(None, None),
                "order_number": _text(None, None),
                "glass_type": _text("transparent tek", "transparent tek"),
            },
            global_warnings=["Dimension unit is unclear."],
        ),
        preferred_dimension_unit="cm",
    )

    assert len(result.rows) == 6
    assert result.document.glass_type.value == "3/3 transparent tek"
    assert all(row.glass_type.value == "3/3 transparent tek" for row in result.rows)
    assert result.rows[0].width.normalized_mm == 1080
    assert result.rows[0].height.normalized_mm == 1375
    assert all(row.notes.value is None for row in result.rows)
    assert result.global_warnings == []


def test_photo_assist_unreadable_row_is_preserved_with_nulls_and_warnings():
    unreadable = _row(
        source_line="? x 120 = ?",
        client_reference=_text(None, None),
        index_number=_index(None, None),
        width=_dimension("?", None, "unknown", "Unclear handwritten digit"),
        height=_dimension("120", 120, "cm"),
        quantity=_quantity("?", None),
        glass_type=_text(None, None),
        notes={"raw": None, "value": None},
    )

    result = photo.normalize_extraction(_ai_result(rows=[unreadable]))

    assert len(result.rows) == 1
    assert result.rows[0].source_line == "? x 120 = ?"
    assert result.rows[0].width.value is None
    assert result.rows[0].quantity.value is None
    assert "Width is missing or unclear." in result.rows[0].warnings
    assert "Quantity not detected or invalid." in result.rows[0].warnings


def test_photo_assist_retries_one_malformed_response_then_returns_review_data():
    calls = []

    def fake_call(image, *, model, client):
        calls.append((image, model, client))
        if len(calls) == 1:
            raise photo.InvalidExtractionError("The AI response could not be validated.")
        return _ai_result()

    response = photo.extract_photo_assist(
        _normalized_image(),
        request_id="request-test",
        client=object(),
        model="vision-test",
        call_once=fake_call,
    )

    assert len(calls) == 2
    assert response.request_id == "request-test"
    assert response.model == "vision-test"
    assert response.status == "ready"


def test_photo_assist_surfaces_model_timeout_as_timeout_error():
    def timeout_call(image, *, model, client):
        raise photo.APITimeoutError(httpx.Request("POST", "https://example.invalid"))

    with pytest.raises(photo.PhotoAssistTimeoutError, match="Extraction timed out"):
        photo.extract_photo_assist(
            _normalized_image(),
            request_id="request-timeout",
            client=object(),
            model="vision-test",
            call_once=timeout_call,
        )


def test_photo_assist_sends_direct_image_with_high_detail_and_strict_schema():
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text=json.dumps(_ai_result().model_dump()))

    result = photo._call_model_once(
        _normalized_image(),
        model="vision-test",
        client=SimpleNamespace(responses=FakeResponses()),
        preferred_dimension_unit="cm",
    )

    assert result.rows[0].source_line
    assert calls[0]["model"] == "vision-test"
    assert calls[0]["store"] is False
    assert calls[0]["text"]["format"]["type"] == "json_schema"
    assert calls[0]["text"]["format"]["strict"] is True
    assert "currently set to centimetres (cm)" in calls[0]["instructions"]
    image_input = calls[0]["input"][0]["content"][1]
    assert image_input["type"] == "input_image"
    assert image_input["detail"] == "high"
    assert image_input["image_url"].startswith("data:image/png;base64,")


def test_photo_assist_never_creates_a_manual_order(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_DIR", str(tmp_path))
    previous_db = sys.modules.get("db")
    sys.modules.pop("db", None)
    try:
        db = importlib.import_module("db")
        db.init_db()

        response = photo.extract_photo_assist(
            _normalized_image(),
            request_id="request-no-save",
            client=object(),
            model="vision-test",
            call_once=lambda image, *, model, client: _ai_result(),
        )

        assert response.rows
        assert db.list_manual_orders() == []
    finally:
        if previous_db is None:
            sys.modules.pop("db", None)
        else:
            sys.modules["db"] = previous_db


def test_photo_assist_values_continue_through_existing_manual_label_logic():
    response = photo.extract_photo_assist(
        _normalized_image(),
        request_id="request-labels",
        client=object(),
        model="vision-test",
        call_once=lambda image, *, model, client: _ai_result(),
    )
    detected = response.rows[0]
    order = {
        "order_number": "L-26-0001",
        "client_name": response.document.client_name.value,
        "order_date": "2026-07-10",
        "status": "approved",
        "manual_format": "client_positions_red_index",
        "rows": [
            {
                "section": detected.section,
                "client_position": detected.client_reference.value,
                "index_number": detected.index_number.value,
                "glass_type": detected.glass_type.value,
                "width_mm": detected.width.normalized_mm,
                "height_mm": detected.height.normalized_mm,
                "quantity": detected.quantity.value,
                "notes": detected.notes.value or "",
            }
        ],
    }
    documents = importlib.import_module("manual_documents")
    labels = fitz.open(stream=documents.build_manual_labels_pdf(order), filetype="pdf")

    assert len(labels) == 2
    assert all("POS K1" in page.get_text() for page in labels)
    assert all("#1" in page.get_text() for page in labels)


def test_photo_assist_feature_flag_and_model_are_centralized(monkeypatch):
    monkeypatch.setenv("MANUAL_PHOTO_ASSIST_ENABLED", "true")
    monkeypatch.setenv("MANUAL_PHOTO_ASSIST_MODEL", "configured-vision-model")

    assert photo.photo_assist_enabled() is True
    assert photo.photo_assist_model() == "configured-vision-model"


def test_photo_assist_frontend_requires_review_and_explicit_apply():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = APP_JS.read_text(encoding="utf-8")
    app_py = APP_PY.read_text(encoding="utf-8")

    assert 'id="manualPhotoAssist"' in html
    assert "Photo Assist" in html
    assert "BETA" in html
    assert 'id="manualPhotoReview"' in html
    assert 'id="manualPhotoApply"' in html
    assert 'id="manualPhotoApplyReplace"' in html
    assert 'id="manualPhotoApplyAppend"' in html
    assert 'id="manualPhotoApplyCancel"' in html
    assert "AI may misread handwriting. Review every value before applying." in html
    assert "function extractManualPhoto" in js
    assert "function renderManualPhotoReview" in js
    assert "function applyManualPhotoResult" in js
    assert "manualPhotoFormHasUnsavedData()" in js
    assert "async function prepareManualPhotoUpload" in js
    assert "maxEdge = 1800" in js
    assert 'canvas.toBlob(' in js
    assert '"image/jpeg"' in js
    assert "const preparedFile = await prepareManualPhotoUpload(file)" in js
    assert 'applyManualPhotoResult("replace")' in js
    assert 'applyManualPhotoResult("append")' in js
    assert 'formData.append("image"' in js
    assert 'formData.append("dimension_unit"' in js
    assert 'manualApi("/api/manual-orders/photo-assist/extract"' in js
    assert '@app.post(\n    "/api/manual-orders/photo-assist/extract"' in app_py
    apply_block = js[js.index("function applyManualPhotoResult"):js.index("function manualToday")]
    assert 'manualApi("/manual-orders"' not in apply_block
    assert "saveManualOrder(" not in apply_block


def test_manual_orders_disables_global_scan_studio_drop_routing():
    js = APP_JS.read_text(encoding="utf-8")
    global_drop_block = js[
        js.index("function isScanStudioActive"):
        js.index("function cleanupGlobalScanDropRouting")
    ]

    assert "function isManualOrdersActive" in global_drop_block
    assert "panels.manual.classList.contains(\"active\")" in global_drop_block
    assert "if (isManualOrdersActive() && isFileDragEvent(event))" in global_drop_block
    assert 'target?.closest?.("#manualPhotoDropzone") ? "copy" : "none"' in global_drop_block
