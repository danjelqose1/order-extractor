from __future__ import annotations

import base64
import json
import re
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import fitz
import httpx
import pytest
from openai import OpenAI
from PIL import Image
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from production_sheets import (SheetSource, SheetSettings, SheetRequest, SheetAIRequest,
                               render_sheet, plan_sheet, suggest_sheet)

TITLE = "Mother Sheet – Client: Ëldi | Orders: R-26-0883 | Date: 9/30/2026"
GLASS = "3 VETRI 33.1F +14CALDO+ 4F +16CALDO+ 33.1LOWE (44MM) — Area: 9,890 m²"
ORDER = "[Order R-26-0883 — Ëldi]"


def source(count=10, detail=False):
    rows = [f"{i} – {380+i} × 2250 × 2" for i in range(1, count + 1)]
    text = "\n".join([TITLE, "", GLASS, ORDER, *rows])
    if detail:
        text = text.replace(rows[0], rows[0] + "\n   (Rounded from 388×2249)")
    return SheetSource(text=text, glass_headers=[GLASS], order_headers=[ORDER], row_count=count, piece_count=2 * count)


def doc(result):
    return fitz.open(stream=base64.b64decode(result["pdf_base64"]), filetype="pdf")


def rows_in(text):
    return re.findall(r"^\d+ – [^\n]+", text, re.MULTILINE)


def normalized(text):
    return " ".join(text.split())


def image():
    output = BytesIO()
    Image.new("RGB", (100, 100), "white").save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def test_small_sheet_is_two_independent_copies_on_one_cuttable_a4():
    src = source(detail=True)
    before = src.model_dump_json()
    result = render_sheet(SheetRequest(source=src))
    pdf = doc(result)
    assert result["layout"] == "cuttable" and result["sheet_count"] == 1
    assert len(pdf) == 1 and tuple(round(v) for v in pdf[0].rect[2:]) == (842, 595)
    text = pdf[0].get_text()
    assert rows_in(text) == rows_in(src.text) * 2
    assert text.count("Ëldi") == 4 and text.count("9,890 m²") == 2
    assert text.count("Rounded from 388×2249") == 2
    assert result["row_count"] == 10 and result["piece_count"] == 20
    assert src.model_dump_json() == before
    center = pdf[0].rect.width / 2
    for block in pdf[0].get_text("blocks"):
        assert block[2] < center or block[0] > center


@pytest.mark.parametrize("columns", ["1", "2", "3"])
def test_full_page_copies_are_collated_and_keep_every_row(columns):
    src = source(160)
    request = SheetRequest(source=src, settings=SheetSettings(layout="full", columns=columns, orientation="landscape"))
    result = render_sheet(request)
    pdf = doc(result)
    n = result["pages_per_copy"]
    assert len(pdf) == 2 * n and result["columns"] == int(columns)
    assert [page.get_text() for page in pdf[:n]] == [page.get_text() for page in pdf[n:]]
    copy_text = "\n".join(page.get_text() for page in pdf[:n])
    assert rows_in(copy_text) == rows_in(src.text)
    assert normalized(copy_text).count(normalized(GLASS)) == 1
    assert normalized(copy_text).count(normalized(ORDER)) == 1
    for page in pdf:
        for block in page.get_text("blocks"):
            assert block[0] >= 20 and block[1] >= 20
            assert block[2] <= page.rect.width - 20 and block[3] <= page.rect.height - 20


@pytest.mark.parametrize("columns", ["2", "3"])
@pytest.mark.parametrize("section_start", [4, 100])
def test_glass_sections_flow_across_columns_and_pages_until_the_next_type(columns, section_start):
    src = source(160)
    second_glass = "2 VETRI 4F +20+ 4LOWE (28MM) — Area: 390,580 m²"
    second_order = "[Order R-26-0884 — BONITA]"
    source_rows = rows_in(src.text)
    src.text = "\n".join([TITLE, GLASS, ORDER, *source_rows[:section_start],
                          second_glass, second_order, *source_rows[section_start:]])
    src.glass_headers.append(second_glass)
    src.order_headers.append(second_order)
    before = src.model_dump_json()
    result = render_sheet(SheetRequest(source=src, settings=SheetSettings(
        layout="full", columns=columns, orientation="landscape")))
    pdf = doc(result)
    n = result["pages_per_copy"]
    assert n > 1
    for copy_start in (0, n):
        text = "\n".join(page.get_text() for page in pdf[copy_start:copy_start + n])
        assert rows_in(text) == source_rows
        for heading in (GLASS, ORDER, second_glass, second_order):
            assert normalized(text).count(normalized(heading)) == 1
        assert text.index(source_rows[section_start - 1]) < text.index("2 VETRI 4F") < text.index(source_rows[section_start])
    assert src.model_dump_json() == before


def test_auto_uses_columns_for_long_jobs_without_shrinking_text():
    result = render_sheet(SheetRequest(source=source(300)))
    assert result["layout"] == "full" and result["columns"] == 3
    assert result["copies"] == 2
    plan = plan_sheet(source(300), SheetSettings())
    assert all(item["size"] == 14 for page in plan["pages"] for item in page)


def test_explicit_columns_override_the_small_job_halves():
    result = render_sheet(SheetRequest(source=source(), settings=SheetSettings(columns="3", orientation="landscape")))
    assert result["layout"] == "full" and result["columns"] == 3 and result["sheet_count"] == 2
    plan = plan_sheet(source(), SheetSettings(columns="3", orientation="landscape"))
    assert len({item["x"] for item in plan["pages"][0] if item["kind"] == "row"}) == 3


def test_long_glass_headings_wrap_without_stranding_the_first_row():
    src = source(120)
    plan = plan_sheet(src, SheetSettings(layout="full", columns="2", orientation="portrait", font_size=18))
    for page in plan["pages"]:
        for index, item in enumerate(page):
            if item["kind"] == "glass" and (index + 1 == len(page) or page[index + 1]["kind"] != "glass"):
                assert any(next_item["kind"] == "row" and next_item["x"] == item["x"] for next_item in page[index + 1:])


def test_large_job_cannot_be_forced_into_halves_or_tiny_text():
    with pytest.raises(ValueError, match="does not fit"):
        render_sheet(SheetRequest(source=source(200), settings=SheetSettings(layout="cuttable")))
    with pytest.raises(ValidationError):
        SheetSettings(font_size=9)
    with pytest.raises(ValidationError):
        SheetSettings(copies=4)


def test_space_after_wrapped_glass_heading_moves_only_following_content():
    src = source()
    before = plan_sheet(src, SheetSettings(layout="cuttable"))["pages"][0]
    after = plan_sheet(src, SheetSettings(layout="cuttable", glass_after_pt=9))["pages"][0]
    assert sum(item["kind"] == "glass" for item in before) > 1
    for original, changed in zip(before, after):
        assert original["text"] == changed["text"]
        expected = 9 if original["kind"] in ("order", "row") else 0
        assert original["y"] - changed["y"] == pytest.approx(expected)
    pdf = doc(render_sheet(SheetRequest(source=src, settings=SheetSettings(glass_after_pt=9))))
    assert rows_in(pdf[0].get_text()) == rows_in(src.text) * 2


def test_invalid_or_missing_rows_and_unknown_fields_are_rejected():
    src = source()
    with pytest.raises(ValueError, match="row count"):
        render_sheet(SheetRequest(source=src.model_copy(update={"row_count":11})))
    with pytest.raises(ValueError, match="invalid dimensions"):
        render_sheet(SheetRequest(source=src.model_copy(update={"text":src.text + "  ⚠"})))
    with pytest.raises(ValidationError):
        SheetRequest(source=src, settings={"font_size":14,"quantity":99})


def test_repeated_source_rows_and_reset_numbering_are_not_deduplicated():
    src = source(1)
    src.text += "\n" + GLASS + "\n" + ORDER + "\n1 – 381 × 2250 × 2"
    src.row_count = 2
    src.piece_count = 4
    text = doc(render_sheet(SheetRequest(source=src)))[0].get_text()
    assert rows_in(text) == ["1 – 381 × 2250 × 2"] * 4
    assert normalized(text).count(normalized(GLASS)) == 4
    assert normalized(text).count(normalized(ORDER)) == 4


def test_notes_are_preserved_in_both_copies_and_reset_is_generated_content():
    src = source()
    result = render_sheet(SheetRequest(source=src, settings=SheetSettings(note="Keep this order together.")))
    assert doc(result)[0].get_text().count("Keep this order together.") == 2
    reset = render_sheet(SheetRequest(source=src))
    assert "Production note" not in doc(reset)[0].get_text()
    assert result["source_digest"] == reset["source_digest"]


def fake_client(payload, status="completed"):
    calls = []
    options = []
    client = SimpleNamespace()
    client.with_options = lambda **kwargs: (options.append(kwargs) or client)
    client.responses = SimpleNamespace(create=lambda **kwargs: (calls.append(kwargs) or SimpleNamespace(status=status, output_text=json.dumps(payload))))
    return client, calls, options


def ai_request():
    src = source()
    return SheetAIRequest(source=src, settings=SheetSettings(), instruction="Use three columns.", images=[image()],
                          rendered={"layout":"cuttable","columns":1,"orientation":"landscape", "pages_per_copy":1,"sheet_count":1,"sampled_pages":[1]})


def test_ai_uses_sol_61_medium_vision_and_returns_only_a_validated_proposal(monkeypatch):
    monkeypatch.delenv("PRODUCTION_SHEET_MODEL", raising=False)
    request = ai_request()
    payload = {"settings":SheetSettings(layout="full",columns="3",orientation="landscape").model_dump(),
               "explanation":"Three readable columns inside each copy.","warnings":[]}
    client, calls, options = fake_client(payload)
    result = suggest_sheet(client,request)
    assert result["proposal"] == payload and result["preview"]["columns"] == 3
    assert result["preview"]["source_digest"] == render_sheet(SheetRequest(source=request.source))["source_digest"]
    call = calls[0]
    assert call["model"] == "gpt-6.1-sol" and call["reasoning"] == {"effort":"medium"}
    assert "temperature" not in call and call["store"] is True
    assert call["input"][0]["content"][1]["type"] == "input_image"
    assert options[0]["max_retries"] == 0
    timeout = options[0]["timeout"]
    assert (timeout.read,timeout.connect,timeout.write,timeout.pool) == (900,15,30,15)
    schema = call["text"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["$defs"]["SheetSettings"]["required"]) == set(SheetSettings.model_fields)
    assert "Use glass_after_pt" in call["instructions"]


def test_ai_cannot_rewrite_dimensions_or_return_incomplete_or_unreadable_changes():
    request = ai_request()
    payload = {"settings":SheetSettings().model_dump(),"explanation":"Proposed layout.","warnings":[], "rows":[{"width":999}]}
    client,_,_ = fake_client(payload)
    with pytest.raises(ValidationError):
        suggest_sheet(client,request)
    payload.pop("rows")
    client,_,_ = fake_client(payload,status="incomplete")
    with pytest.raises(RuntimeError,match="complete proposal"):
        suggest_sheet(client,request)
    payload["settings"]["layout"] = "cuttable"
    request.source = source(200)
    client,_,_ = fake_client(payload)
    with pytest.raises(ValueError,match="does not fit"):
        suggest_sheet(client,request)


def test_sol_dashboard_storage_and_medium_reasoning_are_sent_by_the_installed_sdk(monkeypatch):
    monkeypatch.delenv("PRODUCTION_SHEET_MODEL", raising=False)
    requests = []
    payload = {"settings": SheetSettings().model_dump(), "explanation": "Ready for review.", "warnings": []}
    def transport(request):
        requests.append(request)
        return httpx.Response(200, json={"id": "resp_logged_test", "object": "response", "created_at": 0,
            "status": "completed", "model": "gpt-6.1-sol", "output": [{"id": "msg_test", "type": "message",
            "status": "completed", "role": "assistant", "content": [{"type": "output_text",
            "text": json.dumps(payload), "annotations": []}]}]})
    client = OpenAI(api_key="test-only", http_client=httpx.Client(transport=httpx.MockTransport(transport)))
    result = suggest_sheet(client, ai_request())
    sent = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/responses"
    assert sent["store"] is True and sent["model"] == "gpt-6.1-sol"
    assert sent["reasoning"] == {"effort": "medium"} and "temperature" not in sent
    assert result["response_id"] == "resp_logged_test"


def test_invalid_images_are_rejected_before_an_api_call():
    request = ai_request()
    request.images = ["https://example.com/image.jpg"]
    client,calls,_ = fake_client({})
    with pytest.raises(ValueError,match="PNG or JPEG"):
        suggest_sheet(client,request)
    assert calls == []
