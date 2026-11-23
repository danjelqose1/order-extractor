from copy import deepcopy
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import fitz
import pytest
from fastapi import FastAPI, HTTPException, Response
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from manual_mother_sheet import build_manual_mother_sheet_pdf


def sample_order():
    order = json.loads((ROOT / "tests/fixtures/manual_eldi_grouped_sheet.json").read_text())
    order.update(client_name="REZARTI LUIS DURRES", order_number="R-26-1049", order_date="2026-11-20")
    order["rows"] = [
        dict(position="1", width_mm=310, height_mm=1448, quantity=2,
             glass_type="2 VETRI 33.1F + 22 + 4LOWE (32MM)", final_area_m2=2.520),
        dict(position="7", width_mm=310, height_mm=1488, quantity=2,
             glass_type="2 VETRI 33.1F + 22 + 4LOWE (32MM)", final_area_m2=3.000),
        dict(position="A-3", width_mm=310.125, height_mm=1405.5, quantity=3,
             glass_type="4F", final_area_m2=1.234),
    ]
    return order


def test_mother_sheet_matches_reference_without_mutating_or_rounding():
    order = sample_order()
    before = deepcopy(order)
    pdf = fitz.open(stream=build_manual_mother_sheet_pdf(order), filetype="pdf")
    text = " ".join(pdf[0].get_text().split())
    assert "Mother Sheet – Client: REZARTI LUIS DURRES" in text
    assert "Order: R-26-1049 | Date: 20/11/2026 | Dimensions: cm" in text
    assert "2 VETRI 33.1F + 22 + 4LOWE (32MM) — Area: 5,520 m²" in text
    assert "4F — Area: 1,234 m²" in text
    assert text.count("[Order R-26-1049 — REZARTI LUIS DURRES]") == 2
    lines = ["1 – 31,0 × 144,8 × 2", "7 – 31,0 × 148,8 × 2", "A-3 – 31,0125 × 140,55 × 3"]
    assert all(line in text for line in lines)
    assert [text.index(line) for line in lines] == sorted(text.index(line) for line in lines)
    spans = [span for block in pdf[0].get_text("dict")["blocks"]
             for line in block.get("lines", []) for span in line["spans"]]
    assert all(span["color"] == 0 for span in spans)
    assert all("Bold" in span["font"] for span in spans if "Area:" in span["text"] or "Mother Sheet" in span["text"])
    assert not pdf[0].get_drawings()  # No borders or highlight marks.
    assert order == before


def test_saved_multispec_order_paginates_without_missing_items():
    order = json.loads((ROOT / "tests/fixtures/manual_eldi_grouped_sheet.json").read_text())
    before = deepcopy(order)
    pdf = fitz.open(stream=build_manual_mother_sheet_pdf(order), filetype="pdf")
    text = " ".join(" ".join(page.get_text() for page in pdf).split())
    assert len(pdf) > 1
    for row in order["rows"]:
        assert f'{row["index_number"]} ({row["client_position"]}) – ' in text
    for glass in dict.fromkeys(row["glass_type"] for row in order["rows"]):
        assert glass in text
    for page in pdf:
        for x0, y0, x1, y1, *_ in page.get_text("blocks"):
            assert 0 <= x0 < x1 <= page.rect.width
            assert 0 <= y0 < y1 <= page.rect.height
    assert order == before


@pytest.mark.parametrize("client_position, expected", [
    ("P-12", "13 (P-12) – 31,0 × 144,8 × 2"),
    ("", "13 – 31,0 × 144,8 × 2"),
    (None, "13 – 31,0 × 144,8 × 2"),
])
def test_saved_client_position_is_beside_original_index(client_position, expected):
    order = sample_order()
    order["rows"][0].update(position="P-12 13", index_number=13, client_position=client_position)
    before = deepcopy(order)
    pdf = fitz.open(stream=build_manual_mother_sheet_pdf(order), filetype="pdf")
    text = " ".join(pdf[0].get_text().split())
    assert expected in text
    assert "P-12 13" not in text
    assert "7 – 31,0 × 148,8 × 2" in text  # Rows without indexes retain their saved position.
    assert order == before


def test_print_route_is_read_only_and_independent_of_processing_and_grouping():
    order = sample_order()
    before = deepcopy(order)
    app = FastAPI()
    env = dict(app=app, HTTPException=HTTPException, Response=Response,
               build_manual_mother_sheet_pdf=build_manual_mother_sheet_pdf,
               db_module=SimpleNamespace(get_manual_order=lambda _id: order))
    tree = ast.parse((ROOT / "backend/app.py").read_text())
    route = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "preview_manual_mother_sheet")
    exec(compile(ast.Module(body=[route], type_ignores=[]), "app.py", "exec"), env)
    client = TestClient(app)
    response = client.get("/manual-orders/1/mother-sheet.pdf?group_dimensions=true")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline;")
    assert "7 – 31,0" in fitz.open(stream=response.content, filetype="pdf")[0].get_text()
    for layout in ["a4_portrait_2up", "a4_landscape_2up"]:
        two_up = client.get(f"/manual-orders/1/mother-sheet.pdf?layout={layout}")
        assert two_up.status_code == 200
        assert two_up.headers["X-Manual-Mother-Sheet-Layout"] == layout
        pdf = fitz.open(stream=two_up.content, filetype="pdf")
        assert len(pdf) == 1 and pdf[0].get_text().count("Mother Sheet") == 2
    assert client.get("/manual-orders/1/mother-sheet.pdf?layout=invalid").status_code == 400
    assert order == before
    order["status"] = "draft"
    assert client.get("/manual-orders/1/mother-sheet.pdf").status_code == 400
    env["db_module"].get_manual_order = lambda _id: None
    assert client.get("/manual-orders/1/mother-sheet.pdf").status_code == 404


def test_empty_order_rejected():
    with pytest.raises(ValueError, match="no rows"):
        build_manual_mother_sheet_pdf({"rows": []})


@pytest.mark.parametrize("layout", ["a4_portrait_2up", "a4_landscape_2up"])
@pytest.mark.parametrize("row_count", [3, 12, 19])
def test_two_complete_identical_copies_fit_in_separate_halves(layout, row_count):
    order = sample_order()
    if row_count > 3:
        order["rows"] = [
            {**order["rows"][0 if row_count == 12 or i % 3 else 2],
             "position": str(i), "index_number": i, "client_position": "P-12" if i == 1 else ""}
            for i in range(1, row_count + 1)
        ]
    before = deepcopy(order)
    pdf = fitz.open(stream=build_manual_mother_sheet_pdf(order, layout), filetype="pdf")
    assert len(pdf) == 1
    page = pdf[0]
    w, h = page.rect.width, page.rect.height
    assert (w > h) == (layout == "a4_landscape_2up")
    slots = [fitz.Rect(0, 0, w/2, h), fitz.Rect(w/2, 0, w, h)]
    texts = [" ".join(page.get_text(clip=slot).split()) for slot in slots]
    assert texts[0] == texts[1]
    assert texts[0].count("Mother Sheet") == 1
    assert "Date: 20/11/2026" in texts[0] and "Dimensions: cm" in texts[0]
    if row_count == 12:
        assert "1 (P-12) – 31,0 × 144,8 × 2" in texts[0]
        assert "12 – 31,0 × 144,8 × 2" in texts[0]
        assert "Area: 30,240 m²" in texts[0]
    elif row_count == 19:
        for row in order["rows"]:
            width, height, quantity = (
                "31,0125", "140,55", 3
            ) if row["glass_type"] == "4F" else ("31,0", "144,8", 2)
            index = row["index_number"]
            position = "1 (P-12)" if index == 1 else str(index)
            assert f"{position} – {width} × {height} × {quantity}" in texts[0]
    else:
        assert "Area: 5,520 m²" in texts[0]
        assert "4F" in texts[0] and "Area: 1,234 m²" in texts[0]
        assert "A-3 – 31,0125 × 140,55 × 3" in texts[0]
    # Every text span is fully inside one half, with readable type.
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                assert any(slot.contains(fitz.Rect(span["bbox"])) for slot in slots)
                assert span["size"] >= 10
                assert span["color"] == 0
    if layout == "a4_portrait_2up" and row_count in (12, 19):
        assert max(block[3] for block in page.get_text("blocks")) > h * 0.85
    assert order == before


@pytest.mark.parametrize("layout", ["a4_portrait_2up", "a4_landscape_2up"])
def test_oversized_order_rejected_instead_of_clipped_or_split_copies(layout):
    order = sample_order()
    order["rows"] = [{**order["rows"][0], "position": str(i)} for i in range(100)]
    before = deepcopy(order)
    with pytest.raises(ValueError, match="too long for two complete copies"):
        build_manual_mother_sheet_pdf(order, layout)
    assert order == before
    assert len(fitz.open(stream=build_manual_mother_sheet_pdf(order), filetype="pdf")) > 1
