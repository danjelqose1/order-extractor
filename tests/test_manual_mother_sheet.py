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
    assert order == before
    order["status"] = "draft"
    assert client.get("/manual-orders/1/mother-sheet.pdf").status_code == 400
    env["db_module"].get_manual_order = lambda _id: None
    assert client.get("/manual-orders/1/mother-sheet.pdf").status_code == 404


def test_empty_order_rejected():
    with pytest.raises(ValueError, match="no rows"):
        build_manual_mother_sheet_pdf({"rows": []})
