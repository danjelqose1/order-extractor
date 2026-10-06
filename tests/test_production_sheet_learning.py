from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from production_sheets import (SheetFeedbackRequest, SheetRequest, SheetSettings, TYPOGRAPHY_FIELDS,
                               render_sheet, source_digest, suggest_sheet)
from production_sheet_memory import remember_sheet, similar_sheets
from test_production_sheets import ai_request, source, doc, rows_in


def agent_client(*typographies):
    calls = []
    client = SimpleNamespace(with_options=lambda **kwargs: client)
    def create(**kwargs):
        calls.append(kwargs)
        values = typographies[min(len(calls) - 1, len(typographies) - 1)]
        return SimpleNamespace(status="completed", id=f"response-{len(calls)}", output_text=json.dumps({
            "typography": values, "explanation": "Used the available space.", "warnings": []}))
    client.responses = SimpleNamespace(create=create)
    return client, calls


def typography(**updates):
    values = {field: getattr(SheetSettings(), field) for field in TYPOGRAPHY_FIELDS}
    return {**values, **updates}


@pytest.fixture
def memory_db(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_DIR", str(tmp_path))
    import db
    engine = create_engine(f"sqlite:///{tmp_path / 'memory.db'}")
    db.ProductionSheetExample.__table__.create(engine)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(engine, expire_on_commit=False))
    yield db
    engine.dispose()


def feedback(src=None, **settings):
    src = src or source(4)
    return SheetFeedbackRequest(source=src, source_digest=source_digest(src), action="save",
                                settings=SheetSettings(**settings), baseline_settings=SheetSettings())


def test_finished_corrections_persist_are_deduplicated_and_exclude_order_content(memory_db):
    request = feedback(font_size=18, line_spacing=1.4, section_gap_pt=12, glass_after_pt=8, note="private note")
    assert remember_sheet(request)["learned_from_corrections"]
    request.action = "print"
    request.baseline_settings = request.settings
    remember_sheet(request)
    with memory_db.read_session() as session:
        records = session.scalars(select(memory_db.ProductionSheetExample)).all()
        assert len(records) == 1
        saved = records[0].example_json
    assert all(value not in saved for value in ("private note", "R-26-0883", "Ëldi", "VETRI"))
    examples = similar_sheets(source(5), SheetSettings())
    assert len(examples) == 1
    assert examples[0]["typography"]["font_size"] == 18
    assert examples[0]["manual_corrections"] == {"font_size": 4, "line_spacing": .25, "section_gap_pt": 5, "glass_after_pt": 8}
    # Different layout and density do not inherit this small-order preference.
    assert similar_sheets(source(160), SheetSettings()) == []
    assert similar_sheets(source(4), SheetSettings(layout="full", columns="1")) == []


def test_invalid_or_stale_finalizations_do_not_teach(memory_db):
    request = feedback()
    request.source_digest = "0" * 64
    with pytest.raises(ValueError, match="different sheet"):
        remember_sheet(request)
    request = feedback(source(200), layout="cuttable", font_size=18)
    with pytest.raises(ValueError, match="does not fit"):
        remember_sheet(request)
    assert similar_sheets(source(4), SheetSettings()) == []


def test_history_is_bounded_and_manual_examples_outweigh_accepted_defaults(memory_db, monkeypatch):
    monkeypatch.setattr("production_sheet_memory.MAX_EXAMPLES", 3)
    for i in range(5):
        src = source(4).model_copy(update={"text": source(4).text.replace("Ëldi", f"Client {i}")})
        remember_sheet(feedback(src, font_size=18 if i == 2 else 14))
    examples = similar_sheets(source(4), SheetSettings())
    assert len(examples) == 3
    assert examples[0]["typography"]["font_size"] == 18


def test_agent_refines_spare_space_and_receives_real_measurements_and_memory(memory_db):
    remember_sheet(feedback(font_size=18, line_spacing=1.5, section_gap_pt=16, glass_after_pt=16))
    request = ai_request().model_copy(update={"mode": "automatic", "source": source(4)})
    memory = similar_sheets(request.source, request.settings)
    client, calls = agent_client(typography(), typography(font_size=18, line_spacing=1.5, section_gap_pt=16, glass_after_pt=16))
    result = suggest_sheet(client, request, memory)
    assert len(calls) == 2 and result["memory_examples"] == 1
    assert result["preview"]["layout"] == "cuttable" and result["preview"]["sheet_count"] == 1
    first = json.loads(calls[0]["input"][0]["content"][0]["text"])
    second = json.loads(calls[1]["input"][0]["content"][0]["text"])
    assert first["similar_finished_sheets"][0]["typography"]["font_size"] == 18
    assert first["measurements"]["min_free_height_pt"] > 100
    assert second["attempts"][0]["individually_fitting_increases"]["font_size"] == 15
    assert calls[1]["input"][0]["content"][1]["image_url"] != request.images[0]
    assert result["proposal"]["settings"]["font_size"] == 18
    assert rows_in(doc(result["preview"])[0].get_text()) == rows_in(request.source.text) * 2
    assert set(calls[0]["text"]["format"]["schema"]["properties"]) == {"typography", "explanation", "warnings"}


def test_agent_recovers_from_overflow_and_cannot_change_layout_or_notes():
    request = ai_request().model_copy(update={"mode": "automatic", "source": source(20),
                                              "settings": SheetSettings(layout="cuttable", note="Keep together")})
    client, calls = agent_client(typography(font_size=18, line_spacing=1.5), typography())
    result = suggest_sheet(client, request)
    assert len(calls) >= 2
    context = json.loads(calls[1]["input"][0]["content"][0]["text"])
    assert "fit_error" in context["attempts"][0]
    settings = result["proposal"]["settings"]
    assert settings["layout"] == "cuttable" and settings["columns"] == "1"
    assert settings["note"] == "Keep together"
    assert result["preview"]["sheet_count"] == 1


def test_agent_rejects_added_pages_and_has_bounded_failure():
    request = ai_request().model_copy(update={"mode": "automatic", "source": source(36),
                                              "settings": SheetSettings(layout="full", columns="1")})
    client, calls = agent_client(typography(font_size=18, line_spacing=1.5, section_gap_pt=16))
    with pytest.raises(ValueError, match="could not find"):
        suggest_sheet(client, request)
    assert len(calls) == 3
    second = json.loads(calls[1]["input"][0]["content"][0]["text"])
    assert "pages per copy" in second["attempts"][0]["fit_error"]


def test_last_fitting_ai_result_survives_a_later_bad_proposal():
    request = ai_request().model_copy(update={"mode": "automatic", "source": source(4)})
    client, calls = agent_client(typography(font_size=16), typography(font_size=999))
    result = suggest_sheet(client, request)
    assert len(calls) == 3
    assert result["proposal"]["settings"]["font_size"] == 16
    assert result["response_id"] == "response-1"


def test_feedback_and_agent_routes_use_existing_auth_and_hide_internal_errors(monkeypatch):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_smoke import _load_app
    from fastapi.testclient import TestClient
    app, writes = _load_app(monkeypatch)
    app.APP_KEY = "test-app-key"
    client = TestClient(app.app)
    headers = {"X-App-Key": "test-app-key"}
    assert client.post("/api/production-sheets/feedback", json=feedback().model_dump()).status_code == 401
    assert client.post("/api/production-sheets/ai", json=ai_request().model_dump()).status_code == 401
    app.remember_sheet = lambda _: (_ for _ in ()).throw(RuntimeError("private database error"))
    result = client.post("/api/production-sheets/feedback", json=feedback().model_dump(), headers=headers)
    assert result.status_code == 503 and "private" not in result.text
    seen = []
    app.similar_sheets = lambda *_: [{"typography": {"font_size": 18}}]
    app.get_client = lambda: object()
    app.suggest_sheet = lambda client, request, memory: seen.append(memory) or {"ok": True}
    assert client.post("/api/production-sheets/ai", json=ai_request().model_dump(), headers=headers).status_code == 200
    assert seen[0][0]["typography"]["font_size"] == 18
    assert not writes["update_order_rows"] and not writes["update_order_status"]
