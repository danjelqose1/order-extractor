from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient

import backend.app as app_module


def test_invoices_year_filter_current(monkeypatch):
    current_year = datetime.now().year
    payload = {
        "jobs": [
            {"id": "issue-current", "issueDate": f"{current_year}-06-01T00:00:00Z"},
            {"id": "created-current", "issueDate": None, "createdAt": f"{current_year}-03-05T12:00:00Z"},
            {"id": "past", "createdAt": f"{current_year - 1}-11-10T00:00:00Z"},
        ]
    }
    monkeypatch.setattr(app_module, "_load_invoices", lambda: payload)
    client = TestClient(app_module.app)

    response = client.get("/api/invoices?year=current")
    assert response.status_code == 200
    data = response.json()

    ids = {job["id"] for job in data.get("jobs", [])}
    assert ids == {"issue-current", "created-current"}
    assert data.get("currentYear") == current_year


def test_invoices_year_filter_specific_year(monkeypatch):
    payload = {
        "jobs": [
            {"id": "issue-2024", "issueDate": "2024-06-01T00:00:00Z", "createdAt": "2025-01-02T00:00:00Z"},
            {"id": "created-2024", "issueDate": None, "createdAt": "2024-03-05T12:00:00Z"},
            {"id": "created-2023", "createdAt": "2023-11-10T00:00:00Z"},
            {"id": "issue-2025", "issueDate": "2025-02-01"},
        ]
    }
    monkeypatch.setattr(app_module, "_load_invoices", lambda: payload)
    client = TestClient(app_module.app)

    response = client.get("/api/invoices?year=2024")
    assert response.status_code == 200
    data = response.json()

    ids = {job["id"] for job in data.get("jobs", [])}
    assert ids == {"issue-2024", "created-2024"}


def test_invoices_year_filter_all(monkeypatch):
    payload = {
        "jobs": [
            {"id": "issue-2024", "issueDate": "2024-06-01T00:00:00Z"},
            {"id": "created-2025", "createdAt": "2025-03-05T12:00:00Z"},
        ]
    }
    monkeypatch.setattr(app_module, "_load_invoices", lambda: payload)
    client = TestClient(app_module.app)

    response = client.get("/api/invoices?year=all")
    assert response.status_code == 200
    data = response.json()

    ids = {job["id"] for job in data.get("jobs", [])}
    assert ids == {"issue-2024", "created-2025"}


def test_invoices_year_filter_defaults_to_current(monkeypatch):
    current_year = datetime.now().year
    payload = {
        "jobs": [
            {"id": "current", "createdAt": f"{current_year}-01-02T00:00:00Z"},
            {"id": "past", "createdAt": f"{current_year - 1}-01-02T00:00:00Z"},
        ]
    }
    monkeypatch.setattr(app_module, "_load_invoices", lambda: payload)
    client = TestClient(app_module.app)

    response = client.get("/api/invoices")
    assert response.status_code == 200
    data = response.json()

    ids = {job["id"] for job in data.get("jobs", [])}
    assert ids == {"current"}
