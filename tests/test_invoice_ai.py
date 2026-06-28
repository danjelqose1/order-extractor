from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from invoice_ai import analyze_invoice_line, match_invoice_glass_type


class FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _client(*contents):
    completions = FakeCompletions(contents)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return client, completions


def test_glass_match_returns_only_a_supplied_canonical_key():
    client, completions = _client("33.1SATINATO")

    result = match_invoice_glass_type(
        client,
        raw_name="33.1 STAINATO",
        known_types=["4F", "33.1Satinato"],
    )

    assert result == "33.1Satinato"
    assert "33.1 STAINATO" in completions.calls[0]["messages"][1]["content"]


def test_glass_match_rejects_an_invented_key():
    client, _ = _client("8F")

    result = match_invoice_glass_type(
        client,
        raw_name="unknown glass",
        known_types=["4F", "6F"],
    )

    assert result is None


def test_invoice_line_analysis_is_validated_and_canonicalized(monkeypatch):
    monkeypatch.delenv("INVOICE_LINE_ANALYSIS_MODEL", raising=False)
    payload = {
        "normalizedType": "3 vetri 33.1 LOWE + 12 + 4F",
        "glassKey": "33.1lowe",
        "spacerMode": "thermal",
        "isLaminated": True,
        "confidence": 0.91,
        "reason": "Warm-edge laminated unit.",
    }
    client, completions = _client(json.dumps(payload))

    result = analyze_invoice_line(
        client,
        raw_line="3 vetri 33.1 LOE c.caldo + 4F",
        known_glass_types=["4F", "33.1LowE"],
    )

    assert result["glassKey"] == "33.1LowE"
    assert result["spacerMode"] == "thermal"
    assert result["confidence"] == pytest.approx(0.91)
    assert completions.calls[0]["model"] == "gpt-5.4-mini"
    assert completions.calls[0]["response_format"]["type"] == "json_schema"


def test_invoice_line_analysis_rejects_an_invented_key():
    payload = {
        "normalizedType": "8F",
        "glassKey": "8F",
        "spacerMode": "normal",
        "isLaminated": False,
        "confidence": 0.8,
        "reason": "Match",
    }
    client, _ = _client(json.dumps(payload))

    with pytest.raises(RuntimeError, match="unknown glassKey"):
        analyze_invoice_line(
            client,
            raw_line="8F",
            known_glass_types=["4F", "6F"],
        )
