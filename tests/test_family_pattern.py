from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from backend.agents.repair_orchestrator import repair_suspicious_row
from backend.agents.skills.extraction_diagnostics import (
    diagnose_extraction_row_issue,
    without_retired_pattern_warning,
)


@pytest.mark.parametrize("legacy_warning", [False, True])
def test_similar_valid_sizes_do_not_suggest_replacing_the_cutting_dimension(legacy_warning):
    row = {"order_number": "R-26-0781", "position": "5-1", "type": "LOWE",
           "dimension": "629x2087", "quantity": 1, "area": 1.31}
    diagnostics = diagnose_extraction_row_issue(row)
    if legacy_warning:
        diagnostics.update(severity="warning", requires_human_review=True,
                           family_pattern={"suggested_value": "624x2092"})
        diagnostics["issues"].append({"code": "POSSIBLE_DIMENSION_FAMILY_MISMATCH"})
    original = deepcopy((row, diagnostics))
    result = repair_suspicious_row(
        row, diagnostics,
        order_rows=[dict(row, position=position, dimension="624x2092") for position in ("11-1", "11-2")],
    )
    assert result["success"] is False
    assert result["suggested_value"] is None
    assert result["recommended_action"] == "NO_REPAIR_NEEDED"
    assert result["methods_used"] == ["diagnostics_analyzer"]
    assert (row, diagnostics) == original


def test_retired_pattern_cleanup_keeps_real_errors_and_recovered_review():
    row = {"position": "5-1", "dimension": "", "quantity": 1, "area": 1.31}
    diagnostics = diagnose_extraction_row_issue(row)
    expected = deepcopy(diagnostics)
    diagnostics["issues"].append({"code": "POSSIBLE_DIMENSION_FAMILY_MISMATCH"})
    diagnostics["family_pattern"] = {"suggested_value": "624x2092"}
    assert without_retired_pattern_warning(diagnostics) == expected

    recovered = {"severity": "warning", "requires_human_review": True, "issues": [
        {"code": "POSSIBLE_DIMENSION_FAMILY_MISMATCH"}, {"code": "RECOVERED_DIMENSION_REVIEW"},
    ]}
    result = without_retired_pattern_warning(recovered)
    assert result["issues"] == [{"code": "RECOVERED_DIMENSION_REVIEW"}]
    assert result["requires_human_review"] is True
    assert result["severity"] == "warning"


def test_accept_suggestion_preserves_original_rows_source_context():
    js = Path("docs/js/app.js").read_text(encoding="utf-8")
    start = js.index("function acceptRowSuggestion")
    end = js.index("function keepOriginalSuggestion", start)
    accept_body = js[start:end]

    assert "originalRows" in js
    assert "row[field] = coerceSuggestionValue(field, suggestion.suggested_value);" in accept_body
    assert "originalRows" not in accept_body
