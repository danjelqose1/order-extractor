from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from validators import validate_rows


def test_small_piece_high_quantity_is_flagged_without_mutation():
    row = {
        "order_number": "R-26-0042",
        "type": "2 VETRI 4F + 16 + 4 LOWE 24mm",
        "dimension": "300x300",
        "position": "1-1",
        "quantity": 8,
        "area": 0.72,
    }

    result = validate_rows([row])

    assert result["rows"][0]["quantity"] == 8
    assert "warning: unusually_high_quantity" in result["row_warnings"][0]


def test_operator_confirmed_two_digit_dimension_is_preserved():
    row = {
        "order_number": "R-26-0634",
        "type": "2 VETRI 33.1SANT +18+ 4LOWE (28MM)",
        "dimension": "98x2504",
        "position": "6-1",
        "quantity": 1,
        "area": 0.25,
    }

    result = validate_rows([row])

    assert result["rows"][0]["dimension"] == "98x2504"
    assert not any(
        "dimension_invalid_cleared" in warning
        for warning in result.get("row_warnings", {}).get(0, [])
    )
