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
