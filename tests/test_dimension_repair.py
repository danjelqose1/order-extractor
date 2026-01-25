from __future__ import annotations

from backend.dimension_repair import apply_dimension_repair


def test_dimension_repair_truncated_height_from_raw_text():
    raw_text = (
        "R-25-1290/20-3/KATI 2  337 x 1022  0,350\n"
        "R-25-1290/22-1  704 x 2301  1,620"
    )
    items = [
        {"position": "20-3/KATI 2", "dimension": "337x102", "order_number": "R-25-1290"},
        {"position": "22-1", "dimension": "704x230", "order_number": "R-25-1290"},
    ]

    repaired = apply_dimension_repair(raw_text, items)

    first = repaired[0]
    assert first["dimension"] == "337x102"
    assert first["correctedWidth"] == 337
    assert first["correctedHeight"] == 1022
    assert first["correctedDimension"] == "337x1022"
    assert "SUSPICIOUS_TRUNCATION" in first.get("flags", [])

    second = repaired[1]
    assert second["dimension"] == "704x230"
    assert second["correctedWidth"] == 704
    assert second["correctedHeight"] == 2301
    assert second["correctedDimension"] == "704x2301"
    assert "SUSPICIOUS_TRUNCATION" in second.get("flags", [])
