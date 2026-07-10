from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "scripts" / "evaluate_manual_photo_assist.py"
SPEC = importlib.util.spec_from_file_location("manual_photo_assist_eval", SCRIPT_PATH)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_manual_photo_assist_eval_scores_normalized_rows_and_skips_uncertain_fields():
    case = {
        "id": "PTEST",
        "split": "development",
        "gold_status": "needs_user_confirmation",
        "document": {"client_name": "Eldi", "mode": "client_positions_red_index"},
        "rows": [
            {
                "section": "Vila 1",
                "client_position": "K1",
                "index": 1,
                "width_cm": 80,
                "height_cm": 120.5,
                "quantity": 2,
                "glass_type": "uncertain material",
                "uncertain_fields": ["glass_type"],
            }
        ],
    }
    prediction = {
        "status": "ready",
        "model": "observation -> reasoning",
        "document": {
            "client_name": {"value": "Eldi"},
            "order_number": {"value": None},
            "glass_type": {"value": "different material"},
        },
        "rows": [
            {
                "section": "Vila 1",
                "client_reference": {"value": "K1"},
                "index_number": {"value": 1},
                "width": {"normalized_mm": 800},
                "height": {"normalized_mm": 1205},
                "quantity": {"value": 2},
                "glass_type": {"value": "different material"},
                "notes": {"value": None},
            }
        ],
    }

    result = evaluation.score_case(case, prediction)

    assert result["row_count_correct"] is True
    assert result["exact_order"] is True
    assert "glass_type" not in result["fields"]
    assert result["fields"]["client_position"] == {"correct": 1, "total": 1}
    assert result["fields"]["width_cm"] == {"correct": 1, "total": 1}


def test_manual_photo_assist_eval_aggregate_reports_field_and_exact_accuracy():
    summary = evaluation.aggregate(
        [
            {
                "id": "P1",
                "fields": {"quantity": {"correct": 2, "total": 2}},
                "row_count_correct": True,
                "exact_order": True,
                "latency_ms": 100,
                "mismatch_count": 0,
            },
            {
                "id": "P2",
                "fields": {"quantity": {"correct": 1, "total": 2}},
                "row_count_correct": False,
                "exact_order": False,
                "latency_ms": 300,
                "mismatch_count": 2,
            },
        ]
    )

    assert summary["row_count_accuracy"] == 0.5
    assert summary["exact_order_accuracy"] == 0.5
    assert summary["field_accuracy"]["quantity"]["accuracy"] == 0.75
    assert summary["average_latency_ms"] == 200
    assert summary["p95_latency_ms"] == 300
    assert summary["cases_requiring_correction"] == 1
    assert summary["mismatched_values"] == 2


def test_manual_photo_assist_eval_exact_order_rejects_reordered_indexed_rows():
    case = {
        "id": "PORDER",
        "split": "development",
        "document": {"mode": "client_positions_red_index"},
        "rows": [
            {"index": 1, "width_cm": 50, "height_cm": 100, "quantity": 1},
            {"index": 2, "width_cm": 60, "height_cm": 110, "quantity": 1},
        ],
    }
    prediction = {
        "document": {},
        "rows": [
            {
                "index_number": {"value": 2},
                "width": {"normalized_mm": 600},
                "height": {"normalized_mm": 1100},
                "quantity": {"value": 1},
            },
            {
                "index_number": {"value": 1},
                "width": {"normalized_mm": 500},
                "height": {"normalized_mm": 1000},
                "quantity": {"value": 1},
            },
        ],
    }

    result = evaluation.score_case(case, prediction)

    assert result["row_count_correct"] is True
    assert result["fields"]["width_cm"] == {"correct": 2, "total": 2}
    assert result["exact_order"] is False
