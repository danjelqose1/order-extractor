#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx


DEFAULT_MANIFEST = Path("data/private-evals/manual-photo-assist/manifest.json")
DEFAULT_BASE_URL = "https://order-extractor-kdih.onrender.com"
TEXT_FIELDS = {"client_name", "order_number", "section", "glass_type", "notes", "client_position", "position"}
NUMERIC_FIELDS = {"index", "width_cm", "height_cm", "quantity"}


def text_key(value: Any) -> str:
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def values_match(field: str, expected: Any, actual: Any) -> bool:
    if field in TEXT_FIELDS:
        return text_key(expected) == text_key(actual)
    if field in NUMERIC_FIELDS:
        if expected is None or actual is None:
            return expected is None and actual is None
        try:
            return math.isclose(float(expected), float(actual), abs_tol=0.01)
        except (TypeError, ValueError):
            return False
    return expected == actual


def predicted_mode(prediction: Dict[str, Any]) -> str:
    rows = prediction.get("rows") or []
    structured = any(
        str(row.get("section") or "").strip()
        or str((row.get("client_reference") or {}).get("value") or "").strip()
        or (row.get("index_number") or {}).get("value") is not None
        for row in rows
    )
    return "client_positions_red_index" if structured else "standard"


def prediction_row_value(row: Dict[str, Any], field: str, ordinal: int) -> Any:
    if field == "position":
        client_position = (row.get("client_reference") or {}).get("value")
        index_number = (row.get("index_number") or {}).get("value")
        return client_position or index_number or str(ordinal)
    if field == "client_position":
        return (row.get("client_reference") or {}).get("value")
    if field == "index":
        return (row.get("index_number") or {}).get("value")
    if field in {"width_cm", "height_cm"}:
        dimension_name = "width" if field == "width_cm" else "height"
        normalized_mm = (row.get(dimension_name) or {}).get("normalized_mm")
        return float(normalized_mm) / 10 if normalized_mm is not None else None
    if field == "quantity":
        return (row.get("quantity") or {}).get("value")
    if field == "glass_type":
        return (row.get("glass_type") or {}).get("value")
    if field == "notes":
        return (row.get("notes") or {}).get("value")
    if field == "section":
        return row.get("section")
    return row.get(field)


def _row_pairing(expected_rows: List[Dict[str, Any]], predicted_rows: List[Dict[str, Any]]) -> Iterable[Tuple[Dict[str, Any], Optional[Dict[str, Any]], int]]:
    predicted_by_index = {
        (row.get("index_number") or {}).get("value"): row
        for row in predicted_rows
        if (row.get("index_number") or {}).get("value") is not None
    }
    for ordinal, expected in enumerate(expected_rows, start=1):
        predicted = None
        expected_index = expected.get("index")
        if expected_index is not None:
            predicted = predicted_by_index.get(expected_index)
        if predicted is None and ordinal <= len(predicted_rows):
            predicted = predicted_rows[ordinal - 1]
        yield expected, predicted, ordinal


def rows_match_in_order(expected_rows: List[Dict[str, Any]], predicted_rows: List[Dict[str, Any]]) -> bool:
    if len(expected_rows) != len(predicted_rows):
        return False
    for ordinal, (expected, predicted) in enumerate(zip(expected_rows, predicted_rows), start=1):
        uncertain = set(expected.get("uncertain_fields") or [])
        for field, expected_value in expected.items():
            if field == "uncertain_fields" or field in uncertain or field not in TEXT_FIELDS | NUMERIC_FIELDS:
                continue
            if not values_match(field, expected_value, prediction_row_value(predicted, field, ordinal)):
                return False
    return True


def score_case(case: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    latency_ms = (prediction.get("_eval_meta") or {}).get("latency_ms")
    if prediction.get("error"):
        return {
            "id": case["id"],
            "split": case["split"],
            "error": prediction["error"],
            "fields": {},
            "row_count_correct": False,
            "exact_order": False,
            "latency_ms": latency_ms,
        }

    fields: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    mismatches: List[Dict[str, Any]] = []
    document_fields_correct = True
    document = case.get("document") or {}
    predicted_document = prediction.get("document") or {}
    unscored_document = set(case.get("unscored_document_fields") or [])
    document_mapping = {
        "client_name": (predicted_document.get("client_name") or {}).get("value"),
        "order_number": (predicted_document.get("order_number") or {}).get("value"),
        "default_glass_type": (predicted_document.get("glass_type") or {}).get("value"),
        "mode": predicted_mode(prediction),
    }
    for field, expected in document.items():
        if field in unscored_document:
            continue
        fields[field]["total"] += 1
        actual = document_mapping.get(field)
        if values_match("glass_type" if field == "default_glass_type" else field, expected, actual):
            fields[field]["correct"] += 1
        else:
            document_fields_correct = False
            mismatches.append({"scope": "document", "field": field, "expected": expected, "actual": actual})

    expected_rows = case.get("rows") or []
    predicted_rows = prediction.get("rows") or []
    row_count_correct = len(expected_rows) == len(predicted_rows)
    for expected, predicted, ordinal in _row_pairing(expected_rows, predicted_rows):
        uncertain = set(expected.get("uncertain_fields") or [])
        for field, expected_value in expected.items():
            if field in {"uncertain_fields"} or field in uncertain:
                continue
            if field not in TEXT_FIELDS | NUMERIC_FIELDS:
                continue
            fields[field]["total"] += 1
            actual_value = prediction_row_value(predicted or {}, field, ordinal) if predicted else None
            matched = predicted is not None and values_match(field, expected_value, actual_value)
            if matched:
                fields[field]["correct"] += 1
            else:
                mismatches.append(
                    {
                        "scope": "row",
                        "row": ordinal,
                        "field": field,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )

    return {
        "id": case["id"],
        "split": case["split"],
        "gold_status": case.get("gold_status"),
        "model": prediction.get("model"),
        "status": prediction.get("status"),
        "expected_rows": len(expected_rows),
        "predicted_rows": len(predicted_rows),
        "row_count_correct": row_count_correct,
        "exact_order": document_fields_correct and rows_match_in_order(expected_rows, predicted_rows),
        "mismatch_count": len(mismatches) + abs(len(expected_rows) - len(predicted_rows)),
        "latency_ms": latency_ms,
        "fields": dict(fields),
        "mismatches": mismatches,
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    fields: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    successful = [result for result in results if not result.get("error")]
    for result in successful:
        for field, counts in result["fields"].items():
            fields[field]["correct"] += counts["correct"]
            fields[field]["total"] += counts["total"]
    field_accuracy = {
        field: {
            **counts,
            "accuracy": round(counts["correct"] / counts["total"], 4) if counts["total"] else None,
        }
        for field, counts in sorted(fields.items())
    }
    total = len(results)
    latencies = sorted(
        float(result["latency_ms"])
        for result in successful
        if result.get("latency_ms") is not None
    )
    p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1) if latencies else 0
    return {
        "cases": total,
        "successful_cases": len(successful),
        "row_count_accuracy": round(sum(result.get("row_count_correct", False) for result in results) / total, 4) if total else None,
        "exact_order_accuracy": round(sum(result.get("exact_order", False) for result in results) / total, 4) if total else None,
        "fallback_rate": round(
            sum("gpt-5.4-mini" in str(result.get("model") or "") for result in successful) / len(successful),
            4,
        ) if successful else None,
        "cases_requiring_correction": sum(not result.get("exact_order", False) for result in results),
        "mismatched_values": sum(int(result.get("mismatch_count") or 0) for result in results),
        "average_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "p95_latency_ms": round(latencies[p95_index], 1) if latencies else None,
        "field_accuracy": field_accuracy,
    }


def run_extraction(base_url: str, image_path: Path) -> Dict[str, Any]:
    started_at = time.perf_counter()
    with image_path.open("rb") as image_file:
        response = httpx.post(
            f"{base_url.rstrip('/')}/api/manual-orders/photo-assist/extract",
            files={"image": (image_path.name, image_file, "image/jpeg")},
            data={"dimension_unit": "cm"},
            timeout=httpx.Timeout(300.0, connect=30.0),
        )
    latency_ms = round((time.perf_counter() - started_at) * 1000, 1)
    if response.is_success:
        prediction = response.json()
        prediction["_eval_meta"] = {"latency_ms": latency_ms}
        return prediction
    try:
        detail = response.json().get("detail")
    except Exception:
        detail = response.text
    return {"error": f"HTTP {response.status_code}: {detail}", "_eval_meta": {"latency_ms": latency_ms}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and score the private Manual Photo Assist evaluation set.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--split", choices=["development", "holdout", "all"], default="development")
    parser.add_argument("--run", action="store_true", help="Call the configured backend before scoring.")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    predictions_dir = root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = [case for case in cases if args.split == "all" or case.get("split") == args.split]
    results = []

    for case in selected:
        prediction_path = predictions_dir / f"{case['id']}.json"
        if args.run:
            prediction = run_extraction(args.base_url, root / case["image"])
            prediction_path.write_text(json.dumps(prediction, indent=2, ensure_ascii=False), encoding="utf-8")
        elif prediction_path.exists():
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        else:
            prediction = {"error": "No prediction file. Run with --run first."}
        result = score_case(case, prediction)
        results.append(result)
        print(f"{case['id']}: rows={result.get('predicted_rows', 0)}/{result.get('expected_rows', len(case.get('rows') or []))} exact={result.get('exact_order', False)}")

    report = {"summary": aggregate(results), "cases": results}
    report_path = root / f"report-{args.split}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
