from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from utils_text import (
    clean_dimension,
    compute_area_from_dimension,
    normalize_order_number,
    normalize_position,
)


SubtotalPattern = re.compile(r"^\s*m2\s+\d+\s+[\d.,]+\s*$", re.IGNORECASE)


class ValidationResult(Dict[str, Any]):
    rows: List[Dict[str, Any]]
    warnings: List[str]
    row_warnings: Dict[int, List[str]]


def _ensure_quantity(value: Any) -> int:
    try:
        qty = int(value)
    except Exception:
        return 1
    return max(qty, 1)


def _stringify_warning(message: str, idx: Optional[int] = None) -> str:
    if idx is None:
        return message
    return f"Row {idx + 1}: {message}"


def _looks_like_subtotal(row: Dict[str, Any]) -> bool:
    dimension = (row.get("dimension") or "").strip()
    position = (row.get("position") or "").strip()
    type_val = (row.get("type") or "").strip()
    if SubtotalPattern.match(type_val) or SubtotalPattern.match(position):
        return True
    if dimension == "" and type_val.lower().startswith("m2"):
        return True
    return False


def validate_rows(
    rows: List[Dict[str, Any]],
    *,
    context: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    processed: List[Dict[str, Any]] = []
    warnings: List[str] = []
    row_warnings: Dict[int, List[str]] = {}
    last_type_for_order: Dict[str, str] = {}
    split_candidates: List[Tuple[int, Dict[str, Any]]] = []
    split_flag = False

    for idx, row in enumerate(rows or []):
        working = dict(row)
        per_row: List[str] = []

        # Order number normalization
        normalized_order = normalize_order_number(working.get("order_number", ""))
        if normalized_order and normalized_order != working.get("order_number"):
            working["order_number"] = normalized_order
            per_row.append("auto_fix: order_number_normalized")

        # Position normalization
        normalized_position = normalize_position(working.get("position", ""))
        if normalized_position != working.get("position", ""):
            working["position"] = normalized_position
            per_row.append("auto_fix: position_normalized")

        # Dimension normalization
        dimension, dims = clean_dimension(working.get("dimension", ""))
        if dimension != (working.get("dimension") or ""):
            working["dimension"] = dimension
            if dimension:
                per_row.append("auto_fix: dimension_normalized")
            elif working.get("dimension"):
                per_row.append("warning: dimension_invalid_cleared")

        # Quantity sanity
        original_qty = working.get("quantity")
        quantity = _ensure_quantity(original_qty)
        if quantity != original_qty:
            per_row.append("auto_fix: quantity_defaulted_to_one")
        working["quantity"] = quantity

        # Subtotal bleed check
        if _looks_like_subtotal(working) and working.get("quantity", 0) > 1:
            working["quantity"] = 1
            per_row.append("auto_fix: subtotal_quantity_reset")

        if working.get("quantity", 0) > 6 and dims:
            width, height = dims
            if width * height < 400_000:  # roughly <0.4 m²
                working["quantity"] = max(1, min(3, working["quantity"]))
                per_row.append("warning: unusually_high_quantity_adjusted")

        # Area recompute
        area_from_dimension = compute_area_from_dimension(working.get("dimension", ""))
        try:
            area_value = float(working.get("area") or 0)
        except Exception:
            area_value = 0.0
        if area_from_dimension is not None:
            if not math.isclose(area_value, area_from_dimension, rel_tol=0.03, abs_tol=0.02):
                working["area"] = area_from_dimension
                per_row.append("auto_fix: area_recomputed")
            working["computed_area"] = area_from_dimension
            working["area_mismatch"] = not math.isclose(area_value, area_from_dimension, rel_tol=0.01, abs_tol=0.01)
        else:
            working["area"] = round(area_value, 3)
            working["computed_area"] = None
            working["area_mismatch"] = None

        # Type propagation
        order_key = working.get("order_number") or "__default__"
        current_type = (working.get("type") or "").strip()
        if current_type:
            if (
                order_key in last_type_for_order
                and last_type_for_order[order_key]
                and current_type.lower() != last_type_for_order[order_key].lower()
            ):
                per_row.append("warning: type_changed_without_header")
            last_type_for_order[order_key] = current_type
        else:
            prev = last_type_for_order.get(order_key, "")
            if prev:
                working["type"] = prev
                per_row.append("auto_fix: type_propagated")

        if (working.get("dimension") and not working.get("position")) or (
            not working.get("dimension") and working.get("position")
        ):
            split_candidates.append((idx, working))

        if per_row:
            row_warnings[idx] = per_row

        processed.append(working)

    # Split-line detection: dimension/position likely separated
    for i, current in split_candidates:
        next_idx = i + 1
        if next_idx < len(processed):
            next_row = processed[next_idx]
            dim_here = (current.get("dimension") or "").strip()
            pos_here = (current.get("position") or "").strip()
            dim_next = (next_row.get("dimension") or "").strip()
            pos_next = (next_row.get("position") or "").strip()
            area_next = float(next_row.get("area") or 0)
            if dim_here and not pos_here and pos_next:
                row_warnings.setdefault(i, []).append("warning: possible_split_line")
                row_warnings.setdefault(next_idx, []).append("warning: possible_split_line")
                split_flag = True
            elif not dim_here and pos_here and dim_next:
                row_warnings.setdefault(i, []).append("warning: possible_split_line")
                row_warnings.setdefault(next_idx, []).append("warning: possible_split_line")
                split_flag = True
            elif not dim_here and pos_here and area_next > 0:
                row_warnings.setdefault(i, []).append("warning: possible_split_line")
                split_flag = True

    # Aggregate warnings for missing data
    if not processed:
        warnings.append("No rows detected.")
    else:
        if not any(row.get("order_number") for row in processed):
            warnings.append("No order numbers detected.")

    if split_flag:
        warnings.append("possible_split_line detected")

    # Flatten row warnings into global context for quick view
    counted = {}
    for per_row in row_warnings.values():
        for item in per_row:
            counted[item] = counted.get(item, 0) + 1
    for message, count in counted.items():
        if message.startswith("auto_fix:"):
            warnings.append(f"{message} ({count})")

    return ValidationResult(rows=processed, warnings=warnings, row_warnings=row_warnings)


__all__ = ["validate_rows", "ValidationResult"]
