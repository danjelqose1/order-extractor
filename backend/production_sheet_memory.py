"""Bounded factory formatting memory, populated only by explicit finalization."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert

from production_sheets import (TYPOGRAPHY_FIELDS, SheetFeedbackRequest, measure_plan,
                               plan_sheet, sheet_features, source_digest)

MAX_EXAMPLES = 200


def _layout(plan, settings):
    return {"layout": plan["layout"], "columns": plan["columns"], "orientation": plan["orientation"],
            "margin_mm": settings.margin_mm, "pages_per_copy": len(plan["pages"]),
            "note_length": len(settings.note)}


def remember_sheet(request: SheetFeedbackRequest):
    from db import ProductionSheetExample, get_session
    if source_digest(request.source) != request.source_digest:
        raise ValueError("The finished layout belongs to a different sheet. Refresh before saving preferences.")
    plan = plan_sheet(request.source, request.settings)
    typography = {field: getattr(request.settings, field) for field in TYPOGRAPHY_FIELDS}
    changes = {field: round(typography[field] - getattr(request.baseline_settings, field), 3)
               for field in TYPOGRAPHY_FIELDS if typography[field] != getattr(request.baseline_settings, field)}
    example = {"features": sheet_features(request.source), "layout": _layout(plan, request.settings),
               "typography": typography, "manual_corrections": changes,
               "used_fraction": measure_plan(plan, request.settings)["mean_used_fraction"]}
    with get_session() as session:
        previous = session.get(ProductionSheetExample, request.source_digest)
        if previous and not changes:
            old = json.loads(previous.example_json)
            if old.get("typography") == typography:
                example["manual_corrections"] = old.get("manual_corrections", {})
        values = {"source_digest": request.source_digest, "example_json": json.dumps(example, sort_keys=True),
                  "updated_at": datetime.now(timezone.utc)}
        session.execute(insert(ProductionSheetExample).values(**values).on_conflict_do_update(
            index_elements=[ProductionSheetExample.source_digest], set_=values))
        keep = select(ProductionSheetExample.source_digest).order_by(
            ProductionSheetExample.updated_at.desc(), ProductionSheetExample.source_digest).limit(MAX_EXAMPLES)
        session.execute(delete(ProductionSheetExample).where(ProductionSheetExample.source_digest.not_in(keep)))
    return {"ok": True, "learned_from_corrections": bool(example["manual_corrections"])}


def similar_sheets(source, settings):
    from db import ProductionSheetExample, read_session
    target = sheet_features(source)
    layout = _layout(plan_sheet(source, settings), settings)
    with read_session() as session:
        records = session.scalars(select(ProductionSheetExample).order_by(
            ProductionSheetExample.updated_at.desc()).limit(MAX_EXAMPLES)).all()
        examples = [json.loads(record.example_json) for record in records]
    scored = []
    for example in examples:
        old_layout, old = example["layout"], example["features"]
        if any(old_layout[key] != layout[key] for key in ("layout", "columns", "orientation")):
            continue
        distance = sum(weight * abs(math.log((old[key] + 1) / (target[key] + 1))) for key, weight in (
            ("row_count", 1), ("glass_count", .7), ("order_count", .4), ("text_length", .7),
            ("detail_count", .4), ("longest_row_pt", 1)))
        distance += abs(old_layout["margin_mm"] - layout["margin_mm"]) / 5
        distance += abs(old_layout["note_length"] - layout["note_length"]) / 200
        if distance > 2.5:
            continue
        # Similarity leads; manual corrections are stronger than accepting an AI default.
        rank = distance + (0 if example["manual_corrections"] else .4)
        scored.append((rank, {**example, "similarity": round(1 / (1 + distance), 3)}))
    return [example for _, example in sorted(scored, key=lambda item: item[0])[:6]]
