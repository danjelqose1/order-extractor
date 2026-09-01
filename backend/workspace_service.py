from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz
from sqlalchemy import func, select

from area_dimension_validator import apply_area_dimension_validation
from backend.agents.skills.extraction_diagnostics import diagnose_extraction_row_issue
from db import (
    Order,
    ProcessingBatch,
    ProductionFile,
    SessionLocal,
    get_order_with_extraction,
    get_orders,
    get_orders_by_identifiers,
    normalize_order_status,
    record_workspace_action,
    update_order_status,
)
from utils_text import clean_dimension, parse_declared_totals
from validators import validate_rows
from time_utils import utc_isoformat


PRODUCTION_FILE_TYPES = {"processing_pdf", "labels_pdf"}
PENDING_ACTION_TTL_SECONDS = int(os.getenv("WORKSPACE_PENDING_ACTION_TTL_SECONDS", "1200"))
_PENDING_ACTIONS: Dict[str, Dict[str, Any]] = {}


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _data_dir() -> Path:
    root = Path(os.getenv("DB_DIR", "data"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _production_dir() -> Path:
    path = _data_dir() / "production-files"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_filename(value: Any) -> str:
    text = str(value or "order").strip() or "order"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "order"


def _primary_order_number(order: Dict[str, Any]) -> str:
    numbers = order.get("order_numbers") or []
    if isinstance(numbers, list) and numbers:
        return str(numbers[0])
    for row in order.get("rows") or []:
        if row.get("order_number"):
            return str(row["order_number"])
    return str(order.get("id") or "")


def _client_name(order: Dict[str, Any]) -> str:
    return str(order.get("client_name") or order.get("clientName") or order.get("client") or order.get("client_hint") or "").strip()


def _summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_pieces = 0
    total_area = 0.0
    glass_types = set()
    for row in rows or []:
        try:
            total_pieces += int(row.get("quantity") or 0)
        except Exception:
            pass
        try:
            total_area += float(row.get("area") or 0.0)
        except Exception:
            pass
        glass_type = str(row.get("type") or "").strip()
        if glass_type:
            glass_types.add(glass_type)
    return {
        "total_pieces": total_pieces,
        "total_area_m2": round(total_area, 3),
        "glass_type_count": len(glass_types),
        "line_count": len(rows or []),
    }


def _resolve_order(identifier: Any) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if identifier is None:
        return None, []
    text = str(identifier).strip()
    if not text:
        return None, []
    matches = get_orders_by_identifiers([text])
    if not matches:
        return None, []
    if len(matches) == 1:
        return matches[0], matches

    normalized = text.lower()
    exact = []
    for order in matches:
        if str(order.get("id")) == text:
            exact.append(order)
            continue
        nums = [str(num).lower() for num in (order.get("order_numbers") or [])]
        if normalized in nums:
            exact.append(order)
    if len(exact) == 1:
        return exact[0], matches
    return None, matches


def _normalize_workspace_order_token(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "")
    if re.fullmatch(r"R-?\d{2}-\d{3,5}", text, re.IGNORECASE):
        return "R-" + re.sub(r"^R-?", "", text, flags=re.IGNORECASE)
    if re.fullmatch(r"\d{2}-\d{3,5}", text):
        return f"R-{text}"
    return str(value or "").strip()


def _pending_session_key(context: Optional[Dict[str, Any]], actor: str) -> str:
    if isinstance(context, dict):
        session_id = str(context.get("workspace_session_id") or context.get("session_id") or "").strip()
        if session_id:
            return f"session:{session_id}"
    return f"actor:{actor or 'workspace'}"


def _cleanup_pending_actions() -> None:
    now = time.time()
    expired = [key for key, value in _PENDING_ACTIONS.items() if now - float(value.get("created_epoch") or 0) > PENDING_ACTION_TTL_SECONDS]
    for key in expired:
        _PENDING_ACTIONS.pop(key, None)


def infer_merge_across_orders(message: Optional[str]) -> bool:
    lower = str(message or "").lower()
    if any(
        phrase in lower
        for phrase in (
            "keep orders separate",
            "keep them separate",
            "don't mix orders",
            "do not mix orders",
            "without mixing orders",
            "no merge across orders",
            "do not merge across orders",
            "don't merge across orders",
        )
    ):
        return False
    return any(
        phrase in lower
        for phrase in (
            "merge across orders",
            "merge orders",
            "mix orders",
            "mix same glass across orders",
            "combine same glass across orders",
        )
    )


def _frontend_processing_action(
    identifiers: Sequence[Any],
    mode: str,
    *,
    merge_across_orders: bool = False,
    force: bool = False,
    pending_action_id: Optional[str] = None,
) -> Dict[str, Any]:
    cleaned = [str(item) for item in identifiers if str(item or "").strip()]
    normalized_mode = "separate" if mode == "separate" else ("combined" if len(cleaned) > 1 else "single")
    action: Dict[str, Any] = {
        "type": "process_via_existing_modules",
        "identifiers": cleaned,
        "mode": normalized_mode,
        "mergeAcrossOrders": bool(merge_across_orders) if normalized_mode == "combined" else False,
    }
    if cleaned:
        action["identifier"] = cleaned[0]
    if force:
        action["force"] = True
    if pending_action_id:
        action["confirmed_pending_action_id"] = pending_action_id
    return {
        "message": "Use the Processing module workflow for these orders." if len(cleaned) > 1 else "Use the Processing module workflow for this order.",
        "status": "frontend_workflow_required",
        "actions": [action],
    }


def _warning_items(order_number: str, warnings: Sequence[Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for warning in warnings or []:
        text = str(warning)
        row = None
        match = re.search(r"\brow\s+(\d+)\b", text, re.IGNORECASE)
        if match:
            try:
                row = int(match.group(1))
            except Exception:
                row = None
        code = text.split(":", 1)[0].strip() if ":" in text else "validation_warning"
        items.append({
            "order_number": order_number,
            "row": row,
            "code": code,
            "message": text,
        })
    return items


def prepare_workspace_processing_action(
    identifiers: Sequence[Any],
    *,
    mode: str = "combined",
    merge_across_orders: Optional[bool] = None,
    requested_by: str = "workspace_agent",
    original_message: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _cleanup_pending_actions()
    requested = [_normalize_workspace_order_token(item) for item in identifiers if str(item or "").strip()]
    if not requested:
        return {"message": "Tell me which order to process.", "status": "missing_order"}

    resolved_orders: List[Dict[str, Any]] = []
    missing: List[str] = []
    multiple: List[Dict[str, Any]] = []
    for identifier in requested:
        order_ref, matches = _resolve_order(identifier)
        if not order_ref:
            if matches:
                multiple.append({"identifier": identifier, "matches": [_queue_card(match) for match in matches]})
            else:
                missing.append(identifier)
            continue
        detail = get_order_with_extraction(int(order_ref["id"])) or order_ref
        resolved_orders.append(detail)

    if missing or multiple:
        result = {
            "status": "not_found" if missing else "multiple_matches",
            "message": (
                f"I couldn't find {', '.join(missing)}. No files were created."
                if missing else "I found multiple matching orders. Please choose one."
            ),
            "missing": missing,
            "multiple_matches": multiple,
        }
        record_workspace_action(
            actor=requested_by,
            action_type="process_orders_blocked",
            status=result["status"],
            requested_message=original_message,
            input_json={"identifiers": requested, "mode": mode},
            output_json=result,
        )
        return result

    blocked = []
    blockers: List[Dict[str, Any]] = []
    advisories: List[Dict[str, Any]] = []
    blocker_order_ids: List[str] = []
    order_ids: List[str] = []
    order_numbers: List[str] = []
    summaries: Dict[str, Any] = {}
    for order in resolved_orders:
        order_id = str(order.get("id"))
        order_number = _primary_order_number(order)
        order_ids.append(order_id)
        order_numbers.append(order_number)
        normalized_status = normalize_order_status(order.get("status"))
        if normalized_status != "approved":
            blocked.append({"order_id": order_id, "order_number": order_number, "status": normalized_status})
            continue
        validation = _validate_order(order)
        summaries[order_number] = validation["summary"]
        if validation["blocker_warnings"]:
            blockers.extend(_warning_items(order_number, validation["blocker_warnings"]))
            blocker_order_ids.append(order_id)
        if validation["advisory_warnings"]:
            advisories.extend(_warning_items(order_number, validation["advisory_warnings"]))

    if blocked:
        blocked_labels = ", ".join(
            f"{item.get('order_number') or item.get('order_id')} is {item.get('status') or 'not approved'}"
            for item in blocked
        )
        result = {
            "status": "blocked",
            "reason": "order_not_approved",
            "message": (
                "This order is still Draft. Review and approve it before production."
                if len(blocked) == 1 and blocked[0].get("status") == "draft"
                else f"{blocked_labels}. Review and approve before production."
            ),
            "blocked_orders": blocked,
        }
        record_workspace_action(
            actor=requested_by,
            action_type="process_orders_blocked",
            status="blocked",
            requested_message=original_message,
            input_json={"identifiers": requested, "mode": mode},
            output_json=result,
        )
        return result

    if blockers:
        result = {
            "status": "needs_review",
            "reason": "deterministic_preflight_blocked",
            "message": "Production preflight found required data that must be corrected before files can be created.",
            "warnings": blockers,
            "blocked_orders": sorted({item.get("order_number") for item in blockers if item.get("order_number")}),
            "actions": [{
                "type": "open_review",
                "order_ids": blocker_order_ids,
                "order_numbers": sorted({item.get("order_number") for item in blockers if item.get("order_number")}),
            }],
        }
        record_workspace_action(
            actor=requested_by,
            action_type="process_orders_blocked",
            status="needs_review",
            requested_message=original_message,
            input_json={"identifiers": requested, "mode": mode},
            output_json=result,
        )
        return result

    resolved_merge_across_orders = infer_merge_across_orders(original_message) if merge_across_orders is None else bool(merge_across_orders)
    result = _frontend_processing_action(order_ids, mode, merge_across_orders=resolved_merge_across_orders)
    result["order_numbers"] = order_numbers
    result["informational_warnings"] = advisories
    if advisories:
        result["message"] = "Preflight passed with advisory warning(s); continue through the Processing module workflow."
    record_workspace_action(
        actor=requested_by,
        action_type="process_orders_delegated",
        status="frontend_workflow_required",
        requested_message=original_message,
        input_json={"identifiers": requested, "mode": mode, "mergeAcrossOrders": resolved_merge_across_orders},
        output_json=result,
    )
    return result


def confirm_workspace_pending_action(
    pending_action_id: str,
    *,
    decision: str,
    requested_by: str = "workspace_agent",
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _cleanup_pending_actions()
    pending = _PENDING_ACTIONS.get(str(pending_action_id or "").strip())
    if not pending:
        return {"status": "invalid_pending_action", "message": "That pending action is no longer available. Tell me which order to process."}
    if pending.get("session_key") != _pending_session_key(context, requested_by):
        return {"status": "invalid_pending_action", "message": "That pending action belongs to another session."}
    if time.time() - float(pending.get("created_epoch") or 0) > PENDING_ACTION_TTL_SECONDS:
        _PENDING_ACTIONS.pop(pending["pending_action_id"], None)
        return {"status": "expired_pending_action", "message": "That pending action expired. Tell me which order to process again."}

    decision = (decision or "").strip().lower()
    if decision == "cancel":
        _PENDING_ACTIONS.pop(pending["pending_action_id"], None)
        record_workspace_action(
            actor=requested_by,
            action_type="pending_processing_action_cancelled",
            status="cancelled",
            input_json={"pending_action_id": pending["pending_action_id"]},
        )
        return {"status": "cancelled", "message": "Cancelled. No production files were created."}
    if decision == "open_review":
        return {
            "status": "open_review",
            "message": "Open review for the warning orders before processing.",
            "actions": [{"type": "open_review", "order_ids": pending.get("order_ids") or [], "order_numbers": pending.get("order_numbers") or []}],
            "pending_action": {key: value for key, value in pending.items() if key not in {"created_epoch", "session_key"}},
        }
    if decision != "continue":
        return {"status": "invalid_decision", "message": "Choose Continue Anyway, Open Review, or Cancel."}

    blocked = []
    for order_id in pending.get("order_ids") or []:
        order = get_order_with_extraction(int(order_id)) if str(order_id).isdigit() else None
        if not order:
            blocked.append({"order_id": order_id, "reason": "not_found"})
            continue
        normalized_status = normalize_order_status(order.get("status"))
        if normalized_status != "approved":
            blocked.append({"order_id": order_id, "order_number": _primary_order_number(order), "reason": "order_not_approved", "status": normalized_status})
    if blocked:
        result = {
            "status": "blocked",
            "message": "One or more orders changed status after confirmation was requested. I stopped before processing.",
            "blocked_orders": blocked,
        }
        record_workspace_action(
            actor=requested_by,
            action_type="pending_processing_action_blocked",
            status="blocked",
            input_json={"pending_action_id": pending["pending_action_id"]},
            output_json=result,
        )
        return result

    _PENDING_ACTIONS.pop(pending["pending_action_id"], None)
    result = _frontend_processing_action(
        pending.get("order_ids") or pending.get("order_numbers") or [],
        pending.get("mode") or "combined",
        merge_across_orders=bool(pending.get("mergeAcrossOrders")),
        force=True,
        pending_action_id=pending["pending_action_id"],
    )
    result["message"] = "Continuing the pending processing action."
    result["pending_action"] = {key: value for key, value in pending.items() if key not in {"created_epoch", "session_key"}}
    record_workspace_action(
        actor=requested_by,
        action_type="pending_processing_action_confirmed",
        status="frontend_workflow_required",
        input_json={"pending_action_id": pending["pending_action_id"], "decision": decision},
        output_json=result,
        confirmed=True,
    )
    return result


def _existing_batch(order_id: int) -> Optional[Dict[str, Any]]:
    with SessionLocal() as session:
        batch = session.execute(
            select(ProcessingBatch)
            .where(ProcessingBatch.order_id == order_id)
            .where(ProcessingBatch.status != "failed")
            .order_by(ProcessingBatch.created_at.desc())
        ).scalars().first()
        if not batch:
            return None
        return _serialize_batch(session, batch)


def _serialize_file(file: ProductionFile) -> Dict[str, Any]:
    return {
        "id": file.id,
        "created_at": utc_isoformat(file.created_at),
        "order_id": file.order_id,
        "order_number": file.order_number,
        "processing_batch_id": file.processing_batch_id,
        "file_type": file.file_type,
        "download_url": file.download_url,
        "status": file.status,
    }


def _serialize_batch(session, batch: ProcessingBatch) -> Dict[str, Any]:
    files = session.execute(
        select(ProductionFile)
        .where(ProductionFile.processing_batch_id == batch.id)
        .order_by(ProductionFile.created_at.desc())
    ).scalars().all()
    by_type = {file.file_type: _serialize_file(file) for file in files}
    try:
        summary = json.loads(batch.summary_json or "{}")
    except Exception:
        summary = {}
    return {
        "id": batch.id,
        "created_at": utc_isoformat(batch.created_at),
        "updated_at": utc_isoformat(batch.updated_at),
        "order_id": batch.order_id,
        "order_number": batch.order_number,
        "status": batch.status,
        "requested_by": batch.requested_by,
        "forced": bool(batch.forced),
        "summary": summary,
        "files": by_type,
        "processing_pdf_url": by_type.get("processing_pdf", {}).get("download_url"),
        "labels_pdf_url": by_type.get("labels_pdf", {}).get("download_url"),
    }


def _validate_order(order: Dict[str, Any]) -> Dict[str, Any]:
    rows = [dict(row) for row in (order.get("rows") or [])]
    extraction = order.get("extraction") or {}
    prepared_text = extraction.get("prepared_text", "")
    validation = validate_rows(rows, context={"prepared_text": prepared_text})
    normalized_rows = validation.get("rows", rows)
    final_rows = apply_area_dimension_validation(normalized_rows)
    warnings = list(validation.get("warnings", []) or [])
    row_warnings = validation.get("row_warnings", {}) or {}
    declared_units, declared_area = parse_declared_totals(prepared_text or "")
    summary = _summarize_rows(final_rows)
    if declared_units is not None and declared_units != summary["total_pieces"]:
        warnings.append(f"declared_units_mismatch: declared {declared_units}, parsed {summary['total_pieces']}")
    if declared_area is not None and abs((declared_area or 0.0) - summary["total_area_m2"]) > 0.05:
        warnings.append(f"declared_area_mismatch: declared {declared_area:.3f}, parsed {summary['total_area_m2']:.3f}")

    blockers: List[str] = []
    advisories: List[str] = []
    informational: List[str] = []

    def add_warning(message: Any, *, diagnostic_severity: Optional[str] = None) -> None:
        text = str(message or "").strip()
        if not text:
            return
        lower = text.lower()
        if "auto_fix:" in lower:
            informational.append(text)
            return
        blocker_tokens = (
            "missing_required_field",
            "dimension_invalid",
            "area_not_numeric",
            "no rows detected",
            "no order numbers detected",
            "invalid dimension",
            "impossible dimension",
            "missing dimension",
            "missing glass",
            "missing type",
            "missing quantity",
        )
        if diagnostic_severity == "error" or any(token in lower for token in blocker_tokens):
            blockers.append(text)
            return
        advisories.append(text)

    for warning in warnings:
        add_warning(warning)
    for idx, row in enumerate(final_rows, start=1):
        diagnostics = diagnose_extraction_row_issue(row)
        if diagnostics.get("severity") not in {"warning", "error"}:
            continue
        for issue in diagnostics.get("issues") or []:
            message = issue.get("message") or issue.get("code") or "Extraction diagnostic warning"
            add_warning(f"Row {idx}: {message}", diagnostic_severity=diagnostics.get("severity"))
    for key, values in (row_warnings.items() if isinstance(row_warnings, dict) else []):
        try:
            row_label = int(key) + 1
        except (TypeError, ValueError):
            row_label = key
        for value in values or []:
            add_warning(f"Row {row_label}: {value}")

    def dedupe(items: Sequence[Any]) -> List[str]:
        output: List[str] = []
        seen = set()
        for item in items:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                output.append(text)
        return output

    deduped_blockers = dedupe(blockers)
    deduped_advisories = dedupe(advisories)
    deduped_informational = dedupe(informational)

    return {
        "ok": not deduped_blockers,
        "rows": final_rows,
        "warnings": warnings,
        "critical_warnings": deduped_blockers,
        "blocker_warnings": deduped_blockers,
        "advisory_warnings": deduped_advisories,
        "informational_warnings": deduped_informational,
        "row_warnings": row_warnings,
        "summary": summary,
    }


def validate_order_for_processing(order_number_or_id: Any) -> Dict[str, Any]:
    order, matches = _resolve_order(order_number_or_id)
    if not order:
        return {"status": "multiple_matches" if matches else "not_found", "matches": [_queue_card(m) for m in matches]}
    validation = _validate_order(get_order_with_extraction(int(order["id"])) or order)
    return {
        "status": "ok" if validation["ok"] else "needs_review",
        "order": _queue_card(order),
        "warnings": validation["blocker_warnings"],
        "advisories": validation["advisory_warnings"],
        "summary": validation["summary"],
    }


def _draw_wrapped(page, text: str, rect: fitz.Rect, fontsize: float = 10, bold: bool = False) -> float:
    fontname = "helv"
    page.insert_textbox(rect, text, fontsize=fontsize, fontname=fontname, align=0)
    return rect.y1


def _new_pdf(title: str) -> Tuple[fitz.Document, fitz.Page, float]:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((40, 48), title, fontsize=18, fontname="helv")
    return doc, page, 78.0


def _ensure_space(doc: fitz.Document, page: fitz.Page, y: float, required: float = 36) -> Tuple[fitz.Page, float]:
    if y + required <= 800:
        return page, y
    page = doc.new_page(width=595, height=842)
    return page, 44.0


def _generate_processing_pdf(order: Dict[str, Any], rows: Sequence[Dict[str, Any]], batch_id: int, path: Path) -> None:
    doc, page, y = _new_pdf("Processing Sheet")
    order_number = _primary_order_number(order)
    client = _client_name(order) or "-"
    summary = _summarize_rows(rows)
    header = (
        f"Order: {order_number}\nClient: {client}\n"
        f"Batch: {batch_id}\nPieces: {summary['total_pieces']}   Area: {summary['total_area_m2']:.3f} m2"
    )
    y = _draw_wrapped(page, header, fitz.Rect(40, y, 555, y + 58), fontsize=11) + 12
    current_type = ""
    index = 1
    for row in rows:
        page, y = _ensure_space(doc, page, y, 42)
        glass_type = str(row.get("type") or "(Header not set)").strip()
        if glass_type != current_type:
            current_type = glass_type
            y = _draw_wrapped(page, current_type, fitz.Rect(40, y, 555, y + 28), fontsize=12) + 2
        dim = str(row.get("dimension") or "").replace("×", "x")
        qty = int(row.get("quantity") or 0)
        position = str(row.get("position") or "-")
        area = float(row.get("area") or 0.0)
        line = f"{index}. {dim} mm x {qty}   pos {position}   area {area:.3f} m2"
        y = _draw_wrapped(page, line, fitz.Rect(56, y, 555, y + 20), fontsize=10) + 2
        index += 1
    doc.save(path)
    doc.close()


def _generate_labels_pdf(order: Dict[str, Any], rows: Sequence[Dict[str, Any]], batch_id: int, path: Path) -> None:
    doc, page, y = _new_pdf("Production Labels")
    order_number = _primary_order_number(order)
    client = _client_name(order) or "-"
    x_positions = [40, 310]
    label_w = 245
    label_h = 92
    x_index = 0
    for row in rows:
        qty = max(0, int(row.get("quantity") or 0))
        for _ in range(qty):
            if y + label_h > 800:
                page = doc.new_page(width=595, height=842)
                y = 44
                x_index = 0
            x = x_positions[x_index]
            rect = fitz.Rect(x, y, x + label_w, y + label_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            text = (
                f"{order_number}\n{client}\n"
                f"{row.get('dimension') or '-'} mm\n"
                f"{row.get('type') or '-'}\n"
                f"Pos: {row.get('position') or '-'}   Batch: {batch_id}"
            )
            _draw_wrapped(page, text, fitz.Rect(x + 8, y + 8, x + label_w - 8, y + label_h - 8), fontsize=10)
            if x_index == 0:
                x_index = 1
            else:
                x_index = 0
                y += label_h + 14
        if x_index == 1:
            continue
    doc.save(path)
    doc.close()


def _insert_file_record(session, *, order_id: int, batch_id: int, order_number: str, file_type: str, path: Path) -> ProductionFile:
    file = ProductionFile(
        order_id=order_id,
        processing_batch_id=batch_id,
        order_number=order_number,
        file_type=file_type,
        file_path=str(path),
        download_url="",
        status="ready",
    )
    session.add(file)
    session.flush()
    file.download_url = f"/api/workspace/files/{file.id}/download"
    session.flush()
    return file


def process_approved_order(order_number_or_id: Any, requested_by: str = "workspace", force: bool = False) -> Dict[str, Any]:
    identifier = str(order_number_or_id or "").strip()
    record_workspace_action(
        actor=requested_by,
        action_type="process_order_requested",
        status="received",
        order_number=identifier,
        tool_name="process_approved_order",
        input_json={"identifier": identifier, "force": force},
    )

    order_ref, matches = _resolve_order(identifier)
    if not order_ref:
        status = "multiple_matches" if matches else "not_found"
        result = {"status": status, "matches": [_queue_card(match) for match in matches]}
        record_workspace_action(
            actor=requested_by,
            action_type="process_order_blocked",
            status=status,
            order_number=identifier,
            tool_name="process_approved_order",
            output_json=result,
        )
        return result

    order_id = int(order_ref["id"])
    order = get_order_with_extraction(order_id) or order_ref
    order_number = _primary_order_number(order)
    existing = _existing_batch(order_id)
    if existing:
        result = {
            "status": "already_processed",
            "order": _queue_card(order),
            "batch": existing,
            "files": {
                "processing_pdf_url": existing.get("processing_pdf_url"),
                "labels_pdf_url": existing.get("labels_pdf_url"),
            },
            "summary": existing.get("summary") or {},
            "needs_confirmation": True,
            "confirmation_payload": {"action": "reprocess_order", "order_id": order_id},
        }
        record_workspace_action(
            actor=requested_by,
            action_type="process_order_blocked",
            status="already_processed",
            order_id=order_id,
            order_number=order_number,
            processing_batch_id=existing.get("id"),
            tool_name="process_approved_order",
            output_json=result,
            requires_confirmation=True,
        )
        return result

    normalized_status = normalize_order_status(order.get("status"))
    if normalized_status != "approved":
        result = {
            "status": "blocked",
            "reason": "order_not_approved",
            "order": _queue_card(order),
            "message": f"Order {order_number} is {normalized_status}; only approved orders can be processed.",
        }
        record_workspace_action(
            actor=requested_by,
            action_type="process_order_blocked",
            status="blocked",
            order_id=order_id,
            order_number=order_number,
            tool_name="process_approved_order",
            output_json=result,
        )
        return result

    validation = _validate_order(order)
    if validation["blocker_warnings"]:
        result = {
            "status": "needs_review",
            "reason": "deterministic_preflight_blocked",
            "order": _queue_card(order),
            "summary": validation["summary"],
            "message": "Production preflight found required data that must be corrected before files can be created.",
            "warnings": _warning_items(order_number, validation["blocker_warnings"]),
            "actions": [{"type": "open_review", "order_ids": [str(order_id)], "order_numbers": [order_number]}],
        }
        record_workspace_action(
            actor=requested_by,
            action_type="process_order_blocked",
            status="needs_review",
            order_id=order_id,
            order_number=order_number,
            tool_name="process_approved_order",
            output_json=result,
        )
        return result
    result = {
        "status": "frontend_workflow_required",
        "order": _queue_card(order),
        "summary": validation["summary"],
        "message": "Processing PDFs and labels must be generated through the existing Processing and Labels modules.",
        "actions": [{"type": "process_via_existing_modules", "identifier": str(order_id)}],
        "informational_warnings": _warning_items(order_number, validation["advisory_warnings"]),
    }
    record_workspace_action(
        actor=requested_by,
        action_type="process_order_delegated",
        status="frontend_workflow_required",
        order_id=order_id,
        order_number=order_number,
        tool_name="process_approved_order",
        output_json=result,
    )
    return result

    rows = validation["rows"]
    summary = validation["summary"]
    with SessionLocal() as session:
        batch = ProcessingBatch(
            order_id=order_id,
            order_number=order_number,
            status="creating_files",
            requested_by=requested_by,
            forced=bool(force),
            summary_json=json.dumps(summary, ensure_ascii=False),
        )
        session.add(batch)
        session.flush()
        batch_id = batch.id

        base = _production_dir() / f"{_safe_filename(order_number)}-batch-{batch_id}-{_now_stamp()}"
        processing_path = base.with_name(base.name + "-processing.pdf")
        labels_path = base.with_name(base.name + "-labels.pdf")

        try:
            _generate_processing_pdf(order, rows, batch_id, processing_path)
            processing_file = _insert_file_record(
                session,
                order_id=order_id,
                batch_id=batch_id,
                order_number=order_number,
                file_type="processing_pdf",
                path=processing_path,
            )

            _generate_labels_pdf(order, rows, batch_id, labels_path)
            labels_file = _insert_file_record(
                session,
                order_id=order_id,
                batch_id=batch_id,
                order_number=order_number,
                file_type="labels_pdf",
                path=labels_path,
            )
            batch.status = "files_ready"
            batch.updated_at = datetime.now(timezone.utc)
            session.flush()
            serialized = _serialize_batch(session, batch)
        except Exception as exc:
            batch.status = "failed"
            batch.updated_at = datetime.now(timezone.utc)
            session.flush()
            result = {
                "status": "file_generation_failed",
                "error_message": str(exc),
                "batch": _serialize_batch(session, batch),
            }
            session.commit()
            record_workspace_action(
                actor=requested_by,
                action_type="process_order_failed",
                status="file_generation_failed",
                order_id=order_id,
                order_number=order_number,
                processing_batch_id=batch_id,
                output_json=result,
                error_message=str(exc),
            )
            return result

        session.commit()

    record_workspace_action(
        actor=requested_by,
        action_type="processing_batch_created",
        status="success",
        order_id=order_id,
        order_number=order_number,
        processing_batch_id=serialized.get("id"),
        output_json={"batch_id": serialized.get("id")},
        confirmed=bool(force),
    )
    if (serialized.get("files") or {}).get("processing_pdf"):
        record_workspace_action(
            actor=requested_by,
            action_type="processing_pdf_generated",
            status="success",
            order_id=order_id,
            order_number=order_number,
            processing_batch_id=serialized.get("id"),
            output_json=serialized["files"]["processing_pdf"],
        )
    if (serialized.get("files") or {}).get("labels_pdf"):
        record_workspace_action(
            actor=requested_by,
            action_type="labels_pdf_generated",
            status="success",
            order_id=order_id,
            order_number=order_number,
            processing_batch_id=serialized.get("id"),
            output_json=serialized["files"]["labels_pdf"],
        )

    try:
        update_order_status(order_id, status="in_production", note="Processed from Workspace", reason="workspace_process_order")
    except Exception:
        pass

    result = {
        "status": "success",
        "order": _queue_card(order),
        "batch": serialized,
        "summary": summary,
        "files": {
            "processing_pdf_url": serialized.get("processing_pdf_url"),
            "labels_pdf_url": serialized.get("labels_pdf_url"),
        },
    }
    record_workspace_action(
        actor=requested_by,
        action_type="process_order_completed",
        status="success",
        order_id=order_id,
        order_number=order_number,
        processing_batch_id=serialized.get("id"),
        tool_name="process_approved_order",
        output_json=result,
        confirmed=bool(force),
    )
    return result


def _file_urls_for_order(order_id: int) -> Dict[str, Any]:
    existing = _existing_batch(order_id)
    if not existing:
        return {}
    return {
        "batch_id": existing.get("id"),
        "batch_status": existing.get("status"),
        "processing_pdf_url": existing.get("processing_pdf_url"),
        "labels_pdf_url": existing.get("labels_pdf_url"),
    }


def _parse_workspace_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _age_days(value: Any) -> int:
    parsed = _parse_workspace_datetime(value)
    if not parsed:
        return 0
    return max(0, int((datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() // 86400))


def _glass_profile(rows: Sequence[Dict[str, Any]]) -> List[str]:
    seen = set()
    profile: List[str] = []
    for row in rows or []:
        glass_type = re.sub(r"\s+", " ", str(row.get("type") or "").strip()).upper()
        if glass_type and glass_type not in seen:
            seen.add(glass_type)
            profile.append(glass_type)
    return sorted(profile)


def _queue_attention(card: Dict[str, Any]) -> Dict[str, Any]:
    status = normalize_order_status(card.get("status"))
    blockers = int(card.get("blocker_count") or 0)
    advisories = int(card.get("advisory_count") or 0)
    age = int(card.get("age_days") or 0)
    if status in {"completed", "archived"}:
        return {
            "severity": "complete",
            "reason": "Workflow completed",
            "recommended_action": "Open the audit trail or production files if needed.",
            "production_eligibility": "complete",
            "priority_score": 0,
        }
    if blockers:
        return {
            "severity": "blocker",
            "reason": f"{blockers} required-data blocker{'s' if blockers != 1 else ''}",
            "recommended_action": "Open Preflight and correct the source data before production.",
            "production_eligibility": "blocked",
            "priority_score": 1000 + (blockers * 20) + min(age, 30),
        }
    if status in {"draft", "reviewed"}:
        reason = f"{advisories} advisory warning{'s' if advisories != 1 else ''}" if advisories else "Awaiting operator review"
        return {
            "severity": "advisory" if advisories else "review",
            "reason": reason,
            "recommended_action": "Review the order and approve it before production.",
            "production_eligibility": "not_approved",
            "priority_score": 700 + min(age, 60),
        }
    if status == "approved":
        if advisories:
            return {
                "severity": "advisory",
                "reason": f"Preflight passed with {advisories} advisory warning{'s' if advisories != 1 else ''}",
                "recommended_action": "Review the advisory, then include this order in a production plan.",
                "production_eligibility": "eligible",
                "priority_score": 500 + min(age, 60) + min(advisories, 10),
            }
        return {
            "severity": "ready",
            "reason": "Approved and deterministic preflight passed",
            "recommended_action": "Include this order in a compatible production plan.",
            "production_eligibility": "eligible",
            "priority_score": 400 + min(age, 60),
        }
    if status == "in_production":
        return {
            "severity": "active",
            "reason": "Production files have been prepared",
            "recommended_action": "Complete production checks and labels.",
            "production_eligibility": "in_production",
            "priority_score": 300 + min(age, 30),
        }
    return {
        "severity": "review",
        "reason": "Status requires operator review",
        "recommended_action": "Open the order and confirm its next workflow stage.",
        "production_eligibility": "not_ready",
        "priority_score": 100 + min(age, 30),
    }


def _queue_card(order: Dict[str, Any]) -> Dict[str, Any]:
    rows = order.get("rows") or []
    summary = _summarize_rows(rows)
    blockers: List[str] = []
    advisories: List[str] = []
    if rows:
        validation = _validate_order(order)
        blockers = validation.get("blocker_warnings") or []
        advisories = validation.get("advisory_warnings") or []
    files = _file_urls_for_order(int(order["id"])) if order.get("id") is not None else {}
    approved_at = _approved_at(order)
    card = {
        "id": order.get("id"),
        "order_id": order.get("id"),
        "order_number": _primary_order_number(order),
        "order_numbers": order.get("order_numbers") or [],
        "client_name": _client_name(order),
        "status": normalize_order_status(order.get("status")),
        "status_label": order.get("status_label"),
        "total_pieces": summary["total_pieces"] or order.get("units_total") or 0,
        "total_area_m2": summary["total_area_m2"] if rows else round(float(order.get("area_total") or 0.0), 3),
        "warnings_count": len(blockers) + len(advisories),
        "blocker_count": len(blockers),
        "advisory_count": len(advisories),
        "warning_preview": (blockers + advisories)[:3],
        "glass_profile": _glass_profile(rows),
        "created_at": order.get("created_at"),
        "updated_at": order.get("updated_at"),
        "approved_at": approved_at,
        "age_days": _age_days(approved_at or order.get("created_at")),
        **files,
    }
    card.update(_queue_attention(card))
    return card


def _approved_at(order: Dict[str, Any]) -> Optional[str]:
    for event in reversed(order.get("status_history") or []):
        if event.get("to_status") == "approved":
            return event.get("changed_at")
    return None


def get_workspace_queue() -> Dict[str, Any]:
    items = get_orders(year="all", limit=200, offset=0)
    grouped = {
        "needs_review": [],
        "approved_ready": [],
        "processing_done": [],
        "labels_ready": [],
        "finished": [],
    }
    for item in items:
        detail = get_order_with_extraction(int(item["id"])) or item
        card = _queue_card(detail)
        status = normalize_order_status(card.get("status"))
        if status in {"draft", "reviewed"}:
            grouped["needs_review"].append(card)
            continue
        if status == "approved":
            if card.get("batch_id"):
                grouped["labels_ready" if card.get("labels_pdf_url") else "processing_done"].append(card)
            else:
                grouped["approved_ready"].append(card)
            continue
        if status == "in_production":
            grouped["labels_ready" if card.get("labels_pdf_url") else "processing_done"].append(card)
            continue
        if status in {"completed", "archived"}:
            grouped["finished"].append(card)
    for key in grouped:
        grouped[key].sort(
            key=lambda item: (
                int(item.get("priority_score") or 0),
                str(item.get("approved_at") or item.get("created_at") or ""),
            ),
            reverse=True,
        )

    ready_cards = [
        item for item in grouped["approved_ready"]
        if item.get("production_eligibility") == "eligible"
    ]
    compatibility_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in ready_cards:
        profile = item.get("glass_profile") or []
        signature = " | ".join(profile) if profile else "UNSPECIFIED"
        compatibility_groups.setdefault(signature, []).append(item)
    compatible = max(
        compatibility_groups.values(),
        key=lambda values: (len(values), sum(float(item.get("total_area_m2") or 0.0) for item in values)),
        default=[],
    )[:8]
    recommended_batch = None
    if compatible:
        recommended_batch = {
            "order_ids": [str(item.get("order_id")) for item in compatible],
            "order_numbers": [item.get("order_number") for item in compatible],
            "order_count": len(compatible),
            "total_pieces": sum(int(item.get("total_pieces") or 0) for item in compatible),
            "total_area_m2": round(sum(float(item.get("total_area_m2") or 0.0) for item in compatible), 3),
            "reason": (
                "Exact glass profile match across the selected orders."
                if len(compatible) > 1 else
                "Oldest eligible order ready for a safe production plan."
            ),
        }

    all_cards = [item for values in grouped.values() for item in values]
    attention = {
        "blockers": sum(1 for item in all_cards if item.get("severity") == "blocker"),
        "ready_now": len(ready_cards),
        "advisories": sum(1 for item in all_cards if item.get("severity") == "advisory"),
        "ageing": sum(1 for item in grouped["approved_ready"] if int(item.get("age_days") or 0) >= 7),
        "production_files": sum(1 for item in all_cards if item.get("processing_pdf_url") or item.get("labels_pdf_url")),
        "recommended_batch": recommended_batch,
    }
    return {
        "groups": grouped,
        "counts": {key: len(value) for key, value in grouped.items()},
        "attention": attention,
    }


def _workspace_spacer_width_mm(type_string: Any) -> Optional[float]:
    raw = re.sub(r"^\s*\d+\s*vetri?\s*", "", str(type_string or ""), flags=re.IGNORECASE)
    raw = re.sub(r"\b\d+(?:[.,]\d+)?\s*mm\b", "", raw, flags=re.IGNORECASE)
    for token in raw.split("+"):
        candidate = re.sub(r"\s+", "", token).replace(",", ".")
        if not re.fullmatch(r"\d+(?:\.\d+)?", candidate):
            continue
        try:
            value = float(candidate)
        except ValueError:
            continue
        if 1 <= value <= 40:
            return value
    return None


def build_workspace_batch_plan(
    identifiers: Sequence[Any],
    *,
    keep_order_boundaries: bool = True,
    requested_by: str = "workspace",
) -> Dict[str, Any]:
    requested = [_normalize_workspace_order_token(item) for item in identifiers if str(item or "").strip()]
    if not requested:
        return {"status": "missing_order", "message": "Select at least one order to build a production plan."}
    if len(requested) > 25:
        return {"status": "too_many_orders", "message": "A production plan can contain at most 25 orders."}

    orders: List[Dict[str, Any]] = []
    missing: List[str] = []
    ambiguous: List[str] = []
    seen_ids = set()
    for identifier in requested:
        order, matches = _resolve_order(identifier)
        if not order:
            (ambiguous if matches else missing).append(identifier)
            continue
        order_id = str(order.get("id") or "")
        if order_id in seen_ids:
            continue
        seen_ids.add(order_id)
        orders.append(get_order_with_extraction(int(order_id)) or order)
    if missing or ambiguous:
        return {
            "status": "not_found" if missing else "multiple_matches",
            "message": "Some selected orders could not be resolved. No production data was changed.",
            "missing": missing,
            "ambiguous": ambiguous,
        }

    order_cards: List[Dict[str, Any]] = []
    blockers: List[Dict[str, Any]] = []
    advisories: List[Dict[str, Any]] = []
    total_pieces = 0
    total_area = 0.0
    glass_types = set()
    profiles = set()
    spacer_total_mm = 0.0
    butyl_total_kg = 0.0
    spacer_by_width: Dict[str, float] = {}
    missing_dimensions = 0
    missing_spacer = 0
    not_approved: List[str] = []
    density_g_cm3 = 1.18
    bead_height_mm = 5.0

    for order in orders:
        card = _queue_card(order)
        order_cards.append(card)
        order_number = card.get("order_number") or str(card.get("order_id") or "Order")
        if normalize_order_status(order.get("status")) != "approved":
            not_approved.append(str(order_number))
        validation = _validate_order(order)
        blockers.extend(_warning_items(str(order_number), validation.get("blocker_warnings") or []))
        advisories.extend(_warning_items(str(order_number), validation.get("advisory_warnings") or []))
        summary = validation.get("summary") or {}
        total_pieces += int(summary.get("total_pieces") or 0)
        total_area += float(summary.get("total_area_m2") or 0.0)
        profile = _glass_profile(order.get("rows") or [])
        profiles.add(" | ".join(profile) if profile else "UNSPECIFIED")
        glass_types.update(profile)

        for row in validation.get("rows") or []:
            _cleaned, dims = clean_dimension(row.get("dimension") or "")
            try:
                quantity = max(0, int(row.get("quantity") or 0))
            except (TypeError, ValueError):
                quantity = 0
            if not dims:
                missing_dimensions += 1
                continue
            width, height = dims
            perimeter_mm = 2.0 * (float(width) + float(height)) * quantity
            spacer_total_mm += perimeter_mm
            spacer_width = _workspace_spacer_width_mm(row.get("type"))
            if spacer_width is None:
                missing_spacer += 1
                continue
            spacer_key = f"{spacer_width:g} mm"
            spacer_by_width[spacer_key] = spacer_by_width.get(spacer_key, 0.0) + (perimeter_mm / 1000.0)
            volume_cm3 = (perimeter_mm * spacer_width * bead_height_mm) / 1000.0
            butyl_total_kg += (volume_cm3 * density_g_cm3) / 1000.0

    if not_approved:
        for order_number in not_approved:
            blockers.append({
                "order_number": order_number,
                "row": None,
                "code": "order_not_approved",
                "message": "Order is not approved for production.",
            })

    status = "blocked" if blockers else ("ready_with_advisories" if advisories else "ready")
    result = {
        "status": status,
        "plan_id": f"workspace-plan-{uuid.uuid4().hex[:12]}",
        "created_at": utc_isoformat(datetime.now(timezone.utc)),
        "message": (
            "Production plan is blocked until required data is corrected."
            if blockers else
            "Production plan is ready for operator confirmation."
        ),
        "can_confirm": not blockers,
        "confirmation_required": True,
        "keep_order_boundaries": bool(keep_order_boundaries),
        "orders": [{
            "order_id": str(card.get("order_id") or ""),
            "order_number": card.get("order_number"),
            "client_name": card.get("client_name"),
            "pieces": card.get("total_pieces"),
            "area_m2": card.get("total_area_m2"),
            "updated_at": card.get("updated_at"),
        } for card in order_cards],
        "summary": {
            "order_count": len(order_cards),
            "total_pieces": total_pieces,
            "total_area_m2": round(total_area, 3),
            "glass_type_count": len(glass_types),
            "glass_types": sorted(glass_types),
            "profile_count": len(profiles),
            "compatibility": "exact_profile" if len(profiles) == 1 else "mixed_profiles",
        },
        "materials": {
            "spacer_m": round(spacer_total_mm / 1000.0, 2),
            "spacer_by_width_m": {key: round(value, 2) for key, value in sorted(spacer_by_width.items())},
            "butyl_kg": round(butyl_total_kg, 3),
            "bostik_4000_boxes": int((butyl_total_kg + 6.5 - 1e-9) // 6.5) if butyl_total_kg > 0 else 0,
            "missing_dimension_items": missing_dimensions,
            "missing_spacer_items": missing_spacer,
            "calculation_note": "Deterministic estimate using 5 mm bead height and 1.18 g/cm3 density; stock availability is not connected.",
        },
        "blockers": blockers,
        "advisories": advisories,
        "safety": {
            "mutated_production_data": False,
            "revalidate_on_confirm": True,
            "raw_and_approved_data_unchanged": True,
        },
    }
    record_workspace_action(
        actor=requested_by,
        action_type="workspace_batch_plan_created",
        status=status,
        input_json={"identifiers": requested, "keep_order_boundaries": bool(keep_order_boundaries)},
        output_json={
            "plan_id": result["plan_id"],
            "order_count": len(order_cards),
            "can_confirm": result["can_confirm"],
            "blocker_count": len(blockers),
            "advisory_count": len(advisories),
        },
    )
    return result


def get_recent_production_files(limit: int = 20) -> Dict[str, Any]:
    limit = max(1, min(int(limit or 20), 100))
    with SessionLocal() as session:
        batches = session.execute(
            select(ProcessingBatch)
            .order_by(ProcessingBatch.updated_at.desc(), ProcessingBatch.created_at.desc())
            .limit(limit)
        ).scalars().all()
        items = []
        for batch in batches:
            serialized = _serialize_batch(session, batch)
            order = session.get(Order, batch.order_id)
            items.append(
                {
                    "batch_id": batch.id,
                    "order_id": batch.order_id,
                    "order_number": batch.order_number,
                    "client_name": order.client_name or order.client_hint if order else "",
                    "generated_at": utc_isoformat(batch.updated_at),
                    "batch_status": batch.status,
                    "processing_pdf_url": serialized.get("processing_pdf_url"),
                    "labels_pdf_url": serialized.get("labels_pdf_url"),
                }
            )
        return {"items": items}


def get_production_file(file_id: int) -> Optional[Dict[str, Any]]:
    with SessionLocal() as session:
        file = session.get(ProductionFile, int(file_id))
        if not file:
            return None
        return _serialize_file(file) | {"file_path": file.file_path}


def list_order_production_files(order_id: int) -> List[Dict[str, Any]]:
    """Read the existing persistent file catalogue for a single extracted order."""
    with SessionLocal() as session:
        files = session.scalars(select(ProductionFile).where(
            ProductionFile.order_id == order_id,
            ProductionFile.status == "ready",
        ).order_by(ProductionFile.created_at)).all()
        return [_serialize_file(file) | {"file_path": file.file_path} for file in files]


def get_processing_batch_files(batch_id: int) -> Dict[str, Any]:
    with SessionLocal() as session:
        batch = session.get(ProcessingBatch, int(batch_id))
        if not batch:
            return {"status": "not_found"}
        return {"status": "ok", "batch": _serialize_batch(session, batch)}
