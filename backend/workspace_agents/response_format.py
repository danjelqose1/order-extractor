from __future__ import annotations

from typing import Any, Dict, List, Optional


def _action_to_ui_action(action: Dict[str, Any]) -> Dict[str, Any]:
    action_type = str(action.get("type") or "")
    if action_type == "process_via_existing_modules":
        return {
            "label": "Process in Workspace",
            "kind": "confirm",
            "url": None,
            "payload": action,
        }
    if action_type == "show_last_agent_files":
        return {
            "label": "Show Files",
            "kind": "open",
            "url": None,
            "payload": action,
        }
    if action_type in {"open_last_processing_sheet", "open_last_label_job", "open_review"}:
        return {
            "label": "Open",
            "kind": "open",
            "url": None,
            "payload": action,
        }
    return {
        "label": action.get("label") or action_type or "Action",
        "kind": action.get("kind") or "open",
        "url": action.get("url"),
        "payload": action,
    }


def _file_actions(files: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    processing_url = files.get("processing_pdf_url")
    labels_url = files.get("labels_pdf_url")
    if processing_url:
        actions.append({
            "label": "Download Processing PDF",
            "kind": "download",
            "url": processing_url,
            "payload": None,
        })
    if labels_url:
        actions.append({
            "label": "Download Labels PDF",
            "kind": "download",
            "url": labels_url,
            "payload": None,
        })
    return actions


def cards_for_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    existing = response.get("cards")
    if isinstance(existing, list):
        return existing

    status = str(response.get("status") or "")
    card_type = "success" if status in {"ok", "success", "context_action", "frontend_workflow_required"} else "warning"
    if status in {"blocked", "not_found", "multiple_matches", "error", "file_generation_failed"}:
        card_type = "error"
    if status in {"needs_choice", "needs_review", "missing_order", "missing_pending_action"}:
        card_type = "warning"

    title = "Workspace Assistant"
    if status == "frontend_workflow_required":
        title = "Ready to Process"
    elif status in {"blocked", "not_found", "multiple_matches"}:
        title = "Stopped"
    elif response.get("files"):
        title = "Production Files"

    actions = [_action_to_ui_action(action) for action in response.get("actions") or [] if isinstance(action, dict)]
    files = response.get("files") if isinstance(response.get("files"), dict) else {}
    actions.extend(_file_actions(files))

    return [{
        "type": card_type,
        "title": title,
        "body": str(response.get("message") or ""),
        "actions": actions,
    }]


def derive_last_agent_action(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    existing = response.get("last_agent_action")
    if isinstance(existing, dict):
        return existing

    pending = response.get("pending_action") if isinstance(response.get("pending_action"), dict) else None
    if pending:
        return {
            "type": pending.get("type") or "process_orders",
            "status": "blocked",
            "order_numbers": pending.get("order_numbers") or [],
            "mode": pending.get("mode"),
            "mergeAcrossOrders": bool(pending.get("mergeAcrossOrders")) if pending.get("mode") == "combined" else False,
            "processing_sheet_id": None,
            "label_job_id": None,
            "files": None,
        }

    files = response.get("files") if isinstance(response.get("files"), dict) else None
    actions = response.get("actions") if isinstance(response.get("actions"), list) else []
    process_action = next((action for action in actions if isinstance(action, dict) and action.get("type") == "process_via_existing_modules"), None)
    if process_action:
        identifiers = process_action.get("order_numbers") or process_action.get("identifiers") or []
        mode = process_action.get("mode")
        return {
            "type": "process_orders" if len(identifiers) > 1 else "process_order",
            "status": response.get("status") or "planned",
            "order_numbers": [str(item) for item in identifiers],
            "mode": mode if mode != "single" else None,
            "mergeAcrossOrders": bool(process_action.get("mergeAcrossOrders")) if mode == "combined" else False,
            "processing_sheet_id": None,
            "label_job_id": None,
            "files": files,
        }
    return None


def ensure_workspace_agent_response(response: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(response, dict):
        response = {"message": "I could not complete that workspace action.", "status": "error"}
    response.setdefault("message", "")
    response.setdefault("status", "ok")
    response["cards"] = cards_for_response(response)
    response["last_agent_action"] = derive_last_agent_action(response)
    response.setdefault("refresh", {
        "queue": response.get("status") in {"frontend_workflow_required", "success"},
        "recent_files": bool(response.get("files") or response.get("recent_files")),
    })
    return response
