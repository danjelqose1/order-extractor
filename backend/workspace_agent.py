from __future__ import annotations

print("✅ workspace_agent loaded")

import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx

from db import record_workspace_action
from workspace_service import (
    confirm_workspace_pending_action,
    get_processing_batch_files,
    get_recent_production_files,
    get_workspace_queue,
    prepare_workspace_processing_action,
    process_approved_order,
    validate_order_for_processing,
)
from workspace_agents.intent import legacy_conversational_response
from workspace_agents.response_format import ensure_workspace_agent_response


AGENT_SYSTEM_PROMPT = """You are the Factory Assistant for a glass order extraction and production platform.
You help operators process approved glass orders, understand warnings, and prepare production files.
You must use tools to inspect real platform data.
You must not guess order data.
You must not invent file links.
You must not mutate raw data.
You must not process draft orders.
You must not overwrite approved/history data.
You must not overwrite existing processing batches unless explicit confirmation and backend support exists.
You must prefer deterministic validation results over AI reasoning.
Dimensions are always in millimeters.
Area validation must be quantity-aware.
If there are critical warnings, stop and recommend review.
If the order is approved and clean, you may process it by calling process_approved_order.
Keep responses short, practical, and factory-friendly."""

ORDER_RE = re.compile(r"\b(?:R-?)?\d{2}-\d{3,5}\b", re.IGNORECASE)


def normalize_order_token(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "")
    if not text:
        return ""
    if re.fullmatch(r"R-?\d{2}-\d{3,5}", text, re.IGNORECASE):
        return "R-" + re.sub(r"^R-?", "", text, flags=re.IGNORECASE)
    if re.fullmatch(r"\d{2}-\d{3,5}", text):
        return f"R-{text}"
    return text


def extract_order_tokens(message: str) -> List[str]:
    seen = set()
    tokens: List[str] = []
    for match in ORDER_RE.findall(message or ""):
        normalized = normalize_order_token(match)
        if normalized and normalized not in seen:
            seen.add(normalized)
            tokens.append(normalized)
    return tokens


def infer_processing_mode(message: str, token_count: int = 0) -> str:
    lower = (message or "").lower()
    keep_order_boundaries = any(
        phrase in lower
        for phrase in (
            "keep orders separate",
            "keep them separate",
            "don't mix orders",
            "do not mix orders",
            "without mixing orders",
            "no merge across orders",
        )
    )
    if keep_order_boundaries and any(phrase in lower for phrase in ("together", "same time", "combined", "one sheet")):
        return "combined"
    if any(phrase in lower for phrase in ("separately", "separate jobs", "separate processing", "one by one")):
        return "separate"
    if any(phrase in lower for phrase in ("together", "same time", "combined", "one sheet")):
        return "combined"
    return "combined" if token_count > 1 else "single"


def infer_merge_across_orders(message: str) -> bool:
    lower = (message or "").lower()
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


def _context_order_numbers(context: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(context, dict):
        return []
    values = context.get("selected_order_numbers")
    if not isinstance(values, list):
        selected_orders = context.get("selected_orders")
        if isinstance(selected_orders, list):
            values = [item.get("order_number") for item in selected_orders if isinstance(item, dict)]
    if not isinstance(values, list):
        value = context.get("selected_order_number")
        values = [value] if value else []
    output: List[str] = []
    seen = set()
    for value in values:
        normalized = normalize_order_token(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def _context_order_ids(context: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(context, dict):
        return []
    values = context.get("selected_order_ids")
    if not isinstance(values, list):
        selected_orders = context.get("selected_orders")
        if isinstance(selected_orders, list):
            values = [item.get("order_id") for item in selected_orders if isinstance(item, dict)]
    if not isinstance(values, list):
        value = context.get("selected_order_id")
        values = [value] if value else []
    output: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _visible_orders(context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(context, dict):
        return []
    orders = context.get("visible_orders")
    return [item for item in orders if isinstance(item, dict)] if isinstance(orders, list) else []


def _last_agent_action(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    action = context.get("last_agent_action")
    return action if isinstance(action, dict) else {}


def _visible_approved_orders(context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    approved = []
    for item in _visible_orders(context):
        group = str(item.get("queue_group") or "").lower()
        status = str(item.get("status") or "").lower()
        if group == "approved_ready" or status == "approved":
            approved.append(item)
    return approved


def _ordinal_index(message: str) -> Optional[int]:
    lower = (message or "").lower()
    words = {
        "first": 0,
        "1st": 0,
        "second": 1,
        "2nd": 1,
        "third": 2,
        "3rd": 2,
        "fourth": 3,
        "4th": 3,
        "fifth": 4,
        "5th": 4,
    }
    for key, value in words.items():
        if re.search(rf"\b{re.escape(key)}\b", lower):
            return value
    return None


def _action_for_identifiers(identifiers: List[str], mode: str, *, merge_across_orders: bool = False) -> Dict[str, Any]:
    cleaned = [str(item) for item in identifiers if str(item or "").strip()]
    if not cleaned:
        return {"message": "Tell me which order to process.", "status": "missing_order"}
    if len(cleaned) > 1:
        normalized_mode = "separate" if mode == "separate" else "combined"
        return {
            "message": "Use the Processing module workflow for these orders.",
            "status": "frontend_workflow_required",
            "actions": [{
                "type": "process_via_existing_modules",
                "identifiers": cleaned,
                "order_numbers": cleaned,
                "mode": normalized_mode,
                "mergeAcrossOrders": bool(merge_across_orders) if normalized_mode == "combined" else False,
            }],
        }
    return {
        "message": "Use the Processing module workflow for this order.",
        "status": "frontend_workflow_required",
        "actions": [{
            "type": "process_via_existing_modules",
            "identifier": cleaned[0],
            "identifiers": [cleaned[0]],
            "mode": "single",
        }],
    }


def _make_action_plan(
    *,
    intent: str,
    orders_to_resolve: Optional[List[str]] = None,
    mode: Optional[str] = None,
    merge_across_orders: Optional[bool] = None,
    requires_confirmation: bool = False,
    planned_tool: str = "none",
    reason: str = "",
) -> Dict[str, Any]:
    plan = {
        "intent": intent,
        "orders_to_resolve": orders_to_resolve or [],
        "resolved_order_ids": [],
        "mode": mode,
        "requires_confirmation": bool(requires_confirmation),
        "planned_tool": planned_tool,
        "reason": reason,
    }
    if merge_across_orders is not None:
        plan["mergeAcrossOrders"] = bool(merge_across_orders)
    return plan


def _attach_action_plan(response: Dict[str, Any], plan: Dict[str, Any], requested_by: str, message: str) -> Dict[str, Any]:
    if not isinstance(response, dict):
        response = {"message": "I could not complete that action.", "status": "error"}
    resolved: List[str] = []
    actions = response.get("actions") if isinstance(response.get("actions"), list) else []
    for action in actions:
        if isinstance(action, dict) and action.get("type") == "process_via_existing_modules":
            resolved = [str(item) for item in (action.get("identifiers") or []) if str(item or "").strip()]
            break
    pending = response.get("pending_action") if isinstance(response.get("pending_action"), dict) else None
    if pending and not resolved:
        resolved = [str(item) for item in (pending.get("order_ids") or []) if str(item or "").strip()]
    enriched_plan = {**plan, "resolved_order_ids": resolved}
    response["action_plan"] = enriched_plan
    record_workspace_action(
        actor=requested_by,
        action_type="agent_action_plan",
        status=response.get("status") or "planned",
        requested_message=message,
        input_json=enriched_plan,
        output_json={"response_status": response.get("status")},
    )
    return response


def _resolve_context_action(message: str, context: Optional[Dict[str, Any]], requested_by: str = "workspace_agent") -> Optional[Dict[str, Any]]:
    if not isinstance(context, dict):
        return None
    lower = (message or "").lower()
    selected_ids = _context_order_ids(context)
    selected_numbers = _context_order_numbers(context)
    selected_refs = selected_ids or selected_numbers
    last_action = _last_agent_action(context)
    last_files = last_action.get("files") if isinstance(last_action.get("files"), dict) else {}
    if any(phrase in lower for phrase in ("why did it stop", "why stopped", "why did you stop")):
        pending = context.get("latest_pending_action") if isinstance(context.get("latest_pending_action"), dict) else None
        warnings = pending.get("warnings") if isinstance(pending, dict) else []
        if pending:
            warning_text = "\n".join(
                f"- {item.get('order_number', 'Order')}: {item.get('message', item)}"
                for item in (warnings or [])[:8]
                if isinstance(item, dict)
            )
            message_out = "It stopped because warnings need confirmation before production files are created."
            if warning_text:
                message_out += f"\n{warning_text}"
            return {
                "message": message_out,
                "status": "needs_review",
                "pending_action": pending,
                "warnings": warnings or [],
            }
        if last_action and str(last_action.get("status") or "").lower() in {"blocked", "error", "failed"}:
            order_numbers = ", ".join(str(item) for item in (last_action.get("order_numbers") or []) if str(item or "").strip())
            reason = last_action.get("reason") or last_action.get("message") or f"the last action ended with status {last_action.get('status')}"
            target = f" for {order_numbers}" if order_numbers else ""
            return {"message": f"It stopped{target} because {reason}.", "status": "ok", "last_agent_action": last_action}
        return {"message": "I do not have a blocked action in this Workspace session.", "status": "ok"}
    if any(phrase in lower for phrase in ("cancel", "never mind", "stop that")) and _latest_pending_action_id(context):
        return confirm_workspace_pending_action(
            _latest_pending_action_id(context) or "",
            decision="cancel",
            requested_by=requested_by,
            context=context,
        )
    if ("download" in lower or "labels" in lower) and "again" in lower:
        return {
            "message": "Showing the last labels from this Workspace session.",
            "status": "context_action",
            "files": last_files if last_files else None,
            "actions": [{"type": "show_last_agent_files"}],
        }
    if "download" in lower and "processing" in lower:
        return {
            "message": "Showing the last Processing PDF from this Workspace session.",
            "status": "context_action",
            "files": last_files if last_files else None,
            "actions": [{"type": "show_last_agent_files"}],
        }
    if "open" in lower and "processing" in lower and any(word in lower for word in ("that", "sheet", "last")):
        return {
            "message": "Opening the last Processing sheet from this Workspace session.",
            "status": "context_action",
            "actions": [{"type": "open_last_processing_sheet"}],
        }
    if "open" in lower and ("label" in lower or "labels" in lower):
        return {
            "message": "Opening the last label job from this Workspace session.",
            "status": "context_action",
            "actions": [{"type": "open_last_label_job"}],
        }
    if "latest approved" in lower and "process" not in lower:
        queue = get_workspace_queue()
        approved = ((queue.get("groups") or {}).get("approved_ready") or [])
        if not approved:
            return {"message": "No approved orders are waiting for processing.", "status": "not_found"}
        latest = approved[0]
        order_number = latest.get("order_number") or latest.get("order_id") or "latest approved order"
        return {
            "message": f"Latest approved order is {order_number}.",
            "status": "ok",
            "actions": [{"type": "show_queue", "group": "approved_ready"}],
        }
    if (
        ("process" in lower and any(phrase in lower for phrase in ("these", "these two", "selected orders", "selected together", "process selected")))
        or any(phrase in lower for phrase in ("do these two together", "do these together"))
    ):
        if len(selected_refs) >= 2:
            mode = infer_processing_mode(message, len(selected_refs))
            if mode == "single":
                mode = "combined"
            return prepare_workspace_processing_action(selected_refs, mode=mode, requested_by=requested_by, original_message=message, context=context)
        return {
            "message": "I don’t know which orders you mean. Select them or tell me the order numbers.",
            "status": "needs_choice",
            "actions": [{"type": "ask_order_choice", "reason": "not_enough_selected_orders"}],
        }
    if "process" in lower and any(phrase in lower for phrase in ("this one too", "this too", "process this", "same for")):
        if len(selected_refs) == 1:
            return prepare_workspace_processing_action([selected_refs[-1]], mode="single", requested_by=requested_by, original_message=message, context=context)
        if len(selected_refs) > 1:
            return {
                "message": "I don’t know which order you mean. Select one or say “process these together.”",
                "status": "needs_choice",
                "actions": [{"type": "ask_order_choice", "reason": "ambiguous_selected_orders"}],
            }
        return {
            "message": "I don’t know which order you mean. Select one or tell me the order number.",
            "status": "needs_choice",
            "actions": [{"type": "ask_order_choice", "reason": "no_selected_order"}],
        }
    if "same as before" in lower:
        previous_mode = last_action.get("mode") or "combined"
        previous_merge = last_action.get("mergeAcrossOrders")
        refs = selected_refs or [normalize_order_token(item) for item in (last_action.get("order_numbers") or []) if normalize_order_token(item)]
        if refs:
            return prepare_workspace_processing_action(
                refs,
                mode=previous_mode,
                merge_across_orders=previous_merge if isinstance(previous_merge, bool) else None,
                requested_by=requested_by,
                original_message=message,
                context=context,
            )
        return {"message": "I do not have a previous order action to repeat.", "status": "needs_choice"}
    ordinal = _ordinal_index(message)
    if "process" in lower and ordinal is not None:
        candidates = _visible_approved_orders(context) if "approved" in lower else _visible_orders(context)
        if ordinal < len(candidates):
            item = candidates[ordinal]
            identifier = item.get("order_id") or item.get("order_number")
            if identifier:
                return prepare_workspace_processing_action([str(identifier)], mode="single", requested_by=requested_by, original_message=message, context=context)
        return {
            "message": "I could not find that approved order in the visible queue.",
            "status": "needs_choice",
            "actions": [{"type": "ask_order_choice", "reason": "ordinal_not_visible"}],
        }
    return None


def _latest_pending_action_id(context: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(context, dict):
        return None
    direct = str(context.get("latest_pending_action_id") or "").strip()
    if direct:
        return direct
    pending = context.get("latest_pending_action")
    if isinstance(pending, dict):
        value = str(pending.get("pending_action_id") or "").strip()
        if value:
            return value
    return None


def _is_continue_message(message: str) -> bool:
    lower = (message or "").strip().lower()
    return lower in {"continue", "continue anyway", "yes continue", "proceed", "go ahead", "do it", "yes", "ok continue"} or any(
        phrase in lower for phrase in ("ignore warning", "ignore that warning", "ignore warnings")
    )


def _agent_model() -> str:
    return os.getenv("OPENAI_AGENT_MODEL") or "gpt-5.4-mini"


def _fallback_intent(message: str, selected_order_number: Optional[str], selected_order_id: Optional[str]) -> Dict[str, Any]:
    text = (message or "").strip()
    lower = text.lower()
    order_numbers = extract_order_tokens(text)
    order_number = order_numbers[0] if order_numbers else (normalize_order_token(selected_order_number) if selected_order_number else None)
    order_id = selected_order_id if not order_number else None
    mode = infer_processing_mode(text, len(order_numbers))
    if any(phrase in lower for phrase in ("process this", "process selected")) and selected_order_id:
        return {"intent": "process_order", "order_id": selected_order_id}
    if "process latest" in lower:
        return {"intent": "process_latest_approved"}
    if "process" in lower and len(order_numbers) > 1:
        return {"intent": "process_orders", "order_numbers": order_numbers, "mode": mode}
    if "process" in lower and (order_number or order_id):
        return {"intent": "process_order", "order_number": order_number, "order_id": order_id, "order_numbers": order_numbers, "mode": mode}
    if "approved" in lower and any(word in lower for word in ("show", "list", "waiting", "ready")):
        return {"intent": "show_approved_orders"}
    if "review" in lower or "warning" in lower:
        return {"intent": "needs_review"}
    if "recent" in lower or "production files" in lower:
        return {"intent": "recent_files"}
    if "label" in lower:
        if order_numbers and len(order_numbers) > 1:
            return {"intent": "process_orders", "order_numbers": order_numbers, "mode": mode}
        if "latest" in lower or "download" in lower:
            return {"intent": "latest_labels"}
        if order_number or order_id:
            return {"intent": "generate_labels", "order_number": order_number, "order_id": order_id, "order_numbers": order_numbers, "mode": mode}
    if order_number:
        return {"intent": "order_summary", "order_number": order_number}
    return {"intent": "help"}


def _classify_with_openai(message: str, selected_order_number: Optional[str], selected_order_id: Optional[str]) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_intent(message, selected_order_number, selected_order_id)
    payload = {
        "model": _agent_model(),
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": AGENT_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Classify this operator command into one safe tool intent. "
                            "Return JSON only.\n"
                            f"selected_order_id={selected_order_id or ''}\n"
                            f"selected_order_number={selected_order_number or ''}\n"
                            f"message={message or ''}"
                        ),
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "workspace_agent_intent",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "intent": {
                            "type": "string",
                            "enum": [
                                "process_order",
                                "process_orders",
                                "process_latest_approved",
                                "show_approved_orders",
                                "needs_review",
                                "recent_files",
                                "latest_labels",
                                "generate_labels",
                                "order_summary",
                                "help",
                            ],
                        },
                        "order_number": {"type": "string"},
                        "order_numbers": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "order_id": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["single", "combined", "separate"],
                        },
                    },
                    "required": ["intent", "order_number", "order_numbers", "order_id", "mode"],
                },
            }
        },
    }
    try:
        response = httpx.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=httpx.Timeout(12.0, connect=5.0),
        )
        if response.status_code >= 400:
            return _fallback_intent(message, selected_order_number, selected_order_id)
        data = response.json()
        output = data.get("output_text") or ""
        if not output:
            parts = []
            for item in data.get("output") or []:
                for content in item.get("content") or []:
                    if content.get("type") in {"output_text", "text"} and content.get("text"):
                        parts.append(content["text"])
            output = "\n".join(parts)
        parsed = json.loads(output or "{}")
        if not parsed.get("order_number") and selected_order_number:
            parsed["order_number"] = normalize_order_token(selected_order_number)
        if not parsed.get("order_id") and selected_order_id and not parsed.get("order_number"):
            parsed["order_id"] = selected_order_id
        deterministic_tokens = extract_order_tokens(message)
        if deterministic_tokens:
            parsed["order_numbers"] = deterministic_tokens
            parsed["order_number"] = deterministic_tokens[0]
            parsed["mode"] = infer_processing_mode(message, len(deterministic_tokens))
            if len(deterministic_tokens) > 1 and parsed.get("intent") in {"process_order", "generate_labels", "order_summary", "help"}:
                parsed["intent"] = "process_orders"
        elif not parsed.get("mode"):
            parsed["mode"] = "single"
        return parsed
    except Exception:
        return _fallback_intent(message, selected_order_number, selected_order_id)


def _download_files_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    files = result.get("files") if isinstance(result, dict) else None
    if isinstance(files, dict):
        return files
    batch = result.get("batch") if isinstance(result, dict) else None
    if isinstance(batch, dict):
        return {
            "processing_pdf_url": batch.get("processing_pdf_url"),
            "labels_pdf_url": batch.get("labels_pdf_url"),
        }
    return {}


def _format_process_response(result: Dict[str, Any]) -> Dict[str, Any]:
    status = result.get("status")
    order = result.get("order") or {}
    order_number = order.get("order_number") or result.get("order_number") or "order"
    summary = result.get("summary") or (result.get("batch") or {}).get("summary") or {}
    files = _download_files_from_result(result)
    if status == "success":
        message = (
            f"Done. Order {order_number} has been processed.\n"
            f"Pieces: {summary.get('total_pieces', 0)}\n"
            f"Area: {summary.get('total_area_m2', 0)} m2\n"
            f"Glass types: {summary.get('glass_type_count', 0)}"
        )
    elif status == "blocked":
        message = f"I found {order_number}, but it is not approved. Please review and approve it first."
    elif status == "needs_review":
        warnings = result.get("warnings") or []
        warning_text = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(warnings[:5]))
        message = f"I found {len(warnings)} warning(s) before processing:\n{warning_text}\nI stopped before creating production files."
    elif status == "already_processed":
        message = f"{order_number} already has a processing batch. Existing files are available."
    elif status == "not_found":
        message = f"I couldn't find order {order_number}."
    elif status == "multiple_matches":
        message = "I found multiple matching orders. Please choose one."
    elif status == "file_generation_failed":
        message = result.get("error_message") or "Production files failed to generate."
    else:
        message = "I could not complete that workspace action."
    return {
        "message": message,
        "status": status,
        "actions": result.get("actions") or [],
        "files": files,
        "needs_confirmation": bool(result.get("needs_confirmation")),
        "confirmation_payload": result.get("confirmation_payload"),
        "result": result,
    }


def _latest_approved_identifier(queue: Dict[str, Any]) -> Optional[Any]:
    approved = ((queue.get("groups") or {}).get("approved_ready") or [])
    if not approved:
        return None
    return approved[0].get("order_id") or approved[0].get("order_number")


def _plan_for_response(
    message: str,
    response: Dict[str, Any],
    *,
    intent: str,
    orders_to_resolve: Optional[List[str]] = None,
    planned_tool: str = "none",
    mode: Optional[str] = None,
    reason: str = "",
) -> Dict[str, Any]:
    orders = orders_to_resolve or extract_order_tokens(message)
    merge_across_orders: Optional[bool] = None
    action = next((item for item in response.get("actions") or [] if isinstance(item, dict) and item.get("type") == "process_via_existing_modules"), None)
    if action:
        orders = [str(item) for item in (action.get("identifiers") or orders) if str(item or "").strip()]
        mode = action.get("mode") or mode
        if isinstance(action.get("mergeAcrossOrders"), bool):
            merge_across_orders = action.get("mergeAcrossOrders")
        planned_tool = "process_orders_via_existing_modules"
    pending = response.get("pending_action") if isinstance(response.get("pending_action"), dict) else None
    if pending:
        orders = [str(item) for item in (pending.get("order_numbers") or orders) if str(item or "").strip()]
        mode = pending.get("mode") or mode
        planned_tool = "process_orders_via_existing_modules"
    if response.get("status") == "frontend_workflow_required":
        planned_tool = planned_tool if planned_tool != "none" else "process_orders_via_existing_modules"
    return _make_action_plan(
        intent=intent,
        orders_to_resolve=orders,
        mode=mode,
        merge_across_orders=merge_across_orders,
        requires_confirmation=bool(response.get("needs_confirmation") or response.get("pending_action")),
        planned_tool=planned_tool,
        reason=reason or response.get("message") or "",
    )


def run_workspace_agent(
    message: str,
    *,
    selected_order_id: Optional[str] = None,
    selected_order_number: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    requested_by: str = "workspace",
) -> Dict[str, Any]:
    if isinstance(context, dict):
        selected_order_id = selected_order_id or context.get("selected_order_id")
        selected_order_number = selected_order_number or context.get("selected_order_number")

    try:
        from workspace_agents.sdk_agent import run_workspace_agent_sdk, should_use_agents_sdk

        if should_use_agents_sdk():
            return run_workspace_agent_sdk(
                message,
                selected_order_id=selected_order_id,
                selected_order_number=selected_order_number,
                context=context,
                requested_by=requested_by,
            )
    except Exception as exc:
        record_workspace_action(
            actor=requested_by,
            action_type="agent_engine_fallback",
            status="legacy",
            requested_message=message,
            input_json={"engine": "agents_sdk"},
            error_message=str(exc),
        )

    record_workspace_action(
        actor=requested_by,
        action_type="agent_command_received",
        status="received",
        order_id=int(selected_order_id) if str(selected_order_id or "").isdigit() else None,
        order_number=selected_order_number,
        requested_message=message,
    )
    direct_response = legacy_conversational_response(message, context)
    if direct_response:
        response = _attach_action_plan(
            direct_response,
            _make_action_plan(
                intent=str(direct_response.get("intent") or "conversation"),
                planned_tool="none",
                reason="Legacy fallback conversational response; no backend action needed.",
            ),
            requested_by,
            message,
        )
        return ensure_workspace_agent_response(response)
    if _is_continue_message(message):
        pending_action_id = _latest_pending_action_id(context)
        if not pending_action_id:
            response = {"message": "There is no pending action to continue. Tell me which order to process.", "status": "missing_pending_action"}
            return _attach_action_plan(response, _plan_for_response(message, response, intent="continue_pending_action", planned_tool="continue_pending_action", reason="No latest pending action in context."), requested_by, message)
        response = confirm_workspace_pending_action(
            pending_action_id,
            decision="continue",
            requested_by=requested_by,
            context=context,
        )
        return _attach_action_plan(response, _plan_for_response(message, response, intent="continue_pending_action", planned_tool="continue_pending_action", reason="Continue latest pending action."), requested_by, message)
    context_action = _resolve_context_action(message, context, requested_by)
    if context_action:
        intent_name = "answer_question"
        if context_action.get("status") == "frontend_workflow_required" or context_action.get("pending_action"):
            intent_name = "process_orders"
        elif any(action.get("type") == "show_last_agent_files" for action in context_action.get("actions") or [] if isinstance(action, dict)):
            intent_name = "download_labels_pdf"
        elif any(action.get("type") in {"open_last_processing_sheet", "open_last_label_job"} for action in context_action.get("actions") or [] if isinstance(action, dict)):
            intent_name = "open_processing_sheet"
        return _attach_action_plan(context_action, _plan_for_response(message, context_action, intent=intent_name, reason="Resolved from Workspace context."), requested_by, message)
    if any(phrase in (message or "").lower() for phrase in ("what should i do next", "what next", "next step")):
        queue = get_workspace_queue()
        approved = (queue.get("groups") or {}).get("approved_ready") or []
        review = (queue.get("groups") or {}).get("needs_review") or []
        if approved:
            response = {
                "message": f"You have {len(approved)} approved order(s) ready. The next useful step is processing the latest approved order or selecting a few to process together.",
                "status": "ok",
                "actions": [{"type": "show_queue", "group": "approved_ready"}],
                "queue": queue,
            }
        elif review:
            response = {
                "message": f"{len(review)} order(s) need review first. Open the Needs Review group and approve the ones that are ready for production.",
                "status": "ok",
                "actions": [{"type": "show_queue", "group": "needs_review"}],
                "queue": queue,
            }
        else:
            response = {
                "message": "Nothing is waiting in the Workspace queue right now.",
                "status": "ok",
                "actions": [{"type": "show_queue"}],
                "queue": queue,
            }
        return _attach_action_plan(response, _plan_for_response(message, response, intent="workflow_question", planned_tool="get_workspace_queue"), requested_by, message)
    intent = _classify_with_openai(message, selected_order_number, selected_order_id)
    name = intent.get("intent") or "help"

    if name == "show_approved_orders":
        queue = get_workspace_queue()
        approved = (queue.get("groups") or {}).get("approved_ready") or []
        response = {
            "message": f"{len(approved)} approved order(s) are ready for processing.",
            "status": "ok",
            "actions": [{"type": "show_queue", "group": "approved_ready"}],
            "queue": queue,
        }
        return _attach_action_plan(response, _plan_for_response(message, response, intent="show_ready_orders", planned_tool="get_workspace_queue"), requested_by, message)
    if name == "needs_review":
        queue = get_workspace_queue()
        review = (queue.get("groups") or {}).get("needs_review") or []
        response = {
            "message": f"{len(review)} order(s) need review before production.",
            "status": "ok",
            "actions": [{"type": "show_queue", "group": "needs_review"}],
            "queue": queue,
        }
        return _attach_action_plan(response, _plan_for_response(message, response, intent="check_warnings", planned_tool="get_workspace_queue"), requested_by, message)
    if name == "recent_files":
        files = get_recent_production_files()
        response = {
            "message": f"Showing {len(files.get('items') or [])} recent production file set(s).",
            "status": "ok",
            "actions": [{"type": "show_recent_files"}],
            "recent_files": files.get("items") or [],
        }
        return _attach_action_plan(response, _plan_for_response(message, response, intent="show_recent_files", planned_tool="get_recent_production_files"), requested_by, message)
    if name == "latest_labels":
        recent = get_recent_production_files(1).get("items") or []
        latest = recent[0] if recent else None
        if not latest:
            response = {"message": "No production files have been generated yet.", "status": "not_found"}
            return _attach_action_plan(response, _plan_for_response(message, response, intent="download_labels_pdf", planned_tool="get_recent_production_files"), requested_by, message)
        response = {
            "message": f"Latest labels are ready for {latest.get('order_number')}.",
            "status": "ok",
            "files": {
                "processing_pdf_url": latest.get("processing_pdf_url"),
                "labels_pdf_url": latest.get("labels_pdf_url"),
            },
            "recent_files": recent,
        }
        return _attach_action_plan(response, _plan_for_response(message, response, intent="download_labels_pdf", planned_tool="get_recent_production_files"), requested_by, message)
    if name == "process_latest_approved":
        queue = get_workspace_queue()
        identifier = _latest_approved_identifier(queue)
        if not identifier:
            response = {"message": "No approved orders are waiting for processing.", "status": "not_found"}
            return _attach_action_plan(response, _plan_for_response(message, response, intent="process_orders", planned_tool="get_workspace_queue"), requested_by, message)
        response = prepare_workspace_processing_action(
            [identifier],
            mode="single",
            requested_by=requested_by,
            original_message=message,
            context=context,
        )
        return _attach_action_plan(response, _plan_for_response(message, response, intent="process_orders", orders_to_resolve=[str(identifier)], mode="single", planned_tool="process_orders_via_existing_modules", reason="Latest approved order from backend queue."), requested_by, message)
    if name in {"process_order", "generate_labels", "process_orders"}:
        order_numbers = intent.get("order_numbers") if isinstance(intent.get("order_numbers"), list) else []
        order_numbers = [normalize_order_token(item) for item in order_numbers if normalize_order_token(item)]
        if len(order_numbers) > 1:
            mode = intent.get("mode") or infer_processing_mode(message, len(order_numbers))
            if mode == "single":
                mode = "combined"
            response = prepare_workspace_processing_action(
                order_numbers,
                mode=mode,
                requested_by=requested_by,
                original_message=message,
                context=context,
            )
            return _attach_action_plan(response, _plan_for_response(message, response, intent="process_orders", orders_to_resolve=order_numbers, mode=mode, planned_tool="process_orders_via_existing_modules"), requested_by, message)
        identifier = intent.get("order_number") or intent.get("order_id") or selected_order_id or selected_order_number
        if not identifier:
            response = {"message": "Tell me which order number to process.", "status": "missing_order"}
            return _attach_action_plan(response, _plan_for_response(message, response, intent="process_orders", planned_tool="resolve_orders"), requested_by, message)
        if intent.get("order_number"):
            identifier = normalize_order_token(identifier)
        response = prepare_workspace_processing_action(
            [identifier],
            mode="single",
            requested_by=requested_by,
            original_message=message,
            context=context,
        )
        return _attach_action_plan(response, _plan_for_response(message, response, intent="process_orders", orders_to_resolve=[str(identifier)], mode="single", planned_tool="process_orders_via_existing_modules"), requested_by, message)
    if name == "order_summary":
        identifier = intent.get("order_number") or intent.get("order_id") or selected_order_id or selected_order_number
        if not identifier:
            response = {"message": "Tell me which order to check.", "status": "missing_order"}
            return _attach_action_plan(response, _plan_for_response(message, response, intent="inspect_order", planned_tool="resolve_orders"), requested_by, message)
        validation = validate_order_for_processing(identifier)
        response = {
            "message": f"Order check status: {validation.get('status')}.",
            "status": validation.get("status"),
            "result": validation,
        }
        return _attach_action_plan(response, _plan_for_response(message, response, intent="inspect_order", orders_to_resolve=[str(identifier)], planned_tool="validate_orders_for_processing"), requested_by, message)

    response = {
        "message": "I can process approved orders, show approved orders, list recent production files, or show what needs review.",
        "status": "help",
    }
    return _attach_action_plan(response, _plan_for_response(message, response, intent="answer_question"), requested_by, message)
