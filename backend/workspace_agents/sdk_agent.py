from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from db import (
    get_order_with_extraction,
    get_orders,
    get_orders_by_identifiers,
    normalize_order_number,
    normalize_order_status,
    record_workspace_action,
)
from workspace_service import (
    get_processing_batch_files,
    get_recent_production_files as service_get_recent_production_files,
    get_workspace_queue as service_get_workspace_queue,
    prepare_workspace_processing_action,
    validate_order_for_processing,
)
from .response_format import ensure_workspace_agent_response


WORKFLOW_NAME = "Workspace Factory Assistant"
SOURCE = "workspace_agent"
ORDER_RE = re.compile(r"\b(?:R-?)?\d{2}-\d{3,5}\b", re.IGNORECASE)


AGENT_INSTRUCTIONS = """You are the Workspace Factory Assistant, a workflow operator for a factory-grade glass order extraction and production platform.

You run approved-order workflows using safe backend tools. You are focused on Workspace production actions, queue checks, file retrieval, and workflow failure explanations.

Act like a practical factory coworker:
- practical, short, clear
- short answers
- clear next steps
- no long explanations unless asked
- always use tools for real platform data and production actions
- never invent order data or file links

Conversation and intent judgment:
- For casual/general chat, direct the user to Smart Chat below in one short sentence.
- If the user asks what you can do, explain your workflow abilities only.
- If the user asks about live Workspace data, use read-only tools.
- If the user asks about live platform data, use read-only tools.
- If the user asks to process, download, open, generate, or change something, use the appropriate safe tool.
- If the message combines casual chat with an action request, briefly acknowledge the casual part, then use tools for the action.
- If the user's request is ambiguous, ask one short clarification.
- Never guess order data or file links.
- Never call production tools unless the user clearly requested an action.

Examples:
- User: "hi how are you" -> Reply like: "I am mainly for factory workflows. Use Smart Chat below for normal questions." Do not call tools.
- User: "what day is today btw" -> Reply like: "Smart Chat below is better for date and general questions." Do not call tools.
- User: "what can you do?" -> Explain workflow abilities briefly. Do not call tools unless live data is requested.
- User: "hi can you process R-25-1290" -> Briefly acknowledge, then resolve/check/process with tools.
- User: "process these together" -> combined mode with merge_across_orders=false.
- User: "process these together but don't mix orders" -> combined mode with merge_across_orders=false.
- User: "process these together and merge across orders" -> combined mode with merge_across_orders=true.
- User: "mix same glass across orders" -> combined mode with merge_across_orders=true.
- User: "separately" -> separate mode.

Core workflow:
Approved order -> isolated Workspace processing job -> same Processing module logic -> Danko rounding -> group dimensions -> Processing PDF export -> Labels Add from Processing -> Labels PDF export.

Data safety:
- Never overwrite original PDF data, extracted text, raw order rows, raw dimensions, raw area, raw glass type, manual overrides, approved data, or history data.
- Never modify the active manual Processing tab when running Workspace jobs.
- Workspace jobs must be isolated from manual Processing.

Approval policy:
- Draft/unapproved orders must not be processed.
- Approved orders can be processed.
- Old extraction warnings on approved orders are informational only and must not block processing.
- The Check action can explain warnings, but Process should continue for approved orders.

Confirmation policy:
Require confirmation for reprocessing an existing Workspace job, replacing existing files, loading a Workspace job into manual Processing, destructive actions, or overwriting anything.
Do not require confirmation only because an approved order has old warnings.

Use frontend context only to understand phrases like this, these, latest approved, download labels again, continue, and same as before.
Database/tool results are the source of truth. If unclear, ask the user to choose. Never process based only on frontend context.

Use only the exposed high-level tools. Do not ask for raw database mutation tools.
Never process, reprocess, overwrite, or create files unless the user clearly asks for an action.

When you finish, return JSON only with this shape:
{
  "message": "short operator-facing answer",
  "cards": [{"type":"success|warning|error|choices|files","title":"...","body":"...","actions":[{"label":"...","kind":"open|download|confirm|cancel","url":null,"payload":null}]}],
  "last_agent_action": null,
  "refresh": {"queue": false, "recent_files": false}
}
"""


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


def current_datetime_summary() -> Dict[str, str]:
    now = datetime.now().astimezone()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "day_name": now.strftime("%A"),
        "time": now.strftime("%H:%M"),
        "timezone": now.tzname() or str(now.tzinfo or ""),
    }


def agent_model() -> str:
    return os.getenv("OPENAI_AGENT_MODEL") or os.getenv("EXTRACTION_MODEL") or "gpt-5-mini"


def _sdk_imports() -> Tuple[Any, Any, Any, Any, Any, Optional[Exception]]:
    try:
        from agents import Agent, RunConfig, RunContextWrapper, Runner, function_tool
        return Agent, RunConfig, RunContextWrapper, Runner, function_tool, None
    except Exception as exc:
        return None, None, None, None, None, exc


def agents_sdk_available() -> bool:
    return _sdk_imports()[-1] is None


def should_use_agents_sdk() -> bool:
    engine = (os.getenv("WORKSPACE_AGENT_ENGINE") or "agents_sdk").strip().lower()
    if engine in {"legacy", "custom"}:
        return False
    if engine not in {"agents_sdk", "sdk", "openai_agents"}:
        return False
    if not os.getenv("OPENAI_API_KEY"):
        return False
    return agents_sdk_available()


def _context(ctx: Any) -> Dict[str, Any]:
    value = getattr(ctx, "context", None)
    return value if isinstance(value, dict) else {}


def _request_context(ctx: Any) -> Dict[str, Any]:
    value = _context(ctx).get("workspace_context")
    return value if isinstance(value, dict) else {}


def _requested_by(ctx: Any) -> str:
    return str(_context(ctx).get("requested_by") or SOURCE)


def _summarize_queue_card(item: Dict[str, Any], queue_group: str) -> Dict[str, Any]:
    return {
        "order_id": str(item.get("order_id") or item.get("id") or ""),
        "order_number": item.get("order_number"),
        "client_name": item.get("client_name"),
        "status": item.get("status"),
        "queue_group": queue_group,
        "pieces": item.get("total_pieces"),
        "area": item.get("total_area_m2"),
        "warnings_count": item.get("warnings_count"),
        "has_processing_pdf": bool(item.get("processing_pdf_url")),
        "has_labels_pdf": bool(item.get("labels_pdf_url")),
        "processing_sheet_id": item.get("batch_id"),
        "label_job_id": item.get("label_job_id"),
        "created_at": item.get("created_at"),
        "approved_at": item.get("approved_at"),
    }


def _safe_context_summary(context: Dict[str, Any]) -> Dict[str, Any]:
    selected_orders = context.get("selected_orders") if isinstance(context.get("selected_orders"), list) else []
    visible_orders = context.get("visible_orders") if isinstance(context.get("visible_orders"), list) else []
    return {
        "current_page": context.get("current_page"),
        "workspace_session_id": context.get("workspace_session_id") or context.get("session_id"),
        "selected_orders": selected_orders[:10],
        "visible_orders": visible_orders[:50],
        "last_agent_action": context.get("last_agent_action") if isinstance(context.get("last_agent_action"), dict) else None,
        "latest_pending_action": context.get("latest_pending_action") if isinstance(context.get("latest_pending_action"), dict) else None,
    }


def _primary_order_number(order: Dict[str, Any]) -> str:
    numbers = order.get("order_numbers") or []
    if isinstance(numbers, list) and numbers:
        return str(numbers[0])
    for row in order.get("rows") or []:
        if row.get("order_number"):
            return str(row["order_number"])
    return str(order.get("order_number") or order.get("id") or "")


def _order_summary(order: Dict[str, Any]) -> Dict[str, Any]:
    rows = order.get("rows") if isinstance(order.get("rows"), list) else []
    pieces = 0
    area = 0.0
    glass_types = set()
    for row in rows:
        try:
            pieces += int(row.get("quantity") or row.get("qty") or 0)
        except Exception:
            pass
        try:
            area += float(str(row.get("area") or row.get("area_m2") or 0).replace(",", "."))
        except Exception:
            pass
        glass_type = str(row.get("type") or row.get("glass_type") or "").strip()
        if glass_type:
            glass_types.add(glass_type)
    return {
        "pieces": pieces or order.get("units_total") or order.get("total_pieces"),
        "area": round(area, 3) if rows else order.get("area_total") or order.get("total_area_m2"),
        "glass_type_count": len(glass_types),
        "line_count": len(rows),
    }


def _resolve_exact(identifier: Any) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    text = str(identifier or "").strip()
    if not text:
        return None, []
    normalized = normalize_order_token(text)
    matches = get_orders_by_identifiers([normalized or text])
    if not matches:
        return None, []
    if len(matches) == 1:
        return matches[0], matches
    exact: List[Dict[str, Any]] = []
    normalized_db = normalize_order_number(normalized or text).lower()
    for order in matches:
        if str(order.get("id")) == text:
            exact.append(order)
            continue
        numbers = [normalize_order_number(num).lower() for num in (order.get("order_numbers") or [])]
        if normalized_db in numbers:
            exact.append(order)
    if len(exact) == 1:
        return exact[0], matches
    return None, matches


def _resolve_from_context(ref: str, context: Dict[str, Any]) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    lower = str(ref or "").strip().lower()
    selected = context.get("selected_orders") if isinstance(context.get("selected_orders"), list) else []
    visible = context.get("visible_orders") if isinstance(context.get("visible_orders"), list) else []
    if lower in {"this", "selected order", "selected"}:
        if len(selected) == 1:
            return [str(selected[0].get("order_id") or selected[0].get("id") or selected[0].get("order_number"))], None
        return [], {"status": "ambiguous" if selected else "missing", "message": "Select one order or provide the order number."}
    if lower in {"these", "these orders", "selected orders"}:
        if selected:
            return [str(item.get("order_id") or item.get("id") or item.get("order_number")) for item in selected], None
        return [], {"status": "missing", "message": "Select orders or provide order numbers."}
    if lower == "these two":
        if len(selected) == 2:
            return [str(item.get("order_id") or item.get("id") or item.get("order_number")) for item in selected], None
        return [], {"status": "ambiguous", "message": "Select exactly two orders or provide both order numbers."}
    if lower == "latest approved":
        queue = service_get_workspace_queue()
        approved = ((queue.get("groups") or {}).get("approved_ready") or [])
        if approved:
            first = approved[0]
            return [str(first.get("order_id") or first.get("id") or first.get("order_number"))], None
        return [], {"status": "not_found", "message": "No approved orders are waiting for processing."}
    ordinal_match = re.search(r"\b(second|2nd|first|1st|third|3rd)\s+approved", lower)
    if ordinal_match:
        index = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}[ordinal_match.group(1)]
        approved = [
            item for item in visible
            if str(item.get("queue_group") or "").lower() == "approved_ready"
            or str(item.get("status") or "").lower() == "approved"
        ]
        if index < len(approved):
            item = approved[index]
            return [str(item.get("order_id") or item.get("id") or item.get("order_number"))], None
        return [], {"status": "not_found", "message": "I could not find that approved order in the visible queue."}
    return [], None


def _build_tools(function_tool: Any, RunContextWrapper: Any) -> List[Any]:
    @function_tool(strict_mode=False)
    def get_workspace_context(ctx: RunContextWrapper) -> Dict[str, Any]:
        """Return the safe frontend Workspace context for resolving natural references."""
        result = _safe_context_summary(_request_context(ctx))
        record_workspace_action(
            actor=_requested_by(ctx),
            action_type="agent_tool_called",
            status="success",
            tool_name="get_workspace_context",
            output_json={"selected_count": len(result.get("selected_orders") or []), "visible_count": len(result.get("visible_orders") or [])},
        )
        return result

    @function_tool(strict_mode=False)
    def get_workspace_queue(ctx: RunContextWrapper) -> Dict[str, Any]:
        """Return grouped Workspace queue summaries."""
        queue = service_get_workspace_queue()
        groups = {}
        for key, items in (queue.get("groups") or {}).items():
            groups[key] = [_summarize_queue_card(item, key) for item in (items or [])]
        result = {"groups": groups, "counts": queue.get("counts") or {}}
        record_workspace_action(
            actor=_requested_by(ctx),
            action_type="agent_tool_called",
            status="success",
            tool_name="get_workspace_queue",
            output_json={"counts": result["counts"]},
        )
        return result

    @function_tool(strict_mode=False)
    def resolve_orders(ctx: RunContextWrapper, refs: List[str], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Resolve order references and order numbers to database order IDs."""
        request_context = context if isinstance(context, dict) and context else _request_context(ctx)
        requested = [str(item).strip() for item in (refs or []) if str(item or "").strip()]
        if not requested:
            return {"status": "missing", "resolved": [], "message": "No order references were provided."}

        expanded: List[str] = []
        choices: List[Dict[str, Any]] = []
        for ref in requested:
            natural, issue = _resolve_from_context(ref, request_context)
            if issue:
                choices.append({"ref": ref, **issue})
            elif natural:
                expanded.extend(natural)
            else:
                tokens = extract_order_tokens(ref)
                expanded.extend(tokens or [normalize_order_token(ref) or ref])

        resolved: List[Dict[str, Any]] = []
        missing: List[str] = []
        ambiguous: List[Dict[str, Any]] = choices[:]
        seen_ids = set()
        for identifier in expanded:
            order, matches = _resolve_exact(identifier)
            if order:
                order_id = str(order.get("id"))
                if order_id not in seen_ids:
                    seen_ids.add(order_id)
                    resolved.append({
                        "order_id": order_id,
                        "order_number": _primary_order_number(order),
                        "client_name": order.get("client_name") or order.get("client_hint"),
                        "status": normalize_order_status(order.get("status")),
                    })
                continue
            if matches:
                ambiguous.append({
                    "ref": identifier,
                    "status": "ambiguous",
                    "choices": [
                        {
                            "order_id": str(match.get("id")),
                            "order_number": _primary_order_number(match),
                            "client_name": match.get("client_name") or match.get("client_hint"),
                            "status": normalize_order_status(match.get("status")),
                        }
                        for match in matches
                    ],
                })
            else:
                missing.append(identifier)

        status = "ok"
        if ambiguous:
            status = "ambiguous"
        elif missing:
            status = "not_found"
        result = {"status": status, "resolved": resolved, "missing": missing, "ambiguous": ambiguous}
        record_workspace_action(
            actor=_requested_by(ctx),
            action_type="agent_tool_called",
            status=status,
            tool_name="resolve_orders",
            input_json={"refs": requested},
            output_json=result,
        )
        return result

    @function_tool(strict_mode=False)
    def get_order_details(ctx: RunContextWrapper, order_ids: List[str]) -> Dict[str, Any]:
        """Return safe order summaries without raw extracted text."""
        orders: List[Dict[str, Any]] = []
        missing: List[str] = []
        for order_id in order_ids or []:
            if not str(order_id).isdigit():
                missing.append(str(order_id))
                continue
            detail = get_order_with_extraction(int(order_id))
            if not detail:
                missing.append(str(order_id))
                continue
            orders.append({
                "order_id": str(detail.get("id")),
                "order_number": _primary_order_number(detail),
                "client_name": detail.get("client_name") or detail.get("client_hint"),
                "status": normalize_order_status(detail.get("status")),
                "summary": _order_summary(detail),
                "warnings_count": len((validate_order_for_processing(detail.get("id")).get("warnings") or [])),
            })
        result = {"status": "ok" if not missing else "partial", "orders": orders, "missing": missing}
        record_workspace_action(
            actor=_requested_by(ctx),
            action_type="agent_tool_called",
            status=result["status"],
            tool_name="get_order_details",
            input_json={"order_ids": order_ids},
            output_json={"orders": [item.get("order_number") for item in orders], "missing": missing},
        )
        return result

    @function_tool(strict_mode=False)
    def check_orders(ctx: RunContextWrapper, order_ids: List[str]) -> Dict[str, Any]:
        """Check approval status, warnings, and existing production files for orders."""
        checks: List[Dict[str, Any]] = []
        blockers: List[Dict[str, Any]] = []
        for order_id in order_ids or []:
            detail = get_order_with_extraction(int(order_id)) if str(order_id).isdigit() else None
            if not detail:
                blockers.append({"order_id": str(order_id), "reason": "not_found"})
                continue
            status = normalize_order_status(detail.get("status"))
            validation = validate_order_for_processing(detail.get("id"))
            check = {
                "order_id": str(detail.get("id")),
                "order_number": _primary_order_number(detail),
                "status": status,
                "approved": status == "approved",
                "warnings": validation.get("warnings") or [],
                "warnings_blocking": False,
                "existing_files": {},
            }
            if status != "approved":
                blockers.append({"order_id": check["order_id"], "order_number": check["order_number"], "reason": "order_not_approved", "status": status})
            checks.append(check)
        result = {"status": "blocked" if blockers else "ok", "checks": checks, "blockers": blockers}
        record_workspace_action(
            actor=_requested_by(ctx),
            action_type="agent_tool_called",
            status=result["status"],
            tool_name="check_orders",
            input_json={"order_ids": order_ids},
            output_json={"blockers": blockers, "orders": [item.get("order_number") for item in checks]},
        )
        return result

    @function_tool(strict_mode=False)
    def process_orders_via_existing_modules(
        ctx: RunContextWrapper,
        order_ids: List[str],
        mode: str = "combined",
        merge_across_orders: Optional[bool] = None,
        source: str = SOURCE,
        force_reprocess: bool = False,
    ) -> Dict[str, Any]:
        """Prepare the real Workspace Processing/Labels module workflow for approved orders."""
        normalized_mode = "separate" if mode == "separate" else ("combined" if len(order_ids or []) > 1 else "single")
        result = prepare_workspace_processing_action(
            order_ids or [],
            mode=normalized_mode,
            merge_across_orders=merge_across_orders if normalized_mode == "combined" else False,
            requested_by=source or SOURCE,
            original_message=str(_context(ctx).get("message") or ""),
            context=_request_context(ctx),
        )
        record_workspace_action(
            actor=_requested_by(ctx),
            action_type="agent_tool_called",
            status=result.get("status") or "unknown",
            tool_name="process_orders_via_existing_modules",
            input_json={"order_ids": order_ids, "mode": normalized_mode, "mergeAcrossOrders": merge_across_orders if normalized_mode == "combined" else False, "source": source, "force_reprocess": force_reprocess},
            output_json={"status": result.get("status"), "actions": result.get("actions"), "order_numbers": result.get("order_numbers")},
            requires_confirmation=bool(result.get("needs_confirmation") or result.get("pending_action")),
        )
        return result

    @function_tool(strict_mode=False)
    def get_recent_production_files(ctx: RunContextWrapper, limit: int = 20) -> Dict[str, Any]:
        """Return recent production files with download links if available."""
        result = service_get_recent_production_files(limit)
        record_workspace_action(
            actor=_requested_by(ctx),
            action_type="agent_tool_called",
            status="success",
            tool_name="get_recent_production_files",
            input_json={"limit": limit},
            output_json={"count": len(result.get("items") or [])},
        )
        return result

    @function_tool(strict_mode=False)
    def get_file_links_for_order_or_job(ctx: RunContextWrapper, order_ids: Optional[List[str]] = None, processing_sheet_id: Optional[str] = None, label_job_id: Optional[str] = None) -> Dict[str, Any]:
        """Return verified existing Processing PDF and Labels PDF links when known."""
        context = _request_context(ctx)
        last_action = context.get("last_agent_action") if isinstance(context.get("last_agent_action"), dict) else {}
        files = last_action.get("files") if isinstance(last_action.get("files"), dict) else None
        if files:
            result = {"status": "ok", "source": "workspace_context", "files": files, "order_numbers": last_action.get("order_numbers") or []}
        elif processing_sheet_id and str(processing_sheet_id).isdigit():
            result = get_processing_batch_files(int(processing_sheet_id))
        else:
            recent = service_get_recent_production_files(20).get("items") or []
            wanted = {str(item) for item in (order_ids or []) if str(item or "").strip()}
            found = []
            for item in recent:
                if not wanted or str(item.get("order_id")) in wanted or str(item.get("order_number")) in wanted:
                    found.append(item)
            result = {"status": "ok" if found else "not_found", "items": found[:5]}
        record_workspace_action(
            actor=_requested_by(ctx),
            action_type="agent_tool_called",
            status=result.get("status") or "ok",
            tool_name="get_file_links_for_order_or_job",
            input_json={"order_ids": order_ids, "processing_sheet_id": processing_sheet_id, "label_job_id": label_job_id},
            output_json={"status": result.get("status")},
        )
        return result

    @function_tool(strict_mode=False)
    def open_processing_sheet_info(ctx: RunContextWrapper, processing_sheet_id: Optional[str] = None) -> Dict[str, Any]:
        """Return UI action info for opening an isolated Workspace processing sheet."""
        context = _request_context(ctx)
        last_action = context.get("last_agent_action") if isinstance(context.get("last_agent_action"), dict) else {}
        sheet_id = processing_sheet_id or last_action.get("processing_sheet_id") or last_action.get("workspace_job_id")
        result = {
            "status": "ok" if sheet_id else "not_found",
            "processing_sheet_id": sheet_id,
            "message": "Workspace job - isolated from manual sheet." if sheet_id else "No Workspace processing sheet is available in this session.",
            "actions": [{"type": "open_last_processing_sheet"}] if sheet_id else [],
        }
        record_workspace_action(
            actor=_requested_by(ctx),
            action_type="agent_tool_called",
            status=result["status"],
            tool_name="open_processing_sheet_info",
            output_json=result,
        )
        return result

    @function_tool(strict_mode=False)
    def open_label_job_info(ctx: RunContextWrapper, label_job_id: Optional[str] = None) -> Dict[str, Any]:
        """Return UI action info for opening a Workspace label job."""
        context = _request_context(ctx)
        last_action = context.get("last_agent_action") if isinstance(context.get("last_agent_action"), dict) else {}
        job_id = label_job_id or last_action.get("label_job_id")
        result = {
            "status": "ok" if job_id else "not_found",
            "label_job_id": job_id,
            "message": "Opening the Workspace label job." if job_id else "No Workspace label job is available in this session.",
            "actions": [{"type": "open_last_label_job"}] if job_id else [],
        }
        record_workspace_action(
            actor=_requested_by(ctx),
            action_type="agent_tool_called",
            status=result["status"],
            tool_name="open_label_job_info",
            output_json=result,
        )
        return result

    @function_tool(strict_mode=False)
    def explain_last_blocked_action(ctx: RunContextWrapper) -> Dict[str, Any]:
        """Explain why the last action stopped or failed using Workspace session context."""
        context = _request_context(ctx)
        pending = context.get("latest_pending_action") if isinstance(context.get("latest_pending_action"), dict) else None
        last = context.get("last_agent_action") if isinstance(context.get("last_agent_action"), dict) else None
        if pending:
            result = {"status": "ok", "message": "It stopped because a pending action needs confirmation.", "pending_action": pending}
        elif last and last.get("status") in {"blocked", "error"}:
            result = {"status": "ok", "message": f"It stopped because the last action ended with status {last.get('status')}.", "last_agent_action": last}
        else:
            result = {"status": "ok", "message": "I do not have a blocked action in this Workspace session."}
        record_workspace_action(
            actor=_requested_by(ctx),
            action_type="agent_tool_called",
            status=result["status"],
            tool_name="explain_last_blocked_action",
            output_json={"message": result["message"]},
        )
        return result

    return [
        get_workspace_context,
        get_workspace_queue,
        resolve_orders,
        get_order_details,
        check_orders,
        process_orders_via_existing_modules,
        get_recent_production_files,
        get_file_links_for_order_or_job,
        open_processing_sheet_info,
        open_label_job_info,
        explain_last_blocked_action,
    ]


def _create_agent() -> Any:
    Agent, _RunConfig, RunContextWrapper, _Runner, function_tool, error = _sdk_imports()
    if error:
        raise RuntimeError(f"OpenAI Agents SDK is not available: {error}") from error
    return Agent(
        name="Workspace Factory Assistant",
        instructions=AGENT_INSTRUCTIONS,
        model=agent_model(),
        tools=_build_tools(function_tool, RunContextWrapper),
    )


def _parse_final_output(output: Any) -> Dict[str, Any]:
    if isinstance(output, dict):
        return output
    text = str(output or "").strip()
    if not text:
        return {"message": "Done.", "status": "ok"}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"message": text, "status": "ok"}


def run_workspace_agent_sdk(
    message: str,
    *,
    selected_order_id: Optional[str] = None,
    selected_order_number: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    requested_by: str = SOURCE,
) -> Dict[str, Any]:
    Agent, RunConfig, _RunContextWrapper, Runner, _function_tool, error = _sdk_imports()
    if error:
        raise RuntimeError(f"OpenAI Agents SDK is not available: {error}") from error
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for WORKSPACE_AGENT_ENGINE=agents_sdk")

    safe_context = context if isinstance(context, dict) else {}
    record_workspace_action(
        actor=requested_by,
        action_type="agent_command_received",
        status="received",
        order_id=int(selected_order_id) if str(selected_order_id or "").isdigit() else None,
        order_number=selected_order_number,
        requested_message=message,
        input_json={
            "engine": "agents_sdk",
            "model": agent_model(),
            "context_summary": {
                "current_page": safe_context.get("current_page"),
                "selected_count": len(safe_context.get("selected_orders") or []),
                "visible_count": len(safe_context.get("visible_orders") or []),
            },
        },
    )

    agent = _create_agent()
    run_context = {
        "message": message,
        "workspace_context": safe_context,
        "requested_by": requested_by,
    }
    input_payload = {
        "message": message,
        "context": _safe_context_summary(safe_context),
    }
    run_config = RunConfig(
        workflow_name=WORKFLOW_NAME,
        group_id=str(safe_context.get("workspace_session_id") or safe_context.get("session_id") or "workspace"),
        trace_metadata={
            "current_page": str(safe_context.get("current_page") or "Workspace"),
            "source": SOURCE,
            "action_type": "workspace_agent_command",
            "order_numbers": [str(item) for item in (safe_context.get("selected_order_numbers") or [])],
            "model": agent_model(),
        },
    )
    result = Runner.run_sync(
        agent,
        json.dumps(input_payload, ensure_ascii=False, default=str),
        context=run_context,
        max_turns=8,
        run_config=run_config,
    )
    response = _parse_final_output(getattr(result, "final_output", None))
    response = ensure_workspace_agent_response(response)
    record_workspace_action(
        actor=requested_by,
        action_type="agent_response_completed",
        status=response.get("status") or "ok",
        requested_message=message,
        input_json={"engine": "agents_sdk"},
        output_json={"message": response.get("message"), "cards": response.get("cards"), "refresh": response.get("refresh")},
    )
    return response
