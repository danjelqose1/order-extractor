from __future__ import annotations

print("✅ smart_chat loaded")

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from db import record_workspace_action
from .sdk_agent import (
    _safe_context_summary,
    _sdk_imports,
    _summarize_queue_card,
    current_datetime_summary,
)
from workspace_service import get_workspace_queue as service_get_workspace_queue


WORKFLOW_NAME = "Smart Chat"
SOURCE = "workspace_smart_chat"
LOGGER = logging.getLogger(__name__)
PRODUCTION_ACTION_RE = re.compile(
    r"\b(process|reprocess|generate|create|open|download|export|approve|delete|overwrite)\b.*\b(order|orders|label|labels|processing|pdf|file|files|job|sheet)\b"
    r"|\b(process|reprocess|generate labels|download labels|open processing|open label)\b",
    re.IGNORECASE,
)
WORKSPACE_HELP_RE = re.compile(r"\b(explain\s+(?:this\s+page|workspace)|workspace|processing\s+work|labels?\s+work|what\s+should\s+i\s+do\s+next)\b", re.IGNORECASE)


SMART_CHAT_INSTRUCTIONS = """You are Smart Chat, a conversational mini ChatGPT embedded inside a glass factory order platform. Reply naturally and directly to the user. You can chat casually, answer general questions, explain the platform, and help the user think. Do not use canned fallback responses. Do not constantly redirect to Factory Assistant. Only mention Factory Assistant if the user asks for production actions like processing orders, generating PDFs, creating labels, downloading production files, or changing data.

Safety:
- You cannot execute production actions.
- Do not call or invent production tools.
- Never claim you processed orders, generated PDFs, created labels, changed data, approved records, reprocessed jobs, or downloaded files.
- Use read-only platform context only to explain the app or help the user decide what to do next.

Return JSON only:
{
  "message": "short helpful answer",
  "actions": [{"label":"Send to Factory Assistant","kind":"send_to_factory","payload":{"message":"original request"}}],
  "cards": [],
  "refresh": {"queue": false, "recent_files": false}
}
"""


def smart_chat_model() -> str:
    return os.getenv("OPENAI_SMART_CHAT_MODEL") or os.getenv("OPENAI_AGENT_MODEL") or "gpt-5-mini"


def _debug_log(event: str, **fields: Any) -> None:
    safe_fields = {}
    for key, value in fields.items():
        if key.lower() in {"api_key", "authorization", "token", "secret"}:
            continue
        text = str(value)
        safe_fields[key] = text[:500]
    LOGGER.info("Smart Chat debug: %s %s", event, safe_fields)


def _is_development_mode() -> bool:
    env = (os.getenv("APP_ENV") or os.getenv("ENV") or os.getenv("FASTAPI_ENV") or "development").strip().lower()
    return env not in {"production", "prod"}


def _extract_responses_output_text(response_json: Dict[str, Any]) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: List[str] = []
    for output_item in response_json.get("output") or []:
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content") or []:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") in {"output_text", "text"}:
                text_val = content_item.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    parts.append(text_val.strip())
    return "\n".join(parts).strip()


def _strip_json_fences(content: str) -> str:
    text = (content or "").strip()
    if not text.startswith("```"):
        return text
    text = text.strip("`")
    first_newline = text.find("\n")
    if first_newline != -1 and text[:first_newline].lower().startswith("json"):
        text = text[first_newline + 1 :]
    return text.strip()


def _parse_smart_chat_output(output: Any) -> Dict[str, Any]:
    if isinstance(output, dict):
        return output
    text = _strip_json_fences(str(output or "").strip())
    if not text:
        raise RuntimeError("OpenAI returned no output text.")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"message": text, "status": "ok", "actions": [], "cards": [], "refresh": {"queue": False, "recent_files": False}}


def _api_error_response(message: str, exc: Exception) -> Dict[str, Any]:
    detail = str(exc) or exc.__class__.__name__
    _debug_log("fallback used", input_message=message, fallback_used=True, error=detail)
    record_workspace_action(
        actor=SOURCE,
        action_type="smart_chat_api_error",
        status="error",
        requested_message=message,
        output_json={"error": detail[:1000], "fallback_used": True},
    )
    if _is_development_mode():
        response_message = f"Smart Chat API error: {detail}"
    else:
        response_message = "Smart Chat is temporarily unavailable. Try again in a moment."
    return {
        "message": response_message,
        "status": "error",
        "actions": [],
        "cards": [],
        "refresh": {"queue": False, "recent_files": False},
        "fallback_used": True,
    }


def platform_overview_payload() -> Dict[str, str]:
    return {
        "status": "ok",
        "overview": (
            "This platform extracts glass orders, lets operators review and approve them, "
            "then moves approved orders through Workspace, Processing, Labels, and recent production files."
        ),
    }


def workspace_help_payload() -> Dict[str, str]:
    return {
        "status": "ok",
        "help": (
            "Workspace is the production staging page. Needs Review holds orders that still need attention. "
            "Approved / Ready to Process contains orders the Factory Assistant can send through isolated Processing and Labels workflows. "
            "Processing adds approved orders to an isolated Workspace sheet, applies Danko rounding and dimension grouping, then exports the real Processing PDF. "
            "Labels are created from that Processing result and exported as the real Labels PDF. Recent Production Files shows generated Processing PDFs and Labels PDFs."
        ),
    }


def workspace_queue_summary() -> Dict[str, Any]:
    queue = service_get_workspace_queue()
    groups = queue.get("groups") or {}
    counts = queue.get("counts") or {}
    return {
        "status": "ok",
        "counts": counts,
        "sample_ready_orders": [
            _summarize_queue_card(item, "approved_ready")
            for item in (groups.get("approved_ready") or [])[:5]
        ],
    }


def _send_to_factory_response(message: str) -> Dict[str, Any]:
    return {
        "message": "I can explain it, but production actions run through Factory Assistant. Use the Factory Assistant above or click Send to Factory Assistant.",
        "status": "ok",
        "actions": [{
            "label": "Send to Factory Assistant",
            "kind": "send_to_factory",
            "payload": {"message": message},
        }],
        "cards": [],
        "refresh": {"queue": False, "recent_files": False},
    }


def _normalize_smart_chat_response(response: Dict[str, Any], original_message: str) -> Dict[str, Any]:
    if not isinstance(response, dict):
        response = {"message": str(response or ""), "status": "ok"}
    response.setdefault("message", "")
    response.setdefault("status", "ok")
    response.setdefault("actions", [])
    response.setdefault("cards", [])
    response.setdefault("refresh", {"queue": False, "recent_files": False})
    if PRODUCTION_ACTION_RE.search(original_message):
        response["message"] = "I can explain it, but production actions run through Factory Assistant. Use the Factory Assistant above or click Send to Factory Assistant."
        if not any(isinstance(action, dict) and action.get("kind") == "send_to_factory" for action in response.get("actions") or []):
            response["actions"] = [{
                "label": "Send to Factory Assistant",
                "kind": "send_to_factory",
                "payload": {"message": original_message},
            }]
    return response


def _build_tools(function_tool: Any, _RunContextWrapper: Any = None) -> List[Any]:
    @function_tool(strict_mode=False)
    def get_current_datetime() -> Dict[str, str]:
        """Return current local/server date, day, time, and timezone."""
        result = current_datetime_summary()
        record_workspace_action(
            actor=SOURCE,
            action_type="smart_chat_tool_called",
            status="success",
            tool_name="get_current_datetime",
            output_json=result,
        )
        return result

    @function_tool(strict_mode=False)
    def get_platform_overview() -> Dict[str, str]:
        """Return a safe high-level explanation of the platform."""
        result = platform_overview_payload()
        record_workspace_action(actor=SOURCE, action_type="smart_chat_tool_called", status="success", tool_name="get_platform_overview")
        return result

    @function_tool(strict_mode=False)
    def get_workspace_help() -> Dict[str, str]:
        """Return a safe explanation of the Workspace page and workflow."""
        result = workspace_help_payload()
        record_workspace_action(actor=SOURCE, action_type="smart_chat_tool_called", status="success", tool_name="get_workspace_help")
        return result

    @function_tool(strict_mode=False)
    def get_workspace_queue_summary() -> Dict[str, Any]:
        """Return read-only Workspace queue counts and a small approved-order sample."""
        result = workspace_queue_summary()
        record_workspace_action(
            actor=SOURCE,
            action_type="smart_chat_tool_called",
            status="success",
            tool_name="get_workspace_queue_summary",
            output_json={"counts": result.get("counts")},
        )
        return result

    return [
        get_current_datetime,
        get_platform_overview,
        get_workspace_help,
        get_workspace_queue_summary,
    ]


def _create_agent() -> Any:
    Agent, _RunConfig, RunContextWrapper, _Runner, function_tool, error = _sdk_imports()
    if error:
        raise RuntimeError(f"OpenAI Agents SDK is not available: {error}") from error
    return Agent(
        name="Smart Chat",
        instructions=SMART_CHAT_INSTRUCTIONS,
        model=smart_chat_model(),
        tools=_build_tools(function_tool, RunContextWrapper),
    )


def _should_use_sdk() -> bool:
    engine = (os.getenv("WORKSPACE_CHAT_ENGINE") or os.getenv("WORKSPACE_AGENT_ENGINE") or "agents_sdk").strip().lower()
    if engine in {"responses", "responses_api", "openai_responses"}:
        return False
    if engine not in {"agents_sdk", "sdk", "openai_agents"}:
        return False
    return _sdk_imports()[-1] is None


def _responses_api_context(message: str, safe_context: Dict[str, Any]) -> Dict[str, Any]:
    context_payload: Dict[str, Any] = {
        "message": message,
        "workspace_context": _safe_context_summary(safe_context),
        "current_datetime": current_datetime_summary(),
        "platform_overview": platform_overview_payload()["overview"],
    }
    if WORKSPACE_HELP_RE.search(message or ""):
        context_payload["workspace_help"] = workspace_help_payload()["help"]
    if "what should i do next" in (message or "").lower():
        context_payload["workspace_queue_summary"] = workspace_queue_summary()
    return context_payload


def _run_responses_api(message: str, safe_context: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("missing OPENAI_API_KEY")

    model = smart_chat_model()
    _debug_log("OpenAI call started", input_message=message, model=model, engine="responses_api")
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SMART_CHAT_INSTRUCTIONS}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(_responses_api_context(message, safe_context), ensure_ascii=False, default=str),
                    }
                ],
            },
        ],
    }
    try:
        response = httpx.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
    except Exception as exc:
        raise RuntimeError(f"Responses API request failed: {exc}") from exc
    if response.status_code >= 400:
        detail = response.text.strip() or response.reason_phrase
        raise RuntimeError(f"Responses API error ({response.status_code}): {detail}")
    output_text = _extract_responses_output_text(response.json())
    parsed = _parse_smart_chat_output(output_text)
    _debug_log(
        "OpenAI call succeeded",
        input_message=message,
        model=model,
        response_preview=str(parsed.get("message") or "")[:200],
        fallback_used=False,
    )
    return parsed


def run_workspace_smart_chat(
    message: str,
    *,
    context: Optional[Dict[str, Any]] = None,
    requested_by: str = SOURCE,
) -> Dict[str, Any]:
    safe_context = context if isinstance(context, dict) else {}
    _debug_log("route hit /api/agent/smart-chat", input_message=message)
    record_workspace_action(
        actor=requested_by,
        action_type="smart_chat_message_received",
        status="received",
        requested_message=message,
        input_json={
            "context_summary": {
                "current_page": safe_context.get("current_page"),
                "selected_count": len(safe_context.get("selected_orders") or []),
                "visible_count": len(safe_context.get("visible_orders") or []),
            },
        },
    )
    if PRODUCTION_ACTION_RE.search(message or ""):
        response = _normalize_smart_chat_response(_send_to_factory_response(message), message)
        _debug_log(
            "production action redirected",
            input_message=message,
            response_preview=str(response.get("message") or "")[:200],
            fallback_used=False,
        )
        return response

    model = smart_chat_model()

    try:
        if _should_use_sdk():
            Agent, RunConfig, _RunContextWrapper, Runner, _function_tool, error = _sdk_imports()
            if error:
                raise RuntimeError(f"OpenAI Agents SDK is not available: {error}") from error
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("missing OPENAI_API_KEY")
            _debug_log("OpenAI call started", input_message=message, model=model, engine="agents_sdk")
            agent = _create_agent()
            run_config = RunConfig(
                workflow_name=WORKFLOW_NAME,
                group_id=str(safe_context.get("workspace_session_id") or safe_context.get("session_id") or "workspace-smart-chat"),
                trace_metadata={
                    "current_page": str(safe_context.get("current_page") or "Workspace"),
                    "conversation_type": "smart_chat",
                    "source": SOURCE,
                    "action_type": "smart_chat",
                    "model": model,
                },
            )
            input_payload = {
                "message": message,
                "context": _safe_context_summary(safe_context),
            }
            result = Runner.run_sync(
                agent,
                json.dumps(input_payload, ensure_ascii=False, default=str),
                context={"message": message, "workspace_context": safe_context, "requested_by": requested_by},
                max_turns=6,
                run_config=run_config,
            )
            response = _normalize_smart_chat_response(_parse_smart_chat_output(getattr(result, "final_output", None)), message)
        else:
            response = _normalize_smart_chat_response(_run_responses_api(message, safe_context), message)
        _debug_log(
            "OpenAI call succeeded",
            input_message=message,
            model=model,
            response_preview=str(response.get("message") or "")[:200],
            fallback_used=False,
        )
        record_workspace_action(
            actor=requested_by,
            action_type="smart_chat_response_completed",
            status=response.get("status") or "ok",
            requested_message=message,
            output_json={"message": response.get("message"), "actions": response.get("actions"), "fallback_used": False},
        )
        return response
    except Exception as exc:
        LOGGER.exception("Smart Chat API error for message %r", message)
        return _api_error_response(message, exc)
