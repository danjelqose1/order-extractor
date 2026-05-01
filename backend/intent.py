from __future__ import annotations

import re
from typing import Any, Dict, Optional


CONFIRMATION_RE = re.compile(
    r"^\s*(yes|yep|yeah|continue|continue anyway|proceed|go ahead|do it|ok(?:ay)? continue|cancel|stop|never mind)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
UNSAFE_RE = re.compile(
    r"\b(delete\s+all\s+orders|remove\s+all\s+orders|overwrite\s+(?:approved\s+)?history|overwrite\s+raw|change\s+raw\s+(?:pdf|data)|mutate\s+raw)\b",
    re.IGNORECASE,
)
ACTION_OR_DATA_RE = re.compile(
    r"\b(process|download|open|generate|create|show|list|ready|review|queue|order|orders|label|labels|processing|file|files|warning|warnings)\b",
    re.IGNORECASE,
)
CAPABILITY_RE = re.compile(
    r"\b(what\s+can\s+you\s+do|what\s+are\s+your\s+powers|how\s+can\s+you\s+help|what\s+do\s+you\s+do|help\s+.*\buse\s+this)\b",
    re.IGNORECASE,
)
CASUAL_SIGNAL_RE = re.compile(
    r"\b(hi|hello|hey|thanks?|thank\s+you|good\s+(?:morning|afternoon|evening)|lol|haha|nice|cool|how\s+(?:are|r)\s+you|are\s+you\s+(?:here|there)|with\s+me)\b",
    re.IGNORECASE,
)
DATE_TIME_RE = re.compile(
    r"\b(what\s+(?:day|date|time)|today|tonight|current\s+(?:day|date|time)|is\s+it\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.IGNORECASE,
)
PLATFORM_EXPLAIN_RE = re.compile(r"\b(explain\s+this\s+page|explain\s+the\s+page|what\s+is\s+this\s+page)\b", re.IGNORECASE)


def normalize_message(message: str) -> str:
    text = str(message or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip("?!., ")


def has_pending_action(context: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(context, dict):
        return False
    if str(context.get("latest_pending_action_id") or "").strip():
        return True
    pending = context.get("latest_pending_action")
    return isinstance(pending, dict) and bool(str(pending.get("pending_action_id") or "").strip())


def is_clear_confirmation_response(message: str) -> bool:
    return bool(CONFIRMATION_RE.match(str(message or "")))


def is_obviously_unsafe_request(message: str) -> bool:
    return bool(UNSAFE_RE.search(str(message or "")))


def _smart_chat_redirect(intent: str = "use_smart_chat") -> Dict[str, Any]:
    return {
        "message": "I am mainly for factory workflows. Use Smart Chat below for normal questions.",
        "status": "ok",
        "intent": intent,
    }


def legacy_conversational_response(message: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Small no-SDK fallback for non-action chat.

    The Agents SDK is the primary conversational brain. This helper exists only
    when the SDK cannot run locally, so it deliberately avoids detailed phrase
    routing and never performs production work.
    """
    text = str(message or "").strip()
    if not text:
        return _smart_chat_redirect("casual_chat")
    if is_obviously_unsafe_request(text):
        return {
            "message": "I cannot delete orders or overwrite raw/approved history. I can help review, explain, or safely process approved orders.",
            "status": "blocked",
            "intent": "unsupported_or_blocked",
        }
    if is_clear_confirmation_response(text) and not has_pending_action(context):
        return {
            "message": "Got you, but there is nothing pending right now. Tell me what you want to do next.",
            "status": "ok",
            "intent": "confirmation_response",
        }
    if DATE_TIME_RE.search(text) and not ACTION_OR_DATA_RE.search(text):
        return _smart_chat_redirect("general_question")
    if PLATFORM_EXPLAIN_RE.search(text):
        return _smart_chat_redirect("platform_question")
    if CAPABILITY_RE.search(text):
        return {
            "message": (
                "I run Workspace production workflows: show approved orders, show needs review, process approved orders, "
                "combine selected orders, generate real Processing and Labels files through the existing modules, and reopen recent files."
            ),
            "status": "ok",
            "intent": "capability_question",
        }
    if CASUAL_SIGNAL_RE.search(text) and not ACTION_OR_DATA_RE.search(text):
        return _smart_chat_redirect("casual_chat")
    return None
