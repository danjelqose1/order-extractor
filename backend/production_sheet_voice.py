"""Short-lived transcription transport. Dictation cannot perform sheet actions."""
from __future__ import annotations

import json
import re
import secrets
import threading
import time
from collections import deque

import httpx
from pydantic import BaseModel, ConfigDict, Field

class VoiceOffer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sdp: str = Field(min_length=20, max_length=100000)


class VoiceOwnership(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,200}$")
    token: str = Field(min_length=32, max_length=100)


TRANSCRIPTION_CONTEXT = (
    "Spoken requests about a glass-factory production sheet: layout, columns, "
    "font size, line spacing and glass-type headings. The speaker may switch languages."
)

# Only opaque ownership tokens are held in process memory; no audio/transcript storage.
_lock = threading.Lock()
_sessions: dict[str, tuple[str, float]] = {}
_starts: dict[str, deque] = {}
_pending = 0


def _reserve(peer: str):
    global _pending
    now = time.monotonic()
    with _lock:
        for sid, (_, expiry) in list(_sessions.items()):
            if expiry < now:
                del _sessions[sid]
        for key, recent in list(_starts.items()):
            while recent and recent[0] < now - 3600:
                recent.popleft()
            if not recent:
                del _starts[key]
        recent = _starts.setdefault(peer, deque())
        if len(recent) >= 30 or len(_sessions) + _pending >= 4:
            raise ValueError("Dictation is busy. Stop an existing microphone session or try again shortly.")
        recent.append(now)
        _pending += 1


def owned(request: VoiceOwnership):
    with _lock:
        record = _sessions.get(request.session_id)
        if not record or record[1] < time.monotonic() or not secrets.compare_digest(record[0], request.token):
            raise PermissionError("This dictation session has ended. Start dictation again.")


def create_session(client, request: VoiceOffer, peer: str):
    global _pending
    if not request.sdp.startswith("v=0") or "m=audio" not in request.sdp:
        raise ValueError("The microphone connection offer is invalid.")
    _reserve(peer)
    try:
        session = {"type": "transcription", "audio": {"input": {
            "transcription": {"model": "gpt-live-transcribe", "delay": "medium",
                              "prompt": TRANSCRIPTION_CONTEXT},
            "turn_detection": None}}}
        result = client.with_options(timeout=httpx.Timeout(45, connect=15), max_retries=0).post(
            "/realtime/calls", cast_to=httpx.Response,
            files={"sdp": (None, request.sdp),
                   "session": (None, json.dumps(session), "application/json")},
            options={"headers": {"Content-Type": "multipart/form-data"}})
        match = re.search(r"/realtime/calls/([A-Za-z0-9_-]{1,200})$", result.headers.get("Location", ""))
        if not match:
            raise RuntimeError("Dictation returned an incomplete connection.")
        sid, sdp = match.group(1), result.text
        if not sdp.startswith("v=0") or "m=audio" not in sdp:
            _hangup(client, sid)
            raise RuntimeError("Dictation returned an incomplete connection.")
        token = secrets.token_urlsafe(32)
        with _lock:
            _sessions[sid] = (token, time.monotonic() + 3600)
        return {"session_id": sid, "token": token, "sdp": sdp, "model": "gpt-live-transcribe"}
    finally:
        with _lock:
            _pending -= 1


def _hangup(client, sid):
    try:
        client.with_options(timeout=10, max_retries=0).post(
            f"/realtime/calls/{sid}/hangup", cast_to=type(None))
    except Exception as exc:
        # A confirmed-close session can already have disappeared upstream.
        if getattr(exc, "status_code", None) not in (404, 410):
            raise

def close_session(client, request: VoiceOwnership):
    owned(request)
    _hangup(client, request.session_id)
    with _lock:
        _sessions.pop(request.session_id, None)
    return {"ok": True}
