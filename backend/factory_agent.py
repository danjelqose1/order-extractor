"""Isolated Factory Agent control plane. No production database/tools are imported.

OpenAI owns all browser compute. This process only authenticates operators,
persists control state and services the explicitly read-only fixture function.
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Literal
import uuid

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from factory_agent_fixture import (
    FIXTURE_BROWSER_URL, FIXTURE_ORDER_ID, READ_TOOL, call_read_tool,
    environment_files, environment_setup_commands, load_order, workflow_instructions,
)
from factory_agent_provider import FactoryAgentProvider, ProviderError


TERMINAL = frozenset({"completed", "failed", "cancelled", "timed_out", "setup_required"})
MAX_SCREENSHOT = 3_000_000
MAX_RECORDS = 1000
MAX_RETAINED_RESULTS = 20


def enabled():
    return os.getenv("ENABLE_FACTORY_AGENT", "false").strip().lower() == "true"


def resolve_access_key(existing_app_key):
    """Reuse legacy authentication when enabled; otherwise isolate Beta access.

    This never sets APP_KEY or changes authorization on existing factory routes.
    A new dedicated secret must be at least 32 printable non-space characters.
    The OpenAI API key is never an operator credential or a fallback.
    """
    if existing_app_key:
        return existing_app_key
    dedicated = os.getenv("FACTORY_AGENT_ACCESS_KEY", "")
    if (32 <= len(dedicated) <= 4096
            and all(33 <= ord(char) < 127 for char in dedicated)
            and not hmac.compare_digest(dedicated.encode(), os.getenv("OPENAI_API_KEY", "").encode())):
        return dedicated
    return None


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def bounded_env(name, default, low, high):
    try:
        return min(high, max(low, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


class StartTask(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    request_id: uuid.UUID
    order_id: Literal["fixture:factory-agent-001"]
    message: str = Field(min_length=1, max_length=4000)


class StateStore:
    """Separate bounded control journal; never opens or migrates orders.db.

    An OS lock prevents two workers from owning remote tasks. SQLite protects
    committed request identities across restart, including retired tombstones.
    """
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = (directory / "worker.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError("Factory Agent needs one application worker sharing its state directory.") from None
        try:
            self.db = sqlite3.connect(directory / "sessions.sqlite3")
            os.chmod(directory / "sessions.sqlite3", 0o600)
            self.db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, owner TEXT NOT NULL, request_id TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(owner, request_id))")
            self.db.commit()
        except Exception:
            if hasattr(self, "db"):
                self.db.close()
            self.lock.close()
            raise

    def load(self):
        records = {}
        for identity, payload in self.db.execute("SELECT id,payload FROM sessions"):
            row = json.loads(payload)
            required = {"owner", "request_id", "input_hash", "order_id", "message", "created_ts", "created_at",
                        "deadline", "deadline_at", "status", "phase", "remote_session_id", "turn_id", "activity",
                        "result_text", "screenshot", "error", "cleanup_status", "stop_requested", "timeout_requested",
                        "input_attempted", "handled_actions", "retired"}
            if (not isinstance(row, dict) or row.get("id") != identity or not required.issubset(row)
                    or row["order_id"] != FIXTURE_ORDER_ID or not isinstance(row["owner"], str)
                    or not isinstance(row["activity"], list) or not isinstance(row["handled_actions"], list)
                    or row["status"] not in TERMINAL | {"creating", "running", "stopping", "cleanup_required"}
                    or row["cleanup_status"] not in {"pending", "required", "deleted"}
                    or any(type(row[key]) not in (int, float) or not math.isfinite(row[key]) for key in ("created_ts", "deadline"))):
                raise ValueError("Invalid Factory Agent journal record")
            records[identity] = row
            if len(records) > MAX_RECORDS:
                raise ValueError("Factory Agent journal exceeds its bound")
        return records

    def save(self, row):
        payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        with self.db:
            self.db.execute("INSERT INTO sessions VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                            (row["id"], row["owner"], row["request_id"], payload))

    def close(self):
        self.db.close()
        self.lock.close()


class FactoryAgentService:
    def __init__(self, app_key_getter, *, provider_factory=None, directory=None, poll_seconds=2):
        self.app_key_getter = lambda: resolve_access_key(app_key_getter())
        self.provider_factory = provider_factory or (lambda: FactoryAgentProvider(os.getenv("OPENAI_API_KEY", ""), os.getenv("OPENAI_PROJECT_ID") or None))
        self.directory = directory
        self.poll_seconds = poll_seconds
        self.store = None
        self.provider = None
        self.records = {}
        self.tasks = {}
        self.start_error = None
        self.start_missing = None
        self.closing = False
        self.lock = asyncio.Lock()
        self.janitor = None

    def configuration(self):
        missing = []
        if not self.app_key_getter():
            missing.append("FACTORY_AGENT_ACCESS_KEY")
        if not os.getenv("OPENAI_API_KEY", "").strip():
            missing.append("OPENAI_API_KEY")
        if self.start_error:
            missing.append(self.start_missing or "FACTORY_AGENT_STATE_DIR (writable durable directory, single worker)")
        order = load_order(FIXTURE_ORDER_ID)
        return {"enabled": enabled(), "ready": enabled() and not missing,
                "state": "setup_required" if missing else "ready", "missing": missing,
                "error": self.start_error, "mode": "fixture", "auth": "app_key",
                "orders": [{"id": FIXTURE_ORDER_ID, "label": order["order_number"] + " · isolated fixture"}],
                "limits": {"runtime_seconds": bounded_env("FACTORY_AGENT_RUNTIME_SECONDS", 180, 30, 600),
                           "max_concurrency": bounded_env("FACTORY_AGENT_MAX_CONCURRENCY", 1, 1, 2)},
                "required_permissions": ["api.agents.read", "api.agents.write", "api.responses.write"]}

    async def start(self):
        async with self.lock:
            if self.store or not enabled() or self.configuration()["missing"]:
                return
            try:
                directory = self.directory or os.getenv("FACTORY_AGENT_STATE_DIR") or str(Path(os.getenv("DB_DIR", "data")) / "factory-agent")
                self.store = StateStore(directory)
                self.records = self.store.load()
                self.provider = self.provider_factory()
                self.closing = False
                self.prune()
                for row in self.records.values():
                    if not row.get("retired") and (row["status"] not in TERMINAL or row.get("cleanup_status") != "deleted"):
                        self.spawn(row, recovery=True)
                self.janitor = asyncio.create_task(self.housekeeping())
            except (OSError, RuntimeError, sqlite3.Error, ProviderError, ValueError, KeyError, TypeError) as exc:
                self.start_error = exc.message if isinstance(exc, ProviderError) else "Factory Agent state is unavailable or another worker owns it. Check its private state directory and run one worker."
                if isinstance(exc, ProviderError):
                    self.start_missing = "OPENAI_API_KEY / OPENAI_PROJECT_ID (valid server configuration)"
                if self.store:
                    self.store.close()
                    self.store = None
                self.provider = None

    def save(self, row):
        self.store.save(row)

    def public(self, row, summary=False):
        if summary:
            return {key: row.get(key) for key in ("id", "request_id", "order_id", "created_at", "deadline_at", "status", "cleanup_status", "error", "retired")}
        keys = ("id", "request_id", "order_id", "message", "created_at", "deadline_at", "status",
                "activity", "result_text", "screenshot", "error", "remote_session_id", "turn_id",
                "cleanup_status", "retired")
        return {key: row.get(key) for key in keys}

    def owned(self, session_id, owner):
        row = self.records.get(session_id)
        if not row or not hmac.compare_digest(row["owner"], owner):
            raise HTTPException(404, "Factory Agent session not found.")
        return row

    def activity(self, row, identity, title, status="completed", kind="control"):
        record = {"id": identity, "type": kind, "title": str(title)[:1500], "status": str(status)[:80]}
        previous = next((i for i, item in enumerate(row["activity"]) if item["id"] == identity), None)
        if previous is None:
            row["activity"].append(record)
        else:
            row["activity"][previous] = record
        row["activity"] = row["activity"][-200:]

    def spawn(self, row, recovery=False):
        if row["id"] not in self.tasks or self.tasks[row["id"]].done():
            self.tasks[row["id"]] = asyncio.create_task(self.run(row, recovery=recovery))

    async def create(self, owner, payload):
        await self.start()
        config = self.configuration()
        if not config["ready"] or not self.store:
            raise HTTPException(503, {"code": "setup_required", "missing": config["missing"], "message": config["error"] or "Configure the server before starting Factory Agent."})
        digest = hashlib.sha256(json.dumps({"order_id": payload.order_id, "message": payload.message}, sort_keys=True).encode()).hexdigest()
        async with self.lock:
            for row in self.records.values():
                if row["owner"] == owner and row["request_id"] == str(payload.request_id):
                    if row["input_hash"] != digest:
                        raise HTTPException(409, "This request ID already belongs to a different task.")
                    return self.public(row)
            # Unknown remote outcomes reserve a slot until explicitly cleaned up.
            occupied = sum(r["cleanup_status"] != "deleted" for r in self.records.values())
            if occupied >= config["limits"]["max_concurrency"]:
                raise HTTPException(409, "A task is running or awaiting cleanup. Stop or clean it up before starting another.")
            now = time.time()
            if sum(r["created_ts"] > now - 3600 for r in self.records.values()) >= 12:
                raise HTTPException(429, "Factory Agent is limited to 12 task starts per hour.")
            if len(self.records) >= MAX_RECORDS:
                raise HTTPException(503, "Factory Agent control journal is full. Archive it after confirming all remote sessions are deleted.")
            deadline = now + config["limits"]["runtime_seconds"]
            row = dict(id=str(uuid.uuid4()), request_id=str(payload.request_id), owner=owner,
                       input_hash=digest, order_id=payload.order_id, message=payload.message,
                       created_ts=now, created_at=iso(now), deadline=deadline, deadline_at=iso(deadline),
                       status="creating", phase="reserved", remote_session_id=None, turn_id=None,
                       activity=[], result_text="", screenshot=None, error=None, cleanup_status="pending",
                       stop_requested=False, timeout_requested=False, input_attempted=False,
                       handled_actions=[], retired=False)
            self.save(row)
            self.records[row["id"]] = row
            self.spawn(row)
            return self.public(row)

    def session_payload(self, row):
        return {"agent": {"model": os.getenv("FACTORY_AGENT_MODEL") or "gpt-6.1-sol",
                          "instructions": workflow_instructions(), "multi_agent": {"enabled": False},
                          "tools": [{"type": "computer_use", "include_screenshots": True}, READ_TOOL]},
                "environment": {"type": "openai_hosted", "desktop": {"enabled": True},
                                "network": {"access": "disabled"}, "files": environment_files(),
                                "setup_commands": environment_setup_commands()},
                "metadata": {"factory_agent_request_id": row["id"], "application": "order-extractor-factory-beta"}}

    def safe_diagnostic(self, value):
        """Keep provider failure text useful without persisting server credentials."""
        if not isinstance(value, str):
            return ""
        text = value[:16384]
        secrets = {self.app_key_getter(), os.getenv("OPENAI_API_KEY"),
                   os.getenv("APP_KEY"), os.getenv("FACTORY_AGENT_ACCESS_KEY")}
        for secret in sorted((s for s in secrets if s), key=len, reverse=True):
            text = text.replace(secret, "[redacted]")
        text = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", text)
        text = re.sub(r"(?i)bearer\s+[^\s\"']+", "Bearer [redacted]", text)
        return " ".join(text.split())[:2000]

    async def preserve_failure(self, row, session, fallback):
        # A failed session's saved error is available before deletion. Capture it
        # first; optional SSE diagnostics must never hold up bounded cleanup.
        message = self.safe_diagnostic(session.get("error"))
        row.update(status="failed", error=fallback + (" " + message if message else ""))
        self.save(row)
        diagnostics = await self.provider.read_failure_diagnostics(row["remote_session_id"])
        details = []
        for entry in diagnostics:
            detail = self.safe_diagnostic(" ".join(entry.get(k, "") for k in ("code", "message") if isinstance(entry.get(k), str)))
            if detail and detail not in details:
                details.append(detail)
        if details:
            row["error"] = self.safe_diagnostic(row["error"] + " " + " | ".join(details))
            self.save(row)

    async def recover_creation(self, row):
        sessions = await self.provider.list_sessions()
        matches = [s for s in sessions if s.get("metadata", {}).get("factory_agent_request_id") == row["id"]]
        if len(matches) == 1:
            row["remote_session_id"] = matches[0]["id"]
            self.save(row)
            return matches[0]
        row["status"] = "cleanup_required"
        row["error"] = "Session creation has an unknown outcome. No task was resubmitted. Retry cleanup or inspect the Agents dashboard using this request ID: " + row["id"]
        self.save(row)
        return None

    async def run(self, row, recovery=False):
        """Wall-clock watchdog also bounds paginated/slow API requests.

        Expiry interrupts the observer and then explicitly cancels/deletes the
        remote session. Observer interruption alone is never reported as Stop.
        """
        try:
            async with asyncio.timeout(max(0.01, row["deadline"] - time.time())):
                await self._run(row, recovery)
        except TimeoutError:
            if row["status"] not in TERMINAL:
                row.update(stop_requested=True, timeout_requested=True, status="stopping")
            self.save(row)
            try:
                async with asyncio.timeout(30):
                    if not row["remote_session_id"]:
                        await self.recover_creation(row)
                    if row["remote_session_id"]:
                        await self.cleanup(row, stopping=True)
            except (TimeoutError, ProviderError):
                row.update(status="cleanup_required", cleanup_status="required", error="Runtime expired but remote cancellation/deletion could not be confirmed. Retry cleanup or inspect the Agents dashboard.")
                self.save(row)

    async def _run(self, row, recovery=False):
        try:
            if row["remote_session_id"]:
                session = await self.provider.get_session(row["remote_session_id"])
            elif recovery or row["phase"] == "creating":
                session = await self.recover_creation(row)
                if not session:
                    return
            else:
                if row["stop_requested"]:
                    row.update(status="cancelled", cleanup_status="deleted")
                    self.save(row)
                    return
                row["phase"] = "creating"
                self.save(row)  # Persist before the non-idempotent remote create.
                try:
                    session = await self.provider.create_session(self.session_payload(row))
                except ProviderError as exc:
                    if exc.outcome_unknown:
                        session = await self.recover_creation(row)
                        if not session:
                            return
                    else:
                        row.update(status="setup_required" if exc.status in {401, 403, 404} else "failed", error=exc.message, cleanup_status="deleted")
                        self.save(row)
                        return
                row["remote_session_id"] = session["id"]
                row["phase"] = "created"
                self.activity(row, "session-created", "OpenAI session created; hosted browser setup is pending.")
                self.save(row)
            if row["status"] in TERMINAL or row["status"] == "cleanup_required":
                await self.cleanup(row)
                return
            failures = 0
            last_cancel = 0
            while True:
                now = time.time()
                if now >= row["deadline"]:
                    row.update(stop_requested=True, timeout_requested=True)
                if row["stop_requested"]:
                    row["status"] = "stopping"
                    if now - last_cancel > 5:
                        try:
                            await self.provider.send_events(row["remote_session_id"], [{"type": "agent.session.input.cancel"}])
                            self.activity(row, "cancel-requested", "OpenAI accepted a task cancellation request. Waiting for the remote outcome.")
                            last_cancel = now
                        except ProviderError as exc:
                            row["error"] = "Cancellation could not be confirmed. " + exc.message
                        self.save(row)
                    row.setdefault("stop_started", now)
                    if now - row["stop_started"] >= 30:
                        await self.cleanup(row, stopping=True)
                        return
                try:
                    session = await self.provider.get_session(row["remote_session_id"])
                    if session.get("status") not in {"idle", "in_progress", "requires_action", "failed"}:
                        raise RuntimeError("Unknown remote session state.")
                    if session.get("status") == "failed":
                        await self.preserve_failure(row, session, "The OpenAI session failed.")
                        await self.cleanup(row)
                        return
                    if not row["input_attempted"] and not row["stop_requested"]:
                        environment = session.get("environment") or {}
                        if not environment.get("id"):
                            raise RuntimeError("OpenAI did not return a hosted environment ID.")
                        environment = await self.provider.get_environment(environment["id"])
                        self.activity(row, "environment", "Hosted browser: " + str(environment.get("status", "unknown")), environment.get("status", "unknown"))
                        if environment.get("status") == "failed":
                            await self.preserve_failure(row, session, "Hosted fixture/browser setup failed. No task was submitted.")
                            await self.cleanup(row)
                            return
                        if environment.get("status") == "connected":
                            # Never automatically repeat a message after an unknown delivery.
                            row.update(input_attempted=True, phase="input_attempted", status="running")
                            self.save(row)
                            task = ("Read only the selected synthetic order " + FIXTURE_ORDER_ID + ". "
                                    "Verify it in the browser at " + FIXTURE_BROWSER_URL + " and with get_selected_order. "
                                    "Report client, glass types, all dimensions/units, quantities, index numbers and positions; flag ambiguity. "
                                    "Operator request (subject to the packaged read-only workflow):\n" + row["message"])
                            await self.provider.send_events(row["remote_session_id"], [{"type": "agent.session.input.message", "input": [{"role": "user", "content": [{"type": "input_text", "text": task}]}]}], idempotency_key=row["id"])
                            self.activity(row, "input-accepted", "OpenAI accepted the task.")
                    if row["input_attempted"]:
                        turns = await self.provider.list_turns(row["remote_session_id"])
                        roots = [turn for turn in turns if not turn.get("subagent_id")]
                        if len(roots) > 1:
                            raise RuntimeError("Unexpected additional root turn; task stopped for review.")
                        turn = roots[0] if roots else None
                        if turn:
                            if row["turn_id"] and row["turn_id"] != turn["id"]:
                                raise RuntimeError("The remote turn identity changed; task stopped for review.")
                            row["turn_id"] = turn["id"]
                            items = await self.provider.list_items(row["remote_session_id"])
                            self.collect_items(row, items)
                            outcome = turn.get("status")
                            self.activity(row, "turn", "Agent turn: " + str(outcome), outcome)
                            if outcome in {"completed", "cancelled", "failed"}:
                                row["status"] = "timed_out" if row["timeout_requested"] else outcome
                                if outcome == "failed":
                                    row["error"] = "The agent turn failed. Any partial result remains visible."
                                elif outcome == "completed" and not row["result_text"]:
                                    row["error"] = "The turn completed without a saved text result. No order-reading success has been verified."
                                else:
                                    row["error"] = None
                                self.save(row)
                                await self.cleanup(row)
                                return
                        if not row["stop_requested"]:
                            await self.handle_actions(row, session.get("required_actions", []))
                    if row["stop_requested"] and not row["input_attempted"]:
                        # A cancelled setup has no turn; deletion confirms resource removal.
                        await self.cleanup(row, stopping=True)
                        return
                    failures = 0
                    self.save(row)
                except ProviderError as exc:
                    failures += 1
                    row["error"] = exc.message + " Reconnecting to the same task; it will not be resubmitted."
                    self.save(row)
                    if failures >= 5 or exc.status in {401, 403, 404}:
                        row["stop_requested"] = True
                        row.setdefault("stop_started", time.time())
                await asyncio.sleep(min(self.poll_seconds * max(1, failures), 10))
        except asyncio.CancelledError:
            raise
        except Exception:
            # Do not include upstream messages, request bodies, or credentials in logs/UI.
            row.update(status="cleanup_required", error="Factory Agent encountered an unexpected response. No task will be resubmitted; remote cleanup is required.")
            self.save(row)
            with suppress(Exception):
                await self.cleanup(row, stopping=True)

    def collect_items(self, row, items):
        messages = []
        for item in items:
            if item.get("turn_id") != row["turn_id"]:
                continue
            kind = item.get("type")
            if kind == "computer_use_call":
                self.activity(row, item.get("id", "browser"), item.get("title") or "Browser activity", item.get("status", "unknown"), kind)
                output = item.get("output")
                for part in output if isinstance(output, list) else [output]:
                    if isinstance(part, dict) and part.get("type") == "computer_screenshot":
                        value = part.get("image_url", "")
                        if isinstance(value, str) and len(value) <= MAX_SCREENSHOT and re.fullmatch(r"data:image/(?:jpeg|png);base64,[A-Za-z0-9+/=\r\n]+", value):
                            row["screenshot"] = value
            elif kind == "message" and item.get("role") == "assistant":
                # Only saved assistant output, never fabricated from fixture fields.
                text = "\n".join(p.get("text", "") for p in item.get("content", []) if p.get("type") == "output_text" and isinstance(p.get("text"), str))
                if text:
                    messages.append(text)
        row["result_text"] = "\n\n".join(messages)[-60000:]

    async def handle_actions(self, row, actions):
        for action in actions[:25]:
            if row["stop_requested"] or time.time() >= row["deadline"]:
                return
            kind = action.get("type")
            identity = action.get("request_id") or action.get("call_id")
            if not identity or identity in row["handled_actions"]:
                continue
            if len(row["handled_actions"]) >= 100:
                raise RuntimeError("Agent exceeded the tool/approval request bound.")
            if kind == "function_call":
                event = {"type": "agent.session.input.tool_result", "turn_id": action["turn_id"], "call_id": action["call_id"]}
                try:
                    result = call_read_tool(action.get("name"), action.get("arguments"), row["order_id"])
                    event.update(success=True, output=json.dumps(result, ensure_ascii=False))
                except ValueError:
                    event.update(success=False, error="Denied: only get_selected_order({}) for the selected fixture is available.")
                title = "Read-only tool: " + str(action.get("name", "unknown")) + (" returned the fixture." if event["success"] else " denied.")
            elif kind == "computer_use_approval_request":
                request = action.get("request") or {}
                if request.get("type") == "browser_origin_access":
                    approved = request.get("origin") == "http://127.0.0.1:8765"
                    response = {"type": "browser_origin_access", "decision": "approve" if approved else "deny"}
                    title = "Browser access " + ("approved for the isolated fixture." if approved else "denied: origin is outside the isolated fixture.")
                elif request.get("type") == "browser_authentication":
                    response = {"type": "browser_authentication", "action": "cancel"}
                    title = "Browser sign-in cancelled. Credentials are not permitted in this read-only test."
                else:
                    raise RuntimeError("Unknown browser approval requires cancellation.")
                event = {"type": "agent.session.input.computer_use_approval_request_result", "request_id": action["request_id"], "response": response}
            else:
                # Never connect another environment, execute arbitrary functions or grant access.
                raise RuntimeError("Unsupported required action.")
            await self.provider.send_events(row["remote_session_id"], [event], idempotency_key=row["id"] + ":" + identity)
            row["handled_actions"].append(identity)
            self.activity(row, identity, title)
            self.save(row)

    async def cleanup(self, row, stopping=False):
        if row["status"] in TERMINAL:
            row["final_outcome"] = row["status"]
        if not row["remote_session_id"]:
            row.update(status="cleanup_required", error="Remote session identity is unresolved; inspect the Agents dashboard before starting another task.")
            self.save(row)
            return False
        if stopping or row["status"] not in TERMINAL:
            with suppress(ProviderError):
                await self.provider.send_events(row["remote_session_id"], [{"type": "agent.session.input.cancel"}])
        for attempt in range(3):
            try:
                await self.provider.delete_session(row["remote_session_id"])
                row["cleanup_status"] = "deleted"
                if row["status"] not in TERMINAL:
                    row["status"] = row.get("final_outcome") or ("timed_out" if row["timeout_requested"] else "cancelled")
                self.activity(row, "cleanup", "OpenAI confirmed session deletion; hosted environment cleanup is asynchronous.")
                self.save(row)
                self.prune()
                return True
            except ProviderError as exc:
                if exc.status == 404:
                    row["cleanup_status"] = "deleted"
                    if row["status"] not in TERMINAL:
                        row["status"] = row.get("final_outcome") or ("timed_out" if row["timeout_requested"] else "cancelled")
                    self.save(row)
                    self.prune()
                    return True
                if attempt < 2:
                    await asyncio.sleep(min(self.poll_seconds * (attempt + 1), 4))
        row["cleanup_status"] = "required"
        row["status"] = "cleanup_required"
        row["error"] = "Remote deletion could not be confirmed. Retry cleanup; the concurrency slot remains reserved."
        self.save(row)
        return False

    async def stop(self, row):
        if row["cleanup_status"] == "deleted":
            return self.public(row)
        row["stop_requested"] = True
        if row["status"] in TERMINAL:
            row["final_outcome"] = row["status"]
        row["status"] = "stopping"
        self.save(row)
        if row["remote_session_id"]:
            try:
                async with asyncio.timeout(10):
                    await self.provider.send_events(row["remote_session_id"], [{"type": "agent.session.input.cancel"}])
                    self.activity(row, "cancel-requested", "OpenAI accepted a task cancellation request. Waiting for the remote outcome.")
                    self.save(row)
            except (TimeoutError, ProviderError):
                row["error"] = "Cancellation is not confirmed. The backend will retry and check the same remote task."
                self.save(row)
        self.spawn(row, recovery=True)
        return self.public(row)

    async def remove(self, row):
        if row["cleanup_status"] != "deleted":
            await self.stop(row)
            raise HTTPException(409, "Remote cleanup has been requested. Wait for confirmation, then remove the local result.")
        self.retire(row)
        return {"deleted": True, "id": row["id"]}

    def retire(self, row):
        row.update(retired=True, message="", result_text="", screenshot=None, activity=[], error=None)
        self.save(row)

    def prune(self):
        finished = sorted((r for r in self.records.values() if r["cleanup_status"] == "deleted" and not r["retired"]), key=lambda r: r["created_ts"], reverse=True)
        for index, row in enumerate(finished):
            if index >= MAX_RETAINED_RESULTS or row["created_ts"] < time.time() - 86400:
                self.retire(row)

    async def housekeeping(self):
        while True:
            await asyncio.sleep(30)
            self.prune()

    async def close(self):
        if not self.store:
            return
        self.closing = True
        if self.janitor:
            self.janitor.cancel()
            with suppress(asyncio.CancelledError):
                await self.janitor
        for row in self.records.values():
            if row["cleanup_status"] != "deleted":
                row["stop_requested"] = True
                self.save(row)
        # Let an in-flight create persist its ID before shutting down. If this
        # deadline expires, the journal reconciles the same metadata on restart.
        active = [task for task in self.tasks.values() if not task.done()]
        if active:
            _, pending = await asyncio.wait(active, timeout=15)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        for row in self.records.values():
            if row["cleanup_status"] != "deleted" and row["remote_session_id"]:
                try:
                    async with asyncio.timeout(10):
                        await self.cleanup(row, stopping=True)
                except (TimeoutError, ProviderError):
                    row.update(status="cleanup_required", cleanup_status="required", error="Server shutdown interrupted remote cleanup. Reconnect after restart or use the Agents dashboard.")
                    self.save(row)
        await self.provider.aclose()
        self.store.close()
        self.store = None


class FactoryAgentBoundary:
    """Bound bodies before validation and prevent caching authenticated output."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").startswith("/api/factory-agent"):
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > 16384:
                return await JSONResponse({"detail": "Factory Agent request exceeds 16 KB."}, 413, headers={"Cache-Control": "no-store"})(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        async def private_send(message):
            if message["type"] == "http.response.start":
                message["headers"] = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"] + [(b"cache-control", b"no-store")]
            await send(message)

        return await self.app(scope, replay, private_send)


def install_factory_agent(app: FastAPI, app_key_getter, origins_getter):
    service = FactoryAgentService(app_key_getter)
    router = APIRouter(prefix="/api/factory-agent", tags=["Factory Agent Beta"])

    async def authorized(request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        if not enabled():
            raise HTTPException(404, "Factory Agent Beta is disabled.")
        expected = service.app_key_getter()
        if not expected:
            raise HTTPException(503, {"code": "setup_required", "missing": ["FACTORY_AGENT_ACCESS_KEY"], "message": "Set a dedicated FACTORY_AGENT_ACCESS_KEY (32+ printable non-space characters), or reuse an already configured APP_KEY. Never use the OpenAI API key as an access key."})
        supplied = request.headers.get("x-app-key", "")
        if len(supplied) > 4096 or not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise HTTPException(401, "Application key required for Factory Agent.")
        origin = request.headers.get("origin")
        if origin and (origin == "null" or (origin not in origins_getter() and not re.fullmatch(r"http://(?:localhost|127\.0\.0\.1)(?::\d+)?", origin))):
            raise HTTPException(403, "Factory Agent must be opened from the platform.")
        # This shared operator key identifies a group, not an individual login.
        owner = hashlib.sha256(("factory-agent\0" + expected).encode()).hexdigest()
        await service.start()
        return owner

    @router.get("/config")
    async def config(owner=Depends(authorized)):
        return service.configuration()

    @router.get("/sessions")
    async def sessions(owner=Depends(authorized)):
        return {"sessions": [service.public(row, summary=True) for row in sorted(service.records.values(), key=lambda r: r["created_ts"], reverse=True) if row["owner"] == owner and not row["retired"]][:50]}

    @router.post("/sessions", status_code=202)
    async def create(payload: StartTask, owner=Depends(authorized)):
        return await service.create(owner, payload)

    @router.get("/sessions/{session_id}")
    async def get(session_id: uuid.UUID, owner=Depends(authorized)):
        return service.public(service.owned(str(session_id), owner))

    @router.post("/sessions/{session_id}/stop")
    async def stop(session_id: uuid.UUID, owner=Depends(authorized)):
        return await service.stop(service.owned(str(session_id), owner))

    @router.delete("/sessions/{session_id}")
    async def remove(session_id: uuid.UUID, owner=Depends(authorized)):
        return await service.remove(service.owned(str(session_id), owner))

    app.include_router(router)
    app.add_middleware(FactoryAgentBoundary)
    app.add_event_handler("startup", service.start)
    app.add_event_handler("shutdown", service.close)
    app.state.factory_agent = service
    return service
