"""Small, bounded client for the documented Agents API beta HTTP contract.

The application key is used only on requests to api.openai.com. It is never
included in agent instructions or environment configuration. No operation is
retried here: callers reconcile uncertain outcomes against persisted state.

Contract checked 2026-10-06:
https://developers.openai.com/api/docs/guides/agents-api/quickstart
https://developers.openai.com/api/docs/guides/agents-api/sessions
https://developers.openai.com/api/docs/guides/agents-api/tools/computer-use
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx


API_BASE = "https://api.openai.com/v1"
MAX_RESPONSE_BYTES = 12 * 1024 * 1024
MAX_LIST_PAGES = 20
REQUEST_DEADLINE_SECONDS = 60
_RESOURCE_ID = re.compile(r"[A-Za-z0-9_-]{1,200}\Z")


class ProviderError(Exception):
    """Safe public error; never carries an upstream body or credential."""

    def __init__(
        self, status: int, code: str, message: str, *, outcome_unknown: bool = False
    ) -> None:
        self.status = status
        self.code = code
        self.message = message
        self.outcome_unknown = outcome_unknown
        super().__init__(message)


def _resource_id(value: str) -> str:
    if not isinstance(value, str) or not _RESOURCE_ID.fullmatch(value):
        raise ProviderError(400, "invalid_resource_id", "The agent resource ID is invalid.")
    return value


def _status_error(status: int, *, mutation: bool) -> ProviderError:
    if status == 401:
        code, message = "authentication_required", "The configured OpenAI API key was rejected. Check the server's OPENAI_API_KEY."
    elif status == 403:
        code, message = "permission_required", "OpenAI denied access. The project needs Agents API access and api.agents.read, api.agents.write, and api.responses.write permissions."
    elif status == 404:
        code, message = "resource_unavailable", "The OpenAI agent resource or API is unavailable to this project."
    elif status == 409:
        code, message = "conflict", "The agent state changed. Refresh the saved session before continuing."
    elif status == 429:
        code, message = "rate_limited", "OpenAI rejected the request because of a rate, usage, or billing limit. Check the project's limits."
    elif status in (400, 422):
        code, message = "invalid_request", "OpenAI rejected the agent configuration or request. Review the Factory Agent setup documentation."
    elif 300 <= status < 400:
        code, message = "redirect_rejected", "OpenAI returned an unexpected redirect; it was not followed."
    else:
        code, message = "provider_unavailable", "The OpenAI Agents API is temporarily unavailable. Refresh the saved session before trying again."
    return ProviderError(status, code, message, outcome_unknown=mutation and status >= 500)


class FactoryAgentProvider:
    """Non-streaming transport; saved sessions/items/turns are canonical state.

    Listing methods return the complete bounded list, not just the first page.
    Metadata filtering for lost-create reconciliation is done by the caller.
    A single session create must never be retried blindly. Idempotency-Key is
    documented for input events, but not for session creation.
    """

    def __init__(
        self,
        api_key: str,
        project: str | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip() or any(not 32 <= ord(c) < 127 for c in api_key):
            raise ProviderError(503, "setup_required", "Set OPENAI_API_KEY on the backend to enable Factory Agent.")
        headers = {
            "Authorization": "Bearer " + api_key.strip(),
            "OpenAI-Beta": "agents=v1",
            "Accept": "application/json",
        }
        if project:
            if not isinstance(project, str) or not _RESOURCE_ID.fullmatch(project):
                raise ProviderError(503, "setup_required", "OPENAI_PROJECT_ID must be a valid OpenAI project ID.")
            headers["OpenAI-Project"] = project
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(45.0, connect=15.0, read=45.0, write=30.0, pool=10.0),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "FactoryAgentProvider":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        byte_limit: int = MAX_RESPONSE_BYTES,
    ) -> dict[str, Any]:
        headers = {}
        if idempotency_key is not None:
            if not isinstance(idempotency_key, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", idempotency_key):
                raise ProviderError(400, "invalid_idempotency_key", "The task submission ID is invalid.")
            headers["Idempotency-Key"] = idempotency_key
        mutation = method != "GET"
        try:
            async with asyncio.timeout(REQUEST_DEADLINE_SECONDS):
                async with self._client.stream(
                    method, API_BASE + path, json=payload, params=params, headers=headers
                ) as response:
                    if not 200 <= response.status_code < 300:
                        # Deliberately do not read/return error bodies: they can echo
                        # prompts, headers, credentials or upstream internal details.
                        raise _status_error(response.status_code, mutation=mutation)
                    declared_length = response.headers.get("Content-Length", "")
                    if declared_length.isdigit() and int(declared_length) > byte_limit:
                        raise ProviderError(502, "response_too_large", "The saved agent response exceeds the display limit.", outcome_unknown=mutation)
                    chunks = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                        if len(chunks) + len(chunk) > byte_limit:
                            raise ProviderError(502, "response_too_large", "The saved agent response exceeds the display limit.", outcome_unknown=mutation)
                        chunks.extend(chunk)
                    if not chunks and response.status_code in (202, 204):
                        return {}
                    try:
                        data = json.loads(chunks)
                    except (ValueError, UnicodeError, RecursionError):
                        raise ProviderError(502, "invalid_response", "OpenAI returned an unreadable agent response.", outcome_unknown=mutation) from None
                    if not isinstance(data, dict):
                        raise ProviderError(502, "invalid_response", "OpenAI returned an unexpected agent response.", outcome_unknown=mutation)
                    return data
        except (httpx.TimeoutException, TimeoutError):
            raise ProviderError(504, "provider_timeout", "The OpenAI request timed out. Its outcome must be checked before submitting more work.", outcome_unknown=mutation) from None
        except httpx.HTTPError:
            raise ProviderError(502, "provider_connection_error", "The OpenAI connection failed. Refresh the saved session before submitting more work.", outcome_unknown=mutation) from None

    async def _list(self, path: str, *, order: str = "asc") -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        after: str | None = None
        seen_cursors: set[str] = set()
        remaining = MAX_RESPONSE_BYTES
        for _ in range(MAX_LIST_PAGES):
            params: dict[str, Any] = {"order": order, "limit": 100}
            if after:
                params["after"] = after
            page = await self._request("GET", path, params=params, byte_limit=remaining)
            remaining -= len(json.dumps(page, separators=(",", ":")).encode("utf-8"))
            data = page.get("data")
            if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
                raise ProviderError(502, "invalid_response", "OpenAI returned an invalid history page.")
            results.extend(data)
            if page.get("has_more") is False:
                return results
            after = page.get("last_id")
            if page.get("has_more") is not True or not isinstance(after, str) or not after or after in seen_cursors or not data:
                raise ProviderError(502, "invalid_response", "OpenAI returned an invalid history cursor.")
            seen_cursors.add(after)
            if remaining <= 0:
                raise ProviderError(502, "response_too_large", "The saved agent history exceeds the display limit.")
        raise ProviderError(502, "history_limit", "The agent history exceeds the supported page limit; no partial history was presented as complete.")

    async def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("stream"):
            raise ProviderError(400, "invalid_request", "Factory Agent uses persisted session polling.")
        return await self._request("POST", "/agents/sessions", payload=payload)

    async def get_session(self, session_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/agents/sessions/{_resource_id(session_id)}")

    async def list_sessions(self) -> list[dict[str, Any]]:
        return await self._list("/agents/sessions", order="desc")

    async def get_environment(self, environment_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/agents/environments/{_resource_id(environment_id)}")

    async def list_turns(self, session_id: str) -> list[dict[str, Any]]:
        return await self._list(f"/agents/sessions/{_resource_id(session_id)}/turns", order="desc")

    async def list_items(self, session_id: str) -> list[dict[str, Any]]:
        return await self._list(f"/agents/sessions/{_resource_id(session_id)}/items")

    async def send_events(
        self, session_id: str, events: list[dict[str, Any]], idempotency_key: str | None = None
    ) -> dict[str, Any]:
        return await self._request(
            "POST", f"/agents/sessions/{_resource_id(session_id)}/events",
            payload={"events": events}, idempotency_key=idempotency_key,
        )

    async def delete_session(self, session_id: str) -> dict[str, Any]:
        session_id = _resource_id(session_id)
        result = await self._request("DELETE", f"/agents/sessions/{session_id}")
        # The documented 200 response confirms public resource removal, not
        # completion of asynchronous physical sandbox cleanup. An empty 204 or
        # another 2xx object cannot establish deletion of this exact session.
        if (result.get("id") != session_id or result.get("deleted") is not True
                or result.get("object") != "agent.session.deleted"):
            raise ProviderError(
                502, "invalid_response",
                "OpenAI did not confirm deletion of this agent session. Cleanup must be checked again.",
                outcome_unknown=True,
            )
        return result
