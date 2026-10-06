"""Closed, read-only data boundary for the Factory Agent Beta's first test.

This module intentionally does not import the database, factory actions, or app.
Only these reviewed fixture assets enter the hosted sandbox. Credentials never do.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any


FIXTURE_ORDER_ID = "fixture:factory-agent-001"
FIXTURE_BROWSER_URL = "http://127.0.0.1:8765/order"
REMOTE_DIRECTORY = "/workspace/factory-agent"
ASSET_DIRECTORY = Path(__file__).with_name("factory_agent_assets")
ASSET_NAMES = ("SKILL.md", "order.json", "fixture_server.py")
READ_TOOL = {
    "type": "function",
    "name": "get_selected_order",
    "description": "Read the one server-selected isolated synthetic test order. Returns source fields and ambiguity notes. Cannot read production orders or change anything.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}


class FixtureAccessDenied(ValueError):
    """Raised when a caller requests anything outside the read-only fixture."""


def load_order(order_id: str) -> dict[str, Any]:
    if order_id != FIXTURE_ORDER_ID:
        raise FixtureAccessDenied("Only the selected isolated test order is available.")
    # Parsing per read gives the caller a separate snapshot, preserving raw types.
    order = json.loads((ASSET_DIRECTORY / "order.json").read_text(encoding="utf-8"))
    if order.get("order_id") != FIXTURE_ORDER_ID or order.get("fixture") is not True:
        raise FixtureAccessDenied("The isolated fixture package is invalid.")
    return order


def call_read_tool(name: str, args: Any, selected_order_id: str) -> dict[str, Any]:
    if name != "get_selected_order":
        raise FixtureAccessDenied("This tool is unavailable in read-only mode.")
    if not isinstance(args, dict) or args:
        raise FixtureAccessDenied("get_selected_order accepts only an empty argument object.")
    return load_order(selected_order_id)


def environment_files() -> list[dict[str, str]]:
    """Documented Agents API inline files; no traversal or arbitrary upload input.

    Contract: /api/docs/guides/agents-api/environments/openai-hosted and
    /api/docs/guides/agents-api/environments/files, verified 2026-10-06.
    """
    return [
        {
            "type": "inline",
            "path": f"{REMOTE_DIRECTORY}/{name}",
            "data": base64.b64encode((ASSET_DIRECTORY / name).read_bytes()).decode("ascii"),
        }
        for name in ASSET_NAMES
    ]


def environment_setup_commands() -> list[dict[str, str]]:
    """Compatibility helper: fixture startup belongs to the observable first turn.

    Avoid bootstrap commands that can hide fixture failures behind a generic
    environment-connection error. The agent receives the exact command below in
    its packaged workflow; it runs only inside the hosted sandbox, never Render.
    """
    return []


def fixture_start_command() -> str:
    """Return the one reviewed startup command delivered in the workflow file."""
    skill = (ASSET_DIRECTORY / "SKILL.md").read_text(encoding="utf-8")
    if skill.count("```bash\n") != 1:
        raise FixtureAccessDenied("The fixture startup workflow is invalid.")
    command, closing, _ = skill.split("```bash\n", 1)[1].partition("\n```")
    if not closing or not command.strip():
        raise FixtureAccessDenied("The fixture startup workflow is invalid.")
    return command


def workflow_instructions() -> str:
    """Include the actual workflow in agent instructions as well as its file."""
    skill = (ASSET_DIRECTORY / "SKILL.md").read_text(encoding="utf-8")
    return (
        "You are Factory Agent · Beta, operating exclusively in read-only fixture mode. "
        "Read /workspace/factory-agent/SKILL.md before working. "
        "The authoritative packaged workflow is included here so it is always received:\n\n"
        + skill
    )
