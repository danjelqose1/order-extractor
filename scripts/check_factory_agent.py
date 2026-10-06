"""Check Factory Agent setup or perform one explicitly requested bounded smoke run.

Run from repository root using the existing backend virtual environment.
No production database is opened. No credentials are printed or copied.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from dotenv import load_dotenv

if os.getenv("ORDER_EXTRACTOR_LOAD_DOTENV", "true") == "true":
    load_dotenv(ROOT / "backend" / ".env", override=True)

from factory_agent import FactoryAgentService, StartTask
from factory_agent_fixture import FIXTURE_ORDER_ID, load_order


def verify_report(row):
    """Conservative smoke assertions; a human still reviews the actual report."""
    text = row.get("result_text", "")
    source = load_order(FIXTURE_ORDER_ID)
    expected = [source["client"]]
    expected.extend(str(value) for item in source["items"] for key, value in item.items()
                    if key != "ambiguities" and value is not None)
    missing = sorted(set(value for value in expected if value not in text))
    browser_items = [item for item in row.get("activity", []) if item.get("type") == "computer_use_call"]
    browser_read = (any(item.get("status") == "completed" for item in browser_items)
                    and not any(item.get("status") in {"failed", "incomplete"} for item in browser_items))
    tool_read = any(item.get("title") == "Read-only tool: get_selected_order returned the fixture." for item in row.get("activity", []))
    ambiguous = "missing" in text.lower() and any(word in text.lower() for word in ("uncertain", "ambig", "alternative"))
    return {"all_source_values_present": not missing, "missing_source_values": missing,
            "browser_activity_completed": browser_read, "read_tool_completed": tool_read,
            "ambiguity_flagged": ambiguous,
            "passed": row.get("status") == "completed" and row.get("cleanup_status") == "deleted" and not missing and browser_read and tool_read and ambiguous}


async def main(live):
    service = FactoryAgentService(lambda: os.getenv("APP_KEY"))
    configuration = service.configuration()
    print(json.dumps(configuration, indent=2))  # Public, non-secret configuration only.
    if not configuration["enabled"]:
        print("Setup required: ENABLE_FACTORY_AGENT=true. No remote resource was created.")
        return 2
    if not configuration["ready"]:
        print("Setup required. No remote resource was created.")
        return 2
    if not live:
        print("Local configuration is present. API permissions and hosted browser access are unverified; use --live for one bounded test.")
        return 0
    try:
        await service.start()
        owner = hashlib.sha256(("factory-agent\0" + service.app_key_getter()).encode()).hexdigest()
        result = await service.create(owner, StartTask(request_id=uuid.uuid4(), order_id=FIXTURE_ORDER_ID,
            message="Read the selected test order; report every source field and ambiguity, and identify actual browser/tool evidence."))
        session_id = result["id"]
        print("Local session ID:", session_id)
        deadline = time.monotonic() + configuration["limits"]["runtime_seconds"] + 40
        last = None
        while time.monotonic() < deadline:
            row = service.records[session_id]
            state = (row["status"], row["cleanup_status"])
            if state != last:
                print("Session status:", state[0], "Cleanup:", state[1], flush=True)
                last = state
            if service.tasks[session_id].done():
                visible = service.public(row)
                visible["screenshot"] = "Available in authenticated session detail" if visible["screenshot"] else None
                print(json.dumps(visible, indent=2, ensure_ascii=False))
                checks = verify_report(row)
                print(json.dumps({"smoke_assertions": checks}, indent=2, ensure_ascii=False))
                return 0 if checks["passed"] else 1
            await asyncio.sleep(1)
        print("No final outcome was confirmed within the local deadline. Cleanup will be attempted; inspect the saved session.")
        return 1
    finally:
        await service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Create one hosted fixture session (uses API/container billing), run it, then cancel/delete as needed.")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.live)))
