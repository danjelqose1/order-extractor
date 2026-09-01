"""Run the very same functions loaded by the website, without a browser."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


WORKER = Path(__file__).resolve().parents[1] / "workflow_runtime" / "worker.cjs"


def run_workflow(operation: str, **payload):
    # Pass data on stdin, never on a command line; do not inherit application secrets.
    result = subprocess.run(
        [os.getenv("ORDER_EXTRACTOR_NODE_BINARY", "node"), "--max-old-space-size=256", str(WORKER)],
        input=json.dumps({"operation": operation, **payload}, allow_nan=False),
        text=True, capture_output=True, check=True,
        timeout=min(60, max(1, int(os.getenv("ORDER_EXTRACTOR_MCP_TIMEOUT_SECONDS", "30")))),
        env={"PATH": os.environ.get("PATH", ""), "TZ": "Europe/Tirane", "LANG": "en_GB.UTF-8"},
    )
    if len(result.stdout) > 40_000_000:
        raise ValueError("Artifact exceeds the workflow size limit")
    return json.loads(result.stdout)
