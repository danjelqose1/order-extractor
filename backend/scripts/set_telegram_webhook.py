from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env", override=True)


def main() -> int:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    public_backend_url = (os.getenv("PUBLIC_BACKEND_URL") or "").strip().rstrip("/")
    secret = (os.getenv("TELEGRAM_WEBHOOK_SECRET") or "").strip()

    if not token:
        print("TELEGRAM_BOT_TOKEN is not set", file=sys.stderr)
        return 2
    if not public_backend_url:
        print("PUBLIC_BACKEND_URL is not set", file=sys.stderr)
        return 2

    payload = {"url": f"{public_backend_url}/webhook/telegram"}
    if secret:
        payload["secret_token"] = secret

    response = httpx.post(
        f"https://api.telegram.org/bot{token}/setWebhook",
        json=payload,
        timeout=30.0,
    )
    try:
        body = response.json()
    except Exception:
        body = {"ok": False, "description": response.text}

    if response.status_code >= 400 or not body.get("ok"):
        print(f"Failed to set Telegram webhook: {body.get('description') or response.status_code}", file=sys.stderr)
        return 1

    print(f"Telegram webhook set to {payload['url']}")
    if secret:
        print("Telegram webhook secret token configured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
