"""Read-only fixture browser, copied into an OpenAI-hosted sandbox only.

No app imports, database, filesystem routes, proxy, authentication, or outbound
connections exist here. The fixture is loaded once before serving. Setup starts
this process on loopback and verifies /healthz. Session deletion owns cleanup.
Run manually with: python3 /workspace/factory-agent/fixture_server.py
"""

from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


HOST = "127.0.0.1"
PORT = 8765
CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
    "form-action 'none'; frame-ancestors 'none'; sandbox"
)


def escaped(value: Any) -> str:
    """Escape every source value; missing values remain distinct from zero."""
    return html.escape("Missing" if value is None else str(value), quote=True)


def render_order(order: dict[str, Any]) -> bytes:
    rows = []
    for item in order["items"]:
        cells = "".join(
            f"<td>{escaped(item.get(key))}</td>"
            for key in ("index_number", "position", "glass_type", "width", "height", "quantity")
        )
        issues = item.get("ambiguities", [])
        cells += "<td>" + ("<br>".join(escaped(issue) for issue in issues) or "None recorded") + "</td>"
        rows.append(f"<tr>{cells}</tr>")
    notes = "".join(f"<li>{escaped(note)}</li>" for note in order.get("notes", []))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Factory Agent — isolated test order</title>
<style>
:root{{color-scheme:light dark;font-family:system-ui,sans-serif}}body{{margin:0;padding:clamp(16px,4vw,48px);line-height:1.5;background:#101723;color:#f0f4fa}}main{{max-width:1280px;margin:auto}}h1{{font-size:clamp(1.4rem,3vw,2rem);margin:.5rem 0}}.badge{{display:inline-block;border:1px solid #7bd5c6;border-radius:999px;color:#aff2e5;padding:4px 12px;font-weight:700}}.notice{{padding:12px 16px;border-left:4px solid #f4ca77;background:#232333}}.table-wrap{{overflow:auto;border:1px solid #536077;border-radius:8px}}table{{border-collapse:collapse;min-width:880px;width:100%;font-variant-numeric:tabular-nums}}th,td{{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #536077}}th{{background:#243047}}td:nth-child(3){{min-width:160px}}td:last-child{{min-width:220px}}dt{{font-weight:700}}dd{{margin:0 0 12px}}code{{overflow-wrap:anywhere}}li{{margin:8px 0}}
</style></head><body><main>
<span class="badge">ISOLATED FIXTURE · READ ONLY</span>
<h1>{escaped(order.get('order_number'))}</h1>
<p class="notice">Synthetic test data. No customer PDF was checked. No production connection or actions are available.</p>
<dl><dt>Client</dt><dd>{escaped(order.get('client'))}</dd><dt>Selected order ID</dt><dd><code>{escaped(order.get('order_id'))}</code></dd><dt>Dimension unit</dt><dd>{escaped(order.get('dimension_unit'))} — source text preserved</dd></dl>
<div class="table-wrap"><table><caption>All source rows in their original order</caption><thead><tr><th scope="col">Index number</th><th scope="col">Position</th><th scope="col">Glass type</th><th scope="col">Width</th><th scope="col">Height</th><th scope="col">Quantity</th><th scope="col">Ambiguity / review note</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<ul>{notes}</ul></main></body></html>"""
    return page.encode("utf-8")


def make_handler(order: dict[str, Any]) -> type[BaseHTTPRequestHandler]:
    # Materialize immutable response bytes before accepting any request.
    order_bytes = json.dumps(order, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    routes = {
        "/order": ("text/html; charset=utf-8", render_order(order)),
        "/order.json": ("application/json; charset=utf-8", order_bytes),
        "/healthz": ("application/json; charset=utf-8", b'{"status":"ok","fixture":true,"read_only":true}'),
    }

    class FixtureHandler(BaseHTTPRequestHandler):
        server_version = "FactoryFixture/1"
        sys_version = ""

        def _respond(self, status: int, body: bytes, content_type: str = "text/plain; charset=utf-8") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
            self.send_header("Connection", "close")
            if status == 405:
                self.send_header("Allow", "GET, HEAD")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
            self.close_connection = True

        def _read(self) -> None:
            expected_host = f"{HOST}:{self.server.server_port}"
            if self.headers.get("Host") != expected_host:
                self._respond(403, b"Only the loopback fixture origin is allowed.")
                return
            # No URL decoding, query parsing, path resolution, or file access.
            route = routes.get(self.path)
            if route is None:
                self._respond(404, b"Fixture route not found.")
                return
            self._respond(200, route[1], route[0])

        def do_GET(self) -> None:
            self._read()

        def do_HEAD(self) -> None:
            self._read()

        def _deny_mutation(self) -> None:
            self._respond(405, b"Read-only fixture. Only GET and HEAD are allowed.")

        do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_TRACE = do_CONNECT = _deny_mutation

        def log_message(self, format: str, *args: Any) -> None:
            # Do not echo untrusted request targets or data into task logs.
            pass

    return FixtureHandler


def main() -> None:
    order = json.loads(Path(__file__).with_name("order.json").read_text(encoding="utf-8"))
    if order.get("fixture") is not True or order.get("order_id") != "fixture:factory-agent-001":
        raise RuntimeError("Refusing to serve anything except the isolated fixture")
    with ThreadingHTTPServer((HOST, PORT), make_handler(order)) as server:
        server.daemon_threads = True
        server.serve_forever()


if __name__ == "__main__":
    main()
