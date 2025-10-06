from __future__ import annotations
from typing import Dict, Any, List
import os, json, re
from openai import OpenAI

SYSTEM_PROMPT = '''You are an expert production planner for a glass factory.
You convert pasted order text (copied from PDFs) into STRICT JSON rows for manufacturing.

OUTPUT FORMAT:
- Your reply MUST be a single JSON object that matches the provided JSON schema exactly. No extra keys, no commentary.

GENERAL INPUT NOTES:
- The user input is raw pasted text copied from PDF files (possibly multiple pages).
- Do NOT rely on PDF layout or page structure—treat the input purely as text.
- Expect inconsistent spacing, wrapped lines, and multiple orders; normalize and extract everything.
- Always continue extraction until the very end of the pasted text.

EXTRACTION RULES (follow ALL):
1) Multiple orders & pages
   - The input may contain multiple orders and page headers/footers.
   - Extract ALL item rows across all pages.
   - Each row MUST carry the correct `order_number` for its section.

2) order_number normalization
   - Normalize forms like `R - 25-0864` → `R-25-0864`.
   - When a new document/order header appears, use that order number until another appears.

3) type propagation
   - If a block header declares a glass type (e.g., `2 VETRI 33.1F + 14 + 33.1 LOWE C.CALDO 28mm`), repeat that exact `type` for each subsequent row in that block until a new type block appears.
   - Preserve original casing and symbols exactly as seen (do NOT reformat).

4) position
   - Remove any leading order prefix (e.g., `R-25-0864/1-1` → `1-1`).
   - Preserve any suffix after the short code (e.g., keep `/marco`, `/bontempelli` → `1-1/marco`).
   - Only strip stray symbols or noise; never remove legitimate names.

5) dimension
   - Return as `WIDTHxHEIGHT` in mm with NO spaces (e.g., `520x1168`).
   - If a dimension is missing, unclear, or split such that you cannot be certain, set `dimension` to the empty string `""`. Do NOT guess or back-calculate from area.

6) quantity
   - If a quantity is clearly tied to the same row/line, use it.
   - Otherwise default to 1 per row.

7) area (m²)
   - If the row shows an explicit area (often with comma decimal like `1,570`), convert the comma to dot → `1.570`.
   - If no explicit area is shown, compute from dimension: `(W * H) / 1_000_000` and round to **3 decimals**.
   - Use a dot as the decimal separator in JSON.

8) ignore noise
   - Ignore page headers/footers, dates, timestamps, and summary lines like `m2 57 29,780`, `Totale 61 31,660`.

9) warnings
   - If you detect missing dimensions, odd wraps (numbers on next line), or any ambiguity/normalization, add a short human-readable note to `warnings`.

10) decimal commas
   - Treat `,` as the decimal separator in Italian-style numbers; convert to `.` in JSON. Remove thousands separators.

11) Wrapped splits
   - If a position is split like `22-` on one line and `1/<name>` on the next, treat it as `22-1` and ignore the `/name` suffix.

12) Joined tokens
   - If a dimension and order/position appear stuck together (e.g., `682 x 2270R-25-0767/35-1`), separate them into their correct fields before extracting.

Return ONLY the JSON.
'''

JSON_SCHEMA = {
    "name": "order_extraction",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "order_number": {"type": "string"},
                        "type": {"type": "string"},
                        "dimension": {"type": "string"},
                        "position": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "area": {"type": "number"},
                    },
                    "required": ["order_number","type","dimension","position","quantity","area"]
                }
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["rows", "warnings"]
    },
    "strict": True,
}

_FURNITURE_PATTERNS = [
    re.compile(r'^\*{3}.*KELI ALBANIA.*$', re.IGNORECASE),
    re.compile(r'^ORDINE DI VETRO$', re.IGNORECASE),
    re.compile(r'^CLIENTE .*$', re.IGNORECASE),
    re.compile(r'^OGGETTO .*$', re.IGNORECASE),
    re.compile(r'^DOCUMENTO CORRELATO .*$', re.IGNORECASE),
    re.compile(r'^\d{1,2}\.\d{1,2}\.\d{2,4} .* \[\d+\.\d+/\d+\.\d+\]\s*\d+\s*/\s*\d+$', re.IGNORECASE),
    re.compile(r'^m2\s+\d+\s+[\d,.]+$', re.IGNORECASE),
    re.compile(r'^Totale\s+\d+\s+[\d,.]+$', re.IGNORECASE),
]

_DIM_ONLY_RE = re.compile(r'^\d{3,4}\s*[xX×]\s*\d{3,4}$')
_ORDER_PREFIX_RE = re.compile(r'^R-?\d{2}-\d{4}/', re.IGNORECASE)
_POSITION_WRAP_RE = re.compile(r'/\d{1,3}-$')
_NEXT_POSITION_FRAGMENT_RE = re.compile(r'^\d+(?:/[A-Za-zÀ-ÿ.\-]+)?$')
_POSITION_LINE_RE = re.compile(r'^R-?\d{2}-\d{4}/\d{1,3}-\d{1,2}$', re.IGNORECASE)
_NAME_SUFFIX_RE = re.compile(r'^/[A-Za-zÀ-ÿ.\-]+$')
_DECIMAL_COMMA_RE = re.compile(r'(\d),(?=\d{2,3}\b)')
_THOUSANDS_CHAIN_RE = re.compile(r'(\d)\.(\d{3})\.(\d)')
_DIM_ORDER_STUCK_RE = re.compile(r'(?P<dim>\d{3,4}\s*[xX×]\s*\d{3,4})(?=R-?\d{2}-\d{4}/)', re.IGNORECASE)
_TYPE_LINE_RE = re.compile(r'^\d+\s+VETRI\s+.*mm$', re.IGNORECASE)


def normalize_and_stitch(raw_text: str) -> str:
    if not raw_text:
        return ""

    text = raw_text.replace('\r\n', '\n').replace('\r', '\n')

    lines: List[str] = []
    for raw_line in text.split('\n'):
        stripped = re.sub(r'[ \t]+', ' ', raw_line.strip())
        if stripped:
            lines.append(stripped)

    filtered_lines: List[str] = []
    for line in lines:
        if any(pat.search(line) for pat in _FURNITURE_PATTERNS):
            continue
        filtered_lines.append(line)

    lines = filtered_lines

    for _ in range(3):
        changed = False
        new_lines: List[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                if _DIM_ONLY_RE.match(line) and _ORDER_PREFIX_RE.match(nxt):
                    new_lines.append(f"{line} {nxt}")
                    i += 2
                    changed = True
                    continue
                if _POSITION_WRAP_RE.search(line) and _NEXT_POSITION_FRAGMENT_RE.match(nxt):
                    new_lines.append(f"{line}{nxt}")
                    i += 2
                    changed = True
                    continue
                if _POSITION_LINE_RE.match(line) and _NAME_SUFFIX_RE.match(nxt):
                    new_lines.append(f"{line}{nxt}")
                    i += 2
                    changed = True
                    continue
            new_lines.append(line)
            i += 1
        lines = new_lines
        if not changed:
            break

    stitched = "\n".join(lines)
    stitched = _DECIMAL_COMMA_RE.sub(r'\1.', stitched)

    # Remove thousands separators where they create chained decimals like 1.234.56
    while _THOUSANDS_CHAIN_RE.search(stitched):
        stitched = _THOUSANDS_CHAIN_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}.{m.group(3)}", stitched)

    # Track the last detected type line (for debugging/anchor purposes)
    current_type = ""
    for line in stitched.split('\n'):
        if _TYPE_LINE_RE.match(line):
            current_type = line
    if current_type:
        _ = current_type  # retained to satisfy instructions (anchors for type propagation)

    return stitched


def _insert_dim_breaks(text: str) -> str:
    return _DIM_ORDER_STUCK_RE.sub(lambda m: f"{m.group('dim')}\n", text)


def _prepare_text(raw_text: str) -> str:
    processed = normalize_and_stitch(raw_text)
    processed = _insert_dim_breaks(processed)
    return processed.strip()

def get_client() -> OpenAI:
    """Return an OpenAI client using the OPENAI_API_KEY env var."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    # NOTE: Removed 'proxies' argument for compatibility with new OpenAI SDK on Render
    return OpenAI(api_key=key)

def build_messages(pasted_text: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": pasted_text},
    ]

def call_llm_for_extraction(pasted_text: str) -> Dict[str, Any]:
    client = get_client()
    prepared_text = _prepare_text(pasted_text) or pasted_text.strip()
    messages = build_messages(prepared_text)

    preferred_models = [
        "gpt-4o-2024-08-06",
        "gpt-4o-mini-2024-07-18",
    ]

    completion = None
    model_used = None
    last_err: Exception | None = None

    for model_name in preferred_models:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": JSON_SCHEMA,
                },
                temperature=0.0,
            )
            model_used = model_name
            break  # success
        except Exception as e:
            # Keep trying the next model
            last_err = e
            print(f"⚠️ Model '{model_name}' failed, trying next fallback. Error: {e}")
            continue

    if completion is None:
        # If all attempts failed, surface the final error
        raise RuntimeError(f"All model attempts failed: {last_err}")

    # Parse JSON content and return
    content = completion.choices[0].message.content or "{}"

    # Some models occasionally wrap JSON in fences; strip them defensively
    if content.startswith('```'):
        content = content.strip().strip('`')
        # Remove a leading language hint like ```json
        first_newline = content.find('\n')
        if first_newline != -1 and content[:first_newline].lower().startswith('json'):
            content = content[first_newline+1:]

    data = json.loads(content)

    # Log which model actually produced the result (non-breaking; payload is unchanged)
    try:
        print(f"✅ Extraction completed with: {model_used}")
    except Exception:
        pass

    return data


def _merge_and_dedupe(all_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in all_rows:
        try:
            qty = int(r.get("quantity", 0))
        except Exception:
            qty = 0
        try:
            area_val = float(r.get("area", 0.0) or 0.0)
        except Exception:
            area_val = 0.0
        key = (
            r.get("order_number", ""),
            r.get("type", ""),
            r.get("dimension", ""),
            r.get("position", ""),
            qty,
            round(area_val, 3),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _page_messages(page_text: str, carry: Dict[str, str]) -> List[Dict[str, str]]:
    carry_note = ""
    if carry.get("order_number") or carry.get("glass_type"):
        carry_note = (
            "\n\nCarry-over context from previous page:"
            f"\n- current_order_number: {carry.get('order_number', '(unknown)')}"
            f"\n- current_glass_type: {carry.get('glass_type', '(unknown)')}"
            "\nIf the page continues a block, keep propagating these until a new header appears."
        )
    page_instructions = SYSTEM_PROMPT + carry_note + (
        "\n\nIMPORTANT (page mode): Only extract rows that appear on THIS page. "
        "Do not re-output rows from previous pages. Preserve the same rules."
    )
    return [
        {"role": "system", "content": page_instructions},
        {"role": "user", "content": page_text},
    ]


def _update_carry_from_rows(rows: List[Dict[str, Any]], carry: Dict[str, str]) -> None:
    if not rows:
        return
    last = rows[-1]
    order_number = last.get("order_number")
    glass_type = last.get("type")
    if order_number:
        carry["order_number"] = order_number
    if glass_type:
        carry["glass_type"] = glass_type


def call_llm_for_extraction_multi(pages_text: List[str]) -> Dict[str, Any]:
    client = get_client()
    preferred = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]
    carry: Dict[str, str] = {"order_number": "", "glass_type": ""}
    all_rows: List[Dict[str, Any]] = []
    all_warnings: List[str] = []

    for i, page_text in enumerate(pages_text, start=1):
        prepared_page = _prepare_text(page_text) or page_text.strip()
        if not prepared_page:
            continue

        messages = _page_messages(prepared_page, carry)
        completion = None
        last_err: Exception | None = None
        model_used = None

        for model_name in preferred:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
                    temperature=0.0,
                )
                model_used = model_name
                break
            except Exception as e:
                last_err = e
                print(f"⚠️ Page {i}: model {model_name} failed → {e}")

        if completion is None:
            all_warnings.append(f"Page {i} failed: {last_err}")
            continue

        content = completion.choices[0].message.content or "{}"
        if content.startswith("```"):
            content = content.strip().strip("`")
            first_newline = content.find("\n")
            if first_newline != -1 and content[:first_newline].lower().startswith("json"):
                content = content[first_newline + 1:]

        try:
            data = json.loads(content)
            rows = data.get("rows", []) or []
            warnings = data.get("warnings", []) or []
            all_rows.extend(rows)
            _update_carry_from_rows(rows, carry)
            all_warnings.extend([f"Page {i}: {w}" for w in warnings])
            print(f"✅ Page {i} extracted with {model_used}: {len(rows)} rows")
        except Exception as e:
            all_warnings.append(f"Page {i} JSON parse error: {e}")

    merged = _merge_and_dedupe(all_rows)
    main_order = ""
    for row in merged:
        candidate = (row.get("order_number") or "").strip()
        if candidate:
            main_order = candidate
            break

    return {"order_number": main_order, "rows": merged, "warnings": all_warnings}
