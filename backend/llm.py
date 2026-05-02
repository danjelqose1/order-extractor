from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import base64
import json
import os
import re
import time
import httpx
import fitz  # PyMuPDF
from openai import OpenAI
from utils_text import build_signature, parse_declared_totals
from db import find_similar_corrections, bump_correction_hit
from prompts import PROMPTS

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Configure it in Render → Environment.")

SYSTEM_PROMPT = PROMPTS["extraction"]["system"]
PDF_VISUAL_SYSTEM_PROMPT = PROMPTS["extraction"]["pdf_visual_system"]
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gpt-5-mini")


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except Exception:
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        return default
    return value if value >= 0 else default

JSON_SCHEMA = {
    "name": "order_extraction",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "order_number": {"type": "string"},
            "client_name": {"type": "string"},
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
            },
            "confidence": {"type": "number"},
        },
        "required": ["order_number", "client_name", "rows", "warnings", "confidence"]
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
_DIM_AREA_STUCK_RE = re.compile(
    r'(?P<dim>\d{3,4}\s*[xX×]\s*\d{3,4})\s+(?P<area>\d+(?:[.,]\d+)?)\s+(?P<qty>\d+)\s+(?P<area2>\d+(?:[.,]\d+)?)',
    re.IGNORECASE,
)
_TYPE_LINE_RE = re.compile(r'^\d+\s+VETRI\s+.*mm$', re.IGNORECASE)
_AREA_QTY_LINE_RE = re.compile(r'^\s*\d+(?:[.,]\d+)?\s+\d+\s+\d+(?:[.,]\d+)?\s*$', re.IGNORECASE)


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
            if i + 2 < len(lines):
                nxt = lines[i + 1]
                third = lines[i + 2]
                if _DIM_ONLY_RE.match(line) and _POSITION_LINE_RE.match(nxt) and _AREA_QTY_LINE_RE.match(third):
                    new_lines.append(f"{line} {nxt} {third}")
                    i += 3
                    changed = True
                    continue
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
    text = _DIM_ORDER_STUCK_RE.sub(lambda m: f"{m.group('dim')}\n", text)
    text = _DIM_AREA_STUCK_RE.sub(lambda m: f"{m.group('dim')}\n{m.group('area')} {m.group('qty')} {m.group('area2')}", text)
    return text


def _prepare_text(raw_text: str) -> str:
    processed = normalize_and_stitch(raw_text)
    processed = _insert_dim_breaks(processed)
    return processed.strip()


def _strip_json_fences(content: str) -> str:
    text = (content or "").strip()
    if not text.startswith("```"):
        return text
    text = text.strip("`")
    first_newline = text.find("\n")
    if first_newline != -1 and text[:first_newline].lower().startswith("json"):
        text = text[first_newline + 1 :]
    return text.strip()


def _extract_responses_output_text(response_json: Dict[str, Any]) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: List[str] = []
    for output_item in response_json.get("output") or []:
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content") or []:
            if not isinstance(content_item, dict):
                continue
            ctype = content_item.get("type")
            if ctype in {"output_text", "text"}:
                text_val = content_item.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    parts.append(text_val.strip())
    return "\n".join(parts).strip()


def call_llm_for_pdf_base64_visual(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    if not pdf_bytes:
        raise RuntimeError("PDF input is empty.")

    model_name = os.getenv("EXTRACTION_MODEL", EXTRACTION_MODEL)
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    pdf_data_url = f"data:application/pdf;base64,{pdf_b64}"
    user_prompt = (
        "Extract all order rows from the attached PDF. "
        "Read the PDF visually and structurally (tables, lines, layout), not OCR-transcribed text. "
        "Return strict JSON only."
    )
    payload = {
        "model": model_name,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": PDF_VISUAL_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {
                        "type": "input_file",
                        "filename": filename or "upload.pdf",
                        "file_data": pdf_data_url,
                    },
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": JSON_SCHEMA["name"],
                "schema": JSON_SCHEMA["schema"],
                "strict": True,
            }
        },
    }

    timeout = httpx.Timeout(
        timeout=_env_float("EXTRACTION_TOTAL_TIMEOUT_SECONDS", 900.0),
        connect=_env_float("EXTRACTION_CONNECT_TIMEOUT_SECONDS", 30.0),
        read=_env_float("EXTRACTION_READ_TIMEOUT_SECONDS", 900.0),
        write=_env_float("EXTRACTION_WRITE_TIMEOUT_SECONDS", 180.0),
        pool=_env_float("EXTRACTION_POOL_TIMEOUT_SECONDS", 30.0),
    )
    retries = _env_int("EXTRACTION_REQUEST_RETRIES", 1)
    attempt = 0
    response = None

    while True:
        attempt += 1
        try:
            response = httpx.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            break
        except httpx.ReadTimeout as exc:
            if attempt > (retries + 1):
                raise RuntimeError(
                    "Responses API request failed: read timeout while extracting large PDF "
                    f"(attempts={attempt}, read_timeout={timeout.read}s)."
                ) from exc
            time.sleep(min(2.0 * attempt, 5.0))
            continue
        except Exception as exc:
            raise RuntimeError(f"Responses API request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text.strip()
        raise RuntimeError(f"Responses API error ({response.status_code}): {detail}")

    response_json = response.json()
    output_text = _extract_responses_output_text(response_json)
    if not output_text:
        raise RuntimeError("Responses API returned no output text.")

    output_text = _strip_json_fences(output_text)
    parsed_payload = json.loads(output_text)
    client_name = parsed_payload.get("client_name")
    if client_name is None:
        parsed_payload["client_name"] = ""
    confidence = parsed_payload.get("confidence")
    if confidence is None:
        parsed_payload["confidence"] = 0.0

    return {
        "data": parsed_payload,
        "raw": parsed_payload,
        "raw_response": response_json,
        "prepared_text": "",
        "model_used": response_json.get("model") or model_name,
        "applied_corrections": [],
        "corrections": [],
        "declared_units": None,
        "declared_area": None,
        "pdf_data_url": pdf_data_url,
        "output_text": output_text,
    }


def call_llm_for_image_visual(image_bytes: bytes, filename: str, mime_type: str = "image/jpeg") -> Dict[str, Any]:
    if not image_bytes:
        raise RuntimeError("Image input is empty.")

    model_name = os.getenv("EXTRACTION_MODEL", EXTRACTION_MODEL)
    safe_mime = mime_type if str(mime_type or "").startswith("image/") else "image/jpeg"
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    image_data_url = f"data:{safe_mime};base64,{image_b64}"
    user_prompt = (
        "Extract all order rows from the attached order image. "
        "Read the image visually (tables, lines, layout), preserving dimensions, positions, quantities, and glass types. "
        "Return strict JSON only."
    )
    payload = {
        "model": model_name,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": PDF_VISUAL_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                    },
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": JSON_SCHEMA["name"],
                "schema": JSON_SCHEMA["schema"],
                "strict": True,
            }
        },
    }

    timeout = httpx.Timeout(
        timeout=_env_float("EXTRACTION_TOTAL_TIMEOUT_SECONDS", 900.0),
        connect=_env_float("EXTRACTION_CONNECT_TIMEOUT_SECONDS", 30.0),
        read=_env_float("EXTRACTION_READ_TIMEOUT_SECONDS", 900.0),
        write=_env_float("EXTRACTION_WRITE_TIMEOUT_SECONDS", 180.0),
        pool=_env_float("EXTRACTION_POOL_TIMEOUT_SECONDS", 30.0),
    )
    retries = _env_int("EXTRACTION_REQUEST_RETRIES", 1)
    attempt = 0

    while True:
        attempt += 1
        try:
            response = httpx.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            break
        except httpx.ReadTimeout as exc:
            if attempt > (retries + 1):
                raise RuntimeError(
                    "Responses API request failed: read timeout while extracting order image "
                    f"(attempts={attempt}, read_timeout={timeout.read}s)."
                ) from exc
            time.sleep(min(2.0 * attempt, 5.0))
            continue
        except Exception as exc:
            raise RuntimeError(f"Responses API request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text.strip()
        raise RuntimeError(f"Responses API error ({response.status_code}): {detail}")

    response_json = response.json()
    output_text = _extract_responses_output_text(response_json)
    if not output_text:
        raise RuntimeError("Responses API returned no output text.")

    output_text = _strip_json_fences(output_text)
    parsed_payload = json.loads(output_text)
    if parsed_payload.get("client_name") is None:
        parsed_payload["client_name"] = ""
    if parsed_payload.get("confidence") is None:
        parsed_payload["confidence"] = 0.0

    return {
        "data": parsed_payload,
        "raw": parsed_payload,
        "raw_response": response_json,
        "prepared_text": "",
        "model_used": response_json.get("model") or model_name,
        "applied_corrections": [],
        "corrections": [],
        "declared_units": None,
        "declared_area": None,
        "image_data_url": image_data_url,
        "output_text": output_text,
    }

def pdf_to_png_pages(pdf_bytes: bytes, dpi: int = 200) -> List[bytes]:
    """Render each PDF page to PNG bytes (no alpha channel)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        images: List[bytes] = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            images.append(pix.tobytes("png"))
        return images
    finally:
        doc.close()


def ocr_png_with_openai(image_bytes: bytes, model: Optional[str] = None) -> str:
    """OCR a single PNG using OpenAI Vision; return plain text."""
    client = get_client()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"
    system_prompt = "You are a strict OCR engine. Output only the exact text you see. Preserve line breaks; no commentary."
    model = model or os.getenv("OCR_MODEL", "gpt-5-mini")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe all legible text exactly; keep layout line breaks."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


_client: Optional[OpenAI] = None


def _create_openai_client() -> OpenAI:
    try:
        return OpenAI(api_key=API_KEY)
    except Exception as exc:
        raise RuntimeError("Failed to initialize OpenAI client") from exc


_CLIENT_LINE_RE = re.compile(r"^\s*CLIENTE\s+(.+)$", re.IGNORECASE)


def extract_client_name(full_text: str) -> str:
    """Best-effort: pick first 'CLIENTE ...' line text after the keyword."""
    for line in (full_text or "").splitlines():
        match = _CLIENT_LINE_RE.search(line)
        if match:
            name = match.group(1).strip()
            return re.sub(r"\s{2,}.*$", "", name)
    return ""


def get_client() -> OpenAI:
    """Return an OpenAI client using the OPENAI_API_KEY env var."""
    global _client
    if _client is None:
        _client = _create_openai_client()
    return _client

def build_messages(pasted_text: str, corrections: List[Dict[str, Any]]):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for correction in corrections or []:
        if not correction.get("pattern_text") or not correction.get("after_json"):
            continue
        example_input = correction["pattern_text"]
        example_output = correction["after_json"]
        messages.append(
            {
                "role": "user",
                "content": f"EXAMPLE INPUT\n{example_input.strip()[:1800]}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": example_output,
            }
        )
    messages.append({"role": "user", "content": pasted_text})
    return messages

def call_llm_for_extraction(pasted_text: str) -> Dict[str, Any]:
    client = get_client()
    prepared_text = _prepare_text(pasted_text) or pasted_text.strip()
    corrections = find_similar_corrections(prepared_text, top_k=3)
    messages = build_messages(prepared_text, corrections)

    preferred_models = [os.getenv("EXTRACTION_MODEL", EXTRACTION_MODEL)]

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
    content = _strip_json_fences(completion.choices[0].message.content or "{}")

    usage = getattr(completion, "usage", None)
    if usage:
        try:
            print(
                f"tokens prompt={usage.prompt_tokens} completion={usage.completion_tokens} total={usage.total_tokens}"
            )
        except Exception:
            pass

    raw_payload = json.loads(content)
    data = json.loads(content)

    # Log which model actually produced the result (non-breaking; payload is unchanged)
    try:
        print(f"✅ Extraction completed with: {model_used}")
    except Exception:
        pass

    data, applied = _apply_post_corrections(data, prepared_text, corrections)
    declared_units_raw, declared_area_raw = parse_declared_totals(pasted_text)
    declared_units_prepared, declared_area_prepared = parse_declared_totals(prepared_text)
    declared_units = declared_units_prepared if declared_units_prepared is not None else declared_units_raw
    declared_area = declared_area_prepared if declared_area_prepared is not None else declared_area_raw

    return {
        "data": data,
        "raw": raw_payload,
        "prepared_text": prepared_text,
        "model_used": model_used,
        "applied_corrections": applied,
        "corrections": corrections,
        "declared_units": declared_units,
        "declared_area": declared_area,
    }


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


def _apply_post_corrections(
    payload: Dict[str, Any],
    prepared_text: str,
    corrections: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[int]]:
    if not payload.get("rows"):
        return payload, []
    signature = build_signature(prepared_text or "", payload.get("rows"))
    applied: List[int] = []
    for correction in corrections or []:
        if correction.get("pattern_hash") != signature["pattern_hash"]:
            continue
        after_json = correction.get("after_json")
        if not after_json:
            continue
        try:
            parsed = json.loads(after_json)
            if isinstance(parsed, dict) and parsed.get("rows"):
                payload["rows"] = parsed.get("rows")
                applied.append(correction["id"])
                bump_correction_hit(correction["id"])
        except Exception:
            continue
    return payload, applied


def _page_messages(page_text: str, carry: Dict[str, str], corrections: List[Dict[str, Any]]) -> List[Dict[str, str]]:
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
    messages: List[Dict[str, str]] = [{"role": "system", "content": page_instructions}]
    for correction in corrections or []:
        if not correction.get("pattern_text") or not correction.get("after_json"):
            continue
        messages.append(
            {
                "role": "user",
                "content": f"EXAMPLE INPUT\n{correction['pattern_text'].strip()[:1400]}",
            }
        )
        messages.append({"role": "assistant", "content": correction["after_json"]})
    messages.append({"role": "user", "content": page_text})
    return messages


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
    preferred = [os.getenv("EXTRACTION_MODEL", EXTRACTION_MODEL)]
    carry: Dict[str, str] = {"order_number": "", "glass_type": ""}
    all_rows: List[Dict[str, Any]] = []
    all_warnings: List[str] = []
    prepared_segments: List[str] = []
    prepared_pages: List[str] = []

    for page in pages_text:
        processed = (_prepare_text(page) if page else "") or ""
        prepared = processed.strip()
        if not prepared:
            prepared = ((page or "")).strip()
        prepared_pages.append(prepared)
        if prepared:
            prepared_segments.append(prepared)
    prepared_full = "\n\n".join(prepared_segments)
    corrections = find_similar_corrections(prepared_full, top_k=3)
    model_used_global: Optional[str] = None

    for i, prepared_page in enumerate(prepared_pages, start=1):
        if not prepared_page:
            continue

        messages = _page_messages(prepared_page, carry, corrections)
        completion = None
        last_err: Exception | None = None
        model_used = None
        usage_info = None

        for model_name in preferred:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
                    temperature=0.0,
                )
                model_used = model_name
                model_used_global = model_used
                usage_info = getattr(completion, "usage", None)
                break
            except Exception as e:
                last_err = e
                print(f"⚠️ Page {i}: model {model_name} failed → {e}")

        if completion is None:
            all_warnings.append(f"Page {i} failed: {last_err}")
            continue

        content = _strip_json_fences(completion.choices[0].message.content or "{}")

        if usage_info:
            try:
                print(
                    f"tokens page={i} prompt={usage_info.prompt_tokens} completion={usage_info.completion_tokens} total={usage_info.total_tokens}"
                )
            except Exception:
                pass

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

    payload = {"order_number": main_order, "rows": merged, "warnings": all_warnings}
    raw_payload = json.loads(json.dumps(payload))
    payload, applied = _apply_post_corrections(payload, prepared_full, corrections)
    declared_units_raw, declared_area_raw = parse_declared_totals("\n".join(pages_text))
    declared_units_prepared, declared_area_prepared = parse_declared_totals(prepared_full)
    declared_units = declared_units_prepared if declared_units_prepared is not None else declared_units_raw
    declared_area = declared_area_prepared if declared_area_prepared is not None else declared_area_raw
    return {
        "data": payload,
        "raw": raw_payload,
        "prepared_text": prepared_full,
        "model_used": model_used_global,
        "applied_corrections": applied,
        "corrections": corrections,
        "declared_units": declared_units,
        "declared_area": declared_area,
    }
