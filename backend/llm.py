from __future__ import annotations
import os, json
from typing import Dict, Any
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = '''You are an expert production planner for a glass factory.
You convert pasted order text (copied from a PDF) into STRICT JSON rows.

Rules:
- Use ONLY the JSON schema provided. No extra keys.
- Return `dimension` as WIDTHxHEIGHT in mm, no spaces. Example: "520x1168".
- `area` must be in square meters as a decimal (dot), preferably with 3 decimals. If not given, compute from mm: area=(W*H)/1_000_000.
- Prefer exact transcription for `type` (e.g., "2 vetri 33.1F+14+4 LowE 24mm"). Keep original casing where possible.
- If multiple position lines refer to the last seen type, repeat the same `type` for each row.
- Normalize order number like 'R - 25-0716' -> 'R-25-0716'. If not found, allow empty string.
- Ignore summary lines like 'm2 2 1,220' or 'Totale 4 1,640'.
- Quantity is normally 1 per row unless an explicit quantity is tied to a single dimension/position. If unsure, use 1.
- For `position`, return only the short code (e.g., "1-1"), not the order prefix.
- If you must guess or normalize, add a human-readable note to `warnings`.
- The pasted text may contain MULTIPLE orders. Extract ALL of them.
- Every row MUST include `order_number`. When the text switches to a new order number, continue using that new value for subsequent rows until it changes again.
- If an order number appears with spaces or separators (e.g., "R - 25-0897"), normalize to "R-25-0897".
- If a dimension is NOT explicitly present for a row (or is unclear/split across lines), set `dimension` to an empty string "" (do NOT guess or infer from area).
'''

USER_HINTS = ''

FEW_SHOT_EXAMPLES = [
    {
        "input": "DOCUMENTO CORRELATO R - 25-0716\n2 vetri 33.1F+14+4 LowE 24mm R-25-0716/2-1/MARCO 596 x 360\n0,210 1 0,210",
        "output": {
            "order_number": "R-25-0716",
            "rows": [
                {"order_number":"R-25-0716","type":"2 vetri 33.1F+14+4 LowE 24mm","dimension":"596x360","position":"2-1","quantity":1,"area":0.210}
            ]
        }
    }
]

def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

JSON_SCHEMA = {
    "name": "extraction_result",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "order_number": {"type": "string"},
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "order_number": {"type":"string"},
                        "type": {"type": "string"},
                        "dimension": {"type": "string", "pattern": "^(?:\\d{2,4}x\\d{2,4})?$"},
                        "position": {"type": "string"},
                        "quantity": {"type": "integer", "minimum": 1},
                        "area": {"type": "number", "minimum": 0.0}
                    },
                    "required": ["type","dimension","position","quantity","area"]
                }
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["rows"]
    }
}

def build_messages(pasted_text: str) -> list[dict]:
    msgs = [{"role":"system", "content": SYSTEM_PROMPT}]
    if USER_HINTS.strip():
        msgs.append({"role":"system","content": USER_HINTS.strip()})
    for ex in FEW_SHOT_EXAMPLES:
        msgs.append({"role":"user","content": ex["input"]})
        msgs.append({"role":"assistant","content": json.dumps(ex["output"], ensure_ascii=False)})
    msgs.append({"role":"user","content": pasted_text})
    return msgs

def call_llm_for_extraction(pasted_text: str) -> Dict[str, Any]:
    client = get_client()
    messages = build_messages(pasted_text)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type":"json_schema","json_schema": JSON_SCHEMA},
        temperature=0.0,
    )
    content = completion.choices[0].message.content
    data = json.loads(content)
    return data
