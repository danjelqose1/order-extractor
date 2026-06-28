from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence


INVOICE_LINE_SCHEMA = {
    "name": "invoice_line_analysis",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "normalizedType": {"type": "string"},
            "glassKey": {"type": ["string", "null"]},
            "spacerMode": {"type": "string", "enum": ["normal", "thermal"]},
            "isLaminated": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": [
            "normalizedType",
            "glassKey",
            "spacerMode",
            "isLaminated",
            "confidence",
            "reason",
        ],
    },
    "strict": True,
}


def _clean_known_types(values: Sequence[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values or []:
        text = str(value or "").strip()
        folded = text.casefold()
        if not text or folded in seen:
            continue
        seen.add(folded)
        output.append(text)
    return output


def _completion_text(completion: Any) -> str:
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    return content.strip() if isinstance(content, str) else ""


def _strip_json_fences(value: str) -> str:
    text = str(value or "").strip()
    if not text.startswith("```"):
        return text
    text = text.strip("`").strip()
    if text.lower().startswith("json"):
        text = text[4:].lstrip()
    return text


def match_invoice_glass_type(
    client: Any,
    *,
    raw_name: str,
    known_types: Sequence[Any],
) -> Optional[str]:
    raw_text = str(raw_name or "").strip()
    canonical_types = _clean_known_types(known_types)
    if not raw_text or not canonical_types:
        raise ValueError("raw_name and known_types are required")

    completion = client.chat.completions.create(
        model=os.getenv("INVOICE_GLASS_MATCH_MODEL", "gpt-4o-mini"),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert assistant for a glass processing factory invoicing system. "
                    "Match noisy OCR glass descriptions to one canonical factory price-list key. "
                    "Handle OCR and spelling variants such as STAINATO/SATINATO, LOW E/LOE/LOWE, "
                    "and decimal commas. Compare thickness and treatment, while ignoring pane counts, "
                    "dimensions, and spacer thicknesses. Return exactly one supplied key, or NONE when "
                    "no reasonable match exists. Never invent a key."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Canonical keys:\n{json.dumps(canonical_types, ensure_ascii=False)}\n\n"
                    f"Noisy glass description:\n{raw_text}\n\n"
                    "Examples:\n"
                    '- "33.1STAINATO 44mm" should match "33.1satinato" when supplied.\n'
                    '- "4F low e 6mm" should match "4f lowe" when supplied.\n'
                    '- "33.1STAINATO" must return NONE when only "4f" and "6f" are supplied.\n\n'
                    "Return one canonical key exactly, or NONE."
                ),
            },
        ],
    )
    candidate = _completion_text(completion).strip().strip("\"'")
    if not candidate or candidate.upper() == "NONE":
        return None
    return next(
        (item for item in canonical_types if item.casefold() == candidate.casefold()),
        None,
    )


def analyze_invoice_line(
    client: Any,
    *,
    raw_line: str,
    known_glass_types: Sequence[Any],
) -> Dict[str, Any]:
    raw_text = str(raw_line or "").strip()
    canonical_types = _clean_known_types(known_glass_types)
    if not raw_text:
        raise ValueError("raw_line is required")

    completion = client.chat.completions.create(
        model=os.getenv("INVOICE_LINE_ANALYSIS_MODEL", "gpt-5.4-mini"),
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": INVOICE_LINE_SCHEMA,
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert assistant for a glass processing factory invoicing system. "
                    "Normalize a noisy insulated-glass-unit line while preserving its pane structure "
                    "and thicknesses. Correct obvious OCR errors such as STAINATO to SATINATO, "
                    "LOW E or LOE to LOWE, and 33,1 to 33.1. Produce a clean normalizedType suitable "
                    "for editing. Choose glassKey only from the supplied canonical keys, or null. "
                    "Use spacerMode=thermal for warm-edge terms such as caldo, c.caldo, cald, termico, "
                    "or warm edge; otherwise use normal. Detect laminated glass such as 33.1, 44.1, "
                    "laminato, or stratificato. Return only the requested JSON object. Never invent "
                    "a canonical key; use null and confidence 0 when none is reasonable."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Canonical glass keys:\n{json.dumps(canonical_types, ensure_ascii=False)}\n\n"
                    f"Noisy IGU line:\n{raw_text}"
                ),
            },
        ],
    )
    content = _strip_json_fences(_completion_text(completion))
    if not content:
        raise RuntimeError("Invoice AI returned no content")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("Invoice AI returned an invalid object")

    normalized_type = str(parsed.get("normalizedType") or "").strip()
    if not normalized_type:
        raise RuntimeError("Invoice AI returned an empty normalizedType")

    glass_key: Optional[str] = None
    candidate = parsed.get("glassKey")
    if candidate is not None and str(candidate).strip():
        candidate_text = str(candidate).strip()
        glass_key = next(
            (item for item in canonical_types if item.casefold() == candidate_text.casefold()),
            None,
        )
        if glass_key is None:
            raise RuntimeError("Invoice AI returned an unknown glassKey")

    spacer_mode = str(parsed.get("spacerMode") or "").strip().lower()
    if spacer_mode not in {"normal", "thermal"}:
        raise RuntimeError("Invoice AI returned an invalid spacerMode")

    try:
        confidence = float(parsed.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Invoice AI returned an invalid confidence") from exc
    if not 0 <= confidence <= 1:
        raise RuntimeError("Invoice AI returned confidence outside 0..1")

    return {
        "normalizedType": normalized_type,
        "glassKey": glass_key,
        "spacerMode": spacer_mode,
        "isLaminated": bool(parsed.get("isLaminated")),
        "confidence": confidence,
        "reason": str(parsed.get("reason") or "").strip(),
    }


__all__ = [
    "INVOICE_LINE_SCHEMA",
    "analyze_invoice_line",
    "match_invoice_glass_type",
]
