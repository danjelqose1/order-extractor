from __future__ import annotations
import csv, hashlib, json, os, re
from copy import deepcopy
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, ValidationError
from schema import ExtractionResult, Row
from llm import (
    call_llm_for_extraction,
    call_llm_for_extraction_multi,
    pdf_to_png_pages,
    ocr_png_with_openai,
    _prepare_text,
    extract_client_name,
    get_client,
)
from dotenv import load_dotenv
import traceback
import httpx
import openai as openai_pkg
from db import (
    init_db,
    insert_extraction_with_rows,
    update_order_rows,
    get_orders,
    get_orders_by_identifiers,
    get_order_with_extraction,
    delete_order,
    get_all_rows_for_export,
    save_correction,
    list_corrections,
    delete_correction,
)
from validators import validate_rows
from dimension_repair import apply_dimension_repair
from area_dimension_validator import apply_area_dimension_validation
from utils_text import clean_dimension, parse_declared_totals
from prompts import PROMPTS
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH, override=True)

# Data/config paths
DATA_DIR = Path(os.getenv("DB_DIR", "data"))
PRICE_CONFIG_PATH = DATA_DIR / "price-config.json"
INVOICES_PATH = DATA_DIR / "invoices.json"

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
APP_KEY = os.getenv("APP_KEY")  # optional shared secret

ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "gpt-4o-mini")
try:
    ANALYSIS_DATASET_MAX_CHARS = int(os.getenv("ANALYSIS_MAX_DATASET_CHARS", "50000"))
except ValueError:
    ANALYSIS_DATASET_MAX_CHARS = 50000
ANALYSIS_FALLBACK_ANSWER = "AI analysis is unavailable. The dashboard above still works offline."
ANALYSIS_SYSTEM_PROMPT = PROMPTS["analysis"]["system"]
ANALYSIS_STYLE_PROMPT = PROMPTS["analysis"].get("style", "")
ANALYSIS_MAX_TURNS = 7

analysis_memory: Dict[str, Dict[str, Any]] = {}

# Invoice price config defaults (mirrors frontend defaults)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_glass_key(raw: Any) -> str:
    text = "" if raw is None else str(raw).lower().strip()
    if not text:
        return ""
    text = re.sub(r"^\s*\d+\s*vetri?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bvetri?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s+", "", text)


DEFAULT_GLASS_PRICES: Dict[str, float] = {
    _normalize_glass_key("4F"): 1000,
    _normalize_glass_key("4 Satinato"): 1800,
    _normalize_glass_key("4 LowE"): 2000,
    _normalize_glass_key("33.1F"): 2500,
    _normalize_glass_key("33.1 Satinato"): 3500,
    _normalize_glass_key("33.1 LowE"): 3700,
}

DEFAULT_SPACER_PRICES: Dict[Any, Dict[str, float]] = {
    6: {"normal": 500, "thermal": 1000},
    9: {"normal": 500, "thermal": 1000},
    10: {"normal": 500, "thermal": 1000},
    12: {"normal": 500, "thermal": 1000},
    14: {"normal": 600, "thermal": 1100},
    16: {"normal": 700, "thermal": 1200},
    18: {"normal": 800, "thermal": 1300},
    20: {"normal": 900, "thermal": 1400},
    22: {"normal": 1000, "thermal": 1500},
    24: {"normal": 1100, "thermal": 1600},
}

def _tokenize_igu_composition(type_string: str) -> List[str]:
    raw = "" if type_string is None else str(type_string)
    compact = re.sub(r"^\s*\d+\s*vetri?\s*", "", raw, flags=re.IGNORECASE)
    compact = re.sub(r"\s+", " ", compact).strip()
    match = re.search(r"\d", compact)
    core = compact[match.start():] if match else compact
    without_mm = re.sub(r"\b\d+\s*mm\b", "", core, flags=re.IGNORECASE)
    without_mm = re.sub(r"\bmm\b", "", without_mm, flags=re.IGNORECASE)
    return [token.strip() for token in without_mm.split("+") if token.strip()]


def _extract_spacer_width_mm(type_string: str) -> Optional[float]:
    for token in _tokenize_igu_composition(type_string):
        candidate = re.sub(r"\s+", "", token).replace(",", ".")
        try:
            value = float(candidate)
        except ValueError:
            continue
        if value == value:
            return value
    return None


def _default_price_config() -> Dict[str, Any]:
    return {
        "glassPrices": dict(DEFAULT_GLASS_PRICES),
        "spacerPrices": {k: dict(v) for k, v in DEFAULT_SPACER_PRICES.items()},
        "typeCorrections": [],
    }


def _sanitize_glass_prices(payload: Any) -> Dict[str, float]:
    prices = dict(DEFAULT_GLASS_PRICES)
    if isinstance(payload, dict):
        for key, value in payload.items():
            norm_key = _normalize_glass_key(key)
            try:
                price_val = float(value)
            except (TypeError, ValueError):
                continue
            if not norm_key or not (price_val == price_val):  # filter NaN
                continue
            prices[norm_key] = price_val
    return prices


def _sanitize_spacer_prices(payload: Any) -> Dict[Any, Dict[str, float]]:
    prices = {k: dict(v) for k, v in DEFAULT_SPACER_PRICES.items()}
    if isinstance(payload, dict):
        for thickness, entry in payload.items():
            try:
                thickness_num = int(thickness)
            except (TypeError, ValueError):
                continue
            baseline = prices.get(thickness_num, {})
            if isinstance(entry, dict):
                normal = entry.get("normal")
                thermal = entry.get("thermal")
                try:
                    normal_val = float(normal) if normal is not None else None
                except (TypeError, ValueError):
                    normal_val = None
                try:
                    thermal_val = float(thermal) if thermal is not None else None
                except (TypeError, ValueError):
                    thermal_val = None
                if normal_val is None and thermal_val is None and thickness_num not in prices:
                    continue
                prices[thickness_num] = {
                    "normal": normal_val if normal_val is not None else baseline.get("normal"),
                    "thermal": thermal_val if thermal_val is not None else baseline.get("thermal"),
                }
    return prices


def _sanitize_type_corrections(payload: Any) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            raw = (entry.get("raw") or "").strip()
            corrected = (entry.get("corrected") or "").strip()
            if raw and corrected:
                output.append({"raw": raw, "corrected": corrected})
    return output


def _coerce_price_config(payload: Any) -> Dict[str, Any]:
    glass_prices = _sanitize_glass_prices((payload or {}).get("glassPrices") if isinstance(payload, dict) else None)
    spacer_prices = _sanitize_spacer_prices((payload or {}).get("spacerPrices") if isinstance(payload, dict) else None)
    type_corrections = _sanitize_type_corrections((payload or {}).get("typeCorrections") if isinstance(payload, dict) else None)
    return {
        "glassPrices": glass_prices,
        "spacerPrices": spacer_prices,
        "typeCorrections": type_corrections,
    }


def _load_price_config() -> Dict[str, Any]:
    if PRICE_CONFIG_PATH.exists():
        try:
            data = json.loads(PRICE_CONFIG_PATH.read_text())
            if isinstance(data, dict):
                return _coerce_price_config(data)
        except Exception as exc:
            print(f"[prices] failed to read price config: {exc}")
    return _default_price_config()


def _save_price_config(config: Dict[str, Any]) -> Dict[str, Any]:
    PRICE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _coerce_price_config(config)
    PRICE_CONFIG_PATH.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2))
    return sanitized


def _load_invoices() -> Dict[str, Any]:
    if INVOICES_PATH.exists():
        try:
            data = json.loads(INVOICES_PATH.read_text())
            if isinstance(data, dict):
                jobs = data.get("jobs")
                if isinstance(jobs, list):
                    return {"jobs": jobs}
        except Exception as exc:
            print(f"[invoices] failed to read invoices: {exc}")
    return {"jobs": []}


def _save_invoices(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Invoices payload must be an object.")
    jobs = data.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("jobs must be a list.")
    INVOICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"jobs": jobs}
    INVOICES_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload

app = FastAPI(title="LLM Order Extractor (Local)", version="1.0.0")

# Configure CORS: accept comma‑separated FRONTEND_ORIGINS; fall back to safe defaults
_frontend_origins_env = os.getenv("FRONTEND_ORIGINS") or os.getenv("FRONTEND_ORIGIN")
if _frontend_origins_env:
    _ALLOWED_ORIGINS = [o.strip() for o in _frontend_origins_env.split(",") if o.strip()]
else:
    _ALLOWED_ORIGINS = [
        "https://danjelqose1.github.io",               # GitHub Pages frontend
        "https://order-extractor-kdih.onrender.com",   # Backend self-origin
        "http://localhost:5055",                       # Local backend
        "http://127.0.0.1:5055",                       # Local backend
        "http://localhost:5500",                       # Local frontend (VS Code Live Server)
        "http://127.0.0.1:5500",                       # Local frontend (Python http.server)
        "null",                                        # file:// origin for local testing
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route for basic health/info, quiets 404s on /
@app.get("/")
def root():
    return {"ok": True, "service": "order-extractor"}

class PasteIn(BaseModel):
    text: str


class ApprovePayload(BaseModel):
    rows: List[Row]
    notes: Optional[str] = None


class AnalysisAskPayload(BaseModel):
    question: str
    dataset: Dict[str, Any]
    settings: Optional[Dict[str, Any]] = None


def _debug_log_rows(rows):
    try:
        print(f"[debug] rows extracted: {len(rows)}")
        sample_positions = ["22-1", "40-1", "41-1", "65-2", "79-2", "93-2", "100-2"]
        lookup = {}
        for row in rows:
            if hasattr(row, "position"):
                lookup[row.position] = row
            elif isinstance(row, dict):
                lookup[row.get("position")] = row
        for pos in sample_positions:
            row = lookup.get(pos)
            if row:
                if hasattr(row, "dimension"):
                    dim = row.dimension
                    area = row.area
                else:
                    dim = row.get("dimension")
                    area = row.get("area")
                print(f"[debug] sample {pos}: dimension={dim}, area={area}")
            else:
                print(f"[debug] sample {pos}: missing")
    except Exception as exc:
        print(f"[debug] logging error: {exc}")

DEFAULT_HISTORY_LIMIT = 25
MAX_HISTORY_LIMIT = 100


def _compute_hash(content: Optional[str]) -> Optional[str]:
    if not content:
        return None
    return hashlib.sha1(content.encode("utf-8", "ignore")).hexdigest()


def _rows_to_dicts(rows: List[Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows or []:
        if hasattr(row, "model_dump"):
            output.append(row.model_dump(mode="python"))
        elif isinstance(row, dict):
            output.append(dict(row))
    return output


def _primary_order_number(rows: List[Dict[str, Any]]) -> str:
    for row in rows:
        candidate = (row.get("order_number") or "").strip()
        if candidate:
            return candidate
    return ""


def _summarize_totals(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    units = 0
    area = 0.0
    for row in rows:
        try:
            units += int(row.get("quantity") or 0)
        except Exception:
            pass
        try:
            area += float(row.get("area") or 0.0)
        except Exception:
            pass
    return {"units": units, "area": round(area, 3)}


def _parse_order_datetime(value: Any) -> datetime:
    if value is None:
        return datetime.min
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except Exception:
            return datetime.min
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return datetime.min
        try:
            # Handle trailing Z for UTC
            if text.endswith("Z"):
                return datetime.fromisoformat(text.replace("Z", "+00:00"))
            return datetime.fromisoformat(text)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    return datetime.min


def _sorted_orders_newest(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    decorated: List[Tuple[datetime, Dict[str, Any]]] = []
    for item in items:
        if isinstance(item, dict):
            timestamp = _parse_order_datetime(item.get("created_at") or item.get("createdAt"))
            decorated.append((timestamp, deepcopy(item)))
    decorated.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in decorated]


def _prepare_analysis_dataset(dataset: Dict[str, Any], max_chars: int) -> Tuple[Dict[str, Any], str, bool, Dict[str, int]]:
    payload = deepcopy(dataset) if isinstance(dataset, dict) else {}
    orders = payload.get("orders")
    if not isinstance(orders, list):
        orders = []
        payload["orders"] = orders
    processing = payload.get("processing_orders")
    if not isinstance(processing, list):
        processing = []
        payload["processing_orders"] = processing

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        payload["meta"] = meta
    notes = meta.get("notes")
    if not isinstance(notes, list):
        notes = []
        meta["notes"] = notes

    original_orders = len(dataset.get("orders") or [])
    original_processing = len(dataset.get("processing_orders") or [])

    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    truncated = False
    trimmed_orders = 0
    while len(serialized) > max_chars and orders:
        orders.pop()
        trimmed_orders += 1
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    trimmed_processing = 0
    while len(serialized) > max_chars and processing:
        processing.pop()
        trimmed_processing += 1
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(serialized) > max_chars:
        payload["orders"] = []
        payload["processing_orders"] = []
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    if trimmed_orders or trimmed_processing:
        truncated = True
        meta["truncated"] = True
        if trimmed_orders:
            note = f"orders list trimmed to newest {len(payload['orders'])} to fit token budget"
            if note not in notes:
                notes.append(note)
        if trimmed_processing:
            note = "processing orders trimmed to fit token budget"
            if note not in notes:
                notes.append(note)

    meta["size"] = len(serialized)
    return payload, serialized, truncated, {
        "orders_total": original_orders,
        "processing_total": original_processing,
    }


init_db()


# Invoice pricing/type corrections now persist on the server (shared across devices)
@app.get("/api/prices")
def get_price_config() -> Dict[str, Any]:
    return _load_price_config()


@app.post("/api/prices")
def save_price_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object.")
    try:
        saved = _save_price_config(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save pricing config: {exc}") from exc
    return saved


# Invoice jobs now persist on the server (shared across devices)
@app.get("/api/invoices")
def get_invoices() -> Dict[str, Any]:
    return _load_invoices()


@app.post("/api/invoices")
def save_invoices(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object.")
    if not isinstance(payload.get("jobs"), list):
        raise HTTPException(status_code=400, detail="jobs must be a list.")
    try:
        _save_invoices({"jobs": payload["jobs"]})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save invoices: {exc}") from exc
    return {"ok": True}


@app.delete("/api/invoices/{invoice_id}")
def delete_invoice(invoice_id: str) -> Dict[str, Any]:
    data = _load_invoices()
    jobs = data.get("jobs") if isinstance(data, dict) else []
    remaining = []
    deleted = False
    if isinstance(jobs, list):
        for job in jobs:
            try:
                job_id = job.get("id")
            except Exception:
                job_id = None
            if str(job_id) == str(invoice_id):
                deleted = True
                continue
            remaining.append(job)
    try:
        _save_invoices({"jobs": remaining})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete invoice: {exc}") from exc
    return {"ok": True, "deleted": deleted}


@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/extract")
def extract(inb: PasteIn, x_app_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if APP_KEY and x_app_key != APP_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        bundle = call_llm_for_extraction(inb.text)
        raw_payload = bundle.get("raw") or {}
        result_raw = ExtractionResult(**raw_payload)
        result_data = ExtractionResult(**(bundle.get("data") or raw_payload))

        _debug_log_rows(result_data.rows)

        rows_dict = _rows_to_dicts(result_data.rows)
        repaired_rows = apply_dimension_repair(inb.text, rows_dict)
        validation = validate_rows(repaired_rows, context={"prepared_text": bundle.get("prepared_text")})
        normalized_rows = validation.get("rows", repaired_rows)
        final_rows = apply_area_dimension_validation(normalized_rows)
        row_warnings = validation.get("row_warnings", {})

        combined_warnings: List[str] = []
        for source_list in (
            raw_payload.get("warnings") or [],
            (bundle.get("data") or {}).get("warnings") or [],
            validation.get("warnings", []) or [],
        ):
            combined_warnings.extend(source_list)

        applied = bundle.get("applied_corrections") or []
        if applied:
            combined_warnings.append(f"auto_corrections_applied:{','.join(str(i) for i in applied)}")

        prepared_text = bundle.get("prepared_text") or inb.text.strip()
        client_name = extract_client_name(inb.text) or extract_client_name(prepared_text)
        hash_value = _compute_hash(prepared_text or inb.text)
        llm_output_json = json.dumps(raw_payload, ensure_ascii=False)
        declared_units = bundle.get("declared_units")
        declared_area = bundle.get("declared_area")
        totals = _summarize_totals(final_rows)
        if declared_units is not None and declared_units != totals["units"]:
            combined_warnings.append(
                f"declared_units_mismatch: declared {declared_units}, parsed {totals['units']}"
            )
        if declared_area is not None and abs((declared_area or 0) - totals["area"]) > 0.05:
            combined_warnings.append(
                f"declared_area_mismatch: declared {declared_area:.3f}, parsed {totals['area']:.3f}"
            )

        insert_result = insert_extraction_with_rows(
            source="paste",
            rows=final_rows,
            raw_input=inb.text,
            prepared_text=prepared_text,
            llm_output_json=llm_output_json,
            model_used=bundle.get("model_used"),
            hash_value=hash_value,
        )
        draft_order_id = insert_result["order_id"]
        status = insert_result.get("status", "draft")
        saved_order_id = draft_order_id if status == "approved" else None

        payload = {
            "order_number": (bundle.get("data") or {}).get("order_number") or _primary_order_number(final_rows),
            "rows": final_rows,
            "warnings": combined_warnings,
            "row_warnings": row_warnings,
            "draft_order_id": draft_order_id,
            "saved_order_id": saved_order_id,
            "status": status,
            "applied_corrections": applied,
            "model_used": bundle.get("model_used"),
            "declared_units": declared_units,
            "declared_area": declared_area,
            "parsed_units": totals["units"],
            "parsed_area": totals["area"],
            "client": client_name or "—",
        }
        return payload
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=f"Validation error: {ve.errors()}")
    except Exception as e:
        print("[extract] fatal error:\n" + "".join(traceback.format_exc()))
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.post("/extract_pdf")
async def extract_pdf(file: UploadFile = File(...), x_app_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """Accept a PDF file upload, OCR each page, then run the multi-page LLM extraction."""
    if APP_KEY and x_app_key != APP_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not file or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    try:
        raw_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {e}")

    if not raw_bytes:
        raise HTTPException(
            status_code=422,
            detail="The PDF contains no extractable text (document may be blank or OCR is disabled).",
        )

    try:
        png_pages = pdf_to_png_pages(raw_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR pipeline failed: {e}")

    if not png_pages:
        raise HTTPException(
            status_code=422,
            detail="The PDF appears to contain no pages.",
        )

    try:
        raw_ocr_pages: List[str] = []
        prepared_pages: List[str] = []
        for image_bytes in png_pages:
            ocr_text = ocr_png_with_openai(image_bytes)
            cleaned = (ocr_text or "").strip()
            raw_ocr_pages.append(cleaned)
            prepared = (_prepare_text(ocr_text) if ocr_text else "") or cleaned
            prepared_pages.append(prepared)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR pipeline failed: {e}")

    if not any((page or "").strip() for page in raw_ocr_pages):
        raise HTTPException(
            status_code=422,
            detail="The PDF contains no legible text after OCR.",
        )

    pages_for_llm = prepared_pages

    full_text = "\n".join(raw_ocr_pages)
    client_name = extract_client_name(full_text)

    try:
        bundle = call_llm_for_extraction_multi(pages_for_llm)
        raw_payload = bundle.get("raw") or {}
        result_raw = ExtractionResult(**raw_payload)
        result_data = ExtractionResult(**(bundle.get("data") or raw_payload))

        _debug_log_rows(result_data.rows)

        rows_dict = _rows_to_dicts(result_data.rows)
        prepared_text = bundle.get("prepared_text") or ""
        raw_joined = "\n\n".join(page for page in raw_ocr_pages if page)
        repaired_rows = apply_dimension_repair(raw_joined, rows_dict)
        validation = validate_rows(repaired_rows, context={"prepared_text": prepared_text})
        normalized_rows = validation.get("rows", repaired_rows)
        final_rows = apply_area_dimension_validation(normalized_rows)
        row_warnings = validation.get("row_warnings", {})

        combined_warnings: List[str] = []
        for source_list in (
            raw_payload.get("warnings") or [],
            (bundle.get("data") or {}).get("warnings") or [],
            validation.get("warnings", []) or [],
        ):
            combined_warnings.extend(source_list)

        applied = bundle.get("applied_corrections") or []
        if applied:
            combined_warnings.append(f"auto_corrections_applied:{','.join(str(i) for i in applied)}")

        hash_value = _compute_hash(prepared_text or raw_joined)
        llm_output_json = json.dumps(raw_payload, ensure_ascii=False)
        declared_units = bundle.get("declared_units")
        declared_area = bundle.get("declared_area")
        totals = _summarize_totals(final_rows)
        if declared_units is not None and declared_units != totals["units"]:
            combined_warnings.append(
                f"declared_units_mismatch: declared {declared_units}, parsed {totals['units']}"
            )
        if declared_area is not None and abs((declared_area or 0) - totals["area"]) > 0.05:
            combined_warnings.append(
                f"declared_area_mismatch: declared {declared_area:.3f}, parsed {totals['area']:.3f}"
            )

        insert_result = insert_extraction_with_rows(
            source="pdf",
            rows=final_rows,
            raw_input=raw_joined,
            prepared_text=prepared_text,
            llm_output_json=llm_output_json,
            model_used=bundle.get("model_used"),
            hash_value=hash_value,
        )
        draft_order_id = insert_result["order_id"]
        status = insert_result.get("status", "draft")
        saved_order_id = draft_order_id if status == "approved" else None

        payload = {
            "order_number": (bundle.get("data") or {}).get("order_number") or _primary_order_number(final_rows),
            "rows": final_rows,
            "warnings": combined_warnings,
            "row_warnings": row_warnings,
            "draft_order_id": draft_order_id,
            "saved_order_id": saved_order_id,
            "status": status,
            "applied_corrections": applied,
            "model_used": bundle.get("model_used"),
            "declared_units": declared_units,
            "declared_area": declared_area,
            "parsed_units": totals["units"],
            "parsed_area": totals["area"],
            "client": client_name or "—",
        }
        return payload
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=f"Validation error: {ve.errors()}")
    except Exception as e:
        print("[extract_pdf] fatal error:\n" + "".join(traceback.format_exc()))
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.post("/extract-pdf")
async def extract_pdf_dash_alias(file: UploadFile = File(...), x_app_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    return await extract_pdf(file=file, x_app_key=x_app_key)


@app.get("/orders")
def list_orders(
    query: Optional[str] = Query(default=None, description="Search by order number or client hint"),
    status: Optional[str] = Query(default=None, description="Filter by status draft|approved"),
    year: Optional[str] = Query(default=None, description="Year filter: YYYY (defaults to current year) or all"),
    limit: int = Query(default=DEFAULT_HISTORY_LIMIT, ge=1, le=MAX_HISTORY_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    try:
        items = get_orders(query=query, status=status, year=year, limit=limit, offset=offset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "items": items,
        "limit": limit,
        "offset": offset,
        "count": len(items),
        "has_more": len(items) == limit,
    }


@app.get("/api/history/orders")
def list_history_orders(ids: str = Query(default="", description="Comma-separated order numbers or ids")) -> Dict[str, Any]:
    identifiers = [part.strip() for part in (ids or "").split(",") if part.strip()]
    if not identifiers:
        return {"orders": []}

    orders = get_orders_by_identifiers(identifiers)
    response_orders: List[Dict[str, Any]] = []
    for order in orders:
        order_numbers = order.get("order_numbers") or []
        order_id = ", ".join(order_numbers) if order_numbers else str(order.get("id") or "")
        items = []
        for row in order.get("rows") or []:
            _, dims = clean_dimension(row.get("dimension") or "")
            width_mm = dims[0] if dims else None
            height_mm = dims[1] if dims else None
            try:
                quantity = int(row.get("quantity") or 0)
            except (TypeError, ValueError):
                quantity = 0
            perimeter_mm = None
            if width_mm is not None and height_mm is not None:
                perimeter_mm = 2 * (width_mm + height_mm)
            spacer_width_mm = _extract_spacer_width_mm(row.get("type") or "")
            items.append(
                {
                    "position": row.get("position") or "",
                    "width_mm": width_mm,
                    "height_mm": height_mm,
                    "quantity": quantity,
                    "spacer_width_mm": spacer_width_mm,
                    "perimeter_mm": perimeter_mm,
                }
            )
        response_orders.append({"order_id": order_id, "items": items})
    return {"orders": response_orders}


@app.get("/orders/{order_id}")
def get_order_detail(order_id: int) -> Dict[str, Any]:
    order = get_order_with_extraction(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    rows = order.get("rows") or []
    extraction = order.get("extraction") or {}
    validation = validate_rows([dict(row) for row in rows], context={"prepared_text": extraction.get("prepared_text", "")})
    normalized_rows = validation.get("rows", rows)
    order["rows"] = apply_area_dimension_validation(normalized_rows)
    order["row_warnings"] = validation.get("row_warnings", {})
    order["warnings"] = validation.get("warnings", [])
    declared_units, declared_area = parse_declared_totals(extraction.get("prepared_text", ""))
    order["declared_units"] = declared_units
    order["declared_area"] = declared_area
    totals = _summarize_totals(order["rows"])
    order["parsed_units"] = totals["units"]
    order["parsed_area"] = totals["area"]
    if declared_units is not None and declared_units != totals["units"]:
        order["warnings"].append(
            f"declared_units_mismatch: declared {declared_units}, parsed {totals['units']}"
        )
    if declared_area is not None and abs((declared_area or 0) - totals["area"]) > 0.05:
        order["warnings"].append(
            f"declared_area_mismatch: declared {declared_area:.3f}, parsed {totals['area']:.3f}"
        )
    return order


@app.delete("/orders/{order_id}")
def remove_order(order_id: int) -> Dict[str, bool]:
    deleted = delete_order(order_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Order not found")
    return {"ok": True}


def _rows_to_csv(rows: List[Dict[str, Any]]) -> str:
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["order_number", "type", "dimension", "position", "quantity", "area"])
    for row in rows:
        writer.writerow(
            [
                row.get("order_number", ""),
                row.get("type", ""),
                row.get("dimension", ""),
                row.get("position", ""),
                row.get("quantity", 0),
                row.get("area", 0.0),
            ]
        )
    return output.getvalue()


@app.post("/orders/{order_id}/approve")
def approve_order(order_id: int, payload: ApprovePayload) -> Dict[str, Any]:
    snapshot = get_order_with_extraction(order_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Order not found")

    rows_dict = [row.model_dump(mode="python") for row in payload.rows]

    # Ensure prepared_text is available BEFORE validation below
    extraction = snapshot.get("extraction") or {}
    prepared_text = extraction.get("prepared_text", "")

    validation = validate_rows(rows_dict, context={"prepared_text": prepared_text})
    final_rows = validation.get("rows", rows_dict)
    row_warnings = validation.get("row_warnings", {})

    combined_warnings: List[str] = []
    for source_list in (validation.get("warnings", []) or []):
        combined_warnings.extend(source_list)

    try:
        updated_order = update_order_rows(
            order_id,
            final_rows,
            status="approved",
            notes=payload.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        print("[approve_order] update failed:\n" + "".join(traceback.format_exc()))
        raise HTTPException(status_code=500, detail=f"Failed to update order: {exc}")

    validated_after = validate_rows([dict(row) for row in updated_order.get("rows", [])], context={"prepared_text": prepared_text})
    updated_order["rows"] = validated_after.get("rows", updated_order.get("rows", []))
    row_warnings = validated_after.get("row_warnings", {})
    combined_warnings.extend(validated_after.get("warnings", []) or [])

    correction_info = None
    extraction = snapshot.get("extraction") or {}
    before_json = extraction.get("llm_output_json")
    prepared_text = extraction.get("prepared_text") or ""
    declared_units, declared_area = parse_declared_totals(prepared_text)
    if before_json:
        try:
            before_data = json.loads(before_json)
        except Exception:
            before_data = {}
        before_rows = before_data.get("rows") if isinstance(before_data, dict) else None
        if before_rows is not None and before_rows != final_rows:
            correction_info = save_correction(
                before_json=before_json,
                after_json=json.dumps({"rows": final_rows}, ensure_ascii=False),
                prepared_text=prepared_text,
                notes=payload.notes,
            )
    totals = _summarize_totals(updated_order.get("rows") or [])
    if declared_units is not None and declared_units != totals["units"]:
        combined_warnings.append(
            f"declared_units_mismatch: declared {declared_units}, parsed {totals['units']}"
        )
    if declared_area is not None and abs((declared_area or 0) - totals["area"]) > 0.05:
        combined_warnings.append(
            f"declared_area_mismatch: declared {declared_area:.3f}, parsed {totals['area']:.3f}"
        )
    return {
        "saved_order_id": order_id,
        "warnings": combined_warnings,
        "row_warnings": row_warnings,
        "order": updated_order,
        "correction": correction_info,
        "declared_units": declared_units,
        "declared_area": declared_area,
        "parsed_units": totals["units"],
        "parsed_area": totals["area"],
    }


@app.get("/orders/{order_id}/csv")
def download_order_csv(order_id: int):
    order = get_order_with_extraction(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    if (order.get("status") or "").lower() != "approved":
        raise HTTPException(status_code=400, detail="Order not approved")
    csv_content = _rows_to_csv(order.get("rows") or [])
    filename = f"order-{order_id}.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=csv_content, media_type="text/csv", headers=headers)


@app.get("/orders/export.csv")
def export_all_orders_csv(
    query: Optional[str] = Query(default=None, description="Search by order number or client hint"),
    status: Optional[str] = Query(default="approved", description="Filter by status draft|approved (default approved)"),
    year: Optional[str] = Query(default=None, description="Year filter: YYYY (defaults to current year) or all"),
):
    try:
        rows = get_all_rows_for_export(query=query, status=status, year=year)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "order_id",
            "created_at",
            "source",
            "client_hint",
            "order_numbers",
            "units_total",
            "area_total",
            "order_number",
            "type",
            "dimension",
            "position",
            "quantity",
            "area",
        ]
    )
    for item in rows:
        order_numbers = ";".join(item.get("order_numbers") or [])
        row = item.get("row") or {}
        writer.writerow(
            [
                item.get("order_id"),
                item.get("created_at"),
                item.get("source"),
                item.get("client_hint") or "",
                order_numbers,
                item.get("units_total"),
                item.get("area_total"),
                row.get("order_number", ""),
                row.get("type", ""),
                row.get("dimension", ""),
                row.get("position", ""),
                row.get("quantity", 0),
                row.get("area", 0.0),
            ]
        )
    filename = "orders-export.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=output.getvalue(), media_type="text/csv", headers=headers)


@app.get("/corrections")
def get_corrections() -> Dict[str, Any]:
    return {"items": list_corrections()}


@app.delete("/corrections/{correction_id}")
def remove_correction(correction_id: int) -> Dict[str, bool]:
    deleted = delete_correction(correction_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Correction not found")
    return {"ok": True}


@app.post("/analysis/ask")
def ask_analysis(request: Request, payload: AnalysisAskPayload) -> Dict[str, str]:
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    dataset = payload.dataset or {}
    if not isinstance(dataset, dict):
        raise HTTPException(status_code=400, detail="Dataset must be an object.")

    settings = payload.settings or {}
    if not isinstance(settings, dict):
        settings = {}

    try:
        payload_dataset, serialized_dataset, truncated, totals = _prepare_analysis_dataset(
            dataset,
            ANALYSIS_DATASET_MAX_CHARS,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid dataset payload: {exc}") from exc

    meta = payload_dataset.setdefault("meta", {})
    notes = meta.setdefault("notes", []) if isinstance(meta, dict) else []
    included_orders = len(payload_dataset.get("orders") or [])
    included_processing = len(payload_dataset.get("processing_orders") or [])
    meta.setdefault("payloadSize", len(serialized_dataset))
    meta.setdefault("ordersIncluded", included_orders)
    meta.setdefault("processingIncluded", included_processing)
    meta.setdefault("ordersOriginal", totals.get("orders_total", included_orders))
    meta.setdefault("processingOriginal", totals.get("processing_total", included_processing))
    if truncated:
        meta["truncated"] = True

    try:
        settings_json = json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        settings = {}
        settings_json = "{}"

    dataset_hash = hashlib.sha1(serialized_dataset.encode("utf-8")).hexdigest()

    session_id = request.headers.get("x-session-id") or "local"
    session_id = session_id.strip() or "local"
    state = analysis_memory.get(session_id)
    if not state or state.get("dataset_hash") != dataset_hash:
        base_messages = [{"role": "system", "content": ANALYSIS_SYSTEM_PROMPT}]
    else:
        base_messages = [msg.copy() for msg in state.get("messages", [])]
        if not base_messages or base_messages[0].get("role") != "system":
            base_messages.insert(0, {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT})

    if not base_messages:
        base_messages = [{"role": "system", "content": ANALYSIS_SYSTEM_PROMPT}]

    messages = base_messages[:]
    if ANALYSIS_STYLE_PROMPT:
        messages.append({"role": "system", "content": ANALYSIS_STYLE_PROMPT})

    def _clip(text: str, limit: int = ANALYSIS_DATASET_MAX_CHARS) -> str:
        return text if len(text) <= limit else text[:limit] + "… (truncated)"

    ordered_dataset = {
        "aggregates": payload_dataset.get("aggregates") or {},
        "orders": payload_dataset.get("orders") or [],
        "processing_orders": payload_dataset.get("processing_orders") or [],
        "meta": payload_dataset.get("meta") or {},
        "scope": payload_dataset.get("scope", "all-time"),
    }
    dataset_json = json.dumps(ordered_dataset, ensure_ascii=False, separators=(",", ":"))

    user_content = (
        f"Question:\n{question}\n\n"
        f"Dataset (all-time):\n{_clip(dataset_json, ANALYSIS_DATASET_MAX_CHARS)}\n\n"
        f"Settings:\n{settings_json}"
    )

    messages.append({"role": "user", "content": user_content})

    try:
        client = get_client()
    except Exception as exc:
        print(f"[analysis] client error: {exc}")
        return {"answerMarkdown": ANALYSIS_FALLBACK_ANSWER}

    included_orders = len(payload_dataset.get("orders") or [])
    included_processing = len(payload_dataset.get("processing_orders") or [])

    print(
        "[analysis] dataset_chars="
        f"{len(serialized_dataset)} truncated={truncated} orders={included_orders}/{totals.get('orders_total', included_orders)}"
        f" processing={included_processing}/{totals.get('processing_total', included_processing)}"
    )

    try:
        completion = client.chat.completions.create(
            model=ANALYSIS_MODEL,
            messages=messages,
            temperature=0.3,
        )
    except Exception as exc:
        print(f"[analysis] request failed: {exc}")
        return {"answerMarkdown": ANALYSIS_FALLBACK_ANSWER}

    choice = completion.choices[0] if completion and completion.choices else None
    content = choice.message.content if choice and getattr(choice, "message", None) else None
    if not content:
        return {"answerMarkdown": ANALYSIS_FALLBACK_ANSWER}

    answer = content.strip()
    if not answer:
        return {"answerMarkdown": ANALYSIS_FALLBACK_ANSWER}

    session_messages = base_messages[:]
    session_messages.append({"role": "user", "content": question})
    session_messages.append({"role": "assistant", "content": answer})

    while len(session_messages) > 1 and (len(session_messages) - 1) // 2 > ANALYSIS_MAX_TURNS:
        if len(session_messages) > 2:
            session_messages.pop(1)
            session_messages.pop(1)
        else:
            break

    analysis_memory[session_id] = {
        "messages": session_messages,
        "dataset_hash": dataset_hash,
    }

    return {"answerMarkdown": answer}

@app.get("/diag")
def diag():
    try:
        sdk_ver = getattr(openai_pkg, "__version__", "unknown")
    except Exception:
        sdk_ver = "unknown"
    try:
        httpx_ver = getattr(httpx, "__version__", "unknown")
    except Exception:
        httpx_ver = "unknown"
    return {
        "status": "ok",
        "openai_sdk_version": sdk_ver,
        "httpx_version": httpx_ver,
        "env_has_api_key": bool(os.getenv("OPENAI_API_KEY")),
        "has_http_proxy_env": bool(os.getenv("HTTP_PROXY") or os.getenv("http_proxy")),
        "has_https_proxy_env": bool(os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")),
    }
