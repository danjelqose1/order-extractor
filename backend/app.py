from __future__ import annotations
import asyncio
import base64, csv, hashlib, json, os, re, sys
from collections import deque
from contextvars import ContextVar
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from io import StringIO
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional, Set, Tuple, Union
import uuid
import threading
import time
import logging
import random

BACKEND_DIR = os.path.dirname(__file__)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from schema import ExtractionResult, Row
from llm import (
    call_llm_for_extraction,
    call_llm_for_extraction_multi,
    call_llm_for_pdf_base64_visual,
    call_llm_for_image_visual,
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
    update_order_status,
    get_orders,
    get_analysis_orders,
    get_orders_by_identifiers,
    get_order_with_extraction,
    delete_order,
    get_all_rows_for_export,
    save_correction,
    list_corrections,
    delete_correction,
    normalize_order_status,
    is_processing_eligible_status,
    ORDER_STATUS_SEQUENCE,
    APPROVABLE_STATUSES,
    create_or_get_telegram_intake,
    create_telegram_file_record,
    update_telegram_file_record,
    touch_telegram_file_record,
    soft_delete_telegram_file_record,
    mark_telegram_file_labels_printed,
    mark_telegram_file_linked_order_opened,
    mark_telegram_file_pdf_printed,
    list_telegram_files,
    count_untouched_telegram_files,
    get_telegram_file_counts,
    find_telegram_file_record,
    find_telegram_file_by_sha256,
    find_possible_duplicate_order,
    list_unfinished_telegram_file_ids,
    get_telegram_file,
)
import db as db_module
from validators import validate_rows
from dimension_repair import apply_dimension_repair
from area_dimension_validator import apply_area_dimension_validation
from backend.agents.skills.extraction_diagnostics import (
    attach_pdf_row_locations,
    diagnose_extraction_row_issue,
    diagnose_extraction_row_warning,
    extract_pdf_text_for_row_location,
    extract_pdf_text_layer_text,
    ocr_fallback_row_repair,
)
from backend.agents.skills.family_pattern import attach_family_pattern_diagnostic
from backend.agents.repair_orchestrator import repair_suspicious_row
from extraction_normalizer import normalize_extracted_rows
from utils_text import build_order_total_diagnostics, clean_dimension, parse_declared_totals
from prompts import PROMPTS
from analysis_signals import generate_analysis_signals
from analytics_summary import ANALYTICS_STATUSES, build_analysis_summary
from services.pdf_native_text_editor import native_text_replace
from invoice_ai import analyze_invoice_line, match_invoice_glass_type
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH, override=True)

# Data/config paths
DATA_DIR = Path(os.getenv("DB_DIR", "data"))
PRICE_CONFIG_PATH = DATA_DIR / "price-config.json"
INVOICES_PATH = DATA_DIR / "invoices.json"

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
APP_KEY = os.getenv("APP_KEY")  # optional shared secret
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gpt-5.4-nano")
LEGACY_OCR_ENABLED = os.getenv("LEGACY_OCR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
TELEGRAM_MAX_FILE_BYTES = 5 * 1024 * 1024
TELEGRAM_SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"
TELEGRAM_EXTRACTION_MAX_RETRIES = 2
TELEGRAM_NETWORK_MAX_ATTEMPTS = 4
TELEGRAM_GET_FILE_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=5.0)
TELEGRAM_FILE_DOWNLOAD_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
TELEGRAM_REPLY_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
try:
    TELEGRAM_EXTRACTION_CONCURRENCY = max(1, int(os.getenv("TELEGRAM_EXTRACTION_CONCURRENCY", "1")))
except ValueError:
    TELEGRAM_EXTRACTION_CONCURRENCY = 1

_telegram_queue: Deque[int] = deque()
_telegram_queued_ids: Set[int] = set()
_telegram_queue_lock = threading.Lock()
_telegram_queue_task: Optional[asyncio.Task] = None
_telegram_event_clients: Set[asyncio.Queue] = set()
_telegram_event_clients_lock = threading.Lock()
_telegram_event_loop: Optional[asyncio.AbstractEventLoop] = None
_telegram_log_context: ContextVar[Dict[str, Any]] = ContextVar("telegram_log_context", default={})
logger = logging.getLogger(__name__)

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
_awa_action_state: Dict[str, Dict[str, Any]] = {}

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


@app.on_event("startup")
async def load_workspace_agent_modules() -> None:
    try:
        import workspace_agent as _workspace_agent_module  # noqa: F401
        from workspace_agents import smart_chat as _smart_chat_module  # noqa: F401
    except Exception as exc:
        print("IMPORT ERROR:", exc)
        raise
    _recover_telegram_extraction_queue()

class PasteIn(BaseModel):
    text: str


class ApprovePayload(BaseModel):
    rows: List[Row]
    notes: Optional[str] = None
    client_name: Optional[str] = None
    client: Optional[str] = None
    clientName: Optional[str] = None


class ExtractionRowDiagnosisPayload(BaseModel):
    row: Dict[str, Any]
    diagnostics: Optional[Dict[str, Any]] = None
    order_context: Optional[Dict[str, Any]] = None


class ExtractionRowOcrFallbackPayload(BaseModel):
    order_id: Optional[Union[str, int]] = None
    pdf_id: Optional[str] = None
    row_index: int = 0
    row: Dict[str, Any]
    diagnostics: Optional[Dict[str, Any]] = None
    target_field: Optional[str] = None
    order_context: Optional[Dict[str, Any]] = None


class ExtractionRowRepairPayload(BaseModel):
    order_id: Optional[Union[str, int]] = None
    pdf_id: Optional[str] = None
    row_index: int = 0
    row: Dict[str, Any]
    diagnostics: Optional[Dict[str, Any]] = None
    nearby_rows: Optional[List[Dict[str, Any]]] = None
    order_rows: Optional[List[Dict[str, Any]]] = None
    order_context: Optional[Dict[str, Any]] = None
    optional_pdf_context: Optional[Dict[str, Any]] = None
    target_field: Optional[str] = None


class StatusUpdatePayload(BaseModel):
    status: str
    note: Optional[str] = None


class AnalysisAskPayload(BaseModel):
    question: str
    dataset: Dict[str, Any]
    settings: Optional[Dict[str, Any]] = None


class AnalysisSignalsPayload(BaseModel):
    summary: Dict[str, Any]


class NativeTextReplacePayload(BaseModel):
    pdfBase64: str
    pageIndex: int
    originalText: str
    replacementText: str
    bounds: Optional[Dict[str, float]] = None
    fontSize: Optional[float] = None
    fontFamily: Optional[str] = None
    nativeEditMode: Optional[str] = "content_stream_first"
    preserveNearbyLines: Optional[bool] = True


class WorkspaceAgentPayload(BaseModel):
    message: str
    selected_order_id: Optional[str] = None
    selected_order_number: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class WorkspaceProcessPayload(BaseModel):
    order_number: Optional[str] = None
    order_id: Optional[str] = None
    force: Optional[bool] = False


class WorkspaceConfirmPayload(BaseModel):
    action: str
    order_id: Optional[str] = None
    order_number: Optional[str] = None
    force: Optional[bool] = False
    pending_action_id: Optional[str] = None
    decision: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class AwaActionPayload(BaseModel):
    action_id: str


class AwaExplainPayload(BaseModel):
    action_id: Optional[str] = None
    question: Optional[str] = None


class InvoiceAiGlassMatchPayload(BaseModel):
    raw_name: str
    known_types: List[str]


class InvoiceAiLineAnalysisPayload(BaseModel):
    raw_line: str
    known_glass_types: List[str]


class ManualOrderRowPayload(BaseModel):
    position: str = ""
    glass_type: str = Field(min_length=1)
    width_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)
    quantity: int = Field(gt=0)
    area_override_m2: Optional[float] = Field(default=None, ge=0)
    notes: str = ""


class ManualOrderPayload(BaseModel):
    client_name: str = Field(min_length=1)
    order_number: str = Field(min_length=1)
    order_date: date
    notes: str = ""
    status: Literal["draft", "approved", "processing", "finished", "cancelled"] = "draft"
    rows: List[ManualOrderRowPayload] = Field(min_length=1)


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


def _compute_hash_bytes(content: bytes) -> Optional[str]:
    if not content:
        return None
    return hashlib.sha1(content).hexdigest()


def _compute_sha256_bytes(content: bytes) -> Optional[str]:
    if not content:
        return None
    return hashlib.sha256(content).hexdigest()


def _dedupe_warnings(warnings: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for warning in warnings or []:
        text = str(warning).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _normalize_client_name_payload(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text_value = str(value).strip()
        if text_value and text_value not in {"-", "\u2014"}:
            return text_value
    return ""


def _pdf_data_url(pdf_bytes: bytes) -> str:
    encoded = base64.b64encode(pdf_bytes).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


def _file_data_url(file_bytes: bytes, content_type: str) -> str:
    encoded = base64.b64encode(file_bytes).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _pdf_bytes_from_data_url(value: Any) -> Optional[bytes]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.lower().startswith("data:"):
        content_type, _, encoded = text.partition(",")
        if "application/pdf" not in content_type.lower() or not encoded:
            return None
    else:
        encoded = text
    try:
        pdf_bytes = base64.b64decode(encoded, validate=True)
    except Exception:
        return None
    return pdf_bytes if pdf_bytes.startswith(b"%PDF") else None


def _stored_pdf_bytes_for_order_id(order_id: Optional[str]) -> Optional[bytes]:
    if not order_id or not str(order_id).isdigit():
        return None
    try:
        order = get_order_with_extraction(int(order_id))
    except Exception:
        return None
    extraction = (order or {}).get("extraction") or {}
    return _pdf_bytes_from_data_url(extraction.get("raw_input"))


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


def _declared_totals_from_sources(bundle: Dict[str, Any], *texts: Any) -> Tuple[Optional[int], Optional[float]]:
    declared_units = bundle.get("declared_units") if isinstance(bundle, dict) else None
    declared_area = bundle.get("declared_area") if isinstance(bundle, dict) else None
    for text in texts:
        parsed_units, parsed_area = parse_declared_totals(str(text or ""))
        if declared_units is None and parsed_units is not None:
            declared_units = parsed_units
        if declared_area is None and parsed_area is not None:
            declared_area = parsed_area
        if declared_units is not None and declared_area is not None:
            break
    return declared_units, declared_area


def _declared_totals_with_fallback(
    primary_units: Optional[int],
    primary_area: Optional[float],
    fallback_units: Optional[int],
    fallback_area: Optional[float],
) -> Tuple[Optional[int], Optional[float]]:
    return (
        primary_units if primary_units is not None else fallback_units,
        primary_area if primary_area is not None else fallback_area,
    )


def _order_total_diagnostics_from_totals(
    declared_units: Optional[int],
    declared_area: Optional[float],
    totals: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    return build_order_total_diagnostics(
        declared_units,
        declared_area,
        int(totals.get("units") or 0),
        float(totals.get("area") or 0.0),
    )


def _append_order_total_warning(
    warnings: List[str],
    diagnostics: Optional[Dict[str, Any]],
) -> List[str]:
    if diagnostics and diagnostics.get("message"):
        warnings.append(str(diagnostics["message"]))
    return warnings


def _with_extraction_diagnostics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    order_rows = [dict(row) for row in rows or [] if isinstance(row, dict)]
    for row in rows or []:
        working = dict(row)
        diagnostics = diagnose_extraction_row_issue(working)
        diagnostics = attach_family_pattern_diagnostic(
            working,
            diagnostics,
            order_rows=order_rows,
            order_context={"order_rows": order_rows},
        )
        working["diagnostics"] = diagnostics
        output.append(working)
    return output


CRITICAL_EXTRACTION_FIELDS: Tuple[str, ...] = ("position", "dimension", "quantity", "type", "area")
REPAIR_FIELD_KEYS: Dict[str, str] = {
    "position": "position_repaired",
    "dimension": "dimension_repaired",
    "quantity": "quantity_repaired",
    "type": "type_repaired",
    "area": "area_repaired",
}


def _critical_field_label(field: str) -> str:
    return "glass_type" if field == "type" else field


def _row_field_value(row: Dict[str, Any], field: str) -> Any:
    if field == "type":
        return row.get("type", row.get("glass_type"))
    return row.get(field)


def _is_critical_field_missing(row: Dict[str, Any], field: str) -> bool:
    value = _row_field_value(row, field)
    if field == "quantity":
        if value is None or str(value).strip() == "":
            return True
        try:
            return int(value) <= 0
        except (TypeError, ValueError):
            return True
    if field == "area":
        if value is None or str(value).strip() == "":
            return True
        try:
            return float(str(value).replace(",", ".")) <= 0
        except (TypeError, ValueError):
            return True
    return not str(value or "").strip()


def _raw_base64_values_for_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    source = row if isinstance(row, dict) else {}
    return {
        "position": source.get("position", ""),
        "dimension": source.get("dimension", ""),
        "quantity": source.get("quantity", ""),
        "type": source.get("type", source.get("glass_type", "")),
        "area": source.get("area", ""),
    }


def _merge_critical_row_warnings(
    row_warnings: Dict[Any, List[str]],
    rows: List[Dict[str, Any]],
) -> Dict[Any, List[str]]:
    merged: Dict[Any, List[str]] = {
        key: list(value or [])
        for key, value in (row_warnings or {}).items()
    }
    for idx, row in enumerate(rows or []):
        for field in CRITICAL_EXTRACTION_FIELDS:
            if not _is_critical_field_missing(row, field):
                continue
            message = f"warning: critical_missing:{_critical_field_label(field)}"
            messages = merged.setdefault(idx, [])
            if message not in messages:
                messages.append(message)
    return merged


def _row_repair_context(rows: List[Dict[str, Any]], index: int, order_number: str = "") -> Dict[str, Any]:
    return {
        "order_number": order_number or (rows[index].get("order_number") if 0 <= index < len(rows) else ""),
        "rows_before": [dict(row) for row in rows[max(0, index - 3):index]],
        "rows_after": [dict(row) for row in rows[index + 1:index + 4]],
        "order_rows": [dict(row) for row in rows],
    }


def _repair_confidence_threshold(field: str) -> float:
    if field == "position":
        return 0.78
    if field == "type":
        return 0.78
    return 0.80


def _coerce_repair_value(field: str, value: Any) -> Any:
    if value is None:
        return ""
    if field == "quantity":
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if field == "area":
        try:
            return round(float(str(value).replace(",", ".")), 3)
        except (TypeError, ValueError):
            return value
    return str(value).strip()


def _ocr_raw_value_from_result(result: Dict[str, Any]) -> Any:
    if result.get("suggested_value") is not None:
        return result.get("suggested_value")
    evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    return evidence.get("ocr_text") or evidence.get("matched_text") or ""


def _extract_ocr_overlay(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = {
        "raw_base64_value",
        "raw_ocr_value",
        "repaired_by",
        "repair_confidence",
        "repair_warning",
        "repair_warnings",
        "needs_manual_review",
        "ocr_repaired_fields",
        "ocr_repair_attempted_fields",
        "glass_type_repaired",
        *REPAIR_FIELD_KEYS.values(),
    }
    return {key: deepcopy(row.get(key)) for key in keys if key in row}


def _apply_ocr_overlay_to_row(row: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    working = dict(row)
    if not isinstance(overlay, dict):
        return working
    raw_values = overlay.get("raw_base64_value")
    if isinstance(raw_values, dict):
        working["raw_base64_value"] = deepcopy(raw_values)
    if isinstance(overlay.get("raw_ocr_value"), dict):
        working["raw_ocr_value"] = deepcopy(overlay["raw_ocr_value"])
    for field, repair_key in REPAIR_FIELD_KEYS.items():
        if repair_key not in overlay:
            continue
        if not _is_critical_field_missing(working, field):
            continue
        working[repair_key] = deepcopy(overlay[repair_key])
        if field == "type":
            working["glass_type_repaired"] = deepcopy(overlay[repair_key])
    active_repair_warnings: Dict[str, str] = {}
    overlay_warnings = overlay.get("repair_warnings")
    if isinstance(overlay_warnings, dict):
        for field, message in overlay_warnings.items():
            field_name = str(field)
            if field_name in REPAIR_FIELD_KEYS and _is_critical_field_missing(working, field_name):
                active_repair_warnings[field_name] = str(message or "Needs manual review")
    for key in ("repaired_by", "repair_confidence", "ocr_repaired_fields", "ocr_repair_attempted_fields"):
        if key in overlay:
            working[key] = deepcopy(overlay[key])
    if active_repair_warnings:
        working["repair_warnings"] = active_repair_warnings
        working["needs_manual_review"] = True
        labels = ", ".join(_critical_field_label(field) for field in active_repair_warnings.keys())
        working["repair_warning"] = f"Needs manual review: {labels}"
    return working


def _with_stored_ocr_overlays(
    rows: List[Dict[str, Any]],
    extraction: Dict[str, Any],
) -> List[Dict[str, Any]]:
    raw_json = extraction.get("llm_output_json") if isinstance(extraction, dict) else None
    if not raw_json:
        return rows
    try:
        payload = json.loads(raw_json)
    except Exception:
        return rows
    meta = payload.get("_meta") if isinstance(payload, dict) else None
    if not isinstance(meta, dict):
        return rows
    overlays = meta.get("ocr_row_overlays")
    if not isinstance(overlays, list):
        return rows
    output: List[Dict[str, Any]] = []
    for index, row in enumerate(rows or []):
        overlay = overlays[index] if index < len(overlays) and isinstance(overlays[index], dict) else {}
        output.append(_apply_ocr_overlay_to_row(dict(row), overlay))
    return output


def _run_targeted_ocr_repair_for_missing_fields(
    rows: List[Dict[str, Any]],
    raw_base64_rows: List[Dict[str, Any]],
    *,
    pdf_bytes: Optional[bytes],
    pdf_id: Optional[str],
    order_number: str = "",
    enabled: bool = True,
) -> Dict[str, Any]:
    repaired_rows: List[Dict[str, Any]] = []
    repair_attempts: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for index, row in enumerate(rows or []):
        working = dict(row)
        raw_row = raw_base64_rows[index] if index < len(raw_base64_rows) and isinstance(raw_base64_rows[index], dict) else {}
        working["raw_base64_value"] = _raw_base64_values_for_row(raw_row or row)
        working.setdefault("raw_ocr_value", {})

        target_fields = [
            field
            for field in CRITICAL_EXTRACTION_FIELDS
            if _is_critical_field_missing(working, field)
        ]
        if not target_fields:
            repaired_rows.append(working)
            continue

        attempted_fields: List[str] = []
        repaired_fields: List[str] = []
        repair_warnings: Dict[str, str] = {}
        best_confidence = 0.0
        methods: List[str] = []
        order_context = _row_repair_context(rows, index, order_number=order_number)

        for field in target_fields:
            attempted_fields.append(field)
            if not enabled:
                repair_warnings[field] = "Needs manual review"
                repair_attempts.append(
                    {
                        "row_index": index,
                        "target_field": field,
                        "success": False,
                        "reason": "OCR fallback disabled for this extraction.",
                    }
                )
                continue

            diagnostics = diagnose_extraction_row_issue(working)
            result = ocr_fallback_row_repair(
                row=deepcopy(working),
                diagnostics=deepcopy(diagnostics),
                target_field=field,
                order_context=deepcopy(order_context),
                row_index=index,
                pdf_id=pdf_id,
                pdf_bytes=pdf_bytes,
            )
            confidence = float(result.get("confidence") or 0.0)
            raw_ocr_value = _ocr_raw_value_from_result(result)
            working.setdefault("raw_ocr_value", {})[field] = raw_ocr_value
            method = str(result.get("method") or result.get("repaired_by") or "ocr_fallback")
            if method and method not in methods:
                methods.append(method)

            accepted = (
                bool(result.get("success"))
                and result.get("suggested_value") not in (None, "")
                and confidence >= _repair_confidence_threshold(field)
            )
            if accepted:
                repair_key = REPAIR_FIELD_KEYS[field]
                repaired_value = _coerce_repair_value(field, result.get("suggested_value"))
                working[repair_key] = repaired_value
                if field == "type":
                    working["glass_type_repaired"] = repaired_value
                repaired_fields.append(field)
                best_confidence = max(best_confidence, confidence)
            else:
                repair_warnings[field] = "Needs manual review"

            repair_attempts.append(
                {
                    "row_index": index,
                    "target_field": field,
                    "success": accepted,
                    "suggested_value": result.get("suggested_value"),
                    "confidence": confidence,
                    "method": method,
                    "reason": result.get("reason") or result.get("reasoning") or "",
                    "raw_base64_value": working["raw_base64_value"].get(field),
                    "raw_ocr_value": raw_ocr_value,
                }
            )

        if attempted_fields:
            working["ocr_repair_attempted_fields"] = attempted_fields
        if repaired_fields:
            working["ocr_repaired_fields"] = repaired_fields
            working["repaired_by"] = ", ".join(methods) if methods else "ocr_fallback"
            working["repair_confidence"] = round(best_confidence, 2)
        if repair_warnings:
            working["repair_warnings"] = repair_warnings
            working["needs_manual_review"] = True
            labels = ", ".join(_critical_field_label(field) for field in repair_warnings.keys())
            working["repair_warning"] = f"Needs manual review: {labels}"
            warnings.append(f"Row {index + 1}: Needs manual review after OCR fallback ({labels}).")

        repaired_rows.append(working)

    return {
        "rows": repaired_rows,
        "repair_attempts": repair_attempts,
        "row_overlays": [_extract_ocr_overlay(row) for row in repaired_rows],
        "warnings": warnings,
    }


def _stored_row_locations(extraction: Dict[str, Any]) -> List[Any]:
    raw_json = extraction.get("llm_output_json") if isinstance(extraction, dict) else None
    if not raw_json:
        return []
    try:
        payload = json.loads(raw_json)
    except Exception:
        return []
    meta = payload.get("_meta") if isinstance(payload, dict) else None
    locations = meta.get("row_locations") if isinstance(meta, dict) else None
    return locations if isinstance(locations, list) else []


def _stored_declared_totals(extraction: Dict[str, Any]) -> Tuple[Optional[int], Optional[float]]:
    raw_json = extraction.get("llm_output_json") if isinstance(extraction, dict) else None
    if not raw_json:
        return None, None
    try:
        payload = json.loads(raw_json)
    except Exception:
        return None, None
    meta = payload.get("_meta") if isinstance(payload, dict) else None
    if not isinstance(meta, dict):
        return None, None
    units = meta.get("declared_units")
    area = meta.get("declared_area")
    try:
        units = int(units) if units is not None else None
    except (TypeError, ValueError):
        units = None
    try:
        area = float(area) if area is not None else None
    except (TypeError, ValueError):
        area = None
    return units, area


def _with_stored_row_locations(
    rows: List[Dict[str, Any]],
    extraction: Dict[str, Any],
) -> List[Dict[str, Any]]:
    locations = _stored_row_locations(extraction)
    if not locations:
        return rows
    output: List[Dict[str, Any]] = []
    for index, row in enumerate(rows or []):
        working = dict(row)
        if "row_location" not in working and index < len(locations):
            location = locations[index]
            working["row_location"] = location if isinstance(location, dict) else None
        output.append(working)
    return output


def _optional_id_to_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_pdf_file(filename: str, content_type: str) -> bool:
    return (content_type or "").lower() == "application/pdf" or (filename or "").lower().endswith(".pdf")


def _is_image_file(filename: str, content_type: str) -> bool:
    lowered = (filename or "").lower()
    return (content_type or "").lower().startswith("image/") or lowered.endswith((".png", ".jpg", ".jpeg", ".webp"))


def _telegram_files_dir() -> Path:
    path = DATA_DIR / "telegram-files"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_download_filename(value: Any) -> str:
    basename = Path(str(value or "telegram-order.pdf")).name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", basename).strip(".-")
    return safe or "telegram-order.pdf"


def _store_telegram_file(raw_bytes: bytes, original_filename: str) -> Dict[str, Any]:
    safe_original = _safe_download_filename(original_filename)
    suffix = Path(safe_original).suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".pdf"
    stored_filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex}{suffix}"
    target_dir = _telegram_files_dir()
    target_path = (target_dir / stored_filename).resolve()
    root_path = target_dir.resolve()
    if root_path not in target_path.parents:
        raise RuntimeError("Invalid Telegram file path")
    target_path.write_bytes(raw_bytes)
    return {
        "original_filename": safe_original,
        "stored_filename": stored_filename,
        "file_path": str(target_path),
        "file_size": len(raw_bytes),
    }


def _store_telegram_pdf(raw_bytes: bytes, original_filename: str) -> Dict[str, Any]:
    return _store_telegram_file(raw_bytes, original_filename)


def _safe_telegram_file_path(record: Dict[str, Any]) -> Path:
    root = _telegram_files_dir().resolve()
    path = Path(str(record.get("file_path") or "")).expanduser().resolve()
    if path == root or root not in path.parents:
        raise HTTPException(status_code=404, detail="File not found")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File missing on disk")
    return path


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


@app.post("/api/invoices/ai/glass-match")
def invoice_ai_glass_match(payload: InvoiceAiGlassMatchPayload) -> Dict[str, Any]:
    try:
        match = match_invoice_glass_type(
            get_client(),
            raw_name=payload.raw_name,
            known_types=payload.known_types,
        )
        return {"match": match}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("invoice AI glass match failed")
        raise HTTPException(status_code=502, detail="Invoice AI glass matching is unavailable.") from exc


@app.post("/api/invoices/ai/analyze-line")
def invoice_ai_analyze_line(payload: InvoiceAiLineAnalysisPayload) -> Dict[str, Any]:
    try:
        analysis = analyze_invoice_line(
            get_client(),
            raw_line=payload.raw_line,
            known_glass_types=payload.known_glass_types,
        )
        return {"analysis": analysis}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("invoice AI line analysis failed")
        raise HTTPException(status_code=502, detail="Invoice AI line analysis is unavailable.") from exc


@app.get("/healthz")
def healthz():
    return {"ok": True}


def _awa_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _awa_order_ref(item: Dict[str, Any]) -> Tuple[List[Any], List[str]]:
    order_id = item.get("order_id", item.get("id"))
    order_ids = [order_id] if order_id is not None else []
    order_number = str(item.get("order_number") or "").strip()
    order_numbers = [order_number] if order_number else []
    return order_ids, order_numbers


def _awa_action(
    *,
    action_id: str,
    action_type: str,
    title: str,
    explanation: str,
    order_ids: Optional[List[Any]] = None,
    order_numbers: Optional[List[str]] = None,
    confidence: float = 0.75,
    safety_level: str = "review",
    result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    stored = _awa_action_state.get(action_id) or {}
    return {
        "id": action_id,
        "type": action_type,
        "status": stored.get("status") or "suggested",
        "title": title,
        "explanation": explanation,
        "order_ids": order_ids or [],
        "order_numbers": order_numbers or [],
        "confidence": round(float(confidence), 2),
        "safety_level": safety_level,
        "created_at": stored.get("created_at") or _awa_now(),
        "result": stored.get("result", result),
    }


def _awa_type_key_for_order(order_id: Any) -> str:
    try:
        detail = get_order_with_extraction(int(order_id))
    except Exception:
        detail = None
    rows = (detail or {}).get("rows") or []
    for row in rows:
        glass_type = str(row.get("type") or "").strip()
        if glass_type:
            return re.sub(r"\s+", " ", glass_type.lower())
    return ""


def _awa_workspace_snapshot() -> Dict[str, Any]:
    from workspace_service import get_recent_production_files, get_workspace_queue

    queue = get_workspace_queue()
    recent = get_recent_production_files(limit=25)
    return {
        "groups": queue.get("groups") or {},
        "counts": queue.get("counts") or {},
        "recent_files": recent.get("items") or [],
    }


def _awa_build_suggestions(snapshot: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    snapshot = snapshot or _awa_workspace_snapshot()
    groups = snapshot.get("groups") or {}
    suggestions: List[Dict[str, Any]] = []

    approved_ready = list(groups.get("approved_ready") or [])
    needs_review = list(groups.get("needs_review") or [])
    processing_done = list(groups.get("processing_done") or [])
    labels_ready = list(groups.get("labels_ready") or [])
    recent_files = list(snapshot.get("recent_files") or [])

    for item in approved_ready[:8]:
        order_ids, order_numbers = _awa_order_ref(item)
        order_label = order_numbers[0] if order_numbers else f"#{order_ids[0]}" if order_ids else "approved order"
        suggestions.append(_awa_action(
            action_id=f"awa-process-{order_ids[0] if order_ids else order_label}",
            action_type="process_approved_order",
            title="Process approved order",
            explanation=f"{order_label} is approved and has no production batch recorded yet. AWA can prepare the handoff, but processing still needs supervised approval.",
            order_ids=order_ids,
            order_numbers=order_numbers,
            confidence=0.91,
            safety_level="safe",
        ))

    type_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in approved_ready:
        order_id = item.get("order_id", item.get("id"))
        key = _awa_type_key_for_order(order_id)
        if key:
            type_groups.setdefault(key, []).append(item)
    for key, items in type_groups.items():
        if len(items) < 2:
            continue
        order_ids: List[Any] = []
        order_numbers: List[str] = []
        for item in items[:6]:
            ids, nums = _awa_order_ref(item)
            order_ids.extend(ids)
            order_numbers.extend(nums)
        suggestions.append(_awa_action(
            action_id="awa-group-" + "-".join(str(item) for item in order_ids),
            action_type="process_selected_orders_together",
            title="Process selected orders together",
            explanation=f"{len(order_ids)} approved orders share a similar glass type profile. AWA suggests reviewing them as a supervised processing group.",
            order_ids=order_ids,
            order_numbers=order_numbers,
            confidence=0.78,
            safety_level="requires_confirmation",
        ))
        break

    for item in needs_review[:8]:
        warnings_count = int(item.get("warnings_count") or 0)
        if warnings_count <= 0:
            continue
        order_ids, order_numbers = _awa_order_ref(item)
        order_label = order_numbers[0] if order_numbers else f"#{order_ids[0]}" if order_ids else "draft order"
        suggestions.append(_awa_action(
            action_id=f"awa-review-{order_ids[0] if order_ids else order_label}",
            action_type="review_suspicious_order",
            title="Review suspicious order",
            explanation=f"{order_label} is still draft/review and has {warnings_count} validation warning(s). Draft orders must be reviewed before any production action.",
            order_ids=order_ids,
            order_numbers=order_numbers,
            confidence=0.86,
            safety_level="review",
        ))

    for item in processing_done[:6]:
        order_ids, order_numbers = _awa_order_ref(item)
        order_label = order_numbers[0] if order_numbers else f"#{order_ids[0]}" if order_ids else "processed order"
        suggestions.append(_awa_action(
            action_id=f"awa-labels-{item.get('batch_id') or (order_ids[0] if order_ids else order_label)}",
            action_type="create_labels_from_processing",
            title="Create labels from processing",
            explanation=f"{order_label} has a processing batch but no labels file is visible in the Workspace queue.",
            order_ids=order_ids,
            order_numbers=order_numbers,
            confidence=0.82,
            safety_level="requires_confirmation",
        ))

    for item in labels_ready[:6]:
        order_ids, order_numbers = _awa_order_ref(item)
        order_label = order_numbers[0] if order_numbers else f"#{order_ids[0]}" if order_ids else "production order"
        suggestions.append(_awa_action(
            action_id=f"awa-invoice-{item.get('batch_id') or (order_ids[0] if order_ids else order_label)}",
            action_type="create_invoice_draft",
            title="Create invoice draft",
            explanation=f"{order_label} appears ready after production file preparation. AWA can suggest an invoice draft, but final invoices stay manual.",
            order_ids=order_ids,
            order_numbers=order_numbers,
            confidence=0.72,
            safety_level="review",
        ))

    for item in recent_files[:5]:
        order_number = str(item.get("order_number") or "").strip()
        if not (item.get("processing_pdf_url") or item.get("labels_pdf_url")):
            continue
        suggestions.append(_awa_action(
            action_id=f"awa-download-{item.get('batch_id') or order_number or len(suggestions)}",
            action_type="download_ready_production_files",
            title="Download ready production files",
            explanation=f"Production files are ready for {order_number or 'a recent batch'}. AWA can point you to the files; it will not print or send them.",
            order_ids=[item.get("order_id")] if item.get("order_id") is not None else [],
            order_numbers=[order_number] if order_number else [],
            confidence=0.88,
            safety_level="safe",
        ))

    return suggestions[:24]


def _awa_build_timeline(snapshot: Optional[Dict[str, Any]] = None, suggestions: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    snapshot = snapshot or _awa_workspace_snapshot()
    suggestions = suggestions if suggestions is not None else _awa_build_suggestions(snapshot)
    groups = snapshot.get("groups") or {}
    events: List[Dict[str, Any]] = []
    now = _awa_now()

    for item in list(groups.get("needs_review") or [])[:8]:
        order_number = str(item.get("order_number") or "").strip()
        warnings_count = int(item.get("warnings_count") or 0)
        events.append({
            "timestamp": item.get("created_at") or now,
            "order_number": order_number,
            "title": "Order extracted",
            "explanation": "Draft order is waiting for manual review.",
            "status": "review",
        })
        if warnings_count:
            events.append({
                "timestamp": item.get("created_at") or now,
                "order_number": order_number,
                "title": "Warning detected",
                "explanation": f"{warnings_count} validation warning(s) need review before production.",
                "status": "warning",
            })

    for item in list(groups.get("approved_ready") or [])[:8]:
        events.append({
            "timestamp": item.get("approved_at") or item.get("created_at") or now,
            "order_number": item.get("order_number") or "",
            "title": "Validation passed",
            "explanation": "Approved order is eligible for supervised processing.",
            "status": "safe",
        })

    for item in list(snapshot.get("recent_files") or [])[:8]:
        order_number = str(item.get("order_number") or "").strip()
        if item.get("processing_pdf_url"):
            events.append({
                "timestamp": item.get("generated_at") or now,
                "order_number": order_number,
                "title": "Processing draft prepared",
                "explanation": "Production processing file exists in Workspace files.",
                "status": "ready",
            })
        if item.get("labels_pdf_url"):
            events.append({
                "timestamp": item.get("generated_at") or now,
                "order_number": order_number,
                "title": "Labels draft prepared",
                "explanation": "Label file exists for supervised printing/download.",
                "status": "ready",
            })

    for suggestion in suggestions[:8]:
        events.append({
            "timestamp": suggestion.get("created_at") or now,
            "order_number": ", ".join(suggestion.get("order_numbers") or []),
            "title": "Waiting for approval",
            "explanation": suggestion.get("title") or "Suggested action",
            "status": suggestion.get("safety_level") or "review",
        })

    return sorted(events, key=lambda event: str(event.get("timestamp") or ""), reverse=True)[:30]


def _awa_find_suggestion(action_id: str) -> Optional[Dict[str, Any]]:
    for suggestion in _awa_build_suggestions():
        if str(suggestion.get("id")) == str(action_id):
            return suggestion
    stored = _awa_action_state.get(str(action_id))
    if stored:
        return {"id": action_id, **stored}
    return None


def _awa_chat_answer(question: str) -> str:
    snapshot = _awa_workspace_snapshot()
    counts = snapshot.get("counts") or {}
    suggestions = _awa_build_suggestions(snapshot)
    lower = question.lower()
    if "safe" in lower or "process" in lower or "production" in lower:
        return (
            f"{counts.get('approved_ready', 0)} approved order(s) look ready for supervised processing. "
            "AWA only suggests processing for approved orders and does not process drafts."
        )
    if "review" in lower or "warning" in lower:
        return (
            f"{counts.get('needs_review', 0)} order(s) need review. Draft orders and warning cases stay manual before production."
        )
    if "next" in lower or "recommend" in lower:
        if suggestions:
            first = suggestions[0]
            return f"Next supervised recommendation: {first.get('title')}. {first.get('explanation')}"
        return "No supervised AWA recommendations are available right now."
    return (
        "AWA is in beta and read-only by default. It can explain ready orders, review items, production files, and why a suggestion was made."
    )


@app.get("/api/workspace/queue")
def workspace_queue() -> Dict[str, Any]:
    from workspace_service import get_workspace_queue

    return get_workspace_queue()


@app.get("/api/workspace/recent-files")
def workspace_recent_files() -> Dict[str, Any]:
    from workspace_service import get_recent_production_files

    return get_recent_production_files()


@app.get("/api/awa/summary")
def awa_summary() -> Dict[str, Any]:
    snapshot = _awa_workspace_snapshot()
    suggestions = _awa_build_suggestions(snapshot)
    counts = snapshot.get("counts") or {}
    waiting = len([item for item in suggestions if item.get("status") == "suggested"])
    return {
        "suggested_actions_today": len(suggestions),
        "ready_to_process": int(counts.get("approved_ready") or 0),
        "needs_review": int(counts.get("needs_review") or 0),
        "waiting_approval": waiting,
    }


@app.get("/api/awa/timeline")
def awa_timeline() -> Dict[str, Any]:
    snapshot = _awa_workspace_snapshot()
    suggestions = _awa_build_suggestions(snapshot)
    return {"items": _awa_build_timeline(snapshot, suggestions)}


@app.get("/api/awa/suggestions")
def awa_suggestions() -> Dict[str, Any]:
    return {"items": _awa_build_suggestions()}


@app.post("/api/awa/explain")
def awa_explain(payload: AwaExplainPayload) -> Dict[str, Any]:
    action_id = str(payload.action_id or "").strip()
    question = str(payload.question or "").strip()
    if action_id:
        suggestion = _awa_find_suggestion(action_id)
        if not suggestion:
            raise HTTPException(status_code=404, detail="AWA suggestion not found")
        orders = ", ".join(suggestion.get("order_numbers") or []) or "the affected order(s)"
        return {
            "message": (
                f"{suggestion.get('title')}: {suggestion.get('explanation')} "
                f"Affected orders: {orders}. Safety level: {suggestion.get('safety_level')}. "
                "AWA Beta will not mutate raw data or run production actions without supervised approval."
            ),
            "suggestion": suggestion,
        }
    if question:
        return {"message": _awa_chat_answer(question)}
    raise HTTPException(status_code=400, detail="action_id or question is required")


@app.post("/api/awa/approve-action")
def awa_approve_action(payload: AwaActionPayload) -> Dict[str, Any]:
    action_id = str(payload.action_id or "").strip()
    if not action_id:
        raise HTTPException(status_code=400, detail="action_id is required")
    suggestion = _awa_find_suggestion(action_id)
    if not suggestion:
        raise HTTPException(status_code=404, detail="AWA suggestion not found")
    result = {
        "message": "AWA action approval is not connected yet.",
        "safety_note": "No production data was changed. Use the related module to complete this workflow manually.",
    }
    _awa_action_state[action_id] = {
        "status": "approved",
        "created_at": suggestion.get("created_at") or _awa_now(),
        "result": result,
    }
    return {"status": "approved", "message": result["message"], "result": result, "suggestion": _awa_find_suggestion(action_id)}


@app.post("/api/awa/reject-action")
def awa_reject_action(payload: AwaActionPayload) -> Dict[str, Any]:
    action_id = str(payload.action_id or "").strip()
    if not action_id:
        raise HTTPException(status_code=400, detail="action_id is required")
    suggestion = _awa_find_suggestion(action_id)
    if not suggestion:
        raise HTTPException(status_code=404, detail="AWA suggestion not found")
    result = {"message": "Suggestion rejected for this session. No production data was changed."}
    _awa_action_state[action_id] = {
        "status": "rejected",
        "created_at": suggestion.get("created_at") or _awa_now(),
        "result": result,
    }
    return {"status": "rejected", "message": result["message"], "result": result, "suggestion": _awa_find_suggestion(action_id)}


@app.post("/api/workspace/process-order")
def workspace_process_order(payload: WorkspaceProcessPayload) -> Dict[str, Any]:
    identifier = payload.order_id or payload.order_number
    if not identifier:
        raise HTTPException(status_code=400, detail="order_id or order_number is required")
    return {
        "message": "Processing must run through the frontend Processing and Labels modules.",
        "status": "frontend_workflow_required",
        "actions": [{"type": "process_via_existing_modules", "identifier": str(identifier)}],
    }


@app.post("/api/workspace/confirm-action")
def workspace_confirm_action(payload: WorkspaceConfirmPayload) -> Dict[str, Any]:
    from workspace_service import confirm_workspace_pending_action

    if payload.pending_action_id:
        return confirm_workspace_pending_action(
            payload.pending_action_id,
            decision=payload.decision or "continue",
            requested_by="workspace_agent",
            context=payload.context,
        )

    from workspace_agent import _format_process_response
    from workspace_service import process_approved_order

    if payload.action not in {"process_order"}:
        raise HTTPException(status_code=400, detail="Unsupported confirmation action")
    identifier = payload.order_id or payload.order_number
    if not identifier:
        raise HTTPException(status_code=400, detail="order_id or order_number is required")
    result = process_approved_order(identifier, requested_by="workspace", force=bool(payload.force))
    return _format_process_response(result)


@app.post("/api/agent/workspace")
def workspace_agent(payload: WorkspaceAgentPayload) -> Dict[str, Any]:
    from workspace_agent import run_workspace_agent
    from workspace_agents.response_format import ensure_workspace_agent_response

    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    result = run_workspace_agent(
        message,
        selected_order_id=payload.selected_order_id,
        selected_order_number=payload.selected_order_number,
        context=payload.context,
        requested_by="workspace_agent",
    )
    return ensure_workspace_agent_response(result)


@app.post("/api/agent/smart-chat")
@app.post("/api/agent/workspace-chat")
def workspace_smart_chat(payload: WorkspaceAgentPayload) -> Dict[str, Any]:
    from workspace_agents.smart_chat import run_workspace_smart_chat

    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    return run_workspace_smart_chat(
        message,
        context=payload.context,
        requested_by="workspace_smart_chat",
    )


@app.get("/api/workspace/files/{file_id}/download")
def download_workspace_file(file_id: int):
    from workspace_service import get_production_file

    file = get_production_file(file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    path = Path(file.get("file_path") or "")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File missing on disk")
    filename = path.name
    return FileResponse(path, media_type="application/pdf", filename=filename)


@app.post("/api/extraction/diagnose-row")
def diagnose_extraction_row(payload: ExtractionRowDiagnosisPayload) -> Dict[str, Any]:
    row = deepcopy(payload.row or {})
    diagnostics = diagnose_extraction_row_issue(row)
    order_context = deepcopy(payload.order_context or {})
    nearby_rows = []
    if isinstance(order_context.get("rows_before"), list):
        nearby_rows.extend(item for item in order_context.get("rows_before") if isinstance(item, dict))
    if isinstance(order_context.get("rows_after"), list):
        nearby_rows.extend(item for item in order_context.get("rows_after") if isinstance(item, dict))
    order_rows = order_context.get("order_rows") if isinstance(order_context.get("order_rows"), list) else []
    diagnostics = attach_family_pattern_diagnostic(
        row,
        diagnostics,
        nearby_rows=nearby_rows,
        order_rows=order_rows,
        order_context=order_context,
    )
    diagnosis = diagnose_extraction_row_warning(
        deepcopy(row),
        deepcopy(diagnostics),
        order_context,
    )
    diagnosis["diagnostics"] = diagnostics
    return diagnosis


@app.post("/api/extraction/repair-row")
def repair_extraction_row(payload: ExtractionRowRepairPayload) -> Dict[str, Any]:
    row = deepcopy(payload.row or {})
    normalized_order_id = _optional_id_to_string(payload.order_id)
    order_context = deepcopy(payload.order_context or {})
    if normalized_order_id is not None:
        order_context.setdefault("order_id", normalized_order_id)
    pdf_bytes = _stored_pdf_bytes_for_order_id(normalized_order_id)
    row_location = row.get("row_location")
    if isinstance(row_location, dict):
        region_text = extract_pdf_text_for_row_location(pdf_bytes or b"", row_location)
        if region_text:
            updated_location = dict(row_location)
            updated_location["matched_text"] = region_text
            row["row_location"] = updated_location

    diagnostics = deepcopy(payload.diagnostics) if isinstance(payload.diagnostics, dict) else diagnose_extraction_row_issue(row)
    nearby_rows = deepcopy(payload.nearby_rows or [])
    order_rows = deepcopy(payload.order_rows or [])
    diagnostics = attach_family_pattern_diagnostic(
        row,
        diagnostics,
        nearby_rows=nearby_rows,
        order_rows=order_rows,
        order_context=order_context,
    )
    optional_pdf_context = deepcopy(payload.optional_pdf_context or {})
    if payload.pdf_id is not None:
        optional_pdf_context.setdefault("pdf_id", payload.pdf_id)
    return repair_suspicious_row(
        row=deepcopy(row),
        diagnostics=deepcopy(diagnostics),
        nearby_rows=nearby_rows,
        order_rows=order_rows,
        order_context=order_context,
        optional_pdf_context=optional_pdf_context,
        target_field=payload.target_field,
        row_index=payload.row_index,
        pdf_id=payload.pdf_id,
        pdf_bytes=pdf_bytes,
    )


@app.post("/api/extraction/ocr-fallback-row")
def ocr_fallback_extraction_row(payload: ExtractionRowOcrFallbackPayload) -> Dict[str, Any]:
    row = deepcopy(payload.row or {})
    diagnostics = diagnose_extraction_row_issue(row)
    normalized_order_id = _optional_id_to_string(payload.order_id)
    order_context = deepcopy(payload.order_context or {})
    if normalized_order_id is not None:
        order_context.setdefault("order_id", normalized_order_id)
    pdf_bytes = _stored_pdf_bytes_for_order_id(normalized_order_id)
    row_location = row.get("row_location")
    if isinstance(row_location, dict):
        region_text = extract_pdf_text_for_row_location(pdf_bytes or b"", row_location)
        if region_text:
            updated_location = dict(row_location)
            updated_location["matched_text"] = region_text
            row["row_location"] = updated_location
    return ocr_fallback_row_repair(
        row=deepcopy(row),
        diagnostics=deepcopy(diagnostics),
        target_field=payload.target_field,
        order_context=order_context,
        row_index=payload.row_index,
        pdf_id=payload.pdf_id,
        pdf_bytes=pdf_bytes,
    )


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
        extracted_rows, normalization_warnings = normalize_extracted_rows(repaired_rows)
        validation = validate_rows(extracted_rows, context={"prepared_text": bundle.get("prepared_text")})
        normalized_rows = validation.get("rows", extracted_rows)
        final_rows = apply_area_dimension_validation(normalized_rows)
        row_warnings = validation.get("row_warnings", {})

        combined_warnings: List[str] = []
        for source_list in (
            raw_payload.get("warnings") or [],
            (bundle.get("data") or {}).get("warnings") or [],
            normalization_warnings,
            validation.get("warnings", []) or [],
        ):
            combined_warnings.extend(source_list)

        applied = bundle.get("applied_corrections") or []
        if applied:
            combined_warnings.append(f"auto_corrections_applied:{','.join(str(i) for i in applied)}")

        prepared_text = bundle.get("prepared_text") or inb.text.strip()
        data_payload = bundle.get("data") or {}
        client_name = _normalize_client_name_payload(
            data_payload.get("client_name"),
            data_payload.get("clientName"),
            data_payload.get("client"),
            raw_payload.get("client_name") if isinstance(raw_payload, dict) else None,
            raw_payload.get("clientName") if isinstance(raw_payload, dict) else None,
            raw_payload.get("client") if isinstance(raw_payload, dict) else None,
            extract_client_name(inb.text),
            extract_client_name(prepared_text),
        )
        hash_value = _compute_hash(prepared_text or inb.text)
        llm_output_json = json.dumps(raw_payload, ensure_ascii=False)
        declared_units, declared_area = _declared_totals_from_sources(
            bundle,
            prepared_text,
            inb.text,
            bundle.get("output_text"),
        )
        totals = _summarize_totals(final_rows)
        order_total_diagnostics = _order_total_diagnostics_from_totals(declared_units, declared_area, totals)
        _append_order_total_warning(combined_warnings, order_total_diagnostics)
        combined_warnings = _dedupe_warnings(combined_warnings)
        response_rows = _with_extraction_diagnostics(final_rows)

        insert_result = insert_extraction_with_rows(
            source="paste",
            rows=final_rows,
            raw_input=inb.text,
            prepared_text=prepared_text,
            llm_output_json=llm_output_json,
            model_used=bundle.get("model_used"),
            hash_value=hash_value,
            confidence=(bundle.get("data") or {}).get("confidence"),
            client_name=client_name,
        )
        draft_order_id = insert_result["order_id"]
        status = insert_result.get("status", "draft")
        saved_order_id = draft_order_id if status == "approved" else None

        payload = {
            "order_number": (bundle.get("data") or {}).get("order_number") or _primary_order_number(final_rows),
            "rows": response_rows,
            "warnings": combined_warnings,
            "row_warnings": row_warnings,
            "draft_order_id": draft_order_id,
            "saved_order_id": saved_order_id,
            "status": status,
            "version": insert_result.get("version", 1),
            "source_hash": insert_result.get("source_hash"),
            "created_new_version": bool(insert_result.get("created_new_version")),
            "protected_order_id": insert_result.get("protected_order_id"),
            "applied_corrections": applied,
            "model_used": bundle.get("model_used"),
            "declared_units": declared_units,
            "declared_area": declared_area,
            "parsed_units": totals["units"],
            "parsed_area": totals["area"],
            "order_total_diagnostics": order_total_diagnostics,
            "client_name": client_name,
            "clientName": client_name,
            "client": client_name or "—",
        }
        return payload
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=f"Validation error: {ve.errors()}")
    except Exception as e:
        print("[extract] fatal error:\n" + "".join(traceback.format_exc()))
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


def _extract_pdf_via_legacy_ocr(raw_bytes: bytes) -> Dict[str, Any]:
    """Legacy OCR pipeline retained for rollback behind LEGACY_OCR_ENABLED."""
    try:
        png_pages = pdf_to_png_pages(raw_bytes)
    except Exception as exc:
        raise RuntimeError(f"OCR pipeline failed: {exc}") from exc

    if not png_pages:
        raise RuntimeError("The PDF appears to contain no pages.")

    try:
        raw_ocr_pages: List[str] = []
        prepared_pages: List[str] = []
        for image_bytes in png_pages:
            ocr_text = ocr_png_with_openai(image_bytes)
            cleaned = (ocr_text or "").strip()
            raw_ocr_pages.append(cleaned)
            prepared = (_prepare_text(ocr_text) if ocr_text else "") or cleaned
            prepared_pages.append(prepared)
    except Exception as exc:
        raise RuntimeError(f"OCR pipeline failed: {exc}") from exc

    if not any((page or "").strip() for page in raw_ocr_pages):
        raise RuntimeError("The PDF contains no legible text after OCR.")

    bundle = call_llm_for_extraction_multi(prepared_pages)
    return {
        "bundle": bundle,
        "raw_joined": "\n\n".join(page for page in raw_ocr_pages if page),
        "client_name": extract_client_name("\n".join(raw_ocr_pages)),
    }


@app.post("/api/pdf-editor/native-text-replace")
def pdf_editor_native_text_replace(
    payload: NativeTextReplacePayload,
    x_app_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Attempt a conservative PyMuPDF native text replacement and return a new PDF."""
    if APP_KEY and x_app_key != APP_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    encoded = (payload.pdfBase64 or "").strip()
    if "," in encoded and encoded.lower().startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    try:
        pdf_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid PDF payload: {exc}") from exc
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty PDF payload")

    result = native_text_replace(
        pdf_bytes,
        page_index=payload.pageIndex,
        original_text=payload.originalText,
        replacement_text=payload.replacementText,
        bounds=payload.bounds,
        font_size=payload.fontSize,
        font_family=payload.fontFamily,
        native_edit_mode=payload.nativeEditMode or "content_stream_first",
        preserve_nearby_lines=payload.preserveNearbyLines is not False,
    )
    response = result.to_api_dict()
    if result.success and result.edited_pdf_bytes:
        response["editedPdfBase64"] = base64.b64encode(result.edited_pdf_bytes).decode("ascii")
    return response


def _extract_order_file_bytes(
    *,
    raw_bytes: bytes,
    filename: str,
    content_type: str,
    source: str,
    source_metadata: Optional[Dict[str, Any]] = None,
    force_ocr: bool = False,
) -> Dict[str, Any]:
    if not raw_bytes:
        raise HTTPException(
            status_code=422,
            detail="The uploaded order file is empty.",
        )

    normalized_filename = filename or "upload"
    normalized_content_type = (content_type or "").lower()
    is_pdf = _is_pdf_file(normalized_filename, normalized_content_type)
    is_image = _is_image_file(normalized_filename, normalized_content_type)
    if not is_pdf and not is_image:
        raise HTTPException(status_code=400, detail="Please upload a PDF or image file")

    raw_input_data_url = _file_data_url(
        raw_bytes,
        "application/pdf" if is_pdf else (normalized_content_type or "image/jpeg"),
    )
    extraction_method = "base64_pdf_visual" if is_pdf else "image_visual"
    client_name = ""
    prepared_text = ""
    fallback_warning: Optional[str] = None
    pdf_text_layer_text = extract_pdf_text_layer_text(raw_bytes) if is_pdf else ""
    scanned_image_only = bool(is_pdf and not (pdf_text_layer_text or "").strip())

    try:
        if is_pdf:
            if force_ocr:
                try:
                    legacy_result = _extract_pdf_via_legacy_ocr(raw_bytes)
                    bundle = legacy_result["bundle"]
                    client_name = legacy_result.get("client_name") or ""
                    prepared_text = bundle.get("prepared_text") or legacy_result.get("raw_joined") or ""
                    extraction_method = "forced_ocr"
                    fallback_warning = "forced_ocr_extraction_used"
                except Exception as ocr_exc:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Forced OCR extraction failed: {ocr_exc}",
                    ) from ocr_exc
            else:
                try:
                    bundle = call_llm_for_pdf_base64_visual(raw_bytes, filename=normalized_filename)
                except Exception as visual_exc:
                    if not LEGACY_OCR_ENABLED or not scanned_image_only:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Base64 PDF extraction failed: {visual_exc}",
                        ) from visual_exc
                    try:
                        legacy_result = _extract_pdf_via_legacy_ocr(raw_bytes)
                        bundle = legacy_result["bundle"]
                        client_name = legacy_result.get("client_name") or ""
                        prepared_text = bundle.get("prepared_text") or legacy_result.get("raw_joined") or ""
                        extraction_method = "legacy_ocr"
                        fallback_warning = f"legacy_ocr_fallback_used: {visual_exc}"
                    except Exception as legacy_exc:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Base64 PDF extraction failed: {visual_exc}; legacy OCR failed: {legacy_exc}",
                        ) from legacy_exc
        else:
            bundle = call_llm_for_image_visual(
                raw_bytes,
                filename=normalized_filename,
                mime_type=normalized_content_type or "image/jpeg",
            )

        raw_payload = bundle.get("raw") or {}
        _ = ExtractionResult(**raw_payload)
        result_data = ExtractionResult(**(bundle.get("data") or raw_payload))

        _debug_log_rows(result_data.rows)

        rows_dict = _rows_to_dicts(result_data.rows)
        prepared_text = prepared_text or bundle.get("prepared_text") or pdf_text_layer_text or ""
        repair_context = prepared_text or (bundle.get("output_text") or "")
        repaired_rows = apply_dimension_repair(repair_context, rows_dict)
        extracted_rows, normalization_warnings = normalize_extracted_rows(repaired_rows)
        validation = validate_rows(extracted_rows, context={"prepared_text": prepared_text})
        normalized_rows = validation.get("rows", extracted_rows)
        final_rows = apply_area_dimension_validation(normalized_rows)
        localized_response_rows = attach_pdf_row_locations(final_rows, raw_bytes) if is_pdf else final_rows
        row_warnings = _merge_critical_row_warnings(validation.get("row_warnings", {}), final_rows)
        hash_value = _compute_hash_bytes(raw_bytes)
        targeted_ocr = _run_targeted_ocr_repair_for_missing_fields(
            localized_response_rows,
            rows_dict,
            pdf_bytes=raw_bytes if is_pdf else None,
            pdf_id=hash_value if is_pdf else None,
            order_number=(bundle.get("data") or {}).get("order_number") or _primary_order_number(final_rows),
            enabled=bool(is_pdf and extraction_method == "base64_pdf_visual"),
        )
        localized_response_rows = targeted_ocr["rows"]

        combined_warnings: List[str] = []
        for source_list in (
            raw_payload.get("warnings") or [],
            (bundle.get("data") or {}).get("warnings") or [],
            normalization_warnings,
            validation.get("warnings", []) or [],
            targeted_ocr.get("warnings", []) or [],
        ):
            combined_warnings.extend(source_list)
        if fallback_warning:
            combined_warnings.append(fallback_warning)
        if scanned_image_only and is_pdf:
            combined_warnings.append("pdf_scanned_or_image_only_detected")

        applied = bundle.get("applied_corrections") or []
        if applied:
            combined_warnings.append(f"auto_corrections_applied:{','.join(str(i) for i in applied)}")

        metadata = dict(source_metadata or {})
        metadata.setdefault("source", source)
        metadata.setdefault("original_filename", normalized_filename)
        declared_units, declared_area = _declared_totals_from_sources(
            bundle,
            prepared_text,
            pdf_text_layer_text,
            bundle.get("output_text"),
        )

        llm_output_payload = dict(raw_payload) if isinstance(raw_payload, dict) else {"raw_payload": raw_payload}
        llm_output_payload["_meta"] = {
            "raw_response": bundle.get("raw_response"),
            "parsed_result": bundle.get("data") or raw_payload,
            "extraction_method": extraction_method,
            "source_metadata": metadata,
            "declared_units": declared_units,
            "declared_area": declared_area,
            "row_locations": [
                row.get("row_location") if isinstance(row, dict) else None
                for row in localized_response_rows
            ],
            "raw_base64_values": [
                _raw_base64_values_for_row(row if isinstance(row, dict) else {})
                for row in rows_dict
            ],
            "ocr_repair_attempts": targeted_ocr.get("repair_attempts", []),
            "ocr_row_overlays": targeted_ocr.get("row_overlays", []),
            "scanned_image_only": scanned_image_only,
            "force_ocr": bool(force_ocr),
        }
        llm_output_json = json.dumps(llm_output_payload, ensure_ascii=False)
        data_payload = bundle.get("data") or {}
        client_name = _normalize_client_name_payload(
            client_name,
            data_payload.get("client_name"),
            data_payload.get("clientName"),
            data_payload.get("client"),
            raw_payload.get("client_name") if isinstance(raw_payload, dict) else None,
            raw_payload.get("clientName") if isinstance(raw_payload, dict) else None,
            raw_payload.get("client") if isinstance(raw_payload, dict) else None,
            extract_client_name(prepared_text),
        )
        totals = _summarize_totals(final_rows)
        order_total_diagnostics = _order_total_diagnostics_from_totals(declared_units, declared_area, totals)
        _append_order_total_warning(combined_warnings, order_total_diagnostics)
        combined_warnings = _dedupe_warnings(combined_warnings)
        response_rows = _with_extraction_diagnostics(localized_response_rows)

        possible_duplicate = None
        possible_duplicate_reason = None
        telegram_file_record_id = metadata.get("telegram_file_record_id")
        if source == "telegram" and telegram_file_record_id:
            possible_duplicate = find_possible_duplicate_order(
                order_number=(bundle.get("data") or {}).get("order_number") or _primary_order_number(final_rows),
                client_name=client_name,
                total_units=totals["units"],
                total_area=totals["area"],
                recent_after=datetime.now(timezone.utc) - timedelta(days=30),
            )
            if possible_duplicate:
                possible_duplicate_reason = (
                    "Extracted order appears to match recent draft "
                    f"#{possible_duplicate.get('id')} by order number, client, units, and area."
                )
                combined_warnings = _dedupe_warnings(combined_warnings + ["possible_duplicate_order"])

        insert_result = insert_extraction_with_rows(
            source=source,
            rows=final_rows,
            raw_input=raw_input_data_url,
            prepared_text=prepared_text,
            llm_output_json=llm_output_json,
            model_used=bundle.get("model_used") or EXTRACTION_MODEL,
            hash_value=hash_value,
            confidence=(bundle.get("data") or {}).get("confidence"),
            client_name=client_name,
            source_metadata=metadata,
        )
        draft_order_id = insert_result["order_id"]
        status = insert_result.get("status", "draft")
        saved_order_id = draft_order_id if status == "approved" else None
        if source == "telegram" and telegram_file_record_id and possible_duplicate:
            update_telegram_file_record(
                int(telegram_file_record_id),
                linked_order_id=draft_order_id,
                extraction_status="extracted",
                duplicate_status="possible_duplicate",
                duplicate_reason=possible_duplicate_reason,
                processed_at=datetime.now(timezone.utc),
                clear_last_error=True,
            )

        response = {
            "order_number": (bundle.get("data") or {}).get("order_number") or _primary_order_number(final_rows),
            "rows": response_rows,
            "warnings": combined_warnings,
            "row_warnings": row_warnings,
            "draft_order_id": draft_order_id,
            "saved_order_id": saved_order_id,
            "status": status,
            "version": insert_result.get("version", 1),
            "source_hash": insert_result.get("source_hash"),
            "created_new_version": bool(insert_result.get("created_new_version")),
            "protected_order_id": insert_result.get("protected_order_id"),
            "applied_corrections": applied,
            "model_used": bundle.get("model_used"),
            "declared_units": declared_units,
            "declared_area": declared_area,
            "parsed_units": totals["units"],
            "parsed_area": totals["area"],
            "order_total_diagnostics": order_total_diagnostics,
            "extraction_method": extraction_method,
            "confidence": (bundle.get("data") or {}).get("confidence"),
            "client_name": client_name,
            "clientName": client_name,
            "client": client_name or "—",
        }
        if possible_duplicate:
            response.update(
                {
                    "duplicate_status": "possible_duplicate",
                    "duplicate_of_order_id": possible_duplicate.get("id"),
                    "duplicate_reason": possible_duplicate_reason,
                }
            )
        return response
    except HTTPException:
        raise
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=f"Validation error: {ve.errors()}")
    except Exception as e:
        print(f"[extract_file:{source}] fatal error:\n" + "".join(traceback.format_exc()))
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.post("/extract_pdf")
async def extract_pdf(
    file: UploadFile = File(...),
    force_ocr: bool = Query(default=False),
    x_app_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Accept a PDF file upload and run base64 PDF visual extraction."""
    if APP_KEY and x_app_key != APP_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not file or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    try:
        raw_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {e}")

    return _extract_order_file_bytes(
        raw_bytes=raw_bytes,
        filename=file.filename or "upload.pdf",
        content_type=file.content_type or "application/pdf",
        source="pdf",
        force_ocr=force_ocr,
    )


@app.post("/extract-pdf")
async def extract_pdf_dash_alias(
    file: UploadFile = File(...),
    force_ocr: bool = Query(default=False),
    x_app_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    return await extract_pdf(file=file, force_ocr=force_ocr, x_app_key=x_app_key)


def _telegram_token() -> str:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise HTTPException(status_code=500, detail="TELEGRAM_BOT_TOKEN is not set")
    return token


def _telegram_sender_name(message: Dict[str, Any]) -> str:
    sender = message.get("from") if isinstance(message.get("from"), dict) else {}
    parts = [
        str(sender.get("first_name") or "").strip(),
        str(sender.get("last_name") or "").strip(),
    ]
    full_name = " ".join(part for part in parts if part).strip()
    if full_name:
        return full_name
    username = str(sender.get("username") or "").strip()
    return f"@{username}" if username else ""


class TelegramRequestError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool, status_code: Optional[int] = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


def _telegram_log_fields(**overrides: Any) -> str:
    context = {**_telegram_log_context.get(), **overrides}
    return (
        f"update_id={context.get('update_id')} "
        f"message_id={context.get('message_id')} "
        f"record_id={context.get('record_id')} "
        f"state={context.get('state')}"
    )


def _telegram_retry_after(response: Optional[httpx.Response], payload: Any) -> Optional[float]:
    if isinstance(payload, dict):
        parameters = payload.get("parameters")
        if isinstance(parameters, dict):
            try:
                value = float(parameters.get("retry_after"))
                if value >= 0:
                    return value
            except (TypeError, ValueError):
                pass
    if response is None:
        return None
    raw_value = response.headers.get("Retry-After")
    if not raw_value:
        return None
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(raw_value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _telegram_backoff_delay(attempt: int, retry_after: Optional[float]) -> float:
    if retry_after is not None:
        return retry_after
    return min(8.0, 0.5 * (2 ** max(0, attempt - 1))) + random.uniform(0.0, 0.25)


def _record_telegram_network_retry(*, stage: str, attempt: int, error: str, duration_ms: int) -> None:
    context = _telegram_log_context.get()
    record_id = context.get("record_id")
    if record_id is not None:
        try:
            current = get_telegram_file(int(record_id)) or {}
            update_telegram_file_record(
                int(record_id),
                download_retry_count=int(current.get("download_retry_count") or 0) + 1,
                last_error=error,
            )
        except Exception as exc:
            logger.error(
                "telegram retry state update failed %s stage=%s error_type=%s",
                _telegram_log_fields(),
                stage,
                exc.__class__.__name__,
            )
    logger.warning(
        "telegram network retry %s stage=%s attempt=%s duration_ms=%s error=%s",
        _telegram_log_fields(),
        stage,
        attempt,
        duration_ms,
        error,
    )


async def _telegram_get_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    stage: str,
    timeout: httpx.Timeout,
    params: Optional[Dict[str, Any]] = None,
) -> httpx.Response:
    last_error = "Telegram request failed"
    for attempt in range(1, TELEGRAM_NETWORK_MAX_ATTEMPTS + 1):
        started = time.monotonic()
        response: Optional[httpx.Response] = None
        payload: Any = None
        retryable = False
        retry_after: Optional[float] = None
        try:
            response = await client.get(url, params=params, timeout=timeout)
            try:
                payload = response.json()
            except Exception:
                payload = None

            status_code = int(response.status_code)
            telegram_error_code = None
            if isinstance(payload, dict) and payload.get("ok") is False:
                try:
                    telegram_error_code = int(payload.get("error_code"))
                except (TypeError, ValueError):
                    telegram_error_code = status_code

            effective_code = telegram_error_code or status_code
            retryable = effective_code == 429 or effective_code >= 500
            if status_code >= 400 or telegram_error_code is not None:
                last_error = f"Telegram {stage} returned status {effective_code}"
                retry_after = _telegram_retry_after(response, payload)
                if not retryable:
                    logger.error(
                        "telegram network request failed %s stage=%s attempt=%s "
                        "duration_ms=%s outcome=terminal error=%s",
                        _telegram_log_fields(),
                        stage,
                        attempt,
                        int((time.monotonic() - started) * 1000),
                        last_error,
                    )
                    raise TelegramRequestError(
                        last_error,
                        retryable=False,
                        status_code=effective_code,
                    )
            else:
                logger.info(
                    "telegram network request succeeded %s stage=%s attempt=%s duration_ms=%s outcome=success",
                    _telegram_log_fields(),
                    stage,
                    attempt,
                    int((time.monotonic() - started) * 1000),
                )
                return response
        except TelegramRequestError:
            raise
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            retryable = True
            last_error = f"{exc.__class__.__name__} during Telegram {stage}"

        duration_ms = int((time.monotonic() - started) * 1000)
        if not retryable or attempt >= TELEGRAM_NETWORK_MAX_ATTEMPTS:
            logger.error(
                "telegram network request failed %s stage=%s attempt=%s duration_ms=%s outcome=terminal error=%s",
                _telegram_log_fields(),
                stage,
                attempt,
                duration_ms,
                last_error,
            )
            raise TelegramRequestError(
                last_error,
                retryable=retryable,
                status_code=response.status_code if response is not None else None,
            )

        _record_telegram_network_retry(
            stage=stage,
            attempt=attempt,
            error=last_error,
            duration_ms=duration_ms,
        )
        await asyncio.sleep(_telegram_backoff_delay(attempt, retry_after))

    raise TelegramRequestError(last_error, retryable=True)


async def _telegram_reply(token: str, chat_id: Any, message_id: Any, text: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=TELEGRAM_REPLY_TIMEOUT) as client:
            response = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "reply_to_message_id": message_id,
                    "allow_sending_without_reply": True,
                },
            )
            if response.status_code >= 400:
                logger.warning(
                    "telegram reply failed %s status=%s",
                    _telegram_log_fields(),
                    response.status_code,
                )
                return False
            return True
    except Exception as exc:
        logger.warning(
            "telegram reply failed %s error_type=%s",
            _telegram_log_fields(),
            exc.__class__.__name__,
        )
        return False


async def _telegram_download_file(token: str, file_id: str) -> Tuple[bytes, Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        file_response = await _telegram_get_with_retries(
            client,
            f"https://api.telegram.org/bot{token}/getFile",
            stage="getFile",
            timeout=TELEGRAM_GET_FILE_TIMEOUT,
            params={"file_id": file_id},
        )
        try:
            file_payload = file_response.json()
        except Exception as exc:
            raise TelegramRequestError(
                "Telegram getFile returned invalid JSON",
                retryable=False,
                status_code=file_response.status_code,
            ) from exc
        if not file_payload.get("ok") or not isinstance(file_payload.get("result"), dict):
            raise TelegramRequestError(
                "Telegram getFile returned an invalid response",
                retryable=False,
                status_code=file_response.status_code,
            )
        result = file_payload["result"]
        file_path = result.get("file_path")
        if not file_path:
            raise TelegramRequestError(
                "Telegram getFile did not return a file path",
                retryable=False,
                status_code=file_response.status_code,
            )
        reported_size = result.get("file_size")
        if isinstance(reported_size, int) and reported_size > TELEGRAM_MAX_FILE_BYTES:
            raise ValueError("telegram_file_too_large")

        download_response = await _telegram_get_with_retries(
            client,
            f"https://api.telegram.org/file/bot{token}/{file_path}",
            stage="file_download",
            timeout=TELEGRAM_FILE_DOWNLOAD_TIMEOUT,
        )
        content = download_response.content
        if len(content) > TELEGRAM_MAX_FILE_BYTES:
            raise ValueError("telegram_file_too_large")
        if not content:
            raise TelegramRequestError("Telegram downloaded an empty file", retryable=False)
        return content, result


def _validate_telegram_file_bytes(raw_bytes: bytes, filename: str, mime_type: str) -> None:
    if not raw_bytes:
        raise ValueError("telegram_empty_file")
    if len(raw_bytes) > TELEGRAM_MAX_FILE_BYTES:
        raise ValueError("telegram_file_too_large")
    if _is_pdf_file(filename, mime_type):
        if not raw_bytes[:1024].lstrip().startswith(b"%PDF"):
            raise ValueError("telegram_invalid_pdf")
        return
    if _is_image_file(filename, mime_type):
        is_jpeg = raw_bytes.startswith(b"\xff\xd8")
        is_png = raw_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        is_webp = len(raw_bytes) >= 12 and raw_bytes.startswith(b"RIFF") and raw_bytes[8:12] == b"WEBP"
        if not (is_jpeg or is_png or is_webp):
            raise ValueError("telegram_invalid_image")
        return
    raise ValueError("telegram_unsupported_file")


def _telegram_token_optional() -> str:
    return (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()


def _short_error_message(exc: Exception) -> str:
    if isinstance(exc, TelegramRequestError):
        return str(exc)[:500]
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return f"{exc.__class__.__name__} while contacting Telegram"
    if isinstance(exc, HTTPException):
        detail = exc.detail
        if isinstance(detail, str):
            return detail[:500]
        return json.dumps(detail, ensure_ascii=False)[:500]
    return str(exc)[:500] or exc.__class__.__name__


def _is_retryable_telegram_extraction_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPException):
        return int(exc.status_code or 500) >= 500
    return True


def _public_telegram_file(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None
    blocked = {"file_path", "stored_filename", "telegram_file_id", "file_sha256"}
    return {key: value for key, value in record.items() if key not in blocked}


def _telegram_counts_payload() -> Dict[str, int]:
    try:
        return get_telegram_file_counts()
    except Exception as exc:
        print(f"[telegram-events] count failed: {exc}")
        return {
            "untouched_count": count_untouched_telegram_files(),
            "queued_count": 0,
            "processing_count": 0,
            "failed_count": 0,
        }


def _deliver_telegram_event(message: Dict[str, Any]) -> None:
    stale: List[asyncio.Queue] = []
    with _telegram_event_clients_lock:
        clients = list(_telegram_event_clients)
    for queue in clients:
        try:
            queue.put_nowait(message)
        except Exception:
            stale.append(queue)
    if stale:
        with _telegram_event_clients_lock:
            for queue in stale:
                _telegram_event_clients.discard(queue)


def broadcastTelegramFileEvent(event_type: str, payload: Dict[str, Any]) -> None:
    message = {"type": event_type, **(payload or {})}
    with _telegram_event_clients_lock:
        has_clients = bool(_telegram_event_clients)
    if not has_clients:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    target_loop = _telegram_event_loop or loop
    if target_loop and target_loop.is_running():
        if loop is target_loop:
            _deliver_telegram_event(message)
        else:
            target_loop.call_soon_threadsafe(_deliver_telegram_event, message)


def _broadcast_telegram_counts() -> Dict[str, int]:
    counts = _telegram_counts_payload()
    broadcastTelegramFileEvent("telegram_counts_updated", {"counts": counts})
    return counts


def _broadcast_telegram_file_change(event_type: str, record: Optional[Dict[str, Any]]) -> None:
    public_record = _public_telegram_file(record)
    counts = _telegram_counts_payload()
    payload: Dict[str, Any] = {"counts": counts}
    if public_record is not None:
        payload["file"] = public_record
    broadcastTelegramFileEvent(event_type, payload)
    broadcastTelegramFileEvent("telegram_counts_updated", {"counts": counts})


def _ensure_telegram_queue_runner() -> None:
    global _telegram_queue_task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    with _telegram_queue_lock:
        if _telegram_queue_task and not _telegram_queue_task.done():
            return
        _telegram_queue_task = loop.create_task(_drain_telegram_extraction_queue())


def _enqueue_telegram_extraction(file_id: Any) -> bool:
    try:
        normalized_id = int(file_id)
    except (TypeError, ValueError):
        return False
    with _telegram_queue_lock:
        if normalized_id in _telegram_queued_ids:
            return False
        _telegram_queued_ids.add(normalized_id)
        _telegram_queue.append(normalized_id)
    _ensure_telegram_queue_runner()
    return True


async def _drain_telegram_extraction_queue() -> None:
    global _telegram_queue_task
    while True:
        batch: List[int] = []
        with _telegram_queue_lock:
            while _telegram_queue and len(batch) < TELEGRAM_EXTRACTION_CONCURRENCY:
                batch.append(_telegram_queue.popleft())
        if not batch:
            with _telegram_queue_lock:
                _telegram_queue_task = None
            return
        outcomes = await asyncio.gather(
            *(_process_telegram_queue_job(file_id) for file_id in batch),
            return_exceptions=True,
        )
        for file_id, outcome in zip(batch, outcomes):
            if isinstance(outcome, Exception):
                logger.error(
                    "telegram job crashed record_id=%s state=unknown outcome=failed error_type=%s",
                    file_id,
                    outcome.__class__.__name__,
                )


def _recover_telegram_extraction_queue() -> None:
    # At startup there cannot be a live worker from this process. Recover every
    # row left in an in-progress state, including one updated just before a crash.
    stale_before = datetime.now(timezone.utc) + timedelta(seconds=1)
    try:
        file_ids = list_unfinished_telegram_file_ids(stale_processing_before=stale_before)
    except Exception as exc:
        logger.error("telegram queue recovery failed error_type=%s", exc.__class__.__name__)
        return
    for file_id in file_ids:
        _enqueue_telegram_extraction(file_id)
    logger.info("telegram queue recovery outcome=complete recovered_jobs=%s", len(file_ids))


def _telegram_source_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    received_at = record.get("received_at")
    if hasattr(received_at, "isoformat"):
        received_at = received_at.isoformat()
    return {
        "source": "telegram",
        "telegram_chat_id": record.get("telegram_chat_id"),
        "telegram_message_id": record.get("telegram_message_id"),
        "telegram_sender_name": record.get("telegram_sender_name"),
        "telegram_file_id": record.get("telegram_file_id"),
        "telegram_file_record_id": record.get("id"),
        "original_filename": record.get("original_filename"),
        "received_at": received_at,
        "caption": record.get("telegram_caption") or "",
    }


def _process_telegram_queue_job_sync(file_id: int) -> Dict[str, Any]:
    record = get_telegram_file(file_id)
    if not record:
        return {"status": "missing", "file_id": file_id}
    if record.get("extraction_status") == "extracted":
        return {"status": "already_extracted", "file_id": file_id, "record": record}
    if record.get("extraction_status") == "duplicate":
        return {"status": record.get("extraction_status"), "file_id": file_id, "record": record}

    attempts_done = int(record.get("retry_count") or 0)
    max_attempts = 1 + TELEGRAM_EXTRACTION_MAX_RETRIES
    last_error = ""
    while attempts_done < max_attempts:
        attempts_done += 1
        processing_record = update_telegram_file_record(
            file_id,
            extraction_status="processing",
            processing_started_at=datetime.now(timezone.utc),
            retry_count=attempts_done,
            clear_last_error=True,
        )
        if processing_record:
            _broadcast_telegram_file_change("telegram_file_updated", processing_record)
        try:
            latest = get_telegram_file(file_id) or record
            path = _safe_telegram_file_path(latest)
            raw_bytes = path.read_bytes()
            result = _extract_order_file_bytes(
                raw_bytes=raw_bytes,
                filename=str(latest.get("original_filename") or "telegram-order.pdf"),
                content_type=str(latest.get("mime_type") or "application/pdf"),
                source="telegram",
                source_metadata=_telegram_source_metadata(latest),
            )
            updated = update_telegram_file_record(
                file_id,
                linked_order_id=result.get("draft_order_id"),
                extraction_status="extracted",
                processed_at=datetime.now(timezone.utc),
                retry_count=attempts_done,
                clear_last_error=True,
            )
            response_record = {**latest, **(updated or {})}
            _broadcast_telegram_file_change("telegram_file_updated", response_record)
            return {
                "status": "extracted",
                "file_id": file_id,
                "record": response_record,
                "order_id": result.get("draft_order_id"),
                "duplicate_status": result.get("duplicate_status"),
                "duplicate_of_order_id": result.get("duplicate_of_order_id"),
            }
        except Exception as exc:
            last_error = _short_error_message(exc)
            logger.warning(
                "telegram extraction retry %s attempt=%s error=%s",
                _telegram_log_fields(state="processing"),
                attempts_done,
                last_error,
            )
            retryable = _is_retryable_telegram_extraction_error(exc)
            if retryable and attempts_done < max_attempts:
                update_telegram_file_record(file_id, retry_count=attempts_done, last_error=last_error)
                time.sleep(1)
                continue
            failed = update_telegram_file_record(
                file_id,
                extraction_status="failed",
                processed_at=datetime.now(timezone.utc),
                retry_count=attempts_done,
                last_error=last_error,
            )
            response_record = {**record, **(failed or {})}
            _broadcast_telegram_file_change("telegram_file_updated", response_record)
            return {"status": "failed", "file_id": file_id, "record": response_record, "error": last_error}
    failed = update_telegram_file_record(
        file_id,
        extraction_status="failed",
        processed_at=datetime.now(timezone.utc),
        retry_count=attempts_done,
        last_error=last_error or "Extraction failed",
    )
    response_record = {**record, **(failed or {})}
    _broadcast_telegram_file_change("telegram_file_updated", response_record)
    return {"status": "failed", "file_id": file_id, "record": response_record, "error": last_error}


async def _send_telegram_reply_once(record: Dict[str, Any], *, kind: str, text: str) -> None:
    sent_field = "intake_reply_sent_at" if kind == "intake" else "outcome_reply_sent_at"
    if record.get(sent_field):
        return
    token = _telegram_token_optional()
    chat_id = record.get("telegram_chat_id")
    if not token or chat_id is None:
        return
    sent = await _telegram_reply(token, chat_id, record.get("telegram_message_id"), text)
    if sent is False:
        return
    updated = update_telegram_file_record(
        int(record["id"]),
        **{sent_field: datetime.now(timezone.utc)},
    )
    if updated:
        record.update(updated)


def _telegram_terminal_outcome(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": str(record.get("extraction_status") or "failed"),
        "file_id": record.get("id"),
        "record": record,
    }


async def _download_telegram_queue_file(record: Dict[str, Any]) -> Dict[str, Any]:
    file_id = int(record["id"])
    now = datetime.now(timezone.utc)
    downloading = update_telegram_file_record(
        file_id,
        extraction_status="downloading",
        download_started_at=now,
        clear_last_error=True,
    )
    if downloading:
        record.update(downloading)
        _broadcast_telegram_file_change("telegram_file_updated", record)

    token = _telegram_token_optional()
    if not token:
        failed = update_telegram_file_record(
            file_id,
            extraction_status="failed",
            processed_at=datetime.now(timezone.utc),
            last_error="TELEGRAM_BOT_TOKEN is not set",
        )
        return _telegram_terminal_outcome({**record, **(failed or {})})

    try:
        raw_bytes, _file_info = await _telegram_download_file(token, str(record.get("telegram_file_id") or ""))
        _validate_telegram_file_bytes(
            raw_bytes,
            str(record.get("original_filename") or "telegram-order.pdf"),
            str(record.get("mime_type") or "application/pdf"),
        )
        stored = await asyncio.to_thread(
            _store_telegram_file,
            raw_bytes,
            str(record.get("original_filename") or "telegram-order.pdf"),
        )
        file_sha256 = _compute_sha256_bytes(raw_bytes)
        duplicate_of = await asyncio.to_thread(
            find_telegram_file_by_sha256,
            file_sha256,
            exclude_file_id=file_id,
        )
        downloaded_at = datetime.now(timezone.utc)
        common_update = {
            "original_filename": stored["original_filename"],
            "stored_filename": stored["stored_filename"],
            "file_path": stored["file_path"],
            "file_size": stored["file_size"],
            "file_sha256": file_sha256,
            "downloaded_at": downloaded_at,
            "processed_at": downloaded_at if duplicate_of else None,
            "clear_last_error": True,
        }
        if duplicate_of:
            updated = update_telegram_file_record(
                file_id,
                **common_update,
                extraction_status="duplicate",
                duplicate_status="duplicate",
                duplicate_of_file_id=duplicate_of.get("id"),
                duplicate_reason=f"Exact file SHA-256 match with Telegram file #{duplicate_of.get('id')}.",
            )
            response_record = {**record, **(updated or {})}
            _broadcast_telegram_file_change("telegram_file_updated", response_record)
            return {
                "status": "duplicate",
                "file_id": file_id,
                "record": response_record,
                "duplicate_of_file_id": duplicate_of.get("id"),
            }

        updated = update_telegram_file_record(
            file_id,
            **common_update,
            extraction_status="queued",
            queued_at=downloaded_at,
        )
        response_record = {**record, **(updated or {})}
        _broadcast_telegram_file_change("telegram_file_updated", response_record)
        return {"status": "queued", "file_id": file_id, "record": response_record}
    except ValueError as exc:
        if str(exc) == "telegram_file_too_large":
            updated = update_telegram_file_record(
                file_id,
                extraction_status="too_large",
                processed_at=datetime.now(timezone.utc),
                last_error="Telegram file exceeds the 5 MB limit",
            )
            response_record = {**record, **(updated or {})}
            _broadcast_telegram_file_change("telegram_file_updated", response_record)
            return _telegram_terminal_outcome(response_record)
        error = _short_error_message(exc)
        updated = update_telegram_file_record(
            file_id,
            extraction_status="failed",
            processed_at=datetime.now(timezone.utc),
            last_error=error,
        )
        response_record = {**record, **(updated or {})}
        _broadcast_telegram_file_change("telegram_file_updated", response_record)
        return _telegram_terminal_outcome(response_record)
    except Exception as exc:
        error = _short_error_message(exc)
        updated = update_telegram_file_record(
            file_id,
            extraction_status="failed",
            processed_at=datetime.now(timezone.utc),
            last_error=error,
        )
        logger.error(
            "telegram download failed %s attempt=%s duration_ms=%s outcome=terminal error=%s",
            _telegram_log_fields(state="downloading"),
            int((updated or record).get("download_retry_count") or 0) + 1,
            int((time.monotonic() - _telegram_log_context.get().get("started_at", time.monotonic())) * 1000),
            error,
        )
        response_record = {**record, **(updated or {})}
        _broadcast_telegram_file_change("telegram_file_updated", response_record)
        return _telegram_terminal_outcome(response_record)


async def _process_telegram_queue_job(file_id: int) -> Dict[str, Any]:
    started = time.monotonic()
    record = get_telegram_file(int(file_id))
    context_token = _telegram_log_context.set(
        {
            "update_id": (record or {}).get("telegram_update_id"),
            "message_id": (record or {}).get("telegram_message_id"),
            "record_id": int(file_id),
            "state": (record or {}).get("extraction_status"),
            "started_at": started,
        }
    )
    outcome: Dict[str, Any] = {"status": "missing", "file_id": int(file_id)}
    try:
        if not record:
            return outcome

        initial_status = str(record.get("extraction_status") or "")
        if initial_status in {"download_queued", "downloading"}:
            await _send_telegram_reply_once(
                record,
                kind="intake",
                text="Order received ✅ Queued for extraction.",
            )
            outcome = await _download_telegram_queue_file(record)
            record = outcome.get("record") or record
            if outcome.get("status") == "queued":
                outcome = await asyncio.to_thread(_process_telegram_queue_job_sync, int(file_id))
                record = outcome.get("record") or record
        elif initial_status in {"queued", "received", "processing"}:
            outcome = await asyncio.to_thread(_process_telegram_queue_job_sync, int(file_id))
            record = outcome.get("record") or record
        else:
            outcome = _telegram_terminal_outcome(record)

        status = str(outcome.get("status") or "")
        if outcome.get("duplicate_status") == "possible_duplicate" or status == "possible_duplicate":
            reply_text = "Possible duplicate detected ⚠️ Please review in platform."
        elif status in {"extracted", "already_extracted"}:
            reply_text = "Extraction finished ✅ Please review in platform."
        elif status == "duplicate":
            reply_text = "Duplicate detected ⚠️ This PDF was already received."
        elif status == "too_large":
            reply_text = "File is too large. Max size is 5MB."
        elif status == "unsupported":
            reply_text = "Only PDF/image orders are supported."
        elif status == "failed" and record.get("file_path"):
            reply_text = "Extraction failed ⚠️ Original PDF is saved in Telegram Files."
        elif status in {"missing_file_id", "failed"}:
            reply_text = "Extraction failed. Please try again."
        else:
            reply_text = ""
        if reply_text:
            await _send_telegram_reply_once(record, kind="outcome", text=reply_text)

        logger.info(
            "telegram job complete %s attempt=%s duration_ms=%s outcome=%s",
            _telegram_log_fields(state=status),
            int(record.get("retry_count") or 0) + int(record.get("download_retry_count") or 0),
            int((time.monotonic() - started) * 1000),
            status,
        )
        return outcome
    finally:
        with _telegram_queue_lock:
            _telegram_queued_ids.discard(int(file_id))
        _telegram_log_context.reset(context_token)


def _telegram_document_spec(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    document = message.get("document")
    if not isinstance(document, dict):
        return None
    filename = str(document.get("file_name") or "telegram-upload").strip() or "telegram-upload"
    mime_type = str(document.get("mime_type") or "").strip().lower()
    if not _is_pdf_file(filename, mime_type) and not _is_image_file(filename, mime_type):
        return {
            "unsupported": True,
            "file_id": document.get("file_id"),
            "filename": filename,
            "content_type": mime_type or "application/octet-stream",
            "file_size": document.get("file_size"),
            "original_filename": filename,
        }
    return {
        "file_id": document.get("file_id"),
        "filename": filename,
        "content_type": mime_type or ("application/pdf" if filename.lower().endswith(".pdf") else "image/jpeg"),
        "file_size": document.get("file_size"),
        "original_filename": filename,
    }


def _telegram_photo_spec(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    photos = message.get("photo")
    if not isinstance(photos, list) or not photos:
        return None
    largest = max(
        [photo for photo in photos if isinstance(photo, dict)],
        key=lambda photo: (int(photo.get("file_size") or 0), int(photo.get("width") or 0) * int(photo.get("height") or 0)),
        default=None,
    )
    if not largest:
        return None
    file_unique = str(largest.get("file_unique_id") or largest.get("file_id") or "photo")
    return {
        "file_id": largest.get("file_id"),
        "filename": f"telegram-photo-{file_unique}.jpg",
        "content_type": "image/jpeg",
        "file_size": largest.get("file_size"),
        "original_filename": None,
    }


async def _handle_telegram_update_legacy(update: Dict[str, Any]) -> Dict[str, Any]:
    message = update.get("message")
    if not isinstance(message, dict):
        return {"ok": True, "status": "ignored"}

    chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
    chat_id = chat.get("id")
    message_id = message.get("message_id")
    spec = _telegram_document_spec(message) or _telegram_photo_spec(message)
    if spec is None:
        return {"ok": True, "status": "ignored"}

    token = _telegram_token()
    if spec.get("unsupported"):
        if chat_id is not None:
            await _telegram_reply(token, chat_id, message_id, "Only PDF/image orders are supported.")
        return {"ok": True, "status": "unsupported"}

    file_size = spec.get("file_size")
    if isinstance(file_size, int) and file_size > TELEGRAM_MAX_FILE_BYTES:
        if chat_id is not None:
            await _telegram_reply(token, chat_id, message_id, "File is too large. Max size is 5MB.")
        return {"ok": True, "status": "too_large"}

    if not spec.get("file_id"):
        if chat_id is not None:
            await _telegram_reply(token, chat_id, message_id, "Extraction failed. Please try again.")
        return {"ok": True, "status": "missing_file_id"}

    filename_for_type = str(spec.get("filename") or "")
    content_type_for_type = str(spec.get("content_type") or "")
    is_pdf = _is_pdf_file(filename_for_type, content_type_for_type)
    is_image = _is_image_file(filename_for_type, content_type_for_type)
    if is_pdf or is_image:
        existing = find_telegram_file_record(
            telegram_chat_id=chat_id,
            telegram_message_id=message_id,
            telegram_file_id=str(spec.get("file_id") or ""),
        )
        if existing:
            existing_status = str(existing.get("extraction_status") or "").lower()
            if existing_status in {"received", "queued", "processing"}:
                _enqueue_telegram_extraction(existing.get("id"))
            if chat_id is not None and existing_status in {"received", "queued", "processing"}:
                await _telegram_reply(token, chat_id, message_id, "Order received ✅ Queued for extraction.")
            elif chat_id is not None and existing_status == "failed":
                await _telegram_reply(token, chat_id, message_id, "Extraction failed ⚠️ Original PDF is saved in Telegram Files.")
            return {"ok": True, "status": "duplicate", "telegram_file_id": existing.get("id")}

    try:
        raw_bytes, file_info = await _telegram_download_file(token, str(spec["file_id"]))
        received_at = datetime.now(timezone.utc)
        if is_pdf or is_image:
            file_sha256 = _compute_sha256_bytes(raw_bytes)
            stored = _store_telegram_file(raw_bytes, str(spec.get("original_filename") or spec.get("filename") or "telegram-order.pdf"))
            mime_type = "application/pdf" if is_pdf else str(spec.get("content_type") or "image/jpeg")
            duplicate_of = find_telegram_file_by_sha256(file_sha256, exclude_file_id=None)
            if duplicate_of:
                duplicate_record = create_telegram_file_record(
                    original_filename=stored["original_filename"],
                    stored_filename=stored["stored_filename"],
                    file_path=stored["file_path"],
                    mime_type=mime_type,
                    file_size=stored["file_size"],
                    file_sha256=file_sha256,
                    telegram_file_id=str(spec.get("file_id") or ""),
                    telegram_chat_id=chat_id,
                    telegram_message_id=message_id,
                    telegram_sender_name=_telegram_sender_name(message),
                    telegram_caption=message.get("caption") or "",
                    received_at=received_at,
                    extraction_status="duplicate",
                    duplicate_status="duplicate",
                    duplicate_of_file_id=duplicate_of.get("id"),
                    duplicate_reason=f"Exact file SHA-256 match with Telegram file #{duplicate_of.get('id')}.",
                )
                _broadcast_telegram_file_change("telegram_file_created", duplicate_record)
                if chat_id is not None:
                    await _telegram_reply(token, chat_id, message_id, "Duplicate detected ⚠️ This PDF was already received.")
                return {
                    "ok": True,
                    "status": "duplicate",
                    "telegram_file_id": duplicate_record.get("id"),
                    "duplicate_of_file_id": duplicate_of.get("id"),
                    "file_path": file_info.get("file_path"),
                }
            telegram_file_record = create_telegram_file_record(
                original_filename=stored["original_filename"],
                stored_filename=stored["stored_filename"],
                file_path=stored["file_path"],
                mime_type=mime_type,
                file_size=stored["file_size"],
                file_sha256=file_sha256,
                telegram_file_id=str(spec.get("file_id") or ""),
                telegram_chat_id=chat_id,
                telegram_message_id=message_id,
                telegram_sender_name=_telegram_sender_name(message),
                telegram_caption=message.get("caption") or "",
                received_at=received_at,
                extraction_status="queued",
                queued_at=received_at,
            )
            _broadcast_telegram_file_change("telegram_file_created", telegram_file_record)
            _enqueue_telegram_extraction(telegram_file_record["id"])
            if chat_id is not None:
                await _telegram_reply(token, chat_id, message_id, "Order received ✅ Queued for extraction.")
            return {
                "ok": True,
                "status": "queued",
                "telegram_file_id": telegram_file_record.get("id"),
                "file_path": file_info.get("file_path"),
            }

        if chat_id is not None:
            await _telegram_reply(token, chat_id, message_id, "Order received ✅")
        metadata = {
            "source": "telegram",
            "telegram_chat_id": chat_id,
            "telegram_message_id": message_id,
            "telegram_sender_name": _telegram_sender_name(message),
            "original_filename": spec.get("original_filename") or spec.get("filename"),
            "received_at": received_at.isoformat(),
            "caption": message.get("caption") or "",
            "telegram_file_record_id": None,
        }
        result = _extract_order_file_bytes(
            raw_bytes=raw_bytes,
            filename=str(spec.get("filename") or "telegram-upload"),
            content_type=str(spec.get("content_type") or "application/octet-stream"),
            source="telegram",
            source_metadata=metadata,
        )
        if chat_id is not None:
            await _telegram_reply(token, chat_id, message_id, "Extraction finished ✅ Please review in platform.")
        return {
            "ok": True,
            "status": "extracted",
            "order_id": result.get("draft_order_id"),
            "telegram_file_id": None,
            "file_path": file_info.get("file_path"),
        }
    except ValueError as exc:
        if str(exc) == "telegram_file_too_large":
            if chat_id is not None:
                await _telegram_reply(token, chat_id, message_id, "File is too large. Max size is 5MB.")
            return {"ok": True, "status": "too_large"}
        logger.error(
            "legacy telegram handler failed message_id=%s error_type=%s",
            message_id,
            exc.__class__.__name__,
        )
    except HTTPException as exc:
        logger.error(
            "legacy telegram handler failed message_id=%s status=%s",
            message_id,
            exc.status_code,
        )
    except Exception as exc:
        logger.error(
            "legacy telegram handler failed message_id=%s error_type=%s",
            message_id,
            exc.__class__.__name__,
        )

    if chat_id is not None:
        await _telegram_reply(token, chat_id, message_id, "Extraction failed. Please try again.")
    return {"ok": True, "status": "failed", "telegram_file_id": None}


async def _handle_telegram_update(update: Dict[str, Any]) -> Dict[str, Any]:
    started = time.monotonic()
    message = update.get("message")
    if not isinstance(message, dict):
        return {"ok": True, "status": "ignored"}

    chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
    chat_id = chat.get("id")
    message_id = message.get("message_id")
    spec = _telegram_document_spec(message) or _telegram_photo_spec(message)
    if spec is None:
        return {"ok": True, "status": "ignored"}

    raw_update_id = update.get("update_id")
    if raw_update_id is not None and (
        not isinstance(raw_update_id, (int, str)) or not str(raw_update_id).strip()
    ):
        return {"ok": True, "status": "invalid_payload"}
    if raw_update_id is None:
        # Telegram always supplies update_id. This tuple-derived fallback keeps
        # legacy callers idempotent through the same unique database key.
        fallback = f"{chat_id}|{message_id}|{spec.get('file_id')}"
        telegram_update_id = f"message:{hashlib.sha256(fallback.encode('utf-8')).hexdigest()}"
    else:
        telegram_update_id = str(raw_update_id).strip()

    try:
        declared_size = max(0, int(spec.get("file_size") or 0))
    except (TypeError, ValueError):
        declared_size = 0
    if spec.get("unsupported"):
        initial_status = "unsupported"
    elif declared_size > TELEGRAM_MAX_FILE_BYTES:
        initial_status = "too_large"
    elif not spec.get("file_id"):
        initial_status = "missing_file_id"
    else:
        initial_status = "download_queued"

    try:
        record, created = await asyncio.to_thread(
            create_or_get_telegram_intake,
            telegram_update_id=telegram_update_id,
            original_filename=str(
                spec.get("original_filename") or spec.get("filename") or "telegram-upload"
            ),
            mime_type=str(spec.get("content_type") or "application/octet-stream"),
            file_size=declared_size,
            telegram_file_id=str(spec.get("file_id") or "") or None,
            telegram_chat_id=chat_id,
            telegram_message_id=message_id,
            telegram_sender_name=_telegram_sender_name(message),
            telegram_caption=message.get("caption") or "",
            extraction_status=initial_status,
            received_at=datetime.now(timezone.utc),
        )
    except Exception as exc:
        logger.error(
            "telegram intake persistence failed update_id=%s message_id=%s state=%s "
            "duration_ms=%s outcome=failed error_type=%s",
            telegram_update_id,
            message_id,
            initial_status,
            int((time.monotonic() - started) * 1000),
            exc.__class__.__name__,
        )
        raise HTTPException(status_code=503, detail="Unable to durably accept Telegram update") from exc

    record_status = str(record.get("extraction_status") or initial_status)
    unfinished = record_status in {
        "download_queued",
        "downloading",
        "queued",
        "received",
        "processing",
    }
    pending_terminal_reply = (
        record_status
        in {"unsupported", "too_large", "missing_file_id", "failed", "duplicate", "extracted"}
        and not record.get("outcome_reply_sent_at")
    )
    if unfinished or (created and pending_terminal_reply):
        _enqueue_telegram_extraction(record.get("id"))
    if created:
        _broadcast_telegram_file_change("telegram_file_created", record)

    logger.info(
        "telegram intake accepted update_id=%s message_id=%s record_id=%s state=%s "
        "attempt=1 duration_ms=%s outcome=%s",
        telegram_update_id,
        message_id,
        record.get("id"),
        record_status,
        int((time.monotonic() - started) * 1000),
        "created" if created else "deduplicated",
    )
    return {
        "ok": True,
        "status": (
            ("queued" if initial_status == "download_queued" else initial_status)
            if created
            else "duplicate"
        ),
        "telegram_file_id": record.get("id"),
    }


@app.post("/webhook/telegram")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None, alias=TELEGRAM_SECRET_HEADER),
) -> Dict[str, Any]:
    expected_secret = (os.getenv("TELEGRAM_WEBHOOK_SECRET") or "").strip()
    if expected_secret and x_telegram_bot_api_secret_token != expected_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        update = await request.json()
    except Exception:
        return {"ok": True, "status": "invalid_json"}
    if not isinstance(update, dict):
        return {"ok": True, "status": "ignored"}
    return await _handle_telegram_update(update)


def _format_sse(message: Dict[str, Any], event_name: Optional[str] = None) -> str:
    name = event_name or str(message.get("type") or "message")
    data = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {data}\n\n"


@app.get("/events/telegram-files")
async def telegram_file_events(
    request: Request,
    app_key: Optional[str] = Query(default=None),
    x_app_key: Optional[str] = Header(default=None),
):
    if APP_KEY and x_app_key != APP_KEY and app_key != APP_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    global _telegram_event_loop
    _telegram_event_loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    with _telegram_event_clients_lock:
        _telegram_event_clients.add(queue)

    async def event_stream():
        try:
            yield _format_sse({"type": "telegram_counts_updated", "counts": _telegram_counts_payload()})
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=25)
                    yield _format_sse(message)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            with _telegram_event_clients_lock:
                _telegram_event_clients.discard(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/telegram-files")
def telegram_files(
    status: Optional[str] = Query(default=None),
    query: Optional[str] = Query(default=None),
    touched: str = Query(default="false", description="Filter touched files: false|true|all"),
) -> Dict[str, Any]:
    touched_filter: Optional[bool]
    normalized_touched = str(touched or "false").strip().lower()
    if normalized_touched in {"all", "any", ""}:
        touched_filter = None
    elif normalized_touched in {"1", "true", "yes", "on", "touched"}:
        touched_filter = True
    elif normalized_touched in {"0", "false", "no", "off", "active", "new", "untouched"}:
        touched_filter = False
    else:
        raise HTTPException(status_code=400, detail="Invalid touched filter")
    items = list_telegram_files(status=status, query=query, touched=touched_filter, limit=250)
    counts = _telegram_counts_payload()
    return {"items": items, **counts, "counts": counts}


@app.post("/telegram-files/{file_id}/touch")
def touch_telegram_file(file_id: int, request: Request):
    touched_by = request.headers.get("X-User") or None
    record = touch_telegram_file_record(file_id, touched_by=touched_by)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    _broadcast_telegram_file_change("telegram_file_updated", record)
    counts = _telegram_counts_payload()
    return {"ok": True, "file": record, **counts, "counts": counts}


@app.post("/telegram-files/{file_id}/mark-labels-printed")
def mark_telegram_labels_printed(file_id: int, request: Request):
    existing = get_telegram_file(file_id)
    if not existing:
        raise HTTPException(status_code=404, detail="File not found")
    if not existing.get("linked_order_id"):
        raise HTTPException(status_code=409, detail="No linked order yet")
    touched_by = request.headers.get("X-User") or None
    record = mark_telegram_file_labels_printed(file_id, touched_by=touched_by)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    _broadcast_telegram_file_change("telegram_file_updated", record)
    counts = _telegram_counts_payload()
    return {"ok": True, "file": record, **counts, "counts": counts}


@app.post("/telegram-files/{file_id}/mark-linked-order-opened")
def mark_telegram_linked_order_opened(file_id: int, request: Request):
    existing = get_telegram_file(file_id)
    if not existing:
        raise HTTPException(status_code=404, detail="File not found")
    if not existing.get("linked_order_id"):
        raise HTTPException(status_code=409, detail="No linked order yet")
    touched_by = request.headers.get("X-User") or None
    record = mark_telegram_file_linked_order_opened(file_id, touched_by=touched_by)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    _broadcast_telegram_file_change("telegram_file_updated", record)
    counts = _telegram_counts_payload()
    return {"ok": True, "file": record, **counts, "counts": counts}


@app.post("/telegram-files/{file_id}/mark-pdf-printed")
def mark_telegram_pdf_printed(file_id: int):
    print(f"[telegram-files] mark pdf printed called for telegram file {file_id}")
    record = mark_telegram_file_pdf_printed(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    _broadcast_telegram_file_change("telegram_file_updated", record)
    counts = _telegram_counts_payload()
    return {"ok": True, "file": record, **counts, "counts": counts}


@app.delete("/telegram-files/{file_id}")
def delete_telegram_file(
    file_id: int,
    request: Request,
    also_delete_linked_order: bool = Query(default=False),
) -> Dict[str, Any]:
    existing = get_telegram_file(file_id)
    if not existing:
        raise HTTPException(status_code=404, detail="File not found")
    warning = None
    linked_order_deleted = False
    linked_order = existing.get("linked_order") or None
    if existing.get("linked_order_id") and also_delete_linked_order:
        linked_status = normalize_order_status(linked_order.get("status") if linked_order else None, default="")
        if linked_order and linked_status == "draft":
            archived = update_order_status(
                int(existing["linked_order_id"]),
                status="archived",
                note="Archived from Telegram file delete action.",
                reason="telegram_file_delete",
            )
            linked_order_deleted = bool(archived)
        else:
            warning = "Linked order is not draft and was not deleted."
    elif existing.get("linked_order_id"):
        linked_status = normalize_order_status(linked_order.get("status") if linked_order else None, default="")
        if linked_status and linked_status != "draft":
            warning = "Linked order is not draft and was not deleted."
    record = soft_delete_telegram_file_record(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    _broadcast_telegram_file_change("telegram_file_deleted", record)
    counts = _telegram_counts_payload()
    return {
        "ok": True,
        "file": record,
        "linked_order_deleted": linked_order_deleted,
        "warning": warning,
        **counts,
        "counts": counts,
    }


@app.get("/telegram-files/{file_id}/view")
def view_telegram_file(file_id: int):
    record = get_telegram_file(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    mime_type = str(record.get("mime_type") or "application/pdf")
    if mime_type != "application/pdf" and not mime_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only PDF and image files can be viewed")
    path = _safe_telegram_file_path(record)
    headers = {"Content-Disposition": f'inline; filename="{_safe_download_filename(record.get("original_filename"))}"'}
    return FileResponse(path, media_type=mime_type, headers=headers)


@app.get("/telegram-files/{file_id}/download")
def download_telegram_file(file_id: int):
    record = get_telegram_file(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    path = _safe_telegram_file_path(record)
    return FileResponse(
        path,
        media_type=record.get("mime_type") or "application/pdf",
        filename=_safe_download_filename(record.get("original_filename")),
    )


@app.get("/manual-orders")
def get_manual_orders(
    query: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    try:
        items = db_module.list_manual_orders(
            query=query,
            status=status,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "items": items,
        "count": len(items),
        "limit": limit,
        "offset": offset,
        "has_more": len(items) == limit,
    }


@app.get("/manual-orders/check-number")
def check_manual_order_number(
    order_number: str = Query(min_length=1),
    exclude_id: Optional[int] = Query(default=None, ge=1),
) -> Dict[str, Any]:
    return {
        "exists": db_module.manual_order_number_exists(
            order_number,
            exclude_id=exclude_id,
        )
    }


@app.post("/manual-orders", status_code=201)
def add_manual_order(payload: ManualOrderPayload) -> Dict[str, Any]:
    try:
        return db_module.create_manual_order(payload.model_dump(mode="json"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/manual-orders/{order_id}")
def get_manual_order_detail(order_id: int) -> Dict[str, Any]:
    order = db_module.get_manual_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Manual order not found")
    return order


@app.put("/manual-orders/{order_id}")
def replace_manual_order(order_id: int, payload: ManualOrderPayload) -> Dict[str, Any]:
    try:
        return db_module.update_manual_order(
            order_id,
            payload.model_dump(mode="json"),
        )
    except ValueError as exc:
        message = str(exc)
        raise HTTPException(
            status_code=404 if "not found" in message.lower() else 400,
            detail=message,
        )


@app.post("/manual-orders/{order_id}/duplicate", status_code=201)
def copy_manual_order(order_id: int) -> Dict[str, Any]:
    try:
        return db_module.duplicate_manual_order(order_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/manual-orders/{order_id}/processing")
def process_manual_order(order_id: int) -> Dict[str, Any]:
    try:
        return db_module.send_manual_order_to_processing(order_id)
    except ValueError as exc:
        message = str(exc)
        raise HTTPException(
            status_code=404 if "not found" in message.lower() else 400,
            detail=message,
        )


@app.delete("/manual-orders/{order_id}")
def remove_manual_order(order_id: int) -> Dict[str, bool]:
    if not db_module.delete_manual_order(order_id):
        raise HTTPException(status_code=404, detail="Manual order not found")
    return {"ok": True}


@app.get("/orders")
def list_orders(
    query: Optional[str] = Query(default=None, description="Search by order number or client hint"),
    status: Optional[str] = Query(
        default=None,
        description="Filter by status: draft|reviewed|approved|in_production|completed|archived",
    ),
    client: Optional[str] = Query(default=None, description="Filter by client (contains, case-insensitive)"),
    date_from: Optional[str] = Query(default=None, description="Start date (inclusive), ISO date YYYY-MM-DD"),
    date_to: Optional[str] = Query(default=None, description="End date (inclusive), ISO date YYYY-MM-DD"),
    approved_only: bool = Query(default=False, description="Shortcut for status=approved"),
    year: Optional[str] = Query(default=None, description="Year filter: YYYY (defaults to current year) or all"),
    limit: int = Query(default=DEFAULT_HISTORY_LIMIT, ge=1, le=MAX_HISTORY_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    try:
        items = get_orders(
            query=query,
            status=status,
            client=client,
            date_from=date_from,
            date_to=date_to,
            approved_only=approved_only,
            year=year,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "items": items,
        "limit": limit,
        "offset": offset,
        "count": len(items),
        "has_more": len(items) == limit,
    }


@app.get("/orders/export.csv")
def export_all_orders_csv(
    query: Optional[str] = Query(default=None, description="Search by order number or client hint"),
    status: Optional[str] = Query(
        default="approved",
        description="Filter by status (default approved). Supports lifecycle statuses.",
    ),
    client: Optional[str] = Query(default=None, description="Filter by client (contains, case-insensitive)"),
    date_from: Optional[str] = Query(default=None, description="Start date (inclusive), ISO date YYYY-MM-DD"),
    date_to: Optional[str] = Query(default=None, description="End date (inclusive), ISO date YYYY-MM-DD"),
    approved_only: bool = Query(default=False, description="Shortcut for status=approved"),
    year: Optional[str] = Query(default=None, description="Year filter: YYYY (defaults to current year) or all"),
):
    try:
        rows = get_all_rows_for_export(
            query=query,
            status=status,
            client=client,
            date_from=date_from,
            date_to=date_to,
            approved_only=approved_only,
            year=year,
        )
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
    order["rows"] = _with_stored_row_locations(order["rows"], extraction)
    order["rows"] = _with_stored_ocr_overlays(order["rows"], extraction)
    order["row_warnings"] = validation.get("row_warnings", {})
    order["row_warnings"] = _merge_critical_row_warnings(order["row_warnings"], order["rows"])
    order["warnings"] = validation.get("warnings", [])
    parsed_declared_units, parsed_declared_area = parse_declared_totals(extraction.get("prepared_text", ""))
    stored_declared_units, stored_declared_area = _stored_declared_totals(extraction)
    declared_units, declared_area = _declared_totals_with_fallback(
        parsed_declared_units,
        parsed_declared_area,
        stored_declared_units,
        stored_declared_area,
    )
    order["declared_units"] = declared_units
    order["declared_area"] = declared_area
    totals = _summarize_totals(order["rows"])
    order["parsed_units"] = totals["units"]
    order["parsed_area"] = totals["area"]
    order_total_diagnostics = _order_total_diagnostics_from_totals(declared_units, declared_area, totals)
    order["order_total_diagnostics"] = order_total_diagnostics
    order["warnings"] = _dedupe_warnings(_append_order_total_warning(order["warnings"], order_total_diagnostics))
    order["rows"] = _with_extraction_diagnostics(order["rows"])
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
    current_status = normalize_order_status(snapshot.get("status"))
    if current_status not in APPROVABLE_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Order in status '{current_status}' cannot be approved.",
        )

    rows_dict = [row.model_dump(mode="python") for row in payload.rows]
    client_name = _normalize_client_name_payload(
        payload.client_name,
        payload.clientName,
        payload.client,
        snapshot.get("client_name"),
        snapshot.get("clientName"),
        snapshot.get("client"),
        snapshot.get("client_hint"),
    )

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
            client_name=client_name,
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
    parsed_declared_units, parsed_declared_area = parse_declared_totals(prepared_text)
    stored_declared_units, stored_declared_area = _stored_declared_totals(extraction)
    declared_units, declared_area = _declared_totals_with_fallback(
        parsed_declared_units,
        parsed_declared_area,
        stored_declared_units,
        stored_declared_area,
    )
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
    order_total_diagnostics = _order_total_diagnostics_from_totals(declared_units, declared_area, totals)
    _append_order_total_warning(combined_warnings, order_total_diagnostics)
    combined_warnings = _dedupe_warnings(combined_warnings)
    updated_order["rows"] = _with_extraction_diagnostics(updated_order.get("rows") or [])
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
        "order_total_diagnostics": order_total_diagnostics,
    }


@app.post("/orders/{order_id}/status")
def set_order_status(order_id: int, payload: StatusUpdatePayload) -> Dict[str, Any]:
    target = normalize_order_status(payload.status)
    if target not in ORDER_STATUS_SEQUENCE:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{payload.status}'. Allowed: {', '.join(ORDER_STATUS_SEQUENCE)}",
        )
    try:
        updated = update_order_status(
            order_id,
            status=target,
            note=payload.note,
            reason="api_status_update",
        )
        return {"ok": True, "order": updated}
    except ValueError as exc:
        message = str(exc)
        code = 404 if "not found" in message.lower() else 400
        raise HTTPException(status_code=code, detail=message)
    except Exception as exc:
        print("[set_order_status] update failed:\n" + "".join(traceback.format_exc()))
        raise HTTPException(status_code=500, detail=f"Failed to update status: {exc}")


@app.post("/orders/{order_id}/archive")
def archive_order(order_id: int, payload: Optional[StatusUpdatePayload] = None) -> Dict[str, Any]:
    note = payload.note if payload else None
    try:
        updated = update_order_status(
            order_id,
            status="archived",
            note=note,
            reason="archive",
        )
        return {"ok": True, "order": updated}
    except ValueError as exc:
        message = str(exc)
        code = 404 if "not found" in message.lower() else 400
        raise HTTPException(status_code=code, detail=message)
    except Exception as exc:
        print("[archive_order] update failed:\n" + "".join(traceback.format_exc()))
        raise HTTPException(status_code=500, detail=f"Failed to archive order: {exc}")


@app.get("/orders/{order_id}/csv")
def download_order_csv(order_id: int):
    order = get_order_with_extraction(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    if not is_processing_eligible_status(order.get("status")):
        raise HTTPException(status_code=400, detail="Order not approved")
    csv_content = _rows_to_csv(order.get("rows") or [])
    filename = f"order-{order_id}.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=csv_content, media_type="text/csv", headers=headers)


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
        f"Analytics snapshot:\n{_clip(dataset_json, ANALYSIS_DATASET_MAX_CHARS)}\n\n"
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


@app.get("/analysis/summary")
def analysis_summary(
    start_date: Optional[str] = Query(default=None, description="Current period start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(default=None, description="Current period end date (YYYY-MM-DD)"),
    compare_previous: bool = Query(default=True),
    client: Optional[str] = Query(default=None),
    glass_type: Optional[str] = Query(default=None),
    tolerance_mm: float = Query(default=1.0, ge=0, le=5),
    orientation_agnostic: bool = Query(default=True),
    all_time: bool = Query(default=False),
) -> Dict[str, Any]:
    try:
        orders = get_analysis_orders(ANALYTICS_STATUSES)
        return build_analysis_summary(
            orders,
            start_date=start_date,
            end_date=end_date,
            compare_previous=compare_previous,
            client=(client or "").strip(),
            glass_type=(glass_type or "").strip(),
            tolerance_mm=tolerance_mm,
            orientation_agnostic=orientation_agnostic,
            all_time=all_time,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print("[analysis_summary] failed:\n" + "".join(traceback.format_exc()))
        raise HTTPException(status_code=500, detail=f"Failed to build analytics summary: {exc}")


@app.post("/analysis/signals")
def analysis_signals(payload: AnalysisSignalsPayload) -> Dict[str, Any]:
    summary = payload.summary or {}
    if not isinstance(summary, dict):
        raise HTTPException(status_code=400, detail="Summary must be an object.")
    signals = generate_analysis_signals(summary)
    return {
        "signals": signals,
        "count": len(signals),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

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
