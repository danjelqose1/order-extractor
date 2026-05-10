from __future__ import annotations
import asyncio
import base64, csv, hashlib, json, os, re, sys
from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
import uuid
import threading
import time

BACKEND_DIR = os.path.dirname(__file__)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, ValidationError
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
    create_telegram_file_record,
    update_telegram_file_record,
    touch_telegram_file_record,
    soft_delete_telegram_file_record,
    mark_telegram_file_labels_printed,
    mark_telegram_file_linked_order_opened,
    list_telegram_files,
    count_untouched_telegram_files,
    get_telegram_file_counts,
    find_telegram_file_record,
    find_telegram_file_by_sha256,
    find_possible_duplicate_order,
    list_unfinished_telegram_file_ids,
    get_telegram_file,
)
from validators import validate_rows
from dimension_repair import apply_dimension_repair
from area_dimension_validator import apply_area_dimension_validation
from extraction_normalizer import normalize_extracted_rows
from utils_text import clean_dimension, parse_declared_totals
from prompts import PROMPTS
from analysis_signals import generate_analysis_signals
from services.pdf_native_text_editor import native_text_replace
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH, override=True)

# Data/config paths
DATA_DIR = Path(os.getenv("DB_DIR", "data"))
PRICE_CONFIG_PATH = DATA_DIR / "price-config.json"
INVOICES_PATH = DATA_DIR / "invoices.json"

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
APP_KEY = os.getenv("APP_KEY")  # optional shared secret
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gpt-5-mini")
LEGACY_OCR_ENABLED = os.getenv("LEGACY_OCR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
TELEGRAM_MAX_FILE_BYTES = 5 * 1024 * 1024
TELEGRAM_SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"
TELEGRAM_EXTRACTION_MAX_RETRIES = 2
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


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/workspace/queue")
def workspace_queue() -> Dict[str, Any]:
    from workspace_service import get_workspace_queue

    return get_workspace_queue()


@app.get("/api/workspace/recent-files")
def workspace_recent_files() -> Dict[str, Any]:
    from workspace_service import get_recent_production_files

    return get_recent_production_files()


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
        combined_warnings = _dedupe_warnings(combined_warnings)

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
            "rows": final_rows,
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

    try:
        if is_pdf:
            try:
                bundle = call_llm_for_pdf_base64_visual(raw_bytes, filename=normalized_filename)
            except Exception as visual_exc:
                if not LEGACY_OCR_ENABLED:
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
        prepared_text = prepared_text or bundle.get("prepared_text") or ""
        repair_context = prepared_text or (bundle.get("output_text") or "")
        repaired_rows = apply_dimension_repair(repair_context, rows_dict)
        extracted_rows, normalization_warnings = normalize_extracted_rows(repaired_rows)
        validation = validate_rows(extracted_rows, context={"prepared_text": prepared_text})
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
        if fallback_warning:
            combined_warnings.append(fallback_warning)

        applied = bundle.get("applied_corrections") or []
        if applied:
            combined_warnings.append(f"auto_corrections_applied:{','.join(str(i) for i in applied)}")

        metadata = dict(source_metadata or {})
        metadata.setdefault("source", source)
        metadata.setdefault("original_filename", normalized_filename)

        llm_output_payload = dict(raw_payload) if isinstance(raw_payload, dict) else {"raw_payload": raw_payload}
        llm_output_payload["_meta"] = {
            "raw_response": bundle.get("raw_response"),
            "parsed_result": bundle.get("data") or raw_payload,
            "extraction_method": extraction_method,
            "source_metadata": metadata,
        }
        llm_output_json = json.dumps(llm_output_payload, ensure_ascii=False)
        hash_value = _compute_hash_bytes(raw_bytes)
        declared_units = bundle.get("declared_units")
        declared_area = bundle.get("declared_area")
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
        if declared_units is not None and declared_units != totals["units"]:
            combined_warnings.append(
                f"declared_units_mismatch: declared {declared_units}, parsed {totals['units']}"
            )
        if declared_area is not None and abs((declared_area or 0) - totals["area"]) > 0.05:
            combined_warnings.append(
                f"declared_area_mismatch: declared {declared_area:.3f}, parsed {totals['area']:.3f}"
            )
        combined_warnings = _dedupe_warnings(combined_warnings)

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
                reason = (
                    "Extracted order appears to match recent draft "
                    f"#{possible_duplicate.get('id')} by order number, client, units, and area."
                )
                update_telegram_file_record(
                    int(telegram_file_record_id),
                    extraction_status="possible_duplicate",
                    duplicate_status="possible_duplicate",
                    duplicate_reason=reason,
                    processed_at=datetime.now(timezone.utc),
                    clear_last_error=True,
                )
                return {
                    "order_number": (bundle.get("data") or {}).get("order_number") or _primary_order_number(final_rows),
                    "rows": final_rows,
                    "warnings": _dedupe_warnings(combined_warnings + ["possible_duplicate_order"]),
                    "row_warnings": row_warnings,
                    "draft_order_id": None,
                    "saved_order_id": None,
                    "status": "possible_duplicate",
                    "duplicate_status": "possible_duplicate",
                    "duplicate_of_order_id": possible_duplicate.get("id"),
                    "duplicate_reason": reason,
                    "parsed_units": totals["units"],
                    "parsed_area": totals["area"],
                    "extraction_method": extraction_method,
                    "confidence": (bundle.get("data") or {}).get("confidence"),
                    "client_name": client_name,
                    "clientName": client_name,
                    "client": client_name or "—",
                }

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

        return {
            "order_number": (bundle.get("data") or {}).get("order_number") or _primary_order_number(final_rows),
            "rows": final_rows,
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
            "extraction_method": extraction_method,
            "confidence": (bundle.get("data") or {}).get("confidence"),
            "client_name": client_name,
            "clientName": client_name,
            "client": client_name or "—",
        }
    except HTTPException:
        raise
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=f"Validation error: {ve.errors()}")
    except Exception as e:
        print(f"[extract_file:{source}] fatal error:\n" + "".join(traceback.format_exc()))
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.post("/extract_pdf")
async def extract_pdf(file: UploadFile = File(...), x_app_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
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
    )


@app.post("/extract-pdf")
async def extract_pdf_dash_alias(file: UploadFile = File(...), x_app_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    return await extract_pdf(file=file, x_app_key=x_app_key)


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


async def _telegram_reply(token: str, chat_id: Any, message_id: Any, text: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "reply_to_message_id": message_id,
                    "allow_sending_without_reply": True,
                },
            )
    except Exception as exc:
        print(f"[telegram] reply failed: {exc}")


async def _telegram_download_file(token: str, file_id: str) -> Tuple[bytes, Dict[str, Any]]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        file_response = await client.get(
            f"https://api.telegram.org/bot{token}/getFile",
            params={"file_id": file_id},
        )
        file_response.raise_for_status()
        file_payload = file_response.json()
        if not file_payload.get("ok") or not isinstance(file_payload.get("result"), dict):
            raise RuntimeError("Telegram getFile returned an invalid response")
        result = file_payload["result"]
        file_path = result.get("file_path")
        if not file_path:
            raise RuntimeError("Telegram getFile did not return a file path")
        reported_size = result.get("file_size")
        if isinstance(reported_size, int) and reported_size > TELEGRAM_MAX_FILE_BYTES:
            raise ValueError("telegram_file_too_large")

        download_response = await client.get(f"https://api.telegram.org/file/bot{token}/{file_path}")
        download_response.raise_for_status()
        content = download_response.content
        if len(content) > TELEGRAM_MAX_FILE_BYTES:
            raise ValueError("telegram_file_too_large")
        return content, result


def _telegram_token_optional() -> str:
    return (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()


def _short_error_message(exc: Exception) -> str:
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
        await asyncio.gather(*(_process_telegram_queue_job(file_id) for file_id in batch))


def _recover_telegram_extraction_queue() -> None:
    stale_before = datetime.now(timezone.utc) - timedelta(minutes=10)
    try:
        file_ids = list_unfinished_telegram_file_ids(stale_processing_before=stale_before)
    except Exception as exc:
        print(f"[telegram-queue] recovery failed: {exc}")
        return
    for file_id in file_ids:
        _enqueue_telegram_extraction(file_id)


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
    if record.get("extraction_status") in {"duplicate", "possible_duplicate"}:
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
            if result.get("status") == "possible_duplicate":
                refreshed = get_telegram_file(file_id) or latest
                response_record = {**latest, **refreshed}
                _broadcast_telegram_file_change("telegram_file_updated", response_record)
                return {
                    "status": "possible_duplicate",
                    "file_id": file_id,
                    "record": response_record,
                    "duplicate_of_order_id": result.get("duplicate_of_order_id"),
                }
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
            }
        except Exception as exc:
            last_error = _short_error_message(exc)
            print(f"[telegram-queue] extraction attempt failed file_id={file_id} attempt={attempts_done}: {last_error}")
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


async def _process_telegram_queue_job(file_id: int) -> Dict[str, Any]:
    try:
        outcome = await asyncio.to_thread(_process_telegram_queue_job_sync, int(file_id))
    finally:
        with _telegram_queue_lock:
            _telegram_queued_ids.discard(int(file_id))

    record = outcome.get("record") or {}
    token = _telegram_token_optional()
    chat_id = record.get("telegram_chat_id")
    message_id = record.get("telegram_message_id")
    if token and chat_id is not None:
        if outcome.get("status") == "extracted":
            await _telegram_reply(token, chat_id, message_id, "Extraction finished ✅ Please review in platform.")
        elif outcome.get("status") == "possible_duplicate":
            await _telegram_reply(token, chat_id, message_id, "Possible duplicate detected ⚠️ Please review in platform.")
        elif outcome.get("status") == "failed":
            await _telegram_reply(token, chat_id, message_id, "Extraction failed ⚠️ Original PDF is saved in Telegram Files.")
    return outcome


def _telegram_document_spec(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    document = message.get("document")
    if not isinstance(document, dict):
        return None
    filename = str(document.get("file_name") or "telegram-upload").strip() or "telegram-upload"
    mime_type = str(document.get("mime_type") or "").strip().lower()
    if not _is_pdf_file(filename, mime_type) and not _is_image_file(filename, mime_type):
        return {
            "unsupported": True,
            "file_size": document.get("file_size"),
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


async def _handle_telegram_update(update: Dict[str, Any]) -> Dict[str, Any]:
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
            duplicate_of = find_telegram_file_by_sha256(file_sha256)
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
        print(f"[telegram] value error chat={chat_id} message={message_id}: {exc}")
    except HTTPException as exc:
        print(f"[telegram] extraction failed chat={chat_id} message={message_id}: {exc.detail}")
    except Exception as exc:
        print(f"[telegram] extraction failed chat={chat_id} message={message_id}: {exc}")
        print("[telegram] traceback:\n" + "".join(traceback.format_exc()))

    if chat_id is not None:
        await _telegram_reply(token, chat_id, message_id, "Extraction failed. Please try again.")
    return {"ok": True, "status": "failed", "telegram_file_id": None}


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
