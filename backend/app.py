from __future__ import annotations
import json, os, re
from io import BytesIO
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from schema import ExtractionResult
from llm import call_llm_for_extraction, call_llm_for_extraction_multi
from dotenv import load_dotenv
import traceback
import openai as openai_pkg
load_dotenv()

try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LTTextLine
except Exception:
    extract_pages = None
    LTTextContainer = None
    LTTextLine = None

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
APP_KEY = os.getenv("APP_KEY")  # optional shared secret

app = FastAPI(title="LLM Order Extractor (Local)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PasteIn(BaseModel):
    text: str


def normalize_position(value: str) -> str:
    if not value:
        return value

    cleaned = re.sub(r"\s+", "", value.strip())

    # Remove leading order prefix like R-25-0864/
    cleaned = re.sub(r"^[A-Za-z]-?\d{2,}-\d{4}/", "", cleaned, flags=re.IGNORECASE)

    segments = cleaned.split('/')
    base = segments[0] if segments else ''
    suffix = '/'.join(segments[1:]) if len(segments) > 1 else ''

    if base and '-' not in base:
        digits = re.findall(r"\d+", base)
        if len(digits) >= 2:
            base = f"{digits[0]}-{digits[1]}"
        elif len(digits) == 1 and len(digits[0]) >= 3:
            base = f"{digits[0][:-1]}-{digits[0][-1]}"

    base = base.strip()
    if suffix:
        return f"{base}/{suffix.strip()}" if base else suffix.strip()
    return base


def _debug_log_rows(rows):
    try:
        print(f"[debug] rows extracted: {len(rows)}")
        sample_positions = ["22-1", "40-1", "41-1", "65-2", "79-2", "93-2", "100-2"]
        lookup = {r.position: r for r in rows}
        for pos in sample_positions:
            row = lookup.get(pos)
            if row:
                print(f"[debug] sample {pos}: dimension={row.dimension}, area={row.area}")
            else:
                print(f"[debug] sample {pos}: missing")
    except Exception as exc:
        print(f"[debug] logging error: {exc}")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/extract")
def extract(inb: PasteIn, x_app_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if APP_KEY and x_app_key != APP_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        raw = call_llm_for_extraction(inb.text)
        result = ExtractionResult(**raw)

        for row in result.rows:
            row.position = normalize_position(row.position)

        _debug_log_rows(result.rows)

        return json.loads(result.model_dump_json())
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=f"Validation error: {ve.errors()}")
    except Exception as e:
        print("[extract] fatal error:\n" + "".join(traceback.format_exc()))
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

def extract_pages_text(pdf_bytes: bytes) -> list[str]:
    if not extract_pages or not LTTextContainer or not LTTextLine:
        raise RuntimeError("PDF extraction backend not installed. Ensure pdfminer.six is in requirements.txt")

    pages_text: list[str] = []
    for page_layout in extract_pages(BytesIO(pdf_bytes)):
        lines: list[str] = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for line in element:
                    if isinstance(line, LTTextLine):
                        txt = line.get_text()
                        txt = txt.replace("-\n", "").replace("\r", "")
                        lines.append(txt)
        pages_text.append("".join(lines))
    return pages_text


@app.post("/extract_pdf")
async def extract_pdf(file: UploadFile = File(...), x_app_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """Accept a PDF file upload, extract text per page (no OCR), then run the multi-page LLM extraction."""
    if APP_KEY and x_app_key != APP_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not file or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    try:
        raw_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {e}")

    if not raw_bytes:
        raise HTTPException(status_code=422, detail="The PDF contains no extractable text (is it scanned? No OCR is performed).")

    try:
        pages_text = extract_pages_text(raw_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {e}")

    if not any((page or "").strip() for page in pages_text):
        raise HTTPException(status_code=422, detail="The PDF contains no extractable text (is it scanned? No OCR is performed).")

    try:
        raw = call_llm_for_extraction_multi(pages_text)
        result = ExtractionResult(**raw)

        for row in result.rows:
            row.position = normalize_position(row.position)

        _debug_log_rows(result.rows)

        return json.loads(result.model_dump_json())
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=f"Validation error: {ve.errors()}")
    except Exception as e:
        print("[extract_pdf] fatal error:\n" + "".join(traceback.format_exc()))
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.post("/extract-pdf")
async def extract_pdf_dash_alias(file: UploadFile = File(...), x_app_key: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    return await extract_pdf(file=file, x_app_key=x_app_key)

@app.get("/diag")
def diag():
    try:
        sdk_ver = getattr(openai_pkg, "__version__", "unknown")
    except Exception:
        sdk_ver = "unknown"
    return {
        "status": "ok",
        "openai_sdk_version": sdk_ver,
        "env_has_api_key": bool(os.getenv("OPENAI_API_KEY")),
    }
