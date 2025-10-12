from __future__ import annotations
import csv, hashlib, json, os
from io import StringIO
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Query
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
)
from dotenv import load_dotenv
import traceback
import openai as openai_pkg
from db import (
    init_db,
    insert_extraction_with_rows,
    update_order_rows,
    get_orders,
    get_order_with_extraction,
    delete_order,
    get_all_rows_for_export,
    save_correction,
    list_corrections,
    delete_correction,
)
from validators import validate_rows
from utils_text import parse_declared_totals
load_dotenv()

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
APP_KEY = os.getenv("APP_KEY")  # optional shared secret

app = FastAPI(title="LLM Order Extractor (Local)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://danjelqose1.github.io",  # your GitHub Pages frontend
        "https://order-extractor-kdih.onrender.com",  # backend self-origin
        "http://127.0.0.1:5055",  # local dev
        "http://localhost:5055"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PasteIn(BaseModel):
    text: str


class ApprovePayload(BaseModel):
    rows: List[Row]
    notes: Optional[str] = None


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


init_db()

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
        validation = validate_rows(rows_dict, context={"prepared_text": bundle.get("prepared_text")})
        final_rows = validation.get("rows", rows_dict)
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
        validation = validate_rows(rows_dict, context={"prepared_text": prepared_text})
        final_rows = validation.get("rows", rows_dict)
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

        raw_joined = "\n\n".join(page for page in raw_ocr_pages if page)
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
    limit: int = Query(default=DEFAULT_HISTORY_LIMIT, ge=1, le=MAX_HISTORY_LIMIT),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    items = get_orders(query=query, status=status, limit=limit, offset=offset)
    return {
        "items": items,
        "limit": limit,
        "offset": offset,
        "count": len(items),
        "has_more": len(items) == limit,
    }


@app.get("/orders/{order_id}")
def get_order_detail(order_id: int) -> Dict[str, Any]:
    order = get_order_with_extraction(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    rows = order.get("rows") or []
    extraction = order.get("extraction") or {}
    validation = validate_rows([dict(row) for row in rows], context={"prepared_text": extraction.get("prepared_text", "")})
    order["rows"] = validation.get("rows", rows)
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
    validation = validate_rows(rows_dict, context={"prepared_text": snapshot.get("extraction", {}).get("prepared_text", "")})
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
def export_all_orders_csv():
    rows = get_all_rows_for_export()
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
