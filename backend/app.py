from __future__ import annotations
import json, os
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from schema import ExtractionResult
from llm import call_llm_for_extraction
from dotenv import load_dotenv
load_dotenv()

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

        # Normalize positions to short code (strip prefix like R-25-0864/)
        for row in result.rows:
            if "/" in row.position:
                row.position = row.position.split("/")[-1].strip()

        return json.loads(result.model_dump_json())
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=f"Validation error: {ve.errors()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
