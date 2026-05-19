from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

from backend.agents.skills.extraction_diagnostics import (
    OPENAI_VISION_PAGE_UNAVAILABLE_REASON,
    diagnose_extraction_row_issue,
    diagnose_extraction_row_warning,
    ocr_fallback_row_repair,
)
from backend.agents.skills.family_pattern import analyze_dimension_family
from backend.agents.skills.pattern_repair import suggest_pattern_repair


FAMILY_REPAIR_THRESHOLD = 0.65
PATTERN_REPAIR_CONFIDENCE_THRESHOLD = 0.8


def _issue_codes(diagnostics: Optional[Dict[str, Any]]) -> List[str]:
    issues = diagnostics.get("issues") if isinstance(diagnostics, dict) else None
    if not isinstance(issues, list):
        return []
    return [str(issue.get("code") or "").strip().upper() for issue in issues if isinstance(issue, dict)]


def _select_target_field(diagnostics: Dict[str, Any], requested: Optional[str] = None) -> str:
    allowed = {"dimension", "type", "quantity", "area", "position"}
    if requested in allowed:
        return str(requested)
    codes = set(_issue_codes(diagnostics))
    if codes & {
        "MISSING_DIMENSION",
        "INVALID_DIMENSION",
        "INVALID_DIMENSION_FORMAT",
        "DIMENSION_OUT_OF_RANGE",
        "SUSPICIOUS_DIMENSION_SIZE",
        "AREA_MISMATCH",
        "POSSIBLE_DIMENSION_OCR_ERROR",
        "POSSIBLE_DIMENSION_FAMILY_MISMATCH",
    }:
        return "dimension"
    if codes & {"INVALID_AREA", "MISSING_EXTRACTED_AREA", "INVALID_EXTRACTED_AREA"}:
        return "area"
    if codes & {"MISSING_QUANTITY", "INVALID_QUANTITY"}:
        return "quantity"
    if "GLASS_TYPE_UNCLEAR" in codes:
        return "type"
    if codes & {"MISSING_POSITION", "POSITION_WARNING", "EMPTY_POSITION", "DUPLICATE_POSITION"}:
        return "position"
    return "dimension"


def _original_value(row: Dict[str, Any], target_field: str) -> Any:
    if target_field == "type":
        return row.get("type", row.get("glass_type"))
    if target_field == "area":
        for key in ("area", "extracted_area", "area_m2"):
            if key in row:
                return row.get(key)
        return None
    return row.get(target_field)


def _has_row_location(row: Dict[str, Any]) -> bool:
    return isinstance(row.get("row_location"), dict)


def _should_attempt_openai_vision_page_ocr(codes: List[str], target_field: str) -> bool:
    code_set = set(codes or [])
    if target_field == "dimension":
        return bool(code_set & {"MISSING_DIMENSION", "INVALID_DIMENSION", "INVALID_DIMENSION_FORMAT"})
    if target_field == "position":
        return bool(code_set & {"MISSING_POSITION", "POSITION_WARNING", "EMPTY_POSITION", "DUPLICATE_POSITION"})
    return False


def _response(
    *,
    success: bool,
    target_field: str,
    original_value: Any,
    suggested_value: Any,
    confidence: float,
    recommended_action: str,
    reasoning: str,
    evidence: Dict[str, Any],
    trace: List[str],
    methods_used: List[str],
) -> Dict[str, Any]:
    return {
        "success": success,
        "target_field": target_field,
        "original_value": original_value,
        "suggested_value": suggested_value,
        "confidence": max(0.0, min(1.0, float(confidence))),
        "recommended_action": recommended_action,
        "reasoning": reasoning,
        "reason": reasoning,
        "evidence": evidence,
        "trace": trace,
        "methods_used": methods_used,
        "method": methods_used[-1] if methods_used else "manual_review",
        "safe_to_auto_apply": False,
    }


def repair_suspicious_row(
    row: Dict[str, Any],
    diagnostics: Optional[Dict[str, Any]] = None,
    nearby_rows: Optional[List[Dict[str, Any]]] = None,
    order_rows: Optional[List[Dict[str, Any]]] = None,
    order_context: Optional[Dict[str, Any]] = None,
    optional_pdf_context: Optional[Dict[str, Any]] = None,
    target_field: Optional[str] = None,
    row_index: Optional[int] = None,
    pdf_id: Optional[str] = None,
    pdf_bytes: Optional[bytes] = None,
    openai_vision_repair_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    working_row = deepcopy(row or {})
    working_context = deepcopy(order_context or {})
    working_nearby_rows = deepcopy(nearby_rows or [])
    working_order_rows = deepcopy(order_rows or [])
    working_pdf_context = deepcopy(optional_pdf_context or {})
    working_diagnostics = deepcopy(diagnostics) if isinstance(diagnostics, dict) else diagnose_extraction_row_issue(working_row)
    target = _select_target_field(working_diagnostics, target_field)
    original = _original_value(working_row, target)
    codes = _issue_codes(working_diagnostics)

    trace: List[str] = []
    methods_used = ["diagnostics_analyzer"]
    if codes:
        trace.append(f"Detected {', '.join(codes)}")
    else:
        trace.append("Diagnostics analyzer found no row-level warning or error")

    diagnosis = diagnose_extraction_row_warning(
        deepcopy(working_row),
        deepcopy(working_diagnostics),
        deepcopy(working_context),
    )
    trace.append(f"Deterministic diagnostics recommended {diagnosis.get('recommended_action') or 'MANUAL_REVIEW'}")

    family_result: Optional[Dict[str, Any]] = None
    if target == "dimension":
        family_result = analyze_dimension_family(
            deepcopy(working_row),
            nearby_rows=working_nearby_rows,
            order_rows=working_order_rows,
            order_context=working_context,
        )
        for step in family_result.get("trace") or []:
            if step not in trace:
                trace.append(str(step))
        if family_result.get("success") and float(family_result.get("confidence") or 0.0) >= FAMILY_REPAIR_THRESHOLD:
            methods_used.append("family_pattern_repair")
            return _response(
                success=True,
                target_field="dimension",
                original_value=family_result.get("original_value"),
                suggested_value=family_result.get("suggested_value"),
                confidence=float(family_result.get("confidence") or 0.0),
                recommended_action="PATTERN_REPAIR",
                reasoning=family_result.get("reasoning") or "Family pattern analysis found a supported candidate.",
                evidence={
                    "diagnostic_codes": codes,
                    "diagnosis": diagnosis,
                    "family_pattern": family_result.get("evidence") or {},
                },
                trace=trace,
                methods_used=methods_used,
            )
        trace.append(f"Family pattern confidence {float(family_result.get('confidence') or 0.0):.2f} below threshold")

    if str(working_diagnostics.get("severity") or "ok") not in {"warning", "error"}:
        trace.append("Repair skipped because diagnostics severity is ok")
        return _response(
            success=False,
            target_field=target,
            original_value=original,
            suggested_value=None,
            confidence=0.0,
            recommended_action="NO_REPAIR_NEEDED",
            reasoning="Backend diagnostics did not report a suspicious extraction row.",
            evidence={
                "diagnostic_codes": codes,
                "diagnosis": diagnosis,
                "family_pattern": (family_result or {}).get("evidence") or {},
            },
            trace=trace,
            methods_used=methods_used,
        )

    openai_result: Optional[Dict[str, Any]] = None
    if _should_attempt_openai_vision_page_ocr(codes, target):
        trace.append("Attempting OpenAI vision OCR fallback using page context")
        openai_result = ocr_fallback_row_repair(
            row=deepcopy(working_row),
            diagnostics=deepcopy(working_diagnostics),
            target_field=target,
            order_context=working_context,
            row_index=row_index,
            pdf_id=pdf_id or working_pdf_context.get("pdf_id"),
            pdf_bytes=pdf_bytes,
            openai_vision_repair_fn=openai_vision_repair_fn,
        )
        openai_method = str(openai_result.get("method") or "openai_vision_page_ocr")
        if openai_method not in methods_used:
            methods_used.append(openai_method)
        if openai_result.get("success"):
            trace.append(f"OpenAI vision OCR confidence {float(openai_result.get('confidence') or 0.0):.2f}")
            openai_evidence = deepcopy(openai_result.get("evidence") or {})
            return _response(
                success=True,
                target_field=openai_result.get("target_field") or target,
                original_value=openai_result.get("original_value"),
                suggested_value=openai_result.get("suggested_value"),
                confidence=float(openai_result.get("confidence") or 0.0),
                recommended_action="ACCEPT_OCR_SUGGESTION_OR_REVIEW",
                reasoning=openai_result.get("reason") or "OpenAI vision OCR found a supported correction.",
                evidence={
                    **openai_evidence,
                    "diagnostic_codes": codes,
                    "diagnosis": diagnosis,
                    "family_pattern": (family_result or {}).get("evidence") or {},
                    "ocr_fallback": openai_evidence,
                },
                trace=trace,
                methods_used=methods_used,
            )
        trace.append(openai_result.get("reason") or "OpenAI vision OCR fallback did not find a confident correction")
        if openai_result.get("reason") == OPENAI_VISION_PAGE_UNAVAILABLE_REASON:
            openai_evidence = deepcopy(openai_result.get("evidence") or {})
            return _response(
                success=False,
                target_field=openai_result.get("target_field") or target,
                original_value=openai_result.get("original_value"),
                suggested_value=None,
                confidence=0.0,
                recommended_action="MANUAL_REVIEW",
                reasoning=OPENAI_VISION_PAGE_UNAVAILABLE_REASON,
                evidence={
                    **openai_evidence,
                    "diagnostic_codes": codes,
                    "diagnosis": diagnosis,
                    "family_pattern": (family_result or {}).get("evidence") or {},
                    "ocr_fallback": openai_evidence,
                },
                trace=trace,
                methods_used=methods_used,
            )

    pattern_result: Optional[Dict[str, Any]] = None
    if target == "dimension":
        methods_used.append("pattern_repair")
        trace.append("Checked area consistency")
        pattern_result = suggest_pattern_repair(
            deepcopy(working_row),
            diagnostics=deepcopy(working_diagnostics),
            nearby_rows=working_nearby_rows,
            order_context=working_context,
        )
        for step in pattern_result.get("trace") or []:
            if step not in trace:
                trace.append(str(step))
        if pattern_result.get("success") and float(pattern_result.get("confidence") or 0.0) >= PATTERN_REPAIR_CONFIDENCE_THRESHOLD:
            trace.append("OCR fallback skipped")
            return _response(
                success=True,
                target_field="dimension",
                original_value=pattern_result.get("original_value"),
                suggested_value=pattern_result.get("suggested_value"),
                confidence=float(pattern_result.get("confidence") or 0.0),
                recommended_action=pattern_result.get("recommended_action") or "ACCEPT_PATTERN_SUGGESTION_OR_REVIEW",
                reasoning=pattern_result.get("reasoning") or "Pattern repair found a supported correction.",
                evidence={
                    "diagnostic_codes": codes,
                    "diagnosis": diagnosis,
                    "family_pattern": (family_result or {}).get("evidence") or {},
                    "pattern_repair": pattern_result.get("evidence") or {},
                },
                trace=trace,
                methods_used=methods_used,
            )
        trace.append(
            f"Pattern repair confidence {float((pattern_result or {}).get('confidence') or 0.0):.2f} below threshold"
        )

    if _has_row_location(working_row):
        methods_used.append("ocr_fallback")
        trace.append("Attempting OCR fallback using row_location")
        ocr_result = ocr_fallback_row_repair(
            row=deepcopy(working_row),
            diagnostics=deepcopy(working_diagnostics),
            target_field=target,
            order_context=working_context,
            row_index=row_index,
            pdf_id=pdf_id or working_pdf_context.get("pdf_id"),
        )
        if ocr_result.get("success"):
            trace.append(f"OCR fallback confidence {float(ocr_result.get('confidence') or 0.0):.2f}")
            return _response(
                success=True,
                target_field=ocr_result.get("target_field") or target,
                original_value=ocr_result.get("original_value"),
                suggested_value=ocr_result.get("suggested_value"),
                confidence=float(ocr_result.get("confidence") or 0.0),
                recommended_action="ACCEPT_OCR_SUGGESTION_OR_REVIEW",
                reasoning=ocr_result.get("reason") or "OCR fallback found a supported correction.",
                evidence={
                    "diagnostic_codes": codes,
                    "diagnosis": diagnosis,
                    "family_pattern": (family_result or {}).get("evidence") or {},
                    "pattern_repair": (pattern_result or {}).get("evidence") or {},
                    "ocr_fallback": ocr_result.get("evidence") or {},
                },
                trace=trace,
                methods_used=methods_used,
            )
        trace.append("OCR fallback did not find a confident correction")
    else:
        trace.append("OCR fallback skipped because row_location is unavailable")

    return _response(
        success=False,
        target_field=target,
        original_value=original,
        suggested_value=None,
        confidence=max(
            float((family_result or {}).get("confidence") or 0.0),
            float((pattern_result or {}).get("confidence") or 0.0),
        ),
        recommended_action="MANUAL_REVIEW",
        reasoning="No repair method produced a reliable supported suggestion.",
        evidence={
            "diagnostic_codes": codes,
            "diagnosis": diagnosis,
            "family_pattern": (family_result or {}).get("evidence") or {},
            "pattern_repair": (pattern_result or {}).get("evidence") or {},
        },
        trace=trace,
        methods_used=methods_used,
    )


__all__ = ["repair_suspicious_row"]
