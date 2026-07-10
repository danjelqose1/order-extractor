from __future__ import annotations

import base64
from difflib import SequenceMatcher
import json
import os
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, ValidationError


DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_LONG_EDGE = 1600
MAX_SOURCE_PIXELS = 60_000_000
JPEG_IMAGE_FORMATS = {"JPEG", "MPO"}
SUPPORTED_IMAGE_FORMATS = {
    "JPEG": "image/jpeg",
    # iPhone photos with auxiliary frames can be detected by Pillow as MPO.
    # Flatten the primary frame to a standard JPEG before sending it onward.
    "MPO": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
TRANSIENT_OPENAI_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


def _env_bool(name: str, default: bool = False) -> bool:
    fallback = "true" if default else "false"
    return os.getenv(name, fallback).strip().lower() in {"1", "true", "yes", "on"}


def photo_assist_enabled() -> bool:
    return _env_bool("MANUAL_PHOTO_ASSIST_ENABLED", False)


def photo_assist_model() -> str:
    return os.getenv("MANUAL_PHOTO_ASSIST_REASONING_MODEL") or "gpt-5.6-terra"


def photo_assist_observation_model() -> str:
    return os.getenv("MANUAL_PHOTO_ASSIST_OBSERVATION_MODEL") or "gpt-5.6-luna"


def photo_assist_fallback_model() -> str:
    return os.getenv("MANUAL_PHOTO_ASSIST_FALLBACK_MODEL") or "gpt-5.4-mini"


def photo_assist_max_upload_bytes() -> int:
    try:
        value = int(os.getenv("MANUAL_PHOTO_ASSIST_MAX_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES)))
    except ValueError:
        return DEFAULT_MAX_UPLOAD_BYTES
    return value if value > 0 else DEFAULT_MAX_UPLOAD_BYTES


def photo_assist_timeout_seconds() -> float:
    try:
        value = float(os.getenv("MANUAL_PHOTO_ASSIST_TIMEOUT_SECONDS", "120"))
    except ValueError:
        return 120.0
    return value if value > 0 else 120.0


class PhotoAssistError(RuntimeError):
    category = "failed"


class UnsupportedImageError(PhotoAssistError):
    category = "unsupported_image"


class InvalidImageError(PhotoAssistError):
    category = "invalid_image"


class PhotoAssistTimeoutError(PhotoAssistError):
    category = "timeout"


class PhotoAssistUnavailableError(PhotoAssistError):
    category = "unavailable"


class InvalidExtractionError(PhotoAssistError):
    category = "invalid_response"


class NoOrderRowsError(PhotoAssistError):
    category = "no_rows"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


RawScalar = Optional[Union[str, int, float]]


class TextDetection(StrictModel):
    raw: Optional[str]
    value: Optional[str]
    warning: Optional[str]


class IndexDetection(StrictModel):
    raw: Optional[str]
    value: Optional[int]
    warning: Optional[str]


class QuantityDetection(StrictModel):
    raw: RawScalar
    value: Optional[int]
    warning: Optional[str]


class DimensionDetection(StrictModel):
    raw: RawScalar
    value: Optional[float]
    unit: Literal["mm", "cm", "unknown"]
    normalized_mm: Optional[float]
    warning: Optional[str]


class NotesDetection(StrictModel):
    raw: Optional[str]
    value: Optional[str]


class PhotoAssistDocument(StrictModel):
    client_name: TextDetection
    order_number: TextDetection
    glass_type: TextDetection


class PhotoAssistRow(StrictModel):
    source_line: Optional[str]
    section: Optional[str]
    client_reference: TextDetection
    index_number: IndexDetection
    width: DimensionDetection
    height: DimensionDetection
    quantity: QuantityDetection
    glass_type: TextDetection
    notes: NotesDetection
    warnings: List[str]


class PhotoAssistAIResult(StrictModel):
    document: PhotoAssistDocument
    rows: List[PhotoAssistRow]
    global_warnings: List[str]
    raw_detected_text: Optional[str]


class PhotoAssistResponse(PhotoAssistAIResult):
    status: Literal["ready", "needs_review", "failed"]
    request_id: str
    model: str


class PhotoAssistVisualSpan(StrictModel):
    text: str
    color: Literal["black", "blue", "red", "green", "other", "unknown"]
    uncertain: bool


class PhotoAssistVisualLine(StrictModel):
    raw_text: str
    region: Literal["header", "body", "footer", "unknown"]
    alignment: Literal["left", "indented", "center", "right", "full", "unknown"]
    spans: List[PhotoAssistVisualSpan]
    separator_before: bool
    separator_after: bool
    uncertain_fragments: List[str]


class PhotoAssistObservation(StrictModel):
    page_summary: str
    lines: List[PhotoAssistVisualLine]
    visual_warnings: List[str]


@dataclass(frozen=True)
class NormalizedImage:
    content: bytes
    mime_type: str
    width: int
    height: int
    original_format: str


OBSERVATION_PROMPT = """
Create a neutral visual transcription of this handwritten page before interpreting its business meaning.

Read from top to bottom. Preserve every meaningful line, punctuation mark, number, decimal, fraction, arrow, parenthesis, and separator. Split a line into spans whenever ink color changes so later processing can distinguish parallel numbering systems. Record alignment, page region, horizontal separator lines, and uncertain fragments. Do not decide which text is a client, glass type, section, position, index, dimension, quantity, or note. Do not normalize spelling or units. Do not omit headings or footer text.

Return only data matching the required JSON schema.
""".strip()


EXTRACTION_PROMPT = """
You are an experienced glass-factory order clerk. Convert the supplied neutral visual transcription and photograph into Manual Order fields by understanding the document as a whole.

Privately infer the document's layout grammar from repeated spatial patterns, color changes, separators, headings, row shapes, and surrounding context. The layout is not fixed: it may contain one or many material groups, optional sections, one or two numbering systems, notes, annotations, and header or footer metadata. Determine roles from consistency across the page rather than from any single hard-coded phrase or position.

Use these domain constraints:
- A production row requires a plausible width and height. Context-only headings and annotations are not rows.
- Quantity may be absent or unreadable. Return that production row with quantity null and a focused warning instead of discarding it.
- A material heading governs subsequent rows until the next material heading. Preserve complete material constructions, while keeping non-material annotations separate.
- A section label governs subsequent rows within its group until another section or group begins.
- Parallel columns or ink colors can represent client positions and internal indexes. Keep them separate; do not invent either when absent.
- A single same-color sequential number at the start of each line is ordinary row ordering, not an internal red index. Leave client_reference and index_number null unless a separate role is supported by color, column, or repeated-position evidence.
- Arithmetic separators are syntax. Meaningful trailing words are notes.
- Dimensions use the selected Manual Orders unit unless the page explicitly states another unit or the values make that interpretation implausible.
- Header or footer metadata can contain a client name or order number. Use page isolation, vocabulary similarity, and document structure as evidence.
- Saved client and glass-type vocabularies are soft context, not allowed-value lists. Correct only close visible matches and allow new values.
- Preserve source_line and raw fields. If a digit or role remains genuinely ambiguous after considering the full page, retain the raw text, use null where necessary, and add a focused warning.
- Do not calculate normalized_mm; the backend does that deterministically.

Return only final production rows and document metadata matching the required JSON schema. Do not include context-only lines as rows. Do not return markdown or hidden reasoning.
""".strip()


def _model_instructions(
    preferred_dimension_unit: Optional[str],
    *,
    known_glass_types: Optional[List[str]] = None,
    known_clients: Optional[List[str]] = None,
) -> str:
    unit = preferred_dimension_unit if preferred_dimension_unit in {"mm", "cm"} else None
    context_lines = []
    if unit:
        label = "centimetres" if unit == "cm" else "millimetres"
        context_lines.append(
            f"Manual Orders is currently set to {label} ({unit}). Treat ordinary handwritten "
            f"dimensions as {unit} unless the image explicitly indicates a different unit or the "
            "values are implausible for architectural glass."
        )
    glass_types = [str(value).strip() for value in known_glass_types or [] if str(value).strip()]
    clients = [str(value).strip() for value in known_clients or [] if str(value).strip()]
    if glass_types:
        context_lines.append(
            "Known glass types (reference vocabulary, not an exhaustive list): "
            + json.dumps(glass_types[:100], ensure_ascii=False)
        )
    if clients:
        context_lines.append(
            "Known clients (reference vocabulary, not an exhaustive list): "
            + json.dumps(clients[:100], ensure_ascii=False)
        )
    return EXTRACTION_PROMPT + ("\n\n" + "\n".join(context_lines) if context_lines else "")


def _looks_like_heic(content: bytes) -> bool:
    header = content[:32].lower()
    return b"ftypheic" in header or b"ftypheif" in header or b"ftypheix" in header or b"ftypmif1" in header


def normalize_uploaded_image(content: bytes) -> NormalizedImage:
    if not content:
        raise InvalidImageError("Image could not be read.")
    if _looks_like_heic(content):
        raise UnsupportedImageError("HEIC/HEIF is not available in this beta. Upload JPG or PNG instead.")

    try:
        with Image.open(BytesIO(content)) as opened:
            actual_format = str(opened.format or "").upper()
            if actual_format not in SUPPORTED_IMAGE_FORMATS:
                raise UnsupportedImageError("Unsupported image format. Upload JPG, PNG, or WebP.")
            if opened.width * opened.height > MAX_SOURCE_PIXELS:
                raise InvalidImageError("Image dimensions are too large. Use a smaller photo.")

            # JPEG draft mode asks the decoder to downsample before allocating the
            # full pixel buffer. This keeps large phone photos within Render's
            # memory limit without resizing small handwriting into unreadability.
            if actual_format in JPEG_IMAGE_FORMATS:
                opened.draft("RGB", (MAX_IMAGE_LONG_EDGE, MAX_IMAGE_LONG_EDGE))
            opened.load()
            image = ImageOps.exif_transpose(opened)
            if image.width < 1 or image.height < 1:
                raise InvalidImageError("Image could not be read.")

            if max(image.size) > MAX_IMAGE_LONG_EDGE:
                image.thumbnail(
                    (MAX_IMAGE_LONG_EDGE, MAX_IMAGE_LONG_EDGE),
                    Image.Resampling.LANCZOS,
                )

            output = BytesIO()
            if actual_format in JPEG_IMAGE_FORMATS:
                normalized = image.convert("RGB")
                normalized.save(output, format="JPEG", quality=92, subsampling=2)
                mime_type = "image/jpeg"
            else:
                normalized = image.convert("RGBA" if "A" in image.getbands() else "RGB")
                normalized.save(output, format="PNG", compress_level=3)
                mime_type = "image/png"
            return NormalizedImage(
                content=output.getvalue(),
                mime_type=mime_type,
                width=normalized.width,
                height=normalized.height,
                original_format=actual_format,
            )
    except UnsupportedImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("Image could not be read.") from exc


def _append_warning(existing: Optional[str], message: str) -> str:
    existing_text = str(existing or "").strip()
    if not existing_text:
        return message
    if message.lower() in existing_text.lower():
        return existing_text
    return f"{existing_text} {message}"


def _row_has_meaning(row: PhotoAssistRow) -> bool:
    values = [
        row.source_line,
        row.section,
        row.client_reference.raw,
        row.client_reference.value,
        row.index_number.raw,
        row.index_number.value,
        row.width.raw,
        row.width.value,
        row.height.raw,
        row.height.value,
        row.quantity.raw,
        row.quantity.value,
        row.glass_type.raw,
        row.glass_type.value,
        row.notes.raw,
        row.notes.value,
    ]
    return any(value is not None and str(value).strip() for value in values)


def _normalize_dimension(dimension: DimensionDetection, label: str, row_warnings: List[str]) -> None:
    value = dimension.value
    if value is not None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
    if value is None or value <= 0:
        dimension.value = None
        dimension.normalized_mm = None
        message = f"{label} is missing or unclear."
        dimension.warning = _append_warning(dimension.warning, message)
        row_warnings.append(message)
        return

    dimension.value = value
    if dimension.unit == "cm":
        dimension.normalized_mm = round(value * 10, 3)
    elif dimension.unit == "mm":
        dimension.normalized_mm = round(value, 3)
    else:
        dimension.normalized_mm = None
        message = "Dimensions may be written in centimetres. Confirm before applying."
        dimension.warning = _append_warning(dimension.warning, message)
        row_warnings.append(message)
        return

    if dimension.normalized_mm < 100:
        message = f"{label} is unusually small. Confirm the value."
        dimension.warning = _append_warning(dimension.warning, message)
        row_warnings.append(message)
    elif dimension.normalized_mm > 6000:
        message = f"{label} is unusually large. Confirm the value."
        dimension.warning = _append_warning(dimension.warning, message)
        row_warnings.append(message)


def _is_heading_only_row(row: PhotoAssistRow) -> bool:
    source = str(row.source_line or "").lower()
    has_dimension_syntax = " x " in source or "×" in source
    has_dimensions = row.width.value is not None or row.height.value is not None
    has_quantity = row.quantity.value is not None
    return bool(row.glass_type.value and not has_dimension_syntax and not has_dimensions and not has_quantity)


def _heading_glass_value(source: str, row_glass: str) -> str:
    source_without_thickness = re.sub(
        r"\(\s*\d+(?:[.,]\d+)?\s*mm\s*\)",
        "",
        source,
        flags=re.IGNORECASE,
    ).strip()
    if row_glass and row_glass.lower() in source_without_thickness.lower():
        return source_without_thickness
    return row_glass or source_without_thickness


def _is_unit_uncertainty(message: Optional[str]) -> bool:
    text = str(message or "").lower()
    return "unit" in text or "centimet" in text or "millimet" in text


def _vocabulary_key(value: str) -> str:
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def _closest_vocabulary_match(value: str, candidates: Optional[List[str]]) -> Optional[str]:
    source_key = _vocabulary_key(value)
    if not source_key:
        return None
    scored = [
        (SequenceMatcher(None, source_key, _vocabulary_key(candidate)).ratio(), candidate)
        for candidate in candidates or []
        if _vocabulary_key(candidate)
    ]
    if not scored:
        return None
    score, candidate = max(scored, key=lambda item: item[0])
    return str(candidate).strip() if score >= 0.78 else None


def _production_line_match(value: str) -> Optional[re.Match[str]]:
    return re.search(
        r"(?<![/\d])(?P<width>\d+(?:[.,]\d+)?)\s*(?:x|×|\+)\s*"
        r"(?P<height>\d+(?:[.,]\d+)?)"
        r"(?:\s*(?:x|×|\+|=)\s*(?P<quantity>\d+)(?![.,]\d))?",
        str(value or ""),
        flags=re.IGNORECASE,
    )


def _span_integer(span: PhotoAssistVisualSpan) -> Optional[int]:
    match = re.search(r"(?<![\d.,])\d+(?![\d.,])", span.text)
    return int(match.group(0)) if match else None


def _observation_row_contexts(
    observation: PhotoAssistObservation,
    known_glass_types: Optional[List[str]],
) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    active_glass: Optional[str] = None
    section_candidates: List[str] = []

    for line in observation.lines:
        text = str(line.raw_text or "").strip()
        if line.separator_before:
            active_glass = None
            section_candidates = []

        production_match = _production_line_match(text)
        if production_match:
            red_span_index = next(
                (
                    index
                    for index, span in enumerate(line.spans)
                    if span.color == "red" and _span_integer(span) is not None
                ),
                None,
            )
            red_index = (
                _span_integer(line.spans[red_span_index])
                if red_span_index is not None
                else None
            )
            client_position = None
            if red_span_index is not None:
                prefix_numbers = [
                    number
                    for span in line.spans[:red_span_index]
                    if span.color != "red"
                    for number in re.findall(r"(?<![\d.,])\d+(?![\d.,])", span.text)
                ]
                if prefix_numbers:
                    client_position = prefix_numbers[-1]
            contexts.append(
                {
                    "raw_text": text,
                    "width": float(production_match.group("width").replace(",", ".")),
                    "height": float(production_match.group("height").replace(",", ".")),
                    "quantity": int(production_match.group("quantity")) if production_match.group("quantity") else None,
                    "client_position": client_position,
                    "red_index": red_index,
                    "glass_type": active_glass,
                    "section_candidates": list(section_candidates),
                    "notes": text[production_match.end():].strip(" \t-=→>"),
                }
            )
        elif text:
            cleaned_heading = _heading_glass_value(text, "")
            vocabulary_match = _closest_vocabulary_match(cleaned_heading, known_glass_types)
            looks_like_material = bool(
                vocabulary_match
                or (
                    not _production_line_match(text)
                    and (
                        text.count("+") >= 2
                        or bool(re.search(r"\d+\s*/\s*\d+", text))
                    )
                )
            )
            if looks_like_material:
                active_glass = vocabulary_match or cleaned_heading
                section_candidates = []
            elif active_glass and re.search(r"[A-Za-zÀ-ÿ]", text) and not re.fullmatch(r"\([^)]*\)", text):
                section_candidates.append(text)

        if line.separator_after:
            active_glass = None
            section_candidates = []

    return contexts


def _select_section_candidate(current: Optional[str], candidates: List[str]) -> Optional[str]:
    cleaned = [str(candidate).strip() for candidate in candidates if str(candidate).strip()]
    if not cleaned:
        return current
    current_key = _vocabulary_key(str(current or ""))
    if not current_key:
        return cleaned[-1]
    score, candidate = max(
        (
            SequenceMatcher(None, current_key, _vocabulary_key(item)).ratio(),
            item,
        )
        for item in cleaned
    )
    return candidate if score >= 0.45 else current


def _is_non_actionable_extraction_warning(message: str) -> bool:
    text = str(message or "").casefold()
    return any(
        phrase in text
        for phrase in (
            "glass type from page header applies",
            "source line numbering appears inconsistent",
            "leading measurement line appears",
            "page shows two grouped sections",
            "trailing '->",
        )
    )


def _repair_rows_from_observation(
    result: PhotoAssistAIResult,
    observation: Optional[PhotoAssistObservation],
    known_glass_types: Optional[List[str]],
) -> None:
    if observation is None:
        return
    contexts = _observation_row_contexts(observation, known_glass_types)
    if not contexts:
        return

    def context_matches_row(context: Dict[str, Any], row: PhotoAssistRow) -> bool:
        try:
            return (
                abs(float(row.width.value or 0) - context["width"]) < 0.01
                and abs(float(row.height.value or 0) - context["height"]) < 0.01
            )
        except (TypeError, ValueError):
            return False

    def new_row(context: Dict[str, Any]) -> PhotoAssistRow:
        return PhotoAssistRow(
            source_line=context["raw_text"],
            section=None,
            client_reference=TextDetection(raw=None, value=None, warning=None),
            index_number=IndexDetection(raw=None, value=None, warning=None),
            width=DimensionDetection(raw=str(context["width"]), value=context["width"], unit="unknown", normalized_mm=None, warning=None),
            height=DimensionDetection(raw=str(context["height"]), value=context["height"], unit="unknown", normalized_mm=None, warning=None),
            quantity=QuantityDetection(raw=context["quantity"], value=context["quantity"], warning=None),
            glass_type=TextDetection(raw=None, value=None, warning=None),
            notes=NotesDetection(raw=context["notes"] or None, value=context["notes"] or None),
            warnings=[],
        )

    predicted_rows = list(result.rows)
    repaired_rows: List[PhotoAssistRow] = []
    predicted_index = 0
    for context_index, context in enumerate(contexts):
        row = None
        if predicted_index < len(predicted_rows):
            current_row = predicted_rows[predicted_index]
            if context_matches_row(context, current_row):
                row = current_row
                predicted_index += 1
            elif any(
                context_matches_row(future_context, current_row)
                for future_context in contexts[context_index + 1:]
            ):
                # The reasoning pass skipped this observed line. Keep its next row
                # available for the matching visual line and synthesize this one.
                row = new_row(context)
            else:
                matching_prediction = next(
                    (
                        candidate_index
                        for candidate_index in range(predicted_index + 1, len(predicted_rows))
                        if context_matches_row(context, predicted_rows[candidate_index])
                    ),
                    None,
                )
                if matching_prediction is not None:
                    row = predicted_rows[matching_prediction]
                    predicted_index = matching_prediction + 1
                else:
                    # The reasoning pass interpreted this line differently. Reuse
                    # its semantic fields, but let the visual pass replace numbers.
                    row = current_row
                    predicted_index += 1
        if row is None:
            row = new_row(context)

        row.source_line = context["raw_text"]
        if context["client_position"] is not None:
            row.client_reference.raw = context["client_position"]
            row.client_reference.value = context["client_position"]
            row.client_reference.warning = None
        if context["red_index"] is not None:
            row.index_number.raw = str(context["red_index"])
            row.index_number.value = context["red_index"]
            row.index_number.warning = None
        if context["glass_type"]:
            row.glass_type.raw = context["glass_type"]
            row.glass_type.value = context["glass_type"]
            if _closest_vocabulary_match(context["glass_type"], known_glass_types):
                row.glass_type.warning = None
        row.width.raw = str(context["width"])
        row.width.value = context["width"]
        row.height.raw = str(context["height"])
        row.height.value = context["height"]
        row.quantity.raw = context["quantity"]
        row.quantity.value = context["quantity"]
        if context["notes"]:
            row.notes.raw = context["notes"]
            row.notes.value = context["notes"]
        row.section = _select_section_candidate(row.section, context["section_candidates"])
        row.warnings = [
            warning
            for warning in row.warnings
            if not _is_non_actionable_extraction_warning(warning)
        ]
        repaired_rows.append(row)

    result.rows = repaired_rows

    has_red_index_evidence = any(context["red_index"] is not None for context in contexts)
    has_sections = any(str(row.section or "").strip() for row in result.rows)
    if not has_red_index_evidence and not has_sections:
        for row in result.rows:
            row.client_reference = TextDetection(raw=None, value=None, warning=None)
            row.index_number = IndexDetection(raw=None, value=None, warning=None)

    first_glass = next((row.glass_type.value for row in result.rows if row.glass_type.value), None)
    if first_glass:
        result.document.glass_type.value = first_glass
        result.document.glass_type.raw = first_glass
        if _closest_vocabulary_match(first_glass, known_glass_types):
            result.document.glass_type.warning = None
    result.global_warnings = [
        warning
        for warning in result.global_warnings
        if not _is_non_actionable_extraction_warning(warning)
    ]


def normalize_extraction(
    ai_result: PhotoAssistAIResult,
    *,
    preferred_dimension_unit: Optional[str] = None,
    observation: Optional[PhotoAssistObservation] = None,
    known_glass_types: Optional[List[str]] = None,
) -> PhotoAssistAIResult:
    result = ai_result.model_copy(deep=True)
    result.rows = [row for row in result.rows if _row_has_meaning(row)]
    active_glass = str(result.document.glass_type.value or "").strip() or None
    active_section: Optional[str] = None
    first_heading = True
    data_rows = []
    for row in result.rows:
        if _is_heading_only_row(row):
            source = str(row.source_line or "").strip()
            row_glass = str(row.glass_type.value or "").strip()
            active_glass = _heading_glass_value(source, row_glass)
            if first_heading and active_glass:
                result.document.glass_type.raw = source or row.glass_type.raw
                result.document.glass_type.value = active_glass
                first_heading = False
            active_section = None
            continue
        if not row.glass_type.value and active_glass:
            row.glass_type.value = active_glass
            if row.glass_type.raw is None:
                row.glass_type.raw = active_glass
        elif row.glass_type.value:
            active_glass = row.glass_type.value
        if row.section:
            active_section = row.section
        elif active_section:
            row.section = active_section
        data_rows.append(row)
    result.rows = data_rows
    _repair_rows_from_observation(result, observation, known_glass_types)
    if not result.rows:
        raise NoOrderRowsError("No order rows were detected.")

    preferred_unit = preferred_dimension_unit if preferred_dimension_unit in {"mm", "cm"} else None
    detected_units = set()
    index_rows: Dict[int, List[int]] = {}
    inherited_glass = result.document.glass_type.value

    for row_number, row in enumerate(result.rows, start=1):
        row.warnings = [str(item).strip() for item in row.warnings if str(item).strip()]
        if preferred_unit:
            for dimension in (row.width, row.height):
                if dimension.value is not None and dimension.unit == "unknown":
                    dimension.unit = preferred_unit
                    if _is_unit_uncertainty(dimension.warning):
                        dimension.warning = None
            row.warnings = [warning for warning in row.warnings if not _is_unit_uncertainty(warning)]

        note_value = str(row.notes.value or "").strip()
        quantity_text = str(row.quantity.value or row.quantity.raw or "").strip()
        source_text = str(row.source_line or "").strip()
        trailing_operator_note = bool(
            note_value
            and re.search(
                rf"(?:-|->|→)\s*{re.escape(note_value)}\s*$",
                source_text,
                flags=re.IGNORECASE,
            )
        )
        if (note_value in {"=", "x", "×", "+", "->", "→"} and not trailing_operator_note) or (
            note_value and quantity_text and note_value == quantity_text and "x" in str(row.source_line or "").lower()
        ):
            row.notes.value = None
        _normalize_dimension(row.width, "Width", row.warnings)
        _normalize_dimension(row.height, "Height", row.warnings)
        for dimension in (row.width, row.height):
            if dimension.value is not None and dimension.unit in {"mm", "cm"}:
                detected_units.add(dimension.unit)

        quantity = row.quantity.value
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            row.quantity.value = None
            message = "Quantity not detected or invalid."
            row.quantity.warning = _append_warning(row.quantity.warning, message)
            row.warnings.append(message)

        index_number = row.index_number.value
        if index_number is not None:
            if not isinstance(index_number, int) or isinstance(index_number, bool) or index_number <= 0:
                row.index_number.value = None
                message = "Index number is invalid or unclear."
                row.index_number.warning = _append_warning(row.index_number.warning, message)
                row.warnings.append(message)
            else:
                index_rows.setdefault(index_number, []).append(row_number)

        if not row.glass_type.value and inherited_glass:
            row.glass_type.value = inherited_glass
            if row.glass_type.raw is None:
                row.glass_type.raw = result.document.glass_type.raw
        elif row.glass_type.value:
            inherited_glass = row.glass_type.value

        if not row.glass_type.value:
            message = "Glass type could not be read."
            row.glass_type.warning = _append_warning(row.glass_type.warning, message)
            row.warnings.append(message)

        row.warnings = list(dict.fromkeys(row.warnings))

    for index_number, row_numbers in index_rows.items():
        if len(row_numbers) < 2:
            continue
        message = f"Duplicate index number {index_number}."
        for row_number in row_numbers:
            row = result.rows[row_number - 1]
            row.index_number.warning = _append_warning(row.index_number.warning, message)
            if message not in row.warnings:
                row.warnings.append(message)

    if len(detected_units) > 1:
        message = "Inconsistent dimension units were detected. Confirm every row before applying."
        if message not in result.global_warnings:
            result.global_warnings.append(message)

    global_warnings = [str(item).strip() for item in result.global_warnings if str(item).strip()]
    if preferred_unit:
        global_warnings = [warning for warning in global_warnings if not _is_unit_uncertainty(warning)]
    result.global_warnings = list(dict.fromkeys(global_warnings))
    return result


def _result_has_warnings(result: PhotoAssistAIResult) -> bool:
    if result.global_warnings:
        return True
    document_fields = (
        result.document.client_name,
        result.document.order_number,
        result.document.glass_type,
    )
    if any(field.warning for field in document_fields):
        return True
    for row in result.rows:
        if row.warnings:
            return True
        if any(
            field.warning
            for field in (
                row.client_reference,
                row.index_number,
                row.width,
                row.height,
                row.quantity,
                row.glass_type,
            )
        ):
            return True
    return False


def _response_schema() -> Dict[str, Any]:
    return PhotoAssistAIResult.model_json_schema()


def _observation_schema() -> Dict[str, Any]:
    return PhotoAssistObservation.model_json_schema()


def _default_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise PhotoAssistUnavailableError("Photo Assist is temporarily unavailable.")
    return OpenAI(api_key=api_key)


def _image_data_url(image: NormalizedImage) -> str:
    encoded = base64.b64encode(image.content).decode("ascii")
    return f"data:{image.mime_type};base64,{encoded}"


def _is_model_unavailable_error(exc: Exception) -> bool:
    if isinstance(exc, (NotFoundError, PermissionDeniedError)):
        return True
    if not isinstance(exc, (BadRequestError, RuntimeError)):
        return False
    message = str(exc).lower()
    return "model" in message and any(
        phrase in message
        for phrase in (
            "not found",
            "does not exist",
            "do not have access",
            "not have access",
            "access denied",
            "not supported",
            "unsupported",
        )
    )


def _observe_image_once(
    image: NormalizedImage,
    *,
    model: str,
    client: Any,
) -> PhotoAssistObservation:
    reasoning_options = {"reasoning": {"effort": "low"}} if model.startswith("gpt-5.6") else {}
    response = client.responses.create(
        model=model,
        **reasoning_options,
        instructions=OBSERVATION_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Transcribe the visible page structure without assigning business roles.",
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(image),
                        "detail": "high",
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "manual_photo_assist_observation",
                "schema": _observation_schema(),
                "strict": True,
            }
        },
        max_output_tokens=10000,
        store=False,
        timeout=httpx.Timeout(photo_assist_timeout_seconds(), connect=20.0),
    )
    output_text = str(getattr(response, "output_text", "") or "").strip()
    if not output_text:
        raise InvalidExtractionError("The visual transcription could not be validated.")
    try:
        return PhotoAssistObservation.model_validate(json.loads(output_text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise InvalidExtractionError("The visual transcription could not be validated.") from exc


def _call_model_once(
    image: NormalizedImage,
    *,
    model: str,
    client: Any,
    preferred_dimension_unit: Optional[str] = None,
    known_glass_types: Optional[List[str]] = None,
    known_clients: Optional[List[str]] = None,
    observation: Optional[PhotoAssistObservation] = None,
) -> PhotoAssistAIResult:
    observation_text = (
        observation.model_dump_json(indent=2)
        if observation is not None
        else '{"page_summary":"No separate observation pass was supplied.","lines":[],"visual_warnings":[]}'
    )
    reasoning_options = {"reasoning": {"effort": "medium"}} if model.startswith("gpt-5.6") else {}
    response = client.responses.create(
        model=model,
        **reasoning_options,
        instructions=_model_instructions(
            preferred_dimension_unit,
            known_glass_types=known_glass_types,
            known_clients=known_clients,
        ),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Interpret this handwritten glass order for user review. Use the photograph "
                            "to verify the neutral visual transcription below.\n\n"
                            f"VISUAL TRANSCRIPTION JSON:\n{observation_text}"
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(image),
                        # openai==2.26.0 accepts auto, low, or high; original is unsupported.
                        "detail": "high",
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "manual_photo_assist_extraction",
                "schema": _response_schema(),
                "strict": True,
            }
        },
        max_output_tokens=12000,
        store=False,
        timeout=httpx.Timeout(photo_assist_timeout_seconds(), connect=20.0),
    )
    output_text = str(getattr(response, "output_text", "") or "").strip()
    if not output_text:
        raise InvalidExtractionError("The AI response could not be validated.")
    try:
        return PhotoAssistAIResult.model_validate(json.loads(output_text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise InvalidExtractionError("The AI response could not be validated.") from exc


def extract_photo_assist(
    image: NormalizedImage,
    *,
    request_id: str,
    client: Any = None,
    model: Optional[str] = None,
    observation_model: Optional[str] = None,
    preferred_dimension_unit: Optional[str] = None,
    known_glass_types: Optional[List[str]] = None,
    known_clients: Optional[List[str]] = None,
    call_once: Optional[Callable[..., PhotoAssistAIResult]] = None,
    observe_once: Optional[Callable[..., PhotoAssistObservation]] = None,
    fallback_model: Optional[str] = None,
) -> PhotoAssistResponse:
    selected_model = model or photo_assist_model()
    selected_observation_model = observation_model or photo_assist_observation_model()
    selected_fallback_model = fallback_model or photo_assist_fallback_model()
    active_model = selected_model
    active_observation_model = selected_observation_model
    openai_client = client or _default_client()
    caller = call_once or _call_model_once
    observer = observe_once or _observe_image_once
    use_observation_pass = call_once is None or observe_once is not None
    observation: Optional[PhotoAssistObservation] = None
    last_error: Optional[Exception] = None

    for attempt in range(2):
        try:
            if use_observation_pass and observation is None:
                try:
                    observation = observer(image, model=active_observation_model, client=openai_client)
                except Exception as exc:
                    if not call_once and _is_model_unavailable_error(exc) and active_observation_model != selected_fallback_model:
                        active_observation_model = selected_fallback_model
                        observation = observer(image, model=active_observation_model, client=openai_client)
                    else:
                        raise
            if call_once and not use_observation_pass:
                extracted = caller(image, model=active_model, client=openai_client)
            else:
                try:
                    extracted = caller(
                        image,
                        model=active_model,
                        client=openai_client,
                        preferred_dimension_unit=preferred_dimension_unit,
                        known_glass_types=known_glass_types,
                        known_clients=known_clients,
                        observation=observation,
                    )
                except Exception as exc:
                    if not call_once and _is_model_unavailable_error(exc) and active_model != selected_fallback_model:
                        active_model = selected_fallback_model
                        extracted = caller(
                            image,
                            model=active_model,
                            client=openai_client,
                            preferred_dimension_unit=preferred_dimension_unit,
                            known_glass_types=known_glass_types,
                            known_clients=known_clients,
                            observation=observation,
                        )
                    else:
                        raise
            normalized = normalize_extraction(
                extracted,
                preferred_dimension_unit=preferred_dimension_unit,
                observation=observation,
                known_glass_types=known_glass_types,
            )
            if active_observation_model != selected_observation_model:
                normalized.global_warnings.append(
                    f"Visual transcription used fallback model {active_observation_model}; "
                    f"{selected_observation_model} was unavailable."
                )
            if active_model != selected_model:
                normalized.global_warnings.append(
                    f"Contextual interpretation used fallback model {active_model}; "
                    f"{selected_model} was unavailable."
                )
            normalized.global_warnings = list(dict.fromkeys(normalized.global_warnings))
            return PhotoAssistResponse(
                **normalized.model_dump(),
                status="needs_review" if _result_has_warnings(normalized) else "ready",
                request_id=request_id,
                model=(
                    f"{active_observation_model} -> {active_model}"
                    if use_observation_pass
                    else active_model
                ),
            )
        except NoOrderRowsError:
            raise
        except APITimeoutError as exc:
            raise PhotoAssistTimeoutError("Extraction timed out. Try again.") from exc
        except (InvalidExtractionError, *TRANSIENT_OPENAI_ERRORS) as exc:
            last_error = exc
            if attempt == 0:
                continue
            break
        except Exception as exc:
            last_error = exc
            break

    if isinstance(last_error, APITimeoutError):
        raise PhotoAssistTimeoutError("Extraction timed out. Try again.") from last_error
    if isinstance(last_error, InvalidExtractionError):
        raise last_error
    raise PhotoAssistUnavailableError("Photo Assist is temporarily unavailable. Try again.") from last_error
