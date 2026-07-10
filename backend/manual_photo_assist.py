from __future__ import annotations

import base64
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
    InternalServerError,
    OpenAI,
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
- A production row requires a width, height, and positive quantity. Context-only headings and annotations are not rows.
- A material heading governs subsequent rows until the next material heading. Preserve complete material constructions, while keeping non-material annotations separate.
- A section label governs subsequent rows within its group until another section or group begins.
- Parallel columns or ink colors can represent client positions and internal indexes. Keep them separate; do not invent either when absent.
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


def normalize_extraction(
    ai_result: PhotoAssistAIResult,
    *,
    preferred_dimension_unit: Optional[str] = None,
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
        if note_value in {"=", "x", "×", "+", "->", "→"} or (
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


def _observe_image_once(
    image: NormalizedImage,
    *,
    model: str,
    client: Any,
) -> PhotoAssistObservation:
    response = client.responses.create(
        model=model,
        reasoning={"effort": "low"},
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
    response = client.responses.create(
        model=model,
        reasoning={"effort": "medium"},
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
) -> PhotoAssistResponse:
    selected_model = model or photo_assist_model()
    selected_observation_model = observation_model or photo_assist_observation_model()
    openai_client = client or _default_client()
    caller = call_once or _call_model_once
    observer = observe_once or _observe_image_once
    use_observation_pass = call_once is None or observe_once is not None
    observation: Optional[PhotoAssistObservation] = None
    last_error: Optional[Exception] = None

    for attempt in range(2):
        try:
            if use_observation_pass and observation is None:
                observation = observer(image, model=selected_observation_model, client=openai_client)
            if call_once and not use_observation_pass:
                extracted = caller(image, model=selected_model, client=openai_client)
            else:
                extracted = caller(
                    image,
                    model=selected_model,
                    client=openai_client,
                    preferred_dimension_unit=preferred_dimension_unit,
                    known_glass_types=known_glass_types,
                    known_clients=known_clients,
                    observation=observation,
                )
            normalized = normalize_extraction(
                extracted,
                preferred_dimension_unit=preferred_dimension_unit,
            )
            return PhotoAssistResponse(
                **normalized.model_dump(),
                status="needs_review" if _result_has_warnings(normalized) else "ready",
                request_id=request_id,
                model=(
                    f"{selected_observation_model} -> {selected_model}"
                    if use_observation_pass
                    else selected_model
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
