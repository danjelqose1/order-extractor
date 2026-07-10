from __future__ import annotations

import base64
import json
import os
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
MAX_IMAGE_LONG_EDGE = 3200
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
    # No Terra API identifier exists in this project yet. Configure the exact
    # identifier on Render through MANUAL_PHOTO_ASSIST_MODEL when it is available.
    return (
        os.getenv("MANUAL_PHOTO_ASSIST_MODEL")
        or os.getenv("OCR_MODEL")
        or "gpt-5.4-mini"
    )


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


@dataclass(frozen=True)
class NormalizedImage:
    content: bytes
    mime_type: str
    width: int
    height: int
    original_format: str


EXTRACTION_PROMPT = """
You are extracting handwritten glass-order data from a photograph. Read the document conservatively.

Never invent missing numbers. When a digit, decimal, quantity, index, client reference, or glass type is unclear, return null or preserve the uncertain raw text and add a warning. Preserve original detected values in raw fields. Do not calculate normalized_mm; the backend does that deterministically.

Possible layouts include glass-type headings, section headings such as Vila 1/Vila 2/Vila 3, black client references such as K1/K2/SHK, red handwritten index numbers, and rows written as width x height x quantity, width x height = quantity, width x height + quantity, or with arrows. Decimal dimensions are common. Notes such as WC must stay separate.

Rules:
1. client_reference and index_number are separate fields. Never combine them.
2. Red numbers may be index numbers. Black codes may be client references.
3. A section applies to following rows until the next section.
4. A glass-type heading applies to following rows until another type appears.
5. The final integer after two dimensions usually represents quantity, not part of height.
6. If there are no index numbers or client references, leave those values null.
7. Never manufacture client name or order number when absent.
8. Preserve source_line for every detected row.
9. Preserve raw values before normalization.
10. Dimensions may be mm or cm. Suggest the unit only when the document is sufficiently clear; otherwise use unknown and warn.
11. Do not use area to correct or infer dimensions.
12. Return incomplete rows with warnings instead of discarding them.

Return only data matching the required JSON schema. Do not return markdown or hidden reasoning.
""".strip()


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
        message = "Dimensions were interpreted as centimetres. Confirm the normalized millimetres before applying."
        dimension.warning = _append_warning(dimension.warning, message)
        row_warnings.append(message)
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


def normalize_extraction(ai_result: PhotoAssistAIResult) -> PhotoAssistAIResult:
    result = ai_result.model_copy(deep=True)
    result.rows = [row for row in result.rows if _row_has_meaning(row)]
    if not result.rows:
        raise NoOrderRowsError("No order rows were detected.")

    detected_units = set()
    index_rows: Dict[int, List[int]] = {}
    inherited_glass = result.document.glass_type.value

    for row_number, row in enumerate(result.rows, start=1):
        row.warnings = [str(item).strip() for item in row.warnings if str(item).strip()]
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

    result.global_warnings = list(
        dict.fromkeys(str(item).strip() for item in result.global_warnings if str(item).strip())
    )
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


def _default_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise PhotoAssistUnavailableError("Photo Assist is temporarily unavailable.")
    return OpenAI(api_key=api_key)


def _call_model_once(
    image: NormalizedImage,
    *,
    model: str,
    client: Any,
) -> PhotoAssistAIResult:
    encoded = base64.b64encode(image.content).decode("ascii")
    data_url = f"data:{image.mime_type};base64,{encoded}"
    response = client.responses.create(
        model=model,
        instructions=EXTRACTION_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Extract the handwritten glass order from this image for user review.",
                    },
                    {
                        "type": "input_image",
                        "image_url": data_url,
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
    call_once: Optional[Callable[..., PhotoAssistAIResult]] = None,
) -> PhotoAssistResponse:
    selected_model = model or photo_assist_model()
    openai_client = client or _default_client()
    caller = call_once or _call_model_once
    last_error: Optional[Exception] = None

    for attempt in range(2):
        try:
            extracted = caller(image, model=selected_model, client=openai_client)
            normalized = normalize_extraction(extracted)
            return PhotoAssistResponse(
                **normalized.model_dump(),
                status="needs_review" if _result_has_warnings(normalized) else "ready",
                request_id=request_id,
                model=selected_model,
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
