"""Public tool contracts. All dimensions named *_mm are always millimetres."""
from __future__ import annotations

from datetime import date
from typing import Any, Generic, Literal, TypeVar
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Empty(Contract):
    pass


class ManualRow(Contract):
    section: str = Field(default="", max_length=255)
    position: str = Field(default="", max_length=80)
    client_position: str = Field(default="", max_length=120)
    red_index: int | None = Field(default=None, gt=0, strict=True)
    width_mm: float = Field(gt=0, le=99999)
    height_mm: float = Field(gt=0, le=99999)
    quantity: int = Field(gt=0, le=2147483647, strict=True)
    glass_type: str = Field(min_length=1, max_length=255)
    row_notes: str = Field(default="", max_length=4000)
    area_override_m2: float | None = Field(default=None, ge=0)

    @field_validator("glass_type")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Must contain text")
        return value


class ManualDraft(Contract):
    client_name: str = Field(min_length=1, max_length=255)
    order_number: str = Field(min_length=1, max_length=120)
    order_date: date
    mode: Literal["standard", "client_positions_red_index"] = "standard"
    reference_notes: str = Field(default="", max_length=8000)
    dimension_unit: Literal["mm", "cm"] = Field(default="mm", description="Original entry/display unit; width_mm and height_mm always contain millimetres.")
    rows: list[ManualRow] = Field(min_length=1, max_length=1000)

    @field_validator("client_name", "order_number")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Must contain text")
        return value


class CreateDraft(ManualDraft):
    idempotency_key: str = Field(min_length=8, max_length=128)


class OrderRef(Contract):
    order_id: str = Field(pattern=r"^(manual|pdf):[1-9][0-9]*$", description="Stable ID from list/get, e.g. manual:42 or pdf:42. PDF denotes the extracted-order store, including text/Telegram sources.")


class ManualRef(Contract):
    order_id: str = Field(pattern=r"^manual:[1-9][0-9]*$")


class VersionedOrder(OrderRef):
    expected_version: str = Field(min_length=64, max_length=64, description="Opaque version returned by get_order. Refetch on VERSION_CONFLICT.")


class UpdateDraft(ManualRef):
    expected_version: str = Field(min_length=64, max_length=64)
    replacement: ManualDraft = Field(description="Complete replacement of every editable header and row field; omitted optional fields use their defaults.")


class ConfirmOrder(VersionedOrder):
    confirmed: Literal[True] = Field(description="Set true only after the user explicitly confirms this particular consequential action.")


class ProcessOrder(ConfirmOrder):
    idempotency_key: str = Field(min_length=8, max_length=128)


class DeleteDraft(ConfirmOrder):
    order_id: str = Field(pattern=r"^manual:[1-9][0-9]*$")


class JobRef(Contract):
    processing_job_id: str = Field(pattern=r"^job:[0-9a-f]{32}$")


class ChangeJob(JobRef):
    expected_version: int = Field(ge=1, strict=True)


class GenerateArtifact(ChangeJob):
    idempotency_key: str = Field(min_length=8, max_length=128)


class InvoiceDraft(VersionedOrder):
    idempotency_key: str = Field(min_length=8, max_length=128)


class OrderFilters(Contract):
    year: int | Literal["all"] = "all"
    source: Literal["all", "pdf", "manual"] = "all"
    status: Literal["draft", "reviewed", "approved", "processing", "in_production", "finished", "completed", "cancelled", "archived"] | None = None
    client: str | None = Field(default=None, max_length=255)
    order_number: str | None = Field(default=None, max_length=120)
    date_from: date | None = None
    date_to: date | None = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0, le=1000000)

    @model_validator(mode="after")
    def dates(self):
        if self.year != "all" and not 1900 <= self.year <= 9998:
            raise ValueError("year must be 1900..9998 or all")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        return self


class ManualFilters(OrderFilters):
    source: Literal["manual"] = "manual"


class Issue(Contract):
    field: str
    code: str
    message: str


class ErrorInfo(Contract):
    code: str
    message: str
    issues: list[Issue] = Field(default_factory=list)
    retryable: bool = False


class Artifact(Contract):
    artifact_id: str
    order_id: str
    processing_job_id: str | None = None
    kind: str
    media_type: str
    byte_count: int
    sha256: str
    created_at: str
    download_path: str = Field(description="Authenticated relative HTTP path. Send the bearer header; never put the token in the URL.")
    job_version: int | None = None


class OrderView(Contract):
    order_id: str
    source: str
    storage_source: Literal["manual", "pdf"]
    order_number: str
    client_name: str
    order_date: str
    status: str
    version: str
    row_count: int
    piece_count: int
    calculated_area_m2: float
    total_area_m2: float
    mode: str
    dimension_unit: str
    reference_notes: str
    rows: list[dict[str, Any]]
    raw_values: dict[str, Any] | None = None
    warnings: list[str]
    artifacts: list[Artifact]


class OrderSummary(Contract):
    order_id: str
    source: str
    storage_source: Literal["manual", "pdf"]
    order_number: str
    client_name: str
    order_date: str
    status: str
    version: str
    row_count: int
    piece_count: int
    total_area_m2: float


class OrdersPage(Contract):
    items: list[OrderSummary]
    total: int
    limit: int
    offset: int
    has_more: bool


class JobView(Contract):
    processing_job_id: str
    order_id: str
    order_version: str
    version: int
    state: str
    rounding_applied: bool
    grouped: bool
    original_rows: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    warnings: list[str]
    artifacts: list[Artifact]


class ArtifactsView(Contract):
    artifacts: list[Artifact]


class HealthView(Contract):
    backend: str
    database: str
    storage: str
    mcp: str
    workflow_runtime: str
    durable_storage_configured: bool


class SummaryView(Contract):
    draft_count: int
    approved_count: int
    processing_count: int
    completed_count: int
    pieces: int
    area_m2: float


class DeletedView(Contract):
    order_id: str
    deleted: bool


class InvoiceView(Contract):
    invoice_id: str
    order_id: str
    status: Literal["draft"]
    currency: Literal["ALL"] = "ALL"
    safe_to_price: bool
    invoice: dict[str, Any]
    warnings: list[str]
    artifact: Artifact


T = TypeVar("T")


class Result(Contract, Generic[T]):
    ok: bool
    request_id: str
    data: T | None = None
    error: ErrorInfo | None = None
