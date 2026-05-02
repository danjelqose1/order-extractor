from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    func,
    or_,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from utils_text import build_signature, extract_client_hint, normalize_order_number

# Allow Render to override the local data path
DB_DIR = os.getenv("DB_DIR", "data")  # defaults to ./data when local
DB_FILENAME = "orders.db"
DB_PATH = os.path.join(DB_DIR, DB_FILENAME)

ORDER_STATUS_SEQUENCE = (
    "draft",
    "reviewed",
    "approved",
    "in_production",
    "completed",
    "archived",
)
ORDER_STATUS_LABELS = {
    "draft": "Draft",
    "reviewed": "Reviewed",
    "approved": "Approved",
    "in_production": "In Production",
    "completed": "Completed",
    "archived": "Archived",
}
ORDER_STATUS_ALIASES = {
    "draft": "draft",
    "reviewed": "reviewed",
    "approve": "approved",
    "approved": "approved",
    "in production": "in_production",
    "in-production": "in_production",
    "in_production": "in_production",
    "inproduction": "in_production",
    "completed": "completed",
    "complete": "completed",
    "archived": "archived",
    "archive": "archived",
}
REEXTRACT_PROTECTED_STATUSES = {"reviewed", "approved", "in_production", "completed", "archived"}
APPROVABLE_STATUSES = {"draft", "reviewed"}
PROCESSING_ELIGIBLE_STATUSES = {"approved", "in_production", "completed"}


def normalize_order_status(status: Optional[str], default: str = "draft") -> str:
    raw = (status or "").strip().lower()
    if not raw:
        return default
    return ORDER_STATUS_ALIASES.get(raw, default)


def order_status_label(status: Optional[str]) -> str:
    normalized = normalize_order_status(status)
    return ORDER_STATUS_LABELS.get(normalized, ORDER_STATUS_LABELS["draft"])


def is_processing_eligible_status(status: Optional[str]) -> bool:
    return normalize_order_status(status) in PROCESSING_ELIGIBLE_STATUSES


def _ensure_data_dir() -> None:
    os.makedirs(DB_DIR, exist_ok=True)


_ensure_data_dir()

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    future=True,
    echo=False,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (UniqueConstraint("hash", name="uq_orders_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    source: Mapped[str] = mapped_column(String(20))
    client_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    client_hint: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    order_numbers_raw: Mapped[str] = mapped_column("order_numbers", Text, default="")
    units_total: Mapped[int] = mapped_column(Integer, default=0)
    area_total: Mapped[float] = mapped_column(Float, default=0.0)
    hash: Mapped[Optional[str]] = mapped_column(String(64), unique=True, nullable=True)
    source_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_order_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source_metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="draft")
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    rows: Mapped[List["OrderRow"]] = relationship(
        back_populates="order",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    extraction: Mapped[Optional["Extraction"]] = relationship(
        back_populates="order",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    status_events: Mapped[List["OrderStatusEvent"]] = relationship(
        back_populates="order",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="OrderStatusEvent.changed_at",
    )

    @property
    def order_numbers(self) -> List[str]:
        raw = self.order_numbers_raw or ""
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x) for x in data if x]
        except Exception:
            pass
        return [part.strip() for part in raw.split(",") if part.strip()]

    @order_numbers.setter
    def order_numbers(self, values: Iterable[str]) -> None:
        if values is None:
            self.order_numbers_raw = ""
            return
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        self.order_numbers_raw = json.dumps(sorted(set(cleaned)), ensure_ascii=True)


class OrderRow(Base):
    __tablename__ = "order_rows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), index=True)
    order_number: Mapped[str] = mapped_column(String, default="")
    type: Mapped[str] = mapped_column(String)
    dimension: Mapped[str] = mapped_column(String, default="")
    position: Mapped[str] = mapped_column(String, default="")
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    area: Mapped[float] = mapped_column(Float, default=0.0)

    order: Mapped[Order] = relationship(back_populates="rows")


class Extraction(Base):
    __tablename__ = "extractions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), unique=True)
    raw_input: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    llm_output_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prepared_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_used: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    order: Mapped[Order] = relationship(back_populates="extraction")


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pattern_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    pattern_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    before_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    after_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    hits: Mapped[int] = mapped_column(Integer, default=1)
    last_used_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class OrderStatusEvent(Base):
    __tablename__ = "order_status_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), index=True)
    from_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    to_status: Mapped[str] = mapped_column(String(20), nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    order: Mapped[Order] = relationship(back_populates="status_events")


class ProcessingBatch(Base):
    __tablename__ = "processing_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), index=True)
    order_number: Mapped[Optional[str]] = mapped_column(String(80), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(40), default="created")
    requested_by: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    forced: Mapped[bool] = mapped_column(Boolean, default=False)
    summary_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class ProductionFile(Base):
    __tablename__ = "production_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), index=True)
    processing_batch_id: Mapped[int] = mapped_column(ForeignKey("processing_batches.id", ondelete="CASCADE"), index=True)
    order_number: Mapped[Optional[str]] = mapped_column(String(80), nullable=True, index=True)
    file_type: Mapped[str] = mapped_column(String(40), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    download_url: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="ready")


class TelegramFile(Base):
    __tablename__ = "telegram_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    source: Mapped[str] = mapped_column(String(40), default="telegram", index=True)
    original_filename: Mapped[str] = mapped_column(Text, nullable=False)
    stored_filename: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(120), default="application/pdf")
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    telegram_file_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    telegram_chat_id: Mapped[Optional[str]] = mapped_column(String(120), nullable=True, index=True)
    telegram_message_id: Mapped[Optional[str]] = mapped_column(String(120), nullable=True, index=True)
    telegram_sender_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    telegram_caption: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    linked_order_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    extraction_status: Mapped[str] = mapped_column(String(40), default="received", index=True)
    touched: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    touched_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    touched_by: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    labels_printed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    labels_printed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    linked_order_opened: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    linked_order_opened_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    queued_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class WorkspaceAction(Base):
    __tablename__ = "workspace_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    actor: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    action_type: Mapped[str] = mapped_column(String(80), nullable=False)
    order_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    order_number: Mapped[Optional[str]] = mapped_column(String(80), nullable=True, index=True)
    processing_batch_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(60), default="")
    requested_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tool_name: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    input_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    output_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    requires_confirmation: Mapped[bool] = mapped_column(Boolean, default=False)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


def _ensure_schema() -> None:
    with engine.begin() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        info = conn.execute(text("PRAGMA table_info(orders)")).fetchall()
        columns = {row[1] for row in info}
        if "client_name" not in columns:
            conn.execute(text("ALTER TABLE orders ADD COLUMN client_name TEXT"))
        if "status" not in columns:
            conn.execute(text("ALTER TABLE orders ADD COLUMN status TEXT DEFAULT 'draft'"))
        if "notes" not in columns:
            conn.execute(text("ALTER TABLE orders ADD COLUMN notes TEXT"))
        if "updated_at" not in columns:
            conn.execute(text("ALTER TABLE orders ADD COLUMN updated_at TEXT"))
        if "source_hash" not in columns:
            conn.execute(text("ALTER TABLE orders ADD COLUMN source_hash TEXT"))
        if "version" not in columns:
            conn.execute(text("ALTER TABLE orders ADD COLUMN version INTEGER DEFAULT 1"))
        if "parent_order_id" not in columns:
            conn.execute(text("ALTER TABLE orders ADD COLUMN parent_order_id INTEGER"))
        if "confidence" not in columns:
            conn.execute(text("ALTER TABLE orders ADD COLUMN confidence REAL"))
        if "source_metadata" not in columns:
            conn.execute(text("ALTER TABLE orders ADD COLUMN source_metadata TEXT"))
        info_rows = conn.execute(text("PRAGMA table_info(orders)")).fetchall()
        columns = {row[1] for row in info_rows}
        if "order_numbers" in columns:
            conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.execute(text("UPDATE orders SET updated_at = created_at WHERE updated_at IS NULL"))
        conn.execute(text("UPDATE orders SET source_hash = hash WHERE source_hash IS NULL AND hash IS NOT NULL"))
        conn.execute(text("UPDATE orders SET version = 1 WHERE version IS NULL OR version < 1"))
        conn.execute(text("UPDATE orders SET status = 'draft' WHERE status IS NULL OR TRIM(status) = ''"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_orders_source_hash ON orders(source_hash)"))
        telegram_info = conn.execute(text("PRAGMA table_info(telegram_files)")).fetchall()
        telegram_columns = {row[1] for row in telegram_info}
        if telegram_columns:
            if "telegram_file_id" not in telegram_columns:
                conn.execute(text("ALTER TABLE telegram_files ADD COLUMN telegram_file_id TEXT"))
            if "telegram_caption" not in telegram_columns:
                conn.execute(text("ALTER TABLE telegram_files ADD COLUMN telegram_caption TEXT"))
            if "touched" not in telegram_columns:
                conn.execute(text("ALTER TABLE telegram_files ADD COLUMN touched BOOLEAN DEFAULT 0"))
            if "touched_at" not in telegram_columns:
                conn.execute(text("ALTER TABLE telegram_files ADD COLUMN touched_at TEXT"))
            if "touched_by" not in telegram_columns:
                conn.execute(text("ALTER TABLE telegram_files ADD COLUMN touched_by TEXT"))
            if "labels_printed" not in telegram_columns:
                conn.execute(text("ALTER TABLE telegram_files ADD COLUMN labels_printed BOOLEAN DEFAULT 0"))
            if "labels_printed_at" not in telegram_columns:
                conn.execute(text("ALTER TABLE telegram_files ADD COLUMN labels_printed_at TEXT"))
            if "linked_order_opened" not in telegram_columns:
                conn.execute(text("ALTER TABLE telegram_files ADD COLUMN linked_order_opened BOOLEAN DEFAULT 0"))
            if "linked_order_opened_at" not in telegram_columns:
                conn.execute(text("ALTER TABLE telegram_files ADD COLUMN linked_order_opened_at TEXT"))
            if "deleted" not in telegram_columns:
                conn.execute(text("ALTER TABLE telegram_files ADD COLUMN deleted BOOLEAN DEFAULT 0"))
            if "deleted_at" not in telegram_columns:
                conn.execute(text("ALTER TABLE telegram_files ADD COLUMN deleted_at TEXT"))
            if "queued_at" not in telegram_columns:
                conn.execute(text("ALTER TABLE telegram_files ADD COLUMN queued_at TEXT"))
            if "processing_started_at" not in telegram_columns:
                conn.execute(text("ALTER TABLE telegram_files ADD COLUMN processing_started_at TEXT"))
            if "processed_at" not in telegram_columns:
                conn.execute(text("ALTER TABLE telegram_files ADD COLUMN processed_at TEXT"))
            if "retry_count" not in telegram_columns:
                conn.execute(text("ALTER TABLE telegram_files ADD COLUMN retry_count INTEGER DEFAULT 0"))
            if "last_error" not in telegram_columns:
                conn.execute(text("ALTER TABLE telegram_files ADD COLUMN last_error TEXT"))
            conn.execute(text("UPDATE telegram_files SET touched = 0 WHERE touched IS NULL"))
            conn.execute(text("UPDATE telegram_files SET labels_printed = 0 WHERE labels_printed IS NULL"))
            conn.execute(text("UPDATE telegram_files SET linked_order_opened = 0 WHERE linked_order_opened IS NULL"))
            conn.execute(text("UPDATE telegram_files SET deleted = 0 WHERE deleted IS NULL"))
            conn.execute(text("UPDATE telegram_files SET retry_count = 0 WHERE retry_count IS NULL"))


def _normalize_client_name(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text_value = str(value).strip()
        if text_value and text_value not in {"-", "\u2014"}:
            return text_value
    return None


def _client_name_from_payload(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    direct = _normalize_client_name(
        payload.get("client_name"),
        payload.get("clientName"),
        payload.get("client"),
    )
    if direct:
        return direct
    meta = payload.get("_meta")
    if isinstance(meta, dict):
        parsed = meta.get("parsed_result")
        nested = _client_name_from_payload(parsed)
        if nested:
            return nested
    data = payload.get("data")
    if isinstance(data, dict):
        return _client_name_from_payload(data)
    return None


def _client_name_from_extraction(extraction: Optional["Extraction"]) -> Optional[str]:
    if not extraction or not extraction.llm_output_json:
        return None
    try:
        payload = json.loads(extraction.llm_output_json)
    except Exception:
        return None
    return _client_name_from_payload(payload)


def init_db() -> None:
    _ensure_data_dir()
    Base.metadata.create_all(engine, checkfirst=True)
    _ensure_schema()


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _serialize_order(order: Order, include_rows: bool = False) -> Dict[str, Any]:
    normalized_status = normalize_order_status(order.status)
    client_name = _normalize_client_name(order.client_name, _client_name_from_extraction(order.extraction))
    client_legacy = client_name or _normalize_client_name(order.client_hint) or ""
    source_metadata: Dict[str, Any] = {}
    if order.source_metadata:
        try:
            parsed_metadata = json.loads(order.source_metadata)
            if isinstance(parsed_metadata, dict):
                source_metadata = parsed_metadata
        except Exception:
            source_metadata = {}
    data: Dict[str, Any] = {
        "id": order.id,
        "created_at": order.created_at.isoformat(),
        "updated_at": (order.updated_at or order.created_at).isoformat(),
        "source": order.source,
        "client_name": client_name or "",
        "clientName": client_name or "",
        "client": client_legacy,
        "client_hint": order.client_hint,
        "order_numbers": order.order_numbers,
        "units_total": order.units_total,
        "area_total": order.area_total,
        "hash": order.hash,
        "source_hash": order.source_hash or order.hash,
        "version": int(order.version or 1),
        "parent_order_id": order.parent_order_id,
        "confidence": order.confidence,
        "source_metadata": source_metadata,
        "status": normalized_status,
        "status_label": order_status_label(normalized_status),
        "notes": order.notes,
    }
    if include_rows:
        data["rows"] = [
            {
                "id": row.id,
                "order_number": row.order_number,
                "type": row.type,
                "dimension": row.dimension,
                "position": row.position,
                "quantity": row.quantity,
                "area": row.area,
            }
            for row in order.rows
        ]
    return data


def _serialize_extraction(extraction: Optional[Extraction]) -> Optional[Dict[str, Any]]:
    if not extraction:
        return None
    return {
        "id": extraction.id,
        "order_id": extraction.order_id,
        "raw_input": extraction.raw_input,
        "llm_output_json": extraction.llm_output_json,
        "prepared_text": extraction.prepared_text,
        "model_used": extraction.model_used,
    }


def _serialize_telegram_file(file: TelegramFile, order: Optional[Order] = None) -> Dict[str, Any]:
    linked_order = _serialize_order(order, include_rows=False) if order else None
    return {
        "id": file.id,
        "created_at": file.created_at.isoformat(),
        "received_at": file.received_at.isoformat(),
        "source": file.source,
        "original_filename": file.original_filename,
        "stored_filename": file.stored_filename,
        "file_path": file.file_path,
        "mime_type": file.mime_type,
        "file_size": file.file_size,
        "telegram_file_id": file.telegram_file_id,
        "telegram_chat_id": file.telegram_chat_id,
        "telegram_message_id": file.telegram_message_id,
        "telegram_sender_name": file.telegram_sender_name,
        "telegram_caption": file.telegram_caption,
        "linked_order_id": file.linked_order_id,
        "linked_order": linked_order,
        "extraction_status": file.extraction_status,
        "touched": bool(file.touched),
        "touched_at": file.touched_at.isoformat() if file.touched_at else None,
        "touched_by": file.touched_by,
        "labels_printed": bool(file.labels_printed),
        "labels_printed_at": file.labels_printed_at.isoformat() if file.labels_printed_at else None,
        "linked_order_opened": bool(file.linked_order_opened),
        "linked_order_opened_at": file.linked_order_opened_at.isoformat() if file.linked_order_opened_at else None,
        "deleted": bool(file.deleted),
        "deleted_at": file.deleted_at.isoformat() if file.deleted_at else None,
        "queued_at": file.queued_at.isoformat() if file.queued_at else None,
        "processing_started_at": file.processing_started_at.isoformat() if file.processing_started_at else None,
        "processed_at": file.processed_at.isoformat() if file.processed_at else None,
        "retry_count": int(file.retry_count or 0),
        "last_error": file.last_error,
        "view_url": f"/telegram-files/{file.id}/view",
        "download_url": f"/telegram-files/{file.id}/download",
    }


def _serialize_status_events(events: Sequence[OrderStatusEvent]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for event in events:
        from_status = normalize_order_status(event.from_status) if event.from_status else None
        to_status = normalize_order_status(event.to_status)
        serialized.append(
            {
                "id": event.id,
                "from_status": from_status,
                "from_status_label": order_status_label(from_status) if from_status else None,
                "to_status": to_status,
                "to_status_label": order_status_label(to_status),
                "note": event.note,
                "reason": event.reason,
                "changed_at": event.changed_at.isoformat(),
            }
        )
    return serialized


def _compute_totals(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    units = 0
    area = 0.0
    numbers = set()
    for row in rows:
        try:
            units += int(row.get("quantity") or 0)
        except Exception:
            pass
        try:
            area += float(row.get("area") or 0.0)
        except Exception:
            pass
        num = normalize_order_number(row.get("order_number", ""))
        if num:
            numbers.add(num)
    return {"units": units, "area": round(area, 3), "order_numbers": sorted(numbers)}


def _build_versioned_hash(base_hash: str, version: int) -> str:
    cleaned = (base_hash or "").strip()
    if not cleaned:
        return ""
    if version <= 1:
        return cleaned[:64]
    suffix = f"::v{version}"
    if len(cleaned) + len(suffix) <= 64:
        return f"{cleaned}{suffix}"
    digest = hashlib.sha1(f"{cleaned}:{version}".encode("utf-8", "ignore")).hexdigest()
    return digest[:64]


def _next_version_for_source_hash(session: Session, source_hash: str) -> int:
    if not source_hash:
        return 1
    max_version = session.execute(
        select(func.max(Order.version)).where(func.coalesce(Order.source_hash, Order.hash) == source_hash)
    ).scalar_one_or_none()
    return int(max_version or 0) + 1


def _record_status_event(
    session: Session,
    *,
    order_id: int,
    from_status: Optional[str],
    to_status: str,
    note: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    normalized_to = normalize_order_status(to_status)
    normalized_from = normalize_order_status(from_status) if from_status else None
    if normalized_from == normalized_to:
        return
    session.add(
        OrderStatusEvent(
            order_id=order_id,
            from_status=normalized_from,
            to_status=normalized_to,
            note=(note or "").strip() or None,
            reason=(reason or "").strip() or None,
        )
    )


def _can_transition_status(from_status: str, to_status: str) -> bool:
    source = normalize_order_status(from_status)
    target = normalize_order_status(to_status)
    if source == target:
        return True
    allowed_map = {
        "draft": {"reviewed", "approved", "archived"},
        "reviewed": {"draft", "approved", "archived"},
        "approved": {"in_production", "completed", "archived"},
        "in_production": {"completed", "archived"},
        "completed": {"archived"},
        "archived": set(),
    }
    return target in allowed_map.get(source, set())


def insert_extraction_with_rows(
    *,
    source: str,
    rows: List[Dict[str, Any]],
    raw_input: Optional[str],
    prepared_text: Optional[str],
    llm_output_json: str,
    model_used: Optional[str],
    hash_value: Optional[str],
    confidence: Optional[float] = None,
    client_name: Optional[str] = None,
    source_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("rows are required to insert extraction")

    totals = _compute_totals(rows)
    normalized_client_name = _normalize_client_name(client_name)
    client_hint = extract_client_hint(raw_input or prepared_text or "")
    canonical_hash = (hash_value or "").strip() or None
    metadata_json = (
        json.dumps(source_metadata, ensure_ascii=False, sort_keys=True)
        if isinstance(source_metadata, dict) and source_metadata
        else None
    )

    with get_session() as session:
        order: Optional[Order] = None
        if canonical_hash:
            order = session.execute(
                select(Order)
                .where(or_(Order.hash == canonical_hash, Order.source_hash == canonical_hash))
                .order_by(Order.created_at.desc())
            ).scalars().first()

        created_new_version = False
        protected_order_id: Optional[int] = None

        if not order:
            order = Order(
                source=source,
                client_name=normalized_client_name,
                client_hint=client_hint,
                hash=canonical_hash,
                source_hash=canonical_hash,
                version=1,
                status="draft",
                confidence=confidence,
                source_metadata=metadata_json,
            )
            session.add(order)
            session.flush()
            _record_status_event(
                session,
                order_id=order.id,
                from_status=None,
                to_status="draft",
                reason="initial_extract",
            )
        else:
            current_status = normalize_order_status(order.status)
            if current_status in REEXTRACT_PROTECTED_STATUSES:
                protected_order_id = order.id
                next_version = _next_version_for_source_hash(session, canonical_hash or (order.source_hash or order.hash or ""))
                versioned_hash = _build_versioned_hash(canonical_hash or (order.source_hash or order.hash or ""), next_version)
                order = Order(
                    source=source,
                    client_name=normalized_client_name,
                    client_hint=client_hint or "",
                    hash=versioned_hash or None,
                    source_hash=canonical_hash or (order.source_hash or order.hash),
                    version=next_version,
                    parent_order_id=protected_order_id,
                    status="draft",
                    confidence=confidence,
                    source_metadata=metadata_json,
                )
                session.add(order)
                session.flush()
                created_new_version = True
                _record_status_event(
                    session,
                    order_id=order.id,
                    from_status=None,
                    to_status="draft",
                    reason="reextract_new_version",
                )
            else:
                previous_status = normalize_order_status(order.status)
                order.source = source or order.source
                order.client_name = normalized_client_name or order.client_name
                order.client_hint = client_hint or order.client_hint
                order.status = "draft"
                order.source_hash = canonical_hash or order.source_hash or order.hash
                order.version = int(order.version or 1)
                if confidence is not None:
                    order.confidence = confidence
                if metadata_json is not None:
                    order.source_metadata = metadata_json
                if previous_status != "draft":
                    _record_status_event(
                        session,
                        order_id=order.id,
                        from_status=previous_status,
                        to_status="draft",
                        reason="reextract_reset",
                    )

        session.execute(delete(OrderRow).where(OrderRow.order_id == order.id))
        for row in rows:
            session.add(
                OrderRow(
                    order_id=order.id,
                    order_number=row.get("order_number", ""),
                    type=row.get("type") or "",
                    dimension=row.get("dimension") or "",
                    position=row.get("position") or "",
                    quantity=int(row.get("quantity") or 0),
                    area=float(row.get("area") or 0.0),
                )
            )

        order.units_total = totals["units"]
        order.area_total = totals["area"]
        order.order_numbers = totals["order_numbers"]
        order.client_name = normalized_client_name or order.client_name
        order.client_hint = client_hint or order.client_hint
        if confidence is not None:
            order.confidence = confidence
        if metadata_json is not None:
            order.source_metadata = metadata_json
        order.updated_at = datetime.now(timezone.utc)

        extraction = order.extraction
        if not extraction:
            extraction = Extraction(order_id=order.id)
            session.add(extraction)
        extraction.raw_input = raw_input
        extraction.llm_output_json = llm_output_json
        extraction.prepared_text = prepared_text
        extraction.model_used = model_used

        session.flush()

        return {
            "order_id": order.id,
            "status": normalize_order_status(order.status),
            "order_numbers": order.order_numbers,
            "client_name": order.client_name,
            "client_hint": order.client_hint,
            "version": int(order.version or 1),
            "source_hash": order.source_hash or order.hash,
            "created_new_version": created_new_version,
            "protected_order_id": protected_order_id,
        }


def create_telegram_file_record(
    *,
    original_filename: str,
    stored_filename: str,
    file_path: str,
    mime_type: str,
    file_size: int,
    telegram_file_id: Optional[str] = None,
    telegram_chat_id: Optional[Any] = None,
    telegram_message_id: Optional[Any] = None,
    telegram_sender_name: Optional[str] = None,
    telegram_caption: Optional[str] = None,
    received_at: Optional[datetime] = None,
    extraction_status: str = "received",
    queued_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    with get_session() as session:
        record = TelegramFile(
            source="telegram",
            original_filename=str(original_filename or "telegram-order.pdf"),
            stored_filename=str(stored_filename or ""),
            file_path=str(file_path or ""),
            mime_type=str(mime_type or "application/pdf"),
            file_size=int(file_size or 0),
            telegram_file_id=str(telegram_file_id) if telegram_file_id else None,
            telegram_chat_id=str(telegram_chat_id) if telegram_chat_id is not None else None,
            telegram_message_id=str(telegram_message_id) if telegram_message_id is not None else None,
            telegram_sender_name=telegram_sender_name or None,
            telegram_caption=telegram_caption or None,
            received_at=received_at or datetime.now(timezone.utc),
            extraction_status=extraction_status or "received",
            touched=False,
            labels_printed=False,
            linked_order_opened=False,
            deleted=False,
            queued_at=queued_at,
            retry_count=0,
        )
        session.add(record)
        session.flush()
        return _serialize_telegram_file(record)


def update_telegram_file_record(
    file_id: int,
    *,
    linked_order_id: Optional[int] = None,
    extraction_status: Optional[str] = None,
    queued_at: Optional[datetime] = None,
    processing_started_at: Optional[datetime] = None,
    processed_at: Optional[datetime] = None,
    retry_count: Optional[int] = None,
    last_error: Optional[str] = None,
    clear_last_error: bool = False,
) -> Optional[Dict[str, Any]]:
    with get_session() as session:
        record = session.get(TelegramFile, int(file_id))
        if not record:
            return None
        if linked_order_id is not None:
            record.linked_order_id = int(linked_order_id)
        if extraction_status:
            record.extraction_status = extraction_status
        if queued_at is not None:
            record.queued_at = queued_at
        if processing_started_at is not None:
            record.processing_started_at = processing_started_at
        if processed_at is not None:
            record.processed_at = processed_at
        if retry_count is not None:
            record.retry_count = int(retry_count)
        if clear_last_error:
            record.last_error = None
        elif last_error is not None:
            record.last_error = str(last_error)[:1000]
        session.flush()
        linked_order = session.get(Order, record.linked_order_id) if record.linked_order_id else None
        return _serialize_telegram_file(record, linked_order)


def touch_telegram_file_record(file_id: int, *, touched_by: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with get_session() as session:
        record = session.get(TelegramFile, int(file_id))
        if not record:
            return None
        record.touched = True
        record.touched_at = datetime.now(timezone.utc)
        if touched_by:
            record.touched_by = str(touched_by)[:120]
        session.flush()
        linked_order = session.get(Order, record.linked_order_id) if record.linked_order_id else None
        return _serialize_telegram_file(record, linked_order)


def soft_delete_telegram_file_record(file_id: int) -> Optional[Dict[str, Any]]:
    with get_session() as session:
        record = session.get(TelegramFile, int(file_id))
        if not record:
            return None
        if not record.deleted:
            record.deleted = True
            record.deleted_at = datetime.now(timezone.utc)
        session.flush()
        linked_order = session.get(Order, record.linked_order_id) if record.linked_order_id else None
        return _serialize_telegram_file(record, linked_order)


def mark_telegram_file_labels_printed(file_id: int, *, touched_by: Optional[str] = None) -> Optional[Dict[str, Any]]:
    return _mark_telegram_file_handling(file_id, labels_printed=True, touched_by=touched_by)


def mark_telegram_file_linked_order_opened(file_id: int, *, touched_by: Optional[str] = None) -> Optional[Dict[str, Any]]:
    return _mark_telegram_file_handling(file_id, linked_order_opened=True, touched_by=touched_by)


def _mark_telegram_file_handling(
    file_id: int,
    *,
    labels_printed: bool = False,
    linked_order_opened: bool = False,
    touched_by: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    with get_session() as session:
        record = session.get(TelegramFile, int(file_id))
        if not record:
            return None
        now = datetime.now(timezone.utc)
        if labels_printed:
            record.labels_printed = True
            record.labels_printed_at = now
        if linked_order_opened:
            record.linked_order_opened = True
            record.linked_order_opened_at = now
        can_auto_touch = (
            bool(record.linked_order_id)
            and str(record.extraction_status or "").lower() != "failed"
            and bool(record.labels_printed)
            and bool(record.linked_order_opened)
        )
        if can_auto_touch and not record.touched:
            record.touched = True
            record.touched_at = now
            if touched_by:
                record.touched_by = str(touched_by)[:120]
        session.flush()
        linked_order = session.get(Order, record.linked_order_id) if record.linked_order_id else None
        return _serialize_telegram_file(record, linked_order)


def update_order_rows(
    order_id: int,
    rows: List[Dict[str, Any]],
    *,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    client_name: Optional[str] = None,
) -> Dict[str, Any]:
    totals = _compute_totals(rows)
    normalized_client_name = _normalize_client_name(client_name)
    with get_session() as session:
        order = session.get(Order, order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        session.execute(delete(OrderRow).where(OrderRow.order_id == order_id))
        for row in rows:
            session.add(
                OrderRow(
                    order_id=order.id,
                    order_number=row.get("order_number", ""),
                    type=row.get("type") or "",
                    dimension=row.get("dimension") or "",
                    position=row.get("position") or "",
                    quantity=int(row.get("quantity") or 0),
                    area=float(row.get("area") or 0.0),
                )
            )

        order.units_total = totals["units"]
        order.area_total = totals["area"]
        order.order_numbers = totals["order_numbers"]
        if normalized_client_name and not _normalize_client_name(order.client_name):
            order.client_name = normalized_client_name
        if normalized_client_name and not _normalize_client_name(order.client_hint):
            order.client_hint = normalized_client_name
        if status:
            next_status = normalize_order_status(status, default=normalize_order_status(order.status))
            previous_status = normalize_order_status(order.status)
            order.status = next_status
            if previous_status != next_status:
                _record_status_event(
                    session,
                    order_id=order.id,
                    from_status=previous_status,
                    to_status=next_status,
                    note=notes,
                    reason="manual_update",
                )
        if notes is not None:
            order.notes = notes
        order.updated_at = datetime.now(timezone.utc)

        session.flush()
        return _serialize_order(order, include_rows=True)


def get_orders(
    query: Optional[str] = None,
    status: Optional[str] = None,
    client: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    approved_only: bool = False,
    year: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    limit = max(1, min(limit or 50, 200))
    offset = max(offset or 0, 0)

    now_year = datetime.now(timezone.utc).year
    year_text = (str(year).strip().lower() if year is not None else "")
    if not year_text:
        filter_year: Optional[int] = now_year
    elif year_text == "all":
        filter_year = None
    elif year_text.isdigit() and len(year_text) == 4:
        filter_year = int(year_text)
    else:
        raise ValueError("Invalid year. Use YYYY or 'all'.")

    from_dt: Optional[datetime] = None
    to_dt: Optional[datetime] = None
    if date_from:
        value = str(date_from).strip()
        try:
            from_dt = datetime.fromisoformat(value)
        except ValueError:
            try:
                from_dt = datetime.fromisoformat(f"{value}T00:00:00")
            except ValueError as exc:
                raise ValueError("Invalid date_from. Use ISO date like YYYY-MM-DD.") from exc
        if from_dt.tzinfo is None:
            from_dt = from_dt.replace(tzinfo=timezone.utc)
    if date_to:
        value = str(date_to).strip()
        try:
            to_dt = datetime.fromisoformat(value)
        except ValueError:
            try:
                to_dt = datetime.fromisoformat(f"{value}T00:00:00")
            except ValueError as exc:
                raise ValueError("Invalid date_to. Use ISO date like YYYY-MM-DD.") from exc
        if to_dt.tzinfo is None:
            to_dt = to_dt.replace(tzinfo=timezone.utc)
        if len(str(date_to).strip()) == 10:
            to_dt = to_dt + timedelta(days=1)

    with SessionLocal() as session:
        stmt = select(Order).order_by(Order.updated_at.desc(), Order.created_at.desc()).offset(offset).limit(limit)
        if query:
            like_term = f"%{query.lower()}%"
            stmt = stmt.where(
                func.lower(Order.order_numbers_raw).like(like_term)
                | func.lower(func.coalesce(Order.client_name, "")).like(like_term)
                | func.lower(func.coalesce(Order.client_hint, "")).like(like_term)
            )
        if client:
            client_like = f"%{str(client).strip().lower()}%"
            stmt = stmt.where(
                func.lower(func.coalesce(Order.client_name, "")).like(client_like)
                | func.lower(func.coalesce(Order.client_hint, "")).like(client_like)
            )
        if status:
            normalized_status = normalize_order_status(status, default="")
            if not normalized_status:
                raise ValueError(
                    f"Invalid status '{status}'. Allowed: {', '.join(ORDER_STATUS_SEQUENCE)}"
                )
            stmt = stmt.where(func.lower(Order.status) == normalized_status)
        elif approved_only:
            stmt = stmt.where(func.lower(Order.status) == "approved")
        if filter_year is not None:
            start = datetime(filter_year, 1, 1, tzinfo=timezone.utc)
            end = datetime(filter_year + 1, 1, 1, tzinfo=timezone.utc)
            stmt = stmt.where(Order.created_at >= start).where(Order.created_at < end)
        if from_dt is not None:
            stmt = stmt.where(Order.created_at >= from_dt)
        if to_dt is not None:
            stmt = stmt.where(Order.created_at < to_dt)
        orders = session.execute(stmt).scalars().all()
        return [_serialize_order(order, include_rows=False) for order in orders]


def get_orders_by_identifiers(identifiers: Sequence[str]) -> List[Dict[str, Any]]:
    cleaned: List[str] = []
    for ident in identifiers:
        if ident is None:
            continue
        value = str(ident).strip()
        if value:
            cleaned.append(value)
    if not cleaned:
        return []

    normalized = [normalize_order_number(value) for value in cleaned]
    numeric_ids = {int(value) for value in normalized if value.isdigit()}
    order_numbers = [value for value in normalized if value and not value.isdigit()]

    with SessionLocal() as session:
        stmt = select(Order).order_by(Order.created_at.desc())
        filters = []
        if numeric_ids:
            filters.append(Order.id.in_(numeric_ids))
        if order_numbers:
            for number in order_numbers:
                filters.append(func.lower(Order.order_numbers_raw).like(f"%{number.lower()}%"))
        if filters:
            stmt = stmt.where(or_(*filters))
        orders = session.execute(stmt).scalars().unique().all()
        for order in orders:
            _ = order.rows
        return [_serialize_order(order, include_rows=True) for order in orders]


def get_order_with_extraction(order_id: int) -> Optional[Dict[str, Any]]:
    with SessionLocal() as session:
        order = session.get(Order, order_id)
        if not order:
            return None
        _ = order.rows
        _ = order.extraction
        _ = order.status_events
        data = _serialize_order(order, include_rows=True)
        data["extraction"] = _serialize_extraction(order.extraction)
        data["status_history"] = _serialize_status_events(order.status_events or [])
        return data


def list_telegram_files(
    *,
    status: Optional[str] = None,
    query: Optional[str] = None,
    touched: Optional[bool] = False,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    normalized_status = (status or "").strip().lower()
    search = (query or "").strip().lower()
    safe_limit = max(1, min(int(limit or 100), 250))
    with get_session() as session:
        statement = select(TelegramFile)
        statement = statement.where(TelegramFile.deleted.is_(False))
        if touched is not None:
            statement = statement.where(TelegramFile.touched.is_(bool(touched)))
        if normalized_status:
            statement = statement.where(TelegramFile.extraction_status == normalized_status)
        if search:
            pattern = f"%{search}%"
            statement = statement.where(
                or_(
                    func.lower(TelegramFile.original_filename).like(pattern),
                    func.lower(TelegramFile.telegram_sender_name).like(pattern),
                    func.lower(TelegramFile.telegram_chat_id).like(pattern),
                )
            )
        statement = statement.order_by(TelegramFile.received_at.desc(), TelegramFile.id.desc()).limit(safe_limit)
        records = session.execute(
            statement
        ).scalars().all()
        order_ids = [record.linked_order_id for record in records if record.linked_order_id]
        orders_by_id: Dict[int, Order] = {}
        if order_ids:
            linked_orders = session.execute(select(Order).where(Order.id.in_(order_ids))).scalars().all()
            orders_by_id = {order.id: order for order in linked_orders}
        serialized = [_serialize_telegram_file(record, orders_by_id.get(record.linked_order_id or 0)) for record in records]
    return serialized


def count_untouched_telegram_files() -> int:
    with get_session() as session:
        return int(session.scalar(select(func.count()).select_from(TelegramFile).where(TelegramFile.deleted.is_(False), TelegramFile.touched.is_(False))) or 0)


def get_telegram_file_counts() -> Dict[str, int]:
    with get_session() as session:
        active_filter = TelegramFile.deleted.is_(False)
        untouched = int(session.scalar(select(func.count()).select_from(TelegramFile).where(active_filter, TelegramFile.touched.is_(False))) or 0)
        queued = int(session.scalar(select(func.count()).select_from(TelegramFile).where(active_filter, TelegramFile.extraction_status == "queued")) or 0)
        processing = int(session.scalar(select(func.count()).select_from(TelegramFile).where(active_filter, TelegramFile.extraction_status == "processing")) or 0)
        failed = int(session.scalar(select(func.count()).select_from(TelegramFile).where(active_filter, TelegramFile.extraction_status == "failed")) or 0)
        return {
            "untouched_count": untouched,
            "queued_count": queued,
            "processing_count": processing,
            "failed_count": failed,
        }


def find_telegram_file_record(
    *,
    telegram_chat_id: Optional[Any] = None,
    telegram_message_id: Optional[Any] = None,
    telegram_file_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    chat_value = str(telegram_chat_id) if telegram_chat_id is not None else None
    message_value = str(telegram_message_id) if telegram_message_id is not None else None
    file_value = str(telegram_file_id) if telegram_file_id else None
    with get_session() as session:
        query = select(TelegramFile)
        if chat_value is not None and message_value is not None:
            query = query.where(
                TelegramFile.telegram_chat_id == chat_value,
                TelegramFile.telegram_message_id == message_value,
            )
            if file_value:
                query = query.where(or_(TelegramFile.telegram_file_id == file_value, TelegramFile.telegram_file_id.is_(None)))
        elif file_value:
            query = query.where(TelegramFile.telegram_file_id == file_value)
        else:
            return None
        record = session.execute(query.order_by(TelegramFile.id.desc()).limit(1)).scalar_one_or_none()
        if not record:
            return None
        linked_order = session.get(Order, record.linked_order_id) if record.linked_order_id else None
        return _serialize_telegram_file(record, linked_order)


def list_unfinished_telegram_file_ids(*, stale_processing_before: datetime) -> List[int]:
    with get_session() as session:
        records = session.execute(
            select(TelegramFile.id).where(
                TelegramFile.deleted.is_(False),
                or_(
                    TelegramFile.extraction_status == "queued",
                    TelegramFile.extraction_status == "received",
                    (
                        (TelegramFile.extraction_status == "processing")
                        & (
                            (TelegramFile.processing_started_at.is_(None))
                            | (TelegramFile.processing_started_at < stale_processing_before)
                        )
                    ),
                )
            ).order_by(TelegramFile.queued_at.asc(), TelegramFile.received_at.asc(), TelegramFile.id.asc())
        ).all()
        return [int(row[0]) for row in records]


def get_telegram_file(file_id: int) -> Optional[Dict[str, Any]]:
    with get_session() as session:
        record = session.get(TelegramFile, int(file_id))
        if not record:
            return None
        linked_order = session.get(Order, record.linked_order_id) if record.linked_order_id else None
        return _serialize_telegram_file(record, linked_order)


def delete_order(order_id: int) -> bool:
    with get_session() as session:
        order = session.get(Order, order_id)
        if not order:
            return False
        session.delete(order)
        return True


def update_order_status(
    order_id: int,
    *,
    status: str,
    note: Optional[str] = None,
    reason: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    target_status = normalize_order_status(status, default="")
    if not target_status:
        raise ValueError(f"Invalid status '{status}'. Allowed: {', '.join(ORDER_STATUS_SEQUENCE)}")
    with get_session() as session:
        order = session.get(Order, order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")
        current_status = normalize_order_status(order.status)
        if not force and not _can_transition_status(current_status, target_status):
            raise ValueError(f"Invalid status transition: {current_status} -> {target_status}")
        if current_status != target_status:
            order.status = target_status
            order.updated_at = datetime.now(timezone.utc)
            if note and target_status == "approved":
                order.notes = note
            _record_status_event(
                session,
                order_id=order.id,
                from_status=current_status,
                to_status=target_status,
                note=note,
                reason=reason or "status_update",
            )
        session.flush()
        _ = order.rows
        _ = order.status_events
        data = _serialize_order(order, include_rows=True)
        data["status_history"] = _serialize_status_events(order.status_events or [])
        return data


def get_all_rows_for_export(
    *,
    query: Optional[str] = None,
    status: Optional[str] = "approved",
    client: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    approved_only: bool = False,
    year: Optional[str] = None,
) -> List[Dict[str, Any]]:
    now_year = datetime.now(timezone.utc).year
    year_text = (str(year).strip().lower() if year is not None else "")
    if not year_text:
        filter_year: Optional[int] = now_year
    elif year_text == "all":
        filter_year = None
    elif year_text.isdigit() and len(year_text) == 4:
        filter_year = int(year_text)
    else:
        raise ValueError("Invalid year. Use YYYY or 'all'.")

    from_dt: Optional[datetime] = None
    to_dt: Optional[datetime] = None
    if date_from:
        value = str(date_from).strip()
        try:
            from_dt = datetime.fromisoformat(value)
        except ValueError:
            try:
                from_dt = datetime.fromisoformat(f"{value}T00:00:00")
            except ValueError as exc:
                raise ValueError("Invalid date_from. Use ISO date like YYYY-MM-DD.") from exc
        if from_dt.tzinfo is None:
            from_dt = from_dt.replace(tzinfo=timezone.utc)
    if date_to:
        value = str(date_to).strip()
        try:
            to_dt = datetime.fromisoformat(value)
        except ValueError:
            try:
                to_dt = datetime.fromisoformat(f"{value}T00:00:00")
            except ValueError as exc:
                raise ValueError("Invalid date_to. Use ISO date like YYYY-MM-DD.") from exc
        if to_dt.tzinfo is None:
            to_dt = to_dt.replace(tzinfo=timezone.utc)
        if len(str(date_to).strip()) == 10:
            to_dt = to_dt + timedelta(days=1)

    with SessionLocal() as session:
        stmt = select(Order, OrderRow).join(OrderRow, Order.id == OrderRow.order_id)
        if status:
            normalized_status = normalize_order_status(status, default="")
            if not normalized_status:
                raise ValueError(
                    f"Invalid status '{status}'. Allowed: {', '.join(ORDER_STATUS_SEQUENCE)}"
                )
            stmt = stmt.where(func.lower(Order.status) == normalized_status)
        elif approved_only:
            stmt = stmt.where(func.lower(Order.status) == "approved")
        if query:
            like_term = f"%{query.lower()}%"
            stmt = stmt.where(
                func.lower(Order.order_numbers_raw).like(like_term)
                | func.lower(func.coalesce(Order.client_name, "")).like(like_term)
                | func.lower(func.coalesce(Order.client_hint, "")).like(like_term)
            )
        if client:
            client_like = f"%{str(client).strip().lower()}%"
            stmt = stmt.where(
                func.lower(func.coalesce(Order.client_name, "")).like(client_like)
                | func.lower(func.coalesce(Order.client_hint, "")).like(client_like)
            )
        if filter_year is not None:
            start = datetime(filter_year, 1, 1, tzinfo=timezone.utc)
            end = datetime(filter_year + 1, 1, 1, tzinfo=timezone.utc)
            stmt = stmt.where(Order.created_at >= start).where(Order.created_at < end)
        if from_dt is not None:
            stmt = stmt.where(Order.created_at >= from_dt)
        if to_dt is not None:
            stmt = stmt.where(Order.created_at < to_dt)
        stmt = stmt.order_by(Order.created_at.desc())
        result: List[Dict[str, Any]] = []
        for order, row in session.execute(stmt).all():
            client_name = _normalize_client_name(order.client_name, _client_name_from_extraction(order.extraction))
            result.append(
                {
                    "order_id": order.id,
                    "created_at": order.created_at.isoformat(),
                    "source": order.source,
                    "client_name": client_name,
                    "clientName": client_name,
                    "client": client_name or order.client_hint,
                    "client_hint": order.client_hint,
                    "order_numbers": order.order_numbers,
                    "units_total": order.units_total,
                    "area_total": order.area_total,
                    "row": {
                        "order_number": row.order_number,
                        "type": row.type,
                        "dimension": row.dimension,
                        "position": row.position,
                        "quantity": row.quantity,
                        "area": row.area,
                    },
                }
            )
        return result


def save_correction(
    *,
    before_json: str,
    after_json: str,
    prepared_text: str,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    signature = build_signature(prepared_text, [])
    pattern_hash = signature["pattern_hash"]
    with get_session() as session:
        correction = session.execute(
            select(Correction).where(Correction.pattern_hash == pattern_hash)
        ).scalar_one_or_none()

        if correction:
            correction.before_json = before_json
            correction.after_json = after_json
            correction.pattern_text = signature["pattern_text"]
            correction.notes = notes or correction.notes
            correction.hits += 1
            correction.last_used_at = datetime.now(timezone.utc)
        else:
            correction = Correction(
                pattern_hash=pattern_hash,
                pattern_text=signature["pattern_text"],
                before_json=before_json,
                after_json=after_json,
                notes=notes,
                hits=1,
                last_used_at=datetime.now(timezone.utc),
            )
            session.add(correction)

        session.flush()
        return {
            "id": correction.id,
            "pattern_hash": correction.pattern_hash,
            "hits": correction.hits,
        }


def find_similar_corrections(prepared_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    signature = build_signature(prepared_text or "")
    pattern_hash = signature["pattern_hash"]
    signature_tokens = set(signature["signature"].splitlines())

    with SessionLocal() as session:
        exact = session.execute(
            select(Correction).where(Correction.pattern_hash == pattern_hash)
        ).scalars().all()

        if len(exact) >= top_k:
            corrections = exact[:top_k]
        else:
            others = session.execute(select(Correction)).scalars().all()
            scored = []
            for corr in others:
                tokens = set((corr.pattern_text or "").splitlines())
                score = len(signature_tokens & tokens) + (corr.hits * 0.1)
                scored.append((score, corr))
            scored.sort(key=lambda item: item[0], reverse=True)
            corrections = exact + [item[1] for item in scored if item[1] not in exact]
            corrections = corrections[:top_k]

        output: List[Dict[str, Any]] = []
        for corr in corrections:
            output.append(
                {
                    "id": corr.id,
                    "pattern_hash": corr.pattern_hash,
                    "pattern_text": corr.pattern_text,
                    "before_json": corr.before_json,
                    "after_json": corr.after_json,
                    "notes": corr.notes,
                    "hits": corr.hits,
                }
            )
        return output


def list_corrections() -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        corrections = session.execute(
            select(Correction).order_by(Correction.hits.desc(), Correction.created_at.desc())
        ).scalars().all()
        return [
            {
                "id": corr.id,
                "pattern_hash": corr.pattern_hash,
                "pattern_text": corr.pattern_text,
                "notes": corr.notes,
                "hits": corr.hits,
            }
            for corr in corrections
        ]


def delete_correction(correction_id: int) -> bool:
    with get_session() as session:
        correction = session.get(Correction, correction_id)
        if not correction:
            return False
        session.delete(correction)
        return True


def bump_correction_hit(correction_id: int) -> None:
    with get_session() as session:
        correction = session.get(Correction, correction_id)
        if not correction:
            return
        correction.hits += 1
        correction.last_used_at = datetime.now(timezone.utc)


def record_workspace_action(
    *,
    actor: Optional[str],
    action_type: str,
    status: str,
    order_id: Optional[int] = None,
    order_number: Optional[str] = None,
    processing_batch_id: Optional[int] = None,
    requested_message: Optional[str] = None,
    tool_name: Optional[str] = None,
    input_json: Optional[Any] = None,
    output_json: Optional[Any] = None,
    requires_confirmation: bool = False,
    confirmed: bool = False,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    def _dump(value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

    with get_session() as session:
        action = WorkspaceAction(
            actor=(actor or "").strip() or None,
            action_type=(action_type or "workspace_action").strip(),
            order_id=order_id,
            order_number=(order_number or "").strip() or None,
            processing_batch_id=processing_batch_id,
            status=(status or "").strip(),
            requested_message=requested_message,
            tool_name=tool_name,
            input_json=_dump(input_json),
            output_json=_dump(output_json),
            requires_confirmation=bool(requires_confirmation),
            confirmed=bool(confirmed),
            error_message=error_message,
        )
        session.add(action)
        session.flush()
        return {
            "id": action.id,
            "created_at": action.created_at.isoformat(),
            "status": action.status,
        }


__all__ = [
    "init_db",
    "insert_extraction_with_rows",
    "create_telegram_file_record",
    "update_telegram_file_record",
    "touch_telegram_file_record",
    "soft_delete_telegram_file_record",
    "mark_telegram_file_labels_printed",
    "mark_telegram_file_linked_order_opened",
    "list_telegram_files",
    "count_untouched_telegram_files",
    "get_telegram_file_counts",
    "find_telegram_file_record",
    "list_unfinished_telegram_file_ids",
    "get_telegram_file",
    "update_order_rows",
    "get_orders",
    "get_orders_by_identifiers",
    "get_order_with_extraction",
    "delete_order",
    "get_all_rows_for_export",
    "save_correction",
    "find_similar_corrections",
    "list_corrections",
    "delete_correction",
    "bump_correction_hit",
    "record_workspace_action",
]
