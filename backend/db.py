from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from sqlalchemy import (
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
    client_hint: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    order_numbers_raw: Mapped[str] = mapped_column("order_numbers", Text, default="")
    units_total: Mapped[int] = mapped_column(Integer, default=0)
    area_total: Mapped[float] = mapped_column(Float, default=0.0)
    hash: Mapped[Optional[str]] = mapped_column(String(64), unique=True, nullable=True)
    source_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_order_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
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


def _ensure_schema() -> None:
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        info = conn.execute(text("PRAGMA table_info(orders)")).fetchall()
        columns = {row[1] for row in info}
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
        info_rows = conn.execute(text("PRAGMA table_info(orders)")).fetchall()
        columns = {row[1] for row in info_rows}
        if "order_numbers" in columns:
            conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.execute(text("UPDATE orders SET updated_at = created_at WHERE updated_at IS NULL"))
        conn.execute(text("UPDATE orders SET source_hash = hash WHERE source_hash IS NULL AND hash IS NOT NULL"))
        conn.execute(text("UPDATE orders SET version = 1 WHERE version IS NULL OR version < 1"))
        conn.execute(text("UPDATE orders SET status = 'draft' WHERE status IS NULL OR TRIM(status) = ''"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_orders_source_hash ON orders(source_hash)"))


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
    data: Dict[str, Any] = {
        "id": order.id,
        "created_at": order.created_at.isoformat(),
        "updated_at": (order.updated_at or order.created_at).isoformat(),
        "source": order.source,
        "client": order.client_hint or "",
        "client_hint": order.client_hint,
        "order_numbers": order.order_numbers,
        "units_total": order.units_total,
        "area_total": order.area_total,
        "hash": order.hash,
        "source_hash": order.source_hash or order.hash,
        "version": int(order.version or 1),
        "parent_order_id": order.parent_order_id,
        "confidence": order.confidence,
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
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("rows are required to insert extraction")

    totals = _compute_totals(rows)
    client_hint = extract_client_hint(raw_input or prepared_text or "")
    canonical_hash = (hash_value or "").strip() or None

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
                client_hint=client_hint,
                hash=canonical_hash,
                source_hash=canonical_hash,
                version=1,
                status="draft",
                confidence=confidence,
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
                    client_hint=client_hint or "",
                    hash=versioned_hash or None,
                    source_hash=canonical_hash or (order.source_hash or order.hash),
                    version=next_version,
                    parent_order_id=protected_order_id,
                    status="draft",
                    confidence=confidence,
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
                order.client_hint = client_hint or order.client_hint
                order.status = "draft"
                order.source_hash = canonical_hash or order.source_hash or order.hash
                order.version = int(order.version or 1)
                if confidence is not None:
                    order.confidence = confidence
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
        order.client_hint = client_hint or order.client_hint
        if confidence is not None:
            order.confidence = confidence
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
            "client_hint": order.client_hint,
            "version": int(order.version or 1),
            "source_hash": order.source_hash or order.hash,
            "created_new_version": created_new_version,
            "protected_order_id": protected_order_id,
        }


def update_order_rows(
    order_id: int,
    rows: List[Dict[str, Any]],
    *,
    status: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    totals = _compute_totals(rows)
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
                | func.lower(func.coalesce(Order.client_hint, "")).like(like_term)
            )
        if client:
            client_like = f"%{str(client).strip().lower()}%"
            stmt = stmt.where(func.lower(func.coalesce(Order.client_hint, "")).like(client_like))
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
                | func.lower(func.coalesce(Order.client_hint, "")).like(like_term)
            )
        if client:
            client_like = f"%{str(client).strip().lower()}%"
            stmt = stmt.where(func.lower(func.coalesce(Order.client_hint, "")).like(client_like))
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
            result.append(
                {
                    "order_id": order.id,
                    "created_at": order.created_at.isoformat(),
                    "source": order.source,
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


__all__ = [
    "init_db",
    "insert_extraction_with_rows",
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
]
