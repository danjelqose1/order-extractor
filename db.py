from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
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
    source: Mapped[str] = mapped_column(String(20))
    client_hint: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    order_numbers_raw: Mapped[str] = mapped_column("order_numbers", Text, default="")
    units_total: Mapped[int] = mapped_column(Integer, default=0)
    area_total: Mapped[float] = mapped_column(Float, default=0.0)
    hash: Mapped[Optional[str]] = mapped_column(String(64), unique=True, nullable=True)
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


def _ensure_schema() -> None:
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        info = conn.execute(text("PRAGMA table_info(orders)")).fetchall()
        columns = {row[1] for row in info}
        if "status" not in columns:
            conn.execute(text("ALTER TABLE orders ADD COLUMN status TEXT DEFAULT 'draft'"))
        if "notes" not in columns:
            conn.execute(text("ALTER TABLE orders ADD COLUMN notes TEXT"))
        info_rows = conn.execute(text("PRAGMA table_info(orders)")).fetchall()
        columns = {row[1] for row in info_rows}
        if "order_numbers" in columns:
            conn.execute(text("PRAGMA foreign_keys=ON"))


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
    data: Dict[str, Any] = {
        "id": order.id,
        "created_at": order.created_at.isoformat(),
        "source": order.source,
        "client": order.client_hint or "",
        "client_hint": order.client_hint,
        "order_numbers": order.order_numbers,
        "units_total": order.units_total,
        "area_total": order.area_total,
        "hash": order.hash,
        "status": order.status,
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


def insert_extraction_with_rows(
    *,
    source: str,
    rows: List[Dict[str, Any]],
    raw_input: Optional[str],
    prepared_text: Optional[str],
    llm_output_json: str,
    model_used: Optional[str],
    hash_value: Optional[str],
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("rows are required to insert extraction")

    totals = _compute_totals(rows)
    client_hint = extract_client_hint(raw_input or prepared_text or "")

    with get_session() as session:
        order: Optional[Order] = None
        if hash_value:
            order = session.execute(select(Order).where(Order.hash == hash_value)).scalar_one_or_none()

        if not order:
            order = Order(
                source=source,
                client_hint=client_hint,
                hash=hash_value,
                status="draft",
            )
            session.add(order)
            session.flush()
        else:
            order.source = source or order.source
            order.client_hint = client_hint or order.client_hint
            order.status = "draft"

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
            "status": order.status,
            "order_numbers": order.order_numbers,
            "client_hint": order.client_hint,
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
            order.status = status
        if notes is not None:
            order.notes = notes

        session.flush()
        return _serialize_order(order, include_rows=True)


def get_orders(
    query: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    limit = max(1, min(limit or 50, 200))
    offset = max(offset or 0, 0)
    with SessionLocal() as session:
        stmt = select(Order).order_by(Order.created_at.desc()).offset(offset).limit(limit)
        if query:
            like_term = f"%{query.lower()}%"
            stmt = stmt.where(
                func.lower(Order.order_numbers_raw).like(like_term)
                | func.lower(func.coalesce(Order.client_hint, "")).like(like_term)
            )
        if status:
            stmt = stmt.where(func.lower(Order.status) == status.lower())
        orders = session.execute(stmt).scalars().all()
        return [_serialize_order(order, include_rows=False) for order in orders]


def get_order_with_extraction(order_id: int) -> Optional[Dict[str, Any]]:
    with SessionLocal() as session:
        order = session.get(Order, order_id)
        if not order:
            return None
        _ = order.rows
        _ = order.extraction
        data = _serialize_order(order, include_rows=True)
        data["extraction"] = _serialize_extraction(order.extraction)
        return data


def delete_order(order_id: int) -> bool:
    with get_session() as session:
        order = session.get(Order, order_id)
        if not order:
            return False
        session.delete(order)
        return True


def get_all_rows_for_export() -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        stmt = (
            select(Order, OrderRow)
            .join(OrderRow, Order.id == OrderRow.order_id)
            .where(func.lower(Order.status) == "approved")
            .order_by(Order.created_at.desc())
        )
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
    "get_order_with_extraction",
    "delete_order",
    "get_all_rows_for_export",
    "save_correction",
    "find_similar_corrections",
    "list_corrections",
    "delete_correction",
    "bump_correction_hit",
]
