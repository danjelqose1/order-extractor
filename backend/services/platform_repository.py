"""Persistence for shared workflow services. MCP handlers do not query storage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import uuid
from sqlalchemy import select, func
from time_utils import utc_isoformat, platform_year_utc_bounds, parse_platform_filter_datetime


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def version_of(order):
    # Existing PDF `version` means extraction revision, not mutation revision.
    # A content token also detects edits made through the legacy website.
    return hashlib.sha256(canonical(order).encode()).hexdigest()


def load_invoices(db, legacy_path: Path):
    with db.read_session() as session:
        doc = session.get(db.AppDocument, "invoices")
        if doc:
            return json.loads(doc.content_json)
    if legacy_path.exists():
        data = json.loads(legacy_path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
            raise ValueError("Invalid invoice store; repair before saving")
        return {"jobs": data["jobs"]}
    return {"jobs": []}


def save_invoices(db, data):
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        raise ValueError("Invoices payload must contain a jobs list")
    payload = {"jobs": data["jobs"]}
    with db.get_session() as session:
        doc = session.get(db.AppDocument, "invoices")
        if doc is None:
            doc = db.AppDocument(id="invoices", content_json=canonical(payload))
            session.add(doc)
        else:
            doc.content_json = canonical(payload)
    return payload


class Repository:
    def __init__(self, db):
        self.db = db

    def audit(self, actor, tool, order_ref, request_id, outcome):
        with self.db.get_session() as session:
            session.add(self.db.McpAudit(actor=actor, tool=tool, order_ref=order_ref,
                                         request_id=request_id, outcome=outcome))

    def replay(self, actor, tool, key):
        db = self.db
        with db.read_session() as session:
            return session.scalar(select(db.McpOperation).where(
                db.McpOperation.actor == actor, db.McpOperation.tool == tool,
                db.McpOperation.idempotency_key == key))

    def remember(self, actor, tool, key, request_hash, result):
        with self.db.get_session() as session:
            session.add(self.db.McpOperation(actor=actor, tool=tool, idempotency_key=key,
                                             request_hash=request_hash, result_json=canonical(result)))

    def duplicate_number(self, number, exclude_ref=None):
        db = self.db
        normalized = number.strip().casefold()
        with db.read_session() as session:
            manual = session.execute(select(db.ManualOrder.id, db.ManualOrder.order_number)).all()
            pdf = session.scalars(select(db.Order)).all()
            return any(f"manual:{i}" != exclude_ref and n.strip().casefold() == normalized for i, n in manual) or any(
                f"pdf:{o.id}" != exclude_ref and any(n.strip().casefold() == normalized for n in o.order_numbers) for o in pdf)

    def list_orders(self, filters):
        db = self.db
        matches = []
        with db.read_session() as session:
            for source, model in (("manual", db.ManualOrder), ("pdf", db.Order)):
                if filters.source not in ("all", source):
                    continue
                query = select(model.id, model.updated_at)
                if filters.client:
                    query = query.where(func.lower(model.client_name).contains(filters.client.lower(), autoescape=True))
                if filters.order_number:
                    field = model.order_number if source == "manual" else model.order_numbers_raw
                    query = query.where(func.lower(field).contains(filters.order_number.lower(), autoescape=True))
                if filters.status:
                    query = query.where(model.status == filters.status)
                if source == "manual":
                    if filters.year != "all":
                        query = query.where(model.order_date >= f"{filters.year:04d}-01-01", model.order_date < f"{filters.year+1:04d}-01-01")
                    if filters.date_from:
                        query = query.where(model.order_date >= filters.date_from.isoformat())
                    if filters.date_to:
                        query = query.where(model.order_date <= filters.date_to.isoformat())
                else:
                    if filters.year != "all":
                        start, end = platform_year_utc_bounds(filters.year)
                        query = query.where(model.created_at >= start, model.created_at < end)
                    if filters.date_from:
                        query = query.where(model.created_at >= parse_platform_filter_datetime(filters.date_from.isoformat()))
                    if filters.date_to:
                        query = query.where(model.created_at < parse_platform_filter_datetime(filters.date_to.isoformat(), end_exclusive=True))
                # Pagination is applied after combining stores; it must never hide matches.
                for order in session.execute(query):
                    matches.append((utc_isoformat(order.updated_at), source, order.id))
        matches.sort(reverse=True)
        return [f"{source}:{i}" for _, source, i in matches[filters.offset:filters.offset+filters.limit]], len(matches)

    def summary(self):
        db = self.db
        result = dict(draft_count=0, approved_count=0, processing_count=0, completed_count=0, pieces=0, area_m2=0.0)
        mapping = {"draft":"draft_count", "reviewed":"draft_count", "approved":"approved_count", "processing":"processing_count", "in_production":"processing_count", "finished":"completed_count", "completed":"completed_count"}
        with db.read_session() as session:
            for status, count, pieces, area in session.execute(select(db.Order.status, func.count(), func.sum(db.Order.units_total), func.sum(db.Order.area_total)).group_by(db.Order.status)):
                if status in mapping:
                    result[mapping[status]] += count
                    result["pieces"] += pieces or 0
                    result["area_m2"] += area or 0
            for status, count in session.execute(select(db.ManualOrder.status, func.count()).group_by(db.ManualOrder.status)):
                if status in mapping:
                    result[mapping[status]] += count
            totals = session.execute(select(func.sum(db.ManualOrderRow.quantity), func.sum(db.ManualOrderRow.final_area_m2)).join(db.ManualOrder).where(db.ManualOrder.status.in_(list(mapping)))).one()
            result["pieces"] += totals[0] or 0
            result["area_m2"] = round(result["area_m2"] + (totals[1] or 0), 3)
        return result

    def add_job(self, order_ref, order_version, snapshot, result):
        with self.db.get_session() as session:
            job = self.db.WorkflowJob(id="job:" + uuid.uuid4().hex, order_ref=order_ref,
                order_version=order_version, snapshot_json=canonical(snapshot), result_json=canonical(result))
            session.add(job)
            session.flush()
            return job

    def get_job(self, job_id):
        with self.db.read_session() as session:
            return session.get(self.db.WorkflowJob, job_id)

    def save_job(self, job, result, *, rounded=None, grouped=None):
        with self.db.get_session() as session:
            job = session.merge(job)
            if rounded is not None:
                job.rounded = rounded
            if grouped is not None:
                job.grouped = grouped
            job.version += 1
            job.result_json = canonical(result)
            session.flush()
            return job

    def add_artifact(self, order_ref, kind, content, media_type, job=None):
        with self.db.get_session() as session:
            artifact = self.db.WorkflowArtifact(id="artifact:"+uuid.uuid4().hex, order_ref=order_ref,
                job_id=job.id if job else None, job_version=job.version if job else None, kind=kind,
                content=content, sha256=hashlib.sha256(content).hexdigest(), media_type=media_type)
            session.add(artifact)
            session.flush()
            return self.artifact_view(artifact)

    @staticmethod
    def artifact_view(a, byte_count=None):
        return dict(artifact_id=a.id, order_id=a.order_ref, processing_job_id=a.job_id,
                    kind=a.kind, media_type=a.media_type, byte_count=len(a.content) if byte_count is None else byte_count, sha256=a.sha256,
                    created_at=utc_isoformat(a.created_at), download_path=f"/mcp/artifacts/{a.id}", job_version=a.job_version)

    def artifacts(self, order_ref, job_id=None):
        db = self.db
        with db.read_session() as session:
            from sqlalchemy.orm import defer
            query = select(db.WorkflowArtifact, func.length(db.WorkflowArtifact.content)).options(
                defer(db.WorkflowArtifact.content)).where(db.WorkflowArtifact.order_ref == order_ref)
            if job_id:
                query = query.where(db.WorkflowArtifact.job_id == job_id)
            artifacts = [self.artifact_view(a, size) for a, size in session.execute(query.order_by(db.WorkflowArtifact.created_at))]
        if not job_id and order_ref.startswith("pdf:"):
            from workspace_service import list_order_production_files
            for file in list_order_production_files(int(order_ref.split(":")[1])):
                content = self._legacy_content(file)
                if content is None:
                    continue
                digest = hashlib.sha256(content).hexdigest()
                artifact_id = f"legacy:{file['id']}:{digest}"
                artifacts.append(dict(artifact_id=artifact_id, order_id=order_ref,
                    processing_job_id=None, kind=file["file_type"], media_type="application/pdf",
                    byte_count=len(content), sha256=digest, created_at=file["created_at"],
                    download_path=f"/mcp/artifacts/{artifact_id}", job_version=None))
        return artifacts

    def _legacy_content(self, file):
        # Read only catalogue-backed PDFs inside the established production root.
        # Never expose stored absolute paths or follow links outside that root.
        if not file or file.get("status") != "ready" or file.get("file_type") not in {"processing_pdf", "labels_pdf"}:
            return None
        try:
            path = Path(file["file_path"]).resolve()
            path.relative_to((Path(self.db.DB_DIR) / "production-files").resolve())
            if not path.is_file() or path.stat().st_size > 32_000_000:
                return None
            content = path.read_bytes()
            return content if content.startswith(b"%PDF-") else None
        except (OSError, ValueError):
            return None

    def artifact_content(self, artifact_id):
        match = re.fullmatch(r"legacy:([1-9][0-9]*):([0-9a-f]{64})", artifact_id)
        if match:
            from workspace_service import get_production_file
            content = self._legacy_content(get_production_file(int(match[1])))
            if content is not None and hashlib.sha256(content).hexdigest() == match[2]:
                return content, "application/pdf"
            return None
        with self.db.read_session() as session:
            artifact = session.get(self.db.WorkflowArtifact, artifact_id)
            return (artifact.content, artifact.media_type) if artifact else None

    def set_manual_status(self, order_id, status):
        # Status mutation belongs in the existing application storage layer.
        return self.db.update_manual_order_status(order_id, status=status)
