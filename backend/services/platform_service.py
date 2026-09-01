"""Authorized workflow orchestration, shared by MCP and its protected HTTP surface."""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from sqlalchemy import text

from mcp_contracts import ManualDraft, OrderSummary
from services.platform_repository import Repository, canonical, version_of, load_invoices, save_invoices
from services.workflow_engine import run_workflow


class WorkflowError(Exception):
    def __init__(self, code, message, *, issues=None, retryable=False):
        super().__init__(message)
        self.code, self.message = code, message
        self.issues, self.retryable = issues or [], retryable


class PlatformService:
    def __init__(self, db, price_loader, invoices_path):
        self.db, self.repo = db, Repository(db)
        self.price_loader, self.invoices_path = price_loader, Path(invoices_path)

    def invoke(self, name, args, *, actor, request_id, write=False, output_model=None):
        payload = args.model_dump(mode="json")
        order_ref = payload.get("order_id")
        try:
            if not order_ref and payload.get("processing_job_id"):
                job = self.repo.get_job(payload["processing_job_id"])
                order_ref = job.order_ref if job else None
            if not write:
                result = getattr(self, name)(args)
                if output_model:
                    result = output_model.model_validate(result).model_dump(mode="json")
                self.repo.audit(actor, name, order_ref, request_id, "ok")
                return result
            request_hash = hashlib.sha256(canonical(payload).encode()).hexdigest()
            key = payload.get("idempotency_key")
            with self.db.atomic_workflow():
                if key:
                    previous = self.repo.replay(actor, name, key)
                    if previous:
                        if previous.request_hash != request_hash:
                            raise WorkflowError("IDEMPOTENCY_CONFLICT", "This key was used with different inputs. Use a new key for a different operation.")
                        self.repo.audit(actor, name, order_ref, request_id, "replayed")
                        return json.loads(previous.result_json)
                result = getattr(self, name)(args)
                # Validate the public contract before committing any mutation.
                if output_model:
                    result = output_model.model_validate(result).model_dump(mode="json")
                if key:
                    self.repo.remember(actor, name, key, request_hash, result)
                self.repo.audit(actor, name, order_ref or result.get("order_id"), request_id, "ok")
                return result
        except BaseException as exc:
            # Let the transport record one sanitized final error after rollback.
            exc.mcp_order_ref = order_ref
            raise

    def _snapshot(self, ref):
        source, record_id = ref.split(":")
        order = self.db.get_manual_order(int(record_id)) if source == "manual" else self.db.get_order_with_extraction(int(record_id))
        if not order:
            raise WorkflowError("NOT_FOUND", "Order not found.")
        return order

    def _versioned(self, args):
        order = self._snapshot(args.order_id)
        if version_of(order) != args.expected_version:
            raise WorkflowError("VERSION_CONFLICT", "Order changed. Read it again and review the current version.")
        return order

    def _manual_payload(self, draft):
        raw = draft.model_dump(mode="json", exclude={"idempotency_key"})
        issues, seen = [], set()
        for i, row in enumerate(draft.rows):
            if draft.mode == "client_positions_red_index" and row.red_index is None:
                issues.append(dict(field=f"rows[{i}].red_index", code="required", message="Red index is required in this mode."))
            if row.red_index is not None:
                if row.red_index in seen:
                    issues.append(dict(field=f"rows[{i}].red_index", code="duplicate", message="Red index must be unique within this order."))
                seen.add(row.red_index)
        if issues:
            raise WorkflowError("VALIDATION_ERROR", "Correct the indicated rows.", issues=issues)
        return dict(client_name=draft.client_name, order_number=draft.order_number, order_date=draft.order_date.isoformat(),
                    notes=draft.reference_notes, manual_format=draft.mode, status="draft", raw_values=raw,
                    rows=[dict(position=r.position, section=r.section, client_position=r.client_position,
                               index_number=r.red_index, width_mm=r.width_mm, height_mm=r.height_mm,
                               quantity=r.quantity, glass_type=r.glass_type, notes=r.row_notes,
                               area_override_m2=r.area_override_m2) for r in draft.rows])

    def _validate_saved(self, ref, order):
        if ref.startswith("manual:"):
            self._manual_payload(ManualDraft(client_name=order["client_name"], order_number=order["order_number"],
                order_date=order["order_date"], mode=order["manual_format"], rows=[dict(
                    position=r["position"], section=r["section"], client_position=r["client_position"],
                    red_index=r["index_number"], width_mm=r["width_mm"], height_mm=r["height_mm"],
                    quantity=r["quantity"], glass_type=r["glass_type"], area_override_m2=r["area_override_m2"]
                ) for r in order["rows"]]))
        else:
            from workspace_service import _validate_order
            validation = _validate_order(order)
            if not order.get("client_name") or not order.get("rows") or validation["blocker_warnings"]:
                raise WorkflowError("VALIDATION_ERROR", "Order needs review before this action.", issues=[dict(field="rows", code="review_required", message=m) for m in validation["blocker_warnings"]])
            for original, checked in zip(order["rows"], validation["rows"]):
                if any(original.get(k) != checked.get(k) for k in ("quantity", "dimension", "area", "type")):
                    raise WorkflowError("REVIEW_REQUIRED", "Existing validators propose a correction. Review it in the platform; this tool will not silently rewrite saved values.")

    def _view(self, ref, order=None):
        order = order or self._snapshot(ref)
        manual = ref.startswith("manual:")
        rows = deepcopy(order.get("rows", []))
        warnings = []
        for i, row in enumerate(rows):
            if row.get("quantity", 0) > 100:
                warnings.append(f"Row {i+1}: high quantity {row['quantity']} preserved exactly.")
            if manual:
                row.update(red_index=row["index_number"], row_notes=row["notes"])
        raw = order.get("raw_values")
        return dict(order_id=ref, source=order.get("source", "pdf"), storage_source="manual" if manual else "pdf",
            order_number=order.get("order_number") or ", ".join(order.get("order_numbers", [])),
            client_name=order.get("client_name") or "", order_date=order.get("order_date") or order.get("created_at", "")[:10],
            status=order["status"], version=version_of(order), row_count=len(rows),
            piece_count=order["total_quantity"] if manual else order["units_total"],
            calculated_area_m2=round(sum(r["calculated_area_m2"] for r in rows), 3) if manual else order["area_total"],
            total_area_m2=order["total_area_m2"] if manual else order["area_total"],
            mode=order.get("manual_format", "standard"), dimension_unit=(raw or {}).get("dimension_unit", "mm"),
            reference_notes=order.get("notes") or "", rows=rows, raw_values=raw, warnings=warnings,
            artifacts=self.repo.artifacts(ref))

    def platform_health(self, args):
        with self.db.read_session() as session:
            session.execute(text("SELECT 1"))
        storage = Path(self.db.DB_DIR)
        node = os.getenv("ORDER_EXTRACTOR_NODE_BINARY", "node")
        worker_modules = Path(__file__).resolve().parents[1] / "workflow_runtime/node_modules/pdf-lib"
        runtime_ready = bool(shutil.which(node)) and worker_modules.is_dir()
        return dict(backend="ready", database="ready", storage="ready" if os.access(storage, os.W_OK) else "read_only",
                    mcp="ready", workflow_runtime="ready" if runtime_ready else "unavailable",
                    durable_storage_configured=os.getenv("ORDER_EXTRACTOR_MCP_DURABLE_STORAGE") == "true")

    def list_orders(self, args):
        refs, total = self.repo.list_orders(args)
        fields = OrderSummary.model_fields
        items = [{k:v for k,v in self._view(ref).items() if k in fields} for ref in refs]
        return dict(items=items, total=total, limit=args.limit,
                    offset=args.offset, has_more=args.offset + len(refs) < total)

    list_manual_orders = list_orders

    def get_order(self, args):
        return self._view(args.order_id)

    get_manual_order = get_order

    def get_platform_summary(self, args):
        return self.repo.summary()

    def list_order_artifacts(self, args):
        self._snapshot(args.order_id)
        return {"artifacts": self.repo.artifacts(args.order_id)}

    def create_manual_order_draft(self, args):
        payload = self._manual_payload(args)
        if self.repo.duplicate_number(args.order_number):
            raise WorkflowError("DUPLICATE_ORDER_NUMBER", "This order number already exists. Choose a distinct number.")
        saved = self.db.create_manual_order(payload)
        return self._view(f"manual:{saved['id']}")

    def update_manual_order_draft(self, args):
        order = self._versioned(args)
        if order["status"] != "draft":
            raise WorkflowError("ORDER_PROTECTED", "Only Draft manual orders may be edited.")
        payload = self._manual_payload(args.replacement)
        if self.repo.duplicate_number(args.replacement.order_number, args.order_id):
            raise WorkflowError("DUPLICATE_ORDER_NUMBER", "This order number already exists.")
        self.db.update_manual_order(order["id"], payload)
        return self._view(args.order_id)

    def approve_order(self, args):
        order = self._versioned(args)
        if order["status"] not in ({"draft"} if args.order_id.startswith("manual:") else {"draft", "reviewed"}):
            raise WorkflowError("ORDER_PROTECTED", "Only a draft/reviewed order can be approved.")
        self._validate_saved(args.order_id, order)
        if args.order_id.startswith("manual:"):
            self.db.update_manual_order_status(order["id"], status="approved")
        else:
            self.db.update_order_status(order["id"], status="approved", reason="mcp_approval")
        return self._view(args.order_id)

    def send_order_to_processing(self, args):
        order = self._versioned(args)
        if order["status"] != "approved":
            raise WorkflowError("ORDER_PROTECTED", "Only an Approved order can start a new processing job.")
        self._validate_saved(args.order_id, order)
        result = run_workflow("preview", order=order, rounded=False, grouped=False)
        if args.order_id.startswith("manual:"):
            self.db.send_manual_order_to_processing(order["id"])
        else:
            self.db.update_order_status(order["id"], status="in_production", reason="mcp_processing")
        current = self._snapshot(args.order_id)
        job = self.repo.add_job(args.order_id, version_of(current), current, result)
        return self._job_view(job)

    def _job(self, args, check_version=False):
        job = self.repo.get_job(args.processing_job_id)
        if not job:
            raise WorkflowError("NOT_FOUND", "Processing job not found.")
        if check_version:
            if job.version != args.expected_version:
                raise WorkflowError("VERSION_CONFLICT", "Processing job changed. Read its current version.")
            if version_of(self._snapshot(job.order_ref)) != job.order_version:
                raise WorkflowError("ORDER_CHANGED", "Source order changed after the job was created. Review it before generating documents.")
        return job

    def _job_view(self, job):
        result = json.loads(job.result_json)
        snapshot = json.loads(job.snapshot_json)
        warnings = self._view(job.order_ref, snapshot)["warnings"]
        if job.rounded:
            warnings += [f"Row {i+1}: fractional dimensions are unchanged by the established Danko rule."
                         for i, row in enumerate(result["rows"]) if "." in row.get("__original", {}).get("dimension", "")]
        return dict(processing_job_id=job.id, order_id=job.order_ref, order_version=job.order_version,
            version=job.version, state="processing", rounding_applied=job.rounded, grouped=job.grouped,
            original_rows=snapshot["rows"], rows=result["rows"], groups=result["preview"]["groups"],
            warnings=warnings, artifacts=self.repo.artifacts(job.order_ref, job.id))

    def get_processing_job(self, args):
        return self._job_view(self._job(args))

    def apply_danko_rounding(self, args):
        job = self._job(args, True)
        if not job.rounded:
            result = run_workflow("preview", order=json.loads(job.snapshot_json), rounded=True, grouped=job.grouped)
            job = self.repo.save_job(job, result, rounded=True)
        return self._job_view(job)

    def group_processing_dimensions(self, args):
        job = self._job(args, True)
        if not job.grouped:
            result = run_workflow("preview", order=json.loads(job.snapshot_json), rounded=job.rounded, grouped=True)
            job = self.repo.save_job(job, result, grouped=True)
        return self._job_view(job)

    def _generate(self, args, kind):
        job = self._job(args, True)
        snapshot = json.loads(job.snapshot_json)
        # Quantity is never reduced: reject expensive label jobs as a whole.
        if kind == "labels_pdf" and sum(r["quantity"] for r in snapshot["rows"]) > int(os.getenv("ORDER_EXTRACTOR_MCP_MAX_LABELS", "2000")):
            raise WorkflowError("GENERATION_LIMIT", "This job exceeds the configured label limit. No quantities or orders were changed.")
        result = json.loads(job.result_json)
        if snapshot.get("manual_format") == "client_positions_red_index":
            # This factory format has its own existing Python layout, including red indexes.
            content = self._manual_document(snapshot, result["rows"], kind)
        else:
            content = base64.b64decode(run_workflow(kind, preview=result["preview"])["pdf_base64"], validate=True)
        if not content.startswith(b"%PDF-"):
            raise WorkflowError("GENERATION_FAILED", "Generator did not return a PDF.")
        artifact = self.repo.add_artifact(job.order_ref, kind, content, "application/pdf", job)
        return {"artifacts": [artifact]}

    def _manual_document(self, snapshot, rows, kind):
        # A subprocess enforces a hard timeout even for ReportLab and large label jobs.
        import sys
        payload = dict(order=deepcopy(snapshot), settings=self.db.get_manual_print_settings(), kind=kind)
        for target, processed in zip(payload["order"]["rows"], rows):
            target["width_mm"], target["height_mm"] = processed["width"], processed["height"]
        worker = Path(__file__).with_name("manual_document_worker.py")
        completed = subprocess.run([sys.executable, str(worker)], input=canonical(payload), text=True,
            capture_output=True, check=True, timeout=min(60, max(1, int(os.getenv("ORDER_EXTRACTOR_MCP_TIMEOUT_SECONDS", "30")))),
            env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(worker.parents[1])})
        return base64.b64decode(completed.stdout, validate=True)

    def generate_processing_sheet(self, args):
        return self._generate(args, "processing_pdf")

    def generate_labels(self, args):
        return self._generate(args, "labels_pdf")

    def create_invoice_draft(self, args):
        order = self._versioned(args)
        if order["status"] in {"cancelled", "archived"}:
            raise WorkflowError("ORDER_PROTECTED", "Cancelled or archived orders cannot be invoiced through this connection.")
        self._validate_saved(args.order_id, order)
        invoice = run_workflow("invoice", order=order, price_config=self.price_loader())
        invoice["id"] = "inv-mcp-" + uuid.uuid4().hex
        invoice["status"] = "draft"
        invoice["platformOrderId"] = args.order_id
        safe = all(not line.get("missingPrice") and (line.get("pricingUnderstanding") or {}).get("safeToPrice") for line in invoice["calculated"]["lines"])
        warnings = [] if safe else ["Draft contains unresolved prices. Review the catalog and composition; finalization and sending are unavailable here."]
        store = load_invoices(self.db, self.invoices_path)
        store["jobs"].append(invoice)
        save_invoices(self.db, store)
        artifact = self.repo.add_artifact(args.order_id, "invoice_draft", canonical(invoice).encode(), "application/json")
        return dict(invoice_id=invoice["id"], order_id=args.order_id, status="draft", currency="ALL",
                    safe_to_price=bool(safe), invoice=invoice, warnings=warnings, artifact=artifact)

    def delete_manual_order_draft(self, args):
        order = self._versioned(args)
        if order["status"] != "draft":
            raise WorkflowError("ORDER_PROTECTED", "Only Draft manual orders may be deleted.")
        self.db.delete_manual_order(order["id"])
        return dict(order_id=args.order_id, deleted=True)
