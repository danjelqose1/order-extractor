"""Persistence and HTTP boundary for the advisory engine, isolated from order writes."""
import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Literal
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, MetaData, String, Table, Text, insert, select
import db
import ai_decisions as engine

metadata = MetaData()
decisions = Table('ai_decisions', metadata, Column('id', Integer, primary_key=True), Column('fingerprint', String, unique=True),
                  Column('order_id', Integer, index=True), Column('kind', String), Column('result', Text), Column('created_at', String))
events = Table('ai_decision_events', metadata, Column('id', Integer, primary_key=True), Column('decision_id', Integer, index=True),
               Column('action', String), Column('reason', Text), Column('created_at', String))
_lock = threading.Lock()

def initialize():
    metadata.create_all(db.engine)

def now():
    return datetime.now(timezone.utc).isoformat()

def get_record(order_id, kind):
    order = db.get_manual_order(order_id) if kind == 'manual' else db.get_order_with_extraction(order_id)
    if not order:
        raise HTTPException(404, 'Order not found')
    if kind == 'order':
        extraction = order.pop('extraction', {}) or {}
        try:
            parsed = json.loads(extraction.get('llm_output_json') or '{}')
            order['warnings'] = parsed.get('warnings') or []
            meta = parsed.get('_meta', {})
            if meta.get('scanned_image_only'):
                order['warnings'].append('Scanned original needs visual verification')
        except (ValueError, TypeError):
            order['warnings'] = ['Extraction evidence unavailable']
    order['kind'] = kind
    return order

def candidates(order):
    # Local bounded retrieval by client and order number, including configured client aliases.
    cfg = engine.config()
    normalized_client = engine.normalize(order, cfg)['client']
    clients = {order.get('client_name', '')}
    clients.update(k for k, v in cfg['aliases'].items() if engine.text(v) == normalized_client)
    clients.add(normalized_client)
    records = {}
    for client in clients:
        if not client:
            continue
        for item in db.get_orders(client=client, year='all', limit=100):
            records[('order', item['id'])] = item
        for item in db.list_manual_orders(query=client, limit=100):
            records[('manual', item['id'])] = item
    for number in order.get('order_numbers', []) or [order.get('order_number')]:
        if number:
            for item in db.get_orders(query=number, year='all', limit=100):
                records[('order', item['id'])] = item
    # Source hashes also catch resubmissions with missing client metadata.
    if order.get('source_hash'):
        with db.SessionLocal() as session:
            ids = session.scalars(select(db.Order.id).where(db.Order.source_hash == order['source_hash']).limit(20)).all()
        for oid in ids:
            records[('order', oid)] = {'id': oid}
    return [get_record(oid, kind) for kind, oid in records if (kind, oid) != (order['kind'], order['id'])]

def evaluate_saved(order_id, kind='order'):
    if not engine.enabled():
        return None
    # Serializes same-process evaluations; unique fingerprint also protects persistence across workers.
    with _lock:
        initialize()
        order = get_record(order_id, kind)
        existing = candidates(order)
        payload = {'order': engine.normalize(order, engine.config()), 'candidates': [engine.normalize(c, engine.config()) for c in existing], 'config': engine.config(), 'warnings': order.get('warnings', [])}
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        with db.engine.begin() as conn:
            old = conn.execute(select(decisions).where(decisions.c.fingerprint == fingerprint)).mappings().first()
        if old:
            cached = json.loads(old['result'])
            if not cached.get('api_error'):
                return {**cached, 'id': old['id']}
            # Retry failed evaluations after a bounded cooldown, retaining previous audit rows.
            import time
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(old['created_at'])).total_seconds()
            if age < 60:
                return {**cached, 'id': old['id']}
            fingerprint = hashlib.sha256((fingerprint + str(int(time.time() // 60))).encode()).hexdigest()
        result = engine.evaluate(order, existing)
        result['input_snapshot'] = engine.normalize(order, cfg=engine.config())
        from sqlalchemy.exc import IntegrityError
        try:
            with db.engine.begin() as conn:
                ident = conn.execute(insert(decisions).values(fingerprint=fingerprint, order_id=order_id, kind=kind, result=json.dumps(result), created_at=now())).inserted_primary_key[0]
        except IntegrityError:
            with db.engine.connect() as conn:
                ident = conn.execute(select(decisions.c.id).where(decisions.c.fingerprint == fingerprint)).scalar_one()
        return {**result, 'id': ident}

def install(app, key_provider):
    def authorize(x_app_key: str = Header(default='')):
        import secrets
        key = key_provider()
        if key and not secrets.compare_digest(x_app_key, key):
            raise HTTPException(401, 'Invalid app key')
        if not engine.enabled():
            raise HTTPException(404, 'AI Decisions disabled')
    router = APIRouter(prefix='/api/ai-decisions', dependencies=[Depends(authorize)])

    @router.get('')
    def listing():
        initialize()
        with db.engine.connect() as conn:
            rows = conn.execute(select(decisions).order_by(decisions.c.id.desc())).mappings().all()
            audit = conn.execute(select(events).order_by(events.c.id)).mappings().all()
        latest = {}
        for row in rows:
            latest.setdefault((row['kind'], row['order_id']), {**json.loads(row['result']), 'id': row['id'], 'created_at': row['created_at']})
        items = list(latest.values())
        for item in items:
            item['audit'] = [dict(e) for e in audit if e['decision_id'] == item['id']]
        return {'items': items[:200], 'summary': {'evaluated': len(items), 'exact_duplicates': sum(i['classification'] == 'exact_duplicate' for i in items),
                'possible_duplicates': sum(i['classification'] in ('possible_duplicate', 'probable_duplicate') for i in items),
                **{state: sum(i['readiness'] == state for i in items) for state in ('Ready', 'Review Required', 'Blocked')}}}

    @router.post('/orders/{order_id}/evaluate')
    def evaluate(order_id: int, kind: Literal['order', 'manual'] = 'order'):
        return evaluate_saved(order_id, kind)

    class Override(BaseModel):
        action: Literal['confirm_duplicate', 'legitimate', 'review', 'accept', 'ignore', 'reset']
        reason: str = Field(default='', max_length=2000)

    @router.post('/{decision_id}/override')
    def override(decision_id: int, payload: Override):
        initialize()
        with db.engine.begin() as conn:
            row = conn.execute(select(decisions).where(decisions.c.id == decision_id)).mappings().first()
            if not row:
                raise HTTPException(404, 'Decision not found')
            # Reject accepting evidence after the underlying order changes.
            saved = json.loads(row['result'])
            current = get_record(row['order_id'], row['kind'])
            if payload.action in ('accept', 'confirm_duplicate', 'legitimate', 'review') and engine.normalize(current, saved['config']) != saved.get('input_snapshot'):
                raise HTTPException(409, 'Reevaluate changed order')
            conn.execute(insert(events).values(decision_id=decision_id, action=payload.action, reason=payload.reason, created_at=now()))
        return {'ok': True, 'action': payload.action, 'order_modified': False}
    app.include_router(router)


def safely_evaluate(order_id, kind='order'):
    try:
        return evaluate_saved(order_id, kind)
    except Exception:
        import logging
        logging.getLogger(__name__).warning('AI Decisions unavailable; manual workflow continues')
        return None
