"""Independent delivery documents; no order/customer dependencies."""
import json
import uuid
from datetime import date
import math
import re
from flete_dalje_schema import client_key
from decimal import Decimal
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select, update, delete
from sqlalchemy.exc import IntegrityError
import db

router = APIRouter(prefix='/api/flete-dalje', tags=['Fletë-Dalje'])


def reference_number(value):
    """The original HTML's number() parser, including comma decimal notation."""
    text = re.sub(r'\s', '', str(value or ''))
    if ',' in text:
        text = text.replace('.', '').replace(',', '.', 1)
    try:
        result = float(text)
    except ValueError:
        return 0.0
    return result


def line_value(line):
    value = reference_number(line.quantity) * reference_number(line.price)
    # Math.round ties toward positive infinity (including negative half values).
    floor = math.floor(value)
    return floor + int(value - floor >= 0.5)


class Line(BaseModel):
    description: str = Field(default='', max_length=1000)
    unit: str = Field(default='m²', max_length=40)
    quantity: str = Field(default='', max_length=100)
    price: str = Field(default='', max_length=100)

    @field_validator('quantity', 'price', mode='before')
    @classmethod
    def numeric_text(cls, value):
        text = str(value if value is not None else '')
        number = reference_number(text)
        if not math.isfinite(number) or abs(number) > 1000000000:
            raise ValueError('Vlerë jashtë kufirit të lejuar.')
        return text


class Document(BaseModel):
    number: int = Field(ge=1, le=2147483646)
    recipient: str = Field(default='', max_length=1000)
    date: date
    address: str = Field(default='', max_length=2000)
    notes: str = Field(default='', max_length=5000)
    lines: list[Line] = Field(min_length=1, max_length=500)


class SavedDocument(Document):
    id: str = Field(min_length=1, max_length=80, pattern=r'^[a-zA-Z0-9-]+$')
    version: int = Field(ge=1)


class Payment(BaseModel):
    id: str = Field(min_length=1, max_length=80, pattern=r'^[a-zA-Z0-9-]+$')
    recipient: str = Field(min_length=1, max_length=1000)
    date: date
    amount: Decimal = Field(gt=0, le=1000000000, max_digits=12, decimal_places=2)
    method: Literal['cash', 'bank', 'other'] = 'cash'
    notes: str = Field(default='', max_length=2000)

    @field_validator('recipient')
    @classmethod
    def named_client(cls, value):
        value = ' '.join(value.split())
        if not client_key(value):
            raise ValueError('Zgjidhni klientin.')
        return value


class SavedPayment(Payment):
    version: int = Field(ge=1)
    voided: bool = False
    created_at: datetime


def payment_content(payment):
    return json.dumps(payment.model_dump(mode='json', exclude={'id', 'version', 'voided', 'created_at'}),
                      ensure_ascii=False, sort_keys=True)


def serialize_payment(row):
    return {**json.loads(row.content_json), 'id': row.id, 'version': row.version, 'voided': row.voided}


class OpeningBalance(BaseModel):
    recipient: str = Field(min_length=1, max_length=1000)
    amount: Decimal = Field(ge=0, le=1000000000, max_digits=12, decimal_places=2)
    side: Literal['debt', 'credit'] = 'debt'
    date: date
    notes: str = Field(default='', max_length=2000)
    version: int = Field(default=0, ge=0)

    @field_validator('recipient')
    @classmethod
    def named_client(cls, value):
        value = ' '.join(value.split())
        if not client_key(value):
            raise ValueError('Shkruani emrin e klientit.')
        return value


class SavedOpeningBalance(OpeningBalance):
    version: int = Field(ge=1)
    updated_at: datetime


def serialize_opening(row):
    return {**json.loads(row.content_json), 'version': row.version}


def client_cards(documents, payments, openings=()):
    cards = {}
    def card(name):
        key = client_key(name)
        if key not in cards:
            cards[key] = {'key': key, 'name': ' '.join(name.split()) or 'Pa emër',
                          'opening_cents': 0, 'opening_balance': None, 'documents_cents': 0, 'paid_cents': 0, 'balance_cents': 0, 'entries': []}
        return cards[key]
    for opening in openings:
        entry = card(opening['recipient'])
        amount = int(Decimal(opening['amount']) * 100) * (-1 if opening['side']=='credit' else 1)
        entry['opening_cents'] = amount
        entry['opening_balance'] = opening
        entry['entries'].append({'id': entry['key'], 'kind': 'opening', 'date': opening['date'],
                                 'notes': opening['notes'], 'debit_cents': max(amount, 0),
                                 'credit_cents': max(-amount, 0)})
    for doc in documents:
        entry = card(doc['recipient'])
        cents = int(doc['total']) * 100
        entry['documents_cents'] += cents
        entry['entries'].append({'id': doc['id'], 'kind': 'document', 'date': doc['date'],
                                 'number': doc['number'], 'address': doc['address'], 'debit_cents': cents,
                                 'credit_cents': 0})
    for payment in payments:
        entry = card(payment['recipient'])
        amount = int(Decimal(payment['amount']) * 100)
        if not payment['voided']:
            entry['paid_cents'] += amount
        entry['entries'].append({'id': payment['id'], 'kind': 'payment', 'date': payment['date'],
                                 'amount_cents': amount, 'method': payment['method'], 'notes': payment['notes'],
                                 'version': payment['version'], 'voided': payment['voided'],
                                 'debit_cents': 0, 'credit_cents': 0 if payment['voided'] else amount})
    for entry in cards.values():
        entry['entries'].sort(key=lambda row: (row['kind']!='opening', row['date'], row['kind'], row.get('number', 0), row['id']))
        balance = 0
        for row in entry['entries']:
            balance += row['debit_cents'] - row['credit_cents']
            row['balance_cents'] = balance
        entry['balance_cents'] = balance
    return sorted(cards.values(), key=lambda row: row['key'])


class Backup(BaseModel):
    format: str = Field(pattern=r'^flete-dalje-v1$')
    documents: list[SavedDocument] = Field(max_length=10000)
    payments: list[SavedPayment] = Field(default_factory=list, max_length=10000)
    opening_balances: list[SavedOpeningBalance] = Field(default_factory=list, max_length=10000)

    @model_validator(mode='before')
    @classmethod
    def reference_backup(cls, data):
        # Accept both native backups and the supplied HTML's schema:1 / raw array.
        if isinstance(data, dict) and 'format' in data:
            return data
        records = data if isinstance(data, list) else data.get('records') if isinstance(data, dict) else None
        if not isinstance(records, list):
            raise ValueError('Formati i backup nuk përputhet.')
        documents = []
        for row in records:
            if not isinstance(row, dict) or not isinstance(row.get('lines'), list):
                raise ValueError('Fletë e pavlefshme në backup.')
            if not isinstance(row.get('client', ''), str):
                raise ValueError('Emër klienti i pavlefshëm në backup.')
            documents.append({
                'id': row.get('id', f"legacy-{uuid.uuid5(uuid.NAMESPACE_URL, client_key(row.get('client', '')) + ':' + str(row.get('num')))}"), 'version': row.get('version', 1),
                'number': row.get('num'), 'recipient': row.get('client', ''),
                'date': row.get('date'), 'address': row.get('address', ''), 'notes': row.get('notes', ''),
                'lines': [{'description': line.get('name', ''), 'unit': line.get('unit') or 'm²',
                           'quantity': line.get('qty', ''), 'price': line.get('price', '')}
                          for line in row['lines'] if isinstance(line, dict)]
            })
            if any(not isinstance(line, dict) for line in row['lines']):
                raise ValueError('Rresht i pavlefshëm në backup.')
        return {'format': 'flete-dalje-v1', 'documents': documents}


def content(doc):
    data = doc.model_dump(mode='json', exclude={'id', 'version'})
    data['total'] = str(sum(line_value(line) for line in doc.lines))
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def serialize(row):
    data = json.loads(row.content_json)
    # Recompute on read so pre-reference records also use the corrected arithmetic.
    data['total'] = str(sum(line_value(line) for line in Document.model_validate(data).lines))
    return {**data, 'id': row.id, 'version': row.version}


@router.get('')
def archive(recipient: str = ''):
    with db.read_session() as session:
        rows = session.scalars(select(db.FleteDaljeDocument).order_by(db.FleteDaljeDocument.number.desc())).all()
        documents = [serialize(row) for row in rows]
        payments = [serialize_payment(row) for row in session.scalars(select(db.FleteDaljePayment)).all()]
        openings = [serialize_opening(row) for row in session.scalars(select(db.FleteDaljeOpeningBalance)).all()]
        return {'documents': documents, 'payments': payments, 'opening_balances': openings, 'clients': client_cards(documents, payments, openings),
                'nextNumber': max([0] + [r.number for r in rows if r.client_key == client_key(recipient)]) + 1}


@router.post('/opening-balances')
def set_opening_balance(opening: OpeningBalance):
    key = client_key(opening.recipient)
    data = opening.model_dump(mode='json', exclude={'version'})
    data['updated_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True)
    try:
        with db.get_session() as session:
            if opening.version == 0:
                row = db.FleteDaljeOpeningBalance(client_key=key, version=1, content_json=encoded)
                session.add(row)
                session.flush()
                result = serialize_opening(row)
            else:
                changed = session.execute(update(db.FleteDaljeOpeningBalance).where(
                    db.FleteDaljeOpeningBalance.client_key == key,
                    db.FleteDaljeOpeningBalance.version == opening.version
                ).values(content_json=encoded, version=opening.version+1))
                if changed.rowcount != 1:
                    raise HTTPException(409, 'Gjendja fillestare është ndryshuar. Rifreskoni dhe hapeni përsëri.')
                result = {**data, 'version': opening.version+1}
        return result
    except IntegrityError:
        raise HTTPException(409, 'Klienti ka një gjendje fillestare. Rifreskoni dhe përdorni Ndrysho gjendjen fillestare.')


@router.post('/payments')
def record_payment(payment: Payment):
    # Client-generated immutable ID makes retries idempotent, including after a timeout.
    data = json.loads(payment_content(payment))
    data['created_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    row = db.FleteDaljePayment(id=payment.id, version=1, voided=False,
                              content_json=json.dumps(data, ensure_ascii=False, sort_keys=True))
    try:
        with db.get_session() as session:
            existing = session.get(db.FleteDaljePayment, payment.id)
            if existing:
                if payment_content(Payment.model_validate(serialize_payment(existing))) != payment_content(payment):
                    raise HTTPException(409, 'Pagesa me këtë identifikues ekziston me të dhëna të tjera.')
                return serialize_payment(existing)
            session.add(row)
            session.flush()
            result = serialize_payment(row)
        return result
    except IntegrityError:
        raise HTTPException(409, 'Pagesa ekziston. Rifreskoni kartelën para se të provoni përsëri.')


@router.post('/payments/{payment_id}/void')
def void_payment(payment_id: str, version: int):
    with db.get_session() as session:
        result = session.execute(update(db.FleteDaljePayment).where(
            db.FleteDaljePayment.id == payment_id, db.FleteDaljePayment.version == version,
            db.FleteDaljePayment.voided == False
        ).values(voided=True, version=version + 1))
        if result.rowcount != 1:
            raise HTTPException(409, 'Pagesa është ndryshuar ose anuluar. Rifreskoni kartelën.')
    return {'ok': True}


@router.post('')
def create(doc: Document):
    row = db.FleteDaljeDocument(id=str(uuid.uuid4()), number=doc.number, client_key=client_key(doc.recipient), version=1, content_json=content(doc))
    try:
        with db.get_session() as session:
            session.add(row)
            session.flush()
            result = serialize(row)
        return result
    except IntegrityError:
        raise HTTPException(409, 'Ky numër ekziston për këtë klient. Zgjidhni një numër tjetër.')


@router.put('/{document_id}')
def edit(document_id: str, doc: SavedDocument):
    if document_id != doc.id:
        raise HTTPException(400, 'Identifikues i pasaktë.')
    try:
        with db.get_session() as session:
            result = session.execute(update(db.FleteDaljeDocument).where(
                db.FleteDaljeDocument.id == document_id, db.FleteDaljeDocument.version == doc.version
            ).values(number=doc.number, client_key=client_key(doc.recipient), content_json=content(doc), version=doc.version + 1))
            if result.rowcount != 1:
                raise HTTPException(409, 'Fleta është ndryshuar ose fshirë. Rihapeni nga historiku.')
        return {**json.loads(content(doc)), 'id': document_id, 'version': doc.version + 1}
    except IntegrityError:
        raise HTTPException(409, 'Ky numër ekziston për këtë klient. Zgjidhni një numër tjetër.')


@router.delete('/{document_id}')
def remove(document_id: str, version: int):
    with db.get_session() as session:
        result = session.execute(delete(db.FleteDaljeDocument).where(
            db.FleteDaljeDocument.id == document_id, db.FleteDaljeDocument.version == version))
        if result.rowcount != 1:
            raise HTTPException(409, 'Fleta është ndryshuar ose fshirë. Rifreskoni historikun.')
    return {'ok': True}


@router.post('/import')
def restore(backup: Backup):
    """Merge missing records atomically. Never overwrite an existing record."""
    ids, numbers = set(), set()
    try:
        with db.get_session() as session:
            for doc in backup.documents:
                scope = (client_key(doc.recipient), doc.number)
                if doc.id in ids or scope in numbers:
                    raise HTTPException(409, 'Backup përmban numra ose identifikues të përsëritur.')
                ids.add(doc.id)
                numbers.add(scope)
                existing = session.get(db.FleteDaljeDocument, doc.id)
                if existing is None:
                    existing = session.scalar(select(db.FleteDaljeDocument).where(db.FleteDaljeDocument.number == doc.number,
                        db.FleteDaljeDocument.client_key == client_key(doc.recipient)))
                if existing:
                    if content(Document.model_validate(json.loads(existing.content_json))) != content(doc):
                        raise HTTPException(409, f'Konflikt me fletën {doc.number}. Asnjë ndryshim nuk u ruajt.')
                    continue
                session.add(db.FleteDaljeDocument(id=doc.id, number=doc.number, client_key=client_key(doc.recipient), version=1, content_json=content(doc)))
            payment_ids = set()
            for payment in backup.payments:
                if payment.id in payment_ids:
                    raise HTTPException(409, 'Backup përmban pagesa të përsëritura.')
                payment_ids.add(payment.id)
                data = payment.model_dump(mode='json', exclude={'id', 'version', 'voided'})
                encoded = json.dumps(data, ensure_ascii=False, sort_keys=True)
                existing = session.get(db.FleteDaljePayment, payment.id)
                if existing:
                    if json.loads(existing.content_json) != data or existing.voided != payment.voided:
                        raise HTTPException(409, 'Konflikt me pagesën. Asnjë ndryshim nuk u ruajt.')
                    continue
                session.add(db.FleteDaljePayment(id=payment.id, version=payment.version,
                                                voided=payment.voided, content_json=encoded))
            opening_keys = set()
            for opening in backup.opening_balances:
                key = client_key(opening.recipient)
                if key in opening_keys:
                    raise HTTPException(409, 'Backup përmban gjendje fillestare të përsëritura.')
                opening_keys.add(key)
                data = opening.model_dump(mode='json', exclude={'version'})
                existing = session.get(db.FleteDaljeOpeningBalance, key)
                if existing:
                    if json.loads(existing.content_json) != data:
                        raise HTTPException(409, 'Konflikt me gjendjen fillestare. Asnjë ndryshim nuk u ruajt.')
                    continue
                session.add(db.FleteDaljeOpeningBalance(client_key=key, version=opening.version,
                    content_json=json.dumps(data, ensure_ascii=False, sort_keys=True)))
            session.flush()
    except IntegrityError:
        raise HTTPException(409, 'Numër ekzistues në arkiv. Asnjë ndryshim nuk u ruajt.')
    return archive()
