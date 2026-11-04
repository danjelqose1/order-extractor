"""Independent delivery documents; no order/customer dependencies."""
import json
import uuid
from datetime import date
import math
import re

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


class Backup(BaseModel):
    format: str = Field(pattern=r'^flete-dalje-v1$')
    documents: list[SavedDocument] = Field(max_length=10000)

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
            documents.append({
                'id': row.get('id', f"legacy-{row.get('num')}"), 'version': row.get('version', 1),
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
def archive():
    with db.read_session() as session:
        rows = session.scalars(select(db.FleteDaljeDocument).order_by(db.FleteDaljeDocument.number.desc())).all()
        return {'documents': [serialize(row) for row in rows], 'nextNumber': max([48] + [r.number for r in rows]) + 1}


@router.post('')
def create(doc: Document):
    row = db.FleteDaljeDocument(id=str(uuid.uuid4()), number=doc.number, version=1, content_json=content(doc))
    try:
        with db.get_session() as session:
            session.add(row)
            session.flush()
            result = serialize(row)
        return result
    except IntegrityError:
        raise HTTPException(409, 'Ky numër ekziston. Zgjidhni një numër tjetër.')


@router.put('/{document_id}')
def edit(document_id: str, doc: SavedDocument):
    if document_id != doc.id:
        raise HTTPException(400, 'Identifikues i pasaktë.')
    try:
        with db.get_session() as session:
            result = session.execute(update(db.FleteDaljeDocument).where(
                db.FleteDaljeDocument.id == document_id, db.FleteDaljeDocument.version == doc.version
            ).values(number=doc.number, content_json=content(doc), version=doc.version + 1))
            if result.rowcount != 1:
                raise HTTPException(409, 'Fleta është ndryshuar ose fshirë. Rihapeni nga historiku.')
        return {**json.loads(content(doc)), 'id': document_id, 'version': doc.version + 1}
    except IntegrityError:
        raise HTTPException(409, 'Ky numër ekziston. Zgjidhni një numër tjetër.')


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
                if doc.id in ids or doc.number in numbers:
                    raise HTTPException(409, 'Backup përmban numra ose identifikues të përsëritur.')
                ids.add(doc.id)
                numbers.add(doc.number)
                existing = session.get(db.FleteDaljeDocument, doc.id)
                if existing is None:
                    existing = session.scalar(select(db.FleteDaljeDocument).where(db.FleteDaljeDocument.number == doc.number))
                if existing:
                    if content(Document.model_validate(json.loads(existing.content_json))) != content(doc):
                        raise HTTPException(409, f'Konflikt me fletën {doc.number}. Asnjë ndryshim nuk u ruajt.')
                    continue
                session.add(db.FleteDaljeDocument(id=doc.id, number=doc.number, version=1, content_json=content(doc)))
            session.flush()
    except IntegrityError:
        raise HTTPException(409, 'Numër ekzistues në arkiv. Asnjë ndryshim nuk u ruajt.')
    return archive()
