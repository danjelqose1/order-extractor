"""Isolated delivery API regression checks, using a disposable SQLite database."""
import importlib
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('DB_DIR', str(tmp_path))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'backend'))
    sys.modules.pop('flete_dalje', None)
    sys.modules.pop('db', None)
    db = importlib.import_module('db')
    db.init_db()
    module = importlib.import_module('flete_dalje')
    app = FastAPI()
    app.include_router(module.router)
    return TestClient(app)


def payload(number=49):
    return dict(number=number, recipient='Klienti', date='2026-11-04', address='Tiranë', notes='',
                lines=[dict(description='Xham', unit='m²', quantity='3.782', price='100'),
                       dict(description='Xham', unit='m²', quantity='0.312', price='12.5')])


def test_crud_conflicts_and_backup(client):
    url = '/api/flete-dalje'
    assert client.get(url).json() == {'documents': [], 'payments': [], 'clients': [], 'nextNumber': 49}
    a = client.post(url, json=payload()).json()
    assert a['total'] == '382'
    assert client.post(url, json=payload()).status_code == 409
    b = client.post(url, json=payload(50)).json()
    changed = {**a, 'number': 51, 'recipient': 'Ndryshuar'}
    edited = client.put(f"{url}/{a['id']}", json=changed).json()
    assert edited['version'] == 2
    assert client.put(f"{url}/{a['id']}", json=changed).status_code == 409
    assert client.put(f"{url}/{a['id']}", json={**edited, 'number': 50}).status_code == 409
    archive = client.get(url).json()
    assert archive['nextNumber'] == 52
    backup = {'format': 'flete-dalje-v1', 'documents': archive['documents']}
    assert client.post(url+'/import', json=backup).status_code == 200
    conflicting = {'format':'flete-dalje-v1', 'documents': [{**b,'id':'new-one','number':60}, {**edited,'recipient':'Collision'}]}
    assert client.post(url+'/import', json=conflicting).status_code == 409
    assert len(client.get(url).json()['documents']) == 2
    assert client.delete(f"{url}/{a['id']}?version=1").status_code == 409
    assert client.delete(f"{url}/{a['id']}?version=2").status_code == 200
    assert client.get(url).json()['documents'] == [b]
    assert client.post(url+'/import', json=backup).status_code == 200
    assert len(client.get(url).json()['documents']) == 2


@pytest.mark.parametrize('quantity,price,expected', [
    ('3.782', '100', '378'), ('0.312', '12.5', '4'),
    ('3,782', '100', '378'), ('1.234,56', '1', '1235'),
    ('0.5', '1', '1'), ('-0.5', '1', '0'), ('-1.5', '1', '-1'),
    ('0.312', '0.625', '0'), ('not a number', '10', '0'),
])
def test_reference_number_parsing_and_rounding(client, quantity, price, expected):
    doc = payload()
    doc['lines'] = [dict(quantity=quantity, price=price)]
    assert client.post('/api/flete-dalje', json=doc).json()['total'] == expected


def test_invalid_nonfinite_values(client):
    doc = payload()
    doc['lines'][0]['quantity'] = 'Infinity'
    assert client.post('/api/flete-dalje', json=doc).status_code == 422


def test_reference_backups_import_and_conflicts(client):
    row = {'num': 70, 'client': 'Klienti origjinal', 'date': '2026-11-04',
           'address': 'Tiranë', 'notes': 'Transport',
           'lines': [{'name': 'TR + 12 + TR', 'unit': 'm²', 'qty': '3,782', 'price': '100'}]}
    url = '/api/flete-dalje'
    result = client.post(url+'/import', json={'schema':1, 'records':[row]})
    assert result.status_code == 200
    doc = result.json()['documents'][0]
    assert doc['recipient'] == row['client'] and doc['total'] == '378'
    assert doc['lines'][0]['quantity'] == '3,782'
    assert client.post(url+'/import', json=[row]).status_code == 200
    assert client.post(url+'/import', json=[{**row, 'client': 'Conflict'}]).status_code == 409
    assert client.get(url).json()['documents'] == [doc]
    assert client.post(url+'/import', json={'schema':1,'records':[{**row,'lines':[None]}]}).status_code == 422


def payment(**changes):
    return {**dict(id='payment-1', recipient='Klienti', date='2026-11-05', amount='100.50', method='cash', notes='Arkë'), **changes}


def test_client_ledger_payments_and_void(client):
    url = '/api/flete-dalje'
    doc = client.post(url, json=payload()).json()
    client.post(url, json={**payload(50), 'recipient':'  KLIENTI  '})
    other = client.post(url, json={**payload(51), 'recipient':'Tjetër'}).json()
    saved = client.post(url+'/payments', json=payment()).json()
    assert client.post(url+'/payments', json=payment()).json() == saved  # Retry is idempotent.
    assert client.post(url+'/payments', json=payment(amount='99')).status_code == 409
    archive = client.get(url).json()
    card = next(row for row in archive['clients'] if row['key']=='klienti')
    assert len(card['entries']) == 3
    assert card['documents_cents'] == 76400
    assert card['paid_cents'] == 10050
    assert card['balance_cents'] == 66350
    assert card['entries'][-1]['balance_cents'] == 66350
    assert next(row for row in archive['clients'] if row['key']=='tjetër')['balance_cents'] == 38200
    assert client.get(url).json()['documents'][0] == other
    assert client.post(url+'/payments/payment-1/void?version=99').status_code == 409
    assert client.post(url+'/payments/payment-1/void?version=1').status_code == 200
    assert client.post(url+'/payments/payment-1/void?version=1').status_code == 409
    card = next(row for row in client.get(url).json()['clients'] if row['key']=='klienti')
    assert card['balance_cents'] == 76400
    assert card['entries'][-1]['voided'] is True
    assert card['entries'][-1]['credit_cents'] == 0
    # Renaming/deleting documents moves only their debit, never the payment history.
    assert client.put(url+'/'+doc['id'], json={**doc, 'recipient':'Emër i ri'}).status_code == 200
    assert client.delete(url+'/'+doc['id']+'?version=2').status_code == 200
    assert any(row['key']=='klienti' for row in client.get(url).json()['clients'])


def test_payment_backup_transaction_and_credit(client):
    url = '/api/flete-dalje'
    saved = client.post(url+'/payments', json=payment()).json()
    archive = client.get(url).json()
    assert archive['clients'][0]['balance_cents'] == -10050  # Advance payment, no document required.
    backup = {'format':'flete-dalje-v1', 'documents':[], 'payments':[saved]}
    assert client.post(url+'/import', json=backup).status_code == 200
    # A conflict in a payment rolls back newly imported documents as well.
    doc = {**payload(80),'id':'import-doc','version':1}
    conflict = {**backup, 'documents':[doc], 'payments':[{**saved,'amount':'50'}]}
    assert client.post(url+'/import', json=conflict).status_code == 409
    assert client.get(url).json()['documents'] == []
    assert client.get(url).json()['payments'] == [saved]
    assert client.post(url+'/import',json={**backup, 'payments':[saved,saved]}).status_code == 409
    # Restore a distinct payment, preserving its cancellation and creation timestamp.
    restored = {**saved,'id':'restored-payment','voided':True,'version':2}
    assert client.post(url+'/import', json={**backup,'payments':[restored]}).status_code == 200
    assert client.get(url).json()['clients'][0]['balance_cents'] == -10050


@pytest.mark.parametrize('changes', [{'amount':'0'},{'amount':'-1'},{'amount':'0.001'},{'amount':'NaN'}, {'recipient':'  '}, {'method':'invalid'}])
def test_invalid_payment(client, changes):
    assert client.post('/api/flete-dalje/payments',json=payment(**changes)).status_code == 422


def test_complete_backup_restores_documents_and_payments(client):
    from sqlalchemy import delete
    db = importlib.import_module('db')
    url = '/api/flete-dalje'
    client.post(url, json=payload())
    client.post(url+'/payments', json=payment())
    original = client.get(url).json()
    backup = {'format':'flete-dalje-v1', 'documents':original['documents'], 'payments':original['payments']}
    with db.get_session() as session:
        session.execute(delete(db.FleteDaljeDocument))
        session.execute(delete(db.FleteDaljePayment))
    assert client.get(url).json()['clients'] == []
    restored = client.post(url+'/import', json=backup)
    assert restored.status_code == 200
    assert restored.json() == original
