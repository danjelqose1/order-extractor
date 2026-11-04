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
    assert client.get(url).json() == {'documents': [], 'nextNumber': 49}
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
