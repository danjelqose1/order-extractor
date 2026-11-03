import sys
from pathlib import Path
from copy import deepcopy
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import ai_decisions as ai


def order(**changes):
    value = {'id': 1, 'client_name': 'Client A', 'order_numbers': ['R-101'], 'source_hash': 'document-a',
             'rows': [{'type': '4 Low-E / 8 spacer / 6 clear', 'dimension': '500x600', 'position': '1', 'quantity': 2}]}
    value.update(changes)
    return value


def api(evidence, questions):
    out = {}
    for q in questions:
        if q.type == 'predicate':
            out[q.name] = {'type': q.type, 'name': q.name, 'probability': 0}
        elif q.type == 'choice':
            out[q.name] = {'type': q.type, 'name': q.name, 'choice': 'double_glazing' if q.name == 'category' else 'Processing review' if q.name == 'destination' else 'possible_duplicate', 'confidence': .999}
        else:
            out[q.name] = {'type': q.type, 'name': q.name, 'score': 0, 'confidence': .999}
    return out


def candidate(a, b):
    cfg = ai.config()
    return ai.compare(ai.normalize(a, cfg), ai.normalize(b, cfg), cfg)


def test_exact_document_no_model():
    result = ai.evaluate(order(), [order(id=2)], api=lambda *a: pytest.fail('Exact check should be local'))
    assert result['classification'] == 'exact_duplicate'
    assert result['readiness'] == 'Review Required'


def test_formatting_and_row_order():
    a = order(); a['rows'] *= 2
    b = deepcopy(a); b['id'] = 2; b['rows'].reverse(); b['rows'][0]['dimension'] = '500 × 600'
    assert candidate(a, b)['exact']


@pytest.mark.parametrize('dimension,offset', [('503x603', 3), ('493x593', -7)])
def test_offsets_only_candidates(dimension, offset):
    a, b = order(), order(id=2, source_hash='different')
    a['rows'][0]['dimension'] = dimension
    c = candidate(a, b)
    assert not c['exact'] and offset in c['dimension_offsets']
    assert ai.evaluate(a, [b], api)['classification'] == 'possible_duplicate'


def test_different_clients_not_dimensions():
    assert candidate(order(), order(id=2, client_name='Client B', source_hash='different')) is None


def test_repeated_pieces_preserved():
    a = order(); a['rows'].append({**a['rows'][0], 'position': '2'})
    before = deepcopy(a)
    result = ai.evaluate(a, [], api)
    assert result['readiness'] == 'Ready' and a == before


@pytest.mark.parametrize('field,value', [('quantity', 3), ('type', '4 Low-E / 10 spacer / 6 clear'), ('type', '6 Low-E / 8 spacer / 6 clear'), ('type', '4 clear / 8 spacer / 6 clear')])
def test_modifications_not_exact(field, value):
    b = order(id=2); b['rows'][0][field] = value
    assert not candidate(order(), b)['exact']
    assert ai.evaluate(order(), [b], api)['readiness'] == 'Review Required'


@pytest.mark.parametrize('dimension', ['', 'unclear', '0x600'])
def test_missing_handwriting_dimensions(dimension):
    a = order(); a['rows'][0]['dimension'] = dimension
    assert ai.evaluate(a, [], api)['readiness'] == 'Blocked'


@pytest.mark.parametrize('error', [TimeoutError, RuntimeError])
def test_api_failure(error):
    def broken(*args): raise error()
    result = ai.evaluate(order(), [], broken)
    assert result['readiness'] == 'Review Required'
    assert result['classification'] == 'needs_review'


def test_override_audit_and_disabled_routes(tmp_path, monkeypatch):
    import db
    import ai_decisions_service as service
    from sqlalchemy import create_engine, select
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    monkeypatch.setattr(db, 'engine', create_engine(f'sqlite:///{tmp_path}/advice.db', connect_args={'check_same_thread': False}))
    monkeypatch.setattr(service, 'get_record', lambda *args: order(kind='order'))
    service.initialize()
    result = ai.evaluate(order(), [], api); result['input_snapshot'] = ai.normalize(order(kind='order'), ai.config())
    with db.engine.begin() as conn:
        import json
        conn.execute(service.decisions.insert().values(id=1, fingerprint='one', order_id=1, kind='order', result=json.dumps(result)))
    from fastapi import HTTPException
    app = FastAPI(); service.install(app, lambda: 'test-secret')
    routes = {r.path: r for r in app.routes if r.path.startswith('/api/ai-decisions')}
    authorize = routes['/api/ai-decisions'].dependencies[0].dependency
    override = routes['/api/ai-decisions/{decision_id}/override'].endpoint
    payload_type = override.__annotations__['payload']
    monkeypatch.setenv('AI_DECISIONS_ENABLED', 'true')
    with pytest.raises(HTTPException) as denied:
        authorize('')
    assert denied.value.status_code == 401
    authorize('test-secret')
    for action in ['confirm_duplicate', 'legitimate', 'ignore', 'reset', 'accept']:
        assert override(1, payload_type(action=action))['order_modified'] is False
    with db.engine.connect() as conn:
        assert len(conn.execute(select(service.events)).all()) == 5
    monkeypatch.setattr(service, 'get_record', lambda *args: order(kind='order', client_name='Changed'))
    with pytest.raises(HTTPException) as stale:
        override(1, payload_type(action='accept'))
    assert stale.value.status_code == 409
    monkeypatch.setenv('AI_DECISIONS_ENABLED', 'false')
    with pytest.raises(HTTPException) as disabled:
        authorize('test-secret')
    assert disabled.value.status_code == 404


def test_official_transport_contract_and_refusal(monkeypatch):
    captured = {}
    class Client:
        def __init__(self, **kwargs): captured['options'] = kwargs
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, path, body, cast_to):
            captured.update(path=path, body=body)
            return {'answers': [{'name': 'missing', 'type': 'predicate', 'probability': .8}]}
    monkeypatch.setattr(ai, 'OpenAI', Client)
    monkeypatch.setenv('AI_DECISIONS_MIN_INTERVAL_SECONDS', '0')
    question = ai.PredicateQuestion(name='missing', instructions='Is information missing?')
    result = ai.call_api({'order': order()}, [question])
    assert result['missing']['probability'] == .8
    assert captured['path'] == '/decisions'
    assert captured['body']['model'] == 'gpt-6-luna'
    assert captured['body']['input'][1]['content'][0]['type'] == 'input_json'
    assert captured['body']['input'][1]['content'][0]['data']['order']['rows']
    assert captured['options']['max_retries'] == 2
    monkeypatch.setattr(Client, 'post', lambda *args, **kwargs: {'answers': [{'name': 'missing', 'type': 'refusal'}]})
    with pytest.raises(ValueError): ai.call_api({}, [question])


def test_client_alias_and_warning_conservatism(monkeypatch):
    monkeypatch.setenv('AI_DECISIONS_CLIENT_ALIASES', '{"Client Alias":"Client A"}')
    assert candidate(order(), order(id=2, client_name='Client Alias'))['exact']
    assert ai.evaluate(order(warnings=['unclear handwriting']), [], api)['readiness'] == 'Review Required'
