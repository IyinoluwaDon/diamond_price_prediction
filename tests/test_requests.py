from app import main as web
import pytest

def test_form_and_incomplete_submission(monkeypatch):
    client=web.app.test_client()
    assert client.get('/').status_code==200
    assert client.post('/',data={}).status_code==400
    called=[]
    monkeypatch.setattr(web.m,'predict_pipe',lambda values: called.append(values) or 1234.5)
    response=client.post('/',data={key:'1' for key in web.features})
    assert response.status_code==200 and called

def test_nonfinite_values_rejected_before_model_load():
    from app.model.model import predict_pipe,features,numeric
    values=['Round' for _ in features]
    for key in numeric:values[features.index(key)]='1'
    values[features.index('carat_weight')]='nan'
    with pytest.raises(ValueError):predict_pipe(values)
