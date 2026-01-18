import pytest
from fastapi.testclient import TestClient

def test_health_endpoint():
    """Test API health check."""
    from wildkatze.deployment.api_server import app
    client = TestClient(app)
    
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint():
    """Test prediction endpoint."""
    from wildkatze.deployment.api_server import app
    client = TestClient(app)
    
    response = client.post("/v1/predict", json={
        "message_content": "Test message",
        "target_audience": "general",
        "culture": "de-DE"
    })
    assert response.status_code == 200
    assert "resonance_score" in response.json()

def test_analyze_endpoint():
    """Test analyze endpoint."""
    from wildkatze.deployment.api_server import app
    client = TestClient(app)
    
    response = client.post("/v1/analyze", json={
        "demographic_data": {"age": 30},
        "behavioral_data": {"interests": ["tech"]}
    })
    assert response.status_code == 200
