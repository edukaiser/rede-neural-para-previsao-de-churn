import pytest
from fastapi.testclient import TestClient
from src.api.main import app 

client = TestClient(app)

def test_read_health():
    """Testa se o endpoint de saúde está respondendo 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "online", "version": "1.0.0"}

def test_predict_churn_success():
    """Testa uma predição com dados válidos."""
    payload = {
        "Gender": "Female",
        "Senior Citizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure Months": 5,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "DSL",
        "Online Security": "Yes",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "Yes",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 29.85
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "churn_probability" in data
    assert "probability" in data

def test_predict_invalid_data():
    """Testa se a API barra o valor negativo com erro 422 (Unprocessable Entity)."""
    payload = {
        "Tenure Months": -5,  # Valor inválido
        "Monthly Charges": 29.85
    }
    response = client.post("/predict", json=payload)
    
    # O FastAPI/Pydantic retorna 422 quando a validação do Field(ge=0) falha
    assert response.status_code == 422