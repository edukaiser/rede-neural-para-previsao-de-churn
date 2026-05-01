import pandas as pd
import pytest
from src.data.schema import churn_pandera_schema

def test_churn_schema_valid_data():
    """Testa se o schema aceita dados corretos."""
    data = {
        "Gender": ["Female"],
        "Senior Citizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "Tenure Months": [-5], # Valor inválido para testar a validação
        "Phone Service": ["Yes"],
        "Multiple Lines": ["No"],
        "Internet Service": ["DSL"],
        "Online Security": ["Yes"],
        "Online Backup": ["No"],
        "Device Protection": ["No"],
        "Tech Support": ["Yes"],
        "Streaming TV": ["No"],
        "Streaming Movies": ["No"],
        "Contract": ["Month-to-month"],
        "Paperless Billing": ["Yes"],
        "Payment Method": ["Electronic check"],
        "Monthly Charges": [29.85]
    }
    df = pd.DataFrame(data)
    
    with pytest.raises(Exception):
        churn_pandera_schema.validate(df)