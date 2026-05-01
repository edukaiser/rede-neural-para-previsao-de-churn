import pandera.pandas as pa
from pydantic import BaseModel, ConfigDict, Field

# --- 1. MUNDO PANDERA (Obrigatório para os Testes de Schema) ---
# Esse schema valida o DataFrame inteiro quando carrega o CSV.
churn_pandera_schema = pa.DataFrameSchema({
    "Gender": pa.Column(str),
    "Senior Citizen": pa.Column(int, pa.Check.isin([0, 1])),
    "Partner": pa.Column(str),
    "Dependents": pa.Column(str),
    "Tenure Months": pa.Column(int, pa.Check.ge(0)),
    "Phone Service": pa.Column(str),
    "Multiple Lines": pa.Column(str),
    "Internet Service": pa.Column(str),
    "Online Security": pa.Column(str),
    "Online Backup": pa.Column(str),
    "Device Protection": pa.Column(str),
    "Tech Support": pa.Column(str),
    "Streaming TV": pa.Column(str),
    "Streaming Movies": pa.Column(str),
    "Contract": pa.Column(str),
    "Paperless Billing": pa.Column(str),
    "Payment Method": pa.Column(str),
    "Monthly Charges": pa.Column(float, pa.Check.ge(0)),
})

# --- 2. MUNDO PYDANTIC (Obrigatório para a Validação da API ) ---
# Esse schema valida o JSON que o usuário envia para a FastAPI.
class ChurnInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    Gender: str = Field(..., alias="Gender")
    SeniorCitizen: int = Field(..., alias="Senior Citizen", ge=0, le=1)
    Partner: str = Field(..., alias="Partner")
    Dependents: str = Field(..., alias="Dependents")
    TenureMonths: int = Field(..., alias="Tenure Months", ge=0)
    PhoneService: str = Field(..., alias="Phone Service")
    MultipleLines: str = Field(..., alias="Multiple Lines")
    InternetService: str = Field(..., alias="Internet Service")
    OnlineSecurity: str = Field(..., alias="Online Security")
    OnlineBackup: str = Field(..., alias="Online Backup")
    DeviceProtection: str = Field(..., alias="Device Protection")
    TechSupport: str = Field(..., alias="Tech Support")
    StreamingTV: str = Field(..., alias="Streaming TV")
    StreamingMovies: str = Field(..., alias="Streaming Movies")
    Contract: str = Field(..., alias="Contract")
    PaperlessBilling: str = Field(..., alias="Paperless Billing")
    PaymentMethod: str = Field(..., alias="Payment Method")
    MonthlyCharges: float = Field(..., alias="Monthly Charges", ge=0)