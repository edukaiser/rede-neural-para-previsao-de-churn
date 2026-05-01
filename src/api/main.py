import time
from fastapi import FastAPI, HTTPException, Request
import torch
import joblib
import pandas as pd
from src.data.schema import ChurnInput
from src.models.mlp_model import ChurnMLP 
from src.utils import logger

app = FastAPI(title="FIAP Tech Challenge - API de Previsão de Churn")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    # Inicia o cronômetro
    start_time = time.time()
    
    # Processa a requisição
    response = await call_next(request)
    
    # Calcula o tempo gasto
    process_time = time.time() - start_time
    
    # Loga o tempo no terminal de forma estruturada
    logger.info(f"Metrica de Latencia - Path: {request.url.path} | Tempo: {process_time:.4f}s")
    
    # Adiciona o tempo no cabeçalho da resposta (opcional, mas boa prática)
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# 1. Carregamento dos Artefatos
try:
    # AJUSTE DE SEGURANÇA PYTORCH 2.6+: Autoriza a classe ChurnMLP
    torch.serialization.add_safe_globals([ChurnMLP])

    # Carrega o preprocessor
    preprocessor = joblib.load("models/preprocessor.pkl")
    
    # AJUSTE DE CARREGAMENTO: Adicionado weights_only=False para aceitar o objeto completo
    model = torch.load("models/baseline_model.pth", weights_only=False)
    model.eval()

    logger.info("API: Artefatos carregados com sucesso.")
except Exception as e:
    logger.error(f"API: Erro ao carregar artefatos - {e}")
    raise RuntimeError("Erro na inicialização da API.")

# 2. Endpoints
@app.get("/health")
def health():
    return {"status": "online", "version": "1.0.0"}

# 3. Logica de Predição
@app.post("/predict")
async def predict(payload: ChurnInput):
    try:
        # Converte o JSON recebido para DataFrame
        data_dict = payload.model_dump(by_alias=True)
        df_input = pd.DataFrame([data_dict])

        # Transformação (Engenharia de Features)
        X_processed = preprocessor.transform(df_input)
        X_tensor = torch.FloatTensor(X_processed)

        with torch.no_grad():
            output = model(X_tensor)
            probability = output.item()
            prediction = 1 if probability >= 0.5 else 0
        
        logger.info(f"Predição concluída. Resultado: {prediction}")

        return {
            "churn_probability": probability,
            "probability": round(probability, 4),
            "label": "Churn" if prediction == 1 else "No Churn"
        }
    
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail="Erro interno na predição.")

if __name__ == "__main__":
    import uvicorn
    # 4. Inicialização do servidor local
    uvicorn.run(app, host="0.0.0.0", port=8000)