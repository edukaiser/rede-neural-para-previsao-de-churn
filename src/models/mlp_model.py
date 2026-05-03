import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import json
import mlflow
from datetime import datetime
from src.utils import logger

class ChurnMLP(nn.Module):
    """
    Rede Neural Multilayer Perceptron (MLP) para predição de Churn.
    O modelo inicial apresentou overfitting (Loss 0.01), então apliquei Dropout e 
    removi features redundantes para obter uma generalização melhor.
    """

    def __init__(self, input_size):
        """
        Inicializa a arquitetura da rede neural.

        Args:
            input_size (int): Número de features de entrada após o pré-processamento.
        """
        super(ChurnMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Define o fluxo de dados (forward pass) através da rede.

        Args:
            x (torch.Tensor): Tensor contendo os dados de entrada.

        Returns:
            torch.Tensor: Probabilidade de Churn (valor entre 0 e 1).
        """
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        return self.sigmoid(self.output(x))

    def save_model_card(self, metrics, params, output_path="docs/model_card.json"):
        """
        Gera o Model Card e o salva na pasta /docs do projeto, 
        além de registrar como artefato no MLflow.

        Args:
            metrics (dict): Dicionário com acurácia, precisão, recall, etc.
            params (dict): Hiperparâmetros como learning rate e épocas.
            output_path (str): Caminho relativo para a pasta de documentos.
        """
        model_card = {
            "model_details": {
                "name": "Churn Prediction MLP",
                "version": "1.0.0",
                "developer": "Eduardo",
                "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "architecture": "Multilayer Perceptron (PyTorch)"
            },
            "performance_metrics": metrics,
            "hyperparameters": params,
            "limitations": "O modelo pode apresentar variações se houver mudança brusca no perfil de consumo dos clientes.",
            "failure_scenarios": [
                "Entrada de valores negativos (tratado via Pandera/Pydantic)",
                "Predição para clientes com baixíssimo histórico (tenure)"
            ]
        }

        try:
            # Salva o arquivo JSON na pasta /docs
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(model_card, indent=4, fp=f, ensure_ascii=False)
            
            # Também envia para o MLflow para manter o rastreio do experimento
            mlflow.log_artifact(output_path)
            logger.info(f"✅ Model Card salvo com sucesso em: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Falha ao gravar Model Card em {output_path}: {e}")