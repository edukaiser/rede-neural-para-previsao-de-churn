# Bibliotecas de Sistema e Manipulação de Dados
import os
import pandas as pd
import numpy as np
import joblib  # Essencial para salvar o preprocessor.pkl (Engenharia)

# PyTorch (Para MLP)
import torch
import torch.nn as nn
import torch.optim as optim

# MLflow (Para rastreamento de experimentos)
import mlflow
import mlflow.pytorch

# Scikit-Learn (Para métricas, split e o Pipeline de Engenharia)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Módulos Internos (Pasta src/)
from src.data.preprocessing import build_features_pipeline, initial_cleaning
from src.data.dataset import load_churn_data, get_dataloader
from src.models.mlp_model import ChurnMLP
from src.utils import logger, set_seeds

def train():
    # 1. Configurações e sementes
    set_seeds(42)
    mlflow.set_experiment("TechChallenge_Churn_MLP")

    # 2. Carregar e Limpar os dados 
    X, y = load_churn_data('data/raw/Telco_customer_churn.csv')
    
    # 3. Criar o Pipeline Reproduzível
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Chama a função original 
    preprocessor = build_features_pipeline(numeric_cols, categorical_cols)

    # 4. Split e Transformação (Onde o pipeline "aprende" as escalas)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # O preprocessor gera os dados prontos para a rede neural
    X_train = preprocessor.fit_transform(X_train_raw) 
    X_test = preprocessor.transform(X_test_raw)

    # ENGENHARIA: Salva o objeto que aprendeu as escalas para a futura API
    joblib.dump(preprocessor, "models/preprocessor.pkl") 

    # 5. Criar os DataLoaders (Gerencia o Batching de 32)
    train_loader = get_dataloader(X_train, y_train, batch_size=32)
    test_loader = get_dataloader(X_test, y_test, batch_size=32, shuffle=False)

    # 6. Iniciar Modelo
    input_dim = X_train.shape[1]
    model = ChurnMLP(input_size=input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 7. Loop de Treinamento com MLflow
    epochs = 100 # Aumentamos o teto, pois o Early Stopping vai parar antes se precisar
    patience = 5  # Quantas épocas esperar sem melhora no test_loss
    best_test_loss = float('inf')
    early_stop_count = 0

    with mlflow.start_run(run_name="Treinamento_Com_Metricas"):
        mlflow.log_param("learning_rate", 0.0001)

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
            # Validação e Cálculo de Métricas
            model.eval()
            test_loss = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    test_loss += criterion(outputs, labels).item()
                    
                    # Converte probabilidades em classes (0 ou 1)
                    preds = (outputs > 0.5).float()
                    all_preds.extend(preds.numpy())
                    all_labels.extend(labels.numpy())

            avg_train_loss = total_loss / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)

            # Cálculo das 4 métricas exigidas
            acc = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, zero_division=0)
            rec = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)

            # Log no MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("test_loss", avg_test_loss, step=epoch)
            mlflow.log_metric("accuracy", acc, step=epoch)
            mlflow.log_metric("precision", prec, step=epoch)
            mlflow.log_metric("recall", rec, step=epoch)
            mlflow.log_metric("f1_score", f1, step=epoch)

            if (epoch + 1) % 5 == 0:
                print(f"Época [{epoch+1}] | Loss: {avg_test_loss:.4f} | F1: {f1:.4f} | Recall: {rec:.4f}")

            # Lógica de Early Stopping
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                early_stop_count = 0
                # Salva o melhor estado do modelo se quiser
                # torch.save(model.state_dict(), 'best_model.pth')
            else:
                early_stop_count += 1
            
            if early_stop_count >= patience:
                print(f"Early Stopping acionado na época {epoch+1}!")
                break
        
        # 8. GERAÇÃO AUTOMÁTICA DO MODEL CARD
        final_metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "final_loss": float(avg_test_loss)
        }
        
        final_params = {
            "learning_rate": 0.0001,
            "epochs_executed": epoch + 1,
            "patience": patience,
            "batch_size": 32,
            "input_dim": input_dim
        }

        # Chamada do método na classe ChurnMLP
        model.save_model_card(metrics=final_metrics, params=final_params)

        # 9. Registro Final
        mlflow.pytorch.log_model(model, "model_churn_final")
        torch.save(model, "models/baseline_model.pth")
        logger.info("Modelo treinado e registrado em models/baseline_model.pth e MLFlow com sucesso!")
    


if __name__ == "__main__":
    train()