import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.data.preprocessing import load_and_preprocess_data
from src.data.dataset import get_dataloader
from src.models.mlp_model import ChurnMLP
from src.utils import logger, set_seeds

def train():
    # 1. Configrações e sementes para reprodutibilidade
    set_seeds(42)
    mlflow.set_experiment("TechChallenge_Churn_MLP")

    # 2. Carregar dados usando preprocessing
    X_train, X_test, y_train, y_test = load_and_preprocess_data('data/raw/Telco_customer_churn.csv')

    # 3. Criar os DataLoaders (Gerencia o Batching de 32)
    train_loader = get_dataloader(X_train, y_train, batch_size=32)
    test_loader = get_dataloader(X_test, y_test, batch_size=32, shuffle=False)

    # 4. Iniciar Modelo
    input_dim = X_train.shape[1]
    model = ChurnMLP(input_size=input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 5. Loop de Treinamento com MLflow
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

        mlflow.pytorch.log_model(model, "model_churn_final")
        logger.info("Modelo treinado e registrado no MLflow com sucesso!")

if __name__ == "__main__":
    train()