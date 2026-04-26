import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.data.preprocessing import load_and_preprocess_data

def run_baseline():
    mlflow.set_experiment("TechChallenge_Churn_MLP")
    X_train, X_test, y_train, y_test = load_and_preprocess_data('data/raw/Telco_customer_churn.csv')

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=100)
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=f"Baseline_{name}"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # As 4 métricas exigidas
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }

            mlflow.log_metrics(metrics)
            print(f"Modelo {name} finalizado. Recall: {metrics['recall']:.4f}")

if __name__ == "__main__":
    run_baseline()