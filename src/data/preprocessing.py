import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import logger

def load_and_preprocess_data(filepath):
    logger.info(f"Iniciado pré-processamento: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # 1. Criação do Target e remoção agressiva de vazamentos
    if 'Churn Value' not in df.columns:
        df['Churn Value'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
    
    # 2. Converte Total Charges e remove linhas vazias
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

    df['Total Charges'] = df['Total Charges'].fillna(0)

    df.dropna(subset=['Churn Value'], inplace=True)
    
    # Lista de colunas que causam Data Leakage ou são irrelevantes
    cols_to_drop = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 
        'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 'Churn Reason',
        'Churn Score', 'CLTV', 'Total Charges', 'Satisfaction Score', 
        'Customer Status'
    ]
    
    # 3. Remove qualquer coluna que contenha 'Churn' no nome e não seja o Target
    extra_churn_cols = [c for c in df.columns if 'Churn' in c and c != 'Churn Value']
    final_drop = list(set(cols_to_drop + extra_churn_cols))
    
    df.drop(columns=[c for c in final_drop if c in df.columns], inplace=True)

    # 4. Encoding usando One-Hot Encoding
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # 5. Separação
    X = df.drop('Churn Value', axis=1)
    y = df['Churn Value'].values

    # 6. Print para conferência no terminal
    print(f"\n--- Features utilizadas no treino ---\n{X.columns.tolist()}\n")

    # 7. Split e Escalonamento
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"Pré-processamento concluído. Linhas: {len(df)}")

    return X_train_scaled, X_test_scaled, y_train, y_test