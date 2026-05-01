import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.data.preprocessing import initial_cleaning

def load_churn_data(path: str):
    """
    Carrega os dados e aplica a sua remoção agressiva de colunas 
    para o modelo parar de dar 0.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # 1. Criação do Target 
    if 'Churn Value' not in df.columns:
        df['Churn Value'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
    
    df.dropna(subset=['Churn Value'], inplace=True)
    df = initial_cleaning(df)

    # 2. Define o Y primeiro
    y = df['Churn Value'].values

    # 3. Remoção de Vazamentos
    cols_to_drop = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 
        'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 'Churn Reason',
        'Churn Score', 'CLTV', 'Total Charges', 'Satisfaction Score', 
        'Customer Status', 'Churn Value'
    ]
    
    # Remove qualquer outra que tenha 'Churn' e não seja o Target
    extra_churn_cols = [c for c in df.columns if 'Churn' in c and c != 'Churn Value']
    final_drop = list(set(cols_to_drop + extra_churn_cols))
    
    X = df.drop(columns=[c for c in final_drop if c in df.columns])
    y = df['Churn Value'].values # Pega os valores para o treino

    print(f"\n--- Features que sobraram: ---\n{X.columns.tolist()}\n")

    return X, y

def get_dataloader(X, y, batch_size=32, shuffle=True):
    """
    Transforma os dados processados em DataLoader do PyTorch.
    """
    # Converte matriz esparsa para array denso
    if hasattr(X, "toarray"):
        X = X.toarray()
    
    X_tensor = torch.FloatTensor(X)
    
    # Garante que y seja um array e tenha a forma (n_samples, 1)
    y_vals = y.values if hasattr(y, 'values') else y
    y_tensor = torch.FloatTensor(y_vals).view(-1, 1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)