import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class ChurnDataset(Dataset):
    """
    Converte DataFrames em Tensores do PyTorch para o treinamento da MLP.
    """
    def __init__(self, X, y):
        # Conversão para Tensores
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloader(X, y, batch_size=32, shuffle=True):
    """
    Cria o DataLoader para gerenciar o Batching.
    """
    dataset = ChurnDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)