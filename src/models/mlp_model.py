import torch
import torch.nn as nn

class ChurnMLP(nn.Module):
    """
    O modelo inicial apresentou overfitting (Loss 0.01), então apliquei Dropout e 
    removi features redundantes para obter uma generalização melhor
    """

    def __init__(self, input_size):
        super(ChurnMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2) 
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        return self.sigmoid(self.output(x))