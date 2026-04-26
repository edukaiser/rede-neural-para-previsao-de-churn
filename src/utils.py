import torch
import numpy as np
import random
import logging

# Configurações do Logging Estruturado
logging.basicConfig(level=logging.INFO, format = "%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def set_seeds(seed=42):
    """
    Fixa sementes aleatórias para garantir que os resultados sejam reproduzíveis.
    Abrange bibliotecas Python padrão, Numpy e PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuuda.manual_seed_all(seed)
    logger.info(f"Ambiente configurado com seed {seed} para reprodutibilidade técnica.")