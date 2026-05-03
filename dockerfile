# Use a imagem base do Python
FROM python:3.12-slim

# Instala o uv para gerenciar dependências rapidamente no build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copia os arquivos de dependências
COPY pyproject.toml uv.lock ./

# INSTALA AS DEPENDÊNCIAS NO BUILD 
RUN uv sync --frozen

# Copia o restante do código
COPY . .

# Executa a API diretamente (sem recriar venv)
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]