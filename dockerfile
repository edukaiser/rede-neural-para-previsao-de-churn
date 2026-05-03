# Usa uma imagem oficial do Python 3.12 (versão slim para ser leve)
FROM python:3.12-slim

# Instala o uv manualmente dentro do container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvbin/uv

# Define o diretório de trabalho
WORKDIR /app

# Adiciona o uv ao PATH para podermos usá-lo livremente
ENV PATH="/uvbin:${PATH}"

# Copia os arquivos de dependência
COPY pyproject.toml uv.lock ./

# Instala as dependências usando o uv
RUN uv sync --frozen --no-install-project

# Copia o restante do código
COPY . .

# Expõe a porta do FastAPI
EXPOSE 8000

# Comando para rodar a API
CMD ["uv", "run", "python", "-m", "src.api.main"]