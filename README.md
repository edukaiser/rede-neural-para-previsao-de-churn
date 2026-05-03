## 📁 Documentação Profissional

Abaixo, os documentos que sustentam a governança e a confiabilidade do sistema na pasta `/docs`:

1.  **[Arquitetura de Solução](./docs/ARQUITETURA.md)**: Detalhamento do fluxo de dados e justificativa técnica do deploy síncrono.
2.  **[Plano de Monitoramento](./docs/MONITORAMENTO.md)**: Estratégia de observabilidade, detecção de *Data Drift* e política de retreinamento.
3.  **[Model Card de Governança](./docs/model_card.json)**: Metadados técnicos do modelo em produção, incluindo performance por classe e limitações.

---
**Desenvolvido por:** Eduardo – Data Engineering & Machine Learning

# 📞 Telecom Churn Analytics Pipeline

Este repositório contém uma solução end-to-end para a predição de rotatividade de clientes (churn) no setor de telecomunicações. O foco do projeto é transformar dados operacionais brutos em inteligência acionável através de uma arquitetura de Deep Learning robusta, escalável e pronta para produção.

## 🛠️ Stack Técnica e Diferenciais

*   **Core**: Python 3.12 com gerenciamento via `uv` para garantir builds ultra-rápidos e ambientes determinísticos.
*   **Modelagem**: Rede Neural Multicamadas (MLP) implementada em **PyTorch**, utilizando técnicas de **Early Stopping** e **Dropout** para garantir a generalização do modelo.
*   **MLOps & Governança**:
    *   **MLflow**: Rastreamento completo de experimentos (loss, acurácia, recall) e versionamento de artefatos.
    *   **Model Cards**: Geração automática de documentação técnica em JSON para rastreabilidade de performance.
*   **Deploy**: API de alta performance desenvolvida com **FastAPI**, totalmente conteinerizada para garantir portabilidade.

## 📉 Ciclo de Experimentos e Performance

O desenvolvimento foi baseado em testes iterativos registrados no MLflow. O objetivo principal foi a otimização do **Recall**, métrica essencial no cenário de Telecom para identificar proativamente o maior número de clientes em risco.

![Comparação de Baselines](./img/comparacao_mlflow.png)

*   **Rastreabilidade**: Monitoramento em tempo real de métricas durante o treinamento para ajuste fino da arquitetura MLP.
*   **Análise de Erro**: Comparação visual entre baselines para validação de métricas de negócio antes da promoção para produção.

## 🚀 Guia de Operação

### Execução via Docker (Produção)
Para provisionar a API de inferência e toda a infraestrutura de suporte:
```bash
docker compose up --build