# Arquitetura de Deploy - Predição de Churn

## 1. Visão Geral
A solução foi desenhada utilizando uma arquitetura de **Microserviços**, focada em escalabilidade e portabilidade através de containers Docker. O objetivo é fornecer predições de churn em tempo real para integrar com sistemas de CRM ou atendimento ao cliente.

## 2. Padrão de Deploy: Real-Time (Online Inference)
Optamos pelo deploy em **tempo real** em vez de processamento em batch.

*   **Justificativa**: A retenção de clientes de telecomunicações é mais eficaz quando a probabilidade de cancelamento é conhecida no momento do contato ou de uma ação do usuário. O processamento em batch (lote) poderia gerar informações defasadas para decisões imediatas.
*   **Protocolo**: REST via HTTP/JSON.

## 3. Componentes da Solução
*   **API Layer**: Desenvolvida com **FastAPI**, escolhida pela alta performance (assíncrona) e validação automática de dados com Pydantic.
*   **Inference Engine**: Modelo **MLP (PyTorch)** carregado em memória para baixíssima latência.
*   **Data Pipeline**: Pipeline de pré-processamento serializado com **Joblib**, garantindo que as transformações aplicadas no treino sejam idênticas na inferência.
*   **Containerização**: **Docker** para isolar dependências e garantir que o sistema rode de forma idêntica em qualquer infraestrutura.

## 4. Fluxo de Dados
1. O cliente/sistema envia um JSON com dados do usuário (Tenure, MonthlyCharges, etc).
2. A API valida o schema (via Pydantic).
3. O Pipeline de Engenharia aplica o Scaling e Encoding.
4. O Modelo realiza a inferência e retorna a probabilidade de Churn.