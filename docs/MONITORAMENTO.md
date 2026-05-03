# Plano de Monitoramento e Manutenção - Churn MLP

## 1. Visão Geral
Este documento estabelece a estratégia de sustentação para o modelo de predição de churn, visando garantir que a performance observada durante o Tech Challenge se mantenha estável em ambiente produtivo.

## 2. Monitoramento de Infraestrutura (Saúde da API)
O acompanhamento técnico foca na disponibilidade da solução conteinerizada:
*   **Disponibilidade (Uptime)**: Verificação de status do container Docker `fiap_churn_app`.
*   **Latência de Inferência**: Tempo total entre a recepção dos dados e o retorno da probabilidade de churn (Target: < 150ms).
*   **Taxa de Erros**: Monitoramento de respostas HTTP 4xx (erros de validação do cliente) e 5xx (falhas internas do servidor).
*   **Saúde de Recursos**: Uso de CPU e Memória para evitar degradação por vazamento de memória (*memory leak*) no PyTorch.

## 3. Monitoramento de Modelo (Performance MLOps)
Como o comportamento do cliente no setor de transportes é dinâmico, monitoramos o *Model Decay*:
*   **Data Drift**: Identificar variações estatísticas significativas nas variáveis de entrada (ex: aumento súbito na média de `MonthlyCharges`).
*   **Prediction Drift**: Monitorar se a distribuição das predições está mudando drasticamente (ex: o modelo passou a classificar 90% da base como churn).
*   **Validação Mensal**: Confrontar as predições salvas com o evento real de churn ocorrido após 30 dias para recalcular o F1-Score.

## 4. Estratégia de Retreinamento
O ciclo de vida do modelo prevê atualizações baseadas em gatilhos:
1.  **Degradação de Performance**: Queda superior a 15% no *Recall* em relação ao registrado no `docs/model_card.json`.
2.  **Sazonalidade**: Retreinamento programado a cada 90 dias para capturar novos padrões de consumo.
3.  **Geração de Artefatos**: Cada novo treino deve gerar automaticamente um novo Model Card para garantir a governança e transparência do processo.

## 5. Plano de Resposta
Em caso de falha crítica ou degradação severa, o procedimento padrão é o **Rollback** para a imagem Docker anterior estável, garantindo a continuidade do serviço enquanto o diagnóstico do drift é realizado.