import logging
from dataclasses import dataclass, field

# Configuração de log para ficar profissional no terminal
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class MLCanvas:
    project_name: str = ""
    business_problem: str = ""
    ml_task: str = ""
    success_metrics: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    target: str = ""
    constraints: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    def data_readiness_score(self) -> float:
        checks = [
            bool(self.data_sources),
            bool(self.features),
            bool(self.target),
            len(self.data_sources) >= 1,
            len(self.features) >= 5,
        ]
        return sum(checks) / len(checks)

    def is_viable(self) -> bool:
        return all([self.project_name, self.business_problem, self.ml_task, self.target, self.success_metrics])

    def display(self) -> None:
        logger.info("=" * 60)
        logger.info("ML CANVAS — %s", self.project_name)
        logger.info("=" * 60)
        logger.info("Problema de negócio: %s", self.business_problem)
        logger.info("Tarefa ML: %s", self.ml_task)
        logger.info("Variável alvo: %s", self.target)
        logger.info("Métricas de sucesso: %s", ", ".join(self.success_metrics))
        logger.info("Fontes de dados: %s", ", ".join(self.data_sources))
        logger.info("Features utilizadas: %s", ", ".join(self.features))
        logger.info("Restrições: %s", ", ".join(self.constraints) or "Nenhuma")
        logger.info("Riscos: %s", ", ".join(self.risks) or "Nenhum")
        logger.info("-" * 60)
        score = self.data_readiness_score()
        logger.info("Data Readiness Score: %.0f%%", score * 100)
        logger.info("Projeto viável para Fase 1: %s", "✓" if self.is_viable() else "✗")

def create_churn_canvas() -> MLCanvas:
    return MLCanvas(
        project_name="Predição de Churn — Telco IBM",
        business_problem=(
            "Identificar clientes em risco de cancelamento para aplicar ações de retenção. "
            "Foco em reduzir perda de receita mensal (MRR)."
        ),
        ml_task="Classificação Binária (Churn: Yes/No)",
        success_metrics=[
            "ROI de Retenção Estimado > R$ 200.000,00", 
            "F1-Score Baseline >= 0.50",
            "AUC-ROC Baseline >= 0.70"
        ],
        data_sources=["Telco_customer_churn.csv", "MLflow Tracking (Local)"],
        features=[
            "Tenure Months", "Monthly Charges", "Total Charges", 
            "Churn Score", "CLTV", "Zip Code"
        ],
        target="Churn",
        constraints=[
            "Uso obrigatório de MLflow para governança",
            "Treinamento inicial em CPU local"
        ],
        risks=[
            "Desbalanceamento de classes (mais não-churn do que churn)",
            "Necessidade de conversão de tipos em 'Total Charges'",
            "Limitação de dados demográficos no dataset público"
        ],
    )

if __name__ == "__main__":
    canvas = create_churn_canvas()
    canvas.display()