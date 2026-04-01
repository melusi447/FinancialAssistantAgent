"""
Financial Assistant Agents
--------------------------
Four specialist agents, each integrating with the core LLM engine and RAG service.

Usage
-----
    from agents import BudgetAdvisorAgent, DebtCoachAgent, PortfolioAgent, SandboxExecutorAgent

    budget  = BudgetAdvisorAgent()
    debt    = DebtCoachAgent()
    portfolio = PortfolioAgent()
    sandbox = SandboxExecutorAgent()
"""

from agents.budget_advisor import BudgetAdvisorAgent, BudgetProfile, BUDGET_BENCHMARKS
from agents.debt_coach import DebtCoachAgent, Debt, RepaymentPlan
from agents.portfolio_agent import PortfolioAgent, Holding, PortfolioAnalysis, RISK_PROFILES
from agents.sandbox_executor import SandboxExecutorAgent

__all__ = [
    "BudgetAdvisorAgent",
    "BudgetProfile",
    "BUDGET_BENCHMARKS",
    "DebtCoachAgent",
    "Debt",
    "RepaymentPlan",
    "PortfolioAgent",
    "Holding",
    "PortfolioAnalysis",
    "RISK_PROFILES",
    "SandboxExecutorAgent",
]
