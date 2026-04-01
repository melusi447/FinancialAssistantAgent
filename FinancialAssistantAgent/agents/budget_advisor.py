"""
Budget Advisor Agent
Analyses income, expenses, and spending patterns to give actionable budgeting advice.
Integrates with core LLM engine and RAG service.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 50/30/20 benchmark percentages per category
BUDGET_BENCHMARKS: Dict[str, float] = {
    "housing": 30.0,
    "food": 10.0,
    "transport": 15.0,
    "utilities": 5.0,
    "entertainment": 5.0,
    "healthcare": 5.0,
    "personal": 5.0,
    "savings": 20.0,
}


@dataclass
class BudgetProfile:
    """Holds a user's budget data for a single analysis session."""
    monthly_income: float = 0.0
    expenses: Dict[str, float] = field(default_factory=dict)
    savings_goal_pct: float = 20.0
    currency: str = "USD"
    # Derived — populated by compute()
    total_expenses: float = 0.0
    savings_actual: float = 0.0
    savings_gap: float = 0.0
    expense_breakdown_pct: Dict[str, float] = field(default_factory=dict)

    def compute(self) -> None:
        self.total_expenses = sum(self.expenses.values())
        self.savings_actual = self.monthly_income - self.total_expenses
        savings_target = self.monthly_income * (self.savings_goal_pct / 100)
        self.savings_gap = self.savings_actual - savings_target
        if self.monthly_income > 0:
            self.expense_breakdown_pct = {
                cat: round((amt / self.monthly_income) * 100, 1)
                for cat, amt in self.expenses.items()
            }


class BudgetAdvisorAgent:
    """
    Agent that provides personalised budgeting advice.

    Quick start
    -----------
    >>> agent = BudgetAdvisorAgent()
    >>> result = agent.analyze(income=5000, expenses={"housing": 1500, "food": 600})
    >>> print(result["advice"])
    """

    def __init__(self) -> None:
        try:
            from core.llm_engine import llm_engine
            from core.rag_service import rag_service
            self._llm = llm_engine
            self._rag = rag_service
            logger.info("BudgetAdvisorAgent initialised with LLM + RAG")
        except Exception as exc:
            logger.warning(f"BudgetAdvisorAgent running without LLM/RAG: {exc}")
            self._llm = None
            self._rag = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        income: float,
        expenses: Dict[str, float],
        savings_goal_pct: float = 20.0,
        currency: str = "USD",
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyse a budget and return structured advice.

        Returns a dict with:
            profile      – BudgetProfile with computed metrics
            flags        – over-budget category warnings
            suggestions  – rule-based savings tips
            advice       – LLM-generated narrative
            rag_docs     – retrieved knowledge-base snippets
        """
        profile = BudgetProfile(
            monthly_income=income,
            expenses=expenses,
            savings_goal_pct=savings_goal_pct,
            currency=currency,
        )
        profile.compute()

        flags = self._flag_overspend(profile)
        suggestions = self._rule_based_suggestions(profile)
        rag_docs = self._retrieve_context("budgeting saving money expenses")
        advice = self._llm_advice(profile, flags, suggestions, rag_docs, history or [])

        return {
            "profile": profile,
            "flags": flags,
            "suggestions": suggestions,
            "advice": advice,
            "rag_docs": rag_docs,
        }

    def chat(
        self,
        message: str,
        history: Optional[List[Tuple[str, str]]] = None,
        income: float = 0.0,
        expenses: Optional[Dict[str, float]] = None,
    ) -> str:
        """Free-form budget chat, optionally enriched with the user's budget data."""
        if not self._llm:
            return self._fallback_response()

        context_block = ""
        if income and expenses:
            profile = BudgetProfile(monthly_income=income, expenses=expenses)
            profile.compute()
            context_block = self._format_profile(profile)

        # Compact: skip heavy RAG in chat() — llm_engine handles it internally
        system_prompt = self._build_system_prompt(context_block, "")

        try:
            return self._llm.chat_with_context(message, history or [])
        except Exception as exc:
            logger.error(f"BudgetAdvisorAgent LLM error: {exc}")
            return self._fallback_response()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _flag_overspend(self, p: BudgetProfile) -> List[str]:
        flags: List[str] = []
        if p.monthly_income <= 0:
            return flags
        for category, pct in p.expense_breakdown_pct.items():
            benchmark = BUDGET_BENCHMARKS.get(category.lower())
            if benchmark and pct > benchmark:
                flags.append(
                    f"⚠️  {category.title()} is {pct}% of income (recommended ≤ {benchmark}%)"
                )
        if p.savings_actual < 0:
            flags.append(
                f"🚨 Spending {abs(p.savings_actual):,.2f} {p.currency} MORE than earned monthly."
            )
        elif p.savings_gap < 0:
            flags.append(
                f"⚠️  Savings shortfall of {abs(p.savings_gap):,.2f} {p.currency} "
                f"vs. {p.savings_goal_pct}% goal."
            )
        return flags

    def _rule_based_suggestions(self, p: BudgetProfile) -> List[str]:
        tips: List[str] = []
        if p.savings_gap < 0:
            tips.append(
                f"💡 Reduce spending by {abs(p.savings_gap):,.2f} {p.currency}/month to hit your savings goal."
            )
        for category, pct in p.expense_breakdown_pct.items():
            benchmark = BUDGET_BENCHMARKS.get(category.lower(), 0)
            if benchmark and pct > benchmark * 1.25:
                excess = p.expenses[category] - (p.monthly_income * benchmark / 100)
                tips.append(
                    f"💡 Cut {category.title()} by ~{excess:,.2f} {p.currency}/month "
                    f"to meet the {benchmark}% guideline."
                )
        if 0 < p.savings_actual < p.monthly_income * 0.05:
            tips.append(
                "💡 Savings are very low. Build a 3–6 month emergency fund before investing."
            )
        if not tips:
            tips.append(
                "✅ Budget looks healthy! Maintain the 50/30/20 rule: "
                "50% needs, 30% wants, 20% savings."
            )
        return tips

    def _retrieve_context(self, query: str) -> List[str]:
        if not self._rag:
            return []
        try:
            return self._rag.retrieve_docs(query, k=3)
        except Exception as exc:
            logger.warning(f"RAG retrieval failed: {exc}")
            return []

    def _format_profile(self, p: BudgetProfile) -> str:
        lines = [
            f"Monthly income : {p.monthly_income:,.2f} {p.currency}",
            f"Total expenses : {p.total_expenses:,.2f} {p.currency}",
            f"Net savings    : {p.savings_actual:,.2f} {p.currency}",
            f"Savings goal   : {p.savings_goal_pct}% "
            f"({p.monthly_income * p.savings_goal_pct / 100:,.2f} {p.currency})",
        ]
        if p.expense_breakdown_pct:
            lines.append("\nExpense breakdown (% of income):")
            for cat, pct in sorted(p.expense_breakdown_pct.items(), key=lambda x: -x[1]):
                lines.append(f"  {cat.title():<18} {pct:5.1f}%")
        return "\n".join(lines)

    def _build_system_prompt(self, context_block: str, rag_block: str) -> str:
        # Compact prompt — keeps token count low on CPU
        parts = ["You are FinanceBot. Give concise budgeting advice."]
        if context_block:
            parts += ["\nProfile:", context_block[:300]]
        return " ".join(parts)

    def _llm_advice(
        self,
        profile: BudgetProfile,
        flags: List[str],
        suggestions: List[str],
        rag_docs: List[str],
        history: List[Tuple[str, str]],
    ) -> str:
        if not self._llm:
            return "\n".join(suggestions) or self._fallback_response()

        rag_block = "\n\n".join(rag_docs) if rag_docs else ""
        flag_text = "\n".join(flags) or "No critical issues detected."
        suggestion_text = "\n".join(suggestions)
        user_prompt = (
            f"Here is my monthly budget:\n{self._format_profile(profile)}\n\n"
            f"Issues flagged:\n{flag_text}\n\n"
            f"Initial suggestions:\n{suggestion_text}\n\n"
            "Provide a comprehensive, personalised budget review with a prioritised action plan."
        )
        try:
            return self._llm.chat(
                message=user_prompt,
                history=history,
                system_prompt=self._build_system_prompt("", rag_block),
            )
        except Exception as exc:
            logger.error(f"LLM advice generation failed: {exc}")
            return "\n".join(suggestions)

    @staticmethod
    def _fallback_response() -> str:
        return (
            "I'm your Budget Advisor. Share your monthly income and expense "
            "breakdown and I'll give you personalised advice."
        )

