"""
Debt Coach Agent
Helps users understand, prioritise, and eliminate debts using the Avalanche
(highest-interest-first) and Snowball (smallest-balance-first) strategies.
Integrates with core LLM engine and RAG service.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Debt:
    """Represents a single debt account."""
    name: str
    balance: float           # Current outstanding balance
    interest_rate: float     # Annual rate as a percentage, e.g. 18.5
    minimum_payment: float
    currency: str = "USD"

    @property
    def monthly_interest(self) -> float:
        return self.balance * (self.interest_rate / 100 / 12)


@dataclass
class RepaymentPlan:
    """Repayment schedule produced for a chosen strategy."""
    strategy: str
    ordered_debts: List[Debt] = field(default_factory=list)
    monthly_payment: float = 0.0
    minimum_total: float = 0.0
    extra_payment: float = 0.0
    total_interest_paid: float = 0.0
    months_to_payoff: int = 0
    payoff_schedule: List[Dict[str, Any]] = field(default_factory=list)


class DebtCoachAgent:
    """
    Agent that coaches users toward becoming debt-free.

    Quick start
    -----------
    >>> agent = DebtCoachAgent()
    >>> debts = [
    ...     {"name": "Credit Card", "balance": 3000, "interest_rate": 22.0, "minimum_payment": 60},
    ...     {"name": "Car Loan",    "balance": 8000, "interest_rate":  6.5, "minimum_payment": 200},
    ... ]
    >>> result = agent.analyze(debts=debts, monthly_payment=500)
    >>> print(result["advice"])
    """

    STRATEGIES = ("avalanche", "snowball")

    def __init__(self) -> None:
        try:
            from core.llm_engine import llm_engine
            from core.rag_service import rag_service
            self._llm = llm_engine
            self._rag = rag_service
            logger.info("DebtCoachAgent initialised with LLM + RAG")
        except Exception as exc:
            logger.warning(f"DebtCoachAgent running without LLM/RAG: {exc}")
            self._llm = None
            self._rag = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        debts: List[Dict[str, Any]],
        monthly_payment: float,
        strategy: str = "avalanche",
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyse debts and build a repayment plan.

        Parameters
        ----------
        debts : list of dicts, each with keys:
            name, balance, interest_rate, minimum_payment, currency (optional)
        monthly_payment : float
            Total amount available for debt repayment each month.
        strategy : "avalanche" | "snowball"
        history : prior conversation turns for LLM context.

        Returns a dict with:
            plan        – RepaymentPlan dataclass
            comparison  – side-by-side avalanche vs snowball summary
            advice      – LLM narrative
            rag_docs    – retrieved knowledge snippets
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")

        debt_objects = [self._dict_to_debt(d) for d in debts]
        plan = self._build_plan(debt_objects, monthly_payment, strategy)
        comparison = self._compare_strategies(debt_objects, monthly_payment)
        rag_docs = self._retrieve_context("debt repayment strategies interest savings")
        advice = self._llm_advice(plan, comparison, rag_docs, history or [])

        return {
            "plan": plan,
            "comparison": comparison,
            "advice": advice,
            "rag_docs": rag_docs,
        }

    def chat(
        self,
        message: str,
        history: Optional[List[Tuple[str, str]]] = None,
        debts: Optional[List[Dict[str, Any]]] = None,
        monthly_payment: float = 0.0,
    ) -> str:
        """Free-form debt coaching chat."""
        if not self._llm:
            return self._fallback_response()

        context_block = ""
        if debts and monthly_payment:
            debt_objects = [self._dict_to_debt(d) for d in debts]
            context_block = self._format_debt_summary(debt_objects, monthly_payment)

        # Compact: skip heavy RAG in chat() — llm_engine handles it internally
        system_prompt = self._build_system_prompt(context_block, "")

        try:
            return self._llm.chat_with_context(message, history or [])
        except Exception as exc:
            logger.error(f"DebtCoachAgent LLM error: {exc}")
            return self._fallback_response()

    # ------------------------------------------------------------------ #
    # Plan building                                                        #
    # ------------------------------------------------------------------ #

    def _build_plan(
        self, debts: List[Debt], monthly_payment: float, strategy: str
    ) -> RepaymentPlan:
        minimum_total = sum(d.minimum_payment for d in debts)
        extra = max(monthly_payment - minimum_total, 0)

        # Order debts by strategy
        if strategy == "avalanche":
            ordered = sorted(debts, key=lambda d: -d.interest_rate)
        else:  # snowball
            ordered = sorted(debts, key=lambda d: d.balance)

        plan = RepaymentPlan(
            strategy=strategy,
            ordered_debts=ordered,
            monthly_payment=monthly_payment,
            minimum_total=minimum_total,
            extra_payment=extra,
        )

        plan.total_interest_paid, plan.months_to_payoff, plan.payoff_schedule = (
            self._simulate_payoff(ordered, monthly_payment)
        )
        return plan

    def _simulate_payoff(
        self, ordered_debts: List[Debt], monthly_payment: float
    ) -> Tuple[float, int, List[Dict[str, Any]]]:
        """Simulate month-by-month payoff. Returns (total_interest, months, schedule)."""
        # Work on mutable copies of balances
        balances = {d.name: d.balance for d in ordered_debts}
        total_interest = 0.0
        month = 0
        schedule: List[Dict[str, Any]] = []
        max_months = 600  # 50-year cap to prevent infinite loops

        while any(b > 0 for b in balances.values()) and month < max_months:
            month += 1
            available = monthly_payment
            month_interest = 0.0
            month_log: Dict[str, Any] = {"month": month, "payments": {}}

            # Apply interest and minimums first
            for debt in ordered_debts:
                if balances[debt.name] <= 0:
                    continue
                interest = balances[debt.name] * (debt.interest_rate / 100 / 12)
                balances[debt.name] += interest
                total_interest += interest
                month_interest += interest

                # Pay minimum (or remaining balance, whichever is smaller)
                min_pay = min(debt.minimum_payment, balances[debt.name])
                balances[debt.name] -= min_pay
                available -= min_pay
                month_log["payments"][debt.name] = round(min_pay, 2)

            # Apply extra payment to the priority debt
            for debt in ordered_debts:
                if balances[debt.name] <= 0 or available <= 0:
                    continue
                extra = min(available, balances[debt.name])
                balances[debt.name] -= extra
                available -= extra
                month_log["payments"][debt.name] = (
                    month_log["payments"].get(debt.name, 0) + round(extra, 2)
                )

            month_log["remaining"] = {k: round(v, 2) for k, v in balances.items()}
            month_log["interest_charged"] = round(month_interest, 2)
            schedule.append(month_log)

        return round(total_interest, 2), month, schedule

    def _compare_strategies(
        self, debts: List[Debt], monthly_payment: float
    ) -> Dict[str, Any]:
        """Return a side-by-side comparison of avalanche vs snowball."""
        results = {}
        for strategy in self.STRATEGIES:
            plan = self._build_plan(debts, monthly_payment, strategy)
            results[strategy] = {
                "months_to_payoff": plan.months_to_payoff,
                "total_interest_paid": plan.total_interest_paid,
                "payoff_order": [d.name for d in plan.ordered_debts],
            }

        # Savings from choosing avalanche over snowball
        interest_saved = round(
            results["snowball"]["total_interest_paid"]
            - results["avalanche"]["total_interest_paid"],
            2,
        )
        months_saved = (
            results["snowball"]["months_to_payoff"]
            - results["avalanche"]["months_to_payoff"]
        )
        results["recommendation"] = (
            "avalanche" if interest_saved >= 0 else "snowball"
        )
        results["interest_saved_by_avalanche"] = interest_saved
        results["months_saved_by_avalanche"] = months_saved
        return results

    # ------------------------------------------------------------------ #
    # Formatting & prompting                                               #
    # ------------------------------------------------------------------ #

    def _format_debt_summary(self, debts: List[Debt], monthly_payment: float) -> str:
        currency = debts[0].currency if debts else "USD"
        total_balance = sum(d.balance for d in debts)
        total_min = sum(d.minimum_payment for d in debts)
        lines = [
            f"Total debt     : {total_balance:,.2f} {currency}",
            f"Monthly budget : {monthly_payment:,.2f} {currency}",
            f"Minimum total  : {total_min:,.2f} {currency}",
            f"Extra available: {max(monthly_payment - total_min, 0):,.2f} {currency}",
            "\nDebt accounts:",
        ]
        for d in sorted(debts, key=lambda x: -x.interest_rate):
            lines.append(
                f"  {d.name:<22} Balance: {d.balance:>10,.2f}  "
                f"Rate: {d.interest_rate:>5.1f}%  Min: {d.minimum_payment:>8,.2f}"
            )
        return "\n".join(lines)

    def _build_system_prompt(self, context_block: str, rag_block: str) -> str:
        # Compact prompt — keeps token count low on CPU
        parts = ["You are FinanceBot. Give concise debt repayment advice."]
        if context_block:
            parts += ["\nProfile:", context_block[:300]]
        return " ".join(parts)

    def _llm_advice(
        self,
        plan: RepaymentPlan,
        comparison: Dict[str, Any],
        rag_docs: List[str],
        history: List[Tuple[str, str]],
    ) -> str:
        if not self._llm:
            return self._rule_based_summary(plan, comparison)

        currency = plan.ordered_debts[0].currency if plan.ordered_debts else "USD"
        rag_block = "\n\n".join(rag_docs) if rag_docs else ""
        debt_summary = self._format_debt_summary(plan.ordered_debts, plan.monthly_payment)

        avalanche = comparison.get("avalanche", {})
        snowball = comparison.get("snowball", {})
        user_prompt = (
            f"Debt overview:\n{debt_summary}\n\n"
            f"Chosen strategy : {plan.strategy}\n"
            f"Months to payoff: {plan.months_to_payoff}\n"
            f"Total interest  : {plan.total_interest_paid:,.2f} {currency}\n\n"
            f"Avalanche — {avalanche.get('months_to_payoff')} months, "
            f"{avalanche.get('total_interest_paid'):,.2f} interest\n"
            f"Snowball  — {snowball.get('months_to_payoff')} months, "
            f"{snowball.get('total_interest_paid'):,.2f} interest\n"
            f"Interest saved by avalanche: {comparison.get('interest_saved_by_avalanche'):,.2f}\n\n"
            "Provide a comprehensive debt repayment plan with motivational coaching, "
            "strategy explanation, and a clear action plan."
        )
        try:
            return self._llm.chat(
                message=user_prompt,
                history=history,
                system_prompt=self._build_system_prompt("", rag_block),
            )
        except Exception as exc:
            logger.error(f"LLM advice generation failed: {exc}")
            return self._rule_based_summary(plan, comparison)

    @staticmethod
    def _rule_based_summary(plan: RepaymentPlan, comparison: Dict[str, Any]) -> str:
        currency = plan.ordered_debts[0].currency if plan.ordered_debts else "USD"
        saved = comparison.get("interest_saved_by_avalanche", 0)
        rec = comparison.get("recommendation", "avalanche")
        return (
            f"Using the {plan.strategy} strategy, you'll be debt-free in "
            f"{plan.months_to_payoff} months, paying {plan.total_interest_paid:,.2f} "
            f"{currency} in interest.\n\n"
            f"Recommended strategy: {rec.title()} "
            f"(saves {abs(saved):,.2f} {currency} in interest).\n\n"
            f"Payoff order: {' → '.join(d.name for d in plan.ordered_debts)}"
        )

    def _retrieve_context(self, query: str) -> List[str]:
        if not self._rag:
            return []
        try:
            return self._rag.retrieve_docs(query, k=3)
        except Exception as exc:
            logger.warning(f"RAG retrieval failed: {exc}")
            return []

    @staticmethod
    def _dict_to_debt(d: Dict[str, Any]) -> Debt:
        return Debt(
            name=d["name"],
            balance=float(d["balance"]),
            interest_rate=float(d["interest_rate"]),
            minimum_payment=float(d["minimum_payment"]),
            currency=d.get("currency", "USD"),
        )

    @staticmethod
    def _fallback_response() -> str:
        return (
            "I'm your Debt Coach. Share your debts (name, balance, interest rate, "
            "minimum payment) and your monthly repayment budget and I'll build you "
            "a step-by-step payoff plan."
        )
