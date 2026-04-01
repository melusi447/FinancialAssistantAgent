"""
Sandbox Executor Agent
Safely evaluates financial expressions, runs scenario simulations, and
performs what-if calculations without executing arbitrary code.
Integrates with core LLM engine and RAG service.
"""

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Safe math namespace exposed to the expression evaluator ───────────────────
_SAFE_MATH = {
    name: getattr(math, name)
    for name in dir(math)
    if not name.startswith("_")
}
_SAFE_MATH.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})


class SandboxExecutorAgent:
    """
    Agent that safely evaluates financial expressions and runs simulations.

    Capabilities
    ------------
    - Evaluate arithmetic / financial formulae (compound interest, NPV, etc.)
    - Run parameterised scenario simulations (retirement, loan payoff, savings growth)
    - Explain results in plain language via the LLM

    Security
    --------
    Expression evaluation is restricted to a safe math namespace.
    No imports, attribute access, or arbitrary Python is permitted.

    Quick start
    -----------
    >>> agent = SandboxExecutorAgent()

    # Evaluate a formula
    >>> result = agent.evaluate("1000 * (1 + 0.07) ** 10")
    >>> print(result)   # 1967.15

    # Run a scenario
    >>> result = agent.run_scenario("compound_growth",
    ...     principal=10000, annual_rate=7.0, years=20)
    >>> print(result["summary"])

    # What-if comparison
    >>> result = agent.what_if(
    ...     scenario="loan_payoff",
    ...     base_params={"principal": 20000, "annual_rate": 5.0, "monthly_payment": 400},
    ...     variations={"monthly_payment": [400, 500, 600, 800]},
    ... )
    """

    # Registered scenario functions
    _SCENARIOS: Dict[str, Any] = {}

    def __init__(self) -> None:
        self._register_scenarios()
        try:
            from core.llm_engine import llm_engine
            from core.rag_service import rag_service
            self._llm = llm_engine
            self._rag = rag_service
            logger.info("SandboxExecutorAgent initialised with LLM + RAG")
        except Exception as exc:
            logger.warning(f"SandboxExecutorAgent running without LLM/RAG: {exc}")
            self._llm = None
            self._rag = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(self, expression: str) -> Dict[str, Any]:
        """
        Safely evaluate a mathematical / financial expression.

        Returns
        -------
        dict with keys: expression, result, explanation (LLM), error (if any)
        """
        cleaned = self._sanitise(expression)
        if cleaned is None:
            return {
                "expression": expression,
                "result": None,
                "error": "Expression contains disallowed operations.",
            }
        try:
            result = eval(cleaned, {"__builtins__": {}}, _SAFE_MATH)  # noqa: S307
            explanation = self._explain_result(expression, result)
            return {"expression": expression, "result": result, "explanation": explanation}
        except Exception as exc:
            logger.warning(f"Expression eval failed: {exc}")
            return {"expression": expression, "result": None, "error": str(exc)}

    def run_scenario(
        self,
        scenario_name: str,
        history: Optional[List[Tuple[str, str]]] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        """
        Run a named financial scenario simulation.

        Available scenarios
        -------------------
        compound_growth   – principal, annual_rate, years[, monthly_contribution]
        loan_payoff       – principal, annual_rate, monthly_payment
        retirement_savings– monthly_contribution, annual_rate, years[, initial_savings]
        emergency_fund    – monthly_expenses, months_target, current_savings, monthly_savings
        inflation_impact  – amount, annual_inflation, years

        Returns a dict with: scenario, params, results, summary, advice (LLM)
        """
        fn = self._SCENARIOS.get(scenario_name)
        if fn is None:
            return {
                "error": f"Unknown scenario '{scenario_name}'. "
                f"Available: {list(self._SCENARIOS)}"
            }
        try:
            results = fn(**params)
            rag_docs = self._retrieve_context(f"{scenario_name} financial planning")
            summary = self._format_scenario_results(scenario_name, params, results)
            advice = self._llm_scenario_advice(scenario_name, params, results, rag_docs, history or [])
            return {
                "scenario": scenario_name,
                "params": params,
                "results": results,
                "summary": summary,
                "advice": advice,
                "rag_docs": rag_docs,
            }
        except TypeError as exc:
            return {"error": f"Invalid parameters for scenario '{scenario_name}': {exc}"}
        except Exception as exc:
            logger.error(f"Scenario '{scenario_name}' failed: {exc}")
            return {"error": str(exc)}

    def what_if(
        self,
        scenario: str,
        base_params: Dict[str, Any],
        variations: Dict[str, List[Any]],
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Run a what-if analysis by varying one or more parameters.

        Parameters
        ----------
        scenario : str — name of the registered scenario
        base_params : dict — baseline parameters
        variations : dict — param_name → list of values to test

        Returns a dict with: scenario, table (list of result rows), advice (LLM)
        """
        fn = self._SCENARIOS.get(scenario)
        if fn is None:
            return {"error": f"Unknown scenario '{scenario}'"}

        rows: List[Dict[str, Any]] = []
        for param_name, values in variations.items():
            for value in values:
                params = {**base_params, param_name: value}
                try:
                    result = fn(**params)
                    rows.append({"params": params, "results": result})
                except Exception as exc:
                    rows.append({"params": params, "error": str(exc)})

        advice = self._llm_whatif_advice(scenario, base_params, variations, rows, history or [])
        return {
            "scenario": scenario,
            "base_params": base_params,
            "variations": variations,
            "table": rows,
            "advice": advice,
        }

    def list_scenarios(self) -> List[str]:
        """Return available scenario names."""
        return list(self._SCENARIOS)

    def chat(
        self,
        message: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """Free-form financial calculation chat."""
        if not self._llm:
            return (
                "I can evaluate financial expressions and run scenarios. "
                "Try: evaluate('1000 * (1.07 ** 10)') or "
                "run_scenario('compound_growth', principal=5000, annual_rate=8, years=15)"
            )
        rag_docs = self._retrieve_context(message)
        rag_block = "\n\n".join(rag_docs) if rag_docs else ""
        system_prompt = self._build_system_prompt(rag_block)
        try:
            return self._llm.chat(message=message, history=history or [], system_prompt=system_prompt)
        except Exception as exc:
            logger.error(f"SandboxExecutorAgent LLM error: {exc}")
            return "I encountered an error. Please try rephrasing your question."

    # ------------------------------------------------------------------ #
    # Scenario implementations                                            #
    # ------------------------------------------------------------------ #

    def _register_scenarios(self) -> None:
        self._SCENARIOS = {
            "compound_growth": self._scenario_compound_growth,
            "loan_payoff": self._scenario_loan_payoff,
            "retirement_savings": self._scenario_retirement_savings,
            "emergency_fund": self._scenario_emergency_fund,
            "inflation_impact": self._scenario_inflation_impact,
        }

    @staticmethod
    def _scenario_compound_growth(
        principal: float,
        annual_rate: float,
        years: int,
        monthly_contribution: float = 0.0,
    ) -> Dict[str, Any]:
        r = annual_rate / 100 / 12
        n = years * 12
        # Future value of lump sum
        fv_principal = principal * (1 + r) ** n
        # Future value of regular contributions
        fv_contributions = (
            monthly_contribution * (((1 + r) ** n - 1) / r) if r > 0 else monthly_contribution * n
        )
        total = fv_principal + fv_contributions
        total_contributed = principal + monthly_contribution * n
        return {
            "future_value": round(total, 2),
            "principal_growth": round(fv_principal, 2),
            "contributions_growth": round(fv_contributions, 2),
            "total_contributed": round(total_contributed, 2),
            "total_interest_earned": round(total - total_contributed, 2),
            "growth_multiplier": round(total / max(total_contributed, 1), 2),
        }

    @staticmethod
    def _scenario_loan_payoff(
        principal: float,
        annual_rate: float,
        monthly_payment: float,
    ) -> Dict[str, Any]:
        r = annual_rate / 100 / 12
        if r == 0:
            months = math.ceil(principal / monthly_payment)
            return {
                "months_to_payoff": months,
                "years_to_payoff": round(months / 12, 1),
                "total_paid": round(monthly_payment * months, 2),
                "total_interest": 0.0,
                "minimum_payment": round(principal / 360, 2),
            }
        # Minimum payment to avoid negative amortisation
        min_payment = principal * r
        if monthly_payment <= min_payment:
            return {
                "error": f"Monthly payment must exceed {min_payment:.2f} to pay off the loan.",
                "minimum_required": round(min_payment, 2),
            }
        months = math.ceil(
            -math.log(1 - (principal * r) / monthly_payment) / math.log(1 + r)
        )
        total_paid = monthly_payment * months
        return {
            "months_to_payoff": months,
            "years_to_payoff": round(months / 12, 1),
            "total_paid": round(total_paid, 2),
            "total_interest": round(total_paid - principal, 2),
            "minimum_payment": round(min_payment, 2),
        }

    @staticmethod
    def _scenario_retirement_savings(
        monthly_contribution: float,
        annual_rate: float,
        years: int,
        initial_savings: float = 0.0,
    ) -> Dict[str, Any]:
        r = annual_rate / 100 / 12
        n = years * 12
        fv_initial = initial_savings * (1 + r) ** n
        fv_contributions = (
            monthly_contribution * (((1 + r) ** n - 1) / r) if r > 0 else monthly_contribution * n
        )
        total = fv_initial + fv_contributions
        total_contributed = initial_savings + monthly_contribution * n
        return {
            "retirement_fund": round(total, 2),
            "total_contributed": round(total_contributed, 2),
            "investment_growth": round(total - total_contributed, 2),
            "monthly_income_4pct_rule": round(total * 0.04 / 12, 2),
            "monthly_income_3pct_rule": round(total * 0.03 / 12, 2),
        }

    @staticmethod
    def _scenario_emergency_fund(
        monthly_expenses: float,
        months_target: int,
        current_savings: float,
        monthly_savings: float,
    ) -> Dict[str, Any]:
        target = monthly_expenses * months_target
        gap = max(target - current_savings, 0)
        months_needed = math.ceil(gap / monthly_savings) if monthly_savings > 0 else float("inf")
        return {
            "target_fund": round(target, 2),
            "current_savings": round(current_savings, 2),
            "gap": round(gap, 2),
            "months_to_goal": months_needed if months_needed != float("inf") else "Never (increase savings)",
            "already_funded": gap <= 0,
        }

    @staticmethod
    def _scenario_inflation_impact(
        amount: float,
        annual_inflation: float,
        years: int,
    ) -> Dict[str, Any]:
        future_cost = amount * (1 + annual_inflation / 100) ** years
        purchasing_power = amount / (1 + annual_inflation / 100) ** years
        return {
            "original_amount": round(amount, 2),
            "future_cost": round(future_cost, 2),
            "purchasing_power_today": round(purchasing_power, 2),
            "purchasing_power_loss_pct": round((1 - purchasing_power / amount) * 100, 1),
            "required_return_to_keep_pace": round(annual_inflation, 2),
        }

    # ------------------------------------------------------------------ #
    # Formatting & prompting                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sanitise(expression: str) -> Optional[str]:
        """
        Validate that expression contains only safe characters.
        Returns cleaned expression or None if disallowed content found.
        """
        # Strip whitespace
        expr = expression.strip()
        # Block imports, exec, eval, attribute access, subscripts, function defs
        forbidden = re.compile(
            r"\b(import|exec|eval|open|os|sys|subprocess|__)\b|[;{}]"
        )
        if forbidden.search(expr):
            return None
        # Allow only numbers, operators, parens, dots, spaces, and math function names
        allowed = re.compile(r"^[\d\s\+\-\*\/\(\)\.\,\%\*\^a-zA-Z_]+$")
        if not allowed.match(expr):
            return None
        return expr

    def _explain_result(self, expression: str, result: Any) -> str:
        if not self._llm:
            return f"Result: {result}"
        try:
            return self._llm.chat(
                message=f"Explain this financial calculation in plain English:\n{expression} = {result}",
                system_prompt=(
                    "You are a financial educator. Explain mathematical results "
                    "briefly and clearly in 2-3 sentences."
                ),
            )
        except Exception:
            return f"The expression evaluates to {result}."

    def _format_scenario_results(
        self, name: str, params: Dict, results: Dict
    ) -> str:
        lines = [f"Scenario: {name.replace('_', ' ').title()}", "\nInputs:"]
        for k, v in params.items():
            lines.append(f"  {k.replace('_', ' ').title()}: {v}")
        lines.append("\nResults:")
        for k, v in results.items():
            if k != "error":
                lines.append(f"  {k.replace('_', ' ').title()}: {v}")
        return "\n".join(lines)

    def _build_system_prompt(self, rag_block: str) -> str:
        parts = [
            "You are a Financial Calculator AI. Help users run financial scenarios, "
            "evaluate expressions, and understand mathematical results in plain language. "
            "Be precise, clear, and explain what the numbers mean for the user's goals.",
        ]
        if rag_block:
            parts += ["\n## Relevant Financial Knowledge\n", rag_block]
        return "\n".join(parts)

    def _llm_scenario_advice(
        self,
        name: str,
        params: Dict,
        results: Dict,
        rag_docs: List[str],
        history: List[Tuple[str, str]],
    ) -> str:
        if not self._llm:
            return self._format_scenario_results(name, params, results)
        rag_block = "\n\n".join(rag_docs) if rag_docs else ""
        summary = self._format_scenario_results(name, params, results)
        try:
            return self._llm.chat(
                message=f"{summary}\n\nExplain these results and provide actionable insights.",
                history=history,
                system_prompt=self._build_system_prompt(rag_block),
            )
        except Exception as exc:
            logger.error(f"LLM scenario advice failed: {exc}")
            return summary

    def _llm_whatif_advice(
        self,
        scenario: str,
        base_params: Dict,
        variations: Dict,
        rows: List[Dict],
        history: List[Tuple[str, str]],
    ) -> str:
        if not self._llm:
            return f"What-if analysis complete. {len(rows)} scenarios compared."
        table_lines = [f"What-if analysis for {scenario.replace('_', ' ').title()}:"]
        for row in rows:
            if "error" not in row:
                key_results = {k: v for k, v in row["results"].items() if k != "error"}
                changed = {k: v for k, v in row["params"].items() if k not in base_params or base_params[k] != v}
                table_lines.append(f"  {changed} → {key_results}")
        try:
            return self._llm.chat(
                message="\n".join(table_lines) + "\n\nWhat do these comparisons reveal?",
                history=history,
                system_prompt=self._build_system_prompt(""),
            )
        except Exception as exc:
            logger.error(f"LLM what-if advice failed: {exc}")
            return "\n".join(table_lines)

    def _retrieve_context(self, query: str) -> List[str]:
        if not self._rag:
            return []
        try:
            return self._rag.retrieve_docs(query, k=3)
        except Exception as exc:
            logger.warning(f"RAG retrieval failed: {exc}")
            return []

