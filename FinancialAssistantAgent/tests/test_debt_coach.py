"""
Tests for DebtCoachAgent
Run with: pytest tests/test_debt_coach.py -v
"""

import pytest
from agents.debt_coach import DebtCoachAgent, Debt, RepaymentPlan

SAMPLE_DEBTS = [
    {"name": "Credit Card",  "balance": 3000, "interest_rate": 22.0, "minimum_payment": 60},
    {"name": "Car Loan",     "balance": 8000, "interest_rate":  6.5, "minimum_payment": 200},
    {"name": "Student Loan", "balance": 15000, "interest_rate":  4.5, "minimum_payment": 150},
]


# ── Debt dataclass ─────────────────────────────────────────────────────────────

class TestDebt:
    def test_monthly_interest_calculation(self):
        d = Debt(name="Card", balance=1200, interest_rate=12.0, minimum_payment=25)
        # 1200 * 0.01 = 12.0
        assert d.monthly_interest == pytest.approx(12.0, abs=0.01)

    def test_zero_balance_zero_interest(self):
        d = Debt(name="Paid", balance=0, interest_rate=18.0, minimum_payment=0)
        assert d.monthly_interest == 0.0


# ── DebtCoachAgent ─────────────────────────────────────────────────────────────

class TestDebtCoachAgent:
    @pytest.fixture
    def agent(self):
        return DebtCoachAgent()

    def test_analyze_returns_expected_keys(self, agent):
        result = agent.analyze(debts=SAMPLE_DEBTS, monthly_payment=600)
        assert "plan" in result
        assert "comparison" in result
        assert "advice" in result
        assert "rag_docs" in result

    def test_plan_is_repayment_plan(self, agent):
        result = agent.analyze(debts=SAMPLE_DEBTS, monthly_payment=600)
        assert isinstance(result["plan"], RepaymentPlan)

    def test_avalanche_orders_by_highest_rate(self, agent):
        result = agent.analyze(debts=SAMPLE_DEBTS, monthly_payment=600, strategy="avalanche")
        ordered_names = [d.name for d in result["plan"].ordered_debts]
        # Credit Card (22%) should come first
        assert ordered_names[0] == "Credit Card"

    def test_snowball_orders_by_lowest_balance(self, agent):
        result = agent.analyze(debts=SAMPLE_DEBTS, monthly_payment=600, strategy="snowball")
        ordered_names = [d.name for d in result["plan"].ordered_debts]
        # Credit Card (3000) should come first
        assert ordered_names[0] == "Credit Card"

    def test_avalanche_cheaper_than_snowball(self, agent):
        result = agent.analyze(debts=SAMPLE_DEBTS, monthly_payment=600)
        comparison = result["comparison"]
        # Avalanche should pay less total interest than snowball
        assert (
            comparison["avalanche"]["total_interest_paid"]
            <= comparison["snowball"]["total_interest_paid"]
        )

    def test_invalid_strategy_raises(self, agent):
        with pytest.raises(ValueError):
            agent.analyze(debts=SAMPLE_DEBTS, monthly_payment=600, strategy="random")

    def test_payoff_within_reasonable_time(self, agent):
        result = agent.analyze(debts=SAMPLE_DEBTS, monthly_payment=600)
        assert result["plan"].months_to_payoff < 600  # under 50 years

    def test_extra_payment_calculated_correctly(self, agent):
        result = agent.analyze(debts=SAMPLE_DEBTS, monthly_payment=600)
        plan = result["plan"]
        expected_extra = max(600 - plan.minimum_total, 0)
        assert plan.extra_payment == pytest.approx(expected_extra, abs=1)

    def test_comparison_includes_both_strategies(self, agent):
        result = agent.analyze(debts=SAMPLE_DEBTS, monthly_payment=600)
        comparison = result["comparison"]
        assert "avalanche" in comparison
        assert "snowball" in comparison
        assert "recommendation" in comparison

    def test_chat_fallback_without_llm(self, agent):
        agent._llm = None
        response = agent.chat("Which debt should I pay first?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_single_debt(self, agent):
        single = [{"name": "Card", "balance": 2000, "interest_rate": 15.0, "minimum_payment": 40}]
        result = agent.analyze(debts=single, monthly_payment=200)
        assert result["plan"].months_to_payoff > 0

