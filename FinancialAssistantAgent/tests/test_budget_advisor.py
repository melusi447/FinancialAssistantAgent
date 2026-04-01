"""
Tests for BudgetAdvisorAgent
Run with: pytest tests/test_budget_advisor.py -v
"""

import pytest
from agents.budget_advisor import BudgetAdvisorAgent, BudgetProfile, BUDGET_BENCHMARKS


# ── BudgetProfile unit tests ──────────────────────────────────────────────────

class TestBudgetProfile:
    def test_compute_basic(self):
        p = BudgetProfile(
            monthly_income=5000,
            expenses={"housing": 1500, "food": 500, "transport": 300},
        )
        p.compute()
        assert p.total_expenses == 2300
        assert p.savings_actual == 2700
        assert p.expense_breakdown_pct["housing"] == 30.0

    def test_savings_gap_when_underfunding(self):
        p = BudgetProfile(
            monthly_income=5000,
            expenses={"housing": 4500},
            savings_goal_pct=20.0,
        )
        p.compute()
        # savings_actual = 500, target = 1000, gap = -500
        assert p.savings_gap == pytest.approx(-500, abs=1)

    def test_zero_income_safe(self):
        p = BudgetProfile(monthly_income=0, expenses={"food": 300})
        p.compute()
        assert p.expense_breakdown_pct == {}

    def test_negative_savings_when_overspending(self):
        p = BudgetProfile(monthly_income=3000, expenses={"housing": 3500})
        p.compute()
        assert p.savings_actual < 0


# ── BudgetAdvisorAgent unit tests ─────────────────────────────────────────────

class TestBudgetAdvisorAgent:
    """Tests that run without LLM/RAG (agent gracefully degrades)."""

    @pytest.fixture
    def agent(self):
        return BudgetAdvisorAgent()

    def test_analyze_returns_expected_keys(self, agent):
        result = agent.analyze(
            income=4000,
            expenses={"housing": 1200, "food": 400, "transport": 300},
        )
        assert "profile" in result
        assert "flags" in result
        assert "suggestions" in result
        assert "advice" in result
        assert "rag_docs" in result

    def test_overspend_flag_triggered(self, agent):
        result = agent.analyze(
            income=3000,
            expenses={"housing": 2000},  # 66% — well over 30% benchmark
        )
        flags = result["flags"]
        assert any("Housing" in f or "housing" in f for f in flags)

    def test_healthy_budget_produces_positive_tip(self, agent):
        result = agent.analyze(
            income=6000,
            expenses={"housing": 1500, "food": 500, "transport": 600, "savings": 1400},
        )
        suggestions = result["suggestions"]
        assert any("✅" in s for s in suggestions)

    def test_suggestions_not_empty(self, agent):
        result = agent.analyze(income=5000, expenses={"housing": 1500})
        assert len(result["suggestions"]) > 0

    def test_profile_is_budget_profile_instance(self, agent):
        result = agent.analyze(income=5000, expenses={"food": 600})
        assert isinstance(result["profile"], BudgetProfile)

    def test_chat_without_llm_returns_fallback(self, agent):
        """If LLM not available, agent should return a helpful fallback string."""
        agent._llm = None
        response = agent.chat("How do I reduce my food spending?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_list_categories_returns_all_benchmarks(self, agent):
        categories = agent.list_categories()
        assert set(categories) == set(BUDGET_BENCHMARKS.keys())

    def test_get_benchmark_known_category(self, agent):
        assert agent.get_benchmark("housing") == 30.0

    def test_get_benchmark_unknown_category(self, agent):
        assert agent.get_benchmark("unknown_category") is None

    def test_currency_flows_through(self, agent):
        result = agent.analyze(
            income=10000, expenses={"housing": 3000}, currency="ZAR"
        )
        assert result["profile"].currency == "ZAR"
