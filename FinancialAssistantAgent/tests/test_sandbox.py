"""
Tests for SandboxExecutorAgent
Run with: pytest tests/test_sandbox.py -v
"""

import pytest
from agents.sandbox_executor import SandboxExecutorAgent


@pytest.fixture
def agent():
    return SandboxExecutorAgent()


# ── Expression evaluation ──────────────────────────────────────────────────────

class TestEvaluate:
    def test_simple_arithmetic(self, agent):
        result = agent.evaluate("1000 * 2")
        assert result["result"] == 2000

    def test_compound_interest_formula(self, agent):
        # 1000 * (1.07 ^ 10) ≈ 1967.15
        result = agent.evaluate("1000 * (1 + 0.07) ** 10")
        assert result["result"] == pytest.approx(1967.15, rel=0.01)

    def test_math_function_allowed(self, agent):
        result = agent.evaluate("sqrt(144)")
        assert result["result"] == pytest.approx(12.0)

    def test_forbidden_import_blocked(self, agent):
        result = agent.evaluate("__import__('os')")
        assert result["result"] is None
        assert "error" in result

    def test_forbidden_exec_blocked(self, agent):
        result = agent.evaluate("exec('print(1)')")
        assert result["result"] is None

    def test_attribute_access_blocked(self, agent):
        result = agent.evaluate("(1).__class__")
        assert result["result"] is None

    def test_empty_expression(self, agent):
        result = agent.evaluate("")
        # Should not crash
        assert "result" in result or "error" in result

    def test_division(self, agent):
        result = agent.evaluate("100 / 4")
        assert result["result"] == 25.0


# ── Scenario: compound_growth ──────────────────────────────────────────────────

class TestCompoundGrowth:
    def test_basic_growth(self, agent):
        result = agent.run_scenario("compound_growth", principal=1000, annual_rate=10, years=1)
        assert result["results"]["future_value"] == pytest.approx(1104.71, rel=0.01)

    def test_zero_rate_no_growth(self, agent):
        result = agent.run_scenario("compound_growth", principal=1000, annual_rate=0, years=10)
        assert result["results"]["future_value"] == pytest.approx(1000, rel=0.01)

    def test_with_monthly_contribution(self, agent):
        result = agent.run_scenario(
            "compound_growth", principal=0, annual_rate=7, years=10, monthly_contribution=100
        )
        assert result["results"]["future_value"] > 100 * 12 * 10  # more than simple sum

    def test_returns_expected_keys(self, agent):
        result = agent.run_scenario("compound_growth", principal=5000, annual_rate=7, years=20)
        assert "future_value" in result["results"]
        assert "total_interest_earned" in result["results"]


# ── Scenario: loan_payoff ──────────────────────────────────────────────────────

class TestLoanPayoff:
    def test_basic_payoff(self, agent):
        result = agent.run_scenario("loan_payoff", principal=10000, annual_rate=5, monthly_payment=200)
        assert result["results"]["months_to_payoff"] > 0

    def test_payment_too_low_returns_error(self, agent):
        # Very small payment on a large balance at high rate
        result = agent.run_scenario("loan_payoff", principal=100000, annual_rate=24, monthly_payment=10)
        assert "error" in result or "error" in result.get("results", {})

    def test_zero_rate_loan(self, agent):
        result = agent.run_scenario("loan_payoff", principal=1200, annual_rate=0, monthly_payment=100)
        assert result["results"]["months_to_payoff"] == 12


# ── Scenario: retirement_savings ──────────────────────────────────────────────

class TestRetirementSavings:
    def test_returns_retirement_fund(self, agent):
        result = agent.run_scenario(
            "retirement_savings", monthly_contribution=500, annual_rate=7, years=30
        )
        assert result["results"]["retirement_fund"] > 500 * 12 * 30

    def test_4pct_rule_calculated(self, agent):
        result = agent.run_scenario(
            "retirement_savings", monthly_contribution=500, annual_rate=7, years=30
        )
        fund = result["results"]["retirement_fund"]
        expected_monthly = round(fund * 0.04 / 12, 2)
        assert result["results"]["monthly_income_4pct_rule"] == pytest.approx(expected_monthly, rel=0.01)


# ── Scenario: emergency_fund ──────────────────────────────────────────────────

class TestEmergencyFund:
    def test_already_funded(self, agent):
        result = agent.run_scenario(
            "emergency_fund",
            monthly_expenses=2000,
            months_target=3,
            current_savings=7000,
            monthly_savings=300,
        )
        assert result["results"]["already_funded"] is True

    def test_months_to_goal(self, agent):
        result = agent.run_scenario(
            "emergency_fund",
            monthly_expenses=2000,
            months_target=6,
            current_savings=0,
            monthly_savings=500,
        )
        # target = 12000, monthly = 500 → 24 months
        assert result["results"]["months_to_goal"] == 24


# ── What-if analysis ───────────────────────────────────────────────────────────

class TestWhatIf:
    def test_returns_table(self, agent):
        result = agent.what_if(
            scenario="compound_growth",
            base_params={"principal": 10000, "annual_rate": 7, "years": 10},
            variations={"annual_rate": [5, 7, 9, 12]},
        )
        assert "table" in result
        assert len(result["table"]) == 4

    def test_unknown_scenario_returns_error(self, agent):
        result = agent.what_if(
            scenario="nonexistent",
            base_params={},
            variations={"x": [1, 2]},
        )
        assert "error" in result


# ── list_scenarios ─────────────────────────────────────────────────────────────

def test_list_scenarios(agent):
    scenarios = agent.list_scenarios()
    assert "compound_growth" in scenarios
    assert "loan_payoff" in scenarios
    assert "retirement_savings" in scenarios
    assert "emergency_fund" in scenarios
    assert "inflation_impact" in scenarios
