"""
tests/test_portfolio_agent.py
Unit tests for PortfolioAgent — no LLM or RAG required.
"""

import pytest
from agents.portfolio_agent import (
    PortfolioAgent,
    Holding,
    PortfolioAnalysis,
    RISK_PROFILES,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    return PortfolioAgent()


@pytest.fixture
def balanced_holdings():
    return [
        {"asset_class": "stocks",      "name": "S&P 500 ETF",   "value": 60000},
        {"asset_class": "bonds",       "name": "US Treasuries",  "value": 25000},
        {"asset_class": "real_estate", "name": "REIT",           "value": 10000},
        {"asset_class": "cash",        "name": "Money Market",   "value": 5000},
    ]


@pytest.fixture
def concentrated_holdings():
    """Single asset class — should trigger concentration flag."""
    return [
        {"asset_class": "stocks", "name": "Tech ETF",    "value": 95000},
        {"asset_class": "cash",   "name": "Savings",     "value": 5000},
    ]


@pytest.fixture
def crypto_heavy_holdings():
    """High crypto allocation — should trigger crypto warning."""
    return [
        {"asset_class": "stocks", "name": "S&P 500",    "value": 50000},
        {"asset_class": "crypto", "name": "Bitcoin",    "value": 30000},
        {"asset_class": "bonds",  "name": "Treasuries", "value": 20000},
    ]


# ── Holding dataclass ──────────────────────────────────────────────────────────

class TestHolding:
    def test_holding_stores_values(self):
        h = Holding(asset_class="stocks", name="S&P 500", value=10000)
        assert h.asset_class == "stocks"
        assert h.name == "S&P 500"
        assert h.value == 10000

    def test_holding_expected_return_by_class(self):
        stocks = Holding(asset_class="stocks", name="X", value=1000)
        bonds  = Holding(asset_class="bonds",  name="Y", value=1000)
        cash   = Holding(asset_class="cash",   name="Z", value=1000)
        # Stocks should have higher expected return than bonds and cash
        assert stocks.expected_return > bonds.expected_return
        assert bonds.expected_return  > cash.expected_return

    def test_holding_volatility_by_class(self):
        stocks = Holding(asset_class="stocks", name="X", value=1000)
        bonds  = Holding(asset_class="bonds",  name="Y", value=1000)
        # Stocks should be more volatile than bonds
        assert stocks.volatility > bonds.volatility

    def test_holding_unknown_asset_class_uses_defaults(self):
        h = Holding(asset_class="unknown_class", name="X", value=1000)
        assert h.expected_return >= 0
        assert h.volatility >= 0


# ── RISK_PROFILES ──────────────────────────────────────────────────────────────

class TestRiskProfiles:
    def test_all_profiles_present(self):
        assert "conservative" in RISK_PROFILES
        assert "moderate"     in RISK_PROFILES
        assert "aggressive"   in RISK_PROFILES

    def test_profiles_sum_to_100(self):
        for name, profile in RISK_PROFILES.items():
            total = sum(profile.values())
            assert abs(total - 100) < 0.01, f"{name} profile allocations sum to {total}, not 100"

    def test_aggressive_has_more_stocks_than_conservative(self):
        assert (
            RISK_PROFILES["aggressive"].get("stocks", 0)
            > RISK_PROFILES["conservative"].get("stocks", 0)
        )

    def test_conservative_has_more_bonds_than_aggressive(self):
        assert (
            RISK_PROFILES["conservative"].get("bonds", 0)
            > RISK_PROFILES["aggressive"].get("bonds", 0)
        )


# ── PortfolioAgent.analyze() ───────────────────────────────────────────────────

class TestPortfolioAnalyze:
    def test_returns_expected_keys(self, agent, balanced_holdings):
        result = agent.analyze(balanced_holdings, risk_profile="moderate")
        assert "analysis"  in result
        assert "flags"     in result
        assert "advice"    in result
        assert "rag_docs"  in result

    def test_analysis_is_portfolio_analysis_instance(self, agent, balanced_holdings):
        result = agent.analyze(balanced_holdings, risk_profile="moderate")
        assert isinstance(result["analysis"], PortfolioAnalysis)

    def test_total_value_calculated_correctly(self, agent, balanced_holdings):
        result = agent.analyze(balanced_holdings, risk_profile="moderate")
        expected_total = sum(h["value"] for h in balanced_holdings)
        assert result["analysis"].total_value == expected_total

    def test_current_allocation_sums_to_100(self, agent, balanced_holdings):
        result = agent.analyze(balanced_holdings, risk_profile="moderate")
        total_pct = sum(result["analysis"].current_allocation.values())
        assert abs(total_pct - 100) < 0.1

    def test_sharpe_ratio_is_numeric(self, agent, balanced_holdings):
        result = agent.analyze(balanced_holdings, risk_profile="moderate")
        assert isinstance(result["analysis"].sharpe_ratio, float)

    def test_diversification_score_in_range(self, agent, balanced_holdings):
        result = agent.analyze(balanced_holdings, risk_profile="moderate")
        score = result["analysis"].diversification_score
        assert 0 <= score <= 100

    def test_concentration_flag_triggered(self, agent, concentrated_holdings):
        result = agent.analyze(concentrated_holdings, risk_profile="aggressive")
        flags = " ".join(result["flags"]).lower()
        assert "concentrat" in flags

    def test_crypto_flag_triggered(self, agent, crypto_heavy_holdings):
        result = agent.analyze(crypto_heavy_holdings, risk_profile="aggressive")
        flags = " ".join(result["flags"]).lower()
        assert "crypto" in flags

    def test_balanced_portfolio_has_fewer_flags(self, agent, balanced_holdings, concentrated_holdings):
        balanced_result = agent.analyze(balanced_holdings, risk_profile="moderate")
        concentrated_result = agent.analyze(concentrated_holdings, risk_profile="moderate")
        assert len(balanced_result["flags"]) <= len(concentrated_result["flags"])

    def test_invalid_risk_profile_raises(self, agent, balanced_holdings):
        with pytest.raises(ValueError):
            agent.analyze(balanced_holdings, risk_profile="ultra_risky")

    def test_empty_holdings_raises(self, agent):
        with pytest.raises((ValueError, ZeroDivisionError)):
            agent.analyze([], risk_profile="moderate")

    def test_single_holding(self, agent):
        holdings = [{"asset_class": "stocks", "name": "Single ETF", "value": 100000}]
        result = agent.analyze(holdings, risk_profile="aggressive")
        assert result["analysis"].total_value == 100000

    def test_risk_profile_stored_on_analysis(self, agent, balanced_holdings):
        for profile in ("conservative", "moderate", "aggressive"):
            result = agent.analyze(balanced_holdings, risk_profile=profile)
            assert result["analysis"].risk_profile == profile


# ── Rebalancing ────────────────────────────────────────────────────────────────

class TestRebalancing:
    def test_rebalance_needed_flag_type(self, agent, balanced_holdings):
        result = agent.analyze(balanced_holdings, risk_profile="moderate")
        assert isinstance(result["analysis"].rebalance_needed, bool)

    def test_heavily_drifted_portfolio_needs_rebalance(self, agent, concentrated_holdings):
        # 95% stocks vs a moderate target of ~50% — should require rebalancing
        result = agent.analyze(concentrated_holdings, risk_profile="moderate")
        assert result["analysis"].rebalance_needed is True

    def test_rebalance_trades_have_required_keys(self, agent, concentrated_holdings):
        result = agent.analyze(concentrated_holdings, risk_profile="moderate")
        analysis = result["analysis"]
        if analysis.rebalance_needed:
            for trade in analysis.rebalance_trades:
                assert "asset_class" in trade
                assert "action"      in trade   # BUY or SELL
                assert "amount"      in trade

    def test_rebalance_trade_actions_are_valid(self, agent, concentrated_holdings):
        result = agent.analyze(concentrated_holdings, risk_profile="moderate")
        for trade in result["analysis"].rebalance_trades:
            assert trade["action"] in ("BUY", "SELL")


# ── Risk metrics ───────────────────────────────────────────────────────────────

class TestRiskMetrics:
    def test_higher_stock_allocation_gives_higher_expected_return(self, agent):
        conservative = [
            {"asset_class": "bonds", "name": "Bonds", "value": 80000},
            {"asset_class": "cash",  "name": "Cash",  "value": 20000},
        ]
        aggressive = [
            {"asset_class": "stocks", "name": "Stocks", "value": 90000},
            {"asset_class": "cash",   "name": "Cash",   "value": 10000},
        ]
        r_con = agent.analyze(conservative, risk_profile="conservative")
        r_agg = agent.analyze(aggressive,   risk_profile="aggressive")
        assert (
            r_agg["analysis"].expected_return
            > r_con["analysis"].expected_return
        )

    def test_higher_stock_allocation_gives_higher_volatility(self, agent):
        conservative = [
            {"asset_class": "bonds", "name": "Bonds", "value": 80000},
            {"asset_class": "cash",  "name": "Cash",  "value": 20000},
        ]
        aggressive = [
            {"asset_class": "stocks", "name": "Stocks", "value": 90000},
            {"asset_class": "cash",   "name": "Cash",   "value": 10000},
        ]
        r_con = agent.analyze(conservative, risk_profile="conservative")
        r_agg = agent.analyze(aggressive,   risk_profile="aggressive")
        assert (
            r_agg["analysis"].volatility
            > r_con["analysis"].volatility
        )

    def test_low_sharpe_flag_triggered(self, agent):
        # All cash — very low return relative to any risk — Sharpe will be near zero
        cash_only = [{"asset_class": "cash", "name": "Cash", "value": 100000}]
        result = agent.analyze(cash_only, risk_profile="aggressive")
        flags = " ".join(result["flags"]).lower()
        # Should flag that portfolio is underperforming for the risk profile
        assert len(result["flags"]) > 0


# ── chat() ─────────────────────────────────────────────────────────────────────

class TestPortfolioChat:
    def test_chat_returns_string(self, agent):
        response = agent.chat("What is diversification?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_chat_with_history(self, agent):
        history = [("What is a bond?", "A bond is a fixed-income instrument.")]
        response = agent.chat("How do bonds compare to stocks?", history=history)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_chat_fallback_when_no_llm(self, agent):
        agent._llm = None
        response = agent.chat("Tell me about ETFs")
        assert isinstance(response, str)
        assert len(response) > 0
