"""
Portfolio Agent
Analyses investment portfolios, calculates risk metrics, and recommends
rebalancing strategies aligned with the user's risk profile and goals.
Integrates with core LLM engine and RAG service.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Target allocation templates by risk profile
RISK_PROFILES: Dict[str, Dict[str, float]] = {
    "conservative": {
        "bonds": 60.0,
        "stocks": 25.0,
        "cash": 10.0,
        "real_estate": 5.0,
    },
    "moderate": {
        "stocks": 50.0,
        "bonds": 35.0,
        "real_estate": 10.0,
        "cash": 5.0,
    },
    "aggressive": {
        "stocks": 75.0,
        "bonds": 15.0,
        "real_estate": 7.0,
        "cash": 3.0,
    },
}

ASSET_EXPECTED_RETURNS: Dict[str, float] = {
    "stocks": 10.0,
    "bonds": 4.0,
    "real_estate": 7.0,
    "cash": 2.0,
    "crypto": 15.0,
    "commodities": 5.0,
}

ASSET_VOLATILITIES: Dict[str, float] = {
    "stocks": 18.0,
    "bonds": 6.0,
    "real_estate": 12.0,
    "cash": 0.5,
    "crypto": 60.0,
    "commodities": 20.0,
}


@dataclass
class Holding:
    """A single portfolio position."""
    asset_class: str       # e.g. "stocks", "bonds", "real_estate"
    name: str              # e.g. "S&P 500 ETF", "US Treasury 10yr"
    value: float
    currency: str = "USD"

    @property
    def expected_return(self) -> float:
        return ASSET_EXPECTED_RETURNS.get(self.asset_class.lower(), 7.0)

    @property
    def volatility(self) -> float:
        return ASSET_VOLATILITIES.get(self.asset_class.lower(), 15.0)


@dataclass
class PortfolioAnalysis:
    """Full analysis result for a portfolio."""
    holdings: List[Holding] = field(default_factory=list)
    total_value: float = 0.0
    currency: str = "USD"
    risk_profile: str = "moderate"
    # Allocation
    current_allocation: Dict[str, float] = field(default_factory=dict)   # % by asset class
    target_allocation: Dict[str, float] = field(default_factory=dict)
    allocation_drift: Dict[str, float] = field(default_factory=dict)     # current - target
    # Risk metrics
    portfolio_expected_return: float = 0.0
    portfolio_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    diversification_score: float = 0.0   # 0–100
    # Rebalancing
    rebalance_needed: bool = False
    rebalance_trades: List[Dict[str, Any]] = field(default_factory=list)

    def compute(self, risk_free_rate: float = 2.0) -> None:
        if not self.holdings:
            return

        self.total_value = sum(h.value for h in self.holdings)
        if self.total_value <= 0:
            return

        # Current allocation by asset class
        class_values: Dict[str, float] = {}
        for h in self.holdings:
            class_values[h.asset_class] = class_values.get(h.asset_class, 0) + h.value
        self.current_allocation = {
            cls: round(val / self.total_value * 100, 2)
            for cls, val in class_values.items()
        }

        # Target allocation
        self.target_allocation = RISK_PROFILES.get(self.risk_profile, RISK_PROFILES["moderate"])

        # Drift (positive = overweight, negative = underweight)
        all_classes = set(self.current_allocation) | set(self.target_allocation)
        self.allocation_drift = {
            cls: round(
                self.current_allocation.get(cls, 0) - self.target_allocation.get(cls, 0), 2
            )
            for cls in all_classes
        }

        # Portfolio expected return (weighted average)
        self.portfolio_expected_return = round(
            sum(
                (pct / 100) * ASSET_EXPECTED_RETURNS.get(cls, 7.0)
                for cls, pct in self.current_allocation.items()
            ),
            2,
        )

        # Simplified portfolio volatility (weighted average — ignores correlations)
        self.portfolio_volatility = round(
            math.sqrt(
                sum(
                    ((pct / 100) * ASSET_VOLATILITIES.get(cls, 15.0)) ** 2
                    for cls, pct in self.current_allocation.items()
                )
            ),
            2,
        )

        # Sharpe ratio
        excess_return = self.portfolio_expected_return - risk_free_rate
        self.sharpe_ratio = (
            round(excess_return / self.portfolio_volatility, 2)
            if self.portfolio_volatility > 0
            else 0.0
        )

        # Diversification score — penalise concentration
        n_classes = len(self.current_allocation)
        max_weight = max(self.current_allocation.values()) if self.current_allocation else 100
        self.diversification_score = round(
            min(100, (n_classes / 5) * 50 + (1 - max_weight / 100) * 50), 1
        )

        # Rebalancing
        DRIFT_THRESHOLD = 5.0  # percentage points
        self.rebalance_needed = any(abs(v) > DRIFT_THRESHOLD for v in self.allocation_drift.values())
        if self.rebalance_needed:
            self.rebalance_trades = self._compute_trades()

    def _compute_trades(self) -> List[Dict[str, Any]]:
        trades = []
        for cls, drift in self.allocation_drift.items():
            if abs(drift) > 5.0:
                trade_value = abs(drift / 100) * self.total_value
                trades.append({
                    "asset_class": cls,
                    "action": "SELL" if drift > 0 else "BUY",
                    "amount": round(trade_value, 2),
                    "current_pct": self.current_allocation.get(cls, 0),
                    "target_pct": self.target_allocation.get(cls, 0),
                })
        return sorted(trades, key=lambda t: -abs(t["amount"]))


class PortfolioAgent:
    """
    Agent that analyses portfolios and recommends rebalancing strategies.

    Quick start
    -----------
    >>> agent = PortfolioAgent()
    >>> holdings = [
    ...     {"asset_class": "stocks",    "name": "S&P 500 ETF",  "value": 60000},
    ...     {"asset_class": "bonds",     "name": "US Treasuries", "value": 25000},
    ...     {"asset_class": "real_estate","name": "REIT",         "value": 10000},
    ...     {"asset_class": "cash",      "name": "Money Market", "value": 5000},
    ... ]
    >>> result = agent.analyze(holdings=holdings, risk_profile="moderate")
    >>> print(result["advice"])
    """

    def __init__(self) -> None:
        try:
            from core.llm_engine import llm_engine
            from core.rag_service import rag_service
            self._llm = llm_engine
            self._rag = rag_service
            logger.info("PortfolioAgent initialised with LLM + RAG")
        except Exception as exc:
            logger.warning(f"PortfolioAgent running without LLM/RAG: {exc}")
            self._llm = None
            self._rag = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        holdings: List[Dict[str, Any]],
        risk_profile: str = "moderate",
        risk_free_rate: float = 2.0,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyse a portfolio and return structured recommendations.

        Parameters
        ----------
        holdings : list of dicts, each with:
            asset_class (str), name (str), value (float), currency (optional str)
        risk_profile : "conservative" | "moderate" | "aggressive"
        risk_free_rate : float — used in Sharpe ratio calculation
        history : prior conversation turns

        Returns a dict with:
            analysis    – PortfolioAnalysis dataclass with all metrics
            flags       – risk / concentration warnings
            advice      – LLM narrative with recommendations
            rag_docs    – retrieved knowledge snippets
        """
        if risk_profile not in RISK_PROFILES:
            raise ValueError(f"risk_profile must be one of {list(RISK_PROFILES)}")

        holding_objects = [self._dict_to_holding(h) for h in holdings]
        analysis = PortfolioAnalysis(
            holdings=holding_objects,
            risk_profile=risk_profile,
            currency=holding_objects[0].currency if holding_objects else "USD",
        )
        analysis.compute(risk_free_rate=risk_free_rate)

        flags = self._generate_flags(analysis)
        rag_docs = self._retrieve_context(
            f"portfolio rebalancing {risk_profile} investor asset allocation"
        )
        advice = self._llm_advice(analysis, flags, rag_docs, history or [])

        return {
            "analysis": analysis,
            "flags": flags,
            "advice": advice,
            "rag_docs": rag_docs,
        }

    def chat(
        self,
        message: str,
        history: Optional[List[Tuple[str, str]]] = None,
        holdings: Optional[List[Dict[str, Any]]] = None,
        risk_profile: str = "moderate",
    ) -> str:
        """Free-form portfolio chat."""
        if not self._llm:
            return self._fallback_response()

        context_block = ""
        if holdings:
            holding_objects = [self._dict_to_holding(h) for h in holdings]
            analysis = PortfolioAnalysis(holdings=holding_objects, risk_profile=risk_profile)
            analysis.compute()
            context_block = self._format_summary(analysis)

        # Compact: skip heavy RAG in chat() — llm_engine handles it internally
        system_prompt = self._build_system_prompt(context_block, "")

        try:
            return self._llm.chat_with_context(message, history or [])
        except Exception as exc:
            logger.error(f"PortfolioAgent LLM error: {exc}")
            return self._fallback_response()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _generate_flags(self, a: PortfolioAnalysis) -> List[str]:
        flags: List[str] = []

        # Concentration risk
        for cls, pct in a.current_allocation.items():
            if pct > 70:
                flags.append(f"🚨 High concentration: {cls.title()} is {pct}% of portfolio.")
            elif pct > 50:
                flags.append(f"⚠️  {cls.title()} at {pct}% — consider diversifying.")

        # Rebalancing needed
        if a.rebalance_needed:
            flags.append(
                f"⚠️  Portfolio has drifted from {a.risk_profile} target allocation. "
                "Rebalancing recommended."
            )

        # Low Sharpe ratio
        if a.sharpe_ratio < 0.5:
            flags.append(
                f"⚠️  Sharpe ratio of {a.sharpe_ratio} is low — "
                "returns may not justify the risk taken."
            )

        # High volatility asset (crypto)
        crypto_pct = a.current_allocation.get("crypto", 0)
        if crypto_pct > 10:
            flags.append(
                f"⚠️  Crypto is {crypto_pct}% of portfolio — highly volatile asset class."
            )

        if not flags:
            flags.append("✅ Portfolio looks well-balanced for your risk profile.")

        return flags

    def _format_summary(self, a: PortfolioAnalysis) -> str:
        lines = [
            f"Total value        : {a.total_value:,.2f} {a.currency}",
            f"Risk profile       : {a.risk_profile.title()}",
            f"Expected return    : {a.portfolio_expected_return:.1f}%",
            f"Portfolio volatility: {a.portfolio_volatility:.1f}%",
            f"Sharpe ratio       : {a.sharpe_ratio:.2f}",
            f"Diversification    : {a.diversification_score}/100",
            "\nCurrent vs Target allocation:",
        ]
        all_classes = set(a.current_allocation) | set(a.target_allocation)
        for cls in sorted(all_classes):
            current = a.current_allocation.get(cls, 0)
            target = a.target_allocation.get(cls, 0)
            drift = a.allocation_drift.get(cls, 0)
            indicator = "▲" if drift > 0 else ("▼" if drift < 0 else "=")
            lines.append(
                f"  {cls.title():<18} Current: {current:5.1f}%  "
                f"Target: {target:5.1f}%  {indicator}{abs(drift):.1f}%"
            )
        if a.rebalance_trades:
            lines.append("\nRecommended trades:")
            for t in a.rebalance_trades:
                lines.append(
                    f"  {t['action']} {t['asset_class'].title()}: "
                    f"{t['amount']:,.2f} {a.currency}"
                )
        return "\n".join(lines)

    def _build_system_prompt(self, context_block: str, rag_block: str) -> str:
        # Compact prompt — keeps token count low on CPU
        parts = ["You are FinanceBot. Give concise portfolio and investment advice."]
        if context_block:
            parts += ["\nProfile:", context_block[:300]]
        return " ".join(parts)

    def _llm_advice(
        self,
        analysis: PortfolioAnalysis,
        flags: List[str],
        rag_docs: List[str],
        history: List[Tuple[str, str]],
    ) -> str:
        if not self._llm:
            return self._format_summary(analysis)

        rag_block = "\n\n".join(rag_docs) if rag_docs else ""
        flag_text = "\n".join(flags)
        user_prompt = (
            f"Portfolio overview:\n{self._format_summary(analysis)}\n\n"
            f"Issues identified:\n{flag_text}\n\n"
            "Provide a comprehensive portfolio review: explain the metrics, "
            "justify any rebalancing trades, and suggest next steps."
        )
        try:
            return self._llm.chat(
                message=user_prompt,
                history=history,
                system_prompt=self._build_system_prompt("", rag_block),
            )
        except Exception as exc:
            logger.error(f"LLM advice failed: {exc}")
            return self._format_summary(analysis)

    def _retrieve_context(self, query: str) -> List[str]:
        if not self._rag:
            return []
        try:
            return self._rag.retrieve_docs(query, k=3)
        except Exception as exc:
            logger.warning(f"RAG retrieval failed: {exc}")
            return []

    @staticmethod
    def _dict_to_holding(d: Dict[str, Any]) -> Holding:
        return Holding(
            asset_class=d["asset_class"],
            name=d.get("name", d["asset_class"]),
            value=float(d["value"]),
            currency=d.get("currency", "USD"),
        )

    @staticmethod
    def _fallback_response() -> str:
        return (
            "I'm your Portfolio Manager. Share your holdings (asset class, name, value) "
            "and your risk profile (conservative / moderate / aggressive) and I'll "
            "analyse your portfolio and recommend rebalancing trades."
        )

