"""
Core Orchestrator
Single FastAPI backend — replaces app/orchestrator.py entirely.

Agent routing
-------------
- POST /query          — auto-detects agent from query text; override via `agent` field
- POST /agent/budget   — BudgetAdvisorAgent  (requires budget_data in request)
- POST /agent/debt     — DebtCoachAgent      (requires debt_data in request)
- POST /agent/portfolio— PortfolioAgent      (requires portfolio_data in request)
- POST /agent/sandbox  — SandboxExecutorAgent(requires sandbox_data in request)
- GET  /agents         — list available agents + keyword hints
"""

import os
import sys
import logging
import time
import uuid
import re
from typing import Any, Dict, List, Optional
from datetime import datetime

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from config import config
from core.llm_engine import llm_engine
from core.rag_service import rag_service
from core.prompt_service import prompt_service
from core.database_service import database_service
from core.memory_service import memory_service

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs(config.LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, "orchestrator.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Agent keyword map (used for auto-detection) ────────────────────────────────
AGENT_KEYWORDS: Dict[str, List[str]] = {
    "budget": [
        "budget", "spending", "expenses", "overspend", "50/30/20",
        "saving money", "cut costs", "monthly expenses", "income",
        "where does my money go", "afford",
    ],
    "debt": [
        "debt", "loan", "credit card", "pay off", "payoff", "interest rate",
        "avalanche", "snowball", "minimum payment", "owe", "balance",
        "mortgage", "student loan", "car loan",
    ],
    "portfolio": [
        "portfolio", "invest", "stocks", "bonds", "etf", "asset allocation",
        "rebalance", "diversif", "sharpe", "risk profile", "holdings",
        "mutual fund", "equities", "returns",
    ],
    "sandbox": [
        "calculate", "formula", "compound interest", "what if", "scenario",
        "how much will", "future value", "npv", "roi", "retirement fund",
        "simulate", "projection", "how long to",
    ],
}


def detect_agent(query: str) -> Optional[str]:
    """
    Score each agent by keyword hits and return the best match.
    Returns None if no agent scores above zero.
    """
    q = query.lower()
    scores: Dict[str, int] = {agent: 0 for agent in AGENT_KEYWORDS}
    for agent, keywords in AGENT_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[agent] += 1
    best_agent = max(scores, key=lambda a: scores[a])
    return best_agent if scores[best_agent] > 0 else None


# ── Input validation helpers ───────────────────────────────────────────────────

_VALID_AGENTS = {"budget", "debt", "portfolio", "sandbox", "llm"}
_VALID_STRATEGIES = {"avalanche", "snowball"}
_VALID_RISK_PROFILES = {"conservative", "moderate", "aggressive"}
_VALID_SANDBOX_TYPES = {"evaluate", "scenario", "what_if"}


# ── Pydantic request / response models ────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User's financial question")
    query_type: str = Field("General", max_length=100, description="Type of financial query")
    use_rag: bool = Field(True, description="Whether to use RAG for context")
    session_id: Optional[str] = Field(None, max_length=100, description="Session identifier")
    agent: Optional[str] = Field(
        None,
        description="Force a specific agent: budget|debt|portfolio|sandbox|llm. Leave empty for auto-detection.",
    )

    def model_post_init(self, __context: Any) -> None:
        if self.agent is not None and self.agent not in _VALID_AGENTS:
            raise ValueError(f"agent must be one of {sorted(_VALID_AGENTS)} or null.")


class QueryResponse(BaseModel):
    response: str
    reasoning: str = ""
    risk_evaluation: str = ""
    retrieved_docs: List[str] = []
    response_time: float
    session_id: str
    conversation_id: Optional[int] = None
    agent_used: str = "llm"
    auto_detected: bool = False


class FeedbackRequest(BaseModel):
    conversation_id: int
    rating: int = Field(..., ge=1, le=5)
    feedback_text: Optional[str] = None
    user_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, Any]
    uptime: float


class StatsResponse(BaseModel):
    conversations: Dict[str, Any]
    feedback: Dict[str, Any]
    performance: Dict[str, Any]
    system: Dict[str, Any]


# ── Agent-specific request bodies ─────────────────────────────────────────────

class BudgetRequest(BaseModel):
    query: str = Field("Analyse my budget", description="Optional context question")
    session_id: Optional[str] = None
    budget_data: Dict[str, Any] = Field(
        ...,
        description=(
            "Keys: income (float), expenses (dict[str, float]), "
            "savings_goal_pct (float, default 20), currency (str, default USD)"
        ),
        example={
            "income": 5000,
            "expenses": {"housing": 1500, "food": 600, "transport": 300},
            "savings_goal_pct": 20,
            "currency": "USD",
        },
    )


class DebtRequest(BaseModel):
    query: str = Field("Help me pay off my debts", description="Optional context question")
    session_id: Optional[str] = None
    debt_data: Dict[str, Any] = Field(
        ...,
        description=(
            "Keys: debts (list of {name, balance, interest_rate, minimum_payment}), "
            "monthly_payment (float), strategy ('avalanche' | 'snowball', default avalanche)"
        ),
        example={
            "debts": [
                {"name": "Credit Card", "balance": 3000, "interest_rate": 22.0, "minimum_payment": 60},
                {"name": "Car Loan",    "balance": 8000, "interest_rate":  6.5, "minimum_payment": 200},
            ],
            "monthly_payment": 600,
            "strategy": "avalanche",
        },
    )


class PortfolioRequest(BaseModel):
    query: str = Field("Analyse my portfolio", description="Optional context question")
    session_id: Optional[str] = None
    portfolio_data: Dict[str, Any] = Field(
        ...,
        description=(
            "Keys: holdings (list of {asset_class, name, value}), "
            "risk_profile ('conservative'|'moderate'|'aggressive'), "
            "risk_free_rate (float, default 2.0)"
        ),
        example={
            "holdings": [
                {"asset_class": "stocks",     "name": "S&P 500 ETF",  "value": 60000},
                {"asset_class": "bonds",      "name": "US Treasuries", "value": 25000},
                {"asset_class": "real_estate","name": "REIT",          "value": 10000},
                {"asset_class": "cash",       "name": "Money Market",  "value": 5000},
            ],
            "risk_profile": "moderate",
            "risk_free_rate": 2.0,
        },
    )


class SandboxRequest(BaseModel):
    query: str = Field("Run a financial scenario", description="Optional context question")
    session_id: Optional[str] = None
    sandbox_data: Dict[str, Any] = Field(
        ...,
        description=(
            "For expression eval: {type: 'evaluate', expression: str}. "
            "For scenarios:       {type: 'scenario',  name: str, params: dict}. "
            "For what-if:         {type: 'what_if',   scenario: str, "
            "                      base_params: dict, variations: dict}."
        ),
        example={
            "type": "scenario",
            "name": "compound_growth",
            "params": {"principal": 10000, "annual_rate": 7, "years": 20},
        },
    )


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Financial Assistant AI",
    description=(
        "AI-powered financial advisory system with specialist agents and RAG. "
        "Agents: budget, debt, portfolio, sandbox."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ───────────────────────────────────────────────────────────────
_start_time = time.time()
_request_count = 0
_error_count = 0


# ── Orchestrator ───────────────────────────────────────────────────────────────

class Orchestrator:
    """Manages all services and routes queries to the right agent."""

    def __init__(self):
        self.llm = llm_engine
        self.rag = rag_service
        self.prompts = prompt_service
        self.db = database_service
        self.memory = memory_service
        self._budget_agent = None
        self._debt_agent = None
        self._portfolio_agent = None
        self._sandbox_agent = None
        logger.info("Orchestrator initialised")

    # ── Lazy agent accessors ───────────────────────────────────────────────────

    @property
    def budget_agent(self):
        if self._budget_agent is None:
            from agents.budget_advisor import BudgetAdvisorAgent
            self._budget_agent = BudgetAdvisorAgent()
            logger.info("BudgetAdvisorAgent loaded")
        return self._budget_agent

    @property
    def debt_agent(self):
        if self._debt_agent is None:
            from agents.debt_coach import DebtCoachAgent
            self._debt_agent = DebtCoachAgent()
            logger.info("DebtCoachAgent loaded")
        return self._debt_agent

    @property
    def portfolio_agent(self):
        if self._portfolio_agent is None:
            from agents.portfolio_agent import PortfolioAgent
            self._portfolio_agent = PortfolioAgent()
            logger.info("PortfolioAgent loaded")
        return self._portfolio_agent

    @property
    def sandbox_agent(self):
        if self._sandbox_agent is None:
            from agents.sandbox_executor import SandboxExecutorAgent
            self._sandbox_agent = SandboxExecutorAgent()
            logger.info("SandboxExecutorAgent loaded")
        return self._sandbox_agent

    # ── General query ──────────────────────────────────────────────────────────

    async def process_query(
        self,
        query: str,
        query_type: str = "General",
        use_rag: bool = True,
        session_id: Optional[str] = None,
        agent_override: Optional[str] = None,
    ) -> QueryResponse:
        global _request_count, _error_count
        _request_count += 1
        t0 = time.time()
        session_id = session_id or str(uuid.uuid4())

        # Determine which agent to use
        auto_detected = False
        if agent_override and agent_override != "llm":
            agent_name = agent_override
        else:
            detected = detect_agent(query)
            if detected and agent_override != "llm":
                agent_name = detected
                auto_detected = True
            else:
                agent_name = "llm"

        logger.info(
            f"Query routed to [{agent_name}]"
            f"{'(auto)' if auto_detected else '(manual)'}: {query[:80]}"
        )

        # Load history for this session
        history = self.memory.get_history(session_id)

        try:
            response, reasoning, risk_eval, retrieved_docs = await self._dispatch(
                agent_name, query, use_rag, history
            )
        except Exception as exc:
            _error_count += 1
            logger.error(f"Dispatch error: {exc}")
            response = f"I encountered an error processing your request: {exc}"
            reasoning = "Error during processing"
            risk_eval = "Unable to assess risk"
            retrieved_docs = []

        # Persist this turn to memory
        self.memory.add_turn(session_id, query, response)

        response_time = time.time() - t0

        conversation_id = self.db.log_conversation(
            session_id=session_id,
            user_input=query,
            assistant_response=response,
            query_type=query_type,
            use_rag=use_rag,
            retrieved_docs=retrieved_docs,
            reasoning=reasoning,
            risk_evaluation=risk_eval,
            response_time=response_time,
        )

        self.db.log_analytics("response_time", response_time, {"agent": agent_name})

        return QueryResponse(
            response=response,
            reasoning=reasoning,
            risk_evaluation=risk_eval,
            retrieved_docs=retrieved_docs,
            response_time=round(response_time, 2),
            session_id=session_id,
            conversation_id=conversation_id,
            agent_used=agent_name,
            auto_detected=auto_detected,
        )

    async def _dispatch(
        self, agent_name: str, query: str, use_rag: bool,
        history: list = None,
    ):
        """Route to the correct agent and return (response, reasoning, risk, docs)."""
        history = history or []

        import asyncio as _asyncio

        if agent_name == "budget":
            response = await _asyncio.to_thread(self.budget_agent.chat, query, history)
            reasoning = "Routed to Budget Advisor agent."
            risk_eval = "Budgeting advice — low financial risk. Review suggestions against your circumstances."
            docs = []

        elif agent_name == "debt":
            response = await _asyncio.to_thread(self.debt_agent.chat, query, history)
            reasoning = "Routed to Debt Coach agent."
            risk_eval = "Debt repayment strategies carry no direct financial risk but depend on your income stability."
            docs = []

        elif agent_name == "portfolio":
            response = await _asyncio.to_thread(self.portfolio_agent.chat, query, history)
            reasoning = "Routed to Portfolio Manager agent."
            risk_eval = (
                "Investment advice involves market risk. Past performance does not "
                "guarantee future results. Consult a qualified advisor."
            )
            docs = []

        elif agent_name == "sandbox":
            response = await _asyncio.to_thread(self.sandbox_agent.chat, query, history)
            reasoning = "Routed to Financial Calculator (Sandbox) agent."
            risk_eval = "Projections are estimates based on assumed rates. Actual results will vary."
            docs = []

        else:  # plain LLM
            if use_rag:
                response = await _asyncio.to_thread(self.llm.chat_with_context, query, history)
            else:
                response = await _asyncio.to_thread(self.llm.chat, query, history)
            docs = []
            reasoning = "General financial knowledge via LLM."
            risk_eval = self._generic_risk_eval(query)

        return response, reasoning, risk_eval, docs

    # ── Agent-specific structured endpoints ───────────────────────────────────

    def run_budget_agent(self, data: Dict[str, Any], query: str, session_id: str) -> QueryResponse:
        t0 = time.time()
        try:
            result = self.budget_agent.analyze(
                income=data["income"],
                expenses=data["expenses"],
                savings_goal_pct=data.get("savings_goal_pct", 20.0),
                currency=data.get("currency", "USD"),
            )
            response = result["advice"] or "\n".join(result["suggestions"])
            flags_text = "\n".join(result["flags"]) if result["flags"] else "No issues detected."
            reasoning = f"Budget analysed. Issues:\n{flags_text}"
            risk_eval = "Budgeting advice — review suggestions against your own circumstances."
            docs = result["rag_docs"]
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=f"Missing budget field: {exc}")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        conversation_id = self.db.log_conversation(
            session_id=session_id, user_input=query,
            assistant_response=response, query_type="Budget",
            reasoning=reasoning, risk_evaluation=risk_eval,
            response_time=round(time.time() - t0, 2),
        )
        return QueryResponse(
            response=response, reasoning=reasoning, risk_evaluation=risk_eval,
            retrieved_docs=docs, response_time=round(time.time() - t0, 2),
            session_id=session_id, conversation_id=conversation_id,
            agent_used="budget",
        )

    def run_debt_agent(self, data: Dict[str, Any], query: str, session_id: str) -> QueryResponse:
        t0 = time.time()
        try:
            result = self.debt_agent.analyze(
                debts=data["debts"],
                monthly_payment=data["monthly_payment"],
                strategy=data.get("strategy", "avalanche"),
            )
            plan = result["plan"]
            comparison = result["comparison"]
            response = result["advice"] or self.debt_agent._rule_based_summary(plan, comparison)
            reasoning = (
                f"Strategy: {plan.strategy} | "
                f"Months to payoff: {plan.months_to_payoff} | "
                f"Total interest: {plan.total_interest_paid:,.2f}"
            )
            saved = comparison.get("interest_saved_by_avalanche", 0)
            risk_eval = (
                f"Avalanche strategy saves {abs(saved):,.2f} vs Snowball. "
                "Ensure minimum payments are always met."
            )
            docs = result["rag_docs"]
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=f"Missing debt field: {exc}")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        conversation_id = self.db.log_conversation(
            session_id=session_id, user_input=query,
            assistant_response=response, query_type="Debt",
            reasoning=reasoning, risk_evaluation=risk_eval,
            response_time=round(time.time() - t0, 2),
        )
        return QueryResponse(
            response=response, reasoning=reasoning, risk_evaluation=risk_eval,
            retrieved_docs=docs, response_time=round(time.time() - t0, 2),
            session_id=session_id, conversation_id=conversation_id,
            agent_used="debt",
        )

    def run_portfolio_agent(self, data: Dict[str, Any], query: str, session_id: str) -> QueryResponse:
        t0 = time.time()
        try:
            result = self.portfolio_agent.analyze(
                holdings=data["holdings"],
                risk_profile=data.get("risk_profile", "moderate"),
                risk_free_rate=data.get("risk_free_rate", 2.0),
            )
            analysis = result["analysis"]
            response = result["advice"] or self.portfolio_agent._format_summary(analysis)
            reasoning = (
                f"Risk profile: {analysis.risk_profile} | "
                f"Sharpe: {analysis.sharpe_ratio} | "
                f"Rebalance needed: {analysis.rebalance_needed}"
            )
            risk_eval = "\n".join(result["flags"])
            docs = result["rag_docs"]
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=f"Missing portfolio field: {exc}")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        conversation_id = self.db.log_conversation(
            session_id=session_id, user_input=query,
            assistant_response=response, query_type="Portfolio",
            reasoning=reasoning, risk_evaluation=risk_eval,
            response_time=round(time.time() - t0, 2),
        )
        return QueryResponse(
            response=response, reasoning=reasoning, risk_evaluation=risk_eval,
            retrieved_docs=docs, response_time=round(time.time() - t0, 2),
            session_id=session_id, conversation_id=conversation_id,
            agent_used="portfolio",
        )

    def run_sandbox_agent(self, data: Dict[str, Any], query: str, session_id: str) -> QueryResponse:
        t0 = time.time()
        try:
            sandbox_type = data.get("type", "scenario")

            if sandbox_type == "evaluate":
                result = self.sandbox_agent.evaluate(data["expression"])
                response = (
                    f"Result: {result.get('result')}\n\n"
                    f"{result.get('explanation', '')}"
                )
                reasoning = f"Expression evaluated: {data['expression']}"
                risk_eval = "Mathematical calculation — verify inputs are correct."

            elif sandbox_type == "what_if":
                result = self.sandbox_agent.what_if(
                    scenario=data["scenario"],
                    base_params=data["base_params"],
                    variations=data["variations"],
                )
                response = result.get("advice", "What-if analysis complete.")
                reasoning = f"What-if on {data['scenario']} with {len(result.get('table', []))} scenarios"
                risk_eval = "Projections are estimates. Actual results depend on many variables."

            else:  # scenario
                result = self.sandbox_agent.run_scenario(
                    data["name"], **data.get("params", {})
                )
                if "error" in result:
                    raise HTTPException(status_code=422, detail=result["error"])
                response = result.get("advice") or result.get("summary", "Scenario complete.")
                reasoning = f"Scenario: {data['name']} | Params: {data.get('params', {})}"
                risk_eval = "Projections assume constant rates. Real-world returns vary."

            docs = []

        except HTTPException:
            raise
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=f"Missing sandbox field: {exc}")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        conversation_id = self.db.log_conversation(
            session_id=session_id, user_input=query,
            assistant_response=response, query_type="Sandbox",
            reasoning=reasoning, risk_evaluation=risk_eval,
            response_time=round(time.time() - t0, 2),
        )
        return QueryResponse(
            response=response, reasoning=reasoning, risk_evaluation=risk_eval,
            retrieved_docs=docs, response_time=round(time.time() - t0, 2),
            session_id=session_id, conversation_id=conversation_id,
            agent_used="sandbox",
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _generic_risk_eval(query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["invest", "stock", "crypto", "leverage"]):
            return "🔴 High risk: involves investment decisions. Consult a qualified advisor."
        if any(w in q for w in ["bond", "diversif", "etf", "fund"]):
            return "🟡 Moderate risk: diversified investments carry lower but real risk."
        return "🟢 General financial information — verify with a professional for your situation."


# ── Global orchestrator instance ───────────────────────────────────────────────
orchestrator = Orchestrator()


# ── API routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Financial Assistant AI API v2",
        "status": "running",
        "docs": "/docs",
        "agents": "/agents",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    components = {
        "llm_engine":    {"status": "ready" if orchestrator.llm.is_ready() else "not_ready",
                          "details": orchestrator.llm.get_status()},
        "rag_service":   {"status": "ready" if orchestrator.rag.is_ready() else "not_ready",
                          "details": orchestrator.rag.get_document_info()},
        "prompt_service":{"status": "ready",
                          "details": orchestrator.prompts.get_prompt_info()},
        "database":      {"status": "ready",
                          "details": orchestrator.db.get_database_info()},
        "memory":        {"status": "ready",
                          "details": orchestrator.memory.get_stats()},
    }
    all_ready = all(c["status"] == "ready" for c in components.values())
    return HealthResponse(
        status="healthy" if all_ready else "degraded",
        timestamp=datetime.now().isoformat(),
        components=components,
        uptime=round(time.time() - _start_time, 1),
    )


@app.get("/agents")
async def list_agents():
    """List available agents and their trigger keywords."""
    return {
        "agents": [
            {
                "name": "budget",
                "description": "Analyses income, expenses, and savings. Flags overspending.",
                "endpoint": "/agent/budget",
                "keywords": AGENT_KEYWORDS["budget"],
            },
            {
                "name": "debt",
                "description": "Builds Avalanche/Snowball repayment plans with full simulation.",
                "endpoint": "/agent/debt",
                "keywords": AGENT_KEYWORDS["debt"],
            },
            {
                "name": "portfolio",
                "description": "Analyses holdings, calculates Sharpe ratio, recommends rebalancing.",
                "endpoint": "/agent/portfolio",
                "keywords": AGENT_KEYWORDS["portfolio"],
            },
            {
                "name": "sandbox",
                "description": "Evaluates financial expressions and runs scenario simulations.",
                "endpoint": "/agent/sandbox",
                "keywords": AGENT_KEYWORDS["sandbox"],
            },
        ],
        "auto_detection": "Enabled — set agent='llm' in /query to bypass.",
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    General-purpose endpoint with auto-detection.
    Set `agent` to 'budget'|'debt'|'portfolio'|'sandbox'|'llm' to override.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return await orchestrator.process_query(
        query=request.query,
        query_type=request.query_type,
        use_rag=request.use_rag,
        session_id=request.session_id,
        agent_override=request.agent,
    )


@app.post("/agent/budget", response_model=QueryResponse)
async def budget_endpoint(request: BudgetRequest):
    """
    Structured budget analysis.
    Provide `budget_data` with income and expenses for a full breakdown.
    """
    session_id = request.session_id or str(uuid.uuid4())
    return orchestrator.run_budget_agent(request.budget_data, request.query, session_id)


@app.post("/agent/debt", response_model=QueryResponse)
async def debt_endpoint(request: DebtRequest):
    """
    Structured debt repayment plan.
    Provide `debt_data` with debts list and monthly payment budget.
    """
    session_id = request.session_id or str(uuid.uuid4())
    return orchestrator.run_debt_agent(request.debt_data, request.query, session_id)


@app.post("/agent/portfolio", response_model=QueryResponse)
async def portfolio_endpoint(request: PortfolioRequest):
    """
    Structured portfolio analysis with rebalancing recommendations.
    Provide `portfolio_data` with holdings and risk profile.
    """
    session_id = request.session_id or str(uuid.uuid4())
    return orchestrator.run_portfolio_agent(request.portfolio_data, request.query, session_id)


@app.post("/agent/sandbox", response_model=QueryResponse)
async def sandbox_endpoint(request: SandboxRequest):
    """
    Financial calculator and scenario simulator.
    Provide `sandbox_data` with type ('evaluate'|'scenario'|'what_if') and parameters.
    """
    session_id = request.session_id or str(uuid.uuid4())
    return orchestrator.run_sandbox_agent(request.sandbox_data, request.query, session_id)


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    success = orchestrator.db.submit_feedback(
        conversation_id=request.conversation_id,
        rating=request.rating,
        feedback_text=request.feedback_text,
        user_id=request.user_id,
    )
    if not success:
        raise HTTPException(status_code=400, detail="Failed to submit feedback.")
    return {"message": "Feedback submitted successfully."}


@app.get("/conversations")
async def get_conversations(session_id: str = None, limit: int = 50, offset: int = 0):
    conversations = orchestrator.db.get_conversation_history(
        session_id=session_id, limit=limit, offset=offset
    )
    return {"conversations": conversations}


@app.delete("/memory/{session_id}")
async def clear_memory(session_id: str):
    """Clear conversation memory for a session."""
    orchestrator.memory.clear_session(session_id)
    return {"message": f"Memory cleared for session {session_id}"}


@app.get("/memory/{session_id}")
async def get_memory(session_id: str):
    """Return the current conversation history for a session."""
    mem = orchestrator.memory.get(session_id)
    return {
        "session_id": session_id,
        "turn_count": mem.turn_count,
        "max_turns": mem._max_turns,
        "history": [
            {"user": u, "assistant": a} for u, a in mem.turns
        ],
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    usage = orchestrator.db.get_usage_stats()
    feedback = orchestrator.db.get_feedback_stats()
    uptime = time.time() - _start_time
    performance = {
        "total_requests": _request_count,
        "total_errors": _error_count,
        "error_rate": round(_error_count / max(_request_count, 1), 4),
        "uptime_seconds": round(uptime, 1),
        "requests_per_hour": round(_request_count / max(uptime / 3600, 1), 2),
    }
    system_info = {
        "llm": orchestrator.llm.get_status(),
        "rag": orchestrator.rag.get_document_info(),
        "database": orchestrator.db.get_database_info(),
        "memory": orchestrator.memory.get_stats(),
    }
    return StatsResponse(
        conversations=usage, feedback=feedback,
        performance=performance, system=system_info,
    )


@app.get("/docs/info")
async def get_docs_info():
    return orchestrator.rag.get_document_info()


# ── Startup / shutdown events ──────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("Financial Assistant AI v2 starting up...")
    logger.info(f"Backend : {config.BACKEND_URL}")
    logger.info(f"Docs    : {config.DOCS_FOLDER}")
    logger.info(f"Model   : {config.MODEL_PATH}")
    logger.info("Agent routing: ENABLED (auto-detect + manual override)")

    # Validate config — import directly from file to avoid utils/__init__ path issues
    try:
        import importlib.util, os as _os
        _spec = importlib.util.spec_from_file_location(
            "utils_config",
            _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "utils", "config.py"),
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        report = _mod.validate_config()
        for w in report["warnings"]:
            logger.warning(f"[CONFIG WARNING] {w}")
        for e in report["errors"]:
            logger.error(f"[CONFIG ERROR] {e}")
    except Exception as exc:
        logger.warning(f"Config validation skipped: {exc}")

    # Warn if CORS is wide open
    if config.CORS_ORIGINS == ["*"]:
        logger.warning(
            "[SECURITY] CORS is set to allow all origins ('*'). "
            "Set CORS_ORIGINS env var to specific domains in production."
        )


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Financial Assistant AI shutting down...")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ports = [config.BACKEND_PORT + i for i in range(5)]
    for port in ports:
        try:
            logger.info(f"Starting on port {port}...")
            uvicorn.run(
                "core.orchestrator:app",
                host=config.BACKEND_HOST,
                port=port,
                reload=False,
                log_level=config.LOG_LEVEL.lower(),
            )
            break
        except Exception as exc:
            logger.warning(f"Port {port} unavailable: {exc}")
    else:
        raise RuntimeError(f"No available port in range {ports[0]}–{ports[-1]}")


if __name__ == "__main__":
    main()
