
import os
import sys
import uuid
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import gradio as gr
    import requests
except ImportError as exc:
    print(f"Missing UI dependencies: {exc}")
    print("Run: pip install gradio requests")
    sys.exit(1)

from config import config

logger = logging.getLogger(__name__)

BACKEND = config.BACKEND_URL
TIMEOUT = config.REQUEST_TIMEOUT


# ── Backend helpers ────────────────────────────────────────────────────────────

def _get(path: str, **kwargs) -> Optional[Dict]:
    try:
        r = requests.get(f"{BACKEND}{path}", timeout=10, **kwargs)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _post(path: str, payload: Dict) -> Dict:
    try:
        r = requests.post(f"{BACKEND}{path}", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is the server running?"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Try a simpler question."}
    except Exception as exc:
        return {"error": str(exc)}


def check_backend() -> Tuple[bool, str]:
    data = _get("/health")
    if data is None:
        return False, "Backend offline"
    status = data.get("status", "unknown")
    uptime = data.get("uptime", 0)
    icon = "OK" if status == "healthy" else "DEGRADED"
    return True, f"Backend {icon} — uptime {uptime:.0f}s"


# ── Chat tab ───────────────────────────────────────────────────────────────────

def chat_query(
    message: str,
    history: List,
    agent_choice: str,
    use_rag: bool,
    session_id: str,
) -> Tuple[List, str, str, str]:
    """Send a message to /query and update the chat history (Gradio 6.0 format)."""
    if not message.strip():
        return history, "", "", ""

    agent_override = None if agent_choice == "Auto-detect" else agent_choice.lower()

    payload = {
        "query": message,
        "use_rag": use_rag,
        "session_id": session_id,
        "agent": agent_override,
    }
    data = _post("/query", payload)

    if "error" in data:
        reply = f"Error: {data['error']}"
        agent_badge = ""
        risk_text = ""
    else:
        reply = data.get("response", "No response.")
        agent_used = data.get("agent_used", "llm")
        auto = " (auto)" if data.get("auto_detected") else ""
        agent_badge = f"**Agent:** `{agent_used}`{auto}"
        risk_text = data.get("risk_evaluation", "")

    # Gradio 6.0 messages format
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, "", agent_badge, risk_text


# ── Budget tab ─────────────────────────────────────────────────────────────────

def run_budget(
    income: float,
    housing: float,
    food: float,
    transport: float,
    utilities: float,
    entertainment: float,
    other: float,
    savings_goal: float,
    currency: str,
    session_id: str,
) -> Tuple[str, str]:
    expenses = {}
    for cat, val in [
        ("housing", housing), ("food", food), ("transport", transport),
        ("utilities", utilities), ("entertainment", entertainment), ("other", other),
    ]:
        if val > 0:
            expenses[cat] = val

    payload = {
        "query": "Analyse my budget",
        "session_id": session_id,
        "budget_data": {
            "income": income,
            "expenses": expenses,
            "savings_goal_pct": savings_goal,
            "currency": currency,
        },
    }
    data = _post("/agent/budget", payload)
    if "error" in data:
        return f"❌ {data['error']}", ""
    return data.get("response", ""), data.get("risk_evaluation", "")


# ── Debt tab ───────────────────────────────────────────────────────────────────

def run_debt(
    debts_json: str,
    monthly_payment: float,
    strategy: str,
    session_id: str,
) -> Tuple[str, str]:
    import json
    try:
        debts = json.loads(debts_json)
        if not isinstance(debts, list):
            raise ValueError("Must be a JSON array.")
    except (json.JSONDecodeError, ValueError) as exc:
        return f"❌ Invalid debt JSON: {exc}", ""

    payload = {
        "query": "Help me pay off my debts",
        "session_id": session_id,
        "debt_data": {
            "debts": debts,
            "monthly_payment": monthly_payment,
            "strategy": strategy.lower(),
        },
    }
    data = _post("/agent/debt", payload)
    if "error" in data:
        return f"❌ {data['error']}", ""
    return data.get("response", ""), data.get("risk_evaluation", "")


# ── Portfolio tab ──────────────────────────────────────────────────────────────

def run_portfolio(
    holdings_json: str,
    risk_profile: str,
    session_id: str,
) -> Tuple[str, str]:
    import json
    try:
        holdings = json.loads(holdings_json)
        if not isinstance(holdings, list):
            raise ValueError("Must be a JSON array.")
    except (json.JSONDecodeError, ValueError) as exc:
        return f"❌ Invalid holdings JSON: {exc}", ""

    payload = {
        "query": "Analyse my portfolio",
        "session_id": session_id,
        "portfolio_data": {
            "holdings": holdings,
            "risk_profile": risk_profile.lower(),
        },
    }
    data = _post("/agent/portfolio", payload)
    if "error" in data:
        return f"❌ {data['error']}", ""
    return data.get("response", ""), data.get("risk_evaluation", "")


# ── Sandbox tab ────────────────────────────────────────────────────────────────

def run_sandbox_scenario(
    scenario: str,
    principal: float,
    annual_rate: float,
    years: int,
    monthly_contribution: float,
    session_id: str,
) -> str:
    params: Dict[str, Any] = {"annual_rate": annual_rate, "years": years}
    if scenario == "compound_growth":
        params.update({"principal": principal, "monthly_contribution": monthly_contribution})
    elif scenario == "loan_payoff":
        params.update({"principal": principal, "monthly_payment": monthly_contribution})
    elif scenario == "retirement_savings":
        params.update({"monthly_contribution": monthly_contribution, "initial_savings": principal})
    elif scenario == "inflation_impact":
        params.update({"amount": principal, "annual_inflation": annual_rate})
        params.pop("years", None)
        params["years"] = years

    payload = {
        "query": f"Run {scenario} scenario",
        "session_id": session_id,
        "sandbox_data": {"type": "scenario", "name": scenario, "params": params},
    }
    data = _post("/agent/sandbox", payload)
    if "error" in data:
        return f"❌ {data['error']}"
    return data.get("response", "")


def run_sandbox_expression(expression: str, session_id: str) -> str:
    payload = {
        "query": expression,
        "session_id": session_id,
        "sandbox_data": {"type": "evaluate", "expression": expression},
    }
    data = _post("/agent/sandbox", payload)
    if "error" in data:
        return f"❌ {data['error']}"
    return data.get("response", "")


# ── Interface builder ──────────────────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    session_id = str(uuid.uuid4())

    with gr.Blocks(title=config.UI_TITLE) as demo:

        # ── Header ─────────────────────────────────────────────────────────
        gr.Markdown(f"# 💹 {config.UI_TITLE}")
        gr.Markdown(config.UI_DESCRIPTION)

        with gr.Row():
            status_box = gr.Markdown("Checking backend…")
            refresh_btn = gr.Button("🔄 Refresh", size="sm", scale=0)

        gr.Markdown("---")

        # Hidden session state
        sid = gr.State(session_id)

        # ── Tabs ────────────────────────────────────────────────────────────
        with gr.Tabs():

            # ── Chat ────────────────────────────────────────────────────────
            with gr.TabItem("💬 Chat"):
                chatbot = gr.Chatbot(label="Financial Assistant", height=450)
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ask anything about your finances…",
                        label="Your question",
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    agent_dropdown = gr.Dropdown(
                        choices=["Auto-detect", "Budget", "Debt", "Portfolio", "Sandbox", "LLM"],
                        value="Auto-detect",
                        label="Agent override",
                        scale=2,
                    )
                    use_rag_chk = gr.Checkbox(value=True, label="Use knowledge base", scale=1)
                    clear_btn = gr.Button("🗑️ Clear Chat", scale=1)
                    forget_btn = gr.Button("🧠 Forget Session", scale=1, variant="stop")
                agent_badge = gr.Markdown("")
                with gr.Accordion("⚠️ Risk Assessment", open=False):
                    risk_out = gr.Markdown("")

                send_btn.click(
                    chat_query,
                    [chat_input, chatbot, agent_dropdown, use_rag_chk, sid],
                    [chatbot, chat_input, agent_badge, risk_out],
                )
                chat_input.submit(
                    chat_query,
                    [chat_input, chatbot, agent_dropdown, use_rag_chk, sid],
                    [chatbot, chat_input, agent_badge, risk_out],
                )
                clear_btn.click(lambda: ([], "", "", ""), outputs=[chatbot, chat_input, agent_badge, risk_out])

                def forget_session(session_id: str):
                    _post(f"/memory/{session_id}", {})  # DELETE isn't easy in Gradio, use helper
                    requests.delete(f"{BACKEND}/memory/{session_id}", timeout=5)
                    return [], "", "🧠 Memory cleared — starting fresh.", ""

                forget_btn.click(forget_session, inputs=[sid], outputs=[chatbot, chat_input, agent_badge, risk_out])

                gr.Examples(
                    examples=[
                        ["How do I start investing with R5,000?"],
                        ["Should I pay off debt or invest first?"],
                        ["What is the 4% retirement rule?"],
                        ["How does compound interest work?"],
                    ],
                    inputs=chat_input,
                )

            # ── Budget ──────────────────────────────────────────────────────
            with gr.TabItem("📊 Budget Analyser"):
                gr.Markdown("Enter your monthly income and expenses for a personalised analysis.")
                with gr.Row():
                    with gr.Column():
                        b_income = gr.Number(label="Monthly Income", value=5000, minimum=0)
                        b_savings_goal = gr.Slider(5, 50, value=20, step=5, label="Savings Goal (%)")
                        b_currency = gr.Dropdown(["USD", "EUR", "GBP", "ZAR"], value="USD", label="Currency")
                    with gr.Column():
                        b_housing = gr.Number(label="Housing / Rent", value=1500, minimum=0)
                        b_food = gr.Number(label="Food & Groceries", value=600, minimum=0)
                        b_transport = gr.Number(label="Transport", value=300, minimum=0)
                        b_utilities = gr.Number(label="Utilities", value=200, minimum=0)
                        b_entertainment = gr.Number(label="Entertainment", value=150, minimum=0)
                        b_other = gr.Number(label="Other", value=0, minimum=0)

                b_submit = gr.Button("Analyse Budget", variant="primary")
                b_output = gr.Markdown(label="Analysis")
                with gr.Accordion("⚠️ Risk Assessment", open=False):
                    b_risk = gr.Markdown("")

                b_submit.click(
                    run_budget,
                    [b_income, b_housing, b_food, b_transport, b_utilities,
                     b_entertainment, b_other, b_savings_goal, b_currency, sid],
                    [b_output, b_risk],
                )

            # ── Debt ────────────────────────────────────────────────────────
            with gr.TabItem("💳 Debt Coach"):
                gr.Markdown(
                    "Paste your debts as a JSON array. Each entry needs: "
                    "`name`, `balance`, `interest_rate`, `minimum_payment`."
                )
                d_json = gr.Textbox(
                    label="Debts (JSON)",
                    lines=8,
                    value='[\n  {"name": "Credit Card", "balance": 3000, "interest_rate": 22.0, "minimum_payment": 60},\n  {"name": "Car Loan", "balance": 8000, "interest_rate": 6.5, "minimum_payment": 200}\n]',
                )
                with gr.Row():
                    d_payment = gr.Number(label="Monthly repayment budget", value=600, minimum=0)
                    d_strategy = gr.Radio(["Avalanche", "Snowball"], value="Avalanche", label="Strategy")
                d_submit = gr.Button("Build Repayment Plan", variant="primary")
                d_output = gr.Markdown(label="Repayment Plan")
                with gr.Accordion("⚠️ Risk Assessment", open=False):
                    d_risk = gr.Markdown("")

                d_submit.click(
                    run_debt,
                    [d_json, d_payment, d_strategy, sid],
                    [d_output, d_risk],
                )

            # ── Portfolio ───────────────────────────────────────────────────
            with gr.TabItem("📈 Portfolio Analyser"):
                gr.Markdown(
                    "Paste your holdings as a JSON array. Each entry needs: "
                    "`asset_class`, `name`, `value`."
                )
                p_json = gr.Textbox(
                    label="Holdings (JSON)",
                    lines=8,
                    value='[\n  {"asset_class": "stocks", "name": "S&P 500 ETF", "value": 60000},\n  {"asset_class": "bonds", "name": "US Treasuries", "value": 25000},\n  {"asset_class": "cash", "name": "Money Market", "value": 15000}\n]',
                )
                p_risk_profile = gr.Radio(
                    ["Conservative", "Moderate", "Aggressive"],
                    value="Moderate",
                    label="Risk Profile",
                )
                p_submit = gr.Button("Analyse Portfolio", variant="primary")
                p_output = gr.Markdown(label="Portfolio Analysis")
                with gr.Accordion("⚠️ Risk Assessment", open=False):
                    p_risk = gr.Markdown("")

                p_submit.click(
                    run_portfolio,
                    [p_json, p_risk_profile, sid],
                    [p_output, p_risk],
                )

            # ── Sandbox ─────────────────────────────────────────────────────
            with gr.TabItem("🧮 Calculator"):
                with gr.Tabs():
                    with gr.TabItem("Scenarios"):
                        gr.Markdown("Run a financial projection scenario.")
                        sc_scenario = gr.Dropdown(
                            choices=["compound_growth", "loan_payoff", "retirement_savings", "inflation_impact"],
                            value="compound_growth",
                            label="Scenario",
                        )
                        with gr.Row():
                            sc_principal = gr.Number(label="Principal / Amount", value=10000)
                            sc_rate = gr.Number(label="Annual Rate (%)", value=7.0)
                            sc_years = gr.Number(label="Years", value=20, precision=0)
                            sc_monthly = gr.Number(label="Monthly Contribution / Payment", value=200)
                        sc_submit = gr.Button("Run Scenario", variant="primary")
                        sc_output = gr.Markdown(label="Results")
                        sc_submit.click(
                            run_sandbox_scenario,
                            [sc_scenario, sc_principal, sc_rate, sc_years, sc_monthly, sid],
                            sc_output,
                        )

                    with gr.TabItem("Expression"):
                        gr.Markdown("Evaluate any financial formula safely.")
                        ex_input = gr.Textbox(
                            label="Expression",
                            placeholder="e.g. 1000 * (1 + 0.07) ** 10",
                            value="1000 * (1 + 0.07) ** 10",
                        )
                        ex_submit = gr.Button("Calculate", variant="primary")
                        ex_output = gr.Markdown(label="Result")
                        ex_submit.click(run_sandbox_expression, [ex_input, sid], ex_output)

        # ── Status helpers ──────────────────────────────────────────────────
        def refresh_status():
            _, msg = check_backend()
            return msg

        refresh_btn.click(refresh_status, outputs=status_box)
        demo.load(refresh_status, outputs=status_box)

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Starting Financial Assistant UI")

    ok, status = check_backend()
    if not ok:
        logger.warning("Backend not detected — UI will start with limited functionality.")

    demo = build_interface()

    # Enable queue with extended timeout for slow CPU inference.
    # Without this Gradio kills requests after ~60 seconds regardless
    # of the requests.post timeout set in _post().
    demo.queue(
        max_size=4,
        default_concurrency_limit=1,
    )

    ports = [config.FRONTEND_PORT + i for i in range(5)]
    for port in ports:
        try:
            logger.info(f"Launching UI on port {port}")
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                inbrowser=True,
                show_error=True,
                quiet=False,
                theme=gr.themes.Soft(),
            )
            break
        except OSError:
            logger.warning(f"Port {port} busy, trying next…")
    else:
        raise RuntimeError(f"No available port in range {ports[0]}–{ports[-1]}")


if __name__ == "__main__":
    main()