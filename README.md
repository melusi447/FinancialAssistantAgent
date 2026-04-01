# 💹 Financial Assistant Agent

An AI-powered financial advisory system with specialist agents, Retrieval-Augmented Generation (RAG), and a Gradio web interface.

---

## Features

| Feature | Description |
|---|---|
| **4 specialist agents** | Budget Advisor, Debt Coach, Portfolio Analyser, Financial Calculator |
| **Auto-routing** | Queries are automatically routed to the right agent based on keywords |
| **RAG** | FAISS + Sentence Transformers semantic search over your own financial documents |
| **Conversation memory** | Per-session history persisted to SQLite — survives server restarts |
| **REST API** | FastAPI backend with full Swagger docs at `/docs` |
| **Gradio UI** | Tabbed web interface with dedicated forms for each agent |

---

## Project Structure

```
FinancialAssistantAI/
├── agents/                   # Specialist agents
│   ├── budget_advisor.py     # Income/expense analysis, 50/30/20 benchmarks
│   ├── debt_coach.py         # Avalanche/Snowball repayment strategies
│   ├── portfolio_agent.py    # Allocation, Sharpe ratio, rebalancing
│   └── sandbox_executor.py  # Safe expression eval + scenario simulations
│
├── core/                     # Backend services
│   ├── orchestrator.py       # FastAPI app + agent routing (single source of truth)
│   ├── llm_engine.py         # Llama.cpp LLM wrapper
│   ├── rag_service.py        # FAISS semantic search
│   ├── memory_service.py     # Per-session conversation memory
│   ├── database_service.py   # SQLite logging + analytics
│   └── prompt_service.py     # Prompt template management
│
├── ui/
│   └── app.py                # Gradio web interface (single source of truth)
│
├── utils/
│   ├── data_loader.py        # CSV/JSON loaders, validators, formatters
│   └── config.py             # Env var overrides + config validation
│
├── tests/
│   ├── test_budget_advisor.py
│   ├── test_debt_coach.py
│   └── test_sandbox.py
│
├── docs/                     # Drop your financial documents here for RAG
├── models/                   # Drop your GGUF model file here
├── prompts/                  # Prompt templates (auto-created on first run)
├── logs/                     # Log files (auto-created on first run)
│
├── config.py                 # Central configuration
├── main.py                   # Full system launcher (backend + frontend)
├── run_backend.py            # Backend only
├── run_frontend.py           # Frontend only
├── finance_chat_Term.py      # Legacy shim — re-exports from core/llm_engine
├── downloadModel.py          # Downloads the LLM from Hugging Face
├── install.py                # One-command dependency installer + setup checker
├── kill_ports.py             # Frees ports 8000/7860 when they get stuck (Windows)
├── setup.py                  # Package setup for pip install -e .
└── requirements.txt
```

---

## Quickstart

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/your-username/FinancialAssistantAI.git
cd FinancialAssistantAI

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
# Option A — standard pip
pip install -r requirements.txt

# Option B — one-command installer (checks Python version, creates dirs, verifies install)
python install.py
```

> **Note for Windows users:** `llama-cpp-python` requires a C++ compiler.
> Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> and select **"Desktop development with C++"** before running pip install.

### 3. Download the LLM model

```bash
python downloadModel.py
```

This downloads `finance-Llama3-8B.Q2_K.gguf` (~2.7 GB) from Hugging Face into `models/`.

> The system still works without the model — agents fall back to rule-based responses and RAG still retrieves documents.

### 4. (Optional) Add your own financial documents

Drop `.txt`, `.md`, or `.pdf` files into the `docs/` folder.
These are indexed automatically on startup and used to enrich every response.

### 5. Run the full system

```bash
python main.py
```

This starts both the backend API and the Gradio UI.

| Service | URL |
|---|---|
| Gradio UI | http://127.0.0.1:7861 |
| API | http://127.0.0.1:8001 |
| Swagger docs | http://127.0.0.1:8001/docs |
| Health check | http://127.0.0.1:8001/health |

---

## Running services separately

```bash
# Backend only
python run_backend.py

# Frontend only (assumes backend is already running)
python run_frontend.py

# If ports are stuck (Windows)
python kill_ports.py
```

---

## API Reference

### General chat — auto-detects agent

```http
POST /query
Content-Type: application/json

{
  "query": "How do I pay off my credit card debt?",
  "use_rag": true,
  "agent": null          // null = auto-detect; or "budget"|"debt"|"portfolio"|"sandbox"|"llm"
}
```

### Budget analysis

```http
POST /agent/budget
Content-Type: application/json

{
  "budget_data": {
    "income": 5000,
    "expenses": { "housing": 1500, "food": 600, "transport": 300 },
    "savings_goal_pct": 20,
    "currency": "USD"
  }
}
```

### Debt repayment plan

```http
POST /agent/debt
Content-Type: application/json

{
  "debt_data": {
    "debts": [
      { "name": "Credit Card", "balance": 3000, "interest_rate": 22.0, "minimum_payment": 60 },
      { "name": "Car Loan",    "balance": 8000, "interest_rate": 6.5,  "minimum_payment": 200 }
    ],
    "monthly_payment": 600,
    "strategy": "avalanche"
  }
}
```

### Portfolio analysis

```http
POST /agent/portfolio
Content-Type: application/json

{
  "portfolio_data": {
    "holdings": [
      { "asset_class": "stocks",     "name": "S&P 500 ETF",  "value": 60000 },
      { "asset_class": "bonds",      "name": "US Treasuries", "value": 25000 },
      { "asset_class": "real_estate","name": "REIT",          "value": 10000 },
      { "asset_class": "cash",       "name": "Money Market",  "value": 5000  }
    ],
    "risk_profile": "moderate"
  }
}
```

### Financial calculator

```http
POST /agent/sandbox
Content-Type: application/json

// Expression evaluation
{ "sandbox_data": { "type": "evaluate", "expression": "1000 * (1 + 0.07) ** 10" } }

// Scenario simulation
{ "sandbox_data": { "type": "scenario", "name": "compound_growth",
    "params": { "principal": 10000, "annual_rate": 7, "years": 20 } } }

// What-if analysis
{ "sandbox_data": { "type": "what_if", "scenario": "compound_growth",
    "base_params": { "principal": 10000, "annual_rate": 7, "years": 20 },
    "variations": { "annual_rate": [5, 7, 9, 12] } } }
```

### Conversation memory

```http
GET    /memory/{session_id}    # View history for a session
DELETE /memory/{session_id}    # Clear history for a session
```

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Configuration

All settings live in `config.py`. They can be overridden with environment variables at runtime — no code changes needed.

| Environment variable | Default | Description |
|---|---|---|
| `BACKEND_HOST` | `127.0.0.1` | API host |
| `BACKEND_PORT` | `8000` | API port |
| `FRONTEND_PORT` | `7860` | Gradio UI port |
| `MODEL_PATH` | `models/finance-Llama3-8B.Q2_K.gguf` | Path to GGUF model |
| `DOCS_FOLDER` | `docs/` | RAG document folder |
| `LOG_LEVEL` | `INFO` | Logging level |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

Example — run on a different port without editing any file:

```bash
BACKEND_PORT=9000 FRONTEND_PORT=8080 python main.py
```

---

## Agent auto-detection

The `/query` endpoint automatically routes to the best agent based on keywords in the query.
You can override this by setting `"agent"` in the request body.

| Query contains… | Routed to |
|---|---|
| budget, spending, expenses, income | `budget` |
| debt, loan, credit card, pay off, interest rate | `debt` |
| portfolio, invest, stocks, ETF, rebalance | `portfolio` |
| calculate, compound interest, what if, scenario | `sandbox` |
| anything else | `llm` (direct) |

---

## Tech stack

| Component | Technology |
|---|---|
| LLM | [Llama.cpp](https://github.com/ggerganov/llama.cpp) — `finance-Llama3-8B` GGUF |
| Embeddings | [Sentence Transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2` |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) |
| API | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| UI | [Gradio](https://gradio.app/) |
| Database | SQLite (via Python stdlib) |
| Testing | [pytest](https://pytest.org/) |

---

## Troubleshooting

**Backend won't start — port already in use**
```bash
python kill_ports.py
python run_backend.py
```

**Model not found**
```bash
python downloadModel.py
```

**`llama-cpp-python` install fails on Windows**
Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first, then:
```bash
pip install llama-cpp-python --prefer-binary
```

**RAG returns no results**
- Add `.txt` or `.md` files to the `docs/` folder and restart the backend
- The fallback knowledge base is used automatically if `docs/` is empty

**Frontend connects but gets no response**
- Check the backend is running: `python run_backend.py`
- Check logs in `logs/orchestrator.log`

---

## Disclaimer

This system provides general financial information only. It is **not** a substitute for advice from a qualified financial advisor. Always consult a professional before making investment or financial decisions.
