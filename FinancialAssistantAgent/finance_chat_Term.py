"""
finance_chat_Term.py
--------------------
Legacy compatibility shim.

All LLM logic has moved to core/llm_engine.py.
This file re-exports the same public names so any remaining
imports keep working without changes.

Do not add new logic here — use core/llm_engine directly instead.
"""

from core.llm_engine import (
    llm_engine,
    chat_fn,
    chat_with_context,
    get_retrieved_docs,
)

# DEFAULT_SYSTEM_PROMPT was referenced in the old app/main.py
DEFAULT_SYSTEM_PROMPT = (
    "You are FinanceBot, a professional AI financial assistant. "
    "Provide accurate, structured financial advice. Always include "
    "relevant risk considerations and maintain a professional tone."
)

__all__ = [
    "llm_engine",
    "chat_fn",
    "chat_with_context",
    "get_retrieved_docs",
    "DEFAULT_SYSTEM_PROMPT",
]