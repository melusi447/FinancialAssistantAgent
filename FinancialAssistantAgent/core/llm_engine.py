"""
Core LLM Engine
Robust LLM integration using llama-cpp-python with lazy loading,
thread safety, compact prompts, and timeout protection.

CPU-optimised: uses a minimal prompt format to keep generation fast.
"""

import logging
import os
import sys
import asyncio
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from config import config

logger = logging.getLogger(__name__)

# ── Compact system prompt — kept tiny to save tokens on CPU ───────────────────
_SYSTEM_PROMPT = (
    "You are FinanceBot, a concise AI financial assistant. "
    "Give short, direct answers. Focus on key facts only."
)

# Hard limits for CPU performance
_MAX_TOKENS    = 60    # guarantees response in <60s even at 1 tok/sec
_MAX_HISTORY   = 2     # only last 2 turns to keep prompt short
_MAX_RAG_CHARS = 300   # truncate each RAG doc to 300 chars
_MAX_RAG_DOCS  = 1     # only use the single best RAG doc


class LLMNotReadyError(RuntimeError):
    """Raised when the LLM is called before it has been successfully initialised."""


class LLMEngine:
    """
    Thread-safe LLM engine with lazy loading.

    The model is only loaded on the first call to chat() or generate_response(),
    keeping startup fast even when the model file is large.
    """

    def __init__(self) -> None:
        self._llm = None
        self._initialized = False
        self._lock = threading.Lock()
        self._last_used: float = 0.0

    # ── Initialisation ─────────────────────────────────────────────────────

    def _initialize_llm(self):
        """Load the model. Must be called with self._lock held."""
        if self._initialized:
            return

        model_path = config.MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Run `python downloadModel.py` to download it."
            )

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Run: pip install llama-cpp-python --prefer-binary"
            ) from exc

        logger.info(f"Loading LLM from {model_path} ...")
        self._llm = Llama(
            model_path=model_path,
            n_ctx=config.MODEL_CONTEXT_SIZE,
            n_threads=config.MODEL_N_THREADS,
            n_batch=config.MODEL_N_BATCH,
            verbose=False,
        )
        self._initialized = True
        logger.info("LLM loaded successfully.")

    def _get_llm(self):
        """Return the LLM instance, initialising it on first call."""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._initialize_llm()
        self._last_used = time.time()
        return self._llm

    # ── Core generation ────────────────────────────────────────────────────

    def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate a raw text completion from a prompt string."""
        if not self._is_model_available():
            raise LLMNotReadyError(
                f"Model not found at {config.MODEL_PATH}. "
                "Run `python downloadModel.py`."
            )

        llm = self._get_llm()

        # Hard cap on tokens — never exceed _MAX_TOKENS on CPU
        tokens = min(max_tokens or _MAX_TOKENS, _MAX_TOKENS)

        try:
            result = llm(
                prompt=prompt,
                max_tokens=tokens,
                temperature=temperature or config.MODEL_TEMPERATURE,
                top_p=top_p or config.MODEL_TOP_P,
                stop=stop or ["User:", "###", "\n\n\n"],
                echo=False,
            )
            text = result["choices"][0]["text"].strip()
            if not text:
                logger.warning("LLM returned an empty response.")
            return text

        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected LLM output format: {exc}") from exc
        except Exception as exc:
            logger.error(f"LLM generation error: {exc}")
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

    # ── Prompt builder ─────────────────────────────────────────────────────

    def _build_prompt(
        self,
        message: str,
        history: List[Tuple[str, str]],
        system: str,
        context: str = "",
    ) -> str:
        """
        Build a compact prompt.
        - Only uses the last _MAX_HISTORY turns
        - Truncates RAG context to _MAX_RAG_CHARS
        - Keeps total prompt well under 512 tokens
        """
        if context and len(context) > _MAX_RAG_CHARS:
            context = context[:_MAX_RAG_CHARS] + "..."

        parts = [f"System: {system}"]

        if context:
            parts.append(f"Context: {context}")

        # Only last N turns, truncated
        recent = history[-_MAX_HISTORY:] if history else []
        for user_msg, asst_msg in recent:
            u = user_msg[:200] if len(user_msg) > 200 else user_msg
            a = asst_msg[:200] if len(asst_msg) > 200 else asst_msg
            parts.append(f"User: {u}\nAssistant: {a}")

        parts.append(f"User: {message}\nAssistant:")
        return "\n".join(parts)

    # ── Chat ───────────────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system_prompt: Optional[str] = None,
        use_rag: bool = False,
        context: str = "",
    ) -> str:
        """
        Multi-turn chat with optional RAG context injection.
        Uses compact prompt format optimised for CPU inference.
        """
        history = history or []
        system = system_prompt or _SYSTEM_PROMPT

        # Truncate RAG context — only the first/best doc, 400 chars max
        if use_rag and context:
            first_doc = context.split("\n\n")[0]
            context = first_doc[:_MAX_RAG_CHARS]

        prompt = self._build_prompt(message, history, system, context)
        logger.debug(f"Prompt length: {len(prompt)} chars")

        try:
            return self.generate_response(prompt)
        except LLMNotReadyError as exc:
            logger.error(str(exc))
            return (
                "The AI model is not available. "
                "Please run `python downloadModel.py` to download it."
            )
        except RuntimeError as exc:
            logger.error(f"chat() failed: {exc}")
            return "I encountered an error generating a response. Please try again."

    # ── Async wrappers ─────────────────────────────────────────────────────

    async def async_generate_response(self, prompt, max_tokens=None,
                                       temperature=None, top_p=None, stop=None):
        return await asyncio.to_thread(
            self.generate_response, prompt, max_tokens, temperature, top_p, stop
        )

    async def async_chat(self, message, history=None, system_prompt=None,
                          use_rag=False, context=""):
        return await asyncio.to_thread(
            self.chat, message, history, system_prompt, use_rag, context
        )

    async def async_chat_with_context(self, user_message, history=None):
        return await asyncio.to_thread(self.chat_with_context, user_message, history)

    def chat_with_context(
        self,
        user_message: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """RAG-enhanced chat — retrieves ONE short document then calls chat()."""
        try:
            from core.rag_service import rag_service
            docs = rag_service.retrieve_docs(user_message, k=_MAX_RAG_DOCS)
            context = docs[0][:_MAX_RAG_CHARS] if docs else ""
            return self.chat(user_message, history=history, use_rag=bool(context), context=context)
        except Exception as exc:
            logger.error(f"RAG context retrieval failed: {exc}. Falling back to plain chat.")
            return self.chat(user_message, history=history)

    def get_retrieved_docs(self, user_message: str) -> List[str]:
        try:
            from core.rag_service import rag_service
            return rag_service.retrieve_docs(user_message, k=_MAX_RAG_DOCS)
        except Exception as exc:
            logger.error(f"get_retrieved_docs failed: {exc}")
            return []

    # ── Status ─────────────────────────────────────────────────────────────

    def _is_model_available(self) -> bool:
        return os.path.exists(config.MODEL_PATH)

    def is_ready(self) -> bool:
        return self._is_model_available() and self._initialized

    def get_status(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "model_available": self._is_model_available(),
            "model_path": config.MODEL_PATH,
            "last_used": self._last_used,
            "max_tokens": _MAX_TOKENS,
        }


# ── Global singleton ───────────────────────────────────────────────────────────
llm_engine = LLMEngine()


# ── Backward-compatibility functions ─────────────────────────────────────────
def chat_fn(message, history=None, system_prompt=None, use_rag=False, context=""):
    return llm_engine.chat(message, history, system_prompt, use_rag, context)

def chat_with_context(user_message, history=None):
    return llm_engine.chat_with_context(user_message, history)

def get_retrieved_docs(user_message):
    return llm_engine.get_retrieved_docs(user_message)



