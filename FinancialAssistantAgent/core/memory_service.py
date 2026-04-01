"""
core/memory_service.py
Per-session conversation memory with SQLite persistence and in-memory cache.

Design
------
- Each session has a list of (user, assistant) turn tuples
- History is capped at MAX_TURNS to stay within LLM context limits
- In-memory cache (dict) for fast reads during a session
- SQLite persistence so history survives server restarts
- Thread-safe via a single lock per service instance
"""

import json
import logging
import sqlite3
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum number of (user, assistant) turns kept per session
MAX_TURNS = 10


class ConversationMemory:
    """
    Holds the conversation history for a single session.

    Attributes
    ----------
    session_id : str
    turns      : list of (user_message, assistant_message) tuples
    max_turns  : int — older turns are dropped when this limit is exceeded
    """

    def __init__(self, session_id: str, max_turns: int = MAX_TURNS):
        self.session_id = session_id
        self.max_turns = max_turns
        self._turns: List[Tuple[str, str]] = []

    # ── Mutations ──────────────────────────────────────────────────────────

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """Append a turn and trim if over the cap."""
        self._turns.append((user_message, assistant_message))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]

    def clear(self) -> None:
        """Wipe all turns for this session."""
        self._turns = []

    # ── Accessors ──────────────────────────────────────────────────────────

    @property
    def turns(self) -> List[Tuple[str, str]]:
        """Return a shallow copy so callers cannot mutate internal state."""
        return list(self._turns)

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def is_empty(self) -> bool:
        return len(self._turns) == 0

    def last_n(self, n: int) -> List[Tuple[str, str]]:
        """Return the most recent n turns."""
        return self._turns[-n:] if n < len(self._turns) else list(self._turns)

    def as_text(self, user_label: str = "User", assistant_label: str = "Assistant") -> str:
        """
        Format history as a plain-text block, useful for prompt injection.

        Example output
        --------------
        User: What is dollar-cost averaging?
        Assistant: Dollar-cost averaging is...
        """
        lines: List[str] = []
        for user_msg, asst_msg in self._turns:
            lines.append(f"{user_label}: {user_msg}")
            lines.append(f"{assistant_label}: {asst_msg}")
        return "\n".join(lines)

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(self._turns)

    @classmethod
    def from_json(cls, session_id: str, data: str, max_turns: int = MAX_TURNS) -> "ConversationMemory":
        mem = cls(session_id, max_turns=max_turns)
        try:
            turns = json.loads(data)
            if isinstance(turns, list):
                for item in turns:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        mem._turns.append((str(item[0]), str(item[1])))
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(f"Could not restore memory for session {session_id}: {exc}")
        # Respect cap after restore
        if len(mem._turns) > max_turns:
            mem._turns = mem._turns[-max_turns:]
        return mem


class MemoryService:
    """
    Manages conversation memory for all active sessions.

    - In-memory cache for fast per-request access
    - SQLite persistence for recovery across restarts
    - Thread-safe
    """

    def __init__(self, db_path: str = "financial_assistant.db", max_turns: int = MAX_TURNS):
        self._db_path = db_path
        self._max_turns = max_turns
        self._cache: Dict[str, ConversationMemory] = {}
        self._lock = threading.Lock()
        self._init_db()
        logger.info(f"MemoryService initialised (max_turns={max_turns}, db={db_path})")

    # ── DB setup ───────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_memory (
                        session_id  TEXT PRIMARY KEY,
                        turns_json  TEXT NOT NULL DEFAULT '[]',
                        updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as exc:
            logger.error(f"MemoryService DB init failed: {exc}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Public API ─────────────────────────────────────────────────────────

    def get(self, session_id: str) -> ConversationMemory:
        """
        Return the ConversationMemory for a session.
        Loads from DB on first access, then caches in memory.
        Creates an empty memory if the session does not exist yet.
        """
        with self._lock:
            if session_id not in self._cache:
                self._cache[session_id] = self._load_from_db(session_id)
            return self._cache[session_id]

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Add a turn to a session and persist it."""
        with self._lock:
            mem = self._cache.get(session_id)
            if mem is None:
                mem = self._load_from_db(session_id)
                self._cache[session_id] = mem

            mem.add_turn(user_message, assistant_message)
            self._save_to_db(mem)

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """Convenience method — returns the turns list directly."""
        return self.get(session_id).turns

    def clear_session(self, session_id: str) -> None:
        """Wipe memory for a session (cache + DB)."""
        with self._lock:
            if session_id in self._cache:
                self._cache[session_id].clear()
            self._delete_from_db(session_id)
            logger.info(f"Memory cleared for session {session_id}")

    def get_all_session_ids(self) -> List[str]:
        """Return all session IDs that have stored memory."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT session_id FROM conversation_memory ORDER BY updated_at DESC"
                ).fetchall()
                return [row["session_id"] for row in rows]
        except Exception as exc:
            logger.error(f"Failed to list sessions: {exc}")
            return []

    def get_stats(self) -> Dict:
        """Return memory usage stats."""
        return {
            "active_sessions_cached": len(self._cache),
            "total_sessions_in_db": len(self.get_all_session_ids()),
            "max_turns_per_session": self._max_turns,
        }

    # ── DB helpers ─────────────────────────────────────────────────────────

    def _load_from_db(self, session_id: str) -> ConversationMemory:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT turns_json FROM conversation_memory WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row:
                    return ConversationMemory.from_json(
                        session_id, row["turns_json"], max_turns=self._max_turns
                    )
        except Exception as exc:
            logger.warning(f"Could not load memory for {session_id}: {exc}")
        return ConversationMemory(session_id, max_turns=self._max_turns)

    def _save_to_db(self, mem: ConversationMemory) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO conversation_memory (session_id, turns_json, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(session_id) DO UPDATE SET
                        turns_json = excluded.turns_json,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (mem.session_id, mem.to_json()),
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"Could not persist memory for {mem.session_id}: {exc}")

    def _delete_from_db(self, session_id: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM conversation_memory WHERE session_id = ?",
                    (session_id,),
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"Could not delete memory for {session_id}: {exc}")


# ── Global singleton ───────────────────────────────────────────────────────────

try:
    from config import config
    _db_path = config.DATABASE_PATH
except Exception:
    _db_path = "financial_assistant.db"

memory_service = MemoryService(db_path=_db_path)