"""
utils/config.py
Runtime configuration helpers — environment variable overrides,
validation, and convenience accessors.

The root config object lives in config.py at the project root.
This module adds utilities on top of it without duplicating settings.
"""

import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_env(key: str, default: Any = None, cast: type = str) -> Any:
    """
    Read an environment variable with an optional type cast and default.

    Examples
    --------
    >>> port = get_env("BACKEND_PORT", default=8000, cast=int)
    >>> debug = get_env("DEBUG", default=False, cast=bool)
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        if cast is bool:
            return raw.lower() in ("1", "true", "yes")
        return cast(raw)
    except (ValueError, TypeError) as exc:
        logger.warning(f"Could not cast env var {key}='{raw}' to {cast.__name__}: {exc}. Using default.")
        return default


def get_config_summary() -> Dict[str, Any]:
    """
    Return a safe (no secrets) summary of the current configuration.
    Useful for the /health endpoint and startup logs.
    """
    try:
        from config import config
        return {
            "model_path": config.MODEL_PATH,
            "model_exists": os.path.exists(config.MODEL_PATH),
            "docs_folder": config.DOCS_FOLDER,
            "embedding_model": config.EMBEDDING_MODEL,
            "backend_url": config.BACKEND_URL,
            "frontend_port": config.FRONTEND_PORT,
            "log_level": config.LOG_LEVEL,
            "rag_enabled": config.ENABLE_RAG,
            "analytics_enabled": config.ENABLE_ANALYTICS,
        }
    except Exception as exc:
        logger.error(f"Could not load config summary: {exc}")
        return {"error": str(exc)}


def apply_env_overrides() -> None:
    """
    Apply environment variable overrides to the global config object.
    Call this once at startup (e.g. in main.py) after importing config.

    Supported env vars
    ------------------
    BACKEND_HOST, BACKEND_PORT, FRONTEND_PORT,
    LOG_LEVEL, MODEL_PATH, DOCS_FOLDER,
    CORS_ORIGINS (comma-separated list)
    """
    try:
        from config import config

        host = get_env("BACKEND_HOST")
        if host:
            config.BACKEND_HOST = host
            logger.info(f"Config override: BACKEND_HOST={host}")

        port = get_env("BACKEND_PORT", cast=int)
        if port:
            config.BACKEND_PORT = port
            config.BACKEND_URL = f"http://{config.BACKEND_HOST}:{port}"
            logger.info(f"Config override: BACKEND_PORT={port}")

        frontend_port = get_env("FRONTEND_PORT", cast=int)
        if frontend_port:
            config.FRONTEND_PORT = frontend_port
            logger.info(f"Config override: FRONTEND_PORT={frontend_port}")

        log_level = get_env("LOG_LEVEL")
        if log_level and log_level.upper() in ("DEBUG", "INFO", "WARNING", "ERROR"):
            config.LOG_LEVEL = log_level.upper()
            logging.getLogger().setLevel(getattr(logging, config.LOG_LEVEL))
            logger.info(f"Config override: LOG_LEVEL={config.LOG_LEVEL}")

        model_path = get_env("MODEL_PATH")
        if model_path:
            config.MODEL_PATH = model_path
            logger.info(f"Config override: MODEL_PATH={model_path}")

        docs_folder = get_env("DOCS_FOLDER")
        if docs_folder:
            config.DOCS_FOLDER = docs_folder
            logger.info(f"Config override: DOCS_FOLDER={docs_folder}")

        cors = get_env("CORS_ORIGINS")
        if cors:
            config.CORS_ORIGINS = [o.strip() for o in cors.split(",")]
            logger.info(f"Config override: CORS_ORIGINS={config.CORS_ORIGINS}")

    except Exception as exc:
        logger.error(f"Failed to apply env overrides: {exc}")


def validate_config() -> Dict[str, Any]:
    """
    Validate the current configuration and return a report dict.

    Returns
    -------
    {
        "valid": bool,
        "warnings": [str, ...],
        "errors":   [str, ...]
    }
    """
    warnings: list = []
    errors: list = []

    try:
        from config import config

        # Model
        if not os.path.exists(config.MODEL_PATH):
            warnings.append(
                f"Model file not found at {config.MODEL_PATH}. "
                "Run python downloadModel.py to download it."
            )

        # Docs folder
        if not os.path.exists(config.DOCS_FOLDER):
            warnings.append(
                f"Docs folder missing: {config.DOCS_FOLDER}. "
                "RAG will use fallback knowledge only."
            )

        # Prompts
        if not os.path.exists(config.PROMPTS_DIR):
            warnings.append(
                f"Prompts folder missing: {config.PROMPTS_DIR}. "
                "Default prompts will be created automatically."
            )

        # Ports
        if not (1024 <= config.BACKEND_PORT <= 65535):
            errors.append(f"BACKEND_PORT {config.BACKEND_PORT} is out of valid range.")
        if not (1024 <= config.FRONTEND_PORT <= 65535):
            errors.append(f"FRONTEND_PORT {config.FRONTEND_PORT} is out of valid range.")
        if config.BACKEND_PORT == config.FRONTEND_PORT:
            errors.append("BACKEND_PORT and FRONTEND_PORT cannot be the same.")

        # Log level
        valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if config.LOG_LEVEL not in valid_levels:
            errors.append(f"LOG_LEVEL '{config.LOG_LEVEL}' is invalid. Use one of {valid_levels}.")

    except Exception as exc:
        errors.append(f"Config load failed: {exc}")

    result = {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
    }

    for w in warnings:
        logger.warning(f"Config warning: {w}")
    for e in errors:
        logger.error(f"Config error: {e}")

    return result


def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """
    Return default parameters for a named agent.
    These can be overridden per-request in the API.
    """
    defaults = {
        "budget": {
            "savings_goal_pct": 20.0,
            "currency": "USD",
        },
        "debt": {
            "strategy": "avalanche",
        },
        "portfolio": {
            "risk_profile": "moderate",
            "risk_free_rate": 2.0,
        },
        "sandbox": {
            "default_scenario": "compound_growth",
        },
    }
    return defaults.get(agent_name, {})

