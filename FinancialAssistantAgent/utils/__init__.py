"""
utils/
------
Shared utility helpers used across agents, core services, and the UI.

    from utils.data_loader import load_budget_from_csv, validate_debts, format_currency
    from utils.config     import validate_config, apply_env_overrides, get_agent_config
"""

from utils.data_loader import (
    load_budget_from_csv,
    load_budget_from_dict,
    load_debts_from_csv,
    validate_debts,
    load_portfolio_from_csv,
    validate_portfolio,
    load_json,
    save_json,
    load_text_documents,
    format_currency,
    format_percentage,
    summarise_dict,
)

from utils.config import (
    get_env,
    get_config_summary,
    apply_env_overrides,
    validate_config,
    get_agent_config,
)


__all__ = [
    # data_loader
    "load_budget_from_csv",
    "load_budget_from_dict",
    "load_debts_from_csv",
    "validate_debts",
    "load_portfolio_from_csv",
    "validate_portfolio",
    "load_json",
    "save_json",
    "load_text_documents",
    "format_currency",
    "format_percentage",
    "summarise_dict",
    # config
    "get_env",
    "get_config_summary",
    "apply_env_overrides",
    "validate_config",
    "get_agent_config",
]
