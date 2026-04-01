"""
utils/data_loader.py
Utilities for loading, validating, and preprocessing financial data
before it's passed to agents or the RAG service.
"""

import os
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Budget helpers ─────────────────────────────────────────────────────────────

def load_budget_from_csv(file_path: str) -> Dict[str, Any]:
    """
    Load budget data from a two-column CSV (category, amount).

    Expected format
    ---------------
    income,5000
    housing,1500
    food,600
    transport,300

    Returns
    -------
    {"income": 5000.0, "expenses": {"housing": 1500.0, ...}}
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Budget file not found: {file_path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path.suffix}")

    income = 0.0
    expenses: Dict[str, float] = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, start=1):
            if len(row) < 2 or not row[0].strip():
                continue
            category = row[0].strip().lower()
            try:
                amount = float(row[1].strip().replace(",", ""))
            except ValueError:
                logger.warning(f"Row {i}: could not parse amount '{row[1]}' — skipped.")
                continue
            if category == "income":
                income = amount
            else:
                expenses[category] = amount

    logger.info(f"Loaded budget from {file_path}: income={income}, {len(expenses)} expense categories")
    return {"income": income, "expenses": expenses}


def load_budget_from_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalise a budget dict coming from the API or UI.

    Accepts
    -------
    {"income": 5000, "expenses": {"housing": 1500, "food": 600}}

    Raises
    ------
    ValueError if required keys are missing or values are negative.
    """
    if "income" not in raw:
        raise ValueError("Budget data must include 'income'.")
    if "expenses" not in raw or not isinstance(raw["expenses"], dict):
        raise ValueError("Budget data must include 'expenses' as a dict.")

    income = float(raw["income"])
    if income < 0:
        raise ValueError("Income cannot be negative.")

    expenses = {}
    for cat, amt in raw["expenses"].items():
        val = float(amt)
        if val < 0:
            raise ValueError(f"Expense amount for '{cat}' cannot be negative.")
        expenses[cat.lower().strip()] = val

    return {
        "income": income,
        "expenses": expenses,
        "savings_goal_pct": float(raw.get("savings_goal_pct", 20.0)),
        "currency": str(raw.get("currency", "USD")),
    }


# ── Debt helpers ───────────────────────────────────────────────────────────────

def load_debts_from_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Load debt accounts from a CSV file.

    Expected columns (header row required)
    ---------------------------------------
    name, balance, interest_rate, minimum_payment[, currency]

    Returns
    -------
    List of debt dicts ready for DebtCoachAgent.analyze()
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Debt file not found: {file_path}")

    debts: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"name", "balance", "interest_rate", "minimum_payment"}
        if not required.issubset({c.lower().strip() for c in (reader.fieldnames or [])}):
            raise ValueError(f"CSV must have columns: {required}")

        for i, row in enumerate(reader, start=2):
            try:
                debt = {
                    "name": row["name"].strip(),
                    "balance": float(row["balance"].replace(",", "")),
                    "interest_rate": float(row["interest_rate"].replace(",", "")),
                    "minimum_payment": float(row["minimum_payment"].replace(",", "")),
                    "currency": row.get("currency", "USD").strip(),
                }
                debts.append(debt)
            except (KeyError, ValueError) as exc:
                logger.warning(f"Row {i} skipped: {exc}")

    logger.info(f"Loaded {len(debts)} debt account(s) from {file_path}")
    return debts


def validate_debts(debts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Validate a list of debt dicts. Returns (valid_debts, error_messages).
    """
    valid, errors = [], []
    required_keys = {"name", "balance", "interest_rate", "minimum_payment"}

    for i, debt in enumerate(debts):
        missing = required_keys - set(debt.keys())
        if missing:
            errors.append(f"Debt #{i+1}: missing fields {missing}")
            continue
        try:
            validated = {
                "name": str(debt["name"]).strip(),
                "balance": float(debt["balance"]),
                "interest_rate": float(debt["interest_rate"]),
                "minimum_payment": float(debt["minimum_payment"]),
                "currency": str(debt.get("currency", "USD")),
            }
            if validated["balance"] < 0:
                errors.append(f"Debt '{validated['name']}': balance cannot be negative.")
                continue
            if not (0 < validated["interest_rate"] < 200):
                errors.append(f"Debt '{validated['name']}': interest_rate must be 0–200.")
                continue
            valid.append(validated)
        except (ValueError, TypeError) as exc:
            errors.append(f"Debt #{i+1}: {exc}")

    return valid, errors


# ── Portfolio helpers ──────────────────────────────────────────────────────────

def load_portfolio_from_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Load portfolio holdings from a CSV file.

    Expected columns
    ----------------
    asset_class, name, value[, currency]

    Returns
    -------
    List of holding dicts ready for PortfolioAgent.analyze()
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Portfolio file not found: {file_path}")

    holdings: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"asset_class", "name", "value"}
        if not required.issubset({c.lower().strip() for c in (reader.fieldnames or [])}):
            raise ValueError(f"CSV must have columns: {required}")

        for i, row in enumerate(reader, start=2):
            try:
                holding = {
                    "asset_class": row["asset_class"].strip().lower(),
                    "name": row["name"].strip(),
                    "value": float(row["value"].replace(",", "")),
                    "currency": row.get("currency", "USD").strip(),
                }
                holdings.append(holding)
            except (KeyError, ValueError) as exc:
                logger.warning(f"Row {i} skipped: {exc}")

    logger.info(f"Loaded {len(holdings)} holding(s) from {file_path}")
    return holdings


def validate_portfolio(holdings: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Validate a list of holding dicts. Returns (valid_holdings, error_messages).
    """
    valid, errors = [], []
    required_keys = {"asset_class", "name", "value"}
    valid_asset_classes = {"stocks", "bonds", "real_estate", "cash", "crypto", "commodities"}

    for i, h in enumerate(holdings):
        missing = required_keys - set(h.keys())
        if missing:
            errors.append(f"Holding #{i+1}: missing fields {missing}")
            continue
        try:
            validated = {
                "asset_class": str(h["asset_class"]).strip().lower(),
                "name": str(h["name"]).strip(),
                "value": float(h["value"]),
                "currency": str(h.get("currency", "USD")),
            }
            if validated["value"] < 0:
                errors.append(f"Holding '{validated['name']}': value cannot be negative.")
                continue
            if validated["asset_class"] not in valid_asset_classes:
                logger.warning(
                    f"Holding '{validated['name']}': unknown asset class "
                    f"'{validated['asset_class']}'. Accepted: {valid_asset_classes}"
                )
            valid.append(validated)
        except (ValueError, TypeError) as exc:
            errors.append(f"Holding #{i+1}: {exc}")

    return valid, errors


# ── JSON helpers ───────────────────────────────────────────────────────────────

def load_json(file_path: str) -> Any:
    """Load and return parsed JSON from a file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """Serialise data to a JSON file, creating parent directories if needed."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.info(f"Saved JSON to {file_path}")


# ── Generic financial text loader ─────────────────────────────────────────────

def load_text_documents(folder: str, extensions: Tuple[str, ...] = (".txt", ".md")) -> List[Dict[str, str]]:
    """
    Walk a folder and return all text documents as a list of
    {"filename": ..., "content": ...} dicts.
    Used for seeding the RAG knowledge base.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        logger.warning(f"Documents folder does not exist: {folder}")
        return []

    docs = []
    for path in sorted(folder_path.rglob("*")):
        if path.suffix.lower() in extensions and path.is_file():
            try:
                content = path.read_text(encoding="utf-8", errors="ignore").strip()
                if content:
                    docs.append({"filename": path.name, "content": content})
            except Exception as exc:
                logger.warning(f"Could not read {path.name}: {exc}")

    logger.info(f"Loaded {len(docs)} document(s) from {folder}")
    return docs


# ── Formatting helpers ─────────────────────────────────────────────────────────

def format_currency(amount: float, currency: str = "USD", decimals: int = 2) -> str:
    """Return a human-readable currency string, e.g. '$ 1,234.56'."""
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "ZAR": "R", "JPY": "¥"}
    symbol = symbols.get(currency.upper(), currency)
    return f"{symbol} {amount:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Return a formatted percentage string, e.g. '12.5%'."""
    return f"{value:.{decimals}f}%"


def summarise_dict(d: Dict[str, float], currency: str = "USD") -> str:
    """Format a category→amount dict as a readable summary string."""
    lines = [f"  {k.title():<20} {format_currency(v, currency)}" for k, v in sorted(d.items())]
    return "\n".join(lines)