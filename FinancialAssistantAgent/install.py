"""
install.py
One-command installer for Financial Assistant AI.

What it does
------------
1. Checks Python version (3.8+ required)
2. Upgrades pip
3. Installs all packages from requirements.txt
4. Creates required directories (logs/, docs/, prompts/, models/)
5. Creates default prompt files
6. Verifies the installation by importing core modules
7. Prints a clear next-steps summary

Usage
-----
    python install.py
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────

def run(cmd: str, description: str) -> bool:
    """Run a shell command and return True on success."""
    logger.info(f"{description}…")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True,
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            logger.debug(result.stdout.strip())
        logger.info(f"  ✅ {description} — done")
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(f"  ❌ {description} — failed")
        if exc.stdout:
            logger.error(exc.stdout[-500:])   # last 500 chars to avoid spam
        if exc.stderr:
            logger.error(exc.stderr[-500:])
        return False


def check_python() -> bool:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 8):
        logger.error(f"Python 3.8+ required. You have {major}.{minor}.")
        return False
    logger.info(f"  ✅ Python {major}.{minor} detected")
    return True


def create_directories() -> None:
    dirs = ["logs", "docs", "prompts", "models", "data"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        logger.info(f"  📁 {d}/")


def create_default_prompts() -> None:
    """Write default prompt files if they don't already exist."""
    prompts = {
        "prompts/system_prompt.txt": """\
You are FinanceBot, a professional AI financial assistant.

Your expertise includes:
- Investment analysis and portfolio management
- Risk assessment and financial planning
- Market analysis and economic trends
- Personal finance and budgeting advice
- Retirement planning and wealth management

Guidelines:
- Provide accurate, structured advice based on current best practices
- Always include relevant risk considerations
- Never provide legal, tax, or medical advice
- Maintain a professional, helpful tone

Response format:
- Start with a clear, direct answer
- Provide supporting reasoning and context
- Include relevant risk factors
- Suggest next steps when appropriate
""",
        "prompts/retrieval_prompt.txt": """\
You are FinanceBot, a professional AI financial assistant.

Use the following retrieved financial context to inform your response:

{context}

User Question: {user_input}

Based on the context above and your financial expertise, provide a comprehensive answer that:
- Directly addresses the user's question
- Incorporates relevant information from the context
- Includes appropriate risk considerations
- Provides actionable insights when possible

If the context doesn't contain relevant information, rely on your general financial knowledge.
""",
    }

    for path, content in prompts.items():
        p = Path(path)
        if not p.exists():
            p.write_text(content, encoding="utf-8")
            logger.info(f"  📝 Created {path}")
        else:
            logger.info(f"  ✔  {path} already exists — skipped")


def verify_installation() -> bool:
    """Try importing the key modules to confirm everything installed correctly."""
    checks = [
        ("fastapi",              "FastAPI"),
        ("uvicorn",              "Uvicorn"),
        ("gradio",               "Gradio"),
        ("sentence_transformers","Sentence Transformers"),
        ("faiss",                "FAISS"),
        ("requests",             "Requests"),
        ("numpy",                "NumPy"),
        ("pydantic",             "Pydantic"),
        ("huggingface_hub",      "Hugging Face Hub"),
    ]

    all_ok = True
    for module, label in checks:
        try:
            __import__(module)
            logger.info(f"  ✅ {label}")
        except ImportError:
            logger.error(f"  ❌ {label} — not installed")
            all_ok = False

    # llama-cpp-python is optional (model may not be downloaded yet)
    try:
        import llama_cpp  # noqa: F401
        logger.info("  ✅ llama-cpp-python")
    except ImportError:
        logger.warning(
            "  ⚠️  llama-cpp-python — not installed.\n"
            "     The system works without it, but you won't have LLM responses.\n"
            "     To install on Windows: pip install llama-cpp-python --prefer-binary\n"
            "     To install on Linux/Mac: pip install llama-cpp-python"
        )

    return all_ok


def print_next_steps() -> None:
    model_exists = Path("models/finance-Llama3-8B.Q2_K.gguf").exists()

    print("\n" + "=" * 60)
    print("  🎉  Installation complete!")
    print("=" * 60)

    step = 1
    if not model_exists:
        print(f"\n  {step}. Download the LLM model (~2.7 GB):")
        print("       python downloadModel.py")
        step += 1

    print(f"\n  {step}. (Optional) Add financial documents to docs/")
    print("       Supported: .txt  .md  .pdf  .docx")
    step += 1

    print(f"\n  {step}. Start the full system:")
    print("       python main.py")
    print()
    print("     Or start services separately:")
    print("       python run_backend.py    # API on http://127.0.0.1:8000")
    print("       python run_frontend.py   # UI  on http://127.0.0.1:7860")
    print()
    print("     If ports are stuck (Windows):")
    print("       python kill_ports.py")
    print("\n" + "=" * 60 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    print("\n" + "=" * 60)
    print("  💹  Financial Assistant AI — Installer")
    print("=" * 60 + "\n")

    # 1. Python version
    logger.info("Checking Python version…")
    if not check_python():
        return 1

    # 2. Create directories
    logger.info("Creating project directories…")
    create_directories()

    # 3. Upgrade pip
    if not run(f'"{sys.executable}" -m pip install --upgrade pip', "Upgrading pip"):
        logger.warning("pip upgrade failed — continuing anyway.")

    # 4. Install requirements
    if not Path("requirements.txt").exists():
        logger.error("requirements.txt not found. Are you in the project root?")
        return 1

    if not run(
        f'"{sys.executable}" -m pip install -r requirements.txt',
        "Installing dependencies from requirements.txt",
    ):
        logger.error("Dependency installation failed. Check the errors above.")
        return 1

    # 5. Create default prompts
    logger.info("Setting up prompt templates…")
    create_default_prompts()

    # 6. Verify
    logger.info("Verifying installation…")
    if not verify_installation():
        logger.error("Some packages failed to install. Check errors above.")
        return 1

    # 7. Next steps
    print_next_steps()
    return 0


if __name__ == "__main__":
    sys.exit(main())
