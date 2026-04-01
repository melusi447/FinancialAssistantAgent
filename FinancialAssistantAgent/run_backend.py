"""
Backend Runner
Starts the Financial Assistant API from core/orchestrator.py.
The old app/orchestrator.py is no longer used — core/ is the single source of truth.

Usage
-----
    python run_backend.py
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/backend.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def check_model() -> bool:
    from config import config
    if not os.path.exists(config.MODEL_PATH):
        logger.error(f"Model not found: {config.MODEL_PATH}")
        logger.error("Download it with: python downloadModel.py")
        return False
    logger.info(f"Model found: {config.MODEL_PATH}")
    return True


def check_deps() -> bool:
    required = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "sentence_transformers": "sentence-transformers",
        "numpy": "numpy",
        "pydantic": "pydantic",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error("Run: pip install -r requirements.txt")
        return False
    return True


def main():
    logger.info("Starting Financial Assistant Backend...")

    if not check_deps():
        sys.exit(1)

    if not check_model():
        logger.warning("Continuing without LLM — RAG and rule-based agents still work.")

    # Ensure docs folder exists
    from config import config
    os.makedirs(config.DOCS_FOLDER, exist_ok=True)
    doc_count = len(list(Path(config.DOCS_FOLDER).glob("*.*")))
    if doc_count == 0:
        logger.warning("No documents in docs/ — RAG will use fallback knowledge.")
    else:
        logger.info(f"Found {doc_count} document(s) in docs/")

    try:
        import uvicorn
        from core.orchestrator import app

        logger.info(f"API docs  : {config.BACKEND_URL}/docs")
        logger.info(f"Health    : {config.BACKEND_URL}/health")
        logger.info(f"Agents    : {config.BACKEND_URL}/agents")
        logger.info("Press Ctrl+C to stop.")

        uvicorn.run(
            app,
            host=config.BACKEND_HOST,
            port=config.BACKEND_PORT,
            log_level="info",
            reload=False,
        )

    except ImportError as exc:
        logger.error(f"Import failed: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Server error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()